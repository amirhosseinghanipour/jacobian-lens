use crate::{JacobianLensRegistry, LensError, Result, TokenScore, Unembed, top_k};
use candle_core::Tensor;
use std::collections::BTreeSet;
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct Observation {
    pub layer: usize,
    pub batch: usize,
    pub position: usize,
    pub tokens: Vec<TokenScore>,
}
/// Request-local, bounded tracker. Keeps token summaries, never hidden states or
/// full layer-by-position-by-vocabulary tensors. Share registry/readout only.
pub struct LensTracker<'a> {
    registry: &'a JacobianLensRegistry,
    readout: &'a dyn Unembed,
    tokenizer: &'a Tokenizer,
    layers: BTreeSet<usize>,
    positions: Vec<(usize, usize)>,
    k: usize,
    observations: Vec<Observation>,
}
impl<'a> LensTracker<'a> {
    pub fn new(
        registry: &'a JacobianLensRegistry,
        readout: &'a dyn Unembed,
        tokenizer: &'a Tokenizer,
        layers: impl IntoIterator<Item = usize>,
        positions: Vec<(usize, usize)>,
        k: usize,
    ) -> Result<Self> {
        let layers: BTreeSet<_> = layers.into_iter().collect();
        for &layer in &layers {
            if !registry.contains(layer) {
                return Err(LensError::MissingLayer(layer));
            }
        }
        if positions.is_empty() || k == 0 {
            return Err(LensError::Invalid("select positions and positive k".into()));
        }
        Ok(Self {
            registry,
            readout,
            tokenizer,
            layers,
            positions,
            k,
            observations: Vec::new(),
        })
    }
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }
    pub(crate) fn validate_layer_count(&self, count: usize) -> Result<()> {
        if self.layers.iter().any(|&l| l >= count) {
            return Err(LensError::Invalid("tracked layer outside model".into()));
        }
        Ok(())
    }
    pub fn clear(&mut self) {
        self.observations.clear();
    }
    /// Called immediately after a block, before the model's final norm.
    /// Repeated layers replace earlier observations (bounded across forwards).
    pub fn observe(&mut self, layer: usize, hidden: &Tensor) -> Result<()> {
        if !self.layers.contains(&layer) {
            return Ok(());
        }
        let (batch, seq, _) = hidden.dims3()?;
        let mut selected = Vec::with_capacity(self.positions.len());
        for &(b, p) in &self.positions {
            if b >= batch || p >= seq {
                return Err(LensError::Invalid(format!(
                    "position ({b}, {p}) outside [{batch}, {seq}]"
                )));
            }
            selected.push(hidden.get(b)?.get(p)?);
        }
        let selected = Tensor::stack(&selected, 0)?.contiguous()?;
        let logits = self
            .registry
            .decode_jspace(layer, &selected, self.readout)?;
        let scores = top_k(&logits, self.k, self.tokenizer)?;
        self.observations.retain(|o| o.layer != layer);
        for (&(batch, position), tokens) in self.positions.iter().zip(scores) {
            self.observations.push(Observation {
                layer,
                batch,
                position,
                tokens,
            });
        }
        Ok(())
    }
}
/// A model adapter owns attention masks, rotary positions and KV-cache policy.
/// Each mutable model instance belongs to one inference worker.
pub trait TransformerForward: Send {
    fn embed(&mut self, input_ids: &Tensor) -> Result<Tensor>;
    fn num_layers(&self) -> usize;
    fn block(&mut self, layer: usize, hidden: &Tensor) -> Result<Tensor>;
    fn unembed(&self, hidden: &Tensor) -> Result<Tensor>;
    fn forward(
        &mut self,
        input_ids: &Tensor,
        mut tracker: Option<&mut LensTracker<'_>>,
    ) -> Result<Tensor> {
        if let Some(t) = tracker.as_deref_mut() {
            if t.layers.iter().any(|&l| l >= self.num_layers()) {
                return Err(LensError::Invalid("tracked layer outside model".into()));
            }
            t.clear();
        }
        let mut hidden = self.embed(input_ids)?;
        for layer in 0..self.num_layers() {
            hidden = self.block(layer, &hidden)?;
            if let Some(t) = tracker.as_deref_mut() {
                t.observe(layer, &hidden)?;
            }
        }
        self.unembed(&hidden)
    }
}
