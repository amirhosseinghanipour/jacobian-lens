//! Exact reverse-mode fitting using source variables and summed target cotangents.
use crate::{JacobianLensRegistry, LensError, Result};
use candle_core::{DType, Tensor, Var};
use std::collections::BTreeMap;

/// Full-sequence, deterministic block execution with differentiable operations.
/// No KV cache, dropout, fused inference-only norms or detached attention.
pub trait DifferentiableTransformer: Send + Sync {
    fn hidden_size(&self) -> usize;
    fn layer_count(&self) -> usize;
    fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor>;
    fn forward_block(&self, layer: usize, hidden: &Tensor) -> Result<Tensor>;
}
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FittingConfig {
    pub source_layers: Vec<usize>,
    pub target_layer: usize,
    pub skip_first: usize,
    pub max_seq_len: usize,
    /// Number of independent output cotangents computed in one backward pass.
    pub dim_batch: usize,
}
impl FittingConfig {
    pub fn validate(&self, layers: usize) -> Result<()> {
        if self.source_layers.is_empty()
            || self.target_layer >= layers
            || self.dim_batch == 0
            || self.max_seq_len.saturating_sub(1) <= self.skip_first
            || self.source_layers.iter().any(|&s| s >= self.target_layer)
            || self.source_layers.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(LensError::Invalid(
                "invalid fitting layers, positions or dim_batch".into(),
            ));
        }
        Ok(())
    }
}
pub struct PromptJacobian {
    pub jacobians: BTreeMap<usize, Tensor>,
    pub seq_len: usize,
    pub valid_positions: usize,
}
/// Same reduction as Python: target-position sum, source-position mean.
/// Candle retains leaf gradients, so each source layer is made a fresh variable
/// and its suffix is recomputed. Parameter tensors must be frozen.
pub fn jacobian_for_prompt(
    model: &impl DifferentiableTransformer,
    ids: &Tensor,
    config: &FittingConfig,
) -> Result<PromptJacobian> {
    config.validate(model.layer_count())?;
    let (batch, seq) = ids.dims2()?;
    if batch != 1 {
        return Err(LensError::Invalid(
            "fitting expects one unpadded prompt".into(),
        ));
    }
    let seq = seq.min(config.max_seq_len);
    if seq.saturating_sub(1) <= config.skip_first {
        return Err(LensError::Invalid("prompt too short".into()));
    }
    let valid = seq - 1 - config.skip_first;
    let ids = ids.narrow(1, 0, seq)?;
    let mut h = model.embed_tokens(&ids)?.detach();
    let mut sources = BTreeMap::new();
    for layer in 0..=config.target_layer {
        h = model.forward_block(layer, &h)?.detach();
        if config.source_layers.contains(&layer) {
            sources.insert(layer, h.clone());
        }
    }
    let dimensions: Vec<_> = (0..model.hidden_size()).collect();
    let mut jacobians = BTreeMap::new();
    for (layer, state) in sources {
        jacobians.insert(
            layer,
            jacobian_rows(
                model,
                &state,
                layer,
                config.target_layer,
                config.skip_first,
                &dimensions,
                config.dim_batch,
            )?,
        );
    }
    Ok(PromptJacobian {
        jacobians,
        seq_len: seq,
        valid_positions: valid,
    })
}
/// Corpus mean, weighting successful prompts equally. Only short prompts skip.
pub fn fit(
    model: &impl DifferentiableTransformer,
    prompts: &[Tensor],
    config: &FittingConfig,
) -> Result<(JacobianLensRegistry, usize)> {
    config.validate(model.layer_count())?;
    let mut sums: BTreeMap<usize, Tensor> = BTreeMap::new();
    let mut n = 0;
    for ids in prompts {
        let (batch, seq) = ids.dims2()?;
        if batch != 1 {
            return Err(LensError::Invalid("fitting expects batch one".into()));
        }
        if seq.min(config.max_seq_len).saturating_sub(1) <= config.skip_first {
            continue;
        }
        let result = jacobian_for_prompt(model, ids, config)?;
        for (l, j) in result.jacobians {
            let sum = match sums.remove(&l) {
                Some(prev) => (prev + j)?,
                None => j,
            };
            sums.insert(l, sum);
        }
        n += 1;
    }
    if n == 0 {
        return Err(LensError::Invalid("no valid fitting prompts".into()));
    }
    let mean = sums
        .into_iter()
        .map(|(l, j)| Ok((l, (j / n as f64)?)))
        .collect::<Result<_>>()?;
    Ok((JacobianLensRegistry::new(mean)?, n))
}

/// Selected Jacobian rows for independent parity checks or directional probes.
/// `state` must be a single prompt's source block output, before final norm.
pub fn jacobian_rows(
    model: &impl DifferentiableTransformer,
    state: &Tensor,
    source_layer: usize,
    target_layer: usize,
    skip_first: usize,
    dimensions: &[usize],
    dim_batch: usize,
) -> Result<Tensor> {
    let (batch, seq, d) = state.dims3()?;
    if batch != 1
        || d != model.hidden_size()
        || source_layer >= target_layer
        || target_layer >= model.layer_count()
        || seq.saturating_sub(1) <= skip_first
        || dim_batch == 0
        || dimensions.is_empty()
        || dimensions.iter().any(|&i| i >= d)
    {
        return Err(LensError::Invalid(
            "invalid Jacobian probe configuration".into(),
        ));
    }
    let valid = seq - 1 - skip_first;
    let mut rows = Vec::new();
    for dims in dimensions.chunks(dim_batch) {
        let count = dims.len();
        let source = Var::from_tensor(&state.detach().repeat((count, 1, 1))?)?;
        let mut output = source.as_tensor().clone();
        for layer in source_layer + 1..=target_layer {
            output = model.forward_block(layer, &output)?;
        }
        let mut cotangent = vec![0f32; count * seq * d];
        for (b, &dim) in dims.iter().enumerate() {
            for p in skip_first..seq - 1 {
                cotangent[(b * seq + p) * d + dim] = 1.;
            }
        }
        let cotangent = Tensor::from_vec(cotangent, (count, seq, d), state.device())?
            .to_dtype(output.dtype())?;
        let gradients = output.mul(&cotangent)?.sum_all()?.backward()?;
        let grad = gradients
            .get(source.as_tensor())
            .ok_or_else(|| LensError::Invalid("missing source gradient".into()))?;
        rows.push(
            grad.narrow(1, skip_first, valid)?
                .to_dtype(DType::F32)?
                .mean(1)?
                .detach(),
        );
    }
    Ok(Tensor::cat(&rows, 0)?)
}
