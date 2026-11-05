//! Qwen2/Qwen2.5 dense decoder with explicit block boundaries and autograd.
//! Full-sequence execution avoids stale KV state during causal interventions.
use crate::{DifferentiableTransformer, LensError, Result, TransformerForward, Unembed};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub use_sliding_window: bool,
    pub hidden_act: String,
}
impl Qwen2Config {
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.max_position_embeddings == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !(self.hidden_size / self.num_attention_heads).is_multiple_of(2)
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.
            || self.hidden_act != "silu"
            || self.use_sliding_window
        {
            return Err(LensError::Invalid(
                "unsupported/invalid Qwen2 configuration (dense full attention, SiLU required)"
                    .into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone)]
struct RmsNorm {
    weight: Tensor,
    epsilon: f64,
}
impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + self.epsilon)?.sqrt()?;
        x.broadcast_div(&rms)?
            .to_dtype(dtype)?
            .broadcast_mul(&self.weight)
    }
}
struct Block {
    norm: RmsNorm,
    post_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}
pub type ResidualEdit<'a> = dyn FnMut(usize, &Tensor) -> Result<Tensor> + 'a;

pub struct Qwen2 {
    config: Qwen2Config,
    embedding: candle_nn::Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    head: Linear,
    cos: Tensor,
    sin: Tensor,
}
impl Qwen2 {
    /// Safe buffered loading. For immutable files use `from_var_builder` with
    /// Candle's mmap VarBuilder and its documented safety contract.
    pub fn load(directory: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = directory.as_ref();
        let config: Qwen2Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .map_err(|e| LensError::Invalid(e.to_string()))?;
        config.validate()?;
        let tensors = candle_core::safetensors::load(dir.join("model.safetensors"), device)?;
        Self::from_var_builder(config, VarBuilder::from_tensors(tensors, dtype, device))
    }
    pub fn from_var_builder(config: Qwen2Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let d = config.hidden_size;
        let embedding = candle_nn::embedding(config.vocab_size, d, vb.pp("model.embed_tokens"))?;
        let rms = |path: VarBuilder<'_>| -> Result<RmsNorm> {
            Ok(RmsNorm {
                weight: path.get(d, "weight")?,
                epsilon: config.rms_norm_eps,
            })
        };
        let mut blocks = Vec::new();
        let hd = d / config.num_attention_heads;
        let kv = config.num_key_value_heads * hd;
        for l in 0..config.num_hidden_layers {
            let b = vb.pp(format!("model.layers.{l}"));
            let a = b.pp("self_attn");
            let m = b.pp("mlp");
            blocks.push(Block {
                norm: rms(b.pp("input_layernorm"))?,
                post_norm: rms(b.pp("post_attention_layernorm"))?,
                q: candle_nn::linear(d, d, a.pp("q_proj"))?,
                k: candle_nn::linear(d, kv, a.pp("k_proj"))?,
                v: candle_nn::linear(d, kv, a.pp("v_proj"))?,
                o: candle_nn::linear_no_bias(d, d, a.pp("o_proj"))?,
                gate: candle_nn::linear_no_bias(d, config.intermediate_size, m.pp("gate_proj"))?,
                up: candle_nn::linear_no_bias(d, config.intermediate_size, m.pp("up_proj"))?,
                down: candle_nn::linear_no_bias(config.intermediate_size, d, m.pp("down_proj"))?,
            });
        }
        let head = if config.tie_word_embeddings {
            Linear::new(embedding.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(d, config.vocab_size, vb.pp("lm_head"))?
        };
        let norm = rms(vb.pp("model.norm"))?;
        let frequencies: Vec<f32> = (0..hd)
            .step_by(2)
            .map(|i| config.rope_theta.powf(-(i as f64) / hd as f64) as f32)
            .collect();
        let frequency = Tensor::from_vec(frequencies, (1, hd / 2), vb.device())?;
        let positions = Tensor::arange(0u32, config.max_position_embeddings as u32, vb.device())?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
        let angles = positions.matmul(&frequency)?;
        Ok(Self {
            config,
            embedding,
            blocks,
            norm,
            head,
            cos: angles.cos()?.to_dtype(vb.dtype())?,
            sin: angles.sin()?.to_dtype(vb.dtype())?,
        })
    }
    /// Shared-model inference with request-local tracking and optional edits.
    /// Edits run immediately after a block; observations see the edited state.
    pub fn forward_tracked(
        &self,
        ids: &Tensor,
        mut tracker: Option<&mut crate::LensTracker<'_>>,
        mut edit: Option<&mut ResidualEdit<'_>>,
    ) -> Result<Tensor> {
        if let Some(t) = tracker.as_deref_mut() {
            t.validate_layer_count(self.layer_count())?;
            t.clear();
        }
        let mut h = self.embed_tokens(ids)?;
        for layer in 0..self.layer_count() {
            h = self.forward_block(layer, &h)?;
            if let Some(edit) = edit.as_deref_mut() {
                let next = edit(layer, &h)?;
                if next.dims() != h.dims()
                    || next.dtype() != h.dtype()
                    || !next.device().same_device(h.device())
                {
                    return Err(LensError::Invalid(
                        "intervention changed residual shape/dtype/device".into(),
                    ));
                }
                h = next;
            }
            if let Some(t) = tracker.as_deref_mut() {
                t.observe(layer, &h)?;
            }
        }
        self.logits(&h)
    }

    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }
    pub fn lm_head(&self) -> &Tensor {
        self.head.weight()
    }
    pub fn residuals(&self, ids: &Tensor) -> Result<Vec<Tensor>> {
        let mut h = self.embed_tokens(ids)?;
        let mut states = Vec::with_capacity(self.layer_count());
        for l in 0..self.layer_count() {
            h = self.forward_block(l, &h)?;
            states.push(h.clone());
        }
        Ok(states)
    }
    /// Recompute downstream residuals after a block-output intervention.
    pub fn suffix(&self, source: usize, target: usize, hidden: &Tensor) -> Result<Tensor> {
        if source > target || target >= self.layer_count() {
            return Err(LensError::Invalid("invalid suffix layers".into()));
        }
        let mut h = hidden.clone();
        for l in source + 1..=target {
            h = self.forward_block(l, &h)?;
        }
        Ok(h)
    }
    fn rope(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, s, d) = x.dims4()?;
        let half = d / 2;
        let cos = self.cos.narrow(0, 0, s)?.reshape((1, 1, s, half))?;
        let sin = self.sin.narrow(0, 0, s)?.reshape((1, 1, s, half))?;
        let left = x.narrow(3, 0, half)?;
        let right = x.narrow(3, half, half)?;
        let a = (left.broadcast_mul(&cos)? - right.broadcast_mul(&sin)?)?;
        let b = (right.broadcast_mul(&cos)? + left.broadcast_mul(&sin)?)?;
        Ok(Tensor::cat(&[a, b], 3)?.contiguous()?)
    }
    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let (b, kv, s, d) = x.dims4()?;
        let groups = self.config.num_attention_heads / kv;
        Ok(x.unsqueeze(2)?
            .broadcast_as((b, kv, groups, s, d))?
            .reshape((b, kv * groups, s, d))?
            .contiguous()?)
    }
}
impl DifferentiableTransformer for Qwen2 {
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
    fn layer_count(&self) -> usize {
        self.blocks.len()
    }
    fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        let (batch, seq) = ids.dims2()?;
        if batch == 0 || seq == 0 || seq > self.config.max_position_embeddings {
            return Err(LensError::Invalid(
                "invalid unpadded sequence length".into(),
            ));
        }
        Ok(self.embedding.forward(ids)?)
    }
    fn forward_block(&self, layer: usize, hidden: &Tensor) -> Result<Tensor> {
        let block = self
            .blocks
            .get(layer)
            .ok_or(LensError::MissingLayer(layer))?;
        let (b, s, d) = hidden.dims3()?;
        if d != self.config.hidden_size || s == 0 || s > self.config.max_position_embeddings {
            return Err(LensError::Invalid("invalid block hidden shape".into()));
        }
        let x = block.norm.forward(hidden)?;
        let hd = d / self.config.num_attention_heads;
        let split = |x: Tensor, heads: usize| -> Result<Tensor> {
            Ok(x.reshape((b, s, heads, hd))?.transpose(1, 2)?)
        };
        let q = self.rope(&split(
            block.q.forward(&x)?,
            self.config.num_attention_heads,
        )?)?;
        let k = self.repeat_kv(&self.rope(&split(
            block.k.forward(&x)?,
            self.config.num_key_value_heads,
        )?)?)?;
        let v = self.repeat_kv(&split(
            block.v.forward(&x)?,
            self.config.num_key_value_heads,
        )?)?;
        let mask: Vec<f32> = (0..s)
            .flat_map(|i| (0..s).map(move |j| if j > i { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask =
            Tensor::from_vec(mask, (1, 1, s, s), hidden.device())?.to_dtype(hidden.dtype())?;
        let scores = q
            .matmul(&k.transpose(2, 3)?)?
            .affine(1. / (hd as f64).sqrt(), 0.)?
            .broadcast_add(&mask)?;
        // Composite softmax/RMSNorm/RoPE preserve autograd; fused inference
        // kernels in Candle do not necessarily provide backward implementations.
        let attention = candle_nn::ops::softmax(&scores, D::Minus1)?
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, s, d))?;
        let h = (hidden + block.o.forward(&attention)?)?;
        let x = block.post_norm.forward(&h)?;
        let gate = block.gate.forward(&x)?;
        let activated = gate.mul(&candle_nn::ops::sigmoid(&gate)?)?;
        let mlp = block
            .down
            .forward(&activated.mul(&block.up.forward(&x)?)?)?;
        Ok((h + mlp)?)
    }
}
impl Unembed for Qwen2 {
    fn logits(&self, h: &Tensor) -> Result<Tensor> {
        Ok(self.head.forward(&self.norm.forward(h)?)?)
    }
}
impl TransformerForward for Qwen2 {
    fn embed(&mut self, ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens(ids)
    }
    fn num_layers(&self) -> usize {
        self.layer_count()
    }
    fn block(&mut self, layer: usize, h: &Tensor) -> Result<Tensor> {
        self.forward_block(layer, h)
    }
    fn unembed(&self, h: &Tensor) -> Result<Tensor> {
        self.logits(h)
    }
}
