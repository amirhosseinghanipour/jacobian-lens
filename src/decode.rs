use crate::{JacobianLensRegistry, LensError, Result, contiguous, shape};
use candle_core::{DType, Module, Tensor};
use tokenizers::Tokenizer;

/// Implement using the exact pretrained model's final norm and LM head.
/// Input/output convention: [..., hidden] -> [..., vocab].
pub trait Unembed: Send + Sync {
    fn logits(&self, hidden: &Tensor) -> Result<Tensor>;
}
/// Supports Candle LayerNorm, RMSNorm, or a model-specific normalization Module.
pub struct ModelReadout<N> {
    norm: N,
    head_t: Tensor,
    bias: Option<Tensor>,
    width: usize,
}
impl<N: Module + Send + Sync> ModelReadout<N> {
    /// `head` is PyTorch Linear layout [vocabulary, hidden]. Norm parameters,
    /// epsilon, dtype and device must match the original model.
    pub fn new(norm: N, head: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let (vocab, width) = head.dims2()?;
        if vocab == 0 || width == 0 {
            return Err(shape("lm_head", "nonempty [vocab, hidden]", &head));
        }
        contiguous(&head, "lm_head")?;
        if let Some(b) = &bias {
            if b.dims() != [vocab] {
                return Err(shape("lm_head bias", format!("[{vocab}]"), b));
            }
            if b.dtype() != head.dtype() || !b.device().same_device(head.device()) {
                return Err(LensError::Invalid("head bias dtype/device mismatch".into()));
            }
        }
        Ok(Self {
            norm,
            head_t: head.detach().t()?.contiguous()?,
            bias,
            width,
        })
    }
}
impl<N: Module + Send + Sync> Unembed for ModelReadout<N> {
    fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        if hidden.dims().last() != Some(&self.width) {
            return Err(shape("readout", format!("[..., {}]", self.width), hidden));
        }
        let h = hidden.to_dtype(self.head_t.dtype())?;
        let normalized = self.norm.forward(&h)?;
        let flat = normalized.reshape((hidden.elem_count() / self.width, self.width))?;
        let mut logits = flat.matmul(&self.head_t)?;
        if let Some(b) = &self.bias {
            logits = logits.broadcast_add(b)?;
        }
        let mut dims = hidden.dims().to_vec();
        *dims.last_mut().expect("validated nonzero rank") = self.head_t.dim(1)?;
        Ok(logits.reshape(dims)?)
    }
}
impl JacobianLensRegistry {
    pub fn decode_jspace(
        &self,
        layer: usize,
        hidden: &Tensor,
        readout: &dyn Unembed,
    ) -> Result<Tensor> {
        readout.logits(&self.transport(layer, hidden)?)
    }
}
#[derive(Debug, Clone)]
pub struct TokenScore {
    pub id: u32,
    /// Raw vocabulary spelling, useful for byte/BPE fragments.
    pub token: String,
    /// Decoding one token may be lossy for incomplete UTF-8 fragments.
    pub text: String,
    pub logit: f32,
    /// Full-vocabulary softmax mass, not a calibrated prediction probability.
    pub probability: f64,
}
/// Exact top-K per flattened leading position; deterministic ties use token ID.
/// CPU selection is O(V + K log K), transfers logits once, and retains O(K)
/// results. For large GPU workloads call only on selected positions.
pub fn top_k(logits: &Tensor, k: usize, tokenizer: &Tokenizer) -> Result<Vec<Vec<TokenScore>>> {
    let vocab = *logits
        .dims()
        .last()
        .ok_or_else(|| shape("logits", "[..., vocab]", logits))?;
    if k == 0 || k > vocab || vocab > u32::MAX as usize {
        return Err(LensError::Invalid(
            "top-k must be in 1..=vocab, vocab must fit u32".into(),
        ));
    }
    let rows = logits
        .to_dtype(DType::F32)?
        .reshape((logits.elem_count() / vocab, vocab))?
        .to_vec2::<f32>()?;
    rows.into_iter()
        .map(|row| {
            if row.iter().any(|v| !v.is_finite()) {
                return Err(LensError::Invalid("non-finite logits".into()));
            }
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let z: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
            let mut ids: Vec<usize> = (0..vocab).collect();
            let cmp = |a: &usize, b: &usize| row[*b].total_cmp(&row[*a]).then(a.cmp(b));
            if k < vocab {
                ids.select_nth_unstable_by(k, cmp);
                ids.truncate(k);
            }
            ids.sort_unstable_by(cmp);
            ids.into_iter()
                .map(|id| {
                    let token = tokenizer
                        .id_to_token(id as u32)
                        .ok_or_else(|| LensError::Tokenizer(format!("missing token id {id}")))?;
                    let text = tokenizer
                        .decode(&[id as u32], false)
                        .map_err(|e| LensError::Tokenizer(e.to_string()))?;
                    Ok(TokenScore {
                        id: id as u32,
                        token,
                        text,
                        logit: row[id],
                        probability: (row[id] as f64 - max).exp() / z,
                    })
                })
                .collect()
        })
        .collect()
}
