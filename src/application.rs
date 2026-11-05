//! Reference-style lens application and logit-lens baseline on captured states.
use crate::{
    DifferentiableTransformer, JacobianLensRegistry, LensError, Result, Unembed, qwen2::Qwen2,
};
use candle_core::{DType, Tensor};
use std::collections::BTreeMap;
#[derive(Debug, Clone)]
pub struct ApplyOptions {
    /// None selects every fitted layer.
    pub layers: Option<Vec<usize>>,
    /// None selects every position; negative indices count from the end.
    pub positions: Option<Vec<isize>>,
    pub max_seq_len: usize,
    pub use_jacobian: bool,
}
impl Default for ApplyOptions {
    fn default() -> Self {
        Self {
            layers: None,
            positions: None,
            max_seq_len: 512,
            use_jacobian: true,
        }
    }
}
pub struct AppliedLens {
    pub lens_logits: BTreeMap<usize, Tensor>,
    pub model_logits: Tensor,
    pub input_ids: Tensor,
}
impl JacobianLensRegistry {
    /// Batch-one application, with outputs [selected_positions, vocabulary].
    pub fn apply(
        &self,
        model: &Qwen2,
        ids: &Tensor,
        options: &ApplyOptions,
    ) -> Result<AppliedLens> {
        let (batch, seq) = ids.dims2()?;
        let seq = seq.min(options.max_seq_len);
        if batch != 1 || seq == 0 {
            return Err(LensError::Invalid(
                "apply requires one nonempty prompt".into(),
            ));
        }
        let layers = options
            .layers
            .clone()
            .unwrap_or_else(|| self.layers().collect());
        if layers.is_empty() || layers.iter().any(|&l| l >= model.layer_count()) {
            return Err(LensError::Invalid("apply layer outside model".into()));
        }
        if options.use_jacobian {
            for &l in &layers {
                if !self.contains(l) {
                    return Err(LensError::MissingLayer(l));
                }
            }
        }
        let positions = match &options.positions {
            None => (0..seq).collect::<Vec<_>>(),
            Some(indices) => indices
                .iter()
                .map(|&p| {
                    let p = if p < 0 { seq as isize + p } else { p };
                    if p < 0 || p >= seq as isize {
                        return Err(LensError::Invalid("apply position outside sequence".into()));
                    }
                    Ok(p as usize)
                })
                .collect::<Result<Vec<_>>>()?,
        };
        if positions.is_empty() {
            return Err(LensError::Invalid("empty position selection".into()));
        }
        let input_ids = ids.narrow(1, 0, seq)?;
        let states = model.residuals(&input_ids)?;
        let indices = Tensor::new(
            positions.iter().map(|&p| p as u32).collect::<Vec<_>>(),
            ids.device(),
        )?;
        let select = |h: &Tensor| -> Result<Tensor> {
            Ok(h.squeeze(0)?
                .index_select(&indices, 0)?
                .to_dtype(DType::F32)?
                .contiguous()?)
        };
        let mut lens_logits = BTreeMap::new();
        for layer in layers {
            let h = select(&states[layer])?;
            let logits = if options.use_jacobian {
                self.decode_jspace(layer, &h, model)?
            } else {
                model.logits(&h)?
            };
            lens_logits.insert(layer, logits.to_dtype(DType::F32)?);
        }
        let model_logits = model
            .logits(&select(states.last().unwrap())?)?
            .to_dtype(DType::F32)?;
        Ok(AppliedLens {
            lens_logits,
            model_logits,
            input_ids,
        })
    }
}
