//! Runtime-independent residual-stream access. Adapters own model execution.
use crate::{LensError, Result, Unembed};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDimensions {
    pub hidden_size: usize,
    pub layer_count: usize,
    pub vocab_size: usize,
}
impl ModelDimensions {
    pub fn validate(self) -> Result<Self> {
        if self.hidden_size == 0 || self.layer_count == 0 || self.vocab_size == 0 {
            return Err(LensError::Invalid("empty model dimensions".into()));
        }
        Ok(self)
    }
}

/// Replace a complete block output, before final normalization. To edit one
/// position, construct a replacement that preserves all other positions.
/// Execution must restart from the original inputs with fresh runtime state.
pub struct ResidualReplacement {
    pub layer: usize,
    pub hidden: Tensor,
}

pub struct ModelTrace {
    pub residuals: BTreeMap<usize, Tensor>,
    /// Last block output, before the model's final normalization.
    pub final_residual: Tensor,
}

/// Model-independent interface for readout and causal experiments.
///
/// Implementations may use Candle, a foreign runtime, or another process. All
/// returned tensors must use a consistent device and floating-point dtype.
/// Hidden states have shape [batch, sequence, hidden]; capture indices name
/// block outputs. No assumption is made about attention, recurrence, experts,
/// position encodings, or a block's auxiliary inputs/state. A complete forward
/// pass, rather than independent block calls, preserves those dependencies.
///
/// Implementations must validate capture/edit indices, reject duplicate edits,
/// and validate replacement shape/dtype/device before modifying the execution.
/// Captures see edits. Calls must be deterministic (evaluation mode, no caches
/// carried between calls). Concurrent calls must isolate all request state.
pub trait LensModel: Unembed {
    fn dimensions(&self) -> ModelDimensions;
    fn capture(
        &self,
        input_ids: &Tensor,
        layers: &[usize],
        replacements: &[ResidualReplacement],
    ) -> Result<ModelTrace>;
    /// Selected rows of the model's actual unembedding W_U, [tokens, hidden].
    /// Fetch only selected rows, avoiding a vocabulary-sized host transfer.
    fn unembedding_rows(&self, token_ids: &[u32]) -> Result<Tensor>;
}

pub(crate) fn validate_capture(
    dimensions: ModelDimensions,
    layers: &[usize],
    replacements: &[ResidualReplacement],
) -> Result<()> {
    dimensions.validate()?;
    if let Some(&layer) = layers.iter().find(|&&l| l >= dimensions.layer_count) {
        return Err(LensError::MissingLayer(layer));
    }
    let mut seen = std::collections::BTreeSet::new();
    for edit in replacements {
        if edit.layer >= dimensions.layer_count {
            return Err(LensError::MissingLayer(edit.layer));
        }
        if !seen.insert(edit.layer) {
            return Err(LensError::Invalid("duplicate residual replacement".into()));
        }
    }
    Ok(())
}

/// Rerun a complete model with edits, then read out the final residual. This
/// also works for recurrent/hybrid models whose suffix cannot run in isolation.
pub fn intervention_logits(
    model: &dyn LensModel,
    input_ids: &Tensor,
    replacements: &[ResidualReplacement],
) -> Result<Tensor> {
    let trace = model.capture(input_ids, &[], replacements)?;
    model.logits(&trace.final_residual)
}

/// Combined object-safe interface for applications that both fit and analyze.
/// Readout-only adapters can implement LensModel without an estimator.
pub trait ResearchModel: LensModel + crate::JacobianEstimator {}
impl<T: LensModel + crate::JacobianEstimator + ?Sized> ResearchModel for T {}
