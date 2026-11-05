//! Inference-only Jacobian lens: `h @ J.T -> model final norm -> lm_head`.
//! Layer indices refer to residual block **outputs**, before the final norm.
#![deny(unsafe_op_in_unsafe_fn)]

mod decode;
mod fitting;
pub mod qwen2;
pub use fitting::*;
pub use jlens_research as research;
mod registry;
mod tracking;
pub use decode::*;
pub use registry::*;
pub use tracking::*;

#[derive(Debug, thiserror::Error)]
pub enum LensError {
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("no Jacobian for layer {0}")]
    MissingLayer(usize),
    #[error("{name}: expected {expected}, got {actual:?}")]
    Shape {
        name: &'static str,
        expected: String,
        actual: Vec<usize>,
    },
    #[error("{0} must have contiguous row-major storage; call contiguous() explicitly")]
    NonContiguous(&'static str),
    #[error("invalid lens configuration: {0}")]
    Invalid(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
}
pub type Result<T> = std::result::Result<T, LensError>;

pub(crate) fn shape(
    name: &'static str,
    expected: impl Into<String>,
    t: &candle_core::Tensor,
) -> LensError {
    LensError::Shape {
        name,
        expected: expected.into(),
        actual: t.dims().to_vec(),
    }
}
pub(crate) fn contiguous(t: &candle_core::Tensor, name: &'static str) -> Result<()> {
    if !t.is_contiguous() {
        return Err(LensError::NonContiguous(name));
    }
    Ok(())
}
