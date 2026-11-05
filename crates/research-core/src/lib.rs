//! Deterministic f64 CPU reference algorithms. Not a pretrained model runtime.
pub use nalgebra::{DMatrix as Matrix, DVector as Vector};
use serde::{Deserialize, Serialize};
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid research input: {0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn ensure(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Invalid(message.into()))
    }
}
pub fn finite(matrix: &Matrix<f64>) -> Result<()> {
    ensure(
        matrix.iter().all(|x| x.is_finite()),
        "matrix has non-finite entries",
    )
}
pub mod fitting;
pub mod intervention;
pub mod metrics;
pub mod model;
pub mod slice;

/// Atomic replace, including file fsync. Existing checkpoints survive failed writes.
pub fn save_json(path: &std::path::Path, value: &impl Serialize) -> Result<()> {
    use std::io::Write;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer(&mut tmp, value)?;
    tmp.flush()?;
    tmp.as_file().sync_all()?;
    tmp.persist(path).map_err(|e| Error::Io(e.error))?;
    #[cfg(unix)]
    std::fs::File::open(parent)?.sync_all()?;
    Ok(())
}
pub fn load_json<T: for<'de> Deserialize<'de>>(path: &std::path::Path) -> Result<T> {
    Ok(serde_json::from_reader(std::fs::File::open(path)?)?)
}
