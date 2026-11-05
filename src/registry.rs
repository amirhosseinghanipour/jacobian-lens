use crate::{LensError, Result, contiguous, shape};
use candle_core::{DType, Device, Tensor};
use memmap2::{Mmap, MmapOptions};
use safetensors::{SafeTensors, tensor::TensorView};
use std::{collections::BTreeMap, fs::File, path::Path};

/// Owns a read-only mapping. Views borrow this owner, never an ephemeral buffer.
/// Safetensors specifies little-endian, dense row-major data (no strides).
pub struct MappedLens {
    mmap: Mmap,
}
impl MappedLens {
    /// # Safety
    /// The file must not be modified or truncated by ANY process until this
    /// object and all its borrowed views have been dropped. Read-only opening
    /// does not enforce that contract. Use immutable artifact files.
    pub unsafe fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the caller guarantees file stability for the mapping lifetime.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        SafeTensors::deserialize(&mmap)?;
        Ok(Self { mmap })
    }
    /// Zero-copy tensor views over the mapped bytes; metadata is parsed here.
    pub fn tensors(&self) -> Result<SafeTensors<'_>> {
        Ok(SafeTensors::deserialize(&self.mmap)?)
    }
    /// Materialize once per inference device. Candle owns its storage; this is
    /// an explicit copy/upload, NOT a zero-copy Candle tensor constructor.
    pub fn materialize(&self, device: &Device, dtype: DType) -> Result<JacobianLensRegistry> {
        JacobianLensRegistry::from_safetensors(&self.tensors()?, device, dtype)
    }
}

/// Immutable per-device projections, shareable with `Arc` across workers.
/// Internally stores J.T packed once, avoiding per-token transpose packing.
pub struct JacobianLensRegistry {
    transposes: BTreeMap<usize, Tensor>,
    width: usize,
}
impl JacobianLensRegistry {
    pub fn new(jacobians: BTreeMap<usize, Tensor>) -> Result<Self> {
        let first = jacobians
            .values()
            .next()
            .ok_or_else(|| LensError::Invalid("empty registry".into()))?;
        let (d, out) = first.dims2()?;
        if d == 0 || out != d {
            return Err(shape("Jacobian", "nonempty square matrix", first));
        }
        let dtype = first.dtype();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(LensError::Invalid(
                "compute dtype must be F32, F16 or BF16".into(),
            ));
        }
        let device = first.device().clone();
        let mut transposes = BTreeMap::new();
        for (layer, j) in jacobians {
            if j.dims() != [d, d] {
                return Err(shape("Jacobian", format!("[{d}, {d}]"), &j));
            }
            contiguous(&j, "Jacobian")?;
            if j.dtype() != dtype || !j.device().same_device(&device) {
                return Err(LensError::Invalid(
                    "Jacobians must share device and dtype".into(),
                ));
            }
            transposes.insert(layer, j.detach().t()?.contiguous()?);
        }
        Ok(Self {
            transposes,
            width: d,
        })
    }
    /// Safe alternative to mmap for files that might change concurrently.
    pub fn load(path: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_safetensors(&SafeTensors::deserialize(&bytes)?, device, dtype)
    }
    fn from_safetensors(data: &SafeTensors<'_>, device: &Device, dtype: DType) -> Result<Self> {
        let mut weights = BTreeMap::new();
        for (name, view) in data.tensors() {
            let layer = name
                .strip_prefix("jacobians.")
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or_else(|| {
                    LensError::Invalid(format!("expected jacobians.<layer>, got {name}"))
                })?;
            let j = load_view(&view, device)?.to_dtype(dtype)?;
            if weights.insert(layer, j).is_some() {
                return Err(LensError::Invalid(format!("duplicate layer {layer}")));
            }
        }
        Self::new(weights)
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.transposes.keys().copied()
    }
    pub fn contains(&self, layer: usize) -> bool {
        self.transposes.contains_key(&layer)
    }
    /// Transport [..., d] into the final residual basis. No affine bias:
    /// the corpus-averaged Jacobian is a linear disposition readout.
    pub fn transport(&self, layer: usize, hidden: &Tensor) -> Result<Tensor> {
        let jt = self
            .transposes
            .get(&layer)
            .ok_or(LensError::MissingLayer(layer))?;
        contiguous(hidden, "hidden state")?;
        if hidden.rank() == 0 || hidden.dims().last() != Some(&self.width) {
            return Err(shape(
                "hidden state",
                format!("[..., {}]", self.width),
                hidden,
            ));
        }
        if !hidden.device().same_device(jt.device()) {
            return Err(LensError::Invalid(
                "hidden state and registry devices differ".into(),
            ));
        }
        // Python reference promotes residuals/J to fp32. Choose F32 at load for
        // parity, or explicitly opt into a lower precision deployment.
        let h = hidden.detach().to_dtype(jt.dtype())?;
        let rows = h.elem_count() / self.width;
        Ok(h.reshape((rows, self.width))?
            .matmul(jt)?
            .reshape(hidden.shape())?)
    }
}
fn load_view(view: &TensorView<'_>, device: &Device) -> Result<Tensor> {
    if cfg!(target_endian = "big") {
        return Err(LensError::Invalid("little-endian host required".into()));
    }
    let dtype = match view.dtype() {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        other => {
            return Err(LensError::Invalid(format!(
                "unsupported weight dtype {other:?}"
            )));
        }
    };
    Ok(Tensor::from_raw_buffer(
        view.data(),
        dtype,
        view.shape(),
        device,
    )?)
}

impl JacobianLensRegistry {
    /// Export `J[out,in]` in the canonical Python convention.
    pub fn jacobian(&self, layer: usize) -> Result<Tensor> {
        Ok(self
            .transposes
            .get(&layer)
            .ok_or(LensError::MissingLayer(layer))?
            .t()?
            .contiguous()?)
    }
    /// Write canonical multi-layer safetensors atomically.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.save_with_metadata(path, std::collections::HashMap::new())
    }
    /// Canonical weights plus research provenance in a single atomic artifact.
    pub fn save_with_metadata(
        &self,
        path: impl AsRef<Path>,
        metadata: std::collections::HashMap<String, String>,
    ) -> Result<()> {
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let temp = tempfile::NamedTempFile::new_in(parent)?;
        let tensors = self
            .layers()
            .map(|l| Ok((format!("jacobians.{l}"), self.jacobian(l)?)))
            .collect::<Result<std::collections::HashMap<_, _>>>()?;
        safetensors::serialize_to_file(&tensors, Some(metadata), temp.path())?;
        temp.as_file().sync_all()?;
        temp.persist(path).map_err(|e| LensError::Io(e.error))?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    }
    /// Prompt-weighted merge of independently fitted shards.
    pub fn merge(shards: &[(&Self, usize)]) -> Result<(Self, usize)> {
        let (first, _) = shards
            .first()
            .ok_or_else(|| LensError::Invalid("merge needs shards".into()))?;
        let mut total = 0usize;
        let mut means = BTreeMap::new();
        for &(registry, count) in shards {
            if count == 0 || registry.width != first.width || !registry.layers().eq(first.layers())
            {
                return Err(LensError::Invalid(
                    "incompatible or empty fitting shards".into(),
                ));
            }
            total = total
                .checked_add(count)
                .ok_or_else(|| LensError::Invalid("prompt count overflow".into()))?;
            for layer in registry.layers() {
                let weighted = registry
                    .jacobian(layer)?
                    .to_dtype(DType::F32)?
                    .affine(count as f64, 0.)?;
                let next = match means.remove(&layer) {
                    None => weighted,
                    Some(previous) => (previous + weighted)?,
                };
                means.insert(layer, next);
            }
        }
        let means = means
            .into_iter()
            .map(|(l, j)| Ok((l, j.affine(1. / total as f64, 0.)?)))
            .collect::<Result<_>>()?;
        Ok((Self::new(means)?, total))
    }
    /// Selected rows of W_U J, without constructing a full vocabulary dictionary.
    pub fn token_directions(
        &self,
        layer: usize,
        head: &Tensor,
        token_ids: &[u32],
    ) -> Result<Tensor> {
        let (vocab, d) = head.dims2()?;
        if d != self.width
            || token_ids.is_empty()
            || token_ids.iter().any(|&id| id as usize >= vocab)
        {
            return Err(LensError::Invalid("invalid head/token directions".into()));
        }
        let ids = Tensor::new(token_ids, head.device())?;
        let j = self.jacobian(layer)?;
        Ok(head
            .index_select(&ids, 0)?
            .to_dtype(j.dtype())?
            .matmul(&j)?)
    }
}
