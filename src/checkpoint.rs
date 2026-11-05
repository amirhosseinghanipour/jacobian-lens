//! Atomic, resumable corpus fitting. Model and corpus identities are explicit.
use crate::{
    FittingConfig, JacobianEstimator, JacobianLensRegistry, LensError, Result, estimate_prompt,
};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    path::Path,
};
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Progress {
    version: u32,
    config: FittingConfig,
    width: usize,
    model_id: String,
    corpus_id: String,
    next_index: usize,
    successes: usize,
    skipped: usize,
}
pub struct FitCheckpoint {
    progress: Progress,
    sums: BTreeMap<usize, Tensor>,
}
impl FitCheckpoint {
    /// Identity strings should identify immutable checkpoint and ordered corpus
    /// hashes. Reusing an identity for different artifacts invalidates provenance.
    pub fn new(
        config: FittingConfig,
        width: usize,
        model_id: String,
        corpus_id: String,
    ) -> Result<Self> {
        if width == 0 || model_id.is_empty() || corpus_id.is_empty() {
            return Err(LensError::Invalid(
                "fitting requires width and artifact identities".into(),
            ));
        }
        Ok(Self {
            progress: Progress {
                version: 1,
                config,
                width,
                model_id,
                corpus_id,
                next_index: 0,
                successes: 0,
                skipped: 0,
            },
            sums: BTreeMap::new(),
        })
    }
    pub fn next_index(&self) -> usize {
        self.progress.next_index
    }
    pub fn successes(&self) -> usize {
        self.progress.successes
    }
    pub fn skipped(&self) -> usize {
        self.progress.skipped
    }
    /// Processes [next_index,end); checkpoint saves after each consumed prompt.
    /// Failed model evaluations never advance the cursor. Short prompts do.
    pub fn run_until(
        &mut self,
        model: &(impl JacobianEstimator + ?Sized),
        prompts: &[Tensor],
        end: usize,
        checkpoint: Option<&Path>,
    ) -> Result<()> {
        self.progress
            .config
            .validate(model.fitting_dimensions().1)?;
        if model.fitting_dimensions().0 != self.progress.width
            || end > prompts.len()
            || end < self.progress.next_index
        {
            return Err(LensError::Invalid(
                "invalid fitting dimensions/cursor".into(),
            ));
        }
        while self.progress.next_index < end {
            let ids = &prompts[self.progress.next_index];
            let (batch, seq) = ids.dims2()?;
            if batch != 1 {
                return Err(LensError::Invalid(
                    "fitting expects single unpadded prompts".into(),
                ));
            }
            if seq.min(self.progress.config.max_seq_len).saturating_sub(1)
                <= self.progress.config.skip_first
            {
                self.progress.skipped += 1;
            } else {
                let estimate = estimate_prompt(model, ids, &self.progress.config)?;
                let mut next = BTreeMap::new();
                for (layer, j) in estimate.jacobians {
                    let sum = match self.sums.get(&layer) {
                        Some(previous) => (previous + &j)?,
                        None => j,
                    };
                    next.insert(layer, sum);
                }
                self.sums = next;
                self.progress.successes += 1;
            }
            self.progress.next_index += 1;
            if let Some(path) = checkpoint {
                self.save(path)?;
            }
        }
        Ok(())
    }
    pub fn registry(&self) -> Result<JacobianLensRegistry> {
        if self.progress.successes == 0 {
            return Err(LensError::Invalid("no valid fitting prompts".into()));
        }
        let weights = self
            .sums
            .iter()
            .map(|(&l, sum)| Ok((l, sum.affine(1. / self.progress.successes as f64, 0.)?)))
            .collect::<Result<_>>()?;
        JacobianLensRegistry::new(weights)
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let temp = tempfile::NamedTempFile::new_in(parent)?;
        let json =
            serde_json::to_string(&self.progress).map_err(|e| LensError::Invalid(e.to_string()))?;
        let metadata = HashMap::from([("fit_state".to_string(), json)]);
        let weights: HashMap<_, _> = self
            .sums
            .iter()
            .map(|(&l, j)| (format!("sum.{l}"), j.clone()))
            .collect();
        safetensors::serialize_to_file(&weights, Some(metadata), temp.path())?;
        temp.as_file().sync_all()?;
        temp.persist(path).map_err(|e| LensError::Io(e.error))?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    }
    pub fn load(
        path: impl AsRef<Path>,
        expected: &FittingConfig,
        model_id: &str,
        corpus_id: &str,
        device: &Device,
    ) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)?;
        let json = header
            .metadata()
            .as_ref()
            .and_then(|m| m.get("fit_state"))
            .ok_or_else(|| LensError::Invalid("missing fitting metadata".into()))?;
        let progress: Progress =
            serde_json::from_str(json).map_err(|e| LensError::Invalid(e.to_string()))?;
        if progress.version != 1
            || &progress.config != expected
            || progress.model_id != model_id
            || progress.corpus_id != corpus_id
            || progress.width == 0
            || progress.successes.checked_add(progress.skipped) != Some(progress.next_index)
        {
            return Err(LensError::Invalid(
                "checkpoint configuration/identity/cursor mismatch".into(),
            ));
        }
        let tensors = candle_core::safetensors::load_buffer(&bytes, device)?;
        let mut sums = BTreeMap::new();
        for (name, j) in tensors {
            let layer = name
                .strip_prefix("sum.")
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or_else(|| LensError::Invalid("invalid checkpoint tensor name".into()))?;
            if j.dims() != [progress.width, progress.width]
                || j.dtype() != DType::F32
                || !progress.config.source_layers.contains(&layer)
                || sums.contains_key(&layer)
            {
                return Err(LensError::Invalid(
                    "checkpoint tensor schema mismatch".into(),
                ));
            }
            if j.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .any(|v| !v.is_finite())
            {
                return Err(LensError::Invalid("nonfinite checkpoint sum".into()));
            }
            sums.insert(layer, j);
        }
        if (progress.successes == 0 && !sums.is_empty())
            || (progress.successes > 0
                && !sums
                    .keys()
                    .copied()
                    .eq(progress.config.source_layers.iter().copied()))
        {
            return Err(LensError::Invalid("incomplete checkpoint sums".into()));
        }
        Ok(Self { progress, sums })
    }
}
