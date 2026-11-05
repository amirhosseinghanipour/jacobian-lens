//! Exact source-mean/target-sum estimator, expressed with JVPs rather than VJPs.
//! Linearity makes simultaneous source perturbations algebraically equivalent
//! to the Python implementation's simultaneous target cotangents.
use crate::{Matrix, Result, ensure, finite, model::DifferentiableDecoder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ops::Range, path::Path};

pub const SKIP_FIRST: usize = 16;
pub fn valid_positions(seq_len: usize, skip_first: usize) -> Result<Range<usize>> {
    ensure(
        seq_len.saturating_sub(1) > skip_first,
        "prompt too short for valid positions",
    )?;
    Ok(skip_first..seq_len - 1)
}
pub fn resolve_layers(
    n: usize,
    sources: Option<&[isize]>,
    target: Option<isize>,
) -> Result<(Vec<usize>, usize)> {
    ensure(n > 0 && n <= isize::MAX as usize, "invalid layer count")?;
    let index = |x: isize| -> Result<usize> {
        let x = if x < 0 { n as isize + x } else { x };
        ensure(x >= 0 && x < n as isize, "layer out of range")?;
        Ok(x as usize)
    };
    let target = index(target.unwrap_or(-1))?;
    let mut sources = match sources {
        Some(s) => s.iter().map(|&x| index(x)).collect::<Result<Vec<_>>>()?,
        None => (0..target).collect(),
    };
    sources.sort_unstable();
    sources.dedup();
    ensure(
        !sources.is_empty() && sources.iter().all(|&l| l < target),
        "sources must be nonempty and before target",
    )?;
    Ok((sources, target))
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitConfig {
    pub sources: Vec<usize>,
    pub target: usize,
    pub skip_first: usize,
    pub max_seq_len: usize,
}
impl FitConfig {
    pub fn validate(&self, n_layers: usize) -> Result<()> {
        ensure(
            !self.sources.is_empty() && self.target < n_layers,
            "invalid source/target layers",
        )?;
        ensure(
            self.sources.windows(2).all(|w| w[0] < w[1])
                && self.sources.iter().all(|&s| s < self.target),
            "sources must be sorted, unique, before target",
        )?;
        ensure(
            self.max_seq_len.saturating_sub(1) > self.skip_first,
            "max_seq_len excludes all positions",
        )
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Lens {
    pub jacobians: BTreeMap<usize, Matrix<f64>>,
    pub n_prompts: usize,
    pub model_id: String,
    pub target: usize,
}
impl Lens {
    pub fn validate(&self) -> Result<usize> {
        ensure(
            self.n_prompts > 0 && !self.jacobians.is_empty() && !self.model_id.is_empty(),
            "empty lens",
        )?;
        let d = self.jacobians.values().next().unwrap().nrows();
        ensure(d > 0, "empty Jacobian")?;
        for (&layer, j) in &self.jacobians {
            ensure(
                j.shape() == (d, d) && layer < self.target,
                "invalid Jacobian shape/layer",
            )?;
            finite(j)?;
        }
        Ok(d)
    }
    pub fn transport(&self, layer: usize, h: &Matrix<f64>) -> Result<Matrix<f64>> {
        let d = self.validate()?;
        ensure(h.ncols() == d, "hidden width mismatch")?;
        finite(h)?;
        let j = self
            .jacobians
            .get(&layer)
            .ok_or_else(|| crate::Error::Invalid(format!("missing layer {layer}")))?;
        Ok(h * j.transpose())
    }
    pub fn merge(lenses: &[Self]) -> Result<Self> {
        ensure(!lenses.is_empty(), "merge requires lenses")?;
        let first = &lenses[0];
        let d = first.validate()?;
        let mut count = 0usize;
        let mut sums: BTreeMap<_, _> = first
            .jacobians
            .keys()
            .map(|&l| (l, Matrix::zeros(d, d)))
            .collect();
        for lens in lenses {
            ensure(
                lens.validate()? == d
                    && lens.model_id == first.model_id
                    && lens.target == first.target
                    && lens.jacobians.keys().eq(first.jacobians.keys()),
                "incompatible lenses",
            )?;
            count = count
                .checked_add(lens.n_prompts)
                .ok_or_else(|| crate::Error::Invalid("prompt count overflow".into()))?;
            for (&l, j) in &lens.jacobians {
                *sums.get_mut(&l).unwrap() += j * lens.n_prompts as f64;
            }
        }
        for j in sums.values_mut() {
            *j /= count as f64;
        }
        let lens = Self {
            jacobians: sums,
            n_prompts: count,
            model_id: first.model_id.clone(),
            target: first.target,
        };
        lens.validate()?;
        Ok(lens)
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        crate::save_json(path, self)
    }
    pub fn load(path: &Path) -> Result<Self> {
        let lens: Self = crate::load_json(path)?;
        lens.validate()?;
        Ok(lens)
    }
}

pub fn jacobian_for_prompt(
    model: &impl DifferentiableDecoder,
    prompt: &Matrix<f64>,
    config: &FitConfig,
) -> Result<BTreeMap<usize, Matrix<f64>>> {
    config.validate(model.num_layers())?;
    ensure(prompt.ncols() == model.width(), "prompt width mismatch")?;
    finite(prompt)?;
    let n = prompt.nrows().min(config.max_seq_len);
    let valid = valid_positions(n, config.skip_first)?;
    let h = prompt.rows(0, n).into_owned();
    let states = model.residuals(&h)?;
    ensure(
        states.len() == model.num_layers(),
        "decoder returned wrong layer count",
    )?;
    let d = model.width();
    let mut result = BTreeMap::new();
    for &layer in &config.sources {
        ensure(
            states[layer].shape() == (n, d),
            "decoder residual shape mismatch",
        )?;
        let mut j = Matrix::zeros(d, d);
        for input_dim in 0..d {
            let mut tangent = Matrix::zeros(n, d);
            for p in valid.clone() {
                tangent[(p, input_dim)] = 1.;
            }
            let (_, dy) = model.suffix_jvp(layer, config.target, &states[layer], &tangent)?;
            ensure(dy.shape() == (n, d), "decoder JVP shape mismatch")?;
            finite(&dy)?;
            for output_dim in 0..d {
                j[(output_dim, input_dim)] =
                    valid.clone().map(|p| dy[(p, output_dim)]).sum::<f64>() / valid.len() as f64;
            }
        }
        result.insert(layer, j);
    }
    Ok(result)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitState {
    pub config: FitConfig,
    pub model_id: String,
    pub corpus_sha256: String,
    pub next_index: usize,
    pub successes: usize,
    pub skipped: usize,
    pub sums: BTreeMap<usize, Matrix<f64>>,
}
fn corpus_hash(prompts: &[Matrix<f64>]) -> Result<String> {
    let mut hasher = Sha256::new();
    for p in prompts {
        finite(p)?;
        hasher.update((p.nrows() as u64).to_le_bytes());
        hasher.update((p.ncols() as u64).to_le_bytes());
        for x in p.iter() {
            hasher.update(x.to_le_bytes());
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}
impl FitState {
    pub fn new(
        model: &impl DifferentiableDecoder,
        prompts: &[Matrix<f64>],
        config: FitConfig,
    ) -> Result<Self> {
        config.validate(model.num_layers())?;
        let sums = config
            .sources
            .iter()
            .map(|&l| (l, Matrix::zeros(model.width(), model.width())))
            .collect();
        Ok(Self {
            config,
            model_id: model.fingerprint()?,
            corpus_sha256: corpus_hash(prompts)?,
            next_index: 0,
            successes: 0,
            skipped: 0,
            sums,
        })
    }
    /// An exclusive end cursor permits interruption/resume with unchanged corpus.
    /// Commits each prompt transactionally; skips only too-short sequences.
    pub fn run_until(
        &mut self,
        model: &impl DifferentiableDecoder,
        prompts: &[Matrix<f64>],
        end: usize,
        checkpoint: Option<&Path>,
    ) -> Result<()> {
        self.config.validate(model.num_layers())?;
        ensure(
            self.model_id == model.fingerprint()? && self.corpus_sha256 == corpus_hash(prompts)?,
            "checkpoint model/corpus mismatch",
        )?;
        ensure(
            end <= prompts.len()
                && self.next_index <= end
                && self.successes.checked_add(self.skipped) == Some(self.next_index),
            "invalid resume cursor/count",
        )?;
        ensure(
            self.sums.keys().copied().collect::<Vec<_>>() == self.config.sources,
            "checkpoint source mismatch",
        )?;
        for sum in self.sums.values() {
            ensure(
                sum.shape() == (model.width(), model.width()),
                "checkpoint dimension mismatch",
            )?;
            finite(sum)?;
        }
        while self.next_index < end {
            let prompt = &prompts[self.next_index];
            ensure(prompt.ncols() == model.width(), "prompt width mismatch")?;
            if valid_positions(
                prompt.nrows().min(self.config.max_seq_len),
                self.config.skip_first,
            )
            .is_err()
            {
                self.skipped += 1;
            } else {
                let estimates = jacobian_for_prompt(model, prompt, &self.config)?;
                let mut sums = self.sums.clone();
                for (&l, j) in &estimates {
                    *sums.get_mut(&l).unwrap() += j;
                    finite(&sums[&l])?;
                }
                self.sums = sums;
                self.successes += 1;
            }
            self.next_index += 1;
            if let Some(path) = checkpoint {
                crate::save_json(path, self)?;
            }
        }
        Ok(())
    }
    pub fn lens(&self) -> Result<Lens> {
        ensure(self.successes > 0, "no valid fitting prompts")?;
        let lens = Lens {
            jacobians: self
                .sums
                .iter()
                .map(|(&l, j)| (l, j / self.successes as f64))
                .collect(),
            n_prompts: self.successes,
            model_id: self.model_id.clone(),
            target: self.config.target,
        };
        lens.validate()?;
        Ok(lens)
    }
    pub fn resume(path: &Path, expected_config: &FitConfig) -> Result<Self> {
        let state: Self = crate::load_json(path)?;
        ensure(
            &state.config == expected_config,
            "checkpoint configuration mismatch",
        )?;
        Ok(state)
    }
}
