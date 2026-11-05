//! Layer/position slice computation with full-vocabulary ranks and final output.
use crate::{Matrix, Result, ensure, fitting::Lens, metrics::ranks, model::DifferentiableDecoder};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct SliceConfig {
    pub top_n: usize,
    pub max_tracked: Option<usize>,
    pub pinned: BTreeSet<usize>,
    pub layer_stride: usize,
    pub last_n: Option<usize>,
    pub display_mask: Option<Vec<bool>>,
}
impl Default for SliceConfig {
    fn default() -> Self {
        Self {
            top_n: 10,
            max_tracked: None,
            pinned: BTreeSet::new(),
            layer_stride: 1,
            last_n: None,
            display_mask: None,
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Cell {
    pub position: usize,
    pub layer: usize,
    pub top_ids: Vec<usize>,
    pub top_ranks: Vec<usize>,
    pub tracked_ranks: Vec<usize>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Slice {
    pub layers: Vec<usize>,
    pub context_len: usize,
    pub context_offset: usize,
    pub tracked_ids: Vec<usize>,
    pub pinned: Vec<usize>,
    pub vocab_size: usize,
    pub cells: Vec<Cell>,
}

pub type AppliedLens = (BTreeMap<usize, Matrix<f64>>, Matrix<f64>);

pub fn apply(
    model: &impl DifferentiableDecoder,
    lens: &Lens,
    input: &Matrix<f64>,
    layers: &[usize],
    positions: Option<&[isize]>,
    use_jacobian: bool,
) -> Result<AppliedLens> {
    lens.validate()?;
    ensure(
        lens.model_id == model.fingerprint()?,
        "lens/model fingerprint mismatch",
    )?;
    ensure(
        lens.target == model.num_layers() - 1,
        "readout requires final-layer-targeted lens",
    )?;
    ensure(
        !layers.is_empty() && layers.iter().all(|&l| l < model.num_layers()),
        "invalid readout layers",
    )?;
    let states = model.residuals(input)?;
    let n = input.nrows();
    let indices: Vec<usize> = match positions {
        None => (0..n).collect(),
        Some(ps) => ps
            .iter()
            .map(|&p| {
                let p = if p < 0 { n as isize + p } else { p };
                ensure(p >= 0 && p < n as isize, "position out of range")?;
                Ok(p as usize)
            })
            .collect::<Result<_>>()?,
    };
    ensure(!indices.is_empty(), "empty position selection")?;
    let select =
        |h: &Matrix<f64>| Matrix::from_fn(indices.len(), h.ncols(), |r, c| h[(indices[r], c)]);
    let actual = model.unembed(&select(states.last().unwrap()))?;
    let mut logits = BTreeMap::new();
    for &l in layers {
        let h = select(&states[l]);
        let projected = if use_jacobian {
            lens.transport(l, &h)?
        } else {
            h
        };
        logits.insert(l, model.unembed(&projected)?);
    }
    Ok((logits, actual))
}

pub fn compute_slice(
    model: &impl DifferentiableDecoder,
    lens: &Lens,
    input: &Matrix<f64>,
    config: &SliceConfig,
) -> Result<Slice> {
    ensure(
        config.layer_stride > 0 && config.top_n > 0 && config.last_n != Some(0),
        "invalid slice settings",
    )?;
    let fitted: Vec<_> = lens.jacobians.keys().copied().collect();
    ensure(!fitted.is_empty(), "empty lens")?;
    let mut layers: Vec<_> = fitted
        .iter()
        .step_by(config.layer_stride)
        .copied()
        .collect();
    let last = *fitted.last().unwrap();
    if !layers.contains(&last) {
        layers.push(last);
    }
    let offset = input
        .nrows()
        .saturating_sub(config.last_n.unwrap_or(input.nrows()));
    let positions: Vec<_> = (offset..input.nrows()).map(|i| i as isize).collect();
    let (mut logits, actual) = apply(model, lens, input, &layers, Some(&positions), true)?;
    let final_layer = model.num_layers() - 1;
    layers.push(final_layer);
    logits.insert(final_layer, actual);
    let vocab = logits[&final_layer].ncols();
    ensure(
        config.top_n <= vocab && config.pinned.iter().all(|&id| id < vocab),
        "invalid top_n/pinned token",
    )?;
    if let Some(mask) = &config.display_mask {
        ensure(
            mask.len() == vocab && mask.iter().filter(|&&v| v).count() >= config.top_n,
            "insufficient displayable tokens",
        )?;
    }
    let mut frequencies = BTreeMap::<usize, usize>::new();
    let mut cell_ranks = Vec::new();
    let mut cells = Vec::new();
    for p in 0..positions.len() {
        for &layer in &layers {
            let values: Vec<_> = logits[&layer].row(p).iter().copied().collect();
            let ranks = ranks(&values)?;
            let mut ids: Vec<_> = (0..vocab)
                .filter(|&id| config.display_mask.as_ref().is_none_or(|m| m[id]))
                .collect();
            ids.sort_unstable_by_key(|&id| ranks[id]);
            ids.truncate(config.top_n);
            for &id in &ids {
                *frequencies.entry(id).or_default() += 1;
            }
            cells.push(Cell {
                position: offset + p,
                layer,
                top_ranks: ids.iter().map(|&id| ranks[id]).collect(),
                top_ids: ids,
                tracked_ranks: Vec::new(),
            });
            cell_ranks.push(ranks);
        }
    }
    let mut frequent: Vec<_> = frequencies.into_iter().collect();
    frequent.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    frequent.truncate(config.max_tracked.unwrap_or(frequent.len()));
    let tracked: BTreeSet<_> = frequent
        .into_iter()
        .map(|(id, _)| id)
        .chain(config.pinned.iter().copied())
        .collect();
    let tracked_ids: Vec<_> = tracked.into_iter().collect();
    for (cell, ranks) in cells.iter_mut().zip(cell_ranks) {
        cell.tracked_ranks = tracked_ids.iter().map(|&id| ranks[id]).collect();
    }
    Ok(Slice {
        layers,
        context_len: input.nrows(),
        context_offset: offset,
        tracked_ids,
        pinned: config.pinned.iter().copied().collect(),
        vocab_size: vocab,
        cells,
    })
}
