//! Inspectable layer/position slices with full-vocabulary tracked-token ranks.
use crate::{JacobianLensRegistry, LensError, LensModel, Result};
use candle_core::{DType, Device, Tensor};
use jlens_research::slice::{Cell, Slice, SliceConfig};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelSlice {
    pub slice: Slice,
    pub context_token_ids: Vec<u32>,
    pub context_tokens: Vec<String>,
    pub vocabulary: BTreeMap<usize, String>,
}
fn decode(tokenizer: &Tokenizer, id: u32) -> Result<String> {
    tokenizer
        .decode(&[id], false)
        .map_err(|e| LensError::Tokenizer(e.to_string()))
}
fn row_ranks(row: &[f32]) -> Result<Vec<usize>> {
    if row.iter().any(|x| !x.is_finite()) {
        return Err(LensError::Invalid("nonfinite slice logits".into()));
    }
    let mut order: Vec<_> = (0..row.len()).collect();
    order.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    let mut ranks = vec![0; row.len()];
    for (rank, id) in order.into_iter().enumerate() {
        ranks[id] = rank;
    }
    Ok(ranks)
}
/// Two decoding passes bound peak vocabulary memory to one layer. The first
/// selects displayed tokens; the second records ranks for their global union.
/// Final row is always the actual model output, never a fitted final-layer lens.
pub fn compute_slice(
    model: &dyn LensModel,
    registry: &JacobianLensRegistry,
    tokenizer: &Tokenizer,
    ids: &Tensor,
    config: &SliceConfig,
) -> Result<ModelSlice> {
    let (batch, seq) = ids.dims2()?;
    if batch != 1
        || seq == 0
        || config.layer_stride == 0
        || config.last_n == Some(0)
        || config.top_n == 0
    {
        return Err(LensError::Invalid("invalid slice shape/settings".into()));
    }
    let vocab = model.dimensions().validate()?.vocab_size;
    if config.top_n > vocab || config.pinned.iter().any(|&id| id >= vocab) {
        return Err(LensError::Invalid("invalid slice top_n/pinned id".into()));
    }
    if let Some(mask) = &config.display_mask
        && (mask.len() != vocab || mask.iter().filter(|&&v| v).count() < config.top_n)
    {
        return Err(LensError::Invalid("insufficient display tokens".into()));
    }
    let fitted: Vec<_> = registry.layers().collect();
    let final_layer = model.dimensions().layer_count - 1;
    if fitted.is_empty() || fitted.iter().any(|&l| l >= model.dimensions().layer_count) {
        return Err(LensError::Invalid("invalid slice layers".into()));
    }
    let mut layers: Vec<_> = fitted
        .iter()
        .step_by(config.layer_stride)
        .copied()
        .collect();
    for &l in [fitted.last().unwrap(), &final_layer] {
        if !layers.contains(&l) {
            layers.push(l);
        }
    }
    layers.sort_unstable();
    let trace = model.capture(ids, &layers, &[])?;
    let offset = seq.saturating_sub(config.last_n.unwrap_or(seq));
    let n = seq - offset;
    let logits = |layer: usize| -> Result<Vec<Vec<f32>>> {
        let h = trace
            .residuals
            .get(&layer)
            .ok_or(LensError::MissingLayer(layer))?
            .narrow(1, offset, n)?
            .contiguous()?;
        let logits = if layer == final_layer {
            model.logits(&h)?
        } else {
            registry.decode_jspace(layer, &h, model)?
        };
        Ok(logits
            .to_dtype(DType::F32)?
            .reshape((n, vocab))?
            .to_vec2::<f32>()?)
    };
    let mut cells = BTreeMap::new();
    let mut frequencies = BTreeMap::<usize, usize>::new();
    for &layer in &layers {
        for (position, row) in logits(layer)?.iter().enumerate() {
            let ranks = row_ranks(row)?;
            let mut selected: Vec<_> = (0..vocab)
                .filter(|&id| config.display_mask.as_ref().is_none_or(|m| m[id]))
                .collect();
            let cmp = |a: &usize, b: &usize| ranks[*a].cmp(&ranks[*b]);
            if config.top_n < selected.len() {
                selected.select_nth_unstable_by(config.top_n, cmp);
                selected.truncate(config.top_n);
            }
            selected.sort_unstable_by(cmp);
            for &id in &selected {
                *frequencies.entry(id).or_default() += 1;
            }
            cells.insert(
                (position, layer),
                Cell {
                    position: position + offset,
                    layer,
                    top_ranks: selected.iter().map(|&id| ranks[id]).collect(),
                    top_ids: selected,
                    tracked_ranks: Vec::new(),
                },
            );
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
    if !tracked_ids.is_empty() {
        for &layer in &layers {
            for (position, row) in logits(layer)?.iter().enumerate() {
                let ranks = row_ranks(row)?;
                cells.get_mut(&(position, layer)).unwrap().tracked_ranks =
                    tracked_ids.iter().map(|&id| ranks[id]).collect();
            }
        }
    }
    let mut vocabulary = BTreeMap::new();
    for id in cells
        .values()
        .flat_map(|c| c.top_ids.iter())
        .chain(tracked_ids.iter())
    {
        if !vocabulary.contains_key(id) {
            vocabulary.insert(*id, decode(tokenizer, *id as u32)?);
        }
    }
    let context_token_ids = ids
        .squeeze(0)?
        .to_dtype(DType::U32)?
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    let context_tokens = context_token_ids
        .iter()
        .map(|&id| decode(tokenizer, id))
        .collect::<Result<_>>()?;
    Ok(ModelSlice {
        slice: Slice {
            layers,
            context_len: seq,
            context_offset: offset,
            tracked_ids,
            pinned: config.pinned.iter().copied().collect(),
            vocab_size: vocab,
            cells: cells.into_values().collect(),
        },
        context_token_ids,
        context_tokens,
        vocabulary,
    })
}
/// Standalone HTML; no CDN, model data upload, or external scripts. JSON escapes
/// '<' so prompt text cannot terminate its inert script element.
pub fn build_page(data: &ModelSlice, title: &str) -> Result<String> {
    let json = serde_json::to_string(data)
        .map_err(|e| LensError::Invalid(e.to_string()))?
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026");
    let title = title
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;");
    Ok(include_str!("slice.html")
        .replace("__TITLE__", &title)
        .replace("__DATA__", &json))
}
pub fn save_page(data: &ModelSlice, title: &str, path: impl AsRef<Path>) -> Result<()> {
    std::fs::write(path, build_page(data, title)?)?;
    Ok(())
}

/// Reference-style wordlike display filter. It affects displayed tokens only;
/// full-vocabulary ranks remain unchanged. Cache the result per tokenizer.
pub fn meaningful_token_mask(tokenizer: &Tokenizer, vocab_size: usize) -> Vec<bool> {
    (0..vocab_size)
        .map(|id| {
            let Ok(text) = tokenizer.decode(&[id as u32], false) else {
                return false;
            };
            let text = text.trim();
            let chars: Vec<_> = text.chars().collect();
            !chars.is_empty()
                && !text.contains("<|")
                && !(text.starts_with('<') && text.ends_with('>'))
                && chars.iter().enumerate().all(|(p, c)| {
                    c.is_alphanumeric()
                        || (p > 0 && p + 1 < chars.len() && matches!(c, '\'' | '-' | '’'))
                })
        })
        .collect()
}
