//! Released lens-quality tasks: intermediate-concept recall across layers.
//! Target answers locate readout positions and are never scored as concepts.
use crate::{JacobianLensRegistry, LensError, LensModel, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use tokenizers::Tokenizer;
#[derive(Debug, Clone, Deserialize)]
pub struct EvaluationItem {
    pub name: String,
    pub prompt: String,
    #[serde(default)]
    pub target: Option<String>,
    pub intermediates: Vec<String>,
}
#[derive(Debug, Clone, Deserialize)]
pub struct EvaluationDataset {
    pub items: Vec<EvaluationItem>,
}
#[derive(Debug, Clone, Copy)]
pub enum ReadoutPosition {
    BeforeTarget,
    LastNewline,
    LastToken,
}
#[derive(Debug, Serialize)]
pub struct ConceptRank {
    pub concept: String,
    pub token_ids: Vec<u32>,
    pub minimum_rank_zero_based: Option<usize>,
}
#[derive(Debug, Serialize)]
pub struct ItemEvaluation {
    pub name: String,
    pub readout_position: usize,
    pub concepts: Vec<ConceptRank>,
}
#[derive(Debug, Serialize)]
pub struct Recall {
    pub k: usize,
    pub pass_at_k: f64,
    pub concept_coverage: f64,
    pub items: usize,
}
fn token_error(e: tokenizers::Error) -> LensError {
    LensError::Tokenizer(e.to_string())
}

/// Resolve token position using tokenizer offsets. For target-based tasks,
/// tokenize prompt+target together and remove every token overlapping target;
/// this avoids treating a trailing-space fragment as a standalone prompt token.
pub fn encode_readout(
    tokenizer: &Tokenizer,
    item: &EvaluationItem,
    position: ReadoutPosition,
) -> Result<Vec<u32>> {
    let combined = match position {
        ReadoutPosition::BeforeTarget => format!(
            "{}{}",
            item.prompt,
            item.target
                .as_deref()
                .ok_or_else(|| LensError::Invalid("target-based item has no target".into()))?
        ),
        _ => item.prompt.clone(),
    };
    let encoded = tokenizer.encode(combined, false).map_err(token_error)?;
    let end = match position {
        ReadoutPosition::BeforeTarget => encoded
            .get_offsets()
            .iter()
            .position(|&(_, end)| end > item.prompt.len())
            .ok_or_else(|| LensError::Invalid("target has no token span".into()))?,
        ReadoutPosition::LastNewline => {
            let newline = item
                .prompt
                .rfind('\n')
                .ok_or_else(|| LensError::Invalid("poetry item has no newline".into()))?;
            encoded
                .get_offsets()
                .iter()
                .rposition(|&(start, end)| start <= newline && newline < end)
                .ok_or_else(|| LensError::Invalid("newline token missing".into()))?
                + 1
        }
        ReadoutPosition::LastToken => encoded.len(),
    };
    if end == 0 {
        return Err(LensError::Invalid(
            "no tokens before readout boundary".into(),
        ));
    }
    Ok(encoded.get_ids()[..end].to_vec())
}
/// Single-token spellings only; multi-token concepts are reported as uncovered.
/// Synonyms must be supplied explicitly, e.g. operation symbols/number words.
pub fn concept_tokens(
    tokenizer: &Tokenizer,
    concept: &str,
    synonyms: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<u32>> {
    let mut words = vec![concept.to_string()];
    if let Some(extra) = synonyms.get(concept) {
        words.extend(extra.iter().cloned());
    }
    let mut ids = BTreeSet::new();
    for word in words {
        for variant in [word.clone(), format!(" {word}")] {
            let encoding = tokenizer.encode(variant, false).map_err(token_error)?;
            if encoding.len() == 1 {
                ids.insert(encoding.get_ids()[0]);
            }
        }
    }
    Ok(ids.into_iter().collect())
}
/// Full-vocabulary ranks, deterministic ties by ascending token ID.
pub fn evaluate_item(
    model: &dyn LensModel,
    registry: &JacobianLensRegistry,
    tokenizer: &Tokenizer,
    item: &EvaluationItem,
    position: ReadoutPosition,
    synonyms: &BTreeMap<String, Vec<String>>,
    device: &Device,
) -> Result<ItemEvaluation> {
    if item.intermediates.is_empty() {
        return Err(LensError::Invalid(
            "item has no intermediate concepts".into(),
        ));
    }
    let ids = encode_readout(tokenizer, item, position)?;
    let trace = model.capture(
        &Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?,
        &registry.layers().collect::<Vec<_>>(),
        &[],
    )?;
    let mut concepts = item
        .intermediates
        .iter()
        .map(|concept| {
            Ok(ConceptRank {
                concept: concept.clone(),
                token_ids: concept_tokens(tokenizer, concept, synonyms)?,
                minimum_rank_zero_based: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let p = ids.len() - 1;
    for layer in registry.layers() {
        let state = trace
            .residuals
            .get(&layer)
            .ok_or(LensError::MissingLayer(layer))?
            .narrow(1, p, 1)?
            .contiguous()?;
        let scores = registry
            .decode_jspace(layer, &state, model)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if scores.iter().any(|v| !v.is_finite()) {
            return Err(LensError::Invalid("nonfinite evaluation logits".into()));
        }
        for concept in &mut concepts {
            for &id in &concept.token_ids {
                let id = id as usize;
                if id >= scores.len() {
                    return Err(LensError::Invalid(
                        "tokenizer/head vocabulary mismatch".into(),
                    ));
                }
                let rank = scores
                    .iter()
                    .enumerate()
                    .filter(|&(other, &v)| v > scores[id] || (v == scores[id] && other < id))
                    .count();
                concept.minimum_rank_zero_based = Some(
                    concept
                        .minimum_rank_zero_based
                        .map_or(rank, |r| r.min(rank)),
                );
            }
        }
    }
    Ok(ItemEvaluation {
        name: item.name.clone(),
        readout_position: p,
        concepts,
    })
}
/// Item-macro-averaged fraction of intermediate concepts with best layer rank < k.
/// Uncovered concepts remain misses; coverage is always returned alongside score.
pub fn pass_at_k(items: &[ItemEvaluation], k: usize) -> Result<Recall> {
    if items.is_empty() || k == 0 || items.iter().any(|i| i.concepts.is_empty()) {
        return Err(LensError::Invalid("empty evaluation or invalid k".into()));
    }
    let score = items
        .iter()
        .map(|i| {
            i.concepts
                .iter()
                .filter(|c| c.minimum_rank_zero_based.is_some_and(|r| r < k))
                .count() as f64
                / i.concepts.len() as f64
        })
        .sum::<f64>()
        / items.len() as f64;
    let covered = items
        .iter()
        .map(|i| {
            i.concepts
                .iter()
                .filter(|c| !c.token_ids.is_empty())
                .count()
        })
        .sum::<usize>();
    let total = items.iter().map(|i| i.concepts.len()).sum::<usize>();
    Ok(Recall {
        k,
        pass_at_k: score,
        concept_coverage: covered as f64 / total as f64,
        items: items.len(),
    })
}
