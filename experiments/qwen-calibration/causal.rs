use candle_core::{Device, Tensor};
use jacobian_lens::{JacobianLensRegistry, LensError, LensFrame, Result, Unembed};
use jlens_qwen_reference::Qwen2;
use serde::Deserialize;
use std::collections::BTreeMap;
use tokenizers::Tokenizer;
#[derive(Deserialize)]
struct Dataset {
    categories: Vec<Category>,
}
#[derive(Deserialize)]
struct Category {
    name: String,
    args: Vec<String>,
    funcs: Vec<Function>,
}
#[derive(Deserialize)]
struct Function {
    name: String,
    template: String,
    answers: BTreeMap<String, String>,
}
fn error(e: impl std::fmt::Display) -> LensError {
    LensError::Invalid(e.to_string())
}
fn single(t: &Tokenizer, word: &str) -> Result<Option<u32>> {
    for variant in [format!(" {word}"), word.to_owned()] {
        let e = t.encode(variant, false).map_err(error)?;
        if e.len() == 1 {
            return Ok(Some(e.get_ids()[0]));
        }
    }
    Ok(None)
}
fn random(d: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    let mut uniform = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    };
    (0..d)
        .map(|_| {
            let radius = (-2. * uniform().ln()).sqrt();
            (radius * (std::f64::consts::TAU * uniform()).cos()) as f32
        })
        .collect()
}
fn norm(x: &Tensor) -> Result<f64> {
    Ok((x.sqr()?.sum_all()?.to_scalar::<f32>()? as f64).sqrt())
}
fn matched(delta: &Tensor, reference: &Tensor) -> Result<Tensor> {
    let d = norm(delta)?;
    if d == 0. {
        return Err(error("zero control direction"));
    }
    Ok(delta.affine(norm(reference)? / d, 0.)?)
}
fn values(t: &Tensor) -> Result<Vec<f64>> {
    Ok(t.flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f64::from)
        .collect())
}
fn answer(t: &Tokenizer, prompt: &str, label: &str) -> Result<u32> {
    let p = t.encode(prompt, false).map_err(error)?;
    let full = t
        .encode(format!("{prompt} {label}"), false)
        .map_err(error)?;
    if !full.get_ids().starts_with(p.get_ids()) {
        return Err(error("answer boundary changes prefix"));
    }
    full.get_ids()
        .get(p.len())
        .copied()
        .ok_or_else(|| error("empty answer"))
}
pub fn run(
    model: &Qwen2,
    lens: &JacobianLensRegistry,
    t: &Tokenizer,
    device: &Device,
) -> Result<serde_json::Value> {
    let data: Dataset = serde_json::from_slice(&std::fs::read(
        "data/experiments/flexible-generalization.json",
    )?)
    .map_err(error)?;
    let mut results = Vec::new();
    let mut skipped = Vec::new();
    let mut index = 0;
    for category in data.categories {
        for function in category.funcs {
            for (i, source) in category.args.iter().enumerate() {
                let target = &category.args[(i + 1) % category.args.len()];
                let prompt = function.template.replace("{arg}", source);
                let counterprompt = function.template.replace("{arg}", target);
                let key = format!("{}:{}:{}->{}", category.name, function.name, source, target);
                let Some(sid) = single(t, source)? else {
                    skipped.push(
                        serde_json::json!({"item":key,"reason":"source concept is not one token"}),
                    );
                    continue;
                };
                let Some(tid) = single(t, target)? else {
                    skipped.push(
                        serde_json::json!({"item":key,"reason":"target concept is not one token"}),
                    );
                    continue;
                };
                let original_answer = answer(t, &prompt, &function.answers[source])?;
                let counter_answer = answer(t, &prompt, &function.answers[target])?;
                let encoded = t.encode(prompt.as_str(), false).map_err(error)?;
                let ids = Tensor::new(encoded.get_ids(), device)?.unsqueeze(0)?;
                let p = encoded.len() - 1;
                let states = model.residuals(&ids)?;
                let before = values(&model.logits(&states[23].narrow(1, p, 1)?)?)?;
                let cf = t.encode(counterprompt.as_str(), false).map_err(error)?;
                let cfstates =
                    model.residuals(&Tensor::new(cf.get_ids(), device)?.unsqueeze(0)?)?;
                let cfout = values(&model.logits(&cfstates[23].narrow(1, cf.len() - 1, 1)?)?)?;
                let ranks = jacobian_lens::research::metrics::ranks(&before).map_err(error)?;
                let cfranks = jacobian_lens::research::metrics::ranks(&cfout).map_err(error)?;
                for layer in [6, 10, 14, 18, 22] {
                    let frame =
                        LensFrame::from_tokens(lens, layer, model.lm_head(), &[sid, tid], 1e-6)?;
                    let h = states[layer].narrow(1, p, 1)?.contiguous()?;
                    let full = frame.permute(&h, &[1, 0], 1.)?;
                    let delta = (&full - &h)?;
                    let ablated = frame.ablate(&h, 1.)?;
                    let ablation_delta = (&ablated - &h)?;
                    let mut edits = Vec::new();
                    for strength in [0., 0.5, 1.] {
                        edits.push((
                            format!("swap-{strength}"),
                            frame.permute(&h, &[1, 0], strength)?,
                        ));
                    }
                    for seed in 1..=3 {
                        let r = Tensor::from_vec(
                            random(
                                h.elem_count(),
                                20261105 + seed + index * 101 + layer as u64 * 7,
                            ),
                            h.shape(),
                            device,
                        )?;
                        edits.push((
                            format!("random-swap-norm-{seed}"),
                            (&h + matched(&r, &delta)?)?,
                        ));
                    }
                    edits.push(("concept-ablation".into(), ablated));
                    let random_frame = LensFrame::new(
                        Tensor::from_vec(
                            random(2 * h.elem_count(), 991 + index * 101 + layer as u64),
                            (2, h.elem_count()),
                            device,
                        )?,
                        1e-6,
                    )?;
                    let rd = (random_frame.ablate(&h, 1.)? - &h)?;
                    edits.push((
                        "random-span-ablation-matched-norm".into(),
                        (&h + matched(&rd, &ablation_delta)?)?,
                    ));
                    for (condition, edited) in edits {
                        let perturbation_norm = norm(&(&edited - &h)?)?;
                        let state = if p == 0 {
                            edited
                        } else {
                            Tensor::cat(&[states[layer].narrow(1, 0, p)?, edited], 1)?
                        };
                        let out = model.suffix(layer, 23, &state)?;
                        let after = values(&model.logits(&out.narrow(1, p, 1)?)?)?;
                        let m = jacobian_lens::research::metrics::measure_intervention(
                            &before,
                            &after,
                            counter_answer as usize,
                        )
                        .map_err(error)?;
                        let original = jacobian_lens::research::metrics::measure_intervention(
                            &before,
                            &after,
                            original_answer as usize,
                        )
                        .map_err(error)?;
                        let log_odds_change = (after[counter_answer as usize]
                            - after[original_answer as usize])
                            - (before[counter_answer as usize] - before[original_answer as usize]);
                        results.push(serde_json::json!({"item":key,"category":category.name,"function":function.name,"source":source,"counterfactual":target,"original_answer":function.answers[source],"counterfactual_answer":function.answers[target],"original_answer_token":original_answer,"counterfactual_answer_token":counter_answer,"answer_tokens_identical":original_answer==counter_answer,"source_task_answer_rank":ranks[original_answer as usize],"counterfactual_task_answer_rank":cfranks[counter_answer as usize],"layer":layer,"condition":condition,"perturbation_norm":perturbation_norm,"counterfactual_measurement":m,"original_measurement":original,"counterfactual_log_odds_change":log_odds_change}));
                    }
                }
                index += 1;
                eprintln!("causal item {index}: {key}");
            }
        }
    }
    Ok(
        serde_json::json!({"measurements":results,"skipped":skipped,"endpoint":"first continuation token of supplied upstream answer label"}),
    )
}
