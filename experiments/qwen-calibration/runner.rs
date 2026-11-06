#[path = "causal.rs"]
mod causal;
// Reproducible fitting over frozen token windows and matched lens measurements.
use candle_core::{DType, Device, Tensor};
use jacobian_lens::{
    DifferentiableTransformer, FitCheckpoint, FittingConfig, JacobianLensRegistry, LensError,
    Result, Unembed, evaluation::*,
};
use jlens_qwen_reference::Qwen2;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};
use tokenizers::Tokenizer;
#[derive(Deserialize)]
struct Record {
    ids: Vec<u32>,
}
fn err(e: impl std::fmt::Display) -> LensError {
    LensError::Invalid(e.to_string())
}
fn read<T: serde::de::DeserializeOwned>(p: impl AsRef<Path>) -> Result<T> {
    serde_json::from_slice(&std::fs::read(p)?).map_err(err)
}
fn write(p: impl AsRef<Path>, v: &impl serde::Serialize) -> Result<()> {
    jacobian_lens::research::save_json(p.as_ref(), v).map_err(err)
}
fn scores(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}
fn rank(s: &[f32], ids: &[u32]) -> Option<usize> {
    ids.iter()
        .map(|&id| {
            s.iter()
                .enumerate()
                .filter(|&(i, &v)| v > s[id as usize] || (v == s[id as usize] && i < id as usize))
                .count()
        })
        .min()
}
fn checked_lens(
    path: impl AsRef<Path>,
    model_id: &str,
    device: &Device,
) -> Result<JacobianLensRegistry> {
    let bytes = std::fs::read(path.as_ref())?;
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)?;
    if header
        .metadata()
        .as_ref()
        .and_then(|m| m.get("model_id"))
        .map(String::as_str)
        != Some(model_id)
    {
        return Err(err("lens model/tokenizer identity mismatch"));
    }
    JacobianLensRegistry::load(path, device, DType::F32)
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        return Err(err(
            "usage: jlens-qwen-experiment fit MODEL SHARD OUTPUT | eval MODEL LENS_DIRECTORY OUTPUT",
        ));
    }
    let device = match std::env::var("JLENS_DEVICE").as_deref() {
        Ok("cuda") => Device::new_cuda(0)?,
        Ok("cpu") | Err(_) => Device::Cpu,
        _ => return Err(err("invalid JLENS_DEVICE")),
    };
    let model = Qwen2::load(&a[1], &device, DType::F32)?;
    let model_id = jlens_qwen_reference::checkpoint_identity(&a[1])?;
    let tokenizer = Tokenizer::from_file(Path::new(&a[1]).join("tokenizer.json")).map_err(err)?;
    match a[0].as_str() {
        "causal" => {
            let lens = checked_lens(&a[2], &model_id, &device)?;
            let result = causal::run(&model, &lens, &tokenizer, &device)?;
            write(&a[3], &result)?;
        }
        "merge" | "merge-pilot" => {
            let mut shards = Vec::new();
            for i in 0..if a[0] == "merge-pilot" { 10 } else { 110 } {
                let path = Path::new(&a[2]).join(format!("shard-{i:03}.safetensors"));
                let bytes = std::fs::read(&path)?;
                let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)?;
                let metadata = header
                    .metadata()
                    .as_ref()
                    .ok_or_else(|| err("missing shard provenance"))?;
                if metadata.get("model_id") != Some(&model_id)
                    || metadata.get("n_prompts").map(String::as_str) != Some("10")
                {
                    return Err(err("invalid shard model or count"));
                }
                shards.push(JacobianLensRegistry::load(path, &device, DType::F32)?);
            }
            std::fs::create_dir_all(&a[3])?;
            for (name, range) in [
                ("main-100", 0..10),
                ("main-1000", 0..100),
                ("replicate-100", 100..110),
            ] {
                if a[0] == "merge-pilot" && name != "main-100" {
                    continue;
                }
                let count = range.len() * 10;
                let layers = [6, 10, 14, 18, 22]
                    .into_iter()
                    .map(|l| {
                        let mut sum = Tensor::zeros(
                            (model.hidden_size(), model.hidden_size()),
                            DType::F32,
                            &device,
                        )?;
                        for i in range.clone() {
                            sum = (sum + shards[i].jacobian(l)?)?;
                        }
                        Ok((l, sum.affine(10. / count as f64, 0.)?))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?;
                JacobianLensRegistry::new(layers)?.save_with_metadata(
                    Path::new(&a[3]).join(format!("{name}.safetensors")),
                    HashMap::from([
                        ("model_id".into(), model_id.clone()),
                        ("n_prompts".into(), count.to_string()),
                    ]),
                )?;
            }
        }
        "fit" => {
            let records: Vec<Record> = read(&a[2])?;
            if records.is_empty() || records.iter().any(|r| r.ids.len() != 128) {
                return Err(err("requires nonempty 128-token records"));
            }
            let config = FittingConfig {
                source_layers: vec![6, 10, 14, 18, 22],
                target_layer: 23,
                skip_first: 16,
                max_seq_len: 128,
                dim_batch: std::env::var("JLENS_DIM_BATCH")
                    .unwrap_or("4".into())
                    .parse()
                    .map_err(err)?,
            };
            let corpus_id = jacobian_lens::provenance::sha256_file(&a[2])?;
            let checkpoint = Path::new(&a[3]).with_extension("fit.safetensors");
            let mut state = if checkpoint.exists() {
                FitCheckpoint::load(&checkpoint, &config, &model_id, &corpus_id, &device)?
            } else {
                FitCheckpoint::new(
                    config.clone(),
                    model.hidden_size(),
                    model_id.clone(),
                    corpus_id.clone(),
                )?
            };
            let ids = records
                .iter()
                .map(|r| Ok(Tensor::new(r.ids.as_slice(), &device)?.unsqueeze(0)?))
                .collect::<Result<Vec<_>>>()?;
            while state.next_index() < ids.len() {
                let t = std::time::Instant::now();
                state.run_until(&model, &ids, state.next_index() + 1, Some(&checkpoint))?;
                eprintln!(
                    "{} {}/{} {:.2}s",
                    a[2],
                    state.next_index(),
                    ids.len(),
                    t.elapsed().as_secs_f64()
                );
            }
            state.registry()?.save_with_metadata(
                &a[3],
                HashMap::from([
                    ("model_id".into(), model_id),
                    ("corpus_id".into(), corpus_id),
                    ("n_prompts".into(), state.successes().to_string()),
                    (
                        "fit_config".into(),
                        serde_json::to_string(&config).map_err(err)?,
                    ),
                ]),
            )?;
        }
        "eval" | "eval-development" => {
            let splits: Option<BTreeMap<String, BTreeMap<String, String>>> =
                if a[0] == "eval-development" {
                    Some(read(a.get(4).ok_or_else(|| {
                        err("development evaluation requires splits.json")
                    })?)?)
                } else {
                    None
                };
            let lens_dir = Path::new(&a[2]);
            let mut lenses = BTreeMap::new();
            for name in ["main-100", "main-1000", "replicate-100"] {
                let path = lens_dir.join(format!("{name}.safetensors"));
                if path.exists() {
                    lenses.insert(name, checked_lens(path, &model_id, &device)?);
                }
            }
            if lenses.is_empty() {
                return Err(err("no experiment lenses"));
            }
            let layers = [6, 10, 14, 18, 22];
            let mut results = Vec::new();
            for task in [
                "association",
                "multihop",
                "multilingual",
                "order-ops",
                "poetry",
                "typo",
            ] {
                let dataset: EvaluationDataset =
                    read(format!("data/evaluations/lens-eval-{task}.json"))?;
                let position = match task {
                    "association" | "typo" => ReadoutPosition::LastToken,
                    "poetry" => ReadoutPosition::LastNewline,
                    _ => ReadoutPosition::BeforeTarget,
                };
                for item in dataset.items {
                    if let Some(splits) = &splits {
                        let split = splits
                            .get(&format!("lens-eval-{task}"))
                            .and_then(|items| items.get(&item.name))
                            .ok_or_else(|| err("missing split assignment"))?;
                        if split != "development"
                            && !(task == "multihop" && item.name == "mars-color")
                        {
                            continue;
                        }
                    }
                    let ids = encode_readout(&tokenizer, &item, position)?;
                    let p = ids.len() - 1;
                    let states =
                        model.residuals(&Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?)?;
                    let targets = item
                        .intermediates
                        .iter()
                        .map(|c| concept_tokens(&tokenizer, c, &BTreeMap::new()))
                        .collect::<Result<Vec<_>>>()?;
                    let mut methods = BTreeMap::new();
                    for (name, lens) in std::iter::once(("logit-lens", None))
                        .chain(lenses.iter().map(|(&n, l)| (n, Some(l))))
                    {
                        let mut rows = BTreeMap::new();
                        for l in layers {
                            let h = states[l].narrow(1, p, 1)?.contiguous()?;
                            let logits = if let Some(lens) = lens {
                                lens.decode_jspace(l, &h, &model)?
                            } else {
                                model.logits(&h)?
                            };
                            let s = scores(&logits)?;
                            if s.iter().any(|v| !v.is_finite()) {
                                return Err(err("nonfinite readout"));
                            }
                            rows.insert(l, targets.iter().map(|t| rank(&s, t)).collect::<Vec<_>>());
                        }
                        methods.insert(name, rows);
                    }
                    // Exact greedy continuation prefix: teacher-forced argmax at every target
                    // token equals greedy generation iff all target tokens are recovered.
                    let capability = if let Some(target) = &item.target {
                        let full = tokenizer
                            .encode(format!("{}{}", item.prompt, target), false)
                            .map_err(err)?;
                        if !full.get_ids().starts_with(&ids) {
                            return Err(err("target prefix mismatch"));
                        }
                        let suffix = &full.get_ids()[ids.len()..];
                        let mut context = ids.clone();
                        let mut correct = 0;
                        let mut logp = 0.;
                        for &next in suffix {
                            let states = model.residuals(
                                &Tensor::new(context.as_slice(), &device)?.unsqueeze(0)?,
                            )?;
                            let s = scores(&model.logits(&states[23].narrow(
                                1,
                                context.len() - 1,
                                1,
                            )?)?)?;
                            if rank(&s, &[next]) == Some(0) {
                                correct += 1;
                            }
                            let max = s.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                            logp += s[next as usize] as f64
                                - max
                                - s.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln();
                            context.push(next);
                        }
                        Some(
                            serde_json::json!({"target_tokens":suffix.len(),"exact_greedy_prefix":correct==suffix.len(),"mean_target_log_probability":logp/suffix.len() as f64}),
                        )
                    } else {
                        None
                    };
                    results.push(serde_json::json!({"task":task,"name":item.name,"concepts":item.intermediates,"concept_token_ids":targets,"readout_position":p,"ranks":methods,"capability":capability}));
                    if results.len() % 20 == 0 {
                        eprintln!("evaluated {} items", results.len());
                    }
                }
            }
            write(
                &a[3],
                &serde_json::json!({"model_id":model_id,"layers":layers,"items":results}),
            )?;
        }
        _ => return Err(err("unknown command")),
    }
    Ok(())
}
