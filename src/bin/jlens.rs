//! Research CLI. Model artifacts and fitting corpora are supplied explicitly.
use candle_core::{DType, Device, Tensor};
use jacobian_lens::{
    DifferentiableTransformer, FittingConfig, JacobianLensRegistry, LensError, Result,
    evaluation::*, qwen2::Qwen2,
};
use std::{collections::BTreeMap, path::Path};
use tokenizers::Tokenizer;
fn usage() -> LensError {
    LensError::Invalid("usage: jlens fit MODEL_DIR PROMPTS.json OUTPUT.safetensors SOURCE_LAYERS [DIM_BATCH] [MAX_SEQ_LEN]\n       jlens eval MODEL_DIR LENS.safetensors DATASET.json POSITION OUTPUT.json [K]\nSOURCE_LAYERS: comma-separated block-output indices; POSITION: target|newline|last; PROMPTS.json: array of calibration strings".into())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 5 || (args.first().is_some_and(|s| s == "eval") && args.len() < 6) {
        return Err(usage());
    }
    let device = Device::Cpu;
    let tokenizer = Tokenizer::from_file(Path::new(&args[1]).join("tokenizer.json"))
        .map_err(|e| LensError::Tokenizer(e.to_string()))?;
    let model = Qwen2::load(&args[1], &device, DType::F32)?;
    match args[0].as_str() {
        "fit" => {
            let prompts: Vec<String> = serde_json::from_slice(&std::fs::read(&args[2])?)
                .map_err(|e| LensError::Invalid(e.to_string()))?;
            let mut layers = args[4]
                .split(',')
                .map(|s| s.parse::<usize>().map_err(|_| usage()))
                .collect::<Result<Vec<_>>>()?;
            layers.sort_unstable();
            layers.dedup();
            let dim_batch = args
                .get(5)
                .map(|s| s.parse::<usize>())
                .transpose()
                .map_err(|_| usage())?
                .unwrap_or(8);
            let max_seq_len = args
                .get(6)
                .map(|s| s.parse::<usize>())
                .transpose()
                .map_err(|_| usage())?
                .unwrap_or(128);
            let config = FittingConfig {
                source_layers: layers,
                target_layer: model.layer_count() - 1,
                skip_first: 16,
                max_seq_len,
                dim_batch,
            };
            let ids = prompts
                .iter()
                .map(|p| {
                    let e = tokenizer
                        .encode(p.as_str(), false)
                        .map_err(|e| LensError::Tokenizer(e.to_string()))?;
                    Ok(Tensor::new(e.get_ids(), &device)?.unsqueeze(0)?)
                })
                .collect::<Result<Vec<_>>>()?;
            let (lens, count) = jacobian_lens::fit(&model, &ids, &config)?;
            lens.save(&args[3])?;
            println!("Fitted {} layers on {count} prompts", lens.layers().count());
        }
        "eval" => {
            let registry = JacobianLensRegistry::load(&args[2], &device, DType::F32)?;
            let dataset: EvaluationDataset = serde_json::from_slice(&std::fs::read(&args[3])?)
                .map_err(|e| LensError::Invalid(e.to_string()))?;
            let position = match args[4].as_str() {
                "target" => ReadoutPosition::BeforeTarget,
                "newline" => ReadoutPosition::LastNewline,
                "last" => ReadoutPosition::LastToken,
                _ => return Err(usage()),
            };
            let k = args
                .get(6)
                .map(|s| s.parse::<usize>())
                .transpose()
                .map_err(|_| usage())?
                .unwrap_or(10);
            let mut items = Vec::new();
            for item in dataset.items {
                items.push(evaluate_item(
                    &model,
                    &registry,
                    &tokenizer,
                    &item,
                    position,
                    &BTreeMap::new(),
                    &device,
                )?);
            }
            let score = pass_at_k(&items, k)?;
            let result = serde_json::json!({"metric":score,"items":items,"model_directory":args[1],"lens_file":args[2],"dataset":args[3]});
            jacobian_lens::research::save_json(Path::new(&args[5]), &result)
                .map_err(|e| LensError::Invalid(e.to_string()))?;
            println!(
                "pass@{}: {:.4}; single-token concept coverage: {:.4}",
                k, score.pass_at_k, score.concept_coverage
            );
        }
        _ => return Err(usage()),
    }
    Ok(())
}
