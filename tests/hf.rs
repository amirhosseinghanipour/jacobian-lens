//! Cross-runtime research parity on structurally different Transformers models.
#![cfg(feature = "hf")]
use candle_core::{DType, Device, Tensor};
use jacobian_lens::{
    ApplyOptions, FitCheckpoint, FittingConfig, LensFrame, ResearchModel, ResidualReplacement,
    Result, estimate_prompt,
    hf::{HfModel, HfOptions},
    intervention_logits,
};
use std::{path::Path, sync::Arc, time::Duration};

fn close(actual: &Tensor, expected: &Tensor, abs: f32, rel: f32) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    for (a, e) in actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .zip(
            expected
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        )
    {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= abs + rel * e.abs(),
            "{a} vs {e}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires scripts/backend_oracles.py fixtures and Python runtime dependencies"]
fn architectures_preserve_jacobians_readouts_and_causal_swaps() -> Result<()> {
    let root = std::env::var("JLENS_HF_TEST_DIR").expect("set JLENS_HF_TEST_DIR");
    let python = std::env::var("JLENS_PYTHON").expect("set JLENS_PYTHON");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(Path::new(&root).join("families.json"))?).unwrap();
    let families = manifest["families"].as_array().unwrap();
    assert!(
        families.len() >= 9,
        "exercise dense, MoE, hybrid and different normalization/layouts"
    );
    for family in families {
        let family = family.as_str().unwrap();
        eprintln!("Research backend parity: {family}");
        let directory = Path::new(&root).join(family);
        let oracle =
            candle_core::safetensors::load(directory.join("oracle.safetensors"), &Device::Cpu)?;
        let mut options = HfOptions::local(directory.to_str().unwrap());
        options.tokenizer = Some(Path::new(&root).join("tokenizer").to_str().unwrap().into());
        if family == "custom" {
            options.factory = Some("backend_test_factory:create".into());
        }
        let backend = Arc::new(HfModel::launch(
            &python,
            &options,
            &Device::Cpu,
            Duration::from_secs(120),
        )?);
        // Dynamic dispatch must cover the whole research API, not just inference.
        let model: &dyn ResearchModel = backend.as_ref();
        let ids = oracle["input_ids"].to_dtype(DType::U32)?;
        let trace = model.capture(&ids, &[0, 1, 2], &[])?;
        for layer in 0..3 {
            close(
                &trace.residuals[&layer],
                &oracle[&format!("residual.{layer}")],
                1e-6,
                1e-5,
            )?;
        }
        close(
            &model.logits(&trace.final_residual)?,
            &oracle["logits"],
            1e-5,
            1e-4,
        )?;
        close(
            &model.unembedding_rows(&[7, 8])?,
            &oracle["head_rows"],
            1e-7,
            1e-6,
        )?;
        let encoded = backend.encode("t1 t3 t5 t7 t9 t11", 6)?;
        assert_eq!(encoded.to_vec2::<u32>()?, ids.to_vec2::<u32>()?);
        let config = FittingConfig {
            source_layers: vec![0, 1],
            target_layer: 2,
            skip_first: 1,
            max_seq_len: 6,
            dim_batch: 3,
        };
        let result = estimate_prompt(model, &ids, &config)?;
        for layer in [0, 1] {
            close(
                &result.jacobians[&layer],
                &oracle[&format!("jacobians.{layer}")],
                2e-4,
                2e-4,
            )?;
        }
        let predicted = result.jacobians[&0]
            .matmul(&oracle["direction"].unsqueeze(1)?)?
            .squeeze(1)?;
        close(&predicted, &oracle["finite_difference"], 6e-3, 5e-3)?;
        let registry = jacobian_lens::JacobianLensRegistry::new(result.jacobians)?;
        let mut tracker = jacobian_lens::LensTracker::new(
            &registry,
            model,
            backend.tokenizer()?,
            [0, 1],
            vec![(0, 5)],
            3,
        )?;
        let tracked = jacobian_lens::forward_tracked(model, &ids, Some(&mut tracker), &[])?;
        close(&tracked, &oracle["logits"], 1e-5, 1e-4)?;
        assert_eq!(tracker.observations().len(), 2);
        let frame = LensFrame::from_model(&registry, 0, model, &[7, 8], 1e-6)?;
        let h = &trace.residuals[&0];
        let swapped = frame.permute(&h.narrow(1, 5, 1)?, &[1, 0], 1.)?;
        let replacement = Tensor::cat(&[h.narrow(1, 0, 5)?, swapped], 1)?;
        close(&replacement, &oracle["swapped"], 2e-5, 2e-4)?;
        let edited = intervention_logits(
            model,
            &ids,
            &[ResidualReplacement {
                layer: 0,
                hidden: replacement,
            }],
        )?;
        close(&edited, &oracle["swapped_logits"], 3e-4, 3e-4)?;
        let unchanged = intervention_logits(
            model,
            &ids,
            &[ResidualReplacement {
                layer: 0,
                hidden: h.clone(),
            }],
        )?;
        close(&unchanged, &oracle["logits"], 1e-5, 1e-4)?;
        let applied = registry.apply(
            model,
            &ids,
            &ApplyOptions {
                positions: Some(vec![-1]),
                ..Default::default()
            },
        )?;
        close(
            &applied.model_logits,
            &oracle["logits"].get(0)?.narrow(0, 5, 1)?,
            1e-5,
            1e-4,
        )?;
        let baseline = registry.apply(
            model,
            &ids,
            &ApplyOptions {
                positions: Some(vec![-1]),
                use_jacobian: false,
                ..Default::default()
            },
        )?;
        close(
            &baseline.lens_logits[&0],
            &model.logits(&h.get(0)?.narrow(0, 5, 1)?)?,
            1e-6,
            1e-5,
        )?;
        let slice = jacobian_lens::visualization::compute_slice(
            model,
            &registry,
            backend.tokenizer()?,
            &ids,
            &jacobian_lens::research::slice::SliceConfig {
                top_n: 3,
                max_tracked: Some(4),
                pinned: [7, 8].into_iter().collect(),
                layer_stride: 1,
                last_n: Some(2),
                display_mask: None,
            },
        )?;
        assert_eq!(slice.slice.layers, vec![0, 1, 2]);
        assert_eq!(slice.slice.cells.len(), 6);
        // Failed requests must not leave hooks/caches affecting later experiments.
        assert!(model.capture(&ids, &[3], &[]).is_err());
        assert!(
            model
                .capture(
                    &ids,
                    &[],
                    &[
                        ResidualReplacement {
                            layer: 0,
                            hidden: h.clone()
                        },
                        ResidualReplacement {
                            layer: 0,
                            hidden: h.clone()
                        }
                    ]
                )
                .is_err()
        );
        let invalid = Tensor::new(&[[99u32; 6]], &Device::Cpu)?;
        assert!(model.capture(&invalid, &[0], &[]).is_err());
        close(
            &intervention_logits(model, &ids, &[])?,
            &oracle["logits"],
            1e-5,
            1e-4,
        )?;
        // The same backend supports Rust's atomic resume and short-prompt skips.
        let tmp = tempfile::tempdir()?;
        let checkpoint = tmp.path().join("fit.safetensors");
        let prompts = vec![ids.clone(), ids.narrow(1, 0, 1)?, ids.clone()];
        let mut state = FitCheckpoint::new(config.clone(), 16, family.into(), "fixture".into())?;
        state.run_until(model, &prompts, 2, Some(&checkpoint))?;
        let mut state = FitCheckpoint::load(&checkpoint, &config, family, "fixture", &Device::Cpu)?;
        state.run_until(model, &prompts, 3, Some(&checkpoint))?;
        assert_eq!((state.successes(), state.skipped()), (2, 1));
        close(
            &state.registry()?.jacobian(0)?,
            &registry.jacobian(0)?,
            1e-5,
            1e-4,
        )?;
        // Concurrent callers share a worker without exchanging each other's IDs.
        let workers: Vec<_> = [ids.clone(), ids.narrow(1, 0, 4)?]
            .into_iter()
            .map(|ids| {
                let backend = backend.clone();
                std::thread::spawn(move || -> Result<Tensor> {
                    intervention_logits(backend.as_ref(), &ids, &[])
                })
            })
            .collect();
        let mut results = workers.into_iter().map(|w| w.join().unwrap());
        close(&results.next().unwrap()?, &oracle["logits"], 1e-5, 1e-4)?;
        assert_eq!(results.next().unwrap()?.dims(), &[1, 4, 32]);
    }
    Ok(())
}

#[test]
#[ignore = "requires downloaded Qwen and independent PyTorch checkpoint oracle"]
fn pretrained_qwen_readout_matches_independent_oracle() -> Result<()> {
    use jacobian_lens::{LensModel, Unembed};
    let directory = std::env::var("JLENS_QWEN_DIR").expect("set JLENS_QWEN_DIR");
    let oracle = std::env::var("JLENS_QWEN_ORACLE").expect("set JLENS_QWEN_ORACLE");
    let python = std::env::var("JLENS_PYTHON").expect("set JLENS_PYTHON");
    let model = HfModel::launch(
        python,
        &HfOptions::local(directory),
        &Device::Cpu,
        Duration::from_secs(180),
    )?;
    let fixture = candle_core::safetensors::load(oracle, &Device::Cpu)?;
    let ids = fixture["input_ids"].to_dtype(DType::U32)?;
    let source = model.dimensions().layer_count - 2;
    let trace = model.capture(&ids, &[0, source, source + 1], &[])?;
    for layer in [0, source, source + 1] {
        close(
            &trace.residuals[&layer],
            &fixture[&format!("residual.{layer}")],
            1e-3,
            2e-3,
        )?;
    }
    let last = trace
        .final_residual
        .narrow(1, ids.dim(1)? - 1, 1)?
        .squeeze(1)?;
    close(&model.logits(&last)?, &fixture["logits"], 1e-3, 2e-3)?;
    Ok(())
}

#[test]
#[ignore = "requires generated architecture fixtures and the Python runtime"]
fn default_cli_fits_non_qwen_models_and_rejects_cross_model_lenses() -> Result<()> {
    use jacobian_lens::JacobianLensRegistry;
    use std::process::Command;
    let root = std::env::var("JLENS_HF_TEST_DIR").expect("set JLENS_HF_TEST_DIR");
    let python = std::env::var("JLENS_PYTHON").expect("set JLENS_PYTHON");
    let scratch = tempfile::tempdir()?;
    let prompts = scratch.path().join("prompts.json");
    let dataset = scratch.path().join("evaluation.json");
    let text = (1..25)
        .map(|i| format!("t{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    std::fs::write(&prompts, serde_json::to_vec(&vec![text]).unwrap())?;
    std::fs::write(
        &dataset,
        serde_json::to_vec(&serde_json::json!({"items":[{
            "name":"concept-readout", "prompt":"t1 t3 t5 t7 t9 t11", "intermediates":["t7","t8"]
        }]}))
        .unwrap(),
    )?;
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_jlens"))
            .args(args)
            .env_remove("JLENS_BACKEND")
            .env_remove("JLENS_DEVICE")
            .env("JLENS_PYTHON", &python)
            .output()
            .expect("run CLI")
    };
    let mut previous = None;
    for family in ["llama", "gpt2"] {
        let directory = Path::new(&root).join(family);
        let lens_path = scratch.path().join(format!("{family}.safetensors"));
        let evaluation = scratch.path().join(format!("{family}.json"));
        let out = run(&[
            "fit",
            directory.to_str().unwrap(),
            prompts.to_str().unwrap(),
            lens_path.to_str().unwrap(),
            "0,1",
            "3",
            "24",
        ]);
        assert!(
            out.status.success(),
            "CLI {family}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        // A three-layer non-Qwen model must be fitted to its own final block,
        // with the caller's selected source layers and exact token IDs.
        let model = HfModel::launch(
            &python,
            &HfOptions::local(directory.to_str().unwrap()),
            &Device::Cpu,
            Duration::from_secs(120),
        )?;
        let ids = Tensor::new((1u32..25).collect::<Vec<_>>(), &Device::Cpu)?.unsqueeze(0)?;
        let expected = estimate_prompt(
            &model,
            &ids,
            &FittingConfig {
                source_layers: vec![0, 1],
                target_layer: 2,
                skip_first: 16,
                max_seq_len: 24,
                dim_batch: 1,
            },
        )?;
        let lens = JacobianLensRegistry::load(&lens_path, &Device::Cpu, DType::F32)?;
        assert_eq!(lens.layers().collect::<Vec<_>>(), vec![0, 1]);
        for layer in [0, 1] {
            close(
                &lens.jacobian(layer)?,
                &expected.jacobians[&layer],
                2e-4,
                2e-4,
            )?;
        }
        let out = run(&[
            "eval",
            directory.to_str().unwrap(),
            lens_path.to_str().unwrap(),
            dataset.to_str().unwrap(),
            "last",
            evaluation.to_str().unwrap(),
            "10",
        ]);
        assert!(
            out.status.success(),
            "CLI eval: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let result: serde_json::Value =
            serde_json::from_slice(&std::fs::read(evaluation)?).unwrap();
        assert_eq!(result["items"][0]["readout_position"], 5);
        assert_eq!(result["metric"]["concept_coverage"], 1.0);
        if let Some(ref previous) = previous {
            let previous: &std::path::PathBuf = previous;
            let out = run(&[
                "eval",
                directory.to_str().unwrap(),
                previous.to_str().unwrap(),
                dataset.to_str().unwrap(),
                "last",
                scratch.path().join("wrong-model.json").to_str().unwrap(),
                "10",
            ]);
            assert!(!out.status.success());
            assert!(
                String::from_utf8_lossy(&out.stderr).contains("lens provenance does not match")
            );
        }
        previous = Some(lens_path);
    }
    Ok(())
}
