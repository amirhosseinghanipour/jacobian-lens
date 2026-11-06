//! Local checkpoint research parity tests. Never silently skip missing fixtures.
//! Run scripts/qwen_oracle.py first, then explicitly run this ignored test.
#![cfg(feature = "native-qwen")]
use candle_core::{DType, Device, Tensor};
use jacobian_lens::{DifferentiableTransformer, Result, Unembed, jacobian_rows, qwen2::Qwen2};
fn compare(actual: &Tensor, expected: &Tensor, absolute: f32, relative: f32) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let a = actual.flatten_all()?.to_vec1::<f32>()?;
    let e = expected.flatten_all()?.to_vec1::<f32>()?;
    let max = a
        .iter()
        .zip(&e)
        .map(|(a, e)| (a - e).abs())
        .fold(0f32, f32::max);
    assert!(
        a.iter()
            .zip(&e)
            .all(|(a, e)| a.is_finite() && (a - e).abs() <= absolute + relative * e.abs()),
        "maximum absolute difference {max}"
    );
    Ok(())
}
#[test]
#[ignore = "requires Qwen checkpoint and independently generated PyTorch fixture"]
fn qwen_residual_logits_and_jacobian_rows_match_pytorch() -> Result<()> {
    let directory = std::env::var("JLENS_QWEN_DIR").expect("set JLENS_QWEN_DIR");
    let oracle = std::env::var("JLENS_QWEN_ORACLE").expect("set JLENS_QWEN_ORACLE");
    let device = if std::env::var("JLENS_DEVICE").as_deref() == Ok("cuda") {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let fixture = candle_core::safetensors::load(oracle, &device)?;
    let model = Qwen2::load(directory, &device, DType::F32)?;
    let ids = fixture["input_ids"].to_dtype(DType::U32)?;
    let states = model.residuals(&ids)?;
    let source = model.layer_count() - 2;
    for layer in [0, source, source + 1] {
        compare(
            &states[layer],
            &fixture[&format!("residual.{layer}")],
            2e-3,
            2e-3,
        )?;
    }
    let seq = ids.dim(1)?;
    let logits = model
        .logits(&states[source + 1].narrow(1, seq - 1, 1)?)?
        .squeeze(1)?;
    compare(&logits, &fixture["logits"], 2e-3, 2e-3)?;
    let dimensions = fixture["dimensions"]
        .to_vec1::<i64>()?
        .into_iter()
        .map(|i| i as usize)
        .collect::<Vec<_>>();
    let rows = jacobian_rows(
        &model,
        &states[source],
        source,
        source + 1,
        16,
        &dimensions,
        2,
    )?;
    compare(&rows, &fixture["jacobian_rows"], 1e-3, 2e-3)?;
    // Each row also predicts a genuine downstream directional perturbation.
    let d = model.hidden_size();
    let valid = seq - 17;
    let mut direction = vec![0f32; seq * d];
    for p in 16..seq - 1 {
        for c in 0..d {
            direction[p * d + c] = ((c as f32 + 1.) * 0.13).sin() / (d as f32).sqrt();
        }
    }
    let direction = Tensor::from_vec(direction, (1, seq, d), &device)?;
    let epsilon = 0.02;
    let plus = model.suffix(
        source,
        source + 1,
        &(&states[source] + direction.affine(epsilon, 0.)?)?,
    )?;
    let minus = model.suffix(
        source,
        source + 1,
        &(&states[source] - direction.affine(epsilon, 0.)?)?,
    )?;
    let numerical = ((plus - minus)? / (2. * epsilon))?
        .narrow(1, 16, valid)?
        .mean(1)?
        .squeeze(0)?
        .to_vec1::<f32>()?;
    let rows = rows.to_vec2::<f32>()?;
    let vector = direction.get(0)?.get(16)?.to_vec1::<f32>()?;
    for (r, &dim) in dimensions.iter().enumerate() {
        let predicted: f32 = rows[r].iter().zip(&vector).map(|(a, b)| a * b).sum();
        assert!(
            (predicted - numerical[dim]).abs() < 0.03 + 0.02 * predicted.abs(),
            "directional derivative mismatch at {dim}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "fits a complete Qwen lens and performs a held-out causal concept swap"]
fn qwen_full_fit_serialization_and_concept_swap() -> Result<()> {
    use jacobian_lens::{
        FittingConfig, JacobianLensRegistry, LensFrame,
        evaluation::{EvaluationDataset, ReadoutPosition, encode_readout},
        fit,
    };
    use tokenizers::Tokenizer;
    let directory = std::env::var("JLENS_QWEN_DIR").expect("set JLENS_QWEN_DIR");
    let oracle = std::env::var("JLENS_QWEN_ORACLE").expect("set JLENS_QWEN_ORACLE");
    let device = if std::env::var("JLENS_DEVICE").as_deref() == Ok("cuda") {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let fixture = candle_core::safetensors::load(oracle, &device)?;
    let model = Qwen2::load(&directory, &device, DType::F32)?;
    let ids = fixture["input_ids"].to_dtype(DType::U32)?;
    let source = model.layer_count() - 2;
    let (registry, n) = fit(
        &model,
        &[ids],
        &FittingConfig {
            source_layers: vec![source],
            target_layer: source + 1,
            skip_first: 16,
            max_seq_len: 32,
            dim_batch: 8,
        },
    )?;
    assert_eq!(n, 1);
    let dims = fixture["dimensions"].to_dtype(DType::U32)?;
    compare(
        &registry.jacobian(source)?.index_select(&dims, 0)?,
        &fixture["jacobian_rows"],
        1e-3,
        2e-3,
    )?;
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("lens.safetensors");
    registry.save(&path)?;
    let registry = JacobianLensRegistry::load(&path, &device, DType::F32)?;
    let tok =
        Tokenizer::from_file(std::path::Path::new(&directory).join("tokenizer.json")).unwrap();
    let dataset: EvaluationDataset =
        serde_json::from_str(include_str!("../data/evaluations/lens-eval-multihop.json")).unwrap();
    let item = dataset
        .items
        .iter()
        .find(|i| i.name == "firstletter-paris-country")
        .unwrap();
    let heldout = encode_readout(&tok, item, ReadoutPosition::BeforeTarget)?;
    let heldout = Tensor::new(heldout.as_slice(), &device)?.unsqueeze(0)?;
    let states = model.residuals(&heldout)?;
    let mut tracker = jacobian_lens::LensTracker::new(
        &registry,
        &model,
        &tok,
        [source],
        vec![(0, heldout.dim(1)? - 1)],
        5,
    )?;
    let traced = model.forward_tracked(&heldout, Some(&mut tracker), None)?;
    assert_eq!(tracker.observations().len(), 1);
    compare(&traced, &model.logits(states.last().unwrap())?, 1e-5, 1e-5)?;

    let p = heldout.dim(1)? - 1;
    let hidden = states[source].narrow(1, p, 1)?.contiguous()?;
    let ids = [" France", " Germany"]
        .iter()
        .map(|word| {
            let encoding = tok.encode(*word, false).unwrap();
            assert_eq!(encoding.len(), 1, "concept must be one token");
            encoding.get_ids()[0]
        })
        .collect::<Vec<_>>();
    let frame = LensFrame::from_tokens(&registry, source, model.lm_head(), &ids, 1e-6)?;
    let swap = frame.permute(&hidden, &[1, 0], 1.)?;
    let restored = frame.permute(&swap, &[1, 0], 1.)?;
    compare(&restored, &hidden, 2e-3, 1e-4)?;
    compare(&frame.permute(&hidden, &[1, 0], 0.)?, &hidden, 1e-6, 1e-6)?;
    // Only final prompt position is edited; then run the actual remaining block.
    let edited = Tensor::cat(&[states[source].narrow(1, 0, p)?, swap], 1)?;
    let after = model.logits(&model.suffix(source, source + 1, &edited)?.narrow(1, p, 1)?)?;
    let before = model.logits(&states[source + 1].narrow(1, p, 1)?)?;
    assert!(
        (&after - &before)?.sqr()?.sum_all()?.to_scalar::<f32>()? > 1e-6,
        "causal swap must reach downstream logits"
    );
    let scores_before = before
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    let scores_after = after
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    let result = jacobian_lens::research::metrics::measure_intervention(
        &scores_before,
        &scores_after,
        ids[1] as usize,
    )
    .unwrap();
    println!("Held-out concept swap measurement: {result:?}");
    let slice = jacobian_lens::visualization::compute_slice(
        &model,
        &registry,
        &tok,
        &heldout,
        &jacobian_lens::research::slice::SliceConfig {
            top_n: 5,
            max_tracked: Some(10),
            pinned: ids.iter().map(|&id| id as usize).collect(),
            layer_stride: 1,
            last_n: Some(3),
            display_mask: None,
        },
    )?;
    assert_eq!(slice.slice.layers, vec![source, source + 1]);
    let final_cell = slice
        .slice
        .cells
        .iter()
        .find(|c| c.layer == source + 1 && c.position == p)
        .unwrap();
    let applied = registry.apply(
        &model,
        &heldout,
        &jacobian_lens::ApplyOptions {
            layers: Some(vec![source]),
            positions: Some(vec![-1]),
            max_seq_len: 512,
            use_jacobian: true,
        },
    )?;
    compare(
        &applied.model_logits,
        &before.reshape((1, model.config().vocab_size))?,
        1e-5,
        1e-5,
    )?;
    compare(
        &applied.lens_logits[&source],
        &registry
            .decode_jspace(source, &hidden, &model)?
            .reshape((1, model.config().vocab_size))?,
        1e-5,
        1e-5,
    )?;
    let baseline = registry.apply(
        &model,
        &heldout,
        &jacobian_lens::ApplyOptions {
            layers: Some(vec![source]),
            positions: Some(vec![-1]),
            max_seq_len: 512,
            use_jacobian: false,
        },
    )?;
    compare(
        &baseline.lens_logits[&source],
        &model
            .logits(&hidden)?
            .reshape((1, model.config().vocab_size))?,
        1e-5,
        1e-5,
    )?;
    let expected_ranks = jacobian_lens::research::metrics::ranks(&scores_before).unwrap();
    assert_eq!(expected_ranks[final_cell.top_ids[0]], 0);
    for (column, &id) in slice.slice.tracked_ids.iter().enumerate() {
        assert_eq!(final_cell.tracked_ranks[column], expected_ranks[id]);
    }
    if let Ok(directory) = std::env::var("JLENS_ARTIFACT_DIR") {
        std::fs::create_dir_all(&directory)?;
        jacobian_lens::visualization::save_page(
            &slice,
            "Qwen held-out J-lens readout",
            std::path::Path::new(&directory).join("qwen-slice.html"),
        )?;
        let output = serde_json::json!({"model":"Qwen/Qwen2.5-0.5B-Instruct","revision":"7ae557604adf67be50417f59c2c2f167def9a775","calibration_prompts":n,"source_layer":source,"held_out_item":item.name,"intervention":result});
        jacobian_lens::research::save_json(
            &std::path::Path::new(&directory).join("intervention.json"),
            &output,
        )
        .unwrap();
    }

    // This is an estimator/intervention test, not a minimum quality assertion
    // for a one-prompt, one-layer lens on a 0.5B model.
    Ok(())
}
