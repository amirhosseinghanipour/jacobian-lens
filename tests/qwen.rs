//! Local checkpoint research parity tests. Never silently skip missing fixtures.
//! Run scripts/qwen_oracle.py first, then explicitly run this ignored test.
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
    let device = Device::Cpu;
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
