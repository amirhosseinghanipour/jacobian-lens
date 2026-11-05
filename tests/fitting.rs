//! Research parity: exact reverse-mode estimator versus independent f64 JVPs.
use candle_core::{Device, Tensor};
use jacobian_lens::research::{
    Matrix, fitting as reference,
    model::{DifferentiableDecoder, ReferenceDecoder},
};
use jacobian_lens::{DifferentiableTransformer, FittingConfig, Result, jacobian_for_prompt};
struct Decoder {
    embeddings: Tensor,
    weights: Vec<Tensor>,
    mix: f64,
    nonlinear: bool,
}
impl DifferentiableTransformer for Decoder {
    fn hidden_size(&self) -> usize {
        3
    }
    fn layer_count(&self) -> usize {
        self.weights.len()
    }
    fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(self.embeddings.narrow(1, 0, ids.dim(1)?)?)
    }
    fn forward_block(&self, layer: usize, h: &Tensor) -> Result<Tensor> {
        let (_, seq, _) = h.dims3()?;
        let mut z = h.broadcast_matmul(&self.weights[layer].t()?)?;
        let mut prefix = Vec::new();
        for p in 0..seq {
            prefix.push(h.narrow(1, 0, p + 1)?.mean_keepdim(1)?);
        }
        z = (z + Tensor::cat(&prefix, 1)?.affine(self.mix, 0.)?)?;
        if self.nonlinear {
            z = z.tanh()?;
        }
        Ok((h + z)?)
    }
}
#[test]
fn reverse_mode_matches_forward_mode_for_causal_nonlinear_model() -> Result<()> {
    let device = Device::Cpu;
    let input = Matrix::from_fn(6, 3, |r, c| ((r * 3 + c) as f64 * 0.2).sin());
    let weights =
        vec![Matrix::from_row_slice(3, 3, &[0.1, 0.2, 0., 0., 0.1, -0.1, 0.2, 0., 0.05]); 3];
    let model = ReferenceDecoder {
        weights: weights.clone(),
        head: Matrix::identity(3, 3),
        temporal_mix: 0.3,
        nonlinear: true,
        rms_epsilon: None,
    };
    let tensor = |m: &Matrix<f64>| -> candle_core::Result<Tensor> {
        Tensor::from_vec(
            (0..m.nrows())
                .flat_map(|r| (0..m.ncols()).map(move |c| m[(r, c)] as f32))
                .collect(),
            (m.nrows(), m.ncols()),
            &device,
        )
    };
    let decoder = Decoder {
        embeddings: tensor(&input)?.unsqueeze(0)?,
        weights: weights
            .iter()
            .map(tensor)
            .collect::<candle_core::Result<_>>()?,
        mix: 0.3,
        nonlinear: true,
    };
    let cfg = reference::FitConfig {
        sources: vec![0, 1],
        target: 2,
        skip_first: 1,
        max_seq_len: 6,
    };
    let expected = reference::jacobian_for_prompt(&model, &input, &cfg).unwrap();
    let ids = Tensor::new(&[[0u32; 6]], &device)?;
    for dim_batch in [1, 2, 3, 5] {
        let result = jacobian_for_prompt(
            &decoder,
            &ids,
            &FittingConfig {
                source_layers: vec![0, 1],
                target_layer: 2,
                skip_first: 1,
                max_seq_len: 6,
                dim_batch,
            },
        )?;
        for layer in [0, 1] {
            let actual = result.jacobians[&layer].to_vec2::<f32>()?;
            for r in 0..3 {
                for c in 0..3 {
                    assert!((actual[r][c] as f64 - expected[&layer][(r, c)]).abs() < 2e-5);
                }
            }
        }
        assert_eq!(result.valid_positions, 4);
    }
    assert_eq!(model.num_layers(), 3);
    Ok(())
}

#[test]
fn candle_fit_resume_after_skip_matches_uninterrupted_corpus_mean() -> Result<()> {
    use jacobian_lens::{FitCheckpoint, fit};
    let device = Device::Cpu;
    let decoder = Decoder {
        embeddings: Tensor::new(
            &[[
                [1f32, 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [1., 1., 0.],
                [1., 0., 1.],
                [0., 1., 1.],
            ]],
            &device,
        )?,
        weights: vec![
            Tensor::new(
                &[[0.1f32, 0.2, 0.], [0., 0.1, -0.1], [0.2, 0., 0.05]],
                &device
            )?;
            3
        ],
        mix: 0.3,
        nonlinear: true,
    };
    let ids = vec![
        Tensor::new(&[[0u32; 6]], &device)?,
        Tensor::new(&[[0u32]], &device)?,
        Tensor::new(&[[0u32; 4]], &device)?,
    ];
    let config = FittingConfig {
        source_layers: vec![0, 1],
        target_layer: 2,
        skip_first: 1,
        max_seq_len: 6,
        dim_batch: 2,
    };
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("checkpoint.safetensors");
    let mut state = FitCheckpoint::new(config.clone(), 3, "model-v1".into(), "corpus-v1".into())?;
    state.run_until(&decoder, &ids, 2, Some(&path))?;
    assert_eq!(
        (state.next_index(), state.successes(), state.skipped()),
        (2, 1, 1)
    );
    let mut resumed = FitCheckpoint::load(&path, &config, "model-v1", "corpus-v1", &device)?;
    resumed.run_until(&decoder, &ids, 3, Some(&path))?;
    let actual = resumed.registry()?;
    let (expected, n) = fit(&decoder, &ids, &config)?;
    assert_eq!(n, 2);
    for layer in [0, 1] {
        assert_eq!(
            actual.jacobian(layer)?.to_vec2::<f32>()?,
            expected.jacobian(layer)?.to_vec2::<f32>()?
        );
    }
    assert!(FitCheckpoint::load(&path, &config, "model-v2", "corpus-v1", &device).is_err());
    assert!(FitCheckpoint::load(&path, &config, "model-v1", "corpus-v2", &device).is_err());
    Ok(())
}
