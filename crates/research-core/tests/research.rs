use jlens_research::{
    Matrix, Result, Vector, fitting::*, intervention::*, metrics::*, model::*, slice::*,
};
use std::collections::BTreeSet;
fn decoder(nonlinear: bool, temporal_mix: f64) -> ReferenceDecoder {
    ReferenceDecoder {
        weights: vec![
            Matrix::from_row_slice(3, 3, &[0.1, 0.02, 0., -0.04, 0.05, 0.01, 0.03, 0., -0.02]),
            Matrix::from_row_slice(3, 3, &[0.03, 0.2, -0.1, 0., -0.04, 0.03, 0.02, 0.01, 0.1]),
            Matrix::from_row_slice(3, 3, &[0., 0.1, 0., 0.2, 0., -0.03, -0.02, 0.04, 0.]),
        ],
        head: Matrix::from_row_slice(
            5,
            3,
            &[
                1., 0., 0., 0., 1., 0., 0., 0., 1., -1., 0.3, 0.1, 0.2, -0.5, 0.7,
            ],
        ),
        nonlinear,
        temporal_mix,
        rms_epsilon: Some(1e-6),
    }
}
fn prompt(n: usize, phase: f64) -> Matrix<f64> {
    Matrix::from_fn(n, 3, |r, c| ((r * 3 + c) as f64 * 0.17 + phase).sin())
}
fn config() -> FitConfig {
    FitConfig {
        sources: vec![0, 1],
        target: 2,
        skip_first: 1,
        max_seq_len: 32,
    }
}
fn fit(model: &ReferenceDecoder, prompts: &[Matrix<f64>]) -> Result<Lens> {
    let mut state = FitState::new(model, prompts, config())?;
    state.run_until(model, prompts, prompts.len(), None)?;
    state.lens()
}
fn close(a: &Matrix<f64>, b: &Matrix<f64>, tolerance: f64) {
    assert_eq!(a.shape(), b.shape());
    assert!((a - b).amax() < tolerance, "error {}", (a - b).amax());
}

#[test]
fn python_linear_decoder_identity_plus_weight_and_exact_readout() -> Result<()> {
    let model = decoder(false, 0.);
    let inputs = vec![prompt(7, 0.), prompt(9, 1.)];
    let lens = fit(&model, &inputs)?;
    let expected = Matrix::identity(3, 3) + &model.weights[2];
    close(&lens.jacobians[&1], &expected, 1e-12);
    close(
        &lens.jacobians[&0],
        &(&expected * (Matrix::identity(3, 3) + &model.weights[1])),
        1e-12,
    );
    let (readouts, actual) = apply(&model, &lens, &inputs[0], &[0, 1], None, true)?;
    for l in [0, 1] {
        close(&readouts[&l], &actual, 1e-12);
    }
    let (baseline, _) = apply(&model, &lens, &inputs[0], &[0], None, false)?;
    assert!((&baseline[&0] - actual).norm() > 0.01);
    let (selected, actual_selected) = apply(&model, &lens, &inputs[0], &[1], Some(&[0, -1]), true)?;
    close(&selected[&1], &actual_selected, 1e-12);
    assert_eq!(actual_selected.nrows(), 2);
    Ok(())
}
#[test]
fn nonlinear_causal_estimator_matches_independent_positionwise_finite_differences() -> Result<()> {
    let model = decoder(true, 0.3);
    let x = prompt(6, 0.4);
    let cfg = config();
    let estimates = jacobian_for_prompt(&model, &x, &cfg)?;
    let states = model.residuals(&x)?;
    let valid = valid_positions(6, 1)?;
    // Independent oracle: perturb ONE source coordinate at ONE position,
    // evaluate two suffix forwards, sum target positions, then source-average.
    for source in [0, 1] {
        let h = &states[source];
        let zero = Matrix::zeros(h.nrows(), h.ncols());
        let epsilon = 1e-5;
        let mut expected = Matrix::zeros(3, 3);
        let mut same_position = Matrix::zeros(3, 3);
        for p in valid.clone() {
            for input_dim in 0..3 {
                let mut plus = h.clone();
                let mut minus = h.clone();
                plus[(p, input_dim)] += epsilon;
                minus[(p, input_dim)] -= epsilon;
                let yp = model.suffix_jvp(source, 2, &plus, &zero)?.0;
                let ym = model.suffix_jvp(source, 2, &minus, &zero)?.0;
                let delta = (yp - ym) / (2. * epsilon);
                for output_dim in 0..3 {
                    assert!(
                        (0..p).all(|t| delta[(t, output_dim)].abs() < 1e-10),
                        "future source must not affect past output"
                    );
                    expected[(output_dim, input_dim)] +=
                        valid.clone().map(|t| delta[(t, output_dim)]).sum::<f64>()
                            / valid.len() as f64;
                    same_position[(output_dim, input_dim)] +=
                        delta[(p, output_dim)] / valid.len() as f64;
                }
            }
        }
        close(&estimates[&source], &expected, 1e-8);
        assert!(
            (&estimates[&source] - same_position).norm() > 0.05,
            "must include cross-position effects"
        );
    }
    Ok(())
}
#[test]
fn valid_positions_mask_and_negative_layer_resolution_match_reference() -> Result<()> {
    assert_eq!(valid_positions(32, 16)?, 16..31);
    assert!(valid_positions(17, 16).is_err());
    assert!(valid_positions(0, 0).is_err());
    assert_eq!(
        resolve_layers(4, Some(&[-3, 0, 0]), Some(-1))?,
        (vec![0, 1], 3)
    );
    assert!(resolve_layers(4, Some(&[3]), None).is_err());
    Ok(())
}
#[test]
fn corpus_average_weights_prompts_equally_not_positions() -> Result<()> {
    let model = decoder(true, 0.25);
    let prompts = vec![prompt(4, 0.), prompt(12, 1.7)];
    let a = jacobian_for_prompt(&model, &prompts[0], &config())?;
    let b = jacobian_for_prompt(&model, &prompts[1], &config())?;
    let lens = fit(&model, &prompts)?;
    for l in [0, 1] {
        close(&lens.jacobians[&l], &((&a[&l] + &b[&l]) / 2.), 1e-12);
        assert!((&a[&l] - &b[&l]).norm() > 1e-3);
    }
    let merged = Lens::merge(&[fit(&model, &prompts[..1])?, fit(&model, &prompts[1..])?])?;
    close(&merged.jacobians[&0], &lens.jacobians[&0], 1e-12);
    let mut double = fit(&model, &prompts[..1])?;
    double.n_prompts = 2;
    let merged = Lens::merge(&[double, fit(&model, &prompts[1..])?])?;
    close(&merged.jacobians[&0], &((&a[&0] * 2. + &b[&0]) / 3.), 1e-12);
    Ok(())
}
#[test]
fn checkpoint_resume_after_short_prompt_matches_uninterrupted_fit() -> Result<()> {
    let model = decoder(true, 0.2);
    let prompts = vec![prompt(5, 0.), prompt(1, 0.), prompt(8, 0.8)];
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("checkpoint.json");
    let mut state = FitState::new(&model, &prompts, config())?;
    state.run_until(&model, &prompts, 2, Some(&path))?;
    assert_eq!(
        (state.next_index, state.successes, state.skipped),
        (2, 1, 1)
    );
    let mut resumed = FitState::resume(&path, &config())?;
    resumed.run_until(&model, &prompts, 3, Some(&path))?;
    let reference = fit(&model, &prompts)?;
    assert_eq!(resumed.lens()?.n_prompts, 2);
    close(
        &resumed.lens()?.jacobians[&0],
        &reference.jacobians[&0],
        1e-12,
    );
    resumed.run_until(&model, &prompts, 3, None)?;
    assert_eq!(resumed.successes, 2);
    let mut changed = prompts.clone();
    changed[0][(0, 0)] += 1.;
    assert!(resumed.run_until(&model, &changed, 3, None).is_err());
    let mut changed_model = model.clone();
    changed_model.weights[0][(0, 0)] += 0.01;
    assert!(
        resumed
            .run_until(&changed_model, &prompts, 3, None)
            .is_err()
    );
    let mut cfg = config();
    cfg.skip_first = 0;
    assert!(FitState::resume(&path, &cfg).is_err());
    let file = dir.path().join("lens.json");
    reference.save(&file)?;
    close(
        &Lens::load(&file)?.jacobians[&1],
        &reference.jacobians[&1],
        1e-12,
    );
    Ok(())
}
#[test]
fn coordinate_swap_preserves_orthogonal_content_and_is_an_involution() -> Result<()> {
    // Nonorthogonal directions distinguish pseudoinverse coordinates from dots.
    let basis = Matrix::from_row_slice(3, 2, &[1., 1., 0., 1., 0., 0.]);
    let frame = Frame::new(basis, 1e-10)?;
    let h = Vector::from_vec(vec![5., 2., 7.]);
    let swapped = frame.permute(&h, &[1, 0], 1.)?;
    assert!((frame.coordinates(&h)? - Vector::from_vec(vec![3., 2.])).norm() < 1e-12);
    assert!((frame.coordinates(&swapped)? - Vector::from_vec(vec![2., 3.])).norm() < 1e-12);
    assert!((swapped[2] - 7.).abs() < 1e-12);
    assert!((frame.permute(&swapped, &[1, 0], 1.)? - h).norm() < 1e-12);
    assert!(
        Frame::new(
            Matrix::from_row_slice(3, 2, &[1., 1., 0., 0., 0., 0.]),
            1e-10
        )
        .is_err()
    );
    Ok(())
}
#[test]
fn concept_swap_changes_downstream_target_with_norm_matched_negative_control() -> Result<()> {
    // Constructed causal model: two verbal concepts and one orthogonal feature.
    // This establishes intervention correctness, not natural-language alignment.
    let frame = Frame::new(
        Matrix::from_row_slice(3, 2, &[1., 0., 0., 1., 0., 0.]),
        1e-10,
    )?;
    let head = Matrix::from_row_slice(3, 3, &[1., 0., 0., 0., 1., 0., 0., 0., 0.]);
    let h = Vector::from_vec(vec![4., 0., 9.]);
    let patched = frame.permute(&h, &[1, 0], 1.)?;
    let delta_norm = (&patched - &h).norm();
    let control = steer(&h, &Vector::from_vec(vec![0., 0., 1.]), delta_norm)?;
    assert!(((&control - &h).norm() - delta_norm).abs() < 1e-12);
    let before = &head * &h;
    let after = &head * &patched;
    let null = &head * &control;
    let effect = measure_intervention(before.as_slice(), after.as_slice(), 1)?;
    assert_eq!(effect.rank_after, 0);
    assert!(effect.probability_after > effect.probability_before + 0.8);
    assert!(effect.kl_before_after > 1.);
    assert_eq!(before, null);
    assert_eq!(patched[2], h[2]);
    let ablated = frame.ablate(&h, 1.)?;
    assert!(frame.coordinates(&ablated)?.norm() < 1e-12);
    assert_eq!(ablated[2], h[2]);
    Ok(())
}
#[test]
fn sparse_nonnegative_reconstruction_handles_correlated_dictionary() -> Result<()> {
    let s = 2f64.sqrt();
    let dictionary = Matrix::from_row_slice(3, 3, &[1., 1. / s, 0., 0., 1. / s, 0., 0., 0., 1.]);
    let truth = Vector::from_vec(vec![2., 1., 0.]);
    let h = &dictionary * &truth;
    let fit = gradient_pursuit(&dictionary, &h, 2, 1000, 1e-10)?;
    assert!(fit.residual.norm() < 1e-7);
    assert!(fit.coefficients.iter().all(|&x| x >= 0.));
    assert_eq!(fit.coefficients[2], 0.);
    assert!(fit.squared_error.windows(2).all(|w| w[1] <= w[0] + 1e-12));
    assert!((&fit.reconstruction + &fit.residual - h).norm() < 1e-12);
    Ok(())
}
#[test]
fn slice_final_row_ranks_pins_and_tail_window_match_model() -> Result<()> {
    let model = decoder(false, 0.);
    let x = prompt(8, 0.2);
    let lens = fit(&model, &[prompt(6, 0.)])?;
    let cfg = SliceConfig {
        top_n: 2,
        max_tracked: Some(0),
        pinned: BTreeSet::from([4]),
        layer_stride: 2,
        last_n: Some(3),
        display_mask: None,
    };
    let slice = compute_slice(&model, &lens, &x, &cfg)?;
    assert_eq!(slice.layers, vec![0, 1, 2]);
    assert_eq!(slice.context_offset, 5);
    assert_eq!(slice.tracked_ids, vec![4]);
    assert_eq!(slice.cells.len(), 9);
    let states = model.residuals(&x)?;
    let actual = model.unembed(&states[2])?;
    for cell in slice.cells.iter().filter(|c| c.layer == 2) {
        let scores: Vec<_> = actual.row(cell.position).iter().copied().collect();
        let ranks = ranks(&scores)?;
        assert_eq!(cell.tracked_ranks[0], ranks[4]);
        assert_eq!(ranks[cell.top_ids[0]], 0);
    }
    let masked = SliceConfig {
        display_mask: Some(vec![false, false, false, true, true]),
        ..cfg
    };
    let slice = compute_slice(&model, &lens, &x, &masked)?;
    assert!(
        slice
            .cells
            .iter()
            .all(|c| c.top_ids.iter().all(|&id| id >= 3))
    );
    assert!(
        slice.cells.iter().any(|c| c.top_ranks[0] > 0),
        "mask must not change full-vocabulary rank"
    );
    Ok(())
}
#[test]
fn reporting_uses_tie_aware_correlations_and_wilson_intervals() -> Result<()> {
    assert!((spearman(&[1., 2., 2., 4.], &[4., 2., 2., 1.])? + 1.).abs() < 1e-12);
    assert!(spearman(&[1., 1.], &[1., 2.]).is_err());
    let (low, high) = wilson95(0, 10)?;
    assert!(low.abs() < 1e-12 && (high - 0.2775328).abs() < 1e-6);
    let p = probabilities(&[1000., 1001., 999.])?;
    assert!((p.iter().sum::<f64>() - 1.).abs() < 1e-12);
    Ok(())
}
