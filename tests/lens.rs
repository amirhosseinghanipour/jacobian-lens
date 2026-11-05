use candle_core::{DType, Device, Module, Tensor};
use jacobian_lens::*;
use std::collections::BTreeMap;
use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

fn registry() -> Result<JacobianLensRegistry> {
    // Deliberately nonsymmetric: catches incorrect transpose conventions.
    JacobianLensRegistry::new(BTreeMap::from([(
        0,
        Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?,
    )]))
}
fn tokenizer() -> Tokenizer {
    Tokenizer::new(
        WordLevel::builder()
            .vocab(
                [("zero".into(), 0), ("one".into(), 1), ("two".into(), 2)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("zero".into())
            .build()
            .unwrap(),
    )
}
#[derive(Clone)]
struct Identity;
impl Module for Identity {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        Ok(x.clone())
    }
}
#[test]
fn transport_orientation_batching_and_errors() -> Result<()> {
    let r = registry()?;
    let h = Tensor::new(&[[[5f32, 6.], [1., 0.]]], &Device::Cpu)?;
    assert_eq!(
        r.transport(0, &h)?.to_vec3::<f32>()?,
        vec![vec![vec![17., 39.], vec![1., 3.]]]
    );
    assert!(matches!(
        r.transport(2, &h),
        Err(LensError::MissingLayer(2))
    ));
    assert!(matches!(
        r.transport(0, &Tensor::zeros((1, 3), DType::F32, &Device::Cpu)?),
        Err(LensError::Shape { .. })
    ));
    let strided = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?.t()?;
    assert!(matches!(
        r.transport(0, &strided),
        Err(LensError::NonContiguous(_))
    ));
    Ok(())
}
#[test]
fn readout_and_full_vocabulary_probabilities() -> Result<()> {
    let r = registry()?;
    let head = Tensor::new(&[[1f32, 0.], [0., 1.], [1., 1.]], &Device::Cpu)?;
    let readout = ModelReadout::new(Identity, head, None)?;
    let logits = r.decode_jspace(0, &Tensor::new(&[1f32, 0.], &Device::Cpu)?, &readout)?;
    assert_eq!(logits.to_vec1::<f32>()?, vec![1., 3., 4.]);
    let scores = top_k(&logits, 2, &tokenizer())?;
    assert_eq!(
        scores[0].iter().map(|s| s.id).collect::<Vec<_>>(),
        vec![2, 1]
    );
    let expected = 1. / (1. + (-1f64).exp() + (-3f64).exp());
    assert!((scores[0][0].probability - expected).abs() < 1e-8);
    assert_eq!(scores[0][0].text, "two");
    let ties = top_k(
        &Tensor::new(&[0f32, 0., 0.], &Device::Cpu)?,
        2,
        &tokenizer(),
    )?;
    assert_eq!(ties[0].iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1]);
    assert!(top_k(&logits, 0, &tokenizer()).is_err());
    assert!(top_k(&Tensor::new(&[f32::NAN], &Device::Cpu)?, 1, &tokenizer()).is_err());
    Ok(())
}
#[test]
fn mmap_materialization_outlives_mapping() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("lens.safetensors");
    Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?.save_safetensors("jacobians.0", &path)?;
    let r = {
        // SAFETY: test owns the file exclusively and never mutates it.
        let mapped = unsafe { MappedLens::open(&path)? };
        assert_eq!(mapped.tensors()?.tensor("jacobians.0")?.shape(), &[2, 2]);
        mapped.materialize(&Device::Cpu, DType::F32)?
    };
    assert_eq!(
        r.transport(0, &Tensor::new(&[5f32, 6.], &Device::Cpu)?)?
            .to_vec1::<f32>()?,
        vec![17., 39.]
    );
    let safe = JacobianLensRegistry::load(&path, &Device::Cpu, DType::F32)?;
    assert_eq!(safe.width(), 2);
    std::fs::write(&path, b"invalid")?;
    assert!(JacobianLensRegistry::load(&path, &Device::Cpu, DType::F32).is_err());
    Ok(())
}
struct Model {
    readout: ModelReadout<Identity>,
}
impl TransformerForward for Model {
    fn embed(&mut self, _: &Tensor) -> Result<Tensor> {
        Ok(Tensor::new(&[[[1f32, 0.], [0., 1.]]], &Device::Cpu)?)
    }
    fn num_layers(&self) -> usize {
        1
    }
    fn block(&mut self, _: usize, h: &Tensor) -> Result<Tensor> {
        Ok(h.affine(2., 0.)?)
    }
    fn unembed(&self, h: &Tensor) -> Result<Tensor> {
        self.readout.logits(h)
    }
}
#[test]
fn explicit_hook_and_bounded_tracking_preserve_model_output() -> Result<()> {
    let r = registry()?;
    let tok = tokenizer();
    let make = || {
        ModelReadout::new(
            Identity,
            Tensor::new(&[[1f32, 0.], [0., 1.], [1., 1.]], &Device::Cpu)?,
            None,
        )
    };
    let readout = make()?;
    let mut model = Model { readout: make()? };
    let mut tracker = LensTracker::new(&r, &readout, &tok, [0], vec![(0, 1)], 2)?;
    let ids = Tensor::new(&[[0u32, 1]], &Device::Cpu)?;
    let baseline = model.forward(&ids, None)?.to_vec3::<f32>()?;
    for _ in 0..3 {
        assert_eq!(
            model.forward(&ids, Some(&mut tracker))?.to_vec3::<f32>()?,
            baseline
        );
        assert_eq!(tracker.observations().len(), 1);
        let o = &tracker.observations()[0];
        assert_eq!((o.layer, o.batch, o.position), (0, 0, 1));
        assert_eq!(o.tokens[0].logit, 12.); // block [0,2] -> J [4,8] -> sum
    }
    Ok(())
}
#[test]
fn thread_safety_contracts() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<JacobianLensRegistry>();
    assert_send_sync::<MappedLens>();
    assert_send_sync::<ModelReadout<Identity>>();
    assert_send_sync::<LensTracker<'static>>();
}

struct Rms;
impl Module for Rms {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.broadcast_div(&(x.sqr()?.mean_keepdim(candle_core::D::Minus1)? + 1e-6)?.sqrt()?)
    }
}
#[test]
fn normalization_occurs_after_transport_and_before_head() -> Result<()> {
    let r = registry()?;
    let readout = ModelReadout::new(
        Rms,
        Tensor::new(&[[1f32, 0.], [0., 1.], [1., -1.]], &Device::Cpu)?,
        Some(Tensor::new(&[0f32, 1., 2.], &Device::Cpu)?),
    )?;
    let result = r
        .decode_jspace(0, &Tensor::new(&[1f32, 0.], &Device::Cpu)?, &readout)?
        .to_vec1::<f32>()?;
    // J*[1,0] = [1,3]; RMS^2 = (1+9)/2 = 5.
    let denom = (5f32 + 1e-6).sqrt();
    let expected = [1. / denom, 3. / denom + 1., -2. / denom + 2.];
    for (actual, expected) in result.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5);
    }
    Ok(())
}
