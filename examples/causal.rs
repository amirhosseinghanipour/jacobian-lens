//! Small executable single-head causal decoder demonstrating block-output hooks.
//! Random weights and identity Jacobians exercise plumbing, not interpretability.
use candle_core::{D, DType, Device, Module, Tensor};
use jacobian_lens::*;
use std::collections::BTreeMap;
use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

#[derive(Clone)]
struct Rms {
    weight: Tensor,
}
impl Module for Rms {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let scale = (x.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
        x.broadcast_div(&scale)?.broadcast_mul(&self.weight)
    }
}
struct Block {
    norm: Rms,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    up: Tensor,
    down: Tensor,
}
struct Decoder {
    embedding: Tensor,
    blocks: Vec<Block>,
    readout: ModelReadout<Rms>,
}
impl TransformerForward for Decoder {
    fn embed(&mut self, ids: &Tensor) -> Result<Tensor> {
        let (b, s) = ids.dims2()?;
        Ok(self
            .embedding
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((b, s, 4))?)
    }
    fn num_layers(&self) -> usize {
        self.blocks.len()
    }
    fn block(&mut self, layer: usize, h: &Tensor) -> Result<Tensor> {
        let block = &self.blocks[layer];
        let (_, seq, _) = h.dims3()?;
        let x = block.norm.forward(h)?;
        let q = x.broadcast_matmul(&block.q)?;
        let k = x.broadcast_matmul(&block.k)?;
        let v = x.broadcast_matmul(&block.v)?;
        let mask: Vec<f32> = (0..seq)
            .flat_map(|i| (0..seq).map(move |j| if j > i { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq, seq), h.device())?;
        let scores = q
            .matmul(&k.transpose(1, 2)?)?
            .affine(0.5, 0.)?
            .broadcast_add(&mask)?;
        let attention = candle_nn::ops::softmax_last_dim(&scores)?
            .matmul(&v)?
            .broadcast_matmul(&block.o)?;
        let h = (h + attention)?;
        let mlp = block
            .norm
            .forward(&h)?
            .broadcast_matmul(&block.up)?
            .silu()?
            .broadcast_matmul(&block.down)?;
        Ok((h + mlp)?)
    }
    fn unembed(&self, h: &Tensor) -> Result<Tensor> {
        self.readout.logits(h)
    }
}
fn main() -> Result<()> {
    let device = Device::Cpu;
    let norm = Rms {
        weight: Tensor::ones(4, DType::F32, &device)?,
    };
    let weight = || Tensor::randn(0f32, 0.1, (4, 4), &device);
    let head = Tensor::randn(0f32, 0.1, (3, 4), &device)?;
    let mut model = Decoder {
        embedding: Tensor::randn(0f32, 0.1, (3, 4), &device)?,
        blocks: (0..2)
            .map(|_| {
                Ok(Block {
                    norm: norm.clone(),
                    q: weight()?,
                    k: weight()?,
                    v: weight()?,
                    o: weight()?,
                    up: weight()?,
                    down: weight()?,
                })
            })
            .collect::<Result<_>>()?,
        readout: ModelReadout::new(norm.clone(), head.clone(), None)?,
    };
    let registry = JacobianLensRegistry::new(BTreeMap::from([
        (0, Tensor::eye(4, DType::F32, &device)?),
        (1, Tensor::eye(4, DType::F32, &device)?),
    ]))?;
    let readout = ModelReadout::new(norm, head, None)?;
    let vocab = [("hello".into(), 0), ("world".into(), 1), ("!".into(), 2)]
        .into_iter()
        .collect();
    let tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab)
            .unk_token("!".into())
            .build()
            .map_err(|e| LensError::Tokenizer(e.to_string()))?,
    );
    let mut tracker = LensTracker::new(&registry, &readout, &tokenizer, [0, 1], vec![(0, 1)], 2)?;
    let logits = model.forward(&Tensor::new(&[[0u32, 1]], &device)?, Some(&mut tracker))?;
    println!("model logits: {:?}", logits.dims());
    for observation in tracker.observations() {
        println!("{observation:?}");
    }
    Ok(())
}
