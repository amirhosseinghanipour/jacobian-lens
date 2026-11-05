//! Small causal models for independent numerical validation, with exact JVPs.
use crate::{Matrix, Result, ensure, finite};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub trait DifferentiableDecoder: Send + Sync {
    fn width(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn fingerprint(&self) -> Result<String>;
    fn residuals(&self, input: &Matrix<f64>) -> Result<Vec<Matrix<f64>>>;
    /// Runs blocks source+1..=target, including exact directional derivatives.
    fn suffix_jvp(
        &self,
        source: usize,
        target: usize,
        hidden: &Matrix<f64>,
        tangent: &Matrix<f64>,
    ) -> Result<(Matrix<f64>, Matrix<f64>)>;
    fn unembed(&self, hidden: &Matrix<f64>) -> Result<Matrix<f64>>;
}

/// Positionwise residual MLP with optional causal prefix-mean mixing. This is
/// an exactly analyzable test decoder, not a pretrained language model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceDecoder {
    pub weights: Vec<Matrix<f64>>, // [output, input]
    pub head: Matrix<f64>,         // [vocab, hidden]
    pub temporal_mix: f64,
    pub nonlinear: bool,
    pub rms_epsilon: Option<f64>,
}
impl ReferenceDecoder {
    pub fn validate(&self) -> Result<()> {
        let d = self.head.ncols();
        ensure(
            d > 0 && self.head.nrows() > 0 && !self.weights.is_empty(),
            "empty decoder",
        )?;
        finite(&self.head)?;
        ensure(self.temporal_mix.is_finite(), "nonfinite temporal mix")?;
        if let Some(e) = self.rms_epsilon {
            ensure(e.is_finite() && e > 0., "invalid RMS epsilon")?;
        }
        for w in &self.weights {
            ensure(w.shape() == (d, d), "block width mismatch")?;
            finite(w)?;
        }
        Ok(())
    }
    fn block(&self, layer: usize, h: &Matrix<f64>, dh: &Matrix<f64>) -> (Matrix<f64>, Matrix<f64>) {
        let w = &self.weights[layer];
        let mut x = h * w.transpose();
        let mut dx = dh * w.transpose();
        for p in 0..h.nrows() {
            for c in 0..h.ncols() {
                x[(p, c)] +=
                    self.temporal_mix * (0..=p).map(|q| h[(q, c)]).sum::<f64>() / (p + 1) as f64;
                dx[(p, c)] +=
                    self.temporal_mix * (0..=p).map(|q| dh[(q, c)]).sum::<f64>() / (p + 1) as f64;
            }
        }
        if self.nonlinear {
            for (x, dx) in x.iter_mut().zip(dx.iter_mut()) {
                *x = x.tanh();
                *dx *= 1. - *x * *x;
            }
        }
        (h + x, dh + dx)
    }
}
impl DifferentiableDecoder for ReferenceDecoder {
    fn width(&self) -> usize {
        self.head.ncols()
    }
    fn num_layers(&self) -> usize {
        self.weights.len()
    }
    fn fingerprint(&self) -> Result<String> {
        self.validate()?;
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }
    fn residuals(&self, input: &Matrix<f64>) -> Result<Vec<Matrix<f64>>> {
        self.validate()?;
        finite(input)?;
        ensure(
            input.nrows() > 0 && input.ncols() == self.width(),
            "invalid decoder input",
        )?;
        let zero = Matrix::zeros(input.nrows(), input.ncols());
        let mut h = input.clone();
        let mut out = Vec::new();
        for l in 0..self.num_layers() {
            h = self.block(l, &h, &zero).0;
            finite(&h)?;
            out.push(h.clone());
        }
        Ok(out)
    }
    fn suffix_jvp(
        &self,
        source: usize,
        target: usize,
        hidden: &Matrix<f64>,
        tangent: &Matrix<f64>,
    ) -> Result<(Matrix<f64>, Matrix<f64>)> {
        self.validate()?;
        ensure(
            source <= target && target < self.num_layers(),
            "invalid suffix layers",
        )?;
        ensure(
            hidden.shape() == tangent.shape()
                && hidden.ncols() == self.width()
                && hidden.nrows() > 0,
            "invalid suffix input",
        )?;
        finite(hidden)?;
        finite(tangent)?;
        let (mut h, mut dh) = (hidden.clone(), tangent.clone());
        for l in source + 1..=target {
            (h, dh) = self.block(l, &h, &dh);
        }
        finite(&h)?;
        finite(&dh)?;
        Ok((h, dh))
    }
    fn unembed(&self, hidden: &Matrix<f64>) -> Result<Matrix<f64>> {
        self.validate()?;
        finite(hidden)?;
        ensure(hidden.ncols() == self.width(), "readout width mismatch")?;
        let mut h = hidden.clone();
        if let Some(epsilon) = self.rms_epsilon {
            for p in 0..h.nrows() {
                let norm = (h.row(p).norm_squared() / h.ncols() as f64 + epsilon).sqrt();
                h.row_mut(p).scale_mut(1. / norm);
            }
        }
        let out = h * self.head.transpose();
        finite(&out)?;
        Ok(out)
    }
}
