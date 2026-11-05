//! Device-side J-space interventions with a checked CPU SVD at construction.
use crate::{JacobianLensRegistry, LensError, Result};
use candle_core::{DType, Tensor};
use jlens_research::Matrix;

pub struct LensFrame {
    basis: Tensor,
    dual: Tensor,
    width: usize,
    size: usize,
}
impl LensFrame {
    /// Token vectors form columns V; stores V.T and (V†).T for row activations.
    pub fn from_tokens(
        registry: &JacobianLensRegistry,
        layer: usize,
        head: &Tensor,
        tokens: &[u32],
        relative_tolerance: f64,
    ) -> Result<Self> {
        let basis = registry.token_directions(layer, head, tokens)?;
        Self::new(basis, relative_tolerance)
    }
    pub fn new(basis: Tensor, relative_tolerance: f64) -> Result<Self> {
        let (k, d) = basis.dims2()?;
        if k == 0
            || k > d
            || !relative_tolerance.is_finite()
            || relative_tolerance <= 0.
            || relative_tolerance >= 1.
        {
            return Err(LensError::Invalid(
                "invalid frame shape or SVD tolerance".into(),
            ));
        }
        let values = basis
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(LensError::Invalid("nonfinite frame".into()));
        }
        let matrix =
            Matrix::from_row_slice(k, d, &values.iter().map(|&x| x as f64).collect::<Vec<_>>());
        let svd = matrix.svd(true, true);
        let max = svd.singular_values.max();
        if max == 0. || svd.singular_values.min() <= relative_tolerance * max {
            return Err(LensError::Invalid(
                "ill-conditioned or dependent lens vectors".into(),
            ));
        }
        let inverse = svd
            .pseudo_inverse(relative_tolerance * max)
            .map_err(|e| LensError::Invalid(e.into()))?;
        let mut flat = Vec::with_capacity(d * k);
        for r in 0..d {
            for c in 0..k {
                flat.push(inverse[(r, c)] as f32);
            }
        }
        let dual = Tensor::from_vec(flat, (d, k), basis.device())?.to_dtype(basis.dtype())?;
        Ok(Self {
            basis,
            dual,
            width: d,
            size: k,
        })
    }
    fn flatten(&self, hidden: &Tensor) -> Result<Tensor> {
        if hidden.dims().last() != Some(&self.width)
            || hidden.dtype() != self.basis.dtype()
            || !hidden.device().same_device(self.basis.device())
        {
            return Err(LensError::Invalid(
                "frame hidden shape/dtype/device mismatch".into(),
            ));
        }
        Ok(hidden.reshape((hidden.elem_count() / self.width, self.width))?)
    }
    pub fn ablate(&self, hidden: &Tensor, strength: f64) -> Result<Tensor> {
        if !strength.is_finite() {
            return Err(LensError::Invalid("nonfinite ablation strength".into()));
        }
        let h = self.flatten(hidden)?;
        let projection = h.matmul(&self.dual)?.matmul(&self.basis)?;
        Ok((h - projection.affine(strength, 0.)?)?.reshape(hidden.shape())?)
    }
    /// alpha=1 applies the complete coordinate permutation; alpha=0 is a control.
    pub fn permute(&self, hidden: &Tensor, permutation: &[u32], alpha: f64) -> Result<Tensor> {
        let mut sorted = permutation.to_vec();
        sorted.sort_unstable();
        if !alpha.is_finite()
            || sorted.len() != self.size
            || !sorted.iter().copied().eq(0..self.size as u32)
        {
            return Err(LensError::Invalid("invalid coordinate permutation".into()));
        }
        let h = self.flatten(hidden)?;
        let coords = h.matmul(&self.dual)?;
        let ids = Tensor::new(permutation, hidden.device())?;
        let delta = (coords.index_select(&ids, 1)? - &coords)?
            .matmul(&self.basis)?
            .affine(alpha, 0.)?;
        Ok((h + delta)?.reshape(hidden.shape())?)
    }
}
