//! Research interventions use lens-vector coordinates, not raw dot products.
use crate::{Matrix, Result, Vector, ensure, finite};

/// Rows are token-associated directions W_U J. The model's normalization is
/// nonlinear and is deliberately not folded into these linear directions.
pub fn lens_vectors(head: &Matrix<f64>, jacobian: &Matrix<f64>) -> Result<Matrix<f64>> {
    finite(head)?;
    finite(jacobian)?;
    ensure(
        jacobian.is_square() && head.ncols() == jacobian.nrows() && head.ncols() > 0,
        "lens direction dimension mismatch",
    )?;
    Ok(head * jacobian)
}
/// Full-column-rank frame with SVD conditioning checks. Near-dependent frames
/// are rejected rather than yielding unstable or ambiguous coordinate swaps.
pub struct Frame {
    basis: Matrix<f64>,
    inverse: Matrix<f64>,
}
impl Frame {
    pub fn new(basis: Matrix<f64>, relative_tolerance: f64) -> Result<Self> {
        finite(&basis)?;
        ensure(
            basis.ncols() > 0 && basis.nrows() >= basis.ncols(),
            "frame must be nonempty and at most hidden width",
        )?;
        ensure(
            relative_tolerance.is_finite() && relative_tolerance > 0. && relative_tolerance < 1.,
            "invalid SVD tolerance",
        )?;
        let svd = basis.clone().svd(true, true);
        let max = svd.singular_values.max();
        let min = svd.singular_values.min();
        ensure(
            max > 0. && min > relative_tolerance * max,
            "rank-deficient or ill-conditioned frame",
        )?;
        let inverse = svd
            .pseudo_inverse(relative_tolerance * max)
            .map_err(|e| crate::Error::Invalid(e.into()))?;
        Ok(Self { basis, inverse })
    }
    pub fn coordinates(&self, h: &Vector<f64>) -> Result<Vector<f64>> {
        ensure(
            h.len() == self.basis.nrows() && h.iter().all(|x| x.is_finite()),
            "frame hidden dimension/value mismatch",
        )?;
        Ok(&self.inverse * h)
    }
    pub fn project(&self, h: &Vector<f64>) -> Result<Vector<f64>> {
        Ok(&self.basis * self.coordinates(h)?)
    }
    pub fn ablate(&self, h: &Vector<f64>, strength: f64) -> Result<Vector<f64>> {
        ensure(strength.is_finite(), "nonfinite ablation strength")?;
        Ok(h - self.project(h)? * strength)
    }
    /// h + alpha V (permutation(V†h) - V†h). alpha=1 is a full swap.
    pub fn permute(
        &self,
        h: &Vector<f64>,
        permutation: &[usize],
        alpha: f64,
    ) -> Result<Vector<f64>> {
        ensure(
            alpha.is_finite() && permutation.len() == self.basis.ncols(),
            "invalid permutation/strength",
        )?;
        let mut sorted = permutation.to_vec();
        sorted.sort_unstable();
        ensure(
            sorted.iter().copied().eq(0..self.basis.ncols()),
            "permutation must be bijective",
        )?;
        let c = self.coordinates(h)?;
        let target = Vector::from_iterator(c.len(), permutation.iter().map(|&i| c[i]));
        Ok(h + &self.basis * (target - c) * alpha)
    }
}
pub fn steer(h: &Vector<f64>, direction: &Vector<f64>, alpha: f64) -> Result<Vector<f64>> {
    ensure(
        h.len() == direction.len()
            && alpha.is_finite()
            && h.iter().chain(direction.iter()).all(|x| x.is_finite()),
        "invalid steering inputs",
    )?;
    Ok(h + direction * alpha)
}

#[derive(Debug)]
pub struct SparseDecomposition {
    pub coefficients: Vector<f64>,
    pub reconstruction: Vector<f64>,
    pub residual: Vector<f64>,
    pub squared_error: Vec<f64>,
}
/// Nonnegative gradient pursuit with a support cap. Dictionary columns must be
/// unit length. Each iteration chooses the highest positive residual correlation,
/// then updates the active support in the projected gradient direction with an
/// exact quadratic line step limited to preserve nonnegative coefficients.
/// This is a documented reference algorithm, not a claim of private-code parity.
pub fn gradient_pursuit(
    dictionary: &Matrix<f64>,
    h: &Vector<f64>,
    max_support: usize,
    iterations: usize,
    tolerance: f64,
) -> Result<SparseDecomposition> {
    finite(dictionary)?;
    ensure(
        dictionary.nrows() == h.len() && dictionary.ncols() > 0 && h.iter().all(|x| x.is_finite()),
        "invalid dictionary/activation",
    )?;
    ensure(
        max_support > 0
            && max_support <= dictionary.ncols()
            && iterations > 0
            && tolerance.is_finite()
            && tolerance > 0.,
        "invalid pursuit configuration",
    )?;
    ensure(
        dictionary
            .column_iter()
            .all(|v| (v.norm() - 1.).abs() < 1e-8),
        "dictionary columns must be unit length",
    )?;
    let mut c = Vector::<f64>::zeros(dictionary.ncols());
    let mut support = std::collections::BTreeSet::new();
    let mut residual = h.clone();
    let mut errors = vec![residual.norm_squared()];
    for _ in 0..iterations {
        let gradient = dictionary.transpose() * &residual;
        if support.len() < max_support
            && let Some(best) = (0..gradient.len())
                .filter(|i| !support.contains(i) && gradient[*i] > tolerance)
                .max_by(|&a, &b| gradient[a].total_cmp(&gradient[b]).then(b.cmp(&a)))
        {
            support.insert(best);
        }
        let mut g = Vector::zeros(c.len());
        for &i in &support {
            if c[i] > 0. || gradient[i] > 0. {
                g[i] = gradient[i];
            }
        }
        if g.norm_squared() < tolerance * tolerance {
            break;
        }
        let update = dictionary * &g;
        let denom = update.norm_squared();
        if denom <= f64::EPSILON {
            break;
        }
        let mut step = g.dot(&gradient) / denom;
        for &i in &support {
            if g[i] < 0. {
                step = step.min(-c[i] / g[i]);
            }
        }
        if step <= f64::EPSILON {
            break;
        }
        c += g * step;
        for value in c.iter_mut() {
            *value = value.max(0.);
        }
        residual = h - dictionary * &c;
        errors.push(residual.norm_squared());
        if residual.norm() <= tolerance {
            break;
        }
    }
    let reconstruction = dictionary * &c;
    Ok(SparseDecomposition {
        coefficients: c,
        reconstruction,
        residual,
        squared_error: errors,
    })
}
