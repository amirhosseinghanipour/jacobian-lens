//! Measurements for held-out lens probes and paired intervention experiments.
//! These summarize supplied observations; they never label a model "aligned".
use crate::{Result, ensure};
use serde::{Deserialize, Serialize};
fn valid(x: &[f64]) -> Result<()> {
    ensure(
        !x.is_empty() && x.iter().all(|v| v.is_finite()),
        "scores must be finite and nonempty",
    )
}
pub fn ranks(logits: &[f64]) -> Result<Vec<usize>> {
    valid(logits)?;
    let mut order: Vec<_> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let mut result = vec![0; logits.len()];
    for (rank, id) in order.into_iter().enumerate() {
        result[id] = rank;
    }
    Ok(result)
}
pub fn probabilities(logits: &[f64]) -> Result<Vec<f64>> {
    valid(logits)?;
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
    let z: f64 = p.iter().sum();
    for x in &mut p {
        *x /= z;
    }
    Ok(p)
}
fn midranks(x: &[f64]) -> Vec<f64> {
    let mut order: Vec<_> = (0..x.len()).collect();
    order.sort_unstable_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut ranks = vec![0.; x.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && x[order[end]] == x[order[start]] {
            end += 1;
        }
        let mean = (start + end - 1) as f64 / 2.;
        for i in start..end {
            ranks[order[i]] = mean;
        }
        start = end;
    }
    ranks
}
pub fn spearman(x: &[f64], y: &[f64]) -> Result<f64> {
    valid(x)?;
    valid(y)?;
    ensure(
        x.len() == y.len() && x.len() > 1,
        "correlation requires paired scores",
    )?;
    let (x, y) = (midranks(x), midranks(y));
    let mean = (x.len() - 1) as f64 / 2.;
    let covariance: f64 = x.iter().zip(&y).map(|(a, b)| (a - mean) * (b - mean)).sum();
    let xx: f64 = x.iter().map(|a| (a - mean).powi(2)).sum();
    let yy: f64 = y.iter().map(|a| (a - mean).powi(2)).sum();
    ensure(
        xx > 0. && yy > 0.,
        "correlation undefined for constant scores",
    )?;
    Ok((covariance / (xx * yy).sqrt()).clamp(-1., 1.))
}
/// Wilson score interval with z=1.95996 for a 95% binomial interval.
pub fn wilson95(successes: usize, trials: usize) -> Result<(f64, f64)> {
    ensure(trials > 0 && successes <= trials, "invalid binomial counts")?;
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = 1.959963984540054f64.powi(2);
    let center = p + z2 / (2. * n);
    let delta = (z2 * (p * (1. - p) / n + z2 / (4. * n * n))).sqrt();
    let denom = 1. + z2 / n;
    Ok(((center - delta) / denom, (center + delta) / denom))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionMeasurement {
    pub target_id: usize,
    pub rank_before: usize,
    pub rank_after: usize,
    pub probability_before: f64,
    pub probability_after: f64,
    pub target_logit_delta: f64,
    pub kl_before_after: f64,
}
pub fn measure_intervention(
    before: &[f64],
    after: &[f64],
    target_id: usize,
) -> Result<InterventionMeasurement> {
    ensure(
        before.len() == after.len() && target_id < before.len(),
        "invalid paired logits/target",
    )?;
    let p = probabilities(before)?;
    let q = probabilities(after)?;
    let log_z = |x: &[f64]| {
        let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        max + x.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
    };
    let (zp, zq) = (log_z(before), log_z(after));
    let kl = p
        .iter()
        .enumerate()
        .map(|(i, &p)| p * ((before[i] - zp) - (after[i] - zq)))
        .sum::<f64>()
        .max(0.);
    Ok(InterventionMeasurement {
        target_id,
        rank_before: ranks(before)?[target_id],
        rank_after: ranks(after)?[target_id],
        probability_before: p[target_id],
        probability_after: q[target_id],
        target_logit_delta: after[target_id] - before[target_id],
        kl_before_after: kl,
    })
}
