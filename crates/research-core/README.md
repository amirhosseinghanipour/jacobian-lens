# jlens-research

Deterministic f64 CPU reference algorithms for Jacobian lens research, by
Amirhossein Ghanipour. This is the reference companion to
[jacobian-lens](https://github.com/amirhosseinghanipour/jacobian-lens), not a
pretrained model runtime.

The crate provides source-mean/target-sum Jacobian estimation, atomic resumable
fitting, J-space pseudoinverse coordinate swaps and span ablations, sparse
nonnegative reconstruction, full-vocabulary ranks, and intervention metrics.
Independent research tests compare the estimator to finite differences and check
causal edits on a solvable decoder.

```sh
cargo test
```

Based on the [Jacobian Lens reference](https://github.com/anthropics/jacobian-lens)
and [workspace paper](https://transformer-circuits.pub/2026/workspace/index.html).
Numerical correctness does not establish model alignment or replicate every
empirical finding in the paper. Licensed under Apache-2.0.
