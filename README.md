# Jacobian Lens for Rust

Rust implementations of Jacobian lens fitting, vocabulary decoding, and J-space
interventions, built on Candle. Includes a differentiable Qwen2/Qwen2.5 adapter,
an independent f64 CPU reference, and the public Anthropic evaluation prompts.

Based on [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
and the [Python reference](https://github.com/anthropics/jacobian-lens).

## Mathematics

For row-major activations `h[..., d]`:

```text
transported = h @ J[layer].T
logits = model.lm_head(model.final_norm(transported))
```

Layer indices refer to residual block outputs, before final normalization. The
normalization must match the model: Qwen uses RMSNorm, not LayerNorm. Lens scores
measure verbalization dispositions; their softmax is not a calibrated next-token
prediction or evidence that a model is aligned.

Fitting follows the reference estimator: sum derivatives over valid target
positions, average over valid source positions, then average prompts equally.
By default the first 16 positions and final position are excluded. `dim_batch`
controls independent output cotangents per backward pass. The Candle fitter
recomputes each source layer's suffix to obtain its leaf-variable gradients.

J-space token directions are rows of `W_U J`. `LensFrame` uses an SVD-checked
pseudoinverse for coordinate swaps and span ablations. With columns `V`, a full
swap is `h + V (swap(V†h) - V†h)`, preserving the orthogonal residual. Dependent
or ill-conditioned frames are rejected.

## Use

```sh
cargo test --all-targets
cargo test --manifest-path crates/research-core/Cargo.toml
cargo run --example causal
```

Download the small Qwen checkpoint (approximately 1 GB):

```sh
python3 scripts/download_qwen.py --revision 7ae557604adf67be50417f59c2c2f167def9a775
```

Fit selected block-output layers using a JSON array of calibration strings, then
evaluate a released prompt distribution:

```sh
cargo run --release --bin jlens -- fit \
  models/Qwen2.5-0.5B-Instruct calibration.json lens.safetensors 10,14,18,22 8 128
cargo run --release --bin jlens -- eval \
  models/Qwen2.5-0.5B-Instruct lens.safetensors \
  data/evaluations/lens-eval-multihop.json target results.json 10
```

Fitting automatically resumes from the adjacent `.fit.safetensors` checkpoint.
Content hashes bind checkpoints to the model, tokenizer, and ordered corpus.
Changing any of those artifacts or fitting options rejects resume.

Keep calibration prompts separate from evaluation prompts. Model-wide fitting is
compute intensive; a small corpus or a single late layer is useful for numerical
checks, not a substitute for a representative calibration corpus.

Evaluation positions are `target` (immediately before the answer), `newline`
(poetry), or `last` (association and typo). Target-based evaluation tokenizes the
prompt and target together, then removes target-overlapping tokens. Targets only
locate readout positions; scores use `intermediates`. Pass@k is an item-macro
average over intermediate concepts and fitted layers. Uncovered multi-token
concepts count as misses, and single-token coverage is reported explicitly.
The library accepts explicit synonym maps for arithmetic and operation names.

## Library API

- `JacobianLensRegistry`: load, save, merge, transport, `decode_jspace`, and
  selected token directions. `apply` supports negative positions and the logit-lens
  baseline. Tensors are immutable and shareable across workers.
- `MappedLens`: zero-copy safetensors views borrowing an owned file mapping.
- `LensTracker`: selected layer/position top-K summaries, bounded across forwards.
- `TransformerForward`: explicit block-output interception for model adapters.
- `DifferentiableTransformer`, `jacobian_for_prompt`, `jacobian_rows`, `fit`:
  deterministic full-sequence Jacobian estimation with frozen model weights.
- `qwen2::Qwen2`: dense Qwen2/Qwen2.5 loading, residual capture, differentiable
  block execution, final readout, and suffix execution after interventions.
- `LensFrame`: device-side coordinate permutations and ablations.
- `visualization::compute_slice`, `build_page`: token rank tracking and standalone
  interactive HTML, with the actual model output as the final row.
- `research`: f64 reference fitting, atomic resumable checkpoints, sparse
  nonnegative gradient pursuit, layer/position slices, ranks, and paired metrics.

The Qwen adapter uses full-sequence causal attention, grouped query attention,
RoPE, SwiGLU, tied embeddings where configured, and differentiable normalization.
Sliding-window attention is rejected. It accepts unpadded batches; it does not
maintain an incremental KV cache.

```rust,no_run
use candle_core::{DType, Device};
use jacobian_lens::{JacobianLensRegistry, qwen2::Qwen2};

let device = Device::Cpu;
let model = Qwen2::load("models/Qwen2.5-0.5B-Instruct", &device, DType::F32)?;
let lens = JacobianLensRegistry::load("lens.safetensors", &device, DType::F32)?;
// Obtain ids using the checkpoint's tokenizer.json.
// let states = model.residuals(&ids)?;
// let logits = lens.decode_jspace(14, &states[14], &model)?;
# Ok::<(), jacobian_lens::LensError>(())
```

Share registries/readouts with `Arc`; use request-local trackers. Run synchronous
tensor work on bounded blocking workers when integrating with Tokio. CUDA and
Metal support are exposed through the corresponding Cargo features.

## Weight format and memory

Weights use safetensors names `jacobians.<layer>`, each a square `[output, input]`
F32/F16/BF16 matrix. Convert Anthropic `.pt` lens files with:

```sh
pip install torch safetensors
python3 scripts/convert_lens.py lens.pt lens.safetensors
```

Use F32 transport for the Python reference convention. Match the exact model and
tokenizer checkpoints; dimensions alone cannot establish artifact compatibility.
Non-contiguous direct transport inputs are rejected; callers explicitly opt into
packing with `.contiguous()`. Transposed projections are packed once at load.

`MappedLens::open` requires an unsafe guarantee that no process modifies or
truncates the file while mapped views exist. Safetensors views are zero-copy;
Candle materialization copies/uploads into owned tensor storage. The safe
buffered loader is available when file immutability cannot be guaranteed.

Exact top-K selection transfers selected logits to the CPU and uses partial
selection with deterministic token-ID tie breaking. Only token summaries persist
in a tracker. Token fragments retain both raw vocabulary spelling and decoded
text; individual byte tokens can decode lossily.

## Research tests

The independent CPU tests verify source-mean/target-sum Jacobians against
positionwise finite differences, exact readout on a solvable decoder, prompt
weighting, checkpoint resume, nonorthogonal coordinate swaps, norm-matched causal
controls, sparse reconstruction, and full-vocabulary rank tracking. The Candle
fitter is checked against that independent forward-mode implementation.

Real-checkpoint tests compare Qwen residuals, logits, and Jacobian rows against
PyTorch, verify finite-difference predictions, fit and serialize a complete
late-layer Jacobian, and perform a held-out concept swap. Run them explicitly:

```sh
pip install torch transformers==4.49.0 safetensors
python3 scripts/qwen_oracle.py models/Qwen2.5-0.5B-Instruct /tmp/qwen-oracle.safetensors
JLENS_QWEN_DIR=models/Qwen2.5-0.5B-Instruct \
JLENS_QWEN_ORACLE=/tmp/qwen-oracle.safetensors \
cargo test --release --test qwen -- --ignored --nocapture --test-threads=1
```

These tests establish numerical and intervention behavior. Empirical replication
of workspace or alignment findings requires the corresponding models, calibration
corpora, experimental controls, and held-out outcomes. Public prompt data is
included under Apache-2.0; see `NOTICE` for attribution and upstream revision.
