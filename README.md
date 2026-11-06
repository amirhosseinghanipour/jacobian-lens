# Jacobian Lens for Rust

Rust implementations of Jacobian lens fitting, vocabulary decoding, and J-space
interventions with runtime-independent model interfaces and Candle tensor storage.
Includes an optional Hugging Face runtime bridge, a native differentiable
Qwen2/Qwen2.5 backend, an independent f64 CPU reference, and the public Anthropic
evaluation prompts.

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
retains one suffix graph with zero-valued leaf perturbations at source layers,
reading all source gradients from each backward pass.

J-space token directions are rows of `W_U J`. `LensFrame` uses an SVD-checked
pseudoinverse for coordinate swaps and span ablations. With columns `V`, a full
swap is `h + V (swap(V†h) - V†h)`, preserving the orthogonal residual. Dependent
or ill-conditioned frames are rejected.

## Use

Add the library to a Rust project:

```sh
cargo add jacobian-lens
```

The library import is `jacobian_lens`; the CLI executable is `jlens`.
`jlens-research` is installed automatically as the CPU reference dependency.

From a repository checkout, run the research tests and example:

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
- `LensModel`, `JacobianEstimator`, `ResearchModel`: runtime-independent model contracts.
- `TransformerForward`: explicit block-output interception for native adapters.
- `forward_tracked`, `intervention_logits`: whole-model tracking and causal reruns.
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

## Models and runtimes

The public research APIs use `LensModel` and `JacobianEstimator`, not a concrete
model architecture. `ResearchModel` combines them for dynamic dispatch. `apply`,
evaluation, visualization, tracking, fitting, checkpoints, and J-space
interventions accept these interfaces. Readout-only runtimes need not implement
fitting. Native Candle adapters implementing `DifferentiableTransformer` receive
the estimator implementation automatically.

`LensModel::capture` runs a complete forward pass with selected block-output
captures and optional replacements. The runtime owns positional inputs, masks,
recurrent state, expert routing, and auxiliary context. Causal edits therefore
work without assuming that a transformer suffix can run independently. Final
normalization, head bias, and logit softcapping belong to the model's `Unembed`
implementation. `LensFrame::from_model` fetches only selected unembedding rows.

The optional `hf` feature starts a persistent Python worker using the installed
Transformers runtime and Anthropic's Python fitting implementation. Models are
loaded by Transformers auto classes; the Rust crate has no architecture allowlist
for this backend. The worker discovers decoder layout through embedding endpoints
and module structure. Explicit `HfLayout` overrides handle ambiguous layouts.
For other PyTorch runtimes, `HfOptions::factory` accepts `module:function`; the
function receives the options dictionary and returns a Python `LensModel`, plus
`unembedding_rows(ids)` for constructing J-space frames. Other runtimes, including
Burn integrations, can implement the Rust interfaces directly and exchange
ordinary Candle tensors at the boundary.

```sh
pip install -r scripts/requirements-hf.txt
JLENS_BACKEND=hf JLENS_DEVICE=cuda JLENS_PYTHON=python3 \
  cargo run --release --features hf --bin jlens -- fit \
  models/MODEL calibration.json lens.safetensors 6,10,14 4 128
```

Use the same `JLENS_BACKEND=hf` setting with the `eval` command. The CLI takes
local model/tokenizer directories; the library also accepts Hub identifiers and
revisions. `JLENS_REQUEST_TIMEOUT` sets the per-request timeout in seconds
(default 3600). The Python worker can use CUDA without compiling Candle's `cuda`
feature; Rust-side bridge tensors default to CPU. `HfModel::launch` also accepts
an explicit Rust output device. The native Qwen CLI remains the default.

```rust,ignore
use std::time::Duration;
use candle_core::Device;
use jacobian_lens::{hf::{HfModel, HfOptions}, LensModel, estimate_prompt};

let mut options = HfOptions::local("models/MODEL");
options.device = "cuda".into();
let model = HfModel::launch("python3", &options, &Device::Cpu, Duration::from_secs(3600))?;
let ids = model.encode("A calibration passage long enough for the selected mask", 128)?;
let trace = model.capture(&ids, &[6, 10], &[])?;
// let estimate = estimate_prompt(&model, &ids, &fitting_config)?;
// let readout = lens.apply(&model, &ids, &apply_options)?;
# Ok::<(), jacobian_lens::LensError>(())
```

Tensor exchange uses private safetensors files, not pickle; it copies data across
the runtime boundary. The model stays resident, and its autograd graph stays in
Python. Calls are blocking and serialized per worker, with request IDs, timeouts,
and worker cleanup. Use bounded blocking queues with async inference. Remote
model code is disabled unless explicitly enabled. Input IDs are preserved exactly
during fitting; `force_bos` defaults to false and is an explicit tokenizer policy.
A supplied separate tokenizer must match the checkpoint's token IDs.

Compatibility is a runtime contract, not a claim that every open-weight artifact
has been validated. A model needs accessible fixed-width residual boundaries,
the correct vocabulary readout, deterministic fresh-state execution, and
activation gradients for fitting. Quantized or inference-only kernels may support
readout without fitting. The built-in HF bridge handles text decoder inputs;
image/audio inputs, encoder context, unusual readouts, and nonstandard block
outputs require a custom adapter. Models without this residual/readout structure
need a different scientific definition of the lens.

The backend research suite uses small randomly initialized Llama, Qwen2, Qwen3,
GPT-2, GPT-NeoX, Gemma2, Mixtral, and Qwen3.5 text models, plus an unregistered
context-dependent PyTorch decoder. It compares block outputs, model logits,
full Jacobians, independent finite differences, and pseudoinverse concept swaps;
it also exercises checkpoint resume and simultaneous callers. These are
architecture compatibility checks, not pretrained-model quality measurements.
Qwen3.5 uses reference PyTorch kernels in these CPU tests; fused CUDA kernels need
separate validation.

```sh
python scripts/backend_oracles.py artifacts/backend-tests/models
PYTHONPATH=scripts JLENS_PYTHON=python3 \
  JLENS_HF_TEST_DIR=artifacts/backend-tests/models \
  cargo test --release --features hf --test hf architectures_preserve -- --ignored --nocapture
```

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

## CUDA calibration experiment

Build with a compatible NVIDIA CUDA toolkit and driver:

```sh
cargo build --release --features cuda --bins
JLENS_DEVICE=cuda cargo test --release --features cuda --test qwen -- --ignored --test-threads=1
```

`JLENS_DEVICE=cuda` selects GPU zero explicitly; unavailable CUDA returns an error.
`JLENS_DIM_BATCH` controls the independent cotangent batch for `jlens-experiment`.
The frozen design is in `experiments/qwen-calibration/protocol.json`.
Prepare independent 128-token FineWeb windows, then fit and evaluate:

```sh
pip install pyarrow huggingface-hub==0.30.2 tokenizers==0.22.2
python scripts/prepare_calibration.py models/Qwen2.5-0.5B-Instruct/tokenizer.json artifacts/calibration/corpus --revision 9bb295ddab0e05d785b879661af7260fed5140fc
python scripts/run_calibration.py models/Qwen2.5-0.5B-Instruct artifacts/calibration/corpus artifacts/calibration/results
```

The main 100-prompt sample is a prefix of the 1,000-prompt sample; the independent
100-prompt replication uses disjoint documents. Sampling shuffles source files
and a bounded buffer of documents; it is not an IID draw from all FineWeb or
Qwen's undisclosed pretraining mixture. Corpus manifests retain source attribution,
immutable revision, tokenizer and evaluation hashes, and token-window hashes via
shard files. FineWeb uses ODC-BY; source documents retain their respective rights.

Results include full-vocabulary concept ranks at each fixed layer, matched logit
lens readouts, coverage, target-prefix capability where labels exist, and paired
task-stratified bootstrap intervals. Causal endpoints use first continuation tokens
of upstream answer labels and must not be interpreted as full-answer accuracy.
Calibration shards resume atomically; checkpoints and raw results belong in
`artifacts/`, not the Git history. No minimum empirical score is asserted by tests.

The `merge-pilot` subcommand merges the first ten completed shards into the
100-prompt lens. `eval-development MODEL LENS_DIR OUTPUT SPLITS` evaluates only
the development split, including the recorded prior-smoke exclusions. The
published pilot measurements in `experiments/qwen-calibration/development/`
are development results; final test measurements require the completed run.
