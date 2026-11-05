# Jacobian Lens for Rust

Candle inference library for applying precomputed corpus-averaged Jacobians and
tracking token readouts at explicit transformer block boundaries. Includes a
small causal decoder example, an Anthropic checkpoint converter and numerical
integration tests. Includes a differentiable Qwen2/Qwen2.5 adapter and an exact Jacobian estimator.

```sh
pip install torch safetensors
python3 scripts/convert_lens.py lens.pt lens.safetensors
cargo test --all-targets
cargo run --example causal
```

## Conventions and scientific scope

For row-major residuals `h[..., d]`, compute `h @ J.T`, apply the model's actual
final normalization, then its LM head. `J` on disk is `[output, input]`.
Layer zero means the output of residual block zero. Intercept before the model's
final normalization: applying that norm twice changes the result. Use RMSNorm
for models that use RMSNorm, and the original epsilon, affine weights and head
bias. The readout accepts any Candle normalization module or a custom `Unembed`.

The [paper's methods](https://transformer-circuits.pub/2026/workspace/index.html)
include effects on current and future positions, averaged across contexts.
Lens scores describe verbalization dispositions; their softmax is not a
calibrated next-token prediction. Top-K tracking is a readout of J-space, not an
orthogonal projection onto a uniquely defined subspace. The `research` module provides coordinate swaps, ablations and a documented
nonnegative gradient-pursuit reference algorithm.

The [reference application](https://github.com/anthropics/jacobian-lens/blob/main/jlens/lens.py)
uses block outputs and fp32 residual/Jacobian transport; load with `DType::F32`
for that convention. F16/BF16 compute is opt-in and needs backend support and
numerical validation on your model. The reference checkpoint format is `.pt`,
so run the converter first. The safetensors names are `jacobians.<layer>`.

## Integration

```rust,no_run
use candle_core::{DType, Device};
use jacobian_lens::{JacobianLensRegistry, LensTracker, Result, TransformerForward, Unembed};
use tokenizers::Tokenizer;

fn infer(
    model: &mut impl TransformerForward,
    readout: &dyn Unembed,
    tokenizer: &Tokenizer,
    ids: &candle_core::Tensor,
) -> Result<candle_core::Tensor> {
    // Load once at worker startup in a real service, not per request.
    let registry = JacobianLensRegistry::load("lens.safetensors", &Device::Cpu, DType::F32)?;
    let mut tracker = LensTracker::new(
        &registry, readout, tokenizer, [0, 4], vec![(0, 3)], 10,
    )?;
    let logits = model.forward(ids, Some(&mut tracker))?;
    for observation in tracker.observations() {
        println!("{observation:?}");
    }
    Ok(logits)
}
```

Implement `TransformerForward::embed`, `block`, and `unembed` for your decoder;
the default execution loop implements interception without runtime hooks.
The adapter must handle its architecture's masks, positional embeddings and
request-local KV cache. The example is a single-head causal attention/MLP stack
with random weights and identity lenses, intended only to demonstrate wiring.
It is not a trained language model and its tokens have no scientific meaning.
Tracker positions are `(batch, local_sequence_position)`; streaming adapters
must attach their own absolute token offsets to exported observations.

`ModelReadout::new(norm, head, bias)` accepts the original norm module and a
PyTorch-layout head `[vocab, hidden]`. Alternatively implement `Unembed` directly
to reuse an existing model's exact output path. Registry and head transposes are
packed once; hidden tensors are borrowed, detached for lens computation and
cast only if needed. Non-contiguous registry inputs and direct transport inputs
are rejected: callers explicitly decide whether to pay for `.contiguous()`.

## Memory mapping and concurrency

`MappedLens::open` is unsafe because external modification/truncation of a mapped
file can invalidate memory. Its safety contract requires immutable backing files.
`tensors()` yields genuinely zero-copy safetensors views with lifetimes tied to
the mapping. `materialize(device, dtype)` copies/uploads data into Candle storage
and packs transposes. The registry remains valid after dropping the mapping.

**Candle tensors are not zero-copy views of mmap storage.** Its
[safetensors implementation](https://docs.rs/candle-core/latest/src/candle_core/safetensors.rs.html)
constructs owned tensor storage. End-to-end zero-copy tensor loading would require
a different storage backend/custom kernels. This library exposes the copy
boundary rather than disguising it. The safe `JacobianLensRegistry::load` path
uses a byte buffer when file immutability cannot be guaranteed. Both loaders
accept F32/F16/BF16 weights and validate square dimensions and layer names.
No model/tokenizer identity is inferred from shape: match artifacts to the exact
checkpoint and tokenizer yourself.

Share immutable registries and readouts with `Arc`; keep one tracker and mutable
KV-cache/model adapter per request/worker. Compile-time tests assert `Send + Sync`
for registries, mappings, readouts and trackers. Run synchronous tensor work on a
bounded blocking pool (for example Tokio `spawn_blocking`), not async reactor
threads. Build one registry per device. Thread safety does not imply simultaneous
GPU execution or unlimited throughput.

Selected positions are decoded together per layer. Only token summaries persist;
repeated layers replace prior observations and `forward` clears prior results.
On an inference error discard partial observations. Exact CPU top-K transfers
selected logits and uses partial selection, O(V + K log K) time and O(V) scratch
per row, plus the selected logits buffer. It computes a full-vocabulary stable
softmax denominator. Ties use ascending token ID. GPU top-K and vocabulary-chunked
unembedding are future performance work; no benchmark claims are made here.

## Tests

```sh
cargo test --all-targets
cargo test --manifest-path crates/research-core/Cargo.toml
```

The research reference tests check Jacobian estimates against independent
positionwise finite differences, exact readout on a solvable decoder, causal
coordinate swaps with norm-matched controls, sparse reconstruction, prompt-weighted
averaging, resumable fitting, and full-vocabulary rank tracking. Numerical
correctness tests do not establish alignment or reproduce empirical findings
on pretrained language models.
