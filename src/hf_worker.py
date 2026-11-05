"""Private JSON/safetensors worker embedded in the Rust `hf` feature.

Model execution and Jacobian fitting use Anthropic's LensModel protocol. A custom
factory can supply that protocol without any model architecture in this crate.
"""
import importlib
import json
from pathlib import Path
import sys

# Reserve stdout exclusively for protocol frames, including during imports.
protocol_output = sys.stdout
sys.stdout = sys.stderr


def write_tensors(tensors):
    from safetensors.torch import save_file
    # Clone avoids shared-storage aliases between final and captured residuals.
    packed = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in tensors.items()
    }
    save_file(packed, str(directory / "output.safetensors"))


def discover_layout(hf):
    """Use runtime embedding endpoints rather than a family-name allowlist.

    Only accept an unambiguous decoder with one residual ModuleList and one
    final norm. Unusual layouts must be supplied explicitly or via a factory.
    """
    import torch
    from jlens.hf import Layout
    modules = list(hf.named_modules())
    embedding = hf.get_input_embeddings()
    head = hf.get_output_embeddings()
    embed_paths = [name for name, module in modules if module is embedding]
    head_paths = [name for name, module in modules if module is head and "." not in name]
    if len(embed_paths) != 1 or len(head_paths) != 1 or "." not in embed_paths[0]:
        raise ValueError("cannot discover decoder endpoints; supply layout or factory")
    path, embed = embed_paths[0].rsplit(".", 1)
    decoder = dict(modules)[path]
    count = hf.config.get_text_config().num_hidden_layers
    stacks = [name for name, module in decoder.named_children()
              if isinstance(module, torch.nn.ModuleList) and len(module) == count]
    norms = [name for name, module in decoder.named_children()
             if isinstance(module, torch.nn.LayerNorm) or "norm" in type(module).__name__.lower()]
    if len(stacks) != 1 or len(norms) != 1:
        raise ValueError("ambiguous residual stack/final norm; supply layout or factory")
    return Layout(path, layers=stacks[0], norm=norms[0], embed=embed, lm_head=head_paths[0])


def initialize(request):
    global model, directory, options, input_device, head_rows
    import torch
    import transformers
    from jlens.hf import Layout, from_hf

    options = request["options"]
    directory = Path(request["directory"])
    torch.set_num_threads(options["threads"])
    if options["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("dtype must be float32, float16 or bfloat16")
    dtype = getattr(torch, options["dtype"])
    if options["factory"]:
        module, function = options["factory"].split(":", 1)
        model = getattr(importlib.import_module(module), function)(options)
        input_device = getattr(model, "input_device", torch.device(options["device"]))
        head_rows = getattr(model, "unembedding_rows", None)
    else:
        from transformers import AutoTokenizer
        auto_class = options["auto_class"]
        if auto_class not in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            raise ValueError("unsupported auto class; use a custom factory")
        loading = {
            "revision": options["revision"],
            "local_files_only": options["local_files_only"],
            "trust_remote_code": options["trust_remote_code"],
        }
        hf = getattr(transformers, auto_class).from_pretrained(
            options["model"], dtype=dtype, attn_implementation="eager", **loading
        ).to(options["device"]).eval()
        tokenizer = AutoTokenizer.from_pretrained(options["tokenizer"] or options["model"], **loading)
        layout = Layout(**options["layout"]) if options["layout"] else None
        if layout is None:
            layout = discover_layout(hf)
        model = from_hf(hf, tokenizer, layout=layout, force_bos=options["force_bos"])
        input_device = model.input_device
        head = getattr(hf, model.layout.lm_head)
        head_rows = lambda ids: head.weight[torch.tensor(ids, device=head.weight.device)]
    if model.n_layers <= 0 or model.d_model <= 0:
        raise ValueError("model must expose nonempty residual blocks")
    # The protocol owns readout details (normalization, softcaps, scaling, bias).
    with torch.no_grad():
        logits = model.unembed(torch.zeros(1, model.d_model, device=input_device, dtype=dtype))
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("unembed must preserve leading dimensions")
    tokenizer_backend = getattr(model.tokenizer, "backend_tokenizer", None)
    if tokenizer_backend is not None:
        tokenizer_backend.save(str(directory / "tokenizer.json"))
    return {
        "dimensions": {"hidden_size": model.d_model, "layer_count": model.n_layers, "vocab_size": logits.shape[-1]},
        "metadata": {"model": options["model"], "revision": options["revision"],
                     "adapter": type(model).__name__, "torch": torch.__version__,
                     "transformers": transformers.__version__, "options": options},
    }


def capture(ids, layers, tensors):
    import torch
    from jlens.hooks import ActivationRecorder
    edits = {int(k.split(".")[1]): v for k, v in tensors.items() if k.startswith("replacement.")}
    handles = []
    def replace(value):
        def hook(module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            if hidden.shape != value.shape:
                raise ValueError("replacement shape differs from block output")
            value_on_device = value.to(device=hidden.device, dtype=hidden.dtype)
            if torch.is_tensor(output):
                return value_on_device
            if isinstance(output, tuple):
                return (value_on_device, *output[1:])
            if isinstance(output, list):
                return [value_on_device, *output[1:]]
            raise TypeError("unsupported block output container; supply a custom adapter")
        return hook
    try:
        for layer, value in edits.items():
            if not 0 <= layer < model.n_layers:
                raise ValueError("replacement layer outside model")
            handles.append(model.layers[layer].register_forward_hook(replace(value)))
        with torch.no_grad(), ActivationRecorder(model.layers, at=sorted(set(layers) | {model.n_layers - 1})) as recorder:
            model.forward(ids)
            result = {f"residual.{layer}": recorder.activations[layer].float() for layer in set(layers)}
            result["final_residual"] = recorder.activations[model.n_layers - 1].float()
            write_tensors(result)
    finally:
        for handle in handles:
            handle.remove()


def dispatch(request):
    if request["op"] == "shutdown":
        return {}
    if request["op"] == "init":
        return initialize(request)
    import torch
    from safetensors.torch import load_file
    tensors = load_file(str(directory / "input.safetensors"))
    op = request["op"]
    if op == "encode":
        ids = model.encode(request["text"], max_length=request["max_length"])
        write_tensors({"input_ids": ids.to(torch.int64)})
    elif op == "unembed":
        with torch.no_grad():
            write_tensors({"logits": model.unembed(tensors["hidden"].to(input_device)).float()})
    elif op == "head":
        if head_rows is None:
            raise ValueError("custom adapter must implement unembedding_rows(ids) for J-space frames")
        with torch.no_grad():
            write_tensors({"rows": head_rows(request["tokens"]).float()})
    elif op == "capture":
        capture(tensors["input_ids"].to(device=input_device, dtype=torch.int64), request["layers"], tensors)
    elif op == "jacobian":
        from jlens.fitting import jacobian_for_prompt
        ids = tensors["input_ids"].to(device=input_device, dtype=torch.int64)
        # Preserve caller-tokenized IDs exactly: no decoding/re-tokenization or
        # implicit BOS changes during fitting. Delegate the estimator upstream.
        class TokenizedModel:
            def __getattr__(self, name):
                return getattr(model, name)
            def encode(self, prompt, *, max_length):
                return ids[:, :max_length]
        config = request["config"]
        jacobians, seq_len, valid = jacobian_for_prompt(
            TokenizedModel(), "", config["source_layers"],
            target_layer=config["target_layer"], dim_batch=config["dim_batch"],
            max_seq_len=config["max_seq_len"], skip_first=config["skip_first"],
        )
        write_tensors({f"jacobians.{layer}": j for layer, j in jacobians.items()})
        return {"seq_len": seq_len, "valid_positions": valid}
    else:
        raise ValueError(f"unknown operation {op}")
    return {}


for line in sys.stdin:
    request_id = None
    try:
        request = json.loads(line)
        request_id = request["id"]
        result = dispatch(request)
        result["id"] = request_id
    except Exception as exc:
        result = {"id": request_id, "error": f"{type(exc).__name__}: {exc}"}
    protocol_output.write(json.dumps(result, allow_nan=False) + "\n")
    protocol_output.flush()
    if request_id is not None and request.get("op") == "shutdown":
        break
