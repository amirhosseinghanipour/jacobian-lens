#!/usr/bin/env python3
"""Generate independently computed PyTorch Qwen residual/logit/VJP fixtures.

Requires torch, transformers and safetensors. Uses local checkpoint files only.
"""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("output")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    # Long enough for the estimator's default attention-sink exclusion.
    ids = tokenizer("The researcher carefully compared the hidden representations across many layers of the language model before writing a precise and honest report about the experiment.", return_tensors="pt")["input_ids"][:, :32]
    source = len(model.model.layers) - 2
    captured = {}
    handles = []
    def capture(layer):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if layer == source:
                h = h.detach().requires_grad_(True)
            captured[layer] = h
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return hook
    for layer in [0, source, source + 1]:
        handles.append(model.model.layers[layer].register_forward_hook(capture(layer)))
    output = model(ids, use_cache=False)
    h = captured[source]
    target = captured[source + 1]
    valid = slice(16, ids.shape[1] - 1)
    if ids.shape[1] <= 17:
        raise ValueError("oracle prompt too short")
    # Distinct output coordinates test both orientation and dimension batching.
    dimensions = [0, 1, 7, 31]
    rows = []
    for dimension in dimensions:
        objective = target[:, valid, dimension].sum()
        gradient, = torch.autograd.grad(objective, h, retain_graph=True)
        rows.append(gradient[0, valid].mean(0))
    tensors = {"input_ids": ids.to(torch.int64),
               "logits": output.logits[:, -1].detach().contiguous(),
               "jacobian_rows": torch.stack(rows).detach().contiguous(),
               "dimensions": torch.tensor(dimensions, dtype=torch.int64)}
    for layer, hidden in captured.items():
        tensors[f"residual.{layer}"] = hidden.detach().contiguous()
    save_file(tensors, args.output, metadata={"source_layer": str(source), "skip_first": "16", "dtype": "float32", "attention": "eager"})
    for handle in handles:
        handle.remove()
    print(f"Saved {Path(args.output)} for {ids.shape[1]} positions, source layer {source}")


if __name__ == "__main__":
    main()
