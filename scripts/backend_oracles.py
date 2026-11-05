#!/usr/bin/env python3
"""Build small, random-weight architecture fixtures and independent research oracles.

These verify runtime interoperability, not pretrained model quality. No checkpoint
is downloaded. CPU-only; requires Transformers, PyTorch, tokenizers, safetensors.
"""
import argparse
import json
from pathlib import Path

import torch
import transformers as hf
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def models():
    common = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                  num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                  max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    yield "llama", hf.LlamaConfig(**common)
    yield "qwen2", hf.Qwen2Config(**common)
    yield "qwen3", hf.Qwen3Config(**common, head_dim=8)
    yield "gpt2", hf.GPT2Config(vocab_size=32, n_embd=16, n_layer=3, n_head=2,
                               n_positions=64, bos_token_id=1, eos_token_id=2, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
    yield "gpt_neox", hf.GPTNeoXConfig(**{k: v for k, v in common.items() if k != "num_key_value_heads"}, rotary_pct=0.5)
    yield "gemma2", hf.Gemma2Config(**common, head_dim=8, sliding_window=4,
                                   final_logit_softcapping=3., attn_logit_softcapping=5.)
    yield "mixtral", hf.MixtralConfig(**common, num_local_experts=4, num_experts_per_tok=2)
    yield "qwen3_5_text", hf.Qwen3_5TextConfig(
        **common, head_dim=8, linear_conv_kernel_dim=2, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=1, linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]},
    )

    yield "custom", None


def blocks_and_head(model, name):
    if name == "custom":
        return model.residual_units, model.readout.weight
    if name == "gpt2":
        return model.transformer.h, model.lm_head.weight
    if name == "gpt_neox":
        return model.gpt_neox.layers, model.get_output_embeddings().weight
    return model.model.layers, model.lm_head.weight


def run_with_replacement(model, blocks, ids, replacement):
    def hook(module, inputs, output):
        return replacement if torch.is_tensor(output) else (replacement, *output[1:])
    handle = blocks[0].register_forward_hook(hook)
    try:
        with torch.no_grad():
            return model(ids, use_cache=False).logits
    finally:
        handle.remove()


def oracle(model, name):
    blocks, head = blocks_and_head(model, name)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11]])
    captured = {}
    handles = []
    def hook(layer):
        def save(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            if layer == 0:
                h.requires_grad_(True)
            captured[layer] = h
        return save
    try:
        for layer, block in enumerate(blocks):
            handles.append(block.register_forward_hook(hook(layer)))
        output = model(ids, use_cache=False)
        tensors = {"input_ids": ids, "logits": output.logits.detach()}
        for layer, state in captured.items():
            tensors[f"residual.{layer}"] = state.detach()
        # Unbatched, per-output reverse mode, independent of upstream's batched
        # cotangent implementation. The finite difference below is independent
        # of autograd and probes the same source-mean/target-sum reduction.
        valid = slice(1, ids.shape[1] - 1)
        for layer in [0, 1]:
            rows = []
            for dimension in range(16):
                gradient, = torch.autograd.grad(captured[2][0, valid, dimension].sum(), captured[layer], retain_graph=True)
                rows.append(gradient[0, valid].mean(0))
            tensors[f"jacobians.{layer}"] = torch.stack(rows).detach()
    finally:
        for handle in handles:
            handle.remove()
    source = tensors["residual.0"]
    direction = torch.sin(torch.arange(16).float() + 0.3)
    delta = torch.zeros_like(source)
    delta[:, 1:-1] = direction
    ends = []
    for sign in [1., -1.]:
        final = []
        def save_final(module, inputs, output):
            final.append(output if torch.is_tensor(output) else output[0])
        handle = blocks[-1].register_forward_hook(save_final)
        try:
            run_with_replacement(model, blocks, ids, source + sign * 0.0001 * delta)
            ends.append(final[0][:, 1:-1].mean(1))
        finally:
            handle.remove()
    tensors["direction"] = direction
    tensors["finite_difference"] = ((ends[0] - ends[1]) / 0.0002).squeeze(0)
    tensors["head_rows"] = head[[7, 8]].detach()
    basis = tensors["head_rows"].double() @ tensors["jacobians.0"].double()
    dual = torch.linalg.pinv(basis)
    hidden = source[:, -1:].double()
    coordinates = hidden @ dual
    swap = hidden + (coordinates[..., [1, 0]] - coordinates) @ basis
    edited = source.clone()
    edited[:, -1:] = swap.float()
    tensors["swapped"] = edited
    tensors["swapped_logits"] = run_with_replacement(model, blocks, ids, edited)
    return {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--families", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(2)
    families = []
    for name, config in models():
        if args.families and name not in args.families:
            continue
        torch.manual_seed(17)
        if name == "custom":
            from backend_test_factory import ResearchDecoder
            model = ResearchDecoder().eval()
        else:
            model = hf.AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
        model.requires_grad_(False)
        directory = args.directory / name
        directory.mkdir(parents=True, exist_ok=True)
        # Deliberately sharded: the runtime must own loading and tied weights.
        if name == "custom":
            from safetensors.torch import save_model
            save_model(model, str(directory / "model.safetensors"))
        else:
            model.save_pretrained(directory, max_shard_size="20KB")
        tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
        tokenizer.pre_tokenizer = Whitespace()
        fast = hf.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0")
        fast.save_pretrained(directory)
        # A standalone synthetic tokenizer prevents AutoTokenizer from inferring
        # a model-family BPE implementation from this random model's config.
        fast.save_pretrained(args.directory / "tokenizer")
        save_file(oracle(model, name), str(directory / "oracle.safetensors"))
        print(f"oracle {name}: residuals, logits, Jacobians, finite differences, J-space swap", flush=True)
        families.append(name)
    (args.directory / "families.json").write_text(json.dumps({"families": families, "torch": torch.__version__, "transformers": hf.__version__}, indent=2) + "\n")


if __name__ == "__main__":
    main()
