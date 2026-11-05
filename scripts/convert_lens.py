"""Convert Anthropic JacobianLens.save() checkpoints; pip install torch safetensors."""
import argparse
import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    width = int(checkpoint["d_model"])
    tensors = {}
    for layer, jacobian in checkpoint["J"].items():
        if not isinstance(layer, int) or layer < 0:
            raise ValueError(f"invalid layer {layer!r}")
        if tuple(jacobian.shape) != (width, width) or width <= 0:
            raise ValueError(f"invalid shape for layer {layer}: {jacobian.shape}")
        if not jacobian.is_floating_point() or not torch.isfinite(jacobian).all():
            raise ValueError(f"invalid values for layer {layer}")
        # Preserve J[out, in]. Rust packs its transpose exactly once at load.
        tensors[f"jacobians.{layer}"] = jacobian.detach().float().contiguous().clone()
    if not tensors:
        raise ValueError("empty Jacobian checkpoint")
    save_file(tensors, args.destination, metadata={
        "format": "jacobian-lens-v1", "orientation": "out,in",
        "hook": "block-output-pre-final-norm", "d_model": str(width),
        "n_prompts": str(checkpoint["n_prompts"]),
    })


if __name__ == "__main__":
    main()
