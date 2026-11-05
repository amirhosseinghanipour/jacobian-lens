"""Unregistered PyTorch model fixture for testing the external factory contract."""
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn


class ContextBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, hidden, context):
        prefix = hidden.cumsum(1) / torch.arange(1, hidden.shape[1] + 1, device=hidden.device)[None, :, None]
        return hidden + 0.1 * torch.tanh(self.linear(prefix) + context), context


class ResearchDecoder(nn.Module):
    """Blocks require original-input context, which a bare suffix cannot recover."""
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(32, 16)
        self.residual_units = nn.ModuleList([ContextBlock() for _ in range(3)])
        self.final_norm = nn.LayerNorm(16)
        self.readout = nn.Linear(16, 32)

    def forward(self, ids, use_cache=False):
        hidden = self.tokens(ids)
        context = 0.1 * hidden.cumsum(1)
        for block in self.residual_units:
            hidden, _ = block(hidden, context)
        return SimpleNamespace(logits=self.unembed(hidden))

    def unembed(self, hidden):
        return 2.0 * torch.tanh(self.readout(self.final_norm(hidden)) / 2.0)


def create(options):
    """Return a LensModel; no Transformers architecture registration is needed."""
    from safetensors.torch import load_model
    from transformers import PreTrainedTokenizerFast
    model = ResearchDecoder().to(options["device"]).eval()
    load_model(model, str(Path(options["model"]) / "model.safetensors"))
    model.requires_grad_(False)

    class Adapter:
        n_layers = 3
        d_model = 16
        layers = model.residual_units
        input_device = model.tokens.weight.device
        tokenizer = PreTrainedTokenizerFast.from_pretrained(options["tokenizer"], local_files_only=True)

        def encode(self, text, *, max_length=512):
            return self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(self.input_device)

        def forward(self, ids):
            return model(ids, use_cache=False)

        def unembed(self, residual):
            return model.unembed(residual)

        def unembedding_rows(self, ids):
            return model.readout.weight[ids]

    return Adapter()
