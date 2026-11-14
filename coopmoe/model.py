"""The model: Hugging Face's Qwen3MoeForCausalLM, with each layer's routed experts replaced.

HF runs everything else unchanged: embeddings, attention, norms, the router (`mlp.gate`) and
the LM head. Only `mlp.experts` is swapped for CoopExperts, which hands the router's top-k
choice to the MoeExecutor (own cache, peer, own drive). The non-expert weights come from the
store's dense.safetensors, in HF names; the experts never exist as HF parameters.
"""
from __future__ import annotations

import os

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from coopmoe.store.layout import DENSE_NAME


class CoopExperts(nn.Module):
    """Stands in for Qwen3MoeExperts: same call, (hidden, top_k_index, top_k_weights) -> output."""

    def __init__(self, layer: int, moe):
        super().__init__()
        self.layer, self.moe = layer, moe

    def forward(self, x: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        return self.moe(self.layer, x, top_k_index, top_k_weights)


def hf_config(store_dir: str) -> Qwen3MoeConfig:
    """The checkpoint's config, minus its FP8 quantization: the store's dense weights are plain."""
    cfg = Qwen3MoeConfig.from_json_file(os.path.join(store_dir, "hf_config.json"))
    cfg.quantization_config = None
    if getattr(cfg, "mlp_only_layers", None) or getattr(cfg, "decoder_sparse_step", 1) != 1:
        raise ValueError("only all-sparse checkpoints (every layer MoE) are supported")
    return cfg


def build(store_dir: str, moe, device: torch.device, dtype: torch.dtype) -> Qwen3MoeForCausalLM:
    cfg = hf_config(store_dir)
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(cfg)
    for i, layer in enumerate(model.model.layers):
        layer.mlp.experts = CoopExperts(i, moe)
    # The loaded tensors become the parameters (assign): one copy of the dense weights, not two.
    dense = load_file(os.path.join(store_dir, DENSE_NAME), device=str(device))
    model.load_state_dict({k: v.to(dtype) for k, v in dense.items()}, strict=True, assign=True)
    # Rotary frequencies are a buffer computed at init (not in the state dict): recompute it on the device.
    model.model.rotary_emb = type(model.model.rotary_emb)(cfg).to(device)
    left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if left:
        raise RuntimeError(f"not loaded from {DENSE_NAME}: {left[:5]}")
    return model.eval()
