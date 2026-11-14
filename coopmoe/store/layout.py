"""Expert store format: one aligned record per (layer, expert), as stage repacks it.

    <store>/index.json             StoreIndex (model shape, dtypes, record layout)
    <store>/dense.safetensors      everything that is not a routed expert (attention, norms,
                                   router gates, embeddings, lm_head), already dequantized to
                                   index.dense_dtype, in HF tensor names
    <store>/experts/layer_LLL.bin  num_experts records, record e at offset e * stride
    <store>/tokenizer.json ...     copied from the HF checkpoint when present

One record holds, in this order, each section 64-byte aligned:
    w13 = cat(gate_proj, up_proj)   [2I, H]  in index.dtype
    w2  = down_proj                 [H, I]   in index.dtype
    s13, s2                         float32 block scales, FP8 stores only
FP8 weights use 128x128 block scales as in Qwen's FP8 checkpoints: W = Wq * s[i//b, j//b].
stride is the record size rounded up to 4 KiB, so every record can be read with O_DIRECT
straight into an aligned cache slot. The cache holds records as raw bytes (FP8 stays FP8);
expert_weights() turns a record into compute-dtype matrices when it runs.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from coopmoe.config import ModelConfig

ALIGN = 4096
SECTION_ALIGN = 64
INDEX_NAME = "index.json"
DENSE_NAME = "dense.safetensors"
FP8 = "float8_e4m3fn"

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    FP8: torch.float8_e4m3fn,
}


def dtype_name(dtype: torch.dtype) -> str:
    for name, dt in DTYPES.items():
        if dt == dtype:
            return name
    raise ValueError(f"unsupported dtype {dtype}")


def round_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def _elem(dtype: str) -> int:
    return torch.empty((), dtype=DTYPES[dtype]).element_size()


@dataclass
class StoreIndex:
    model: ModelConfig
    dtype: str                    # expert weights on disk and in the cache
    dense_dtype: str              # dense.safetensors
    block: Optional[int]          # FP8 block size, None for unquantized stores
    sections: dict                # name -> [byte offset, rows, cols, dtype]
    expert_bytes: int             # meaningful bytes per record
    stride: int                   # record pitch on disk and slot size in the cache
    format: int = 2

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]

    @property
    def quantized(self) -> bool:
        return self.block is not None

    @classmethod
    def create(cls, model: ModelConfig, dtype: str, dense_dtype: str = "bfloat16",
               block: Optional[int] = None) -> "StoreIndex":
        H, I = model.hidden_size, model.moe_intermediate_size
        shapes = [("w13", 2 * I, H, dtype), ("w2", H, I, dtype)]
        if dtype == FP8:
            if not block:
                raise ValueError("an FP8 store needs a block size")
            if I % block:
                raise ValueError(f"moe_intermediate_size {I} is not a multiple of the FP8 block {block}")
            shapes += [("s13", math.ceil(2 * I / block), math.ceil(H / block), "float32"),
                       ("s2", math.ceil(H / block), math.ceil(I / block), "float32")]
        else:
            block = None
        sections, off = {}, 0
        for name, r, c, dt in shapes:
            off = round_up(off, SECTION_ALIGN)
            sections[name] = [off, r, c, dt]
            off += r * c * _elem(dt)
        return cls(model=model, dtype=dtype, dense_dtype=dense_dtype, block=block,
                   sections=sections, expert_bytes=off, stride=round_up(off))

    @classmethod
    def load(cls, store_dir: str) -> "StoreIndex":
        with open(os.path.join(store_dir, INDEX_NAME)) as f:
            d = json.load(f)
        if d.get("format") != 2:
            raise ValueError(f"{store_dir}: store format {d.get('format')} is not supported; reconvert")
        d["model"] = ModelConfig(**d["model"])
        return cls(**d)

    def save(self, store_dir: str) -> None:
        with open(os.path.join(store_dir, INDEX_NAME), "w") as f:
            json.dump(asdict(self), f, indent=2)

    def layer_path(self, store_dir: str, layer: int) -> str:
        return os.path.join(store_dir, "experts", f"layer_{layer:03d}.bin")

    def offset(self, expert: int) -> int:
        return expert * self.stride

    def key(self, layer: int, expert: int) -> int:
        return layer * self.model.num_experts + expert

    def unkey(self, key: int) -> tuple[int, int]:
        return divmod(key, self.model.num_experts)


def section(buf: torch.Tensor, index: StoreIndex, name: str) -> torch.Tensor:
    """Typed view of one section of a uint8 record, without copying."""
    off, r, c, dt = index.sections[name]
    n = r * c * _elem(dt)
    return buf[off : off + n].view(DTYPES[dt]).view(r, c)


def pack_expert(index: StoreIndex, gate, up, down, s_gate=None, s_up=None, s_down=None) -> torch.Tensor:
    """HF per-expert tensors (already in index.dtype) -> one uint8 record of index.expert_bytes."""
    out = torch.zeros(index.expert_bytes, dtype=torch.uint8)
    parts = {"w13": torch.cat([gate, up], dim=0), "w2": down}
    if index.quantized:
        parts["s13"] = torch.cat([s_gate, s_up], dim=0).float()
        parts["s2"] = s_down.float()
    for name, t in parts.items():
        dst = section(out, index, name)
        if tuple(dst.shape) != tuple(t.shape):
            raise ValueError(f"{name}: expected shape {tuple(dst.shape)}, got {tuple(t.shape)}")
        dst.copy_(t.to(dst.dtype))
    return out


def dequant_block(wq: torch.Tensor, s: torch.Tensor, block: int, dtype: torch.dtype) -> torch.Tensor:
    """W = Wq * s[i // block, j // block], returned in `dtype`."""
    R, C = wq.shape
    w = wq.to(dtype)
    if R % block == 0 and C % block == 0:
        w.view(R // block, block, C // block, block).mul_(s.to(dtype)[:, None, :, None])
        return w
    full = s.repeat_interleave(block, 0)[:R].repeat_interleave(block, 1)[:, :C]
    return w.mul_(full.to(dtype))


def expert_weights(buf: torch.Tensor, index: StoreIndex, dtype: torch.dtype):
    """(w13 [2I, H], w2 [H, I]) in the compute dtype from a uint8 record (slot or buffer).

    Unquantized stores return views (no copy when the dtypes match). FP8 records are
    dequantized on every call: the fallback for as long as no FP8 expert kernel runs on GB10,
    and it keeps FP8 bytes in the cache.
    """
    w13, w2 = section(buf, index, "w13"), section(buf, index, "w2")
    if index.quantized:
        b = index.block
        return (dequant_block(w13, section(buf, index, "s13"), b, dtype),
                dequant_block(w2, section(buf, index, "s2"), b, dtype))
    if w13.dtype != dtype:
        return w13.to(dtype), w2.to(dtype)
    return w13, w2
