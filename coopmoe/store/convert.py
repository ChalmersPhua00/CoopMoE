"""HF Qwen3-MoE checkpoint (safetensors, per-expert tensor names) -> expert store.

Handles BF16 checkpoints and Qwen's block-FP8 checkpoints (`*.weight` in float8_e4m3fn plus
`*.weight_scale_inv`, quantization_config.weight_block_size = [128, 128]).

  experts: kept in the checkpoint's format by default (FP8 stays FP8 with its block scales),
           one aligned record per (layer, expert)
  dense:   dequantized to bfloat16 (about 16 GB for Qwen3-235B-A22B), so attention runs on
           plain matrices; only experts stay quantized in memory

Shards are memory-mapped through safetensors and read tensor by tensor, so memory stays
small. On the hackathon cluster it runs once, in its own job (stage.py), and writes
the store to the team's shared NFS volume; each experiment job then copies the finished store
to its box's local disk.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import time
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from coopmoe.config import ModelConfig
from coopmoe.store.layout import DENSE_NAME, DTYPES, FP8, StoreIndex, dequant_block, pack_expert

EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight(_scale_inv)?$")
FUSED_RE = re.compile(r"model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)")
COPY_FILES = ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "vocab.json",
              "merges.txt", "chat_template.jinja"]


def _weight_map(hf_dir: str) -> dict[str, str]:
    idx = os.path.join(hf_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return json.load(f)["weight_map"]
    with safe_open(os.path.join(hf_dir, "model.safetensors"), framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def fp8_block(hf_cfg: dict) -> Optional[int]:
    q = hf_cfg.get("quantization_config") or {}
    if q.get("quant_method") != "fp8":
        return None
    b = q.get("weight_block_size")
    if not b or len(b) != 2 or b[0] != b[1]:
        raise NotImplementedError(f"only square FP8 blocks are supported, got {b}")
    return int(b[0])


def convert_hf(hf_dir: str, out_dir: str, dtype: Optional[str] = None, dense_dtype: str = "bfloat16",
               log=print) -> StoreIndex:
    with open(os.path.join(hf_dir, "config.json")) as f:
        hf_cfg = json.load(f)
    model = ModelConfig.from_hf(hf_cfg)
    block = fp8_block(hf_cfg)
    if dtype is None:
        ckpt = hf_cfg.get("torch_dtype") or hf_cfg.get("dtype")
        dtype = FP8 if block else (ckpt if ckpt in DTYPES else "bfloat16")
    if dtype == FP8 and not block:
        raise ValueError("an FP8 store needs an FP8 checkpoint (quantization_config.weight_block_size)")
    index = StoreIndex.create(model, dtype, dense_dtype, block)
    target, dense_target = DTYPES[dtype], DTYPES[dense_dtype]

    wmap = _weight_map(hf_dir)
    for name in wmap:
        if FUSED_RE.match(name):
            raise NotImplementedError(f"{name}: fused expert tensors are not handled; use the "
                                      "original per-expert checkpoint from the Hub")
    handles: dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        shard = wmap[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(hf_dir, shard), framework="pt")
        return handles[shard].get_tensor(name)

    def weight(name: str, out: torch.dtype) -> torch.Tensor:
        """A weight in `out`, dequantizing an FP8 one when the store wants a float type."""
        t = get(name)
        sname = name + "_scale_inv"
        if t.dtype == torch.float8_e4m3fn and out != torch.float8_e4m3fn:
            return dequant_block(t, get(sname).float(), block, torch.float32).to(out)
        return t.to(out) if t.is_floating_point() else t

    os.makedirs(os.path.join(out_dir, "experts"), exist_ok=True)
    m = model
    written = np.zeros((m.num_hidden_layers, m.num_experts), dtype=bool)
    for layer in range(m.num_hidden_layers):
        with open(index.layer_path(out_dir, layer), "w+b") as f:
            f.truncate(m.num_experts * index.stride)
            for e in range(m.num_experts):
                p = f"model.layers.{layer}.mlp.experts.{e}."
                names = [p + f"{w}_proj.weight" for w in ("gate", "up", "down")]
                if dtype == FP8:
                    ws = [get(n) for n in names]
                    ss = [get(n + "_scale_inv") for n in names]
                    rec = pack_expert(index, *ws, *ss)
                else:
                    rec = pack_expert(index, *(weight(n, target) for n in names))
                f.seek(index.offset(e))
                f.write(memoryview(rec.numpy()))
                written[layer, e] = True
        if layer % 8 == 0 or layer == m.num_hidden_layers - 1:
            log(f"[convert] layer {layer + 1}/{m.num_hidden_layers}")
    if not written.all():
        raise RuntimeError("some experts were not written")

    dense = {}
    for name in sorted(wmap):
        if EXPERT_RE.match(name) or name.endswith("_scale_inv"):
            continue
        dense[name] = weight(name, dense_target).contiguous()
    save_file(dense, os.path.join(out_dir, DENSE_NAME))
    index.save(out_dir)
    for name in COPY_FILES:
        src = os.path.join(hf_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, name))
    shutil.copy(os.path.join(hf_dir, "config.json"), os.path.join(out_dir, "hf_config.json"))
    log(f"[convert] done: {m.n_keys} experts x {index.expert_bytes / 2**20:.2f} MiB ({dtype}), "
        f"dense {sum(t.numel() * t.element_size() for t in dense.values()) / 2**30:.2f} GiB ({dense_dtype})")
    return index


def stage(model: str, store_dir: str, hf_dir: Optional[str] = None, log=print,
          wait_s: float = 24 * 3600) -> None:
    """The model's store at store_dir. `model` is a Hub repo (downloaded to hf_dir) or a
    local checkpoint directory. Does nothing when the store is already there.

    Safe on shared storage: one caller builds it (holding <store>.lock) into <store>.partial
    and renames it into place when complete; any other caller waits for that rename. A store
    without index.json is never visible under store_dir, so a crash leaves only .partial."""
    index = os.path.join(store_dir, "index.json")
    if os.path.exists(index):
        log(f"[stage] store already present: {store_dir}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(store_dir)), exist_ok=True)
    lock = store_dir.rstrip("/") + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        with open(lock) as f:
            holder = f.read().strip()
        log(f"[stage] another stage holds {lock} ({holder}); waiting for {store_dir}. "
            "Delete the lock if that stage is dead.")
        t0 = time.time()
        while not os.path.exists(index):
            if time.time() - t0 > wait_s:
                raise SystemExit(f"[stage] {store_dir} did not appear within {wait_s:.0f} s")
            time.sleep(30)
        log(f"[stage] store ready: {store_dir}")
        return
    with os.fdopen(fd, "w") as f:
        f.write(f"{socket.gethostname()} pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        src = model
        if not os.path.isdir(model):
            from huggingface_hub import snapshot_download

            log(f"[stage] downloading {model} -> {hf_dir}")
            src = snapshot_download(model, local_dir=hf_dir, allow_patterns=["*.safetensors", "*.json", "*.txt",
                                                                              "tokenizer*", "*.jinja"])
        partial = store_dir.rstrip("/") + ".partial"
        shutil.rmtree(partial, ignore_errors=True)
        log(f"[stage] repacking {src} -> {partial}")
        convert_hf(src, partial, log=log)
        os.replace(partial, store_dir)
        log(f"[stage] store ready: {store_dir}")
        if src != model:  # the download is only the store's source
            shutil.rmtree(src)
            log(f"[stage] deleted the download: {src}")
    finally:
        os.remove(lock)
