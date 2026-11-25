"""Model shape (read from the checkpoint) and run knobs (read from a JSON file).

Run configs live in configs/*.json. Overrides are section.field=value strings, e.g.
cache.gb=60 or link.enabled=false (a list value is JSON). driver.py uses them to point the
engines at their store and addresses; a cell's overrides go to both engines with the reset.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    """Qwen3-MoE shape. Only all-sparse models (every layer is MoE) are supported."""

    vocab_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool = True
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf(cls, hf: dict) -> "ModelConfig":
        if hf.get("mlp_only_layers"):
            raise ValueError("checkpoints with dense (mlp_only) layers are not supported")
        if hf.get("decoder_sparse_step", 1) != 1:
            raise ValueError("decoder_sparse_step != 1 is not supported")
        head_dim = hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"]
        return cls(
            vocab_size=hf["vocab_size"],
            hidden_size=hf["hidden_size"],
            moe_intermediate_size=hf["moe_intermediate_size"],
            num_hidden_layers=hf["num_hidden_layers"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf["num_key_value_heads"],
            head_dim=head_dim,
            num_experts=hf["num_experts"],
            num_experts_per_tok=hf["num_experts_per_tok"],
            norm_topk_prob=hf.get("norm_topk_prob", True),
            rms_norm_eps=hf.get("rms_norm_eps", 1e-6),
            rope_theta=hf.get("rope_theta", 1_000_000.0),
            tie_word_embeddings=hf.get("tie_word_embeddings", False),
        )

    @property
    def n_keys(self) -> int:
        """Number of (layer, expert) pairs: the length of a residency bitmap or a signature."""
        return self.num_hidden_layers * self.num_experts


@dataclass
class CacheConfig:
    """One plain LRU expert cache per box, plus one rule: duplicate-first eviction."""

    gb: float = 1.0                  # expert-cache budget per box, GiB
    dup_first: bool = True           # evict experts the peer also holds before exclusive ones
    dup_protect_steps: int = 0       # >0: spare duplicates this box used in its last N decode steps
                                     # (0 = the plain rule: any duplicate goes first)
    remote_refresh: bool = True      # handing the peer a copy refreshes the expert's LRU position


@dataclass
class LinkConfig:
    enabled: bool = True             # False = two independent boxes: no peer path, no bitmaps
    transport: str = "rdma"          # how a copied expert's bytes cross: rdma (one-sided READ through
                                     # NIXL/UCX; the request and reply still use TCP) | tcp (the bytes
                                     # ride in the TCP reply)
    busy_fallback: bool = True       # read own NVMe when the peer is slower than the drive
    probe_every: int = 32            # while the peer counts as busy, still try it every N layers
    nvme_ms_per_expert: Optional[float] = None  # reported only (default: expert size / 3.9 GB/s); the
                                                # busy test compares measured copy and drive times


@dataclass
class EngineConfig:
    max_batch: int = 4               # requests sharing one forward on a box (HF continuous batching)
    max_sessions: int = 4            # requests in flight on a box (each box's cap; KV is sized for it)
    max_seq: int = 8192              # prompt + generated tokens per request
    n_transient: int = 16            # buffers for experts read without being admitted (prefill)


@dataclass
class ClusterConfig:
    addrs: list = field(default_factory=lambda: ["127.0.0.1:29600", "127.0.0.1:29601"])
    connect_timeout: float = 600.0


@dataclass
class ReaderConfig:
    threads: int = 8
    direct_io: bool = True  # O_DIRECT on Linux: on UMA the page cache eats the same 128 GB


@dataclass
class BenchConfig:
    warmup_sessions: int = 4         # counters reset after this many requests finish
    sample_s: float = 1.0            # cache-overlap sampling period
    max_new_tokens: int = 64         # tokens generated per request, always (end of turn is ignored,
                                     # so every cell does the same work)


@dataclass
class RunConfig:
    store_dir: Optional[str] = None  # set per box by driver.py / coopmoe run
    device: str = "auto"             # auto | cuda | cpu
    cache: CacheConfig = field(default_factory=CacheConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def copy(self) -> "RunConfig":
        return copy.deepcopy(self)


def _build(cls, data: dict):
    kwargs = {}
    names = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key.startswith("_"):
            continue  # comments
        if key not in names:
            raise KeyError(f"unknown config field {cls.__name__}.{key}")
        sub = names[key].type
        sub_cls = globals().get(sub) if isinstance(sub, str) else sub
        if dataclasses.is_dataclass(sub_cls) and isinstance(value, dict):
            value = _build(sub_cls, value)
        kwargs[key] = value
    return cls(**kwargs)


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(cfg: RunConfig, overrides: Optional[list[str]]) -> RunConfig:
    for item in overrides or []:
        path, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like section.field=value, got {item!r}")
        obj = cfg
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise KeyError(f"unknown config field {path}")
        setattr(obj, parts[-1], _parse_value(raw))
    return cfg


def load_run_config(path: Optional[str], overrides: Optional[list[str]] = None) -> RunConfig:
    cfg = RunConfig()
    if path:
        with open(path) as f:
            cfg = _build(RunConfig, json.load(f))
    return apply_overrides(cfg, overrides or [])

