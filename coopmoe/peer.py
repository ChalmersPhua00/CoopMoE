"""This box's side of the link to the other box.

PeerLink: per-layer compute requests (one message per MoE layer, activations out, partial sums
back), residency-bitmap pushes, and the busy-peer estimate. Only activations and bitmaps cross
the link, never expert weights: an expert's bytes come from a box's own NVMe only.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch


class PeerLink:
    def __init__(self, box, layer_client, epoch: int):
        self.box, self.layer, self.epoch = box, layer_client, epoch
        cfg = box.cfg.link
        self.busy_fallback, self.probe_every = cfg.busy_fallback, max(1, cfg.probe_every)
        self.nvme_s = box.nvme_s_per_expert()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="peer")
        self.bits: Optional[np.ndarray] = None   # peer residency, from its last push
        self.version = -1
        self.pushed_version = -1
        self._lock = threading.Lock()
        self.svc_per_expert: Optional[float] = None  # EWMA of the peer's service time
        self._skips = 0

    # -- residency ------------------------------------------------------------------
    def holds(self, key: int) -> bool:
        bits = self.bits
        return bits is not None and bool(bits[key])

    def receive_bitmap(self, epoch: int, version: int, packed: torch.Tensor) -> None:
        if epoch != self.epoch:
            return  # left over from another benchmark cell
        with self._lock:
            if version <= self.version:
                return
            n = self.box.m.n_keys
            bits = np.unpackbits(packed.numpy())[:n].astype(bool)
            self.bits, self.version = bits, version
        self.box.cache.set_peer(bits)

    def push_bitmap(self) -> None:
        """At most once per decode step, and only when this box's residency changed."""
        version, packed = self.box.cache.bitmap()
        if version == self.pushed_version:
            return
        self.pushed_version = version
        self.box.counters.add("link", "bitmap_out", packed.nbytes)
        self.pool.submit(self._call, "bitmap", {"epoch": self.epoch, "version": version},
                         [torch.from_numpy(packed)])

    def _call(self, op: str, fields: dict, tensors: list):
        try:
            return self.layer.call(dict(fields, op=op), tensors)
        except Exception as exc:  # a lost bitmap push only costs staleness
            self.box.log(f"[peer] {op} failed: {exc}")
            return None

    # -- busy peer --------------------------------------------------------------------
    def usable(self) -> bool:
        """False while the peer's measured service time per expert exceeds an own-drive read;
        every probe_every-th layer still goes to the peer to refresh the estimate."""
        if self.bits is None:
            return False
        if not self.busy_fallback or self.svc_per_expert is None or self.svc_per_expert <= self.nvme_s:
            return True
        self._skips += 1
        if self._skips >= self.probe_every:
            self._skips = 0
            return True
        self.box.counters.add("link", "busy_skips")
        return False

    # -- per-layer compute ------------------------------------------------------------
    def compute_async(self, layer: int, items: list, x: torch.Tensor) -> Future:
        """items: (expert, key, token_idx, weight). One message for the whole layer."""
        needed = torch.unique(torch.cat([ti for _, _, ti, _ in items]))
        remap = torch.cat([torch.searchsorted(needed, ti).to(torch.int32) for _, _, ti, _ in items])
        fields = {"op": "compute", "layer": layer, "experts": [e for e, _, _, _ in items],
                  "counts": [int(ti.numel()) for _, _, ti, _ in items]}
        tensors = [x[needed], remap, torch.cat([w for _, _, _, w in items]).float()]
        return self.pool.submit(self._compute, fields, tensors, needed)

    def _compute(self, fields: dict, tensors: list, needed: torch.Tensor):
        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        rh, rt = self.layer.call(fields, tensors)
        n_run = len(fields["experts"]) - len(rh["miss"])
        if n_run:
            per = rh["svc"] / n_run
            self.svc_per_expert = per if self.svc_per_expert is None else 0.8 * self.svc_per_expert + 0.2 * per
        c = self.box.counters
        c.add("link", "act_out", nbytes)
        c.add("link", "partial_in", rt[0].numel() * rt[0].element_size())
        c.add("link", "requests")
        return needed, rt[0], rh["miss"]

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        try:
            self.layer.close()
        except Exception:
            pass

