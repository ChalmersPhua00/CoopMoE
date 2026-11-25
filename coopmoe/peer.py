"""This box's side of the link to the other box: a miss the peer holds is copied from the
peer's cache (fetch) and run here.

PeerLink: expert copies (fetch: one connection per loader thread, the expert's bytes back),
residency-bitmap pushes, and the busy-peer estimate (is a copy from the peer currently slower
than a read from this box's drive?). Expert weights and bitmaps cross the link. compute_async
(activations out, the peer's partial sums back) is the earlier design and is no longer called.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

from coopmoe.transport import TcpClient


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
        self.svc_per_expert: Optional[float] = None  # EWMA of the time to copy one expert from the peer
        self._skips = 0
        self._tls = threading.local()            # one fetch connection per loader thread
        self._fetch_clients: list = []

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
        nvme_s = self.box.nvme_load_s  # measured on this box; unknown until its first drive read
        if not self.busy_fallback or self.svc_per_expert is None or nvme_s is None or self.svc_per_expert <= nvme_s:
            return True
        self._skips += 1
        if self._skips >= self.probe_every:
            self._skips = 0
            return True
        self.box.counters.add("link", "busy_skips")
        return False

    # -- copying an expert from the peer ----------------------------------------------
    def fetch(self, key: int) -> Optional[torch.Tensor]:
        """The record of `key` from the peer's cache (host uint8), or None if it no longer
        holds it. Called from the loader threads, each on its own connection."""
        client = getattr(self._tls, "client", None)
        if client is None:
            client = self._tls.client = TcpClient(self.layer.addr, self.box.cfg.cluster.connect_timeout)
            with self._lock:
                self._fetch_clients.append(client)
        rh, rt = client.call({"op": "fetch", "key": int(key)})
        if not rt:
            return None
        c = self.box.counters
        c.add("link", "fetch_in", rt[0].numel())
        c.add("link", "fetches")
        return rt[0]

    def note_fetch(self, seconds: float) -> None:
        s = self.svc_per_expert
        self.svc_per_expert = seconds if s is None else 0.9 * s + 0.1 * seconds

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
        for client in [self.layer, *self._fetch_clients]:
            try:
                client.close()
            except Exception:
                pass

