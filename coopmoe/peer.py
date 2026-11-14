"""This box's side of the link to the other box.

PeerLink   per-layer compute requests (one message per MoE layer, activations out, partial
           sums back), residency-bitmap pushes, and the busy-peer estimate.
Admitter   the admission rule: an expert this box sent to the peer k times within its last W
           decode steps is fetched (bytes over the link, when no layer request is in flight)
           and admitted to this box's cache, creating a deliberate duplicate.

Two connections: layer requests and bitmap pushes share one (both come from the engine
thread, in order); admission fetches use the other so a multi-MB transfer never sits in
front of a layer request.
"""
from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

from coopmoe.cache import Acquire


class PeerLink:
    def __init__(self, box, layer_client, bg_client, epoch: int):
        self.box, self.layer, self.bg, self.epoch = box, layer_client, bg_client, epoch
        cfg = box.cfg.link
        self.busy_fallback, self.probe_every = cfg.busy_fallback, max(1, cfg.probe_every)
        self.nvme_s = box.nvme_s_per_expert()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="peer")
        self.bits: Optional[np.ndarray] = None   # peer residency, from its last push
        self.version = -1
        self.pushed_version = -1
        self._lock = threading.Lock()
        self.idle = threading.Event()
        self.idle.set()
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
        self.idle.clear()
        return self.pool.submit(self._compute, fields, tensors, needed)

    def _compute(self, fields: dict, tensors: list, needed: torch.Tensor):
        try:
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
        finally:
            self.idle.set()

    def fetch(self, keys: list[int]) -> list[Optional[torch.Tensor]]:
        rh, rt = self.bg.call({"op": "fetch", "keys": keys})
        it = iter(rt)
        out = [next(it) if have else None for have in rh["have"]]
        self.box.counters.add("link", "fetch_in", sum(t.numel() for t in rt))
        return out

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for c in (self.layer, self.bg):
            try:
                c.close()
            except Exception:
                pass


class Admitter:
    """Admission rule: fetch-and-admit a peer-held expert once this box has sent it to the peer
    k times within its last W decode steps. Runs on its own thread, between layer requests."""

    def __init__(self, box):
        self.box = box
        self.k, self.window = box.cfg.cache.admit_k, box.cfg.cache.admit_window
        self.hist: dict[int, deque] = {}
        self.queue: deque = deque()
        self.pending: set[int] = set()
        self.cv = threading.Condition()
        self.busy = False
        self.stop = False
        self.thread = threading.Thread(target=self._run, name="admitter", daemon=True)
        self.thread.start()

    def note(self, keys: list[int], step: int) -> None:
        """Called by the engine for every expert it sends to the peer during decode."""
        if self.k <= 0:
            return
        new = []
        for key in keys:
            dq = self.hist.get(key)
            if dq is None:
                dq = self.hist[key] = deque()
            dq.append(step)
            while dq and dq[0] <= step - self.window:
                dq.popleft()
            if len(dq) >= self.k and key not in self.pending:
                dq.clear()
                new.append(key)
        if new:
            with self.cv:
                self.pending.update(new)
                self.queue.extend(new)
                self.cv.notify()

    def _run(self) -> None:
        box = self.box
        while True:
            with self.cv:
                while not self.queue and not self.stop:
                    self.cv.wait()
                if self.stop:
                    return
                batch = [self.queue.popleft() for _ in range(min(8, len(self.queue)))]
                self.busy = True
            try:
                peer = box.peer
                if peer is None:
                    continue
                while not peer.idle.wait(timeout=0.05):  # "when the link is idle"
                    if self.stop:
                        return
                todo = [k for k in batch if not box.cache.has(k)]
                blobs = peer.fetch(todo) if todo else []
                for key, blob in zip(todo, blobs):
                    if blob is None:
                        box.counters.add("admit", "peer_miss")
                        continue
                    slot, st = box.cache.acquire(key, reserve=True)
                    if st is Acquire.RESERVED:
                        try:
                            box.loader.copy_in(box.cache.slab[slot], blob)
                        except BaseException:
                            box.cache.abort(key, slot)
                            raise
                        box.cache.commit(key, slot)
                        box.cache.release(key)
                        box.counters.add("admit", "admitted")
                    elif st is Acquire.HIT:
                        box.cache.release(key)
                        box.counters.add("admit", "already_here")
                    else:
                        box.counters.add("admit", "no_slot")
            except Exception as exc:
                box.log(f"[admit] failed: {exc}")
            finally:
                with self.cv:
                    self.pending.difference_update(batch)
                    self.busy = False
                    self.cv.notify_all()

    def drain(self) -> None:
        """Drop queued admissions and wait for the one in progress (before a reset)."""
        with self.cv:
            for k in self.queue:
                self.pending.discard(k)
            self.queue.clear()
            while self.busy:
                self.cv.wait()
        self.hist.clear()

    def close(self) -> None:
        with self.cv:
            self.stop = True
            self.cv.notify_all()
        self.thread.join(timeout=5)
