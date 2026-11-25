"""The MoE layer at one box, and the work it does for its peer.

Per MoE layer, for the box's own batch (MoeExecutor.__call__, called by model.CoopExperts
with HF's routing):
    1. copy the router's top-k ids to the host once
    2. per routed expert, in order of preference:
         own cache  -> run here (pinned so this layer's loads cannot evict it)
         peer holds -> copy its bytes from the peer's cache over the link (unless the link
                       is currently slower than this box's drive), then run here
         neither    -> read own NVMe
       either way the expert is admitted to this box's cache during decode, and goes to a
       transient buffer during prefill
    3. hits run here while the misses load in parallel; an expert the peer no longer holds
       (stale bitmap) is read from own NVMe

For the peer (serve_fetch): hand out the bytes of an expert that is in this box's cache, or
MISS. The peer never reads this box's drive on its behalf. (serve_compute, the earlier design
where the peer ran the expert and sent back partial sums, is no longer called.)
"""
from __future__ import annotations

import time
from concurrent.futures import Future, as_completed
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from coopmoe.cache import Acquire
from coopmoe.stats import LOCAL, NVME, NVME_TMP, PEER, PEER_MISS
from coopmoe.store.layout import expert_weights


def _drain(stream=None) -> None:
    """Wait until the GPU has run everything queued on `stream` (default: the current one).
    A cache slot or transient buffer may only be given back after the kernels reading it ran:
    the drive loader writes the next expert into it on its own stream, and pins only order
    the host, not the GPU."""
    if torch.cuda.is_available():
        (stream or torch.cuda.current_stream()).synchronize()


def expert_ffn(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """SwiGLU expert: down(silu(gate(x)) * up(x)) with gate/up packed as w13."""
    h = x @ w13.t()
    inter = w2.shape[1]
    return (F.silu(h[:, :inter]) * h[:, inter:]) @ w2.t()


def group_by_expert(topw: torch.Tensor, topi_h: torch.Tensor, n_experts: int):
    """-> [(expert, token_idx, weight)] for experts with >= 1 token. topi_h is on the host
    (one device->host copy per layer); index tensors are moved to topw's device once."""
    k = topi_h.shape[1]
    flat = topi_h.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=n_experts)
    nz = torch.nonzero(counts).flatten().tolist()
    sizes = counts[nz].tolist()
    dev = topw.device
    tok = torch.div(order, k, rounding_mode="floor").to(dev)
    w = topw.reshape(-1)[order.to(dev)]
    return list(zip(nz, torch.split(tok, sizes), torch.split(w, sizes)))


class MoeExecutor:
    def __init__(self, box):
        self.box = box
        self.index, self.m = box.index, box.m

    def _apply(self, buf: torch.Tensor, x, ti, w, out) -> None:
        w13, w2 = expert_weights(buf, self.index, x.dtype)
        out.index_add_(0, ti, (expert_ffn(x[ti], w13, w2) * w[:, None]).float())

    def __call__(self, layer: int, x: torch.Tensor, topi: torch.Tensor, topw: torch.Tensor) -> torch.Tensor:
        box, cache = self.box, self.box.cache
        c = box.counters
        decode = box.decode
        phase = "decode" if decode else "prefill"
        t0 = time.perf_counter()
        topi_h = topi.cpu()
        groups = group_by_expert(topw, topi_h, self.m.num_experts)

        peer = box.peer
        use_peer = peer is not None and peer.usable()
        hits, misses = [], []
        for e, ti, w in groups:
            key = self.index.key(layer, e)
            _, st = cache.acquire(key, reserve=False)
            if st is Acquire.HIT:
                hits.append((key, ti, w))
            else:  # copied from the peer's cache if it holds it, else read from own drive
                misses.append((key, ti, w, use_peer and peer.holds(key)))
        c.add("experts", f"{phase}.{LOCAL}", len(hits))
        c.add("pairs", f"{phase}.{LOCAL}", sum(int(ti.numel()) for _, ti, _ in hits))
        t1 = time.perf_counter()

        out = torch.zeros(x.shape[0], self.m.hidden_size, dtype=torch.float32, device=x.device)
        self._run_local(hits, misses, x, out, admit=decode, phase=phase)
        c.add("time", f"{phase}.route_plan", t1 - t0)
        c.add("time", f"{phase}.local", time.perf_counter() - t1)
        return out.to(x.dtype)

    def _run_local(self, hits, misses, x, out, admit: bool, phase: str) -> None:
        """Hits are already pinned. Misses are loaded in parallel (the peer's copy over the
        link, or this box's drive) and run here as they land: into a cache slot during decode
        (admitted), into a transient buffer during prefill or when no slot can be freed."""
        box, cache = self.box, self.box.cache
        c = box.counters
        pending: dict[Future, tuple] = {}
        for key, ti, w, from_peer in misses:
            if admit:
                slot, st = cache.acquire(key, reserve=True)
                if st is Acquire.HIT:  # another thread loaded it meanwhile
                    hits.append((key, ti, w))
                    c.add("experts", f"{phase}.{LOCAL}")
                    c.add("pairs", f"{phase}.{LOCAL}", int(ti.numel()))
                    continue
                if st is Acquire.RESERVED:
                    pending[box.reader.submit(box.load_and_commit, key, slot, from_peer)] = (key, slot, ti, w)
                    continue
            pending[box.reader.submit(box.load_transient, key, from_peer)] = (key, None, ti, w)
        # On an error keep going until every pin and buffer is given back, then raise the first one.
        err: BaseException | None = None
        pinned = []  # released only once the GPU has run every kernel reading them
        for key, ti, w in hits:  # compute hits while the misses load
            pinned.append(key)
            try:
                if err is None:
                    self._apply(cache.slab[int(cache.slot_of[key])], x, ti, w, out)
            except BaseException as exc:
                err = exc
        try:
            for fut in as_completed(pending):
                key, slot, ti, w = pending[fut]
                try:
                    res = fut.result()
                except BaseException as exc:  # a failed load already gave back its slot or buffer
                    err = err or exc
                    continue
                if slot is not None:
                    pinned.append(key)
                    buf, src = cache.slab[slot], res
                else:
                    buf, src = res
                n = int(ti.numel())
                if src == PEER:
                    c.add("experts", f"{phase}.{PEER}")
                    c.add("pairs", f"{phase}.{PEER}", n)
                else:
                    if src == PEER_MISS:
                        c.add("experts", f"{phase}.{PEER_MISS}")
                    kind = NVME if slot is not None else NVME_TMP
                    c.add("experts", f"{phase}.{kind}")
                    c.add("pairs", f"{phase}.{kind}", n)
                try:
                    if err is None:
                        self._apply(buf, x, ti, w, out)
                except BaseException as exc:
                    err = err or exc
                finally:
                    if slot is None:
                        _drain()  # a loader thread may be waiting for this buffer
                        box.transients.put(buf)
        finally:
            _drain()
            for key in pinned:
                cache.release(key)
        if err is not None:
            raise err

    # ------------------------------------------------------------------ for the peer
    def serve_fetch(self, h: dict) -> tuple[dict, list]:
        """The peer copies one expert out of this box's cache: into serve[i] for the peer's RDMA
        READ (h["rdma"] = i), or its bytes in the reply (tcp), or MISS."""
        box, cache = self.box, self.box.cache
        key = int(h["key"])
        slot, st = cache.acquire(key, reserve=False, local=False, refresh=box.cfg.cache.remote_refresh,
                                 wait=False)  # never wait for a load here (cache.py: two boxes would deadlock)
        if st is not Acquire.HIT:
            box.counters.add("served", "miss")
            return {"miss": True}, []
        n, i = self.index.expert_bytes, h.get("rdma")
        try:
            # A real copy, taken while the slot is pinned: the slot may hold another expert as soon
            # as it is released (.cpu() alone returns the slot itself on a CPU box). Synchronous.
            if i is not None:  # serve[i] is only ever read by the peer's loader thread i
                box.rdma.serve[int(i), :n].copy_(cache.slab[slot][:n])
            else:
                blob = cache.slab[slot][:n].to("cpu", copy=True)
        finally:
            cache.release(key)
        box.counters.add("served", "experts")
        return ({"rdma": True}, []) if i is not None else ({}, [blob])

    def serve_compute(self, h: dict, tensors: list) -> tuple[dict, list]:
        box, cache = self.box, self.box.cache
        t0 = time.perf_counter()
        stream = box.peer_stream
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            x, remap, w = tensors
            x = x.to(box.device)
            remap = remap.to(box.device).long()
            w = w.to(box.device).to(x.dtype)
            layer, experts, counts = h["layer"], h["experts"], h["counts"]
            out = torch.zeros(x.shape[0], self.m.hidden_size, dtype=torch.float32, device=box.device)
            miss, pinned = [], []
            refresh = box.cfg.cache.remote_refresh
            try:
                for e, ti, wi in zip(experts, torch.split(remap, counts), torch.split(w, counts)):
                    key = self.index.key(layer, e)
                    slot, st = cache.acquire(key, reserve=False, local=False, refresh=refresh, wait=False)
                    if st is not Acquire.HIT:
                        miss.append(e)
                        continue
                    pinned.append(key)
                    self._apply(cache.slab[slot], x, ti, wi, out)
                out = out.cpu()  # waits for this stream only
            finally:
                _drain(stream)
                for key in pinned:
                    cache.release(key)
        svc = time.perf_counter() - t0
        c = box.counters
        c.add("served", "requests")
        c.add("served", "experts", len(experts) - len(miss))
        c.add("served", "miss", len(miss))
        c.add("time", "served", svc)
        return {"miss": miss, "svc": svc}, [out]
