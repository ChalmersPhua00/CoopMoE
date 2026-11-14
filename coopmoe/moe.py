"""The MoE layer at one box, and the work it does for its peer.

Per MoE layer, for the box's own batch (MoeExecutor.__call__, called by model.CoopExperts
with HF's routing):
    1. copy the router's top-k ids to the host once
    2. per routed expert, in order of preference:
         own cache  -> run here (pinned so this layer's loads cannot evict it)
         peer holds -> goes in this layer's single peer message (unless the peer is busy)
         neither    -> read own NVMe; admit during decode, transient buffer during prefill
    3. the peer message goes out on the link thread while hits and drive reads run here
    4. add the peer's partial sums; experts it reports MISS (stale bitmap) are read from
       own NVMe

For the peer (serve_compute / serve_fetch): run requested experts that are in this box's
cache, report the rest as MISS; hand out expert bytes for the peer's admissions. The peer
never reads this box's drive on its behalf.
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
        hits, remote, misses = [], [], []
        for e, ti, w in groups:
            key = self.index.key(layer, e)
            _, st = cache.acquire(key, reserve=False)
            if st is Acquire.HIT:
                hits.append((key, ti, w))
            elif use_peer and peer.holds(key):
                remote.append((e, key, ti, w))
            else:
                misses.append((key, ti, w))
        fut = None
        if remote:
            fut = peer.compute_async(layer, remote, x)
            if decode:
                box.admitter.note([key for _, key, _, _ in remote], cache.step)
        c.add("experts", f"{phase}.{LOCAL}", len(hits))
        c.add("pairs", f"{phase}.{LOCAL}", sum(int(ti.numel()) for _, ti, _ in hits))
        t1 = time.perf_counter()

        out = torch.zeros(x.shape[0], self.m.hidden_size, dtype=torch.float32, device=x.device)
        self._run_local(hits, misses, x, out, admit=decode, phase=phase)
        t2 = time.perf_counter()
        if fut is not None:
            needed, partial, miss = fut.result()
            t3 = time.perf_counter()
            out.index_add_(0, needed, partial.to(out.device))
            miss = set(miss)
            n_ok = len(remote) - len(miss)
            c.add("experts", f"{phase}.{PEER}", n_ok)
            c.add("pairs", f"{phase}.{PEER}", sum(int(ti.numel()) for e, _, ti, _ in remote if e not in miss))
            if miss:
                c.add("experts", f"{phase}.{PEER_MISS}", len(miss))
                late = [(key, ti, w) for e, key, ti, w in remote if e in miss]
                self._run_local([], late, x, out, admit=decode, phase=phase)
            c.add("time", "peer_wait", t3 - t2)
        c.add("time", f"{phase}.route_plan", t1 - t0)
        c.add("time", f"{phase}.local", t2 - t1)
        return out.to(x.dtype)

    def _run_local(self, hits, misses, x, out, admit: bool, phase: str) -> None:
        """Hits are already pinned. Misses are read from this box's drive in parallel and run
        as they land: into a cache slot during decode (admitted), into a transient buffer
        during prefill or when no slot can be freed."""
        box, cache = self.box, self.box.cache
        c = box.counters
        pending: dict[Future, tuple] = {}
        n_nvme = n_tmp = p_nvme = p_tmp = 0
        for key, ti, w in misses:
            if admit:
                slot, st = cache.acquire(key, reserve=True)
                if st is Acquire.HIT:  # another thread loaded it meanwhile
                    hits.append((key, ti, w))
                    c.add("experts", f"{phase}.{LOCAL}")
                    c.add("pairs", f"{phase}.{LOCAL}", int(ti.numel()))
                    continue
                if st is Acquire.RESERVED:
                    pending[box.reader.submit(box.load_and_commit, key, slot)] = (key, slot, ti, w)
                    n_nvme += 1
                    p_nvme += int(ti.numel())
                    continue
            pending[box.reader.submit(box.load_transient, key)] = (key, None, ti, w)
            n_tmp += 1
            p_tmp += int(ti.numel())
        # On an error keep going until every pin and buffer is given back, then raise the first one.
        err: BaseException | None = None
        for key, ti, w in hits:  # compute hits while the drive works on the misses
            try:
                if err is None:
                    self._apply(cache.slab[int(cache.slot_of[key])], x, ti, w, out)
            except BaseException as exc:
                err = exc
            finally:
                cache.release(key)
        for fut in as_completed(pending):
            key, slot, ti, w = pending[fut]
            try:
                res = fut.result()
            except BaseException as exc:  # a failed load already gave back its slot or buffer
                err = err or exc
                continue
            try:
                if err is None:
                    self._apply(cache.slab[slot] if slot is not None else res, x, ti, w, out)
            except BaseException as exc:
                err = err or exc
            finally:
                if slot is not None:
                    cache.release(key)
                else:
                    box.transients.put(res)
        if err is not None:
            raise err
        if misses:
            c.add("experts", f"{phase}.{NVME}", n_nvme)
            c.add("experts", f"{phase}.{NVME_TMP}", n_tmp)
            c.add("pairs", f"{phase}.{NVME}", p_nvme)
            c.add("pairs", f"{phase}.{NVME_TMP}", p_tmp)

    # ------------------------------------------------------------------ for the peer
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
            miss = []
            refresh = box.cfg.cache.remote_refresh
            for e, ti, wi in zip(experts, torch.split(remap, counts), torch.split(w, counts)):
                key = self.index.key(layer, e)
                slot, st = cache.acquire(key, reserve=False, local=False, refresh=refresh)
                if st is not Acquire.HIT:
                    miss.append(e)
                    continue
                try:
                    self._apply(cache.slab[slot], x, ti, wi, out)
                finally:
                    cache.release(key)
            out = out.cpu()  # waits for this stream only
        svc = time.perf_counter() - t0
        c = box.counters
        c.add("served", "requests")
        c.add("served", "experts", len(experts) - len(miss))
        c.add("served", "miss", len(miss))
        c.add("time", "served", svc)
        return {"miss": miss, "svc": svc}, [out]

    def serve_fetch(self, h: dict) -> tuple[dict, list]:
        box, cache = self.box, self.box.cache
        have, blobs = [], []
        n = self.index.expert_bytes
        for key in h["keys"]:
            slot, st = cache.acquire(int(key), reserve=False, local=False, refresh=False)
            if st is Acquire.HIT:
                try:
                    blobs.append(cache.slab[slot][:n].cpu().clone())
                finally:
                    cache.release(int(key))
                have.append(True)
            else:
                have.append(False)
        box.counters.add("served", "fetch_hit", sum(have))
        box.counters.add("served", "fetch_miss", len(have) - sum(have))
        return {"have": have}, blobs
