"""One PGX box: serves its own requests, and hands the other box copies of experts in its cache.

State on a box:
    non-expert weights   resident, never evicted (both boxes run attention)
    KV caches            HF's paged cache, for the requests this box is serving
    expert cache         one LRU (cache.py) over a fixed slot budget
    drive                the full model (store/)

Requests run through Hugging Face's continuous batching (the same loop `transformers serve`
uses): up to engine.max_batch requests share each forward, and a new request joins as soon as
there is room. The model is HF's Qwen3-MoE with the experts swapped (model.py). Two hooks around
each forward keep the box's own bookkeeping: before it, whether this step is a decode step (one
new token per request: misses are admitted to the cache) or contains prompt tokens (misses go to
a transient buffer); after it, the step count and a residency-bitmap push if the cache changed.

Threads: HF's generation loop, a result thread (hands finished requests back, pushes the bitmap
while idle), one server thread per connection (bench turns block until done; the other box's
fetches are answered right away, from this box's cache), the loader threads (a miss: a copy
from the other box, or a read from this box's drive) and the link thread (bitmap pushes).

Ops (TcpServer -> Box.handle):
    bench:  turn, quiesce, reset, reset_counters, state, stats, ping, shutdown
    peer:   fetch, bitmap, rdma_info   (compute: the earlier design where the other box ran the
                                        expert; this version never sends it)

With link.transport=rdma (rdma.py) a fetch's reply only says the expert is in the agreed serve
buffer, and the asking box RDMA-reads it from there; with tcp the bytes ride in the reply.
"""
from __future__ import annotations

import json
import math
import socket
import threading
import time
from concurrent.futures import Future
from typing import Callable, Optional

import torch
from transformers import ContinuousBatchingConfig, GenerationConfig

from coopmoe import model as hf_model
from coopmoe.cache import ExpertCache
from coopmoe.config import RunConfig, apply_overrides
from coopmoe.moe import MoeExecutor
from coopmoe.peer import PeerLink
from coopmoe.rdma import RdmaPlane
from coopmoe.stats import NVME, PEER, PEER_MISS, Counters
from coopmoe.store.layout import FP8, StoreIndex
from coopmoe.store.reader import BufferPool, ExpertLoader, ExpertReader
from coopmoe.transport import TcpClient, TcpServer

PAGE = 256  # tokens per KV page


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def compute_dtype(index: StoreIndex) -> torch.dtype:
    """FP8 experts are dequantized to bfloat16 to run; otherwise the store's own dtype."""
    return torch.bfloat16 if index.dtype == FP8 else index.torch_dtype


class Box:
    def __init__(self, cfg: RunConfig, rank: int, log: Callable = print):
        self.base_cfg, self.cfg = cfg.copy(), cfg.copy()
        self.rank, self.log = rank, log
        self.index = StoreIndex.load(cfg.store_dir)
        self.m = self.index.model
        self.device = resolve_device(cfg.device)
        self.dtype = compute_dtype(self.index)
        self.reader = ExpertReader(cfg.store_dir, self.index, cfg.reader.threads, cfg.reader.direct_io)
        self.loader = ExpertLoader(self.reader, self.index, self.device, n_staging=cfg.reader.threads)
        self.moe = MoeExecutor(self)
        self.model = hf_model.build(cfg.store_dir, self.moe, self.device, self.dtype)
        self.counters = Counters()
        self.peer_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None  # serve_compute only
        self.peer: Optional[PeerLink] = None
        self.nvme_load_s: Optional[float] = None  # measured seconds per expert read from the drive
        self.rdma: Optional[RdmaPlane] = None    # registered buffers + NIXL agent, made on first use
        self._rdma_lock = threading.Lock()
        self.epoch = 0
        self.step = 0
        self.decode = False
        self.cache: Optional[ExpertCache] = None
        self.transients: Optional[BufferPool] = None
        self._configure(cfg)
        self.model.register_forward_pre_hook(self._before_step, with_kwargs=True)
        self.model.register_forward_hook(self._after_step)
        self.cb = self._start_cb(cfg)
        self.waiting: dict[str, Future] = {}
        self._stop = False
        self.stopped = threading.Event()
        self.server: Optional[TcpServer] = None
        self._thread = threading.Thread(target=self._results, name=f"results{rank}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ configuration
    def _n_slots(self, cfg: RunConfig) -> int:
        n = int(float(cfg.cache.gb) * 2**30 // self.index.stride)
        return max(0, min(n, self.m.n_keys))

    def _configure(self, cfg: RunConfig) -> None:
        """(Re)build what depends on knobs: cache and transient buffers."""
        if cfg.link.transport not in ("rdma", "tcp"):
            raise ValueError(f"link.transport must be rdma or tcp, got {cfg.link.transport!r}")
        self.cfg = cfg
        n = self._n_slots(cfg)
        if self.cache is None or self.cache.n_slots != n:
            self.cache = None
            self.cache = ExpertCache(self.index, n, self.device, cfg.cache.dup_first, cfg.cache.dup_protect_steps)
        else:
            self.cache.clear()
        self.cache.dup_first, self.cache.dup_protect = cfg.cache.dup_first, cfg.cache.dup_protect_steps
        if self.transients is None:
            self.transients = BufferPool(cfg.engine.n_transient, self.index.stride, self.device)

    def _start_cb(self, cfg: RunConfig):
        """HF continuous batching: greedy, end of turn ignored (fixed length), KV for max_sessions
        requests of max_seq tokens, a whole prompt in one forward. No CUDA graphs: the expert
        path decides on the host, layer by layer."""
        e = cfg.engine
        gen = GenerationConfig(do_sample=False, max_new_tokens=cfg.bench.max_new_tokens, eos_token_id=-1)
        cbc = ContinuousBatchingConfig(page_size=PAGE, num_blocks=e.max_sessions * math.ceil(e.max_seq / PAGE),
                                       max_batch_tokens=e.max_seq, max_requests_per_batch=e.max_batch,
                                       use_cuda_graph=False, use_async_batching=False, allow_block_sharing=False,
                                       auto_switch_to_flash=False)
        # HF's KV footprint is ~1.5x the KV itself (masks, batch tensors); it must fit in 0.9 x free.
        self._make_room(int((1.6 * e.max_sessions * self.kv_bytes_per_request() + 2**30) / 0.9))
        cb = self.model.init_continuous_batching(generation_config=gen, continuous_batching_config=cbc)
        cb.start()
        t0 = time.time()
        while cb.batch_processor is None:  # set by the loop thread once its KV cache exists
            if not cb.is_running():
                raise RuntimeError("HF's generation loop failed to start (its error is logged above)")
            if time.time() - t0 > 600:
                raise RuntimeError("HF's generation loop did not start within 600 s")
            time.sleep(0.05)
        self.cb_cfg = cfg
        return cb

    def _make_room(self, nbytes: int) -> None:
        """On the GB10 (unified memory) cudaMemGetInfo's "free" leaves out the page cache, although
        an allocation reclaims it. HF sizes and checks its KV cache against that "free", so on a
        node whose page cache is full (the store copy alone adds ~32-226 GB) it refuses to start.
        Allocating and touching nbytes once makes the kernel reclaim that much; freeing it leaves it
        free for HF's check."""
        if self.device.type != "cuda":
            return
        free, _ = torch.cuda.mem_get_info(self.device)
        if free >= nbytes:
            return
        try:
            x = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            x.fill_(0)
            torch.cuda.synchronize(self.device)
            del x
        except torch.OutOfMemoryError:
            self.log(f"[box{self.rank}] could not reclaim {nbytes / 2**30:.1f} GiB for the KV cache")
        torch.cuda.empty_cache()
        after, _ = torch.cuda.mem_get_info(self.device)
        self.log(f"[box{self.rank}] free memory {free / 2**30:.1f} -> {after / 2**30:.1f} GiB "
                 f"(reclaimed page cache for the KV cache, wanted {nbytes / 2**30:.1f} GiB)")

    @staticmethod
    def _cb_key(cfg: RunConfig) -> tuple:
        return cfg.engine.max_batch, cfg.engine.max_sessions, cfg.engine.max_seq

    def kv_bytes_per_request(self) -> int:
        m = self.m
        return 2 * m.num_hidden_layers * m.num_key_value_heads * m.head_dim * self.cfg.engine.max_seq * self.dtype.itemsize

    def nvme_s_per_expert(self) -> float:
        """Busy-peer threshold: the cost of reading one expert from this box's drive."""
        if self.cfg.link.nvme_ms_per_expert:
            return self.cfg.link.nvme_ms_per_expert / 1e3
        return self.index.expert_bytes / 3.9e9  # reported Phison Gen4 rate

    def describe(self) -> dict:
        return {"rank": self.rank, "host": socket.gethostname(), "device": str(self.device),
                "store_dtype": self.index.dtype, "compute_dtype": str(self.dtype).replace("torch.", ""),
                "slots": self.cache.n_slots, "slot_mib": self.index.stride / 2**20,
                "cache_gib": self.cache.n_slots * self.index.stride / 2**30, "n_keys": self.m.n_keys,
                "kv_session_gib": self.kv_bytes_per_request() / 2**30,
                "nvme_ms_per_expert": self.nvme_s_per_expert() * 1e3, "link": self.cfg.link.enabled,
                "transport": self.cfg.link.transport}

    # ------------------------------------------------------------------ drive helpers
    def _fill(self, key: int, dst: torch.Tensor, from_peer: bool) -> str:
        """One expert's record into dst: copied from the peer's cache if asked and the peer
        still holds it, else read from this box's drive. -> where it came from. Both paths are
        timed (same threads, same parallelism): PeerLink.usable() compares them."""
        src, peer = NVME, self.peer
        if from_peer and peer is not None:
            t0 = time.perf_counter()
            blob = peer.fetch(key)
            if blob is not None:
                self.loader.copy_in(dst, blob)
                peer.note_fetch(time.perf_counter() - t0)
                return PEER
            src = PEER_MISS
        t0 = time.perf_counter()
        self.loader.load(key, dst)
        dt = time.perf_counter() - t0
        self.nvme_load_s = dt if self.nvme_load_s is None else 0.9 * self.nvme_load_s + 0.1 * dt
        return src

    def load_and_commit(self, key: int, slot: int, from_peer: bool = False) -> str:
        try:
            src = self._fill(key, self.cache.slab[slot], from_peer)
        except BaseException:
            self.cache.abort(key, slot)
            raise
        self.cache.commit(key, slot)
        return src

    def load_transient(self, key: int, from_peer: bool = False) -> tuple[torch.Tensor, str]:
        buf = self.transients.get()
        try:
            src = self._fill(key, buf, from_peer)
        except BaseException:
            self.transients.put(buf)
            raise
        return buf, src

    # ------------------------------------------------------------------ peer link
    def rdma_plane(self) -> RdmaPlane:
        """This box's RDMA buffers (one serve and one landing buffer per loader thread) and NIXL
        agent, made on first use and kept for the life of the process."""
        with self._rdma_lock:
            if self.rdma is None:
                self.rdma = RdmaPlane(self.rank, self.cfg.reader.threads, self.index.stride,
                                      self.index.expert_bytes, pin=self.device.type == "cuda")
            return self.rdma

    def _open_peer(self) -> None:
        addr, t = self.cfg.cluster.addrs[1 - self.rank], self.cfg.cluster.connect_timeout
        self.peer = PeerLink(self, TcpClient(addr, t), self.epoch)
        self.cache.set_peer(None)
        if self.cfg.link.transport == "rdma":  # learn where the peer's serve buffers are
            plane = self.rdma_plane()
            plane.connect(*self.peer.layer.call({"op": "rdma_info"}))
            self.peer.rdma = plane

    def _close_peer(self) -> None:
        old, self.peer = self.peer, None
        self.cache.set_peer(None)
        if old is not None:
            old.close()

    # ------------------------------------------------------------------ server
    def serve(self, host: str = "0.0.0.0", port: Optional[int] = None) -> int:
        if port is None:
            port = int(self.cfg.cluster.addrs[self.rank].rsplit(":", 1)[1])
        self.server = TcpServer(host, port, self.handle)
        return self.server.port

    def handle(self, h: dict, tensors: list) -> tuple[dict, list]:
        op = h["op"]
        if op == "compute":
            with torch.inference_mode():
                return self.moe.serve_compute(h, tensors)
        if op == "fetch":
            return self.moe.serve_fetch(h)
        if op == "rdma_info":
            return self.rdma_plane().info()
        if op == "bitmap":
            if self.peer is not None:
                self.peer.receive_bitmap(int(h["epoch"]), int(h["version"]), tensors[0])
            return {}, []
        if op == "turn":
            return self.run_turn(h)
        if op == "quiesce":
            self.control(self._quiesce)
            return {}, []
        if op == "reset":
            self.control(lambda: self._reset(h.get("overrides", []), int(h.get("epoch", 0))))
            if self._cb_key(self.cfg) != self._cb_key(self.cb_cfg):  # the cell changed batch or KV sizes
                self.cb.stop(block=True)
                self.cb = self._start_cb(self.cfg)
            return {"info": self.describe()}, []
        if op == "reset_counters":
            self.control(self.reset_counters)
            return {}, []
        if op == "state":
            return self.state()
        if op == "stats":
            return {"stats": self.snapshot()}, []
        if op == "ping":
            return {"t": time.time(), "info": self.describe()}, tensors
        if op == "shutdown":
            threading.Thread(target=self.close, daemon=True).start()
            return {}, []
        raise ValueError(f"unknown op {op!r}")

    def control(self, fn: Callable) -> None:
        """Run fn between steps (no forward in flight)."""
        with self.cb.pause():
            fn()

    # ------------------------------------------------------------------ requests
    def run_turn(self, h: dict) -> tuple[dict, list]:
        """One request: its prompt, then exactly max_new generated tokens."""
        sid, ids, max_new = str(h["sid"]), list(h["ids"]), max(1, int(h["max_new"]))
        if len(ids) + max_new > self.cfg.engine.max_seq:
            raise ValueError(f"prompt of {len(ids)} tokens + {max_new} new exceeds engine.max_seq={self.cfg.engine.max_seq}")
        fut: Future = Future()
        self.waiting[sid] = fut
        self.cb.add_request(ids, request_id=sid, max_new_tokens=max_new, record_timestamps=True, eos_token_id=-1)
        with self.counters.lock:
            self.counters.turns += 1
        out = fut.result()
        if out.error:
            raise RuntimeError(out.error)
        t0, start, times = out.created_time, out.lifespan[0], out.timestamps
        res = {"tokens": out.generated_tokens, "box": self.rank, "queue_s": start - t0,
               "prefill_s": times[0] - start, "ttft_s": times[0] - t0,
               "times": [x - t0 for x in times], "prompt_tokens": len(ids)}
        return res, []

    def _results(self) -> None:
        """Hands each finished request back to its waiting turn; pushes the bitmap while idle
        (admissions can land between steps)."""
        while not self._stop:
            out = self.cb.get_result(timeout=0.05)
            if out is not None and out.is_finished():
                fut = self.waiting.pop(out.request_id, None)
                if fut is not None:
                    fut.set_result(out)
            if self.peer is not None:
                self.peer.push_bitmap()
        self.stopped.set()

    def _before_step(self, module, args, kwargs) -> None:
        n_tokens = kwargs["input_ids"].shape[-1]
        n_requests = kwargs["cu_seq_lens_q"].numel() - 1
        self.decode = n_tokens == n_requests  # one new token per request; otherwise prompt tokens are in it
        self.cache.step = self.step
        self._step_tokens = n_tokens

    def _after_step(self, module, args, output) -> None:
        with self.counters.lock:
            if self.decode:
                self.step += 1
                self.counters.steps += 1
                self.counters.decode_tokens += self._step_tokens
            else:
                self.counters.prefill_tokens += self._step_tokens
        if self.peer is not None:
            self.peer.push_bitmap()

    # ------------------------------------------------------------------ control ops
    def _quiesce(self) -> None:
        """Stop using and serving the link (first half of a reset across both boxes)."""
        self._close_peer()

    def _reset(self, overrides: list, epoch: int) -> None:
        """Start a benchmark cell: knobs from the cell, empty cache, zero counters."""
        if self.waiting:
            raise RuntimeError("reset while turns are in flight")
        self._quiesce()
        cfg = apply_overrides(self.base_cfg.copy(), overrides)
        self._configure(cfg)
        self.epoch = epoch
        self.step = 0
        self.reset_counters()
        if cfg.link.enabled:
            self._open_peer()
        self.log(f"[box{self.rank}] reset epoch {epoch}: {json.dumps(self.describe())}")

    def reset_counters(self) -> None:
        """Start of the measured window: keep cache contents, zero every counter."""
        self.counters.reset()
        self.reader.reads = self.reader.bytes = 0
        c = self.cache
        self._cache_base = (c.admissions, c.evictions, c.dup_evictions)

    # ------------------------------------------------------------------ reporting
    def state(self) -> tuple[dict, list]:
        version, packed = self.cache.bitmap()
        return ({"step": self.step, "version": version, "in_flight": len(self.waiting),
                 "held": int(self.cache.present.sum())},
                [torch.from_numpy(packed)])

    def snapshot(self) -> dict:
        occ = self.cache.occupancy()
        base = getattr(self, "_cache_base", (0, 0, 0))
        occ["admissions"] -= base[0]
        occ["evictions"] -= base[1]
        occ["dup_evictions"] -= base[2]
        return {"rank": self.rank, "counters": self.counters.snapshot(), "cache": occ,
                "nvme_reads": self.reader.reads, "nvme_bytes": self.reader.bytes,
                "peer_svc_ms_per_expert": (self.peer.svc_per_expert * 1e3
                                           if self.peer is not None and self.peer.svc_per_expert else None),
                "nvme_ms_per_expert_measured": self.nvme_load_s * 1e3 if self.nvme_load_s else None,
                "info": self.describe()}

    def close(self) -> None:
        self.cb.stop(block=True, hard_stop=True)
        self._stop = True
        self.stopped.wait(timeout=10)
        self._close_peer()
        if self.server is not None:
            self.server.close()
        self.reader.close()
