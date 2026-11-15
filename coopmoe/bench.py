"""Measurement for driver.py: replays the selected requests against both boxes, one cell at a
time, and computes the metrics.

A cell is a tag plus config overrides. Bench overrides apply here; the same overrides go to both
engines with the reset (cache.*, link.*, ...), so several cells (an evaluation's setups) run on
the same two engine processes, one after another.

Every request is a single turn: its prompt, then exactly bench.max_new_tokens generated tokens.
All requests arrive at the start, in order. Each goes to the box make_requests.py assigned it
to ("box": a random half of every category per box); nothing here decides placement. Each box
works through its own queue with engine.max_sessions requests in flight, so a box never idles
while it has requests waiting, and a request never moves to the other box.

Metrics per cell: aggregate output tok/s, time per output token (median, p95), TTFT, end-to-end
latency on the box (dispatch to last token) and the wait before dispatch, where routed experts
came from (own cache / peer / own NVMe), cache overlap
|A n B| / |A u B| over time, turns per box, NVMe and link bytes. Every request keeps its
generated text.
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from coopmoe.config import RunConfig
from coopmoe.stats import SOURCES
from coopmoe.transport import TcpClient
from coopmoe.workload import Formatter


class BoxPool:
    """Thread-safe access to both boxes: one connection per calling thread and box."""

    def __init__(self, addrs: list, timeout: float = 600.0):
        self.addrs, self.timeout = addrs, timeout
        self._tls = threading.local()
        self._all: list = []
        self._lock = threading.Lock()
        self._ex = ThreadPoolExecutor(2, thread_name_prefix="boxpool")  # long-lived: reuses connections

    def _client(self, box: int):
        clients = getattr(self._tls, "clients", None)
        if clients is None:
            clients = self._tls.clients = {}
        c = clients.get(box)
        if c is None:
            c = clients[box] = TcpClient(self.addrs[box], self.timeout)
            with self._lock:
                self._all.append(c)
        return c

    def call(self, box: int, header: dict, tensors=()):
        return self._client(box).call(header, tensors)

    def both(self, header: dict) -> list:
        futs = [self._ex.submit(self.call, b, header) for b in (0, 1)]
        return [f.result() for f in futs]

    def release_thread(self) -> None:
        """Close the calling thread's connections (a worker thread that is about to exit)."""
        clients = getattr(self._tls, "clients", None) or {}
        with self._lock:
            for c in clients.values():
                c.close()
                if c in self._all:
                    self._all.remove(c)
        self._tls.clients = {}

    def close(self) -> None:
        self._ex.shutdown(wait=False)
        for c in self._all:
            c.close()


def _pct(xs: list[float], q: float) -> Optional[float]:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def _overlap(a: np.ndarray, b: np.ndarray) -> dict:
    inter, union = int((a & b).sum()), int((a | b).sum())
    return {"a": int(a.sum()), "b": int(b.sum()), "inter": inter, "union": union,
            "jaccard": inter / union if union else 0.0}


def run_cell(cfg: RunConfig, pool: BoxPool, requests: list[dict], fmt: Formatter, overrides: list[str],
             epoch: int, n_keys: int, log=print) -> dict:
    b = cfg.bench
    pool.both({"op": "quiesce"})
    infos = [rh["info"] for rh, _ in pool.both({"op": "reset", "overrides": overrides, "epoch": epoch})]
    lock = threading.Lock()
    queues: list[list] = [[], []]
    for r in requests:
        if r.get("box") not in (0, 1):
            raise ValueError(f"request {r.get('id')!r} has no box: regenerate the requests with make_requests.py")
        queues[r["box"]].append(r)
    records: list[dict] = []
    errors: list[str] = []
    progress = {"done": 0, "t_measure": None, "box_done": [None, None]}
    t_start = time.perf_counter()
    if b.warmup_sessions <= 0:
        progress["t_measure"] = t_start

    samples: list[dict] = []
    stop = threading.Event()

    def sample() -> None:
        res = pool.both({"op": "state"})
        bits = [np.unpackbits(rt[0].numpy())[:n_keys].astype(bool) for _, rt in res]
        s = _overlap(bits[0], bits[1])
        s.update(t=time.perf_counter() - t_start, steps=[rh["step"] for rh, _ in res],
                 measured=progress["t_measure"] is not None)
        samples.append(s)

    def sampler() -> None:
        while not stop.wait(b.sample_s):
            try:
                sample()
            except Exception as exc:
                log(f"[bench] overlap sample failed: {exc}")

    def worker(box: int) -> None:
        try:
            while True:
                with lock:
                    if not queues[box]:
                        return
                    r = queues[box].pop(0)
                run_request(r, box)
        finally:
            pool.release_thread()

    def run_request(r: dict, box: int) -> None:
        sid = f"e{epoch}:{r['id']}"
        try:
            t0 = time.perf_counter()
            rh, _ = pool.call(box, {"op": "turn", "sid": sid, "ids": list(fmt.prompt(r["request"])),
                                    "max_new": int(b.max_new_tokens)})
            t1 = time.perf_counter()
            times, n = rh["times"], len(rh["tokens"])
            rec = {"id": r["id"], "category": r["category"], "box": rh["box"],
                   "t_send": t0 - t_start, "latency_s": t1 - t0, "out_tokens": n,
                   "ttft_s": rh["ttft_s"], "queue_s": rh["queue_s"], "prefill_s": rh["prefill_s"],
                   "prompt_tokens": rh["prompt_tokens"],
                   "tpot_s": (times[-1] - times[0]) / (n - 1) if n > 1 else None,
                   "measured": (progress["t_measure"] or -1.0) > 0 and t0 >= progress["t_measure"],
                   "text": fmt.text(rh["tokens"])}
            with lock:
                records.append(rec)
        except Exception as exc:
            with lock:
                errors.append(f"{sid}: {exc}")
            log(f"[bench] {sid} failed: {exc}")
        finished(box)

    def finished(box: int) -> None:
        start_measuring = False
        with lock:
            progress["done"] += 1
            progress["box_done"][box] = time.perf_counter() - t_start
            if progress["t_measure"] is None and progress["done"] >= b.warmup_sessions:
                progress["t_measure"] = -1.0  # claimed; set below
                start_measuring = True
        if start_measuring:
            pool.both({"op": "reset_counters"})
            with lock:
                progress["t_measure"] = time.perf_counter()
            log(f"[bench] warm-up done after {progress['done']} requests; measuring")

    if progress["t_measure"] is not None:
        pool.both({"op": "reset_counters"})
    smp = threading.Thread(target=sampler, daemon=True)
    smp.start()
    workers = [threading.Thread(target=worker, args=(box,), name=f"box{box}-slot{k}")
               for box in (0, 1) for k in range(cfg.engine.max_sessions)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    t_end = time.perf_counter()
    stop.set()
    smp.join()
    sample()
    stats = [rh["stats"] for rh, _ in pool.both({"op": "stats"})]
    summary = summarize(records, samples, stats, progress["t_measure"] or t_start, t_end, t_start)
    summary["box_done_s"] = progress["box_done"]
    summary["errors"] = len(errors)
    return {"boxes": infos, "summary": summary, "box_stats": stats,
            "overlap": samples, "turns": records, "errors": errors}


def summarize(records, samples, stats, t_measure, t_end, t_start) -> dict:
    meas = [r for r in records if r["measured"]] or records
    wall = max(t_end - t_measure, 1e-9)
    out_tokens = sum(r["out_tokens"] for r in meas)
    tpots = [r["tpot_s"] for r in meas if r["tpot_s"] is not None]
    ttfts = [r["ttft_s"] for r in meas]
    e2e = [r["latency_s"] for r in meas]
    waits = [r["t_send"] for r in meas]  # everything arrives at the start
    src, link, served = Counter(), Counter(), Counter()
    nvme_bytes = 0
    for s in stats:
        c = s["counters"]
        src.update({k: v for k, v in c["experts"].items()})
        link.update(c["link"])
        served.update(c["served"])
        nvme_bytes += s["nvme_bytes"]

    def share(phase: str) -> dict:
        tot = sum(src.get(f"{phase}.{k}", 0) for k in SOURCES)
        return {k: src.get(f"{phase}.{k}", 0) / tot for k in SOURCES} if tot else {}

    msamples = [s for s in samples if s["measured"]] or samples
    link_bytes = sum(v for k, v in link.items() if k in ("act_out", "partial_in", "bitmap_out", "fetch_in"))
    per_box = Counter(r["box"] for r in meas)
    return {
        "turns": len(meas), "wall_s": wall,
        "total_s": t_end - t_start, "out_tokens": out_tokens, "tok_s": out_tokens / wall,
        # The whole run, warm-up included: first request sent -> last request finished.
        "run_s": t_end - t_start, "run_tok_s": sum(r["out_tokens"] for r in records) / max(t_end - t_start, 1e-9),
        "tpot_ms_p50": None if not tpots else statistics.median(tpots) * 1e3,
        "tpot_ms_p95": None if not tpots else _pct(tpots, 0.95) * 1e3,
        "ttft_s_mean": None if not ttfts else statistics.mean(ttfts),
        "ttft_s_p95": _pct(ttfts, 0.95),
        "e2e_s_p50": None if not e2e else statistics.median(e2e), "e2e_s_p95": _pct(e2e, 0.95),
        "wait_s_mean": None if not waits else statistics.mean(waits),
        "decode_sources": share("decode"), "prefill_sources": share("prefill"),
        "peer_miss": src.get("decode.peer_miss", 0) + src.get("prefill.peer_miss", 0),
        "overlap_jaccard_mean": statistics.mean(s["jaccard"] for s in msamples) if msamples else None,
        "overlap_final": samples[-1] if samples else None,
        "nvme_gb_s": nvme_bytes / wall / 1e9, "link_gb_s": link_bytes / wall / 1e9,
        "turns_per_box": [per_box.get(0, 0), per_box.get(1, 0)],
        "busy_skips": link.get("busy_skips", 0),
    }


def format_row(tag: str, s: dict) -> str:
    d = s["decode_sources"] or {}

    def f(x, fmt="{:.1f}"):
        return "-" if x is None else fmt.format(x)

    return (f"{tag:<14} {f(s['tok_s'], '{:7.2f}')} tok/s  tpot p50 {f(s['tpot_ms_p50'], '{:7.1f}')} ms "
            f"p95 {f(s['tpot_ms_p95'], '{:7.1f}')} ms  ttft {f(s['ttft_s_mean'], '{:5.2f}')} s  "
            f"e2e p50 {f(s['e2e_s_p50'], '{:6.2f}')} s  "
            f"decode local/peer/nvme {d.get('local', 0):.2f}/{d.get('peer', 0):.2f}/"
            f"{d.get('nvme', 0) + d.get('nvme_tmp', 0):.2f}  overlap {f(s['overlap_jaccard_mean'], '{:.2f}')}  "
            f"turns A/B {s['turns_per_box'][0]}/{s['turns_per_box'][1]}  errors {s.get('errors', 0)}")
