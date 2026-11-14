"""A cell's metrics from its request records and box stats, the results table per cell, the
per-request outputs, and the overlap series as CSV. Standard library only: driver.py imports it
outside the jobs."""
from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from typing import Optional

from coopmoe.stats import SOURCES


def _pct(xs: list[float], q: float) -> Optional[float]:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def summarize(records, samples, stats, t_measure, t_end, t_start) -> dict:
    meas = [r for r in records if r["measured"]] or records
    wall = max(t_end - t_measure, 1e-9)
    out_tokens = sum(r["out_tokens"] for r in meas)
    tpots = [r["tpot_s"] for r in meas if r["tpot_s"] is not None]
    ttfts = [r["ttft_s"] for r in meas]
    e2e = [r["latency_s"] for r in meas]
    waits = [r["t_send"] for r in meas]  # everything arrives at the start
    src, admit, link, served = Counter(), Counter(), Counter(), Counter()
    nvme_bytes = 0
    for s in stats:
        c = s["counters"]
        src.update({k: v for k, v in c["experts"].items()})
        admit.update(c["admit"])
        link.update(c["link"])
        served.update(c["served"])
        nvme_bytes += s["nvme_bytes"]

    def share(phase: str) -> dict:
        tot = sum(src.get(f"{phase}.{k}", 0) for k in SOURCES)
        return {k: src.get(f"{phase}.{k}", 0) / tot for k in SOURCES} if tot else {}

    msamples = [s for s in samples if s["measured"]] or samples
    link_bytes = sum(v for k, v in link.items() if k in ("act_out", "partial_in", "fetch_in", "bitmap_out"))
    per_box = Counter(r["box"] for r in meas)
    return {
        "turns": len(meas), "wall_s": wall,
        "total_s": t_end - t_start, "out_tokens": out_tokens, "tok_s": out_tokens / wall,
        "tpot_ms_p50": None if not tpots else statistics.median(tpots) * 1e3,
        "tpot_ms_p95": None if not tpots else _pct(tpots, 0.95) * 1e3,
        "ttft_s_mean": None if not ttfts else statistics.mean(ttfts),
        "ttft_s_p95": _pct(ttfts, 0.95),
        "e2e_s_p50": None if not e2e else statistics.median(e2e), "e2e_s_p95": _pct(e2e, 0.95),
        "wait_s_mean": None if not waits else statistics.mean(waits),
        "decode_sources": share("decode"), "prefill_sources": share("prefill"),
        "peer_miss": src.get("decode.peer_miss", 0) + src.get("prefill.peer_miss", 0),
        "admitted_from_peer": admit.get("admitted", 0),
        "overlap_jaccard_mean": statistics.mean(s["jaccard"] for s in msamples) if msamples else None,
        "overlap_final": samples[-1] if samples else None,
        "nvme_gb_s": nvme_bytes / wall / 1e9, "link_gb_s": link_bytes / wall / 1e9,
        "turns_per_box": [per_box.get(0, 0), per_box.get(1, 0)],
        "busy_skips": link.get("busy_skips", 0),
    }


def _f(x, fmt="{:.2f}") -> str:
    return "-" if x is None else fmt.format(x)


def cell_table(rows: list[dict]) -> str:
    head = ("cell", "tok/s", "tpot p50 ms", "tpot p95 ms", "ttft s", "e2e p50 s", "e2e p95 s", "wait s",
            "local", "peer", "nvme", "admits", "overlap", "turns A/B", "nvme GB/s", "link GB/s", "err")
    lines = [" | ".join(head), " | ".join("---" for _ in head)]
    for r in rows:
        s = r["summary"]
        d = s.get("decode_sources") or {}
        lines.append(" | ".join([
            r["tag"], _f(s["tok_s"]), _f(s["tpot_ms_p50"], "{:.1f}"), _f(s["tpot_ms_p95"], "{:.1f}"),
            _f(s["ttft_s_mean"]), _f(s.get("e2e_s_p50")), _f(s.get("e2e_s_p95")), _f(s.get("wait_s_mean")),
            _f(d.get("local")), _f(d.get("peer")),
            _f((d.get("nvme") or 0) + (d.get("nvme_tmp") or 0)), str(s.get("admitted_from_peer", 0)),
            _f(s.get("overlap_jaccard_mean")), "/".join(str(t) for t in s.get("turns_per_box", [])),
            _f(s.get("nvme_gb_s"), "{:.3f}"), _f(s.get("link_gb_s"), "{:.3f}"),
            str(s.get("errors", 0))]))
    return "\n".join(lines)


def render(results: list[dict], csv_path: str | None = None) -> str:
    by_wl: dict[str, list] = defaultdict(list)
    for r in results:
        by_wl[r.get("workload", "?")].append(r)
    parts = []
    for wl, rows in sorted(by_wl.items()):
        parts.append(f"## workload: {wl}\n")
        parts.append(cell_table(rows))
        parts.append("")
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["workload", "cell", "t_s", "steps_a", "steps_b", "held_a", "held_b", "inter", "union",
                        "jaccard", "measured"])
            for r in results:
                for s in r.get("overlap", []):
                    w.writerow([r.get("workload"), r["tag"], f"{s['t']:.3f}", s["steps"][0], s["steps"][1],
                                s["a"], s["b"], s["inter"], s["union"], f"{s['jaccard']:.4f}", s["measured"]])
        parts.append(f"overlap series -> {csv_path}")
    return "\n".join(parts)


def output_records(res: dict, by_id: dict) -> list[dict]:
    return [{"cell": res["tag"], "id": r["id"], "category": r["category"], "box": r["box"],
             "prompt_tokens": r["prompt_tokens"], "output_tokens": r["out_tokens"], "ttft_s": round(r["ttft_s"], 4),
             "tpot_ms": None if r["tpot_s"] is None else round(r["tpot_s"] * 1e3, 2),
             "e2e_s": round(r["latency_s"], 4), "wait_s": round(r["t_send"], 4),
             "measured": r["measured"],
             "request": by_id[r["id"]]["request"], "output": r["text"]} for r in res["turns"]]


def write_cell(out: str, res: dict, outputs: list[dict]) -> None:
    with open(os.path.join(out, f"{res['tag']}.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "turns"}, f, indent=1, default=str)
    with open(os.path.join(out, f"{res['tag']}_outputs.jsonl"), "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
