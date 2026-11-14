"""The results table per cell, the per-request outputs, and the overlap series as CSV."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

def _f(x, fmt="{:.2f}") -> str:
    return "-" if x is None else fmt.format(x)


def cell_table(rows: list[dict]) -> str:
    head = ("cell", "run s", "run tok/s", "tok/s", "tpot p50 ms", "tpot p95 ms", "ttft s", "e2e p50 s", "e2e p95 s", "wait s",
            "local", "peer", "nvme", "overlap", "turns A/B", "nvme GB/s", "link GB/s", "err")
    lines = [" | ".join(head), " | ".join("---" for _ in head)]
    for r in rows:
        s = r["summary"]
        d = s.get("decode_sources") or {}
        lines.append(" | ".join([
            r["tag"], _f(s.get("run_s"), "{:.1f}"), _f(s.get("run_tok_s")), _f(s["tok_s"]), _f(s["tpot_ms_p50"], "{:.1f}"), _f(s["tpot_ms_p95"], "{:.1f}"),
            _f(s["ttft_s_mean"]), _f(s.get("e2e_s_p50")), _f(s.get("e2e_s_p95")), _f(s.get("wait_s_mean")),
            _f(d.get("local")), _f(d.get("peer")),
            _f((d.get("nvme") or 0) + (d.get("nvme_tmp") or 0)),
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
             "request": by_id[r["id"]]["request"], "output": r["text"]} for r in res["turns"]]


def write_cell(out: str, res: dict, outputs: list[dict]) -> None:
    with open(os.path.join(out, f"{res['tag']}.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "turns"}, f, indent=1, default=str)
    with open(os.path.join(out, f"{res['tag']}_outputs.jsonl"), "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
