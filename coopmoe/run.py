"""The experiment itself, on PGX-A, inside the Fuzzball job driver.py submits.

  1. store   check this box's expert store is there (built once on NFS by stage.py; bootstrap.sh
             copied it to this box's local disk)
  2. start   this box's engine as a child process; box 1's engine is started on PGX-B by
             the other Fuzzball job of the pair
  3. run     the requests once per cell: each goes to the box make_requests.py assigned it to,
             and each box serves engine.max_sessions at a time
  4. report  <out>/report.md, <out>/<cell>.json, <out>/<cell>_outputs.jsonl; then both engines stop

With emit, every result also goes to stdout as '@@coopmoe <kind> <i>/<n> <base64 chunk>' lines
(_emit): that is how driver.py collects results from the Fuzzball job log.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time

from coopmoe.bench import BoxPool, format_row, run_cell
from coopmoe.config import apply_overrides, load_run_config
from coopmoe.report import output_records, render, write_cell
from coopmoe.workload import Formatter

MARK = "@@coopmoe"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _emit(kind: str, obj: dict) -> None:
    """One result as '@@coopmoe <kind> <i>/<n> <chunk>' lines: the JSON in base64, cut into
    chunks. The Fuzzball CLI cannot read back a log line longer than 64 KB, and a cell's
    result is longer than that."""
    data = base64.b64encode(json.dumps(obj, ensure_ascii=False, default=str).encode()).decode()
    chunks = [data[i:i + 8000] for i in range(0, len(data), 8000)] or [""]
    for i, c in enumerate(chunks):
        print(f"{MARK} {kind} {i + 1}/{len(chunks)} {c}", flush=True)


def _start_engine(config: str, sets: list[str], log_path: str) -> subprocess.Popen:
    with open(log_path, "w") as f:
        return subprocess.Popen([sys.executable, "-m", "coopmoe", "engine", "--config", config, "--rank", "0",
                                 "--set", *sets], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)


def _wait_ready(pool: BoxPool, engine: subprocess.Popen, path: str, timeout: float, log) -> None:
    t0, said = time.time(), False
    for b in (0, 1):
        while True:
            try:
                pool.call(b, {"op": "ping"})
                break
            except Exception:
                if engine.poll() is not None:
                    raise SystemExit(f"engine0 exited with {engine.returncode}; see {path}")
                if time.time() - t0 > timeout:
                    raise SystemExit(f"box {b} did not answer within {timeout:.0f} s")
                if b == 1 and not said and time.time() - t0 > 60:
                    log("still waiting for box 1 (it may still be copying its store)")
                    said = True
                time.sleep(1)


def run(config: str, model: str, store: str, requests: list[dict], cells: list, out: str, boxes: list[str],
        emit: bool = False, ready_timeout: float = 12 * 3600, log=print) -> list[dict]:
    if not os.path.exists(os.path.join(store, "index.json")):
        raise SystemExit(f"no expert store at {store}: build it once with stage.py")
    sets = [f"store_dir={store}", f"cluster.addrs={json.dumps(boxes)}"]
    cfg = load_run_config(config, sets)
    os.makedirs(out, exist_ok=True)
    by_id = {r["id"]: r for r in requests}

    pool = BoxPool(boxes, timeout=cfg.cluster.connect_timeout)
    path = os.path.join(out, "engine0.log")
    engine = _start_engine(config, sets, path)
    results = []
    try:
        log(f"waiting for both engines ({', '.join(boxes)}; engine0 log in {path})")
        _wait_ready(pool, engine, path, ready_timeout, log)
        infos = [pool.call(b, {"op": "ping"})[0]["info"] for b in (0, 1)]
        for info in infos:
            log(f"box {info['rank']} on {info['host']}: {info['device']}, {info['slots']} expert slots "
                f"({info['cache_gib']:.1f} GiB), store {info['store_dtype']}")
        if infos[0]["n_keys"] != infos[1]["n_keys"]:
            raise SystemExit("the two boxes loaded different models")
        fmt = Formatter.for_store(store)
        for epoch, (tag, cell_overrides) in enumerate(cells, start=1):
            cell_cfg = apply_overrides(cfg.copy(), cell_overrides)
            log(f"cell {tag}: {' '.join(cell_overrides)}")
            res = run_cell(cell_cfg, pool, requests, fmt, cell_overrides, epoch, infos[0]["n_keys"], log)
            res.update(tag=tag, workload=f"{len(requests)} requests", model=model, overrides=cell_overrides,
                       config=cell_cfg.to_dict())
            outputs = output_records(res, by_id)
            write_cell(out, res, outputs)
            log(format_row(tag, res["summary"]))
            if emit:
                _emit("cell", {k: v for k, v in res.items() if k not in ("turns", "config")})
                for o in outputs:
                    _emit("output", o)
            results.append(res)
        with open(os.path.join(out, "report.md"), "w") as f:
            f.write(f"# {model}\n\n" + render(results, os.path.join(out, "overlap.csv")))
        if emit:
            _emit("done", {"out": out})
        return results
    finally:
        for b in (0, 1):
            try:
                pool.call(b, {"op": "shutdown"})
            except Exception:
                pass
        pool.close()
        try:
            engine.wait(timeout=60)
        except subprocess.TimeoutExpired:
            engine.kill()


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="coopmoe run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="Hub repo (the report's title)")
    ap.add_argument("--store", required=True, help="this box's expert store")
    ap.add_argument("--requests", required=True, help="the requests to run, in order (driver.py selects them)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxes", required=True, help="engine addresses in rank order, A:port,B:port")
    ap.add_argument("--cell", action="append", required=True, metavar="'TAG KEY=VALUE ...'",
                    help="one cell: a tag, then config overrides; repeatable")
    ap.add_argument("--emit", action="store_true", help="results to stdout as '@@coopmoe' lines")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(f"[run {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    with open(args.requests, encoding="utf-8") as f:
        requests = json.load(f)
    cells = [(c.split()[0], c.split()[1:]) for c in args.cell]
    run(args.config, args.model, args.store, requests, cells, args.out, args.boxes.split(","), args.emit, log=log)
