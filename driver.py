#!/usr/bin/env python3
"""The baseline for CoopMoE: the same requests on the same two PGX boxes, served by stock
Hugging Face transformers 5.23.0, each box on its own.

    python driver.py --requests data/requests.json --volume coopmoe-qwen3-235b-a22b --num 16 --concurrency 8

Run it anywhere the `fuzzball` CLI is logged in to the cluster. Same arguments as driver.py on
main, so a baseline run and a linked run are the same command on the two branches.

--volume is the model's NFS volume (made by stage.py on main); its name says which model. Then:
  1. select   --num requests for each box (default 64), from the requests make_requests.py
              assigned to that box: make_requests.select, the same call as on main
  2. render   fuzzball/baseline.fz twice, one single-node job per box, each carrying
              baseline.py and that box's requests
  3. submit   both at once; the jobs never talk to each other
  4. inside   each job copies the original checkpoint to its own NVMe (the first job on a
     the jobs volume downloads it there once), loads it with from_pretrained and serves its
              requests --concurrency at a time (baseline.py)
  5. collect  follows both job logs, then writes results/<model>-<time>-baseline/: report.md,
              outputs.jsonl (one line per request), results.json, requests.json, and per box
              box<b>.fz and box<b>.log

The report has one row per box and one for the pair. `run s` is the first request sent to the
last request finished; for the pair it is the slower box's, as if both had started together
(which is what a linked run on main measures). `run tok/s` is every generated token over that.
`tok/s` and the latencies leave out each box's warm-up requests, as on main.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shlex
import statistics
import subprocess
import tarfile
import threading
import time
from collections import Counter

from make_requests import select

REPO = os.path.dirname(os.path.abspath(__file__))
# The models main stages: short name -> Hub repo.
MODELS = {
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
}
# Fixed for every job: the same box as a CoopMoE job on main.
SETTINGS = dict(
    provisioner="team1",                             # where the model volumes live
    data_root="/data/coopmoe",                       # the model's NFS volume, inside the job
    local_root="/scratch", scratch="team1-scratch",  # the node's own NVMe, wiped per job
    image="docker://nvcr.io/nvidia/pytorch:25.09-py3",
    gpus=1, cores=20, memory="120GiB",               # a whole GB10 box (unified memory: a hard limit)
    timeout="24h",
    fuzzball="fuzzball", poll=15.0,                  # the CLI, seconds between status checks
)
MARK = "@@baseline"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
DONE = {"STAGE_STATUS_FINISHED", "STAGE_STATUS_FAILED", "STAGE_STATUS_CANCELED"}
S = argparse.Namespace(**SETTINGS)


def log(msg: str) -> None:
    print(f"[driver {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------- the workflow
def fill(template: str, values: dict) -> str:
    for k, v in values.items():
        template = template.replace("{{" + k + "}}", v)
    left = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", template)))
    if left:
        raise ValueError(f"unfilled placeholders: {left}")
    return template


def block(text: str, indent: int = 4) -> str:
    """Text as the body of a YAML literal block scalar."""
    pad = " " * indent
    return "\n".join(pad + line if line else "" for line in text.rstrip("\n").split("\n"))


def payload(requests: list[dict]) -> str:
    """baseline.py, requirements.txt and this box's requests as a base64 tar.gz."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in ("baseline.py", "requirements.txt"):
            t.add(os.path.join(REPO, name), arcname=name)
        data = json.dumps(requests, ensure_ascii=False).encode()
        info = tarfile.TarInfo("requests.json")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    return base64.encodebytes(buf.getvalue()).decode()


def render(short: str, box: int, requests: list[dict], concurrency: int) -> str:
    """fuzzball/baseline.fz for one box and its requests."""
    repo = MODELS[short]
    values = {"MODEL": shlex.quote(repo), "HF_DIR": f"{S.data_root}/hf/{repo.rpartition('/')[2].lower()}",
              "LOCAL": S.local_root, "BOX": str(box), "CONCURRENCY": str(concurrency), "MEMORY": S.memory}
    with open(os.path.join(REPO, "fuzzball", "bootstrap.sh")) as f:
        boot = fill(f.read(), values)
    with open(os.path.join(REPO, "fuzzball", "baseline.fz")) as f:
        return fill(f.read(), {
            "BOOTSTRAP": block(boot), "PAYLOAD": block(payload(requests)), "DATA": S.data_root,
            "LOCAL": S.local_root, "VOLUME_USE": S.provisioner, "VOLUME": f"coopmoe-{short}", "SCRATCH": S.scratch,
            "IMAGE": S.image, "CORES": str(S.cores), "MEMORY": S.memory, "GPUS": str(S.gpus), "TIMEOUT": S.timeout})


# ---------------------------------------------------------------------- fuzzball
def fuzzball(*cli: str, check: bool = True) -> str:
    p = subprocess.run([S.fuzzball, *cli], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise SystemExit(f"fuzzball {' '.join(cli)} failed ({p.returncode}):\n{p.stdout}{p.stderr}")
    return p.stdout + p.stderr


def start(wf: str, path: str) -> str:
    """Writes, validates and starts a workflow. -> its id."""
    with open(path, "w") as f:
        f.write(wf)
    log(f"workflow: {path} ({len(wf) / 1e3:.0f} kB)")
    fuzzball("workflow", "validate", path)
    m = UUID.search(fuzzball("workflow", "start", path))
    if not m:
        raise SystemExit(f"no workflow id after starting {path}")
    log(f"started workflow {m.group(0)}; Ctrl-C stops it")
    return m.group(0)


def follow(wid: str, prefix: str) -> str:
    """Streams the job's log until the workflow ends. -> its final status."""
    last = None
    while True:
        m = re.search(r"Status:\s*(\S+)", fuzzball("workflow", "describe", wid, check=False))
        st = m.group(1) if m else "?"
        if st != last:
            log(f"{prefix}workflow {wid}: {st}")
            last = st
        if st in DONE:
            return st
        if st == "STAGE_STATUS_STARTED":
            p = subprocess.Popen([S.fuzzball, "workflow", "log", wid, "run", "-f"], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                if MARK not in line:
                    print(prefix + line, end="", flush=True)
            p.wait()
        time.sleep(S.poll)


def run_both(wfs: list[str], paths: list[str]) -> tuple[list[str], list]:
    """Starts one workflow per box, both at once, and follows both to the end. Ctrl-C or a
    failed start stops whatever was started. -> (workflow ids, final statuses)."""
    wids: list[str] = []
    finals: list = [None, None]

    def watch(b: int) -> None:
        finals[b] = follow(wids[b], f"[box{b}] ")

    try:
        for wf, path in zip(wfs, paths):
            wids.append(start(wf, path))
        threads = [threading.Thread(target=watch, args=(b,), daemon=True) for b in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(timeout=1.0)
    except BaseException:
        for wid in wids:
            log(f"stopping workflow {wid}")
            fuzzball("workflow", "stop", wid, check=False)
        raise
    return wids, finals


def collect(job_log: str) -> tuple[dict | None, list[dict]]:
    """'@@baseline' lines of a job log -> (the box's line, its per-request outputs)."""
    box, outputs = None, []
    for line in job_log.splitlines():
        i = line.find(MARK + " ")
        if i < 0:
            continue
        kind, _, data = line[i + len(MARK) + 1:].partition(" ")
        if kind == "box":
            box = json.loads(data)
        elif kind == "output":
            outputs.append(json.loads(data))
    return box, outputs


# ---------------------------------------------------------------------- the report
def pct(xs: list[float], q: float):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def summarize(tag: str, outputs: list[dict], run_s: float, wall_s: float, errors: int) -> dict:
    """The same metrics, computed the same way, as main's report."""
    meas = [o for o in outputs if o["measured"]] or outputs
    tpots = [o["tpot_ms"] for o in meas if o["tpot_ms"] is not None]
    med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
    return {
        "cell": tag, "run_s": run_s, "run_tok_s": sum(o["output_tokens"] for o in outputs) / max(run_s, 1e-9),
        "tok_s": sum(o["output_tokens"] for o in meas) / max(wall_s, 1e-9),
        "tpot_ms_p50": med(tpots), "tpot_ms_p95": pct(tpots, 0.95),
        "ttft_s_mean": statistics.mean(o["ttft_s"] for o in meas) if meas else None,
        "e2e_s_p50": med([o["e2e_s"] for o in meas]), "e2e_s_p95": pct([o["e2e_s"] for o in meas], 0.95),
        "wait_s_mean": statistics.mean(o["wait_s"] for o in meas) if meas else None,
        "requests": len(outputs), "errors": errors}


def table(rows: list[dict]) -> str:
    cols = [("cell", "cell", "{}"), ("run s", "run_s", "{:.1f}"), ("run tok/s", "run_tok_s", "{:.2f}"),
            ("tok/s", "tok_s", "{:.2f}"), ("tpot p50 ms", "tpot_ms_p50", "{:.1f}"), ("tpot p95 ms", "tpot_ms_p95", "{:.1f}"),
            ("ttft s", "ttft_s_mean", "{:.2f}"), ("e2e p50 s", "e2e_s_p50", "{:.2f}"), ("e2e p95 s", "e2e_s_p95", "{:.2f}"),
            ("wait s", "wait_s_mean", "{:.2f}"), ("requests", "requests", "{}"), ("err", "errors", "{}")]
    lines = [" | ".join(c[0] for c in cols), " | ".join("---" for _ in cols)]
    for r in rows:
        lines.append(" | ".join("-" if r[k] is None else fmt.format(r[k]) for _, k, fmt in cols))
    return "\n".join(lines)


# ---------------------------------------------------------------------- the run
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", required=True, help="labeled requests JSON (make_requests.py)")
    ap.add_argument("--volume", required=True, choices=[f"coopmoe-{s}" for s in MODELS],
                    help="the model's NFS volume, made by stage.py on main; it decides the model")
    ap.add_argument("--num", type=int, default=64, help="requests per box (default 64)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="requests each box serves at a time, batched (default 4, as on main)")
    args = ap.parse_args(argv)

    short = args.volume[len("coopmoe-"):]
    with open(args.requests, encoding="utf-8") as f:
        picked = select(json.load(f), args.num)
    log(f"{MODELS[short]}: {len(picked)} requests ({args.num} per box), {args.concurrency} at a time per box")
    for b in (0, 1):
        log(f"box {b}: " + ", ".join(f"{c} {n}" for c, n in Counter(r["category"] for r in picked if r["box"] == b).items()))

    out = os.path.join(REPO, "results", f"{short}-{time.strftime('%Y%m%d-%H%M%S')}-baseline")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "requests.json"), "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=1)
    wfs = [render(short, b, [r for r in picked if r["box"] == b], args.concurrency) for b in (0, 1)]
    wids, finals = run_both(wfs, [os.path.join(out, f"box{b}.fz") for b in (0, 1)])

    boxes, rows, outputs = [], [], []
    for b in (0, 1):
        job_log = fuzzball("workflow", "log", wids[b], "run", check=False)
        with open(os.path.join(out, f"box{b}.log"), "w") as f:
            f.write(job_log)
        box, outs = collect(job_log)
        if finals[b] != "STAGE_STATUS_FINISHED" or box is None:
            log(f"box {b}: workflow ended {finals[b]} without a result; see {out}/box{b}.log")
            continue
        boxes.append(box)
        outputs += outs
        rows.append(summarize(f"hf_box{b}", outs, box["run_s"], box["wall_s"], box["errors"]))
    if len(boxes) == 2:  # the pair: as if both boxes had started together
        rows.append(summarize("hf", outputs, max(x["run_s"] for x in boxes), max(x["wall_s"] for x in boxes),
                              sum(x["errors"] for x in boxes)))
    if not rows:
        return
    with open(os.path.join(out, "outputs.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"model": MODELS[short], "boxes": boxes, "rows": rows}, f, indent=1)
    report = f"# {MODELS[short]} (baseline: stock HF transformers, each box on its own)\n\n{table(rows)}\n"
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(report)
    print("\n" + report)
    log(f"done: {out}")


if __name__ == "__main__":
    main()
