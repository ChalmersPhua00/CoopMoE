#!/usr/bin/env python3
"""Run labeled requests through CoopMoE on the two PGX boxes, set up by Fuzzball, and report.

    python driver.py --requests data/requests.json --volume coopmoe-qwen3-235b-a22b
    python driver.py --requests data/requests.json --volume coopmoe-qwen3-235b-a22b --num 16 --concurrency 8

Run it anywhere the `fuzzball` CLI is logged in to the cluster.

--volume is a model's NFS volume, made once beforehand by stage.py (team1/coopmoe-<model>).
Each volume holds one model's expert store, so its name says which model to run. Then:
  1. select   --num requests for each box (default 64, so 128 in all): box 0's from its
              categories, box 1's from its own, spread evenly over each side's categories.
              Each box serves --concurrency of them at a time, batched (default: the run
              config's engine.max_sessions, 4); more needs more memory for KV caches
  2. render   fuzzball/coopmoe.fz: one two-node job that carries this repo's code and the
              selected requests (fuzzball/bootstrap.sh runs on each node)
  3. submit   fuzzball workflow validate + start
  4. inside   both nodes copy the expert store from the volume to their own NVMe and
     the job  start their engine; PGX-A runs the requests through the scheduler with the model's
              run config as it is (configs/), and prints the results into the job log
  5. collect  follows the job log, then writes the results to results/<model>-<time>/:
              report.md (throughput, TTFT, TPOT, latency, where experts came from), <setup>.json,
              <setup>_outputs.jsonl (one line per request), requests.json, coopmoe.fz, job.log

Everything else is fixed in SETTINGS below. To change the engines or the scheduler (cache.gb,
link, router mode), edit the model's run config. experiment() is the same thing as a function,
for scripts that compare several setups.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from coopmoe.config import load_run_config  # noqa: E402
from coopmoe.report import render, write_cell  # noqa: E402
from coopmoe.workload import select  # noqa: E402

# The two models the configs are written for: short name -> (Hub repo, run config).
MODELS = {
    "qwen3-30b-a3b": ("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "configs/qwen3_30b_a3b_fp8.json"),
    "qwen3-235b-a22b": ("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "configs/qwen3_235b_a22b_fp8.json"),
}
# Fixed for every job (stage.py uses them too).
SETTINGS = dict(
    provisioner="team1",                             # where the model volumes live
    data_root="/data/coopmoe",                       # the model's NFS volume, inside the job
    local_root="/scratch", scratch="team1-scratch",  # each node's own NVMe, wiped per job: the store copy (~226 GiB for 235B)
    image="docker://nvcr.io/nvidia/pytorch:25.09-py3",
    gpus=1, cores=20, memory="120GiB",               # a whole GB10 box (unified memory: a hard limit)
    timeout="24h",
    boxes=["run-0:29600", "run-1:29600"],            # the engines, by the job's hostnames
    fuzzball="fuzzball", poll=15.0,                  # the CLI, seconds between status checks
)
# What the job needs of this repo (shipped inside the workflow).
CODE = ["coopmoe", "configs", "setup.sh", "env.sh", "requirements.txt"]
MARK = "@@coopmoe"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
DONE = {"STAGE_STATUS_FINISHED", "STAGE_STATUS_FAILED", "STAGE_STATUS_CANCELED"}
S = argparse.Namespace(**SETTINGS)


def log(msg: str) -> None:
    print(f"[driver {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------- models and volumes
def volume_name(short: str) -> str:
    """A model's NFS volume, made by stage.py: qwen3-235b-a22b -> coopmoe-qwen3-235b-a22b."""
    return f"coopmoe-{short}"


def volume_model(volume: str) -> str:
    """coopmoe-qwen3-235b-a22b -> qwen3-235b-a22b."""
    for short in MODELS:
        if volume == volume_name(short):
            return short
    raise SystemExit(f"{volume!r} is not a model volume made by stage.py; expected one of: "
                     + ", ".join(volume_name(s) for s in MODELS))


def store_paths(repo: str) -> dict:
    """Where things live inside the jobs."""
    name = repo.rpartition("/")[2].lower()
    data, local = S.data_root, S.local_root
    return {"data": data, "local": local, "nfs_store": f"{data}/store/{name}",
            "store": f"{local}/store/{name}", "hf": f"{data}/hf/{name}"}


# ---------------------------------------------------------------------- the workflow
def code_archive() -> str:
    """This repo's code as base64 tar.gz."""
    buf = io.BytesIO()
    skip = lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti  # noqa: E731
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in CODE:
            t.add(os.path.join(REPO, name), arcname=name, filter=skip)
    return base64.encodebytes(buf.getvalue()).decode()


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


def bootstrap(short: str, sets: str = "", boxes: str = "''", run_args: str = "") -> str:
    """fuzzball/bootstrap.sh for this model."""
    repo, config = MODELS[short]
    p, q = store_paths(repo), shlex.quote
    with open(os.path.join(REPO, "fuzzball", "bootstrap.sh")) as f:
        return fill(f.read(), {
            "MODEL": q(repo), "NFS_STORE": q(p["nfs_store"]), "STORE": q(p["store"]), "HF_DIR": q(p["hf"]),
            "CONFIG": q(config), "BIND1": "0.0.0.0", "SETS": sets, "RESULTS": f"{p['data']}/results",
            "BOXES": boxes, "RUN_ARGS": run_args})


def job_values(short: str, timeout: str) -> dict:
    """The placeholders both workflows share."""
    p = store_paths(MODELS[short][0])
    return {"CODE": block(code_archive()), "DATA": p["data"], "LOCAL": p["local"], "TIMEOUT": timeout,
            "VOLUME_USE": S.provisioner, "VOLUME": volume_name(short), "IMAGE": S.image,
            "CORES": str(S.cores), "MEMORY": S.memory, "SCRATCH": S.scratch}


def render_workflow(short: str, picked: list[dict], cells: list) -> str:
    p, q = store_paths(MODELS[short][0]), shlex.quote
    sets = [f"store_dir={p['store']}", f"cluster.addrs={json.dumps(S.boxes)}"]  # for engine rank 1
    run_args = [a for c in cells for a in ("--cell", " ".join([c[0], *c[1]]))]
    boot = bootstrap(short, " ".join(q(s) for s in sets), q(",".join(S.boxes)), " ".join(q(a) for a in run_args))
    requests = "[\n" + ",\n".join(json.dumps(r) for r in picked) + "\n]"  # ASCII: safe in a YAML block
    with open(os.path.join(REPO, "fuzzball", "coopmoe.fz")) as f:
        return fill(f.read(), {"BOOTSTRAP": block(boot), "REQUESTS": block(requests), "GPUS": str(S.gpus),
                               **job_values(short, S.timeout)})


# ---------------------------------------------------------------------- fuzzball
def fuzzball(*cli: str, check: bool = True) -> str:
    p = subprocess.run([S.fuzzball, *cli], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise SystemExit(f"fuzzball {' '.join(cli)} failed ({p.returncode}):\n{p.stdout}{p.stderr}")
    return p.stdout + p.stderr


def status(wid: str) -> str:
    m = re.search(r"Status:\s*(\S+)", fuzzball("workflow", "describe", wid, check=False))
    return m.group(1) if m else "?"


def follow(wid: str, job: str) -> str:
    """Streams the job's log until the workflow ends. -> its final status."""
    last = None
    while True:
        st = status(wid)
        if st != last:
            log(f"workflow {wid}: {st}")
            last = st
        if st in DONE:
            return st
        if st == "STAGE_STATUS_STARTED":
            p = subprocess.Popen([S.fuzzball, "workflow", "log", wid, job, "-f"], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                if MARK not in line:
                    print(line, end="", flush=True)
            p.wait()
        time.sleep(S.poll)


def submit(wf: str, path: str, job: str) -> tuple[str, str]:
    """Writes, validates, starts and follows a workflow. -> (workflow id, final status)."""
    with open(path, "w") as f:
        f.write(wf)
    log(f"workflow: {path} ({len(wf) / 1e3:.0f} kB)")
    fuzzball("workflow", "validate", path)
    started = fuzzball("workflow", "start", path)
    m = UUID.search(started)
    if not m:
        raise SystemExit(f"no workflow id in: {started}")
    wid = m.group(0)
    log(f"started workflow {wid}; Ctrl-C stops it")
    try:
        return wid, follow(wid, job)
    except KeyboardInterrupt:
        log(f"stopping workflow {wid}")
        fuzzball("workflow", "stop", wid, check=False)
        raise


def collect(job_log: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """'@@coopmoe' lines of the job log -> (results per setup, outputs per setup)."""
    cells, outputs = [], {}
    for line in job_log.splitlines():
        i = line.find(MARK + " ")
        if i < 0:
            continue
        kind, _, payload = line[i + len(MARK) + 1:].partition(" ")
        if kind == "cell":
            cells.append(json.loads(payload))
        elif kind == "output":
            o = json.loads(payload)
            outputs.setdefault(o["cell"], []).append(o)
    return cells, outputs


# ---------------------------------------------------------------------- the experiment
def experiment(volume: str, requests: str, num: int = 64, concurrency: int | None = None,
               cells: list | None = None) -> tuple[str, list[dict]]:
    """One Fuzzball job: num requests per box from the requests file, run on the model in volume,
    each box serving `concurrency` of them at a time, batched (default: the run config's
    engine.max_sessions). cells: [(tag, [KEY=VALUE, ...]), ...] setups to run one after another in
    that job; default is the run config as it is. -> (results directory, results per setup)."""
    short = volume_model(volume)
    repo, config = MODELS[short]
    cfg = load_run_config(os.path.join(REPO, config), [])
    log(f"volume {S.provisioner}/{volume} -> model {repo}")
    with open(requests, encoding="utf-8") as f:
        picked = select(json.load(f), cfg.router.box_categories, 2 * num)
    sides = cfg.router.box_categories
    log(f"{len(picked)} requests ({num} per box) from {requests}: "
        + ", ".join(f"{c} {n}" for c, n in Counter(r["category"] for r in picked).items()))
    log(f"box 0 serves {', '.join(sides[0])}; box 1 serves {', '.join(sides[1])}")
    conc = cfg.engine.max_sessions if concurrency is None else concurrency
    if conc < 1:
        raise SystemExit(f"concurrency must be at least 1, got {conc}")
    log(f"each box serves {conc} requests at a time")
    if cells is None:
        mode, link = cfg.router.mode, cfg.link.enabled
        cells = [(f"{'linked' if link else 'indep'}_{'cat' if mode == 'category' else 'rr'}", [])]
    # Both engines and the scheduler take it with each setup's reset; a setup may still override it.
    per_box = [f"engine.max_sessions={conc}", f"engine.max_batch={conc}"]
    cells = [(tag, per_box + list(overrides)) for tag, overrides in cells]

    out = os.path.join(REPO, "results", f"{short}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "requests.json"), "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=1)
    wid, final = submit(render_workflow(short, picked, cells), os.path.join(out, "coopmoe.fz"), "run")
    job_log = fuzzball("workflow", "log", wid, "run", check=False)
    with open(os.path.join(out, "job.log"), "w") as f:
        f.write(job_log)
    results, outputs = collect(job_log)
    for res in results:
        write_cell(out, res, outputs.get(res["tag"], []))
        res["turns"] = outputs.get(res["tag"], [])
    if final != "STAGE_STATUS_FINISHED" or len(results) < len(cells):
        log(f"workflow ended {final} with {len(results)} of {len(cells)} setups; see {out}/job.log")
    if results:
        with open(os.path.join(out, "report.md"), "w") as f:
            f.write(f"# {repo}\n\n" + render(results, os.path.join(out, "overlap.csv")))
    return out, results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", required=True, help="labeled requests JSON (make_requests.py)")
    ap.add_argument("--volume", required=True, choices=[volume_name(s) for s in MODELS],
                    help="the model's NFS volume, made by stage.py; it decides the model")
    ap.add_argument("--num", type=int, default=64, help="requests per box (default 64)")
    ap.add_argument("--concurrency", type=int,
                    help="requests each box serves at a time, batched (default: the run config's, 4)")
    args = ap.parse_args(argv)

    out, results = experiment(args.volume, args.requests, args.num, args.concurrency)
    if not results:
        return
    with open(os.path.join(out, "report.md")) as f:
        print("\n" + f.read())
    for r in results[-1]["turns"][:4]:
        print(f"--- {r['id']} ({r['category']}, box {r['box']}) -> {r['output'][:300]!r}")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
