#!/usr/bin/env python3
"""Run labeled requests through CoopMoE on the two PGX boxes, set up by Fuzzball, and report.

    python driver.py --requests data/requests.json --volume coopmoe-qwen3-235b-a22b
    python driver.py --requests data/requests.json --volume coopmoe-qwen3-235b-a22b --num 16 --concurrency 8

Run it anywhere the `fuzzball` CLI is logged in to the cluster.

--volume is a model's NFS volume, made once beforehand by stage.py (team1/coopmoe-<model>).
Each volume holds one model's expert store, so its name says which model to run. Then:
  1. select   --num requests for each box (default 64, so 128 in all), from the requests
              make_requests.py assigned to that box ("box"), spread evenly over its categories.
              Each box serves --concurrency of them at a time, batched (default: the run
              config's engine.max_sessions, 4); more needs more memory for KV caches
  2. render   fuzzball/coopmoe.fz twice, one single-node job per box (a multinode job cannot
              mount a node's NVMe): box 0 runs the experiment, box 1 serves as its peer. Each
              carries this repo's code; box 0's also carries the selected requests
  3. submit   fuzzball workflow validate + start, both at once
  4. inside   each job writes its node's CX-7 address to the NFS volume and waits for the
     the jobs other's (fuzzball/bootstrap.sh, rendezvous), copies the expert store from the volume
              to its own NVMe and starts its engine; box 0 runs the requests through the
              scheduler with the model's run config as it is (configs/), prints the results
              into its job log and stops both engines
  5. collect  follows both job logs, then writes the results to results/<model>-<time>/:
              report.md (throughput, TTFT, TPOT, latency, where experts came from), <setup>.json,
              <setup>_outputs.jsonl (one line per request), requests.json, box0.fz, box1.fz,
              job.log (box 0), box1.log
     If box 1's job does not start (Fuzzball sometimes puts both jobs' scratch volumes on one
     node), box 0 gives up after SETTINGS rendezvous seconds and both jobs are submitted again.

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
import threading
import time
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from coopmoe.config import load_run_config  # noqa: E402
from coopmoe.report import render, write_cell  # noqa: E402
from make_requests import select  # noqa: E402

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
    port=29600, rendezvous=120,                      # the engines' port; seconds a box waits for the other box's job
    cx7_net="10.10.11.1",                            # an address on the CX-7 network (the NFS server): picks each box's CX-7 address
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
def code_archive(requests: list[dict]) -> str:
    """This repo's code and the run's requests (requests.json) as base64 tar.gz. Base64 because
    Fuzzball treats a double brace anywhere in a workflow file as a template, and requests
    contain them."""
    buf = io.BytesIO()
    skip = lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti  # noqa: E731
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in CODE:
            t.add(os.path.join(REPO, name), arcname=name, filter=skip)
        data = json.dumps(requests, ensure_ascii=False).encode()
        info = tarfile.TarInfo("requests.json")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
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


def bootstrap(short: str, run_id: str = "", run_args: str = "") -> str:
    """fuzzball/bootstrap.sh for this model. run_id names the directory on the NFS volume where
    the two boxes of a linked run find each other."""
    repo, config = MODELS[short]
    p, q = store_paths(repo), shlex.quote
    with open(os.path.join(REPO, "fuzzball", "bootstrap.sh")) as f:
        return fill(f.read(), {
            "MODEL": q(repo), "NFS_STORE": q(p["nfs_store"]), "STORE": q(p["store"]), "HF_DIR": q(p["hf"]),
            "CONFIG": q(config), "RESULTS": f"{p['data']}/results", "RUN_ARGS": run_args,
            "RDV": q(f"{p['data']}/rendezvous/{run_id}"), "PORT": str(S.port), "CX7_NET": S.cx7_net, "RDV_TIMEOUT": str(S.rendezvous)})


def job_values(short: str, timeout: str, requests: list[dict] = ()) -> dict:
    """The placeholders both workflows share."""
    p = store_paths(MODELS[short][0])
    return {"CODE": block(code_archive(list(requests))), "DATA": p["data"], "LOCAL": p["local"], "TIMEOUT": timeout,
            "VOLUME_USE": S.provisioner, "VOLUME": volume_name(short), "IMAGE": S.image,
            "CORES": str(S.cores), "MEMORY": S.memory, "SCRATCH": S.scratch}


def render_box(short: str, role: str, run_id: str, picked: list[dict], cells: list) -> str:
    """fuzzball/coopmoe.fz for one box. role: run (box 0: its engine and the experiment) or
    engine (box 1). picked: the requests (box 0's job runs them all through the scheduler);
    cells: [(tag, [KEY=VALUE, ...]), ...], for run."""
    q = shlex.quote
    run_args = [a for c in cells for a in ("--cell", " ".join([c[0], *c[1]]))]
    boot = bootstrap(short, run_id, " ".join(q(a) for a in run_args))
    with open(os.path.join(REPO, "fuzzball", "coopmoe.fz")) as f:
        return fill(f.read(), {"BOOTSTRAP": block(boot), "GPUS": str(S.gpus), "ROLE": role,
                               **job_values(short, S.timeout, picked)})


# ---------------------------------------------------------------------- fuzzball
def fuzzball(*cli: str, check: bool = True) -> str:
    p = subprocess.run([S.fuzzball, *cli], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise SystemExit(f"fuzzball {' '.join(cli)} failed ({p.returncode}):\n{p.stdout}{p.stderr}")
    return p.stdout + p.stderr


def status(wid: str) -> str:
    m = re.search(r"Status:\s*(\S+)", fuzzball("workflow", "describe", wid, check=False))
    return m.group(1) if m else "?"


def follow(wid: str, job: str, prefix: str = "") -> str:
    """Streams the job's log until the workflow ends. -> its final status."""
    last = None
    while True:
        st = status(wid)
        if st != last:
            log(f"{prefix}workflow {wid}: {st}")
            last = st
        if st in DONE:
            return st
        if st == "STAGE_STATUS_STARTED":
            p = subprocess.Popen([S.fuzzball, "workflow", "log", wid, job, "-f"], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                if MARK not in line:
                    print(prefix + line, end="", flush=True)
            p.wait()
        time.sleep(S.poll)


def start(wf: str, path: str) -> str:
    """Writes, validates and starts a workflow. -> its id."""
    with open(path, "w") as f:
        f.write(wf)
    log(f"workflow: {path} ({len(wf) / 1e3:.0f} kB)")
    fuzzball("workflow", "validate", path)
    started = fuzzball("workflow", "start", path)
    m = UUID.search(started)
    if not m:
        raise SystemExit(f"no workflow id in: {started}")
    log(f"started workflow {m.group(0)}; Ctrl-C stops it")
    return m.group(0)


def submit(wf: str, path: str, job: str) -> tuple[str, str]:
    """Writes, validates, starts and follows a workflow. -> (workflow id, final status)."""
    wid = start(wf, path)
    try:
        return wid, follow(wid, job)
    except KeyboardInterrupt:
        log(f"stopping workflow {wid}")
        fuzzball("workflow", "stop", wid, check=False)
        raise


def run_pair(wfs: list[str], paths: list[str], until_box0: bool) -> tuple[list[str], list]:
    """Starts one workflow per box, both at once, and follows both job logs. until_box0: the
    run is over when box 0's workflow ends (box 1 only serves, and is stopped if it is still
    there a minute later); otherwise it waits for both. Ctrl-C or a failed start stops whatever
    was started. -> (workflow ids, final statuses)."""
    wids: list[str] = []
    finals: list = [None, None]

    def watch(b: int) -> None:
        finals[b] = follow(wids[b], "run", prefix=f"[box{b}] ")

    try:
        for wf, path in zip(wfs, paths):
            wids.append(start(wf, path))
        threads = [threading.Thread(target=watch, args=(b,), daemon=True) for b in (0, 1)]
        for t in threads:
            t.start()
        while threads[0].is_alive():
            threads[0].join(timeout=1.0)
        t0 = time.time()
        while threads[1].is_alive():
            if until_box0 and time.time() - t0 > 60:
                log(f"box 0's workflow ended {finals[0]}; stopping box 1's workflow {wids[1]}")
                fuzzball("workflow", "stop", wids[1], check=False)
                until_box0 = False
            threads[1].join(timeout=1.0)
    except BaseException:
        for wid in wids:
            log(f"stopping workflow {wid}")
            fuzzball("workflow", "stop", wid, check=False)
        raise
    return wids, finals


def collect(job_log: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """'@@coopmoe <kind> <i>/<n> <base64 chunk>' lines of the job log (run.py, _emit) ->
    (results per setup, outputs per setup)."""
    cells, outputs = [], {}
    parts: list[str] = []
    for line in job_log.splitlines():
        i = line.find(MARK + " ")
        if i < 0:
            continue
        fields = line[i + len(MARK) + 1:].split(" ")
        if len(fields) != 3 or "/" not in fields[1]:
            continue  # same-node and other plain markers
        kind, pos, chunk = fields
        k, n = (int(x) for x in pos.split("/"))
        if k == 1:
            parts = []
        parts.append(chunk.strip())
        if k < n:
            continue
        obj = json.loads(base64.b64decode("".join(parts)).decode())
        if kind == "cell":
            cells.append(obj)
        elif kind == "output":
            outputs.setdefault(obj["cell"], []).append(obj)
    return cells, outputs


# ---------------------------------------------------------------------- the experiment
def plan(volume: str, requests: str, num: int, concurrency: int | None):
    """What a linked run and its baseline share: the model, its run config, the same selected
    requests and the per-box concurrency. -> (short name, run config, picked requests, concurrency)."""
    short = volume_model(volume)
    repo, config = MODELS[short]
    cfg = load_run_config(os.path.join(REPO, config), [])
    log(f"volume {S.provisioner}/{volume} -> model {repo}")
    with open(requests, encoding="utf-8") as f:
        picked = select(json.load(f), num)
    log(f"{len(picked)} requests ({num} per box, as make_requests.py assigned them) from {requests}")
    for b in (0, 1):
        log(f"box {b}: " + ", ".join(f"{c} {n}" for c, n in Counter(r["category"] for r in picked if r["box"] == b).items()))
    conc = cfg.engine.max_sessions if concurrency is None else concurrency
    if conc < 1:
        raise SystemExit(f"concurrency must be at least 1, got {conc}")
    log(f"each box serves {conc} requests at a time")
    return short, cfg, picked, conc


def experiment(volume: str, requests: str, num: int = 64, concurrency: int | None = None,
               cells: list | None = None) -> tuple[str, list[dict]]:
    """One linked run, as two single-node Fuzzball jobs (one per box) that talk over the CX-7
    link: num requests per box from the requests file, run on the model in volume, each box
    serving `concurrency` of them at a time, batched (default: the run config's
    engine.max_sessions). cells: [(tag, [KEY=VALUE, ...]), ...] setups to run one after another in
    those jobs; default is the run config as it is. -> (results directory, results per setup)."""
    short, cfg, picked, conc = plan(volume, requests, num, concurrency)
    repo = MODELS[short][0]
    if cells is None:
        mode, link = cfg.router.mode, cfg.link.enabled
        cells = [(f"{'linked' if link else 'indep'}_{'cat' if mode == 'category' else 'rr'}", [])]
    # Both engines and the scheduler take it with each setup's reset; a setup may still override it.
    per_box = [f"engine.max_sessions={conc}", f"engine.max_batch={conc}"]
    cells = [(tag, per_box + list(overrides)) for tag, overrides in cells]

    run_id = f"{short}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = os.path.join(REPO, "results", run_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "requests.json"), "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=1)
    # Fuzzball decides which node each job's scratch volume lands on. When both land on one
    # node the second job cannot start; the jobs notice (bootstrap.sh, rendezvous) and it is
    # submitted again.
    for attempt in range(1, 7):
        rdv = f"{run_id}-{attempt}"
        wfs = [render_box(short, "run", rdv, picked, cells), render_box(short, "engine", rdv, [], [])]
        wids, finals = run_pair(wfs, [os.path.join(out, f"box{b}.fz") for b in (0, 1)], until_box0=True)
        final = finals[0]
        job_log = fuzzball("workflow", "log", wids[0], "run", check=False)
        if f"{MARK} same-node" not in job_log:
            break
        log(f"attempt {attempt}: both jobs were placed on one node; submitting again")
    with open(os.path.join(out, "job.log"), "w") as f:
        f.write(job_log)
    with open(os.path.join(out, "box1.log"), "w") as f:
        f.write(fuzzball("workflow", "log", wids[1], "run", check=False))
    results, outputs = collect(job_log)
    for res in results:
        write_cell(out, res, outputs.get(res["tag"], []))
        res["turns"] = outputs.get(res["tag"], [])
    if final != "STAGE_STATUS_FINISHED" or len(results) < len(cells):
        log(f"box 0's workflow ended {final} with {len(results)} of {len(cells)} setups; see {out}/job.log and box1.log")
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
