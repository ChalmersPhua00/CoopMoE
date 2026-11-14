#!/usr/bin/env python3
"""Give one model its own NFS volume and build the model's expert store on it. Once per model.

    python stage.py --model qwen3-235b-a22b --size 350GiB
    python stage.py --model qwen3-30b-a3b --size 80GiB

Run it anywhere the `fuzzball` CLI is logged in to the cluster, before the first driver.py run
with that model. It does two things, then stops:

1. volume  team1/coopmoe-<model> is created at --size if missing, grown to --size if smaller,
           never shrunk. Size it for the store (~243 GB for 235B, ~31 GB for 30B) plus results.
2. stage   submits fuzzball/stage.fz: one node downloads the checkpoint to its ephemeral disk
           (/local) and builds the expert store on that volume (/data/coopmoe/store/<model>),
           then waits for it to finish.

It starts no experiment: driver.py --volume coopmoe-<model> does that, and reads the model from the name.
Running it again is harmless: the volume is left as it is, and a finished store is skipped.
stage.fz (what was submitted) and stage.log go to results/stage-<model>-<time>/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

from driver import MODELS, REPO, S, block, bootstrap, fill, fuzzball, job_values, log, store_paths, submit, volume_name

TIMEOUT = "12h"  # the download and the build; every other setting is driver.py's SETTINGS


UNITS = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
         "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}


def to_bytes(size) -> int | None:
    """'350GiB', '200 GB', 214748364800 or '214748364800' -> bytes; None if it is not a size."""
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        return int(size)
    m = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]?I?B)?\s*", str(size), re.IGNORECASE)
    if not m or (m.group(2) or "B").upper() not in UNITS:
        return None
    return int(float(m.group(1)) * UNITS[(m.group(2) or "B").upper()])


def current_size(info: dict) -> int | None:
    """The largest size/capacity field anywhere in `fuzzball volume info -o json` (its name is not known)."""
    found = []

    def walk(x):
        for k, v in (x.items() if isinstance(x, dict) else enumerate(x) if isinstance(x, list) else ()):
            if isinstance(v, (dict, list)):
                walk(v)
            elif isinstance(k, str) and re.search(r"size|capacity", k, re.IGNORECASE) and to_bytes(v):
                found.append(to_bytes(v))

    walk(info)
    return max(found) if found else None


def ensure_volume(name: str, size: str) -> None:
    """Creates the volume at size if missing, grows it to size if smaller. Never shrinks."""
    vol = f"{S.provisioner}/{name}"
    out = fuzzball("volume", "info", S.provisioner, name, "-o", "json", check=False)
    if re.search(r"not ?found|does not exist|no such", out, re.IGNORECASE):
        log(f"volume {vol} does not exist: creating it at {size}")
        fuzzball("volume", "create", S.provisioner, name, "--size", size)
        return
    try:
        have = current_size(json.loads(out))
    except ValueError:
        raise SystemExit(f"fuzzball volume info {S.provisioner} {name} failed:\n{out}")
    if have is not None and have >= to_bytes(size):
        log(f"volume {vol} is {have / 2**30:.1f} GiB, already >= {size}: left as is")
        return
    log(f"volume {vol}: {'size unknown' if have is None else f'{have / 2**30:.1f} GiB'} -> {size}")
    fuzzball("volume", "update", S.provisioner, name, "--size", size)


def render_stage(short: str) -> str:
    with open(os.path.join(REPO, "fuzzball", "stage.fz")) as f:
        return fill(f.read(), {"BOOTSTRAP": block(bootstrap(short)), **job_values(short, TIMEOUT)})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=str.lower, choices=list(MODELS), help="the model to stage")
    ap.add_argument("--size", required=True, help="the model's volume size, e.g. 350GiB (grown to it, never shrunk)")
    args = ap.parse_args(argv)
    if not to_bytes(args.size):
        ap.error(f"--size {args.size!r} is not a size (e.g. 350GiB, 400GB)")

    short, name = args.model, volume_name(args.model)
    repo = MODELS[short][0]
    ensure_volume(name, args.size)

    out = os.path.join(REPO, "results", f"stage-{short}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    log(f"staging {repo} -> {S.provisioner}/{name}:{store_paths(repo)['nfs_store']}")
    wid, final = submit(render_stage(short), os.path.join(out, "stage.fz"), "stage")
    with open(os.path.join(out, "stage.log"), "w") as f:
        f.write(fuzzball("workflow", "log", wid, "stage", check=False))
    log(f"stage workflow ended {final}; log in {out}/stage.log")
    if final == "STAGE_STATUS_FINISHED":
        log(f"store is ready; next: python driver.py --requests data/requests.json --volume {name}")


if __name__ == "__main__":
    main()
