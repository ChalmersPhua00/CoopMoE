"""Command line: python -m coopmoe <command> ...  (run from the project root). fuzzball/bootstrap.sh
calls these inside the Fuzzball jobs that stage.py and driver.py submit:

  stage    download a checkpoint and repack it into an expert store, once (stage.py's job)
  engine   one box: serves its requests and its peer (both nodes of driver.py's job)
  run      the experiment on PGX-A: engine, requests, report (node 0 of driver.py's job)
"""
from __future__ import annotations

import argparse
import json
import sys

from coopmoe.config import load_run_config


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_stage(args) -> None:
    from coopmoe.store.convert import stage

    stage(args.model, args.store, args.hf_dir, log=_log)


def cmd_engine(args) -> None:
    from coopmoe.engine import Box

    cfg = load_run_config(args.config, args.overrides)
    if len(cfg.cluster.addrs) != 2:
        raise SystemExit("cluster.addrs must list exactly two boxes")
    box = Box(cfg, args.rank, _log)
    port = box.serve(args.host)
    _log(f"[engine{args.rank}] ready on {args.host}:{port} {json.dumps(box.describe())}")
    try:
        box.stopped.wait()
    except KeyboardInterrupt:
        box.close()
    _log(f"[engine{args.rank}] stopped")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:  # its own argument parser
        from coopmoe.run import main as run_main

        return run_main(argv[1:])
    ap = argparse.ArgumentParser(prog="coopmoe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stage", help="download + repack into a store (skipped when present; safe on shared storage)")
    p.add_argument("--model", required=True, help="Hub repo or local checkpoint directory")
    p.add_argument("--store", required=True)
    p.add_argument("--hf-dir", help="where to download (default: the Hub cache)")
    p.set_defaults(fn=cmd_stage)

    p = sub.add_parser("engine", help="one box")
    p.add_argument("--config", required=True, help="run config JSON (configs/*.json)")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE",
                   help="config overrides, e.g. store_dir=... cluster.addrs=[...]")
    p.set_defaults(fn=cmd_engine)

    sub.add_parser("run", help="the experiment on PGX-A (see python -m coopmoe run --help)")
    args = ap.parse_args(argv)
    args.fn(args)
