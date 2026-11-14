#!/usr/bin/env python3
"""One PGX box serving its requests with stock Hugging Face transformers 5.23.0. This is the
whole baseline: it runs inside the Fuzzball job that driver.py starts on each box.

    python baseline.py --checkpoint DIR --offload DIR --requests requests.json --box 0 \
        --concurrency 4 --memory 120GiB

  1. load   AutoModelForCausalLM.from_pretrained on the original checkpoint, device_map="auto".
            What does not fit in the box's memory HF offloads to --offload (this box's NVMe) and
            reads back on every forward. All of Qwen3-30B fits; Qwen3-235B does not.
  2. serve  the requests through HF's continuous batching (the loop `transformers serve` uses),
            --concurrency at a time. Each request is one prompt, then exactly MAX_NEW generated
            tokens (end of turn ignored), greedy.
  3. print  one '@@baseline output <json>' line per request and one '@@baseline box <json>' line
            for the box: driver.py reads them from the job log.

Two settings HF needs on the GB10, neither of them a feature:
  experts_implementation="eager"   HF's default experts path (grouped_mm) calls torch._grouped_mm,
                                   which refuses compute capability 12.1
  FineGrainedFP8Config(dequantize=True)
                                   the FP8 checkpoint as BF16 weights: what CoopMoE computes in,
                                   and no kernel download from the Hub (HF's FP8 matmuls come
                                   from kernels-community/finegrained-fp8)

The numbers below are the ones CoopMoE's run configs use on main (configs/*.json), so both
branches do the same work and measure it the same way.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import threading
import time
from concurrent.futures import Future

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, ContinuousBatchingConfig,
                          FineGrainedFP8Config, GenerationConfig)

MAX_SEQ = 8192       # prompt + generated tokens per request (main: engine.max_seq)
MAX_NEW = 128        # generated tokens per request (main: bench.max_new_tokens)
WARMUP = 4           # requests before measuring (main: bench.warmup_sessions = 8 over two boxes)
PAGE = 256           # tokens per KV page (main: engine.PAGE)
HEADROOM_GIB = 12.0  # kept out of the weight budget besides KV: CUDA context, activations, slack
MARK = "@@baseline"


def log(msg: str) -> None:
    print(f"[baseline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(kind: str, obj: dict) -> None:
    print(f"{MARK} {kind} {json.dumps(obj, ensure_ascii=False)}", flush=True)


def gib(size: str) -> float:
    """'120GiB' (the job's memory request) -> 120.0."""
    m = re.fullmatch(r"\s*([0-9.]+)\s*GiB\s*", size)
    if not m:
        raise SystemExit(f"--memory must look like 120GiB, got {size!r}")
    return float(m.group(1))


def kv_bytes(cfg) -> int:
    """One request's KV cache at MAX_SEQ tokens, in BF16."""
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * MAX_SEQ * 2


def make_room(nbytes: int) -> None:
    """On the GB10 (unified memory) CUDA's "free" leaves out the page cache, although an
    allocation reclaims it. HF sizes its KV cache against that "free", so after loading a
    checkpoint it can refuse to start. Allocating and touching nbytes once makes the kernel
    reclaim that much; freeing it leaves it free for HF's check."""
    free, _ = torch.cuda.mem_get_info()
    if free >= nbytes:
        return
    try:
        x = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        x.fill_(0)
        torch.cuda.synchronize()
        del x
    except torch.OutOfMemoryError:
        log(f"could not reclaim {nbytes / 2**30:.1f} GiB for the KV cache")
    torch.cuda.empty_cache()
    log(f"free memory {free / 2**30:.1f} -> {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB (reclaimed page cache)")


def load(checkpoint: str, offload: str, weights_gib: float):
    os.makedirs(offload, exist_ok=True)
    log(f"loading {checkpoint}: device_map=auto, {weights_gib:.1f} GiB of weights on the GPU, the rest offloaded to {offload}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: f"{weights_gib:.1f}GiB", "cpu": "0GiB"}, offload_folder=offload,
        attn_implementation="sdpa", experts_implementation="eager",
        quantization_config=FineGrainedFP8Config(dequantize=True))
    placed: dict = {}
    for where in (getattr(model, "hf_device_map", None) or {"": "cuda:0"}).values():
        placed[str(where)] = placed.get(str(where), 0) + 1
    log(f"loaded in {time.time() - t0:.0f} s; modules per device: {json.dumps(placed)}")
    return model.eval(), placed, time.time() - t0


def start_batching(model, concurrency: int):
    """HF continuous batching: greedy, fixed length, KV for `concurrency` requests of MAX_SEQ
    tokens, a whole prompt in one forward. The same settings as a CoopMoE box."""
    gen = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW, eos_token_id=-1)
    cbc = ContinuousBatchingConfig(page_size=PAGE, num_blocks=concurrency * math.ceil(MAX_SEQ / PAGE),
                                   max_batch_tokens=MAX_SEQ, max_requests_per_batch=concurrency,
                                   use_cuda_graph=False, use_async_batching=False, allow_block_sharing=False,
                                   auto_switch_to_flash=False)
    # HF's KV footprint is ~1.5x the KV itself (masks, batch tensors); it must fit in 0.9 x free.
    make_room(int((1.6 * concurrency * kv_bytes(model.config) + 2**30) / 0.9))
    cb = model.init_continuous_batching(generation_config=gen, continuous_batching_config=cbc)
    cb.start()
    t0 = time.time()
    while cb.batch_processor is None:  # set by the loop thread once its KV cache exists
        if not cb.is_running():
            raise SystemExit("HF's generation loop failed to start (its error is logged above)")
        if time.time() - t0 > 600:
            raise SystemExit("HF's generation loop did not start within 600 s")
        time.sleep(0.05)
    return cb


def serve(cb, tok, requests: list[dict], box: int, concurrency: int) -> dict:
    """All requests are queued at the start; `concurrency` are in flight at a time. Timings are
    taken exactly as on main: wait = queued -> sent, e2e = sent -> finished, ttft and tpot from
    HF's own per-token timestamps."""
    eos = set(tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]))
    waiting: dict[str, Future] = {}
    queue = list(requests)
    lock = threading.Lock()
    state = {"done": 0, "t_measure": None, "errors": 0}
    stop = threading.Event()

    def results() -> None:  # hands each finished request back to the worker waiting for it
        while not stop.is_set():
            out = cb.get_result(timeout=0.05)
            if out is not None and out.is_finished():
                fut = waiting.pop(out.request_id, None)
                if fut is not None:
                    fut.set_result(out)

    def one(r: dict) -> None:
        t0 = time.perf_counter()
        ids = tok.apply_chat_template([{"role": "user", "content": r["request"]}], add_generation_prompt=True,
                                      tokenize=True, return_dict=False)
        if len(ids) + MAX_NEW > MAX_SEQ:
            raise ValueError(f"prompt of {len(ids)} tokens + {MAX_NEW} new exceeds {MAX_SEQ}")
        fut: Future = Future()
        waiting[r["id"]] = fut
        cb.add_request(ids, request_id=r["id"], max_new_tokens=MAX_NEW, record_timestamps=True, eos_token_id=-1)
        out = fut.result()
        t1 = time.perf_counter()
        if out.error:
            raise RuntimeError(out.error)
        times, toks = out.timestamps, out.generated_tokens
        cut = next((i for i, t in enumerate(toks) if t in eos), len(toks))
        with lock:
            measured = state["t_measure"] is not None and t0 >= state["t_measure"]
        emit("output", {
            "id": r["id"], "category": r["category"], "box": box, "prompt_tokens": len(ids),
            "output_tokens": len(toks), "ttft_s": round(times[0] - out.created_time, 4),
            "tpot_ms": None if len(toks) < 2 else round((times[-1] - times[0]) / (len(toks) - 1) * 1e3, 2),
            "e2e_s": round(t1 - t0, 4), "wait_s": round(t0 - t_start, 4), "measured": measured,
            "request": r["request"], "output": tok.decode(toks[:cut])})

    def worker() -> None:
        while True:
            with lock:
                if not queue:
                    return
                r = queue.pop(0)
            try:
                one(r)
            except Exception as exc:
                log(f"{r['id']} failed: {exc}")
                with lock:
                    state["errors"] += 1
            with lock:
                state["done"] += 1
                if state["t_measure"] is None and state["done"] >= WARMUP:
                    state["t_measure"] = time.perf_counter()
                    log(f"warm-up done after {state['done']} requests; measuring")

    threading.Thread(target=results, daemon=True).start()
    t_start = time.perf_counter()
    if WARMUP <= 0:
        state["t_measure"] = t_start
    workers = [threading.Thread(target=worker) for _ in range(concurrency)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    t_end = time.perf_counter()
    stop.set()
    # run_s: first request sent -> last request finished. wall_s: the same, after the warm-up.
    return {"run_s": t_end - t_start, "wall_s": t_end - (state["t_measure"] or t_start), "errors": state["errors"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="the original HF checkpoint directory")
    ap.add_argument("--offload", required=True, help="where HF offloads the weights that do not fit (this box's NVMe)")
    ap.add_argument("--requests", required=True, help="this box's requests (driver.py selects them)")
    ap.add_argument("--box", type=int, required=True, choices=(0, 1))
    ap.add_argument("--concurrency", type=int, required=True, help="requests served at a time, batched")
    ap.add_argument("--memory", required=True, help="the job's memory request, e.g. 120GiB")
    args = ap.parse_args()

    with open(args.requests, encoding="utf-8") as f:
        requests = json.load(f)
    cfg = AutoConfig.from_pretrained(args.checkpoint)
    weights_gib = gib(args.memory) - 1.6 * args.concurrency * kv_bytes(cfg) / 2**30 - HEADROOM_GIB
    model, placed, load_s = load(args.checkpoint, args.offload, weights_gib)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    cb = start_batching(model, args.concurrency)
    log(f"box {args.box} on {socket.gethostname()}: {len(requests)} requests, {args.concurrency} at a time")
    res = serve(cb, tok, requests, args.box, args.concurrency)
    cb.stop(block=True, hard_stop=True)
    emit("box", dict(res, box=args.box, host=socket.gethostname(), requests=len(requests), load_s=load_s,
                     concurrency=args.concurrency, weights_gib_on_gpu=weights_gib, device_map=placed,
                     transformers=__import__("transformers").__version__))


if __name__ == "__main__":
    main()
