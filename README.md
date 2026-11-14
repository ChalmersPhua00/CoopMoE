# CoopMoE baseline: stock Hugging Face transformers on each PGX box

This branch is only the baseline for CoopMoE (branch `main`). It serves the same requests on
the same two DGX Spark (GB10) boxes with stock Hugging Face transformers 5.23.0, each box on
its own: no expert cache, no expert store, no link between the boxes.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install datasets
python make_requests.py --out data/requests.json   # once; the same file main uses
python driver.py --requests data/requests.json --volume coopmoe-qwen3-235b-a22b --num 16 --concurrency 8
```

The `driver.py` command is the same as on `main`. The model's NFS volume
(`team1/coopmoe-<model>`) must exist: `stage.py` on `main` creates it.

## What runs

- **[make_requests.py](make_requests.py)** is identical to `main`'s. It assigns every request
  to a box (`"box"`), and `select()` in it picks `--num` per box, so both branches serve exactly
  the same requests on the same box.
- **[driver.py](driver.py)** starts [fuzzball/baseline.fz](fuzzball/baseline.fz) twice, one
  single-node job per box, follows both and writes `results/<model>-<time>-baseline/`.
- **[fuzzball/bootstrap.sh](fuzzball/bootstrap.sh)** (in each job) installs transformers next
  to the image's torch and copies the original checkpoint to the box's NVMe. The first job on
  a volume downloads the checkpoint to `hf/` on it once (~236 GB for 235B, so the volume needs
  that much room next to main's store).
- **[baseline.py](baseline.py)** (in each job) loads the checkpoint with
  `AutoModelForCausalLM.from_pretrained(device_map="auto")` and serves the box's requests
  through HF's continuous batching, `--concurrency` at a time, each for exactly 128 generated
  tokens. Weights that do not fit in memory are offloaded by HF to the box's NVMe and read
  back on every forward: all of 30B fits, 235B does not.

Two settings HF needs on the GB10, neither of them a feature: `experts_implementation="eager"`
(HF's default `grouped_mm` refuses compute capability 12.1), and the FP8 checkpoint
dequantized to BF16 at load (what CoopMoE computes in; HF's native FP8 needs a kernel download
from the Hub).

## Comparing with main

`report.md` has one row per box and one for the pair (`hf`), with the same columns as the
start of main's table:

- `run s`: first request sent → last request finished. For the pair it is the slower box's,
  as if both boxes had started together, which is what a linked run on main measures.
- `run tok/s`: every generated token over `run s`. **main's `run tok/s` over this one is the
  speedup.**
- `tok/s`, `tpot`, `ttft`, `e2e`, `wait`: after each box's 4 warm-up requests, as on main.

Batching settings, sequence limit, generation length and warm-up are constants at the top of
`baseline.py`, equal to main's run configs. Loading time is not in `run s` on either branch
(it is in each box's line in `results.json` as `load_s`).
