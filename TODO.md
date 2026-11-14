# TODO

Delete each item when it is done. Commands run from the repo root, on a machine where the
`fuzzball` CLI can reach the cluster.

## First runs

- [ ] **Smoke test** ([fuzzball/tests/smoke.fz](fuzzball/tests/smoke.fz)): one 2-node job with
  the same resources as an experiment. Its log answers U1–U6 and U10 below.
  ```bash
  fuzzball workflow start fuzzball/tests/smoke.fz --name coopmoe-smoke
  ```
  ```bash
  fuzzball workflow log <id> smoke -f
  ```
- [ ] **30B** (smaller, faster to stage):
  ```bash
  python stage.py --model qwen3-30b-a3b --size 80GiB
  ```
  ```bash
  python driver.py --requests data/requests.json --volume coopmoe-qwen3-30b-a3b --num 16
  ```
- [ ] **235B**:
  ```bash
  python stage.py --model qwen3-235b-a22b --size 350GiB
  ```
  ```bash
  python driver.py --requests data/requests.json --volume coopmoe-qwen3-235b-a22b --num 16 --concurrency 8
  ```
- [ ] Once U1–U6 and U10 are settled: delete `fuzzball/tests/smoke.fz`, and the old `coopMoE` volume
  (only the smoke test mounts it).

## Open questions

| # | question | how it shows up | if the answer is bad |
|---|---|---|---|
| U1 | Do team1's nodes take jobs? (On Nov 13 at 17:45 every job was rejected with "too few nodes".) | the smoke test runs at all | ask the organizers |
| U2 | Is `/local` the box's own NVMe, and can it be 300 GiB? | smoke: `findmnt` / `df` on `/local` | lower `local_size`, or read from NFS (not built) |
| U3 | How fast are `/local` and NFS? | smoke: `dd` on both | a slow NFS makes the per-job store copy long |
| U4 | Is a 120 GiB exclusive memory request accepted? | smoke: memory section | lower `memory` in `SETTINGS` and `cache.gb` with it |
| U5 | Are the two boxes linked by CX-7, and at which addresses? | smoke: interface list | put the CX-7 addresses in `boxes` in `SETTINGS` (Ethernet also works) |
| U6 | Does the image see the GB10 with CUDA? | smoke: torch section | another arm64 CUDA image |
| U7 | How large can a team volume be? | `stage.py` creating the volume | ask the organizers |
| U8 | How long do the download and the 235B store take? | `stage.py`'s log | — |
| U9 | How big can `cache.gb` go on 235B, and at which `--concurrency`? | the engine's `cache_gib` line; the job is not killed for memory | lower `cache.gb` in the config, or `--concurrency` |
| U10 | Does `transformers==5.23.0` install in the image (Python ≥ 3.11, the image's pip pins), and does HF's continuous batching run on the GB10? | smoke: U10 section prints two finished requests | pin the transformers version the image allows, or another image |

