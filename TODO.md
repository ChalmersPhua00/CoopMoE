# TODO

Delete each item when it is done. Commands run from the repo root, on a machine where the
`fuzzball` CLI can reach the cluster.

## First runs

- [ ] **Two-node smoke test** ([fuzzball/tests/smoke.fz](fuzzball/tests/smoke.fz)) for U5. It was
  only run on one node. It mounts the deleted `coopMoE` volume and the slow `ephemeral` one: point it
  at `coopmoe-qwen3-30b-a3b` and `team1-scratch` first.
  ```bash
  fuzzball workflow start fuzzball/tests/smoke.fz --name coopmoe-smoke
  ```
- [ ] **Run 30B** (the first two-node run: check its `local disk:` log line shows `/dev/nvme0n1p2` on both boxes):
  ```bash
  python driver.py --requests data/requests.json --volume coopmoe-qwen3-30b-a3b --num 16
  ```
- [ ] **Run 235B**:
  ```bash
  python driver.py --requests data/requests.json --volume coopmoe-qwen3-235b-a22b --num 16 --concurrency 8
  ```
- [ ] **HF baseline on the GB10**: stock transformers 5.23.0 experts fail with
  `torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0, 10.0`
  (the GB10 is 12.1). The baseline needs another experts implementation; check that CoopMoE's
  own engine is unaffected.
- [ ] Once U5 is settled: delete `fuzzball/tests/smoke.fz`.

## Open questions

| # | question | how it shows up | if the answer is bad |
|---|---|---|---|
| U5 | Are the two boxes linked by CX-7, and at which addresses? (The cards are there: two ports up at 100 Gb/s per box.) | smoke: link section, on two nodes | put the CX-7 addresses in `boxes` in `SETTINGS` (Ethernet also works) |
| U9 | How big can `cache.gb` go on 235B, and at which `--concurrency`? | the engine's `cache_gib` line; the job is not killed for memory | lower `cache.gb` in the config, or `--concurrency` |
