# Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 (baseline: two single-node jobs, no link)

## workload: 64 requests

cell | tok/s | tpot p50 ms | tpot p95 ms | ttft s | e2e p50 s | e2e p95 s | wait s | local | peer | nvme | admits | overlap | turns A/B | nvme GB/s | link GB/s | err
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
indep_cat_box0 | 2.65 | 1359.0 | 1803.6 | 6.73 | 175.02 | 240.20 | 775.95 | 0.78 | 0.00 | 0.22 | 0 | - | 25/0 | 1.165 | 0.000 | 0
indep_cat_box1 | 5.88 | 462.3 | 1365.3 | 4.49 | 60.26 | 176.33 | 334.00 | 0.92 | 0.00 | 0.08 | 0 | - | 0/25 | 1.018 | 0.000 | 0
indep_cat | 5.30 | 735.2 | 1759.7 | 5.61 | 98.45 | 240.20 | 554.98 | 0.84 | 0.00 | 0.16 | 0 | - | 25/25 | 1.624 | 0.000 | 0
