# Gemma 4-31B FP8 on DGX Spark — cost-per-token analysis

> Tagged 2026-05-04. Numbers from a measurement pass on a 2-node DGX Spark
> cluster (`recipes/gemma-4-31b-fp8.yaml` at `tensorParallel=2`,
> `--enforce-eager`).
> Re-run the measurement scripts (see "How to refresh") whenever hardware,
> driver, vLLM version, or recipe flags change.
>
> All € figures shown in dual-currency at **€1 = $1.087** (May 2026 rate
> used uniformly throughout). Update the rate + re-derive USD if the
> currency basis changes materially.
>
> **Caveat:** This run uses `--enforce-eager` (torch.compile + CUDA graphs
> disabled) because the compiled path hangs during inductor codegen for
> Gemma 4's heterogeneous head dimensions (head_dim=256 / global_head_dim=512)
> when paired with `VLLM_DISABLE_COMPILE_CACHE=1` and cross-node TP=2. Once
> that hang is resolved upstream, re-run with the compiled path — TPS will
> likely improve by 30–50% and per-token cost will drop accordingly. For
> the apples-to-apples Qwen vs Gemma comparison, both should ideally be
> measured under the same compile mode.

---

## TL;DR

<!-- TODO: fill after benchmark run -->

---

## Cluster + workload under measurement

| | |
|---|---|
| Hardware | 2× NVIDIA DGX Spark (GB10, 122 GB unified memory each) |
| Capex | €3,000 / $3,260 per node = €6,000 / $6,520 total |
| Depreciation | 3 years straight-line |
| Power (peak) | ~100 W per node = 200 W combined |
| Electricity | €0.30 / $0.326 per kWh assumed |
| vLLM | 0.20.1rc1.dev152+gc3ad791e1 (cu132) |
| Container | `vllm-node-tf5` from `eugr/spark-vllm-docker` (kreuzhofer fork), pinned wheel |
| Model | `google/gemma-4-31B-it` (31B dense, multimodal text+image+video, runtime FP8) |
| TP | 2 (head spark-03 + worker spark-04) |
| KV cache dtype | FP8 |
| Compile | **Disabled** (`--enforce-eager`) — see caveat above |
| Recipe | `recipes/gemma-4-31b-fp8.yaml` |

---

## Measured throughput

### Decode (output) — concurrency sweep, max_tokens=256, 8 reqs per cell

Source: `scripts/perf_test.py` against the live endpoint.
Results: `/mnt/tank/results/gemma-4-31b-fp8-tp2-perf/decode-sweep.json`

<!-- TODO: fill after benchmark run -->

### Prefill (input) — single-request latency vs prompt size

Source: ad-hoc streaming TTFT measurement with cache-busted prompts (a
nonce per request prevents prefix-cache hits skewing the numbers).
Results: `/mnt/tank/results/gemma-4-31b-fp8-tp2-perf/prefill.json`

<!-- TODO: fill after benchmark run -->

---

## Cost per cluster-second

| Component | €/s | $/s | derivation |
|---|---:|---:|---|
| Capex amortization | 0.0000634 | 0.0000689 | €6,000 / (3 × 365 × 24 × 3600 s) |
| Power | 0.0000167 | 0.0000182 | 200 W × €0.30/kWh / 3600 s/h |
| **Total** | **0.0000801** | **0.0000871** | (capex 79%, power 21%) |

Equivalently: **€0.288 / $0.314 per hour** for the whole 2-node cluster.
Same hardware as the qwen3.6-27b cost analysis, so the cluster €/s is
identical — only the per-token throughput differs.

---

## Cost per million tokens (100% utilization)

<!-- TODO: fill after benchmark run -->

---

## How to refresh these numbers

1. Deploy `recipes/gemma-4-31b-fp8.yaml` at `tensorParallel=2` on
   spark-03 + spark-04 (or whichever pair is idle).
2. Run the decode sweep:
   ```
   python /mnt/tank/src/github/dgx-manager-fine-tune-recipes/scripts/perf_test.py \
     --endpoint http://192.168.44.142:8001 \
     --served-name google/gemma-4-31B-it \
     --concurrencies 1,4,8,16,32 \
     --requests-per-c 8 \
     --max-tokens 256 \
     --output /mnt/tank/results/gemma-4-31b-fp8-tp2-perf/decode-sweep.json
   ```
3. Run the prefill measurement:
   ```
   python /mnt/tank/results/gemma-4-31b-fp8-tp2-perf/prefill_test.py
   ```
4. Update the throughput table above.
5. Re-derive per-million numbers via `cluster_€/s / TPS × 1,000,000`.

---

## Source artifacts

- Decode sweep raw: `/mnt/tank/results/gemma-4-31b-fp8-tp2-perf/decode-sweep.json`
- Prefill raw: `/mnt/tank/results/gemma-4-31b-fp8-tp2-perf/prefill.json`
- Prefill measurement script: `/mnt/tank/results/gemma-4-31b-fp8-tp2-perf/prefill_test.py`
- Recipe: `spark-vllm-docker/recipes/gemma-4-31b-fp8.yaml`
- Concurrency-sweep tool: `dgx-manager-fine-tune-recipes/scripts/perf_test.py`
