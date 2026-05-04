# Qwen 3.6-27B FP8 on DGX Spark — cost-per-token analysis

> Tagged 2026-05-04. Numbers from a measurement pass on a 2-node DGX Spark
> cluster (`recipes/qwen3.6-27b-fp8.yaml` at `tensorParallel=2`).
> Re-run the measurement scripts (see "How to refresh") whenever hardware,
> driver, vLLM version, or recipe flags change.

---

## TL;DR

At **100% utilization** on a 2-node DGX Spark cluster (2 × GB10, 6000 € capex
total, 3-year depreciation, ~200 W peak combined draw):

| | Our cluster | Qwen 3.6-27B API list price |
|---|---:|---:|
| Per 1M **input** tokens | **0.032 €** | ~0.55 € (17× cheaper) |
| Per 1M **output** tokens | **0.82 €** | ~3.31 € (4× cheaper) |
| Output:Input ratio | 25:1 | 6:1 |

At a more realistic **30% duty cycle**: **~0.11 €/M input** and **~2.7 €/M
output** — still cheaper than the API on both.

Per-request example (200 input + 800 output): **~0.066 €¢/request** at
100% util, or about 1,500 requests per €.

The cluster is cost-competitive with the API even at modest utilization.
Capex dominates (~80% of total cost), so the strongest lever for further
reduction is pushing utilization up.

---

## Cluster + workload under measurement

| | |
|---|---|
| Hardware | 2× NVIDIA DGX Spark (GB10, 122 GB unified memory each) |
| Capex | 3000 € per node = 6000 € total |
| Depreciation | 3 years straight-line |
| Power (peak) | ~100 W per node = 200 W combined |
| Electricity | 0.30 €/kWh assumed |
| Network | Per-node fast fabric (100/200 GbE) — included in capex |
| vLLM | 0.20.1rc1.dev152+gc3ad791e1 (cu132) |
| Container | `vllm-node` from `eugr/spark-vllm-docker` (kreuzhofer fork), pinned wheel |
| Model | `Qwen/Qwen3.6-27B-FP8` (27B dense, hybrid GatedDeltaNet+Gated-Attention, multimodal) |
| TP | 2 (head + 1 worker) |
| KV cache dtype | FP8 |
| Attention backend | flashinfer |
| Recipe | `recipes/qwen3.6-27b-fp8.yaml` |

TP=4 is currently blocked by an upstream vLLM issue
(see `docs/vllm-issue-draft-tp4-hang.md`); when that is unblocked the
analysis here should be re-run at TP=4 and the per-token numbers will
likely drop further.

---

## Measured throughput

### Decode (output) — concurrency sweep, max_tokens=256, 8 reqs per cell

Source: `scripts/perf_test.py` against the live endpoint.
Results: `/mnt/tank/results/qwen3.6-27b-fp8-tp2-perf/decode-sweep.json`

| concurrency | aggregate decode TPS | per-stream decode TPS | request latency mean (s) |
|---:|---:|---:|---:|
| 1   | (streaming, untracked here; ~13 from latency) | ~13 | 19.7 |
| 4   | **52.0** | 13.0 | 19.7 |
| 8   | **98.3** | 12.3 | 20.8 |
| 16  | **98.2** (saturated) | 12.3 | 20.8 |

**Decode plateaus at ~98 tok/s aggregate at concurrency ≥ 8.** Per-stream
throughput is rock-stable at ~12-13 tok/s — vLLM continuous batching is
doing its job within this range. The plateau is the relevant number for
saturated-server cost analysis.

### Prefill (input) — single-request latency vs prompt size

Source: ad-hoc streaming TTFT measurement with cache-busted prompts (a
nonce per request prevents prefix-cache hits skewing the numbers).
Results: `/mnt/tank/results/qwen3.6-27b-fp8-tp2-perf/prefill.json`

| prompt tokens | TTFT (= prefill time) | prefill TPS |
|---:|---:|---:|
| 493   | 0.27 s | 1,823 |
| 944   | 0.34 s | 2,784 |
| 1,846 | 0.76 s | 2,420 |
| 4,548 | 1.83 s | 2,489 |
| 9,046 | 5.86 s | 1,543 |
| 18,046 | 11.61 s | 1,554 |

**Sweet spot ~2,500 tok/s prefill** for moderate prompt sizes (1k-5k).
Long prompts (~9k+) drop to ~1,500 tok/s as the attention quadratic kicks
in. **For typical chat / RAG workloads (200-2,000 input tokens), 2,500 tok/s
is the representative number.** This is a single-request measurement;
concurrent prefill batches together, so aggregate prefill TPS at
concurrency > 1 will be higher (unmeasured here — see "Refinements").

---

## Cost per cluster-second

| Component | €/s | derivation |
|---|---:|---|
| Capex amortization | 0.0000634 | 6000 € / (3 × 365 × 24 × 3600 s) |
| Power | 0.0000167 | 200 W × 0.30 €/kWh / 3600 s/h |
| **Total** | **0.0000801** | (capex 79%, power 21%) |

Capex dominates by 4× over power. **The dominant lever to reduce
cost-per-token is pushing utilization up**, not reducing power draw.

---

## Cost per million tokens (100% utilization)

The cluster costs the same per second whether it's prefilling or decoding —
the per-token cost falls out of `cluster_€/s ÷ tokens/s` for each phase.

### Output / decode — at decode saturation (98 tok/s)

```
0.0000801 €/s / 98 tok/s × 1,000,000 = 0.817 €/M output tokens
```

### Input / prefill — at single-request prefill (2,500 tok/s)

```
0.0000801 €/s / 2,500 tok/s × 1,000,000 = 0.032 €/M input tokens
```

### Comparison to API list price

Source: llm-stats leaderboard, $0.60/M input + $3.60/M output for Qwen 3.6
27B. Converted at 0.92 €/$.

| | Cluster (100% util) | Qwen 27B API | Cluster vs API |
|---|---:|---:|---:|
| 1M input tokens | **0.032 €** | 0.55 € | **17× cheaper** |
| 1M output tokens | **0.82 €** | 3.31 € | **4× cheaper** |

### At realistic 30% duty cycle

Multiply both by 1/0.30 ≈ 3.33×:

| | 30% duty cycle | API |
|---|---:|---:|
| 1M input | **0.11 €** | 0.55 € (5× cheaper) |
| 1M output | **2.72 €** | 3.31 € (~20% cheaper) |

Even at 30% duty cycle the cluster beats API pricing. At 50% duty cycle
the gap widens substantially (0.064 €/M input, 1.63 €/M output → both 2-3×
cheaper than API).

---

## Worked example — a typical chat request

Assume 200 input tokens + 800 output tokens per request.

| | Tokens | Cost @ 100% util | Cost @ 30% util |
|---|---:|---:|---:|
| Input | 200 | 0.0064 €¢ | 0.021 €¢ |
| Output | 800 | 0.066 €¢ | 0.218 €¢ |
| **Total** | 1000 | **0.072 €¢** | **0.239 €¢** |

At 100% utilization: ~14,000 such requests per €. At 30% duty cycle:
~4,200 requests per €.

For comparison, the same request hitting the Qwen 27B API list price:
- Input: 200 × 0.55 €/M = 0.011 €¢
- Output: 800 × 3.31 €/M = 0.265 €¢
- **Total: ~0.276 €¢** → ~360 requests per €

So at 100% utilization the cluster is **~38× cheaper per request** than
the API; at 30% duty cycle, **~12× cheaper**.

---

## What's NOT in these numbers

Be honest about the omissions before quoting any figure to a stakeholder:

1. **Sysadmin / engineering time.** Setting up + maintaining the cluster
   has a real cost. Not amortized here.
2. **Cooling, rack, network switch capex/opex.** Probably small at this
   scale (4 small machines fit in a normal office) but non-zero.
3. **Reliability / SLA.** API has a defined uptime contract; our cluster
   does not. Building one means more headroom (warm-spare nodes, etc).
4. **Concurrent prefill batching.** At c > 1, multiple requests' prefills
   batch together. Aggregate prefill TPS could be 2-4× higher than the
   single-request number — would make input cost even cheaper. Untested
   here.
5. **Long contexts hurt.** Prefill TPS halves at ~9k+ token prompts.
   For document-RAG workloads, recompute with the relevant prompt-size
   distribution.
6. **vLLM version drift.** The 0.20.1 wheel we're on is a specific
   pinned snapshot; later versions may change throughput in either
   direction. Re-measure on every wheel bump.
7. **Decode TPS at higher concurrency.** We only swept c=1, 4, 8, 16.
   The benchmark doc for the related Qwen 3.6 35B-A3B FP8 model showed
   continued aggregate-TPS gains at c=16 → c=32 (with longer queue
   depth). Worth measuring c=32+ here for a more honest "saturated"
   number.

---

## Sensitivity analysis

How the headline numbers move when single inputs change:

| Change | Effect on cost/M output |
|---|---|
| Power doubles (400 W per node, 0.30 €/kWh) | +21% (still capex-dominated) |
| Capex doubles (6000 €/node) | +79% |
| 3y depreciation → 5y | -40% |
| Decode TPS at saturation goes 98 → 150 (e.g. better batching) | -35% |
| Duty cycle 100% → 50% → 30% → 10% | ×2, ×3.3, ×10 |
| TP=4 unblocks and gives 1.7× decode TPS | -41% |

Capex amortization length and duty cycle are by far the biggest swing
factors. Power, even at peak, is a minor contributor.

---

## How to refresh these numbers

1. Deploy `recipes/qwen3.6-27b-fp8.yaml` at `tensorParallel=2`.
2. Run the decode sweep:
   ```
   python scripts/perf_test.py \
     --endpoint http://192.168.44.36:8000 \
     --served-name Qwen/Qwen3.6-27B-FP8 \
     --concurrencies 1,4,8,16,32 \
     --requests-per-c 8 \
     --max-tokens 256 \
     --output /workspace/results/<run-name>/decode-sweep.json
   ```
3. Run the prefill measurement:
   ```
   python /mnt/tank/results/qwen3.6-27b-fp8-tp2-perf/prefill_test.py
   ```
4. Update the throughput table above with the new aggregate decode TPS at
   saturation and the prefill TPS at the most-relevant prompt size for
   your workload.
5. Update the cluster cost / second only if hardware, capex, or power
   assumptions changed.
6. Re-derive the per-million numbers via `cluster_€/s / TPS × 1,000,000`.

---

## Source artifacts

- Decode sweep raw: `/mnt/tank/results/qwen3.6-27b-fp8-tp2-perf/decode-sweep.json`
- Prefill raw: `/mnt/tank/results/qwen3.6-27b-fp8-tp2-perf/prefill.json`
- Prefill measurement script: `/mnt/tank/results/qwen3.6-27b-fp8-tp2-perf/prefill_test.py`
- Recipe: `spark-vllm-docker/recipes/qwen3.6-27b-fp8.yaml`
- Concurrency-sweep tool: `dgx-manager-fine-tune-recipes/scripts/perf_test.py`
