# Design: GLM-5.2 eval recipe variant (high-throughput batched serving)

*2026-07-12*

## Problem

Our eval runs are gated by the **serving** recipe, which is tuned for a single
low-latency user, not for a benchmark's many concurrent requests:

- `--max-num-seqs 1` — the endpoint serves **one request at a time**. Every eval
  is therefore serial: GPQA (198) took ~5h, AIME re-runs queue, and a full
  SWE-bench Verified 500 with mini-SWE-agent is estimated at **2–3 days** — almost
  entirely the serial agent phase (measured: 20 astropy tasks = 79 min at
  `-w 1`; more workers just queue on the single-seq endpoint).
- `--speculative-config … mtp …` — the MTP drafter is a **latency** optimization.
  At batch=1 it's a big win (memory-bound, spare compute verifies drafts cheaply;
  acceptance 3.5–3.9). At high batch the GPU is compute-bound and speculation
  *competes* for compute — rejected drafts waste cycles that could serve other
  requests. So MTP is counterproductive for batched throughput.

Evals are throughput-bound, not latency-bound. This variant trades single-stream
latency for aggregate throughput.

## Goal

A GLM-5.2 recipe that serves **many concurrent requests** (lm-eval `num_concurrent`,
mini-SWE-agent `-w N`), cutting eval wall-clock ~5–10×, at the cost of per-request
latency and long context — both of which evals don't need.

## Non-goals

- Replacing the 320K serving recipe. This is a **campaign-mode** recipe (see
  Deployment below). The 320K/MTP recipe stays the daily-driver for coding.
- Long context. Evals need far less than 320K (GPQA/AIME prompts <10K;
  mini-SWE-agent context ~32–64K).

## Recipe: `recipes/dgxrun/glm-5.2-quanttrio-c16-64k.yaml`

Same container, model, DCP2 topology, and env as `…-dcp2-320k.yaml` (no rebuild).
Only the serve knobs change. Diff from the 320K recipe:

| Knob | 320K serving | Eval variant | Why |
|---|---|---|---|
| `--speculative-config` (MTP) | present | **removed** | MTP hurts at high batch; removing it also frees the drafter's VRAM for KV |
| `--max-num-seqs` | 1 | **16** | the core change — batch concurrent eval requests |
| `--max-model-len` | 327680 | **65536** | evals don't need 320K; smaller per-seq KV → room for 16 seqs |
| `--max-num-batched-tokens` | 1024 | **8192** | the 1024 bounded the sparse-indexer prewarm at 320K; at 64K the prewarm is ~5× smaller, so raise it for batched prefill |
| `--kv-cache-memory-bytes` | 9126805504 (~8.5G) | **retune (~26–30G)** | 16 seqs × 64K ≈ 1.05M tokens ÷ 39,223 tok/GiB ≈ **27 GiB** KV. The freed MTP weights + smaller indexer prewarm + smaller per-seq KV supply it. Validate against the startup guard. |
| `--decode-context-parallel-size` | 2 | 2 (keep) | DCP2 is baked into the container/indexer; harmless at 64K |
| `--gpu-memory-utilization` | 0.88 | 0.88 (start) | keep; retune only if the startup guard trips |
| cudagraph | FULL, capture 10 | FULL, **capture ≥16** | must cover the batch sizes actually used; larger capture = more startup memory/time |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | 256 | 256 (keep) | context-sized buffer; smaller at 64K, so fine |
| MTP-related env (`VLLM_DCP_SHARD_DRAFT`) | 1 | drop/ignore | no drafter to shard |

Everything else (NCCL, JIT caches, `reasoning-parser glm45`, `tool-call-parser
glm47`, prefix caching, fp8 KV, async scheduling) is unchanged.

**KV sizing is the one thing to validate empirically** — the memory notes three
independent walls on this stack (the `gmu×total` startup guard vs CUDA-visible
free, `min_free_kbytes`, and the b12x indexer prewarm ∝ `mnbt × max_model_len`).
At 64K + mnbt 8192 the prewarm is ~2.6× the 320K@1024 baseline (5 GiB → ~13 GiB)
— that must be checked; if it's too large, lower mnbt to 4096. Bring KV up
incrementally (start ~24 GiB, raise until the guard or an OOM stops you).

## Deployment mode: campaign swap (no spare nodes)

The 4 GPU nodes run the 320K deployment (full unified pool); there is **no idle
GPU node** to host a second deployment. So the eval variant is deployed **in place
of** the 320K deployment for the duration of an eval campaign:

1. Stop the 320K deployment (`POST /:id/restart` won't switch recipes — DELETE +
   re-POST with the eval recipe, or add recipe-swap support to restart).
2. Deploy `@dgxrun/glm-5.2-quanttrio-c16-64k`.
3. Run the eval batch (GPQA / AIME / SWE-bench) at high concurrency.
4. Delete the eval deployment, redeploy the 320K serving recipe.

This is a deliberate, operator-initiated campaign, not a background mode. Document
the swap in the runbook. (A future option: a small second GPU box, or time-sharing,
but out of scope here.)

## Consumer plumbing: make concurrency a preset field

The batched endpoint is useless unless the runner sends concurrent requests.

- **lm-eval accuracy kind**: `buildLmEvalArgs` currently hardcodes
  `num_concurrent=1`. Add `numConcurrent` to `AccuracyConfig`/the preset (default
  1 for the serving recipe; 16 for eval-campaign presets) and pass it through to
  `--model_args num_concurrent=<n>`.
- **mini-SWE-agent**: run with `-w <n>` (e.g. 16) so N agents' turns batch at the
  endpoint. Each agent is still serial per-task; concurrency comes from N tasks
  in flight.
- **throughput/tool-eval**: leave at their existing concurrency; they measure
  latency-shaped metrics and aren't the target here.

Set the runner concurrency to match (or slightly below) `--max-num-seqs` so the
batch fills without excessive queueing.

## Expected impact

- Agent phase throughput ≈ min(`max-num-seqs`, runner workers) × single-stream,
  minus batching overhead. Realistically **~5–8×** aggregate for many small eval
  requests. SWE-bench Verified 500: **~2–3 days → well under a day**. GPQA 198:
  ~5h → ~40–60 min.
- Per-request latency **increases** (no MTP, shared compute) — fine for evals,
  unacceptable for the interactive 320K use, which is why they're separate recipes.
- Precision is unchanged — batching changes speed, not scores (same greedy/temp,
  same extraction). The ±CI limits from the earlier analysis still hold (±4.3% at
  n=500); this variant makes reaching n=500 affordable, not more precise.

## Validation plan

1. Deploy the eval variant on the 4 nodes (campaign swap); confirm it loads (watch
   the KV/guard walls) and serves.
2. `vllm:num_requests_running` should climb toward `max-num-seqs` under a batched
   client (not cap at 1).
3. Re-run GPQA-Diamond with `num_concurrent=16`; confirm score matches the serial
   run (67.7% ± noise) and wall-clock drops ~5–8×.
4. Then commit to the SWE-bench Verified 500 (or a large shuffled sample) with
   `-w 16`.

## Risks

- **KV/startup walls** (the recurring GB10 failure mode) — mitigate by raising KV
  incrementally and watching the guard; keep mnbt adjustable.
- **cudagraph capture at batch 16** — more startup memory/time; fall back to
  PIECEWISE or a smaller capture set if capture OOMs.
- **Campaign swap disrupts the 320K deployment** — schedule eval campaigns when
  the coding endpoint isn't needed; the swap back is a normal deploy.
- **Batching inefficiency for wildly uneven request lengths** (SWE-bench agent
  turns vary) — real but still far better than serial; tune `max-num-seqs` down if
  decode latency starves short requests.
