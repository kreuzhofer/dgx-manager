# DeepSeek-V4-Flash-0731 (self-hosted, fp8) — head-to-head vs GLM-5.2

*2026-08-09*

## What this is

We ran DeepSeek-V4-Flash-0731 on **two** DGX Sparks and put it through the same two
benchmarks as [`glm-5.2-benchmark-results.md`](glm-5.2-benchmark-results.md), with the
same harnesses and settings, so the numbers are directly comparable. GLM-5.2 holds all
four nodes; DeepSeek holds two, which is the frame for everything below.

Prompted by Artificial Analysis rating the two models within about a point of each
other. On the benchmarks a self-hoster can actually reproduce, they are not.

## Results

| Benchmark | GLM-5.2 (4 nodes) | DeepSeek-V4-Flash (2 nodes) | Verdict |
|---|---|---|---|
| **GPQA-Diamond** (198) | **69.2%** | **55.1%** (±3.5) | GLM, decisively |
| **SWE-bench Verified** (500) | **73.0%** (365/500) | **78.8%** (394/500) | DeepSeek — but see below |
| SWE-bench, like-for-like (472) | **77.3%** (365/472) | **79.2%** (374/472) | tie (p=0.28) |

**The one-line summary: GLM-5.2 is the better reasoner; on agentic coding they are
statistically indistinguishable, and DeepSeek achieves that on half the cluster.**

## GPQA-Diamond — 55.1% vs GLM's 69.2%

- Full 198 questions, `gpqa_diamond_cot_zeroshot` via lm-eval-harness 0.4.12,
  `local-chat-completions`, `num_concurrent=8`, `flexible-extract` filter.
- `strict-match` reads 0.0000 for **both** models — neither one's answer format hits
  lm-eval's exact template — so `flexible-extract` is the real number for both and the
  comparison stays honest.
- 55.05% ±3.54 at 95%. The 14-point gap is far outside the interval; this is not noise.
- Ran against the **sparkrun** launch of the recipe, before the dgxrun port existed.
  Same model, same recipe parameters (see the equivalence note below).

## SWE-bench Verified — 78.8%, and why that overstates it

Full 500 via mini-SWE-agent 2.4.5 (litellm), `-w 8`, `-c swebench.yaml`,
`model.model_kwargs.timeout=1800`, scored with SWE-bench harness 4.1.0. ~9h20m for
inference, ~1h50m for scoring. **500/500 submitted, 0 empty patches, 0 eval errors.**

Both models ran the same 500 instances, so the comparison can be **paired** rather than
resting on overlapping confidence intervals:

|  | count |
|---|---|
| Both resolved | 342 |
| DeepSeek only | 52 |
| GLM only | 23 |
| Neither | 83 |

McNemar over all 500: **χ² = 10.45, p = 0.0012** — significant.

**But that result is mostly an artifact.** GLM produced no patch at all on 28 instances:
8 context-exceeded at its 64K window, 12 step-budget exhaustions, and ~8 Docker-startup
timeouts. Those are scaffold and infrastructure failures, not reasoning failures — and
DeepSeek resolved **20 of the 28**.

Restricting to the 472 instances where GLM actually produced a patch:

|  | GLM-5.2 | DeepSeek |
|---|---|---|
| Resolved | 365 (77.3%) | 374 (79.2%) |
| Exclusive wins | 23 | 32 |

McNemar: **χ² = 1.16, p = 0.281 — not significant.** On like-for-like tasks the two
models are indistinguishable at agentic coding.

**Report the 78.8% only alongside the 79.2%-vs-77.3%**, or it claims a capability
advantage the data does not support.

## What this actually tells us

- **GLM-5.2 wins on knowledge/reasoning.** 14 points on GPQA survives any framing.
- **Neither model wins on agentic coding.** DeepSeek's headline edge is its context
  window plus GLM's operational bad luck.
- **DeepSeek wins decisively on efficiency.** Same coding capability on **2 nodes vs 4**,
  13B active parameters vs 40B — freeing two Sparks for other work.
- **The 1M context eliminated a whole failure class.** GLM's 64K window truncated ~1.6%
  of SWE tasks outright; DeepSeek never truncated once, and the long-prefill throughput
  collapse that killed GLM's 128K variant did not materialise.
- **GLM's 73.0% is arguably understated.** ~8 of its 28 no-patch instances were Docker
  startup timeouts, unrelated to the model, and would not necessarily recur on a rerun.
- **Artificial Analysis's near-parity rating did not hold** on the one cleanly
  apples-to-apples knowledge benchmark a self-hoster can run.

Both numbers measure **model + agent scaffold**, not weights alone. The scaffold does a
lot of the work on SWE-bench; report it as such.

## The served stack

- **Model:** `deepseek-ai/DeepSeek-V4-Flash-0731` (284B total / 13B active, fp8).
- **Hardware:** 2× NVIDIA GB10, tensor-parallel 2, `B12X_MLA_SPARSE` attention,
  dspark speculative decoding, vLLM OpenAI endpoint.
- **Recipe:** `@dgxrun/deepseek-v4-flash-0731-2x` — a faithful port of
  `@official/deepseek-v4-flash-0731-b12x-dspark-vllm`, with the
  `instanttensor-hybrid-draft-loader` mod.
- **Negotiated at runtime:** `max_model_len: auto` → 1,048,576; GPU KV cache 1,061,250
  tokens (11.2 GiB/node); engine init 156 s.
- **Speculative decoding measured live:** mean acceptance length 3.61, draft acceptance
  52.2% at depth 5.

### Note on the two launches

GPQA ran against the **sparkrun** launch; SWE-bench against the **dgxrun** port. These
are equivalent by construction, not by assumption:
`packages/agent/src/runtime/dgxrun/deepseek-golden.test.ts` pins the dgxrun-rendered
argv and env against the launch sparkrun actually performed on this cluster, and every
deliberate difference is enumerated there (forced `mp` executor, `--served-model-name`
placement, injected fabric env). Re-running GPQA under dgxrun would remove even that
caveat and is the obvious next tidy-up.

## Reproduction

- **GPQA:** `POST /api/benchmarks {deploymentId, presetId:"acc-gpqa-diamond-full",
  numConcurrent:8}`.
- **SWE-bench:** `~/swebench/run_ds_swe.sh <outdir>` on agenthost (checked into that
  host, not this repo) wraps:
  ```
  OPENAI_API_BASE=http://<head>:8000/v1 MSWEA_COST_TRACKING=ignore_errors \
  python -m minisweagent.run.benchmarks.swebench --subset verified --split test \
    -m openai/<served-name> --environment-class docker \
    -w 8 -c swebench.yaml -c model.model_kwargs.timeout=1800 -o <outdir>
  ```
  then
  ```
  python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified --split test \
    --predictions_path <outdir>/preds.json --max_workers 8 --run_id <id>
  ```
- `MSWEA_COST_TRACKING=ignore_errors` is required (litellm knows neither model's pricing).
- `-w 8` matches the recipe's `max_num_seqs: 8`; raising one without the other wastes
  either workers or KV.
