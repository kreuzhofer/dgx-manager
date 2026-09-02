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
| SWE-bench Verified (500), GLM at 64K | 73.0% (365/500) | 78.8% (394/500) | DeepSeek (p=0.001) — but confounded |
| **SWE-bench Verified** (500), **GLM context-corrected** | **75.6%** (378/500) | **78.8%** (394/500) | DeepSeek, not significant (p=0.061) |
| SWE-bench, like-for-like (472) | 77.3% (365/472) | 79.2% (374/472) | tie (p=0.28) |

**The one-line summary: GLM-5.2 is the better reasoner; on agentic coding DeepSeek is
somewhat ahead but not decisively, and it gets there on half the cluster.**

Roughly **half** of DeepSeek's apparent SWE-bench win was GLM's 64K context window rather
than capability — see the context-corrected run below. Quote the corrected row, not the
first one.

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

## Context-corrected — re-running GLM's 28 failures at 320K

The obvious objection to the above is that DeepSeek ran at 1M context and GLM at 64K, so
the comparison hands DeepSeek a structural advantage. We tested that directly: re-ran
**only the 28 instances GLM failed to patch**, on
`@dgxrun/zai-glm-5.2-quanttrio-int4-unpruned-dcp2-320k-4x` — 320K context, `--max-num-seqs 1`.

| Of the 28 | |
|---|---|
| Now produce a patch | **23** (was 0) |
| Actually resolve | **13** |
| Still `LimitsExceeded` | 5 |

**Context was a real limiter.** 23 of 28 tasks that produced nothing at 64K produce a
patch at 320K. But a patch is not a solution — only 13 resolved.

**The 5 remaining failures are not about context.** KV usage sat at 6.8% of the window,
so they are step-budget exhaustion — agentic inefficiency, which a bigger window cannot
fix.

Corrected: **365 + 13 = 378/500 = 75.6%**. Paired against DeepSeek:

|  | count |
|---|---|
| Both resolved | 354 |
| DeepSeek only | 40 |
| GLM only | 24 |
| Neither | 82 |

McNemar: **χ² = 3.52, p = 0.061 — no longer significant at 0.05.** Removing GLM's
truncation handicap moves the comparison from p=0.0012 to p=0.061 and halves the gap
(5.8 → 3.2 points). Close enough that "settled tie" would also overclaim.

A residual gap survives for a real reason: on those same 28 hard instances DeepSeek
resolved **20** and GLM resolves **13**.

**Caveat that must travel with the 75.6%:** it mixes two configurations — 472 instances
from the `c16-64k` run and 28 from the 320K `c=1` run, and the latter also carries MTP and
a different recipe. It is a defensible estimate of *GLM with adequate context*, not a
clean single-condition measurement. The only way to remove that caveat is a full 500 at
320K, measured at **~2.8 days** of cluster time (c=1 sustains ~20 tok/s aggregate; GB10
concurrency buys ~2×, not 8×, so serialising costs roughly the whole speedup). Judged not
worth it for an expected ≤1.6-point refinement.

## What this actually tells us

- **GLM-5.2 wins on knowledge/reasoning.** 14 points on GPQA survives any framing.
- **DeepSeek is somewhat ahead on agentic coding, not decisively.** 75.6% vs 78.8% once
  GLM's context handicap is removed, p=0.061.
- **About half the apparent SWE-bench gap was context, not capability.** That is the most
  transferable finding here: on an agentic benchmark, the serving window is a first-order
  variable, and a model evaluated at a short context is being measured with a handicap
  that looks exactly like weakness.
- **DeepSeek wins decisively on efficiency.** Comparable coding on **2 nodes vs 4**,
  13B active parameters vs 40B — freeing two Sparks for other work.
- **Context stops paying at some point.** Going 64K → 320K converted 23 of 28 no-patch
  failures into patches, but only 13 into solutions, and 5 tasks failed on step budget
  with 93% of the window unused.
- **The 1M context eliminated a whole failure class.** GLM's 64K window truncated ~1.6%
  of SWE tasks outright; DeepSeek never truncated once, and the long-prefill throughput
  collapse that killed GLM's 128K variant did not materialise.
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
- **The GLM context-corrected run:** `~/swebench/run_glm_c1_swe.sh <outdir>
  --filter "$(cat ~/glm_28_ids.txt | paste -sd'|' | sed 's/^/^(/;s/$/)$/')"`, where the
  IDs come from `empty_patch_ids` in the baseline report. It uses `-w 4` (not `-w 1`)
  against the `max_num_seqs: 1` server — the server serialises regardless, so extra client
  workers only keep the queue full while agents run bash/docker — and
  `model.model_kwargs.timeout=3600`, because a request can now wait behind up to three
  others and exceeding the timeout triggers the re-prefill death spiral.
