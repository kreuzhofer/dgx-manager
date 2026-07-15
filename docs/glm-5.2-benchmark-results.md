# GLM-5.2 (self-hosted, INT4) — comparable benchmark results

*2026-07-15*

## What this is

We serve GLM-5.2 quantized on a 4-node DGX Spark cluster and ran the subset of
standard benchmarks a self-hoster can actually reproduce off the shelf, to put our
numbers next to the official GLM-5.2 figures. This documents the results, the exact
methodology, and the caveats — the backbone for a write-up.

Companion docs: [`glm-5.2-256k-to-320k.md`](glm-5.2-256k-to-320k.md) (the serving-stack
journey), [`glm-5.2-inference-benchmark.md`](glm-5.2-inference-benchmark.md) (speed).

## The served stack

- **Model:** `QuantTrio/GLM-5.2-Int4-Int8Mix` (INT4 / INT8-mixed quantization).
- **Hardware:** 4× NVIDIA GB10 (Grace-Blackwell, unified memory), tensor-parallel 4,
  decode-context-parallel 2 (DCP2), b12x sparse-MLA indexer, vLLM OpenAI endpoint.
- **Two recipes** — only one can hold the 4-node pool at a time:
  - `@dgxrun/glm-5.2-quanttrio-unpruned-dcp2-320k` — daily driver: 320K context, MTP
    speculative drafter, tuned for single-stream latency (~25–31 tok/s decode).
  - `@dgxrun/glm-5.2-quanttrio-c16-64k` — concurrent recipe for eval throughput: no MTP,
    `--max-num-seqs 16`, 64K context, 6 GiB KV, PIECEWISE cudagraph. Batches many
    requests at once. Accuracy is identical to the serving recipe (batching changes
    speed, not correctness — confirmed: GPQA 67.7% serial vs 69.2% batched, within noise).

## Results — ours vs official

| Benchmark | Ours (INT4, self-hosted) | Official GLM-5.2 | Directly comparable? |
|---|---|---|---|
| **GPQA-Diamond** (198) | **69.2%** (flexible-extract) | 91.2 | ✅ same benchmark |
| **AIME 2026** (30) | **90.0%** (27/30); 93.1% on converged | 99.2 | ~ official set, our harness + token budget |
| **SWE-bench Verified** (500) | **73.0%** (365/500) | 62.1 (on **Pro**, harder) | ~ ours = Verified, official = Pro |

The three barely-overlapping benchmark universes are why this list is short: the
official GLM-5.2 set is almost entirely frontier/agentic/closed (HLE, CritPt,
Terminal-Bench, FrontierSWE, MCP-Atlas, …), and the classic academic set (MMLU-Pro,
IFEval, GSM8K, …) is saturated and no longer officially reported. **GPQA-Diamond is the
single cleanly apples-to-apples number** a self-hoster can run.

## Detail + methodology

### GPQA-Diamond — 69.2% (vs official 91.2)

- Full 198 questions, `gpqa_diamond_cot_zeroshot` via lm-eval-harness 0.4.12,
  `local-chat-completions`, temp 0, single sample, `flexible-extract` filter.
  `num_concurrent=8`, ~3h18m, 0 errors. `strict-match` reads 0.0 (GLM's answer format
  doesn't hit lm-eval's exact template) — `flexible-extract` is the real number.
- The ~22-pt gap is **mostly harness, not weights**: the official 91.2 uses z.ai's own
  prompt/extraction/reasoning stack (likely sampling), which vendor numbers bake in and
  which isn't reproducible off the shelf.
- **Gotcha:** the endpoint's `glm45` reasoning parser already returns clean `content`
  (no `<think>` tags), so lm-eval can hit the vLLM endpoint **directly**; the
  `<think>`-strip proxy is unnecessary and, run on the weak Pi manager, actually breaks
  under concurrent long-lived requests (buffering → 502s). Point lm-eval at the endpoint.
- **Gated dataset:** `Idavidrein/gpqa` needs an HF account that has accepted the gate +
  a token dropped at `~/.cache/huggingface/token` on the eval node.

### AIME 2026 — 90.0% / 27 of 30 (vs official 99.2)

- 27/30 at a 48000-token generation cap; 93.1% on *converged* problems (27/29,
  excluding one that never stops reasoning even at 48k).
- The three misses: one non-convergence (runs to the token cap), two genuine reasoning
  errors. An earlier 24000-token cap truncated three problems; raising to 48000 recovered
  only one, so **part of the AIME gap is real, not just budget** — the honest
  counter-example to "it's all harness."

### SWE-bench Verified — 73.0% / 365 of 500 (vs official 62.1 on Pro)

- Full 500 via **mini-SWE-agent** (v2.4.5, litellm) over the `c16-64k` endpoint,
  `-w 8`, scored with the SWE-bench harness 4.1.0 (`resolved/submitted`). n=500 →
  **±~3.9% at 95% CI**. A shuffled-50 pilot read 80% (±11%), consistent.
- Breakdown: 365 resolved · 107 unresolved (patch applied, tests failed) · 28 no-patch
  (8 context-exceeded at 64K, 12 step-budget, ~8 Docker-startup timeouts) · 0 eval errors.
- **Not directly comparable to the official 62.1**, which is on SWE-bench **Pro** (a
  different, harder set). Ours on Verified is comparable to *other models'* Verified
  numbers, and sits in frontier range for a self-hosted INT4 model.
- **This measures GLM-5.2 + a good agent scaffold**, not the weights alone — the
  scaffold does a lot of the work. Report it as such.

## What generalizes (the operational lessons)

- **Concurrency on GB10 buys ~2×, not 5–8×.** The batched recipe tops out around
  57 tok/s aggregate at batch 16 vs ~28 single-stream — the quantized + sparse-indexer
  stack is throughput-bound, not memory-bound. A full SWE-bench 500 is still ~a day.
- **Longer context + smaller prefill batch is a trap for agentic evals.** A 128K variant
  (`mnbt` halved to hold the indexer prewarm constant) loads fine but death-spirals on
  SWE: 100K-token prefills exceed the client timeout → retry → vLLM re-prefills from
  scratch → ~1 tok/s. Dropped.
- **The stalls were the *client* timeout, not the recipe.** mini-SWE-agent's litellm
  defaults to a 600s timeout; long-context turns exceed it and trigger the re-prefill
  spiral. Fix: `-c model.model_kwargs.timeout=1800`. c16-64k also stalled at ~200/500
  until this was raised.
- **64K context truncates ~1.6% of SWE tasks** (context-exceeded → clean unresolved).
  A bounded, deliberate trade for stability/speed; 128K isn't viable (see above).

## Reproduction

- Accuracy (GPQA/etc.): `POST /api/benchmarks {deploymentId, presetId:"acc-gpqa-diamond-full", numConcurrent:8}`
  against the `c16-64k` deployment, or standalone
  `uvx --from "lm-eval[api,ifeval,math]==0.4.12" lm_eval --model local-chat-completions
  --model_args base_url=http://<endpoint>/v1/chat/completions,model=glm-5.2,num_concurrent=8,tokenized_requests=False,timeout=3600 …`.
- SWE-bench: `python -m minisweagent.run.benchmarks.swebench --subset verified --split test
  -m openai/glm-5.2 --environment-class docker -w 8 -c swebench.yaml -c model.model_kwargs.timeout=1800 -o DIR`,
  then `python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified
  --split test --predictions_path DIR/preds.json --max_workers 8 --run_id <id>`.
  mini-SWE-agent needs `MSWEA_COST_TRACKING=ignore_errors` (litellm doesn't know glm-5.2 pricing).
