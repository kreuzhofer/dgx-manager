# Accuracy-eval benchmark kind (lm-eval-harness) — Design

**Date:** 2026-07-09
**Status:** Approved (design); implementation plan pending
**Roadmap:** Phase 5 — Evaluation & Benchmarks (`docs/ROADMAP.md:248`)

## Goal

Add a third benchmark `kind` — **`accuracy`** — to DGX Manager's benchmark suite, powered by
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (`lm-eval`), alongside
the shipped `throughput` (llama-benchy) and `tool-eval` (tool-eval-bench) kinds. It runs curated
academic benchmarks against a running deployment's OpenAI-compatible `/v1` endpoint and surfaces a
comparable accuracy score plus a per-task breakdown in the dashboard.

**Why:** makes the manual "deploy A → eval → deploy B → eval → compare" loop one-click and
dashboard-comparable. Directly serves two live decisions: the **prune-quality decision**
(GLM-5.2 15pct vs unpruned) and **fine-tune regression** checks. IFEval in particular targets the
*known* expert-prune failure mode (instruction adherence).

## Non-goals (v1)

- **No agent / node dispatch.** Runs server-side on the manager via `uvx`, exactly like the other
  two kinds. The GPU nodes are reserved exclusively for model hosting and are **not** an execution
  target. A future dedicated benchmark host is a deployment change, not an architectural one (see
  Execution model).
- **No multiple-choice / loglikelihood tasks** (MMLU-classic, ARC, HellaSwag, …). They need the
  endpoint to return prompt logprobs and are the wrong mode for a reasoning model like GLM-5.2. All
  v1 benchmarks are generative.
- **No code-execution benchmarks** (HumanEval / MBPP). Executing model-written code needs a sandbox
  we don't have yet.
- **No new SQL-eval integration.** The existing SQL eval is a standalone script in the fine-tune
  recipes repo, not an lm-eval task; out of scope here.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Execution location | Server-side via `uvx` on the manager (Pi now; dedicated host later). Not the GPU nodes. |
| Benchmark lineup | IFEval, MMLU-Pro (CoT), GPQA-Diamond (CoT), GSM8K, BBH, MATH-hard — i.e. HF Open LLM Leaderboard v2 minus MuSR |
| Runtime strategy | A **quick** (`--limit`ed) and a **full** variant per benchmark → ~12 presets |
| Reasoning handling | Per-preset `reasoning` toggle: `--apply_chat_template` + large `max_gen_toks` + an **ephemeral strip proxy** that removes `<think>…</think>` before lm-eval scores |
| Data model | Two nullable columns on `BenchmarkRun` (`accuracyScore`, `accuracyMetrics` JSON); no new table |

## Architecture

### Execution model

A run reuses the existing benchmark lifecycle in `packages/server/src/benchmarks/` and
`routes/benchmarks.ts`:

1. `POST /api/benchmarks { deploymentId, presetId }` → create a `BenchmarkRun` row (`kind: "accuracy"`).
2. Resolve the deployment's `/v1` endpoint and its actual served model name
   (`resolveServedModelName`, reused unchanged).
3. Spawn `uvx --from lm-eval[...] lm_eval …` against the endpoint (or the strip proxy, see below).
4. Stream stdout/stderr line-by-line to SSE (`benchmark:log`) and the per-run log file
   (reused unchanged).
5. On exit, locate + parse lm-eval's results JSON, persist headline score + breakdown, broadcast
   `benchmark:status`.

The lm-eval package spec is pinned via a new `LM_EVAL_VERSION` env var (default: a pinned version),
mirroring `LLAMA_BENCHY_VERSION` / `TOOL_EVAL_BENCH_REF`. Because the endpoint URL is passed in and
nothing is hard-wired to "the Pi", moving execution to a dedicated benchmark host later is a
deployment/ops change, not a code rewrite.

### lm-eval invocation

Model type: **`local-chat-completions`** — hits `/v1/chat/completions`, which every deploy serves,
and needs no logprobs. Base argv (built by `buildLmEvalArgs`):

```
uvx --from "lm-eval[ifeval,math]" lm_eval \
  --model local-chat-completions \
  --model_args base_url=<endpointV1Url>/chat/completions,model=<servedModel>,num_concurrent=1,tokenized_requests=False \
  --tasks <task[,task…]> \
  --apply_chat_template \
  --gen_kwargs max_gen_toks=<n> \
  [--num_fewshot <n>] \
  [--limit <n>] \
  --seed <n> \
  --output_path <outputDir>
```

Notes:
- `<endpointV1Url>` is the same `.../v1` URL the other kinds already build (`deploymentEndpointUrl(d) + "/v1"`),
  or the strip proxy's `/v1` URL. `local-chat-completions` wants `base_url` to be the **full**
  `/v1/chat/completions` path, so `buildLmEvalArgs` appends `/chat/completions`.
- `tokenized_requests=False` avoids lm-eval trying to load a local HF tokenizer for the served model.
- `num_concurrent=1` — a single slow reasoning endpoint; can be surfaced later if needed.
- **`uvx` extras are per-task.** v1 needs at least `[ifeval]` (IFEval scoring: `langdetect`,
  `immutabledict`, `nltk`) and `[math]` (MATH-hard answer checking: `sympy`,
  `antlr4-python3-runtime`). The exact extra set is finalized at build time when task ids are verified.
- **lm-eval writes to a nested `results_*.json`** under `--output_path` (e.g.
  `<outputDir>/<sanitized-model>/results_<timestamp>.json`), **not** a fixed path like
  llama-benchy's `--save-result`. So the orchestrator's result reader is generalized (below).

### Reasoning handling — ephemeral strip proxy

The whole motivation is evaluating **GLM-5.2, a reasoning model**. Two failure modes to defeat:
1. **Cut off mid-think** → no final answer → score 0. Mitigation: a large `max_gen_toks`
   (per-preset default, e.g. 4096) and `--apply_chat_template`.
2. **Thinking text confuses answer extraction** (model states a wrong answer mid-reasoning, then
   corrects). Mitigation: strip the reasoning block before lm-eval's task filters run.

**Chosen mechanism — ephemeral local HTTP proxy** (not per-task YAML filter forks):

- `startReasoningProxy(targetV1BaseUrl)` starts an HTTP server on `127.0.0.1:0` (ephemeral port).
- It forwards requests to the deployment and, on `/v1/chat/completions` responses, rewrites each
  `choices[].message.content` through a pure `stripReasoning(content)` before returning.
- lm-eval runs **stock, unmodified tasks** pointed at the proxy URL → robust, task-agnostic, and
  free for future benchmarks.
- Torn down in a `finally` block and on cancel.

`stripReasoning(content)` (pure, unit-tested) removes `<think>…</think>` blocks; if a stray
`</think>` remains (the opening `<think>` was injected by the chat template and not echoed), it
drops everything up to and including the last `</think>`. No-op when there are no reasoning tags, so
it is harmless for non-reasoning models.

When a preset has `reasoning: false`, no proxy is started and lm-eval points straight at the
deployment endpoint.

### Presets

New `AccuracyConfig` type in `presets.ts`:

```ts
export type AccuracyConfig = {
  tasks: string[];          // lm-eval task/group names
  primaryTask: string;      // which task's metric is the headline (∈ tasks)
  primaryMetric: string;    // metric key without the ",none" suffix, e.g. "exact_match"
  limit: number | null;     // --limit N (quick) or null (full)
  numFewshot: number | null;// --num_fewshot; null = task default
  maxGenToks: number;       // --gen_kwargs max_gen_toks=N
  applyChatTemplate: boolean;
  reasoning: boolean;       // strip <think>…</think> via proxy + bigger budget
  seed: number;
};
```

`BenchmarkKind` becomes `"throughput" | "tool-eval" | "accuracy"`.

Six benchmarks × {quick, full} = 12 presets, grouped under "Accuracy (lm-eval)" in the form. Task
names below are the intended targets and **must be verified against the pinned lm-eval version at
build time** (task ids drift across versions):

| Preset id | lm-eval task | primaryMetric | quick limit |
|---|---|---|---|
| `acc-ifeval-quick` / `-full` | `ifeval` | `prompt_level_strict_acc` | 100 |
| `acc-mmlu-pro-quick` / `-full` | `mmlu_pro` | `exact_match` | 200 |
| `acc-gpqa-diamond-quick` / `-full` | `gpqa_diamond_cot_zeroshot` | `exact_match` | 50 |
| `acc-gsm8k-quick` / `-full` | `gsm8k_cot` | `exact_match` | 200 |
| `acc-bbh-quick` / `-full` | `bbh_cot_zeroshot` | `exact_match` | 40 |
| `acc-math-hard-quick` / `-full` | `leaderboard_math_hard` | `exact_match` | 100 |

All carry `reasoning: true`, `applyChatTemplate: true`, `seed: 42`, and a per-preset `maxGenToks`
(CoT/MATH/BBH larger than IFEval). `full` variants set `limit: null`. `numFewshot` left `null`
(task default) unless a specific task needs otherwise.

BBH and MATH-hard are task **groups** (multiple subtasks with a group aggregate); their `primaryTask`
is the group name and the parser reads the group-level metric for the headline while emitting per-
subtask rows into the breakdown.

### Data model

`BenchmarkRun.kind` gains the value `"accuracy"` — it is a free-text `String`, so no enum migration.
Two new **nullable** columns (honors the roadmap's "reuse the JSON pattern, no new columns/table"
intent — this is leaner than what `tool-eval` added):

```prisma
// --- accuracy / lm-eval headline + breakdown (null on other kinds) ---
accuracyScore   Float?   // primary metric ×100 (0–100), for list/compare
accuracyMetrics String?  // JSON: [{task, metric, value, stderr?, isGroup?, nSamples?}]
```

`accuracyScore` is analogous to `meanTps` / `toolEvalScore` (headline for list/compare without
re-parsing). `accuracyMetrics` is analogous to the JSON-string `toolEvalSafetyWarnings` (rendered
client-side on the detail page — the compare/list views never need to query per-task in SQL).
`config`, `rawOutput`, `status`, `error`, `startedAt`, `completedAt` are reused unchanged. Applied
via `npm run db:push`.

### Parser

`parseLmEvalResults(jsonText, primaryTask, primaryMetric)` in `benchmarks/lm-eval-parser.ts`:

- Walks the lm-eval JSON `results` (and `groups`) objects. For each task, collects metric keys,
  stripping the `,none` filter suffix; pairs each metric with its `_stderr` sibling; records
  `n-samples` when present; flags group-level rows with `isGroup`.
- Produces `{ primaryScore: number /* 0–100 */, metrics: AccuracyMetricInput[] }`.
- **Fail-fast:** if `results[primaryTask][primaryMetric + ",none"]` is missing or non-numeric, throw
  a clear error → the run is marked `failed` with that message. No silent default.

### Orchestrator

`runAccuracy(opts)` in `orchestrator.ts`, mirroring `runBenchmark` / `runToolEval`:

- If `reasoning`, start the strip proxy and use its URL; else use the endpoint directly.
- Spawn via the shared `spawnTracked`, generalized to accept an optional **result-file locator**
  (default: `join(outputDir, "result.json")`; lm-eval passes a locator that finds the newest
  `results_*.json` under `outputDir`).
- Parse with `parseLmEvalResults`; return `{ exitCode, primaryScore, metrics, rawOutput }`.
- Always close the proxy in `finally`. `cancelBenchmark` continues to SIGTERM the process group;
  the proxy close is handled by the `finally`.

### Route

In `routes/benchmarks.ts` `POST /`, add an `accuracy` branch alongside `tool-eval` / `throughput`:
on success, persist `status: "completed"`, `rawOutput`, `accuracyScore`, `accuracyMetrics`
(JSON-stringified), then broadcast the final run. The custom `config` path stays **throughput-only**
(accuracy and tool-eval require a preset) — unchanged contract.

### Dashboard

- `lib/benchmarks.ts`: add `AccuracyConfig`, extend `BenchmarkKind`, add `accuracyScore` /
  `accuracyMetrics` to `BenchmarkRun`, and an `AccuracyMetric` row type.
- `benchmark-form-modal.tsx`: add `"accuracy"` to the kind-group loop with the "Accuracy (lm-eval)"
  heading; render the accuracy presets (label + description + quick/full + limit hint).
- `benchmarks/[id]/page.tsx`: when `kind === "accuracy"`, render a new `accuracy-result-card.tsx`
  (headline score + a table of the per-task/subtask breakdown parsed from `accuracyMetrics`).
- `benchmarks/compare/page.tsx`: add an accuracy score comparison (bar per run) when comparing
  accuracy runs.

## Known caveats (must surface in the build)

- **GPQA is a gated HF dataset** (`Idavidrein/gpqa`) → requires `HF_TOKEN` with granted access in the
  server/uvx environment. Documented; the GPQA preset must surface a clear error if unauthenticated
  rather than a cryptic download failure.
- **First run per task downloads the dataset from HF** (network dependency). uv and the HF cache
  amortize subsequent runs.
- **lm-eval task ids drift** across versions — pin `LM_EVAL_VERSION` and verify the six task ids at
  build time.
- **MMLU-Pro full is ~12k CoT generations** — impractical on the Pi-driven slow endpoint; that's why
  every preset ships a quick variant and full is opt-in.

## Testing (medium/high risk)

Per `CLAUDE.md` risk tiers — new endpoint branch, new config, new parsing:

- **Property** (`stripReasoning`): idempotent; removes `<think>…</think>`; preserves the trailing
  answer; no-op without tags.
- **Unit** (`parseLmEvalResults`): fixture of a real lm-eval JSON (single task + a group task);
  primary-metric extraction; per-subtask rows; **fail-fast** on missing primary metric.
- **Unit** (`buildLmEvalArgs`): argv correctness — `--limit` omitted when null, `--gen_kwargs`
  present, tasks comma-joined, `--num_fewshot` gated.
- **Unit** (presets): every accuracy preset has `primaryTask ∈ tasks` and a non-empty
  `primaryMetric`; both quick and full variants exist per benchmark.
- **Integration** (supertest, stubbed orchestrator/spawn): `POST /api/benchmarks` with an accuracy
  `presetId` creates a `kind: "accuracy"` run; error paths — unknown preset, deployment not running,
  concurrent-run 409.
- **Integration** (reasoning proxy): a fake upstream returns `<think>…</think>answer`; assert the
  proxy forwards a stripped body.

`npm test` must be green before the work is claimed done. Anything genuinely environmental (a real
lm-eval run against a live GLM-5.2 deploy) is verified manually and noted.

## Rollout

Small, reversible, additive: a new kind next to two working ones, two nullable columns, no change to
existing behavior or contracts. Ship + validate manually with an IFEval quick run against a live
deploy first (the prune-quality use case), then the rest.
