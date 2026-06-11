# Tool-Eval-Bench Integration — Design

**Date:** 2026-06-11
**Status:** Approved (design); pending implementation plan
**Author:** Daniel Kreuzhofer (with Claude)

## Summary

Add a second *kind* of benchmark — **tool-eval-bench** — to the existing
Benchmarks feature. tool-eval-bench
([SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench))
is an OpenAI-compatible CLI that evaluates a model's **tool-calling
capability** (tool selection, parameter precision, multi-step chains, error
recovery, prompt-injection resistance, restraint/refusal, ambiguity
resolution, context-under-pressure) across 63 deterministic scenarios in 14
categories. It produces a **correctness score** (0–100), a star rating, a
deployability score, per-category breakdowns, and safety warnings — *not* the
throughput rows the current benchmark feature collects.

The existing benchmark feature is a throughput/latency harness built around the
`llama-benchy` CLI; every preset is a workload shape (pp/tg/depth/concurrency)
and every result is a TPS/TTFR row. tool-eval-bench is a different *kind* of
benchmark that reuses the same run lifecycle but produces a different result
shape. This design integrates it by reusing all the generic plumbing (run
lifecycle, log streaming, SSE, cancel, list/detail pages) and isolating the
three things that genuinely differ per kind behind a small strategy: **the CLI
command + args, the result parser, and the result display component.**

## Decisions (locked)

These were settled during brainstorming and are not open for re-litigation in
the plan:

1. **Data model:** extend `BenchmarkRun` with a `kind` discriminator rather than
   create a separate `ToolEvalRun` model. One unified Benchmarks page; tool-eval
   presets appear in the same launcher.
2. **Scope:** eval-only for v1. Do **not** pass `--perf` (which would shell out
   to llama-benchy for a throughput sweep) — throughput is already covered by the
   existing presets. `--perf` can be added later.
3. **Variants:** expose four presets — Short (`--short`), Full (63 scenarios),
   Hard mode (`--hardmode`), and Context pressure (`--context-pressure 0.75`).
4. **No custom-config UI** for tool-eval in v1 (the existing custom-config form
   is throughput-specific; YAGNI).

## tool-eval-bench facts (from the README + NVIDIA forum post)

- **Install/invoke (ephemeral):**
  `uvx --from "git+https://github.com/SeraphimSerapis/tool-eval-bench.git@c3868bff099592c9a1045de2c9a3dc24abebb7fb" tool-eval-bench …`
  We pin a commit SHA — the documented `git+` install is unpinned, which is
  unacceptable for reproducible benchmark records.
- **Required input:** `--base-url <openai-endpoint>`. Works with vLLM /
  llama.cpp / LiteLLM. No real upstream API calls — mock tool handlers, offline,
  deterministic.
- **Model selection:** auto-detected from `/v1/models`; if multiple models are
  returned it shows an **interactive picker**. We always pass `--model
  <servedModelName>` explicitly to avoid the picker hanging a headless run.
- **Machine-readable output:** `--json` (stdout) and `--json-file PATH` (writes
  JSON, implies `--json`). We use `--json-file <outputDir>/result.json` to match
  the existing orchestrator's `result.json` convention.
- **Actual JSON schema** (captured from a real `--short` run on 2026-06-11,
  tool-eval-bench `2.0.6`, `schema_version: "1"` — sample archived as the test
  fixture `tool-eval-result.json`). Top-level keys:
  `schema_version`, `tool_eval_bench_version`, `final_score` (int 0–100),
  `rating` (star string, e.g. `"★★★ Adequate"`), `safety_warnings` (list),
  `deployability` (int), `responsiveness` (small int — a star-style rating, **not**
  0–100; observed value `2`), `total_scenarios`, `run_id`, `status`, `config`
  (dict), `scores` (dict), `metadata` (dict), `report_path`.
  - **There is no `quality` field** (the forum post was imprecise). Do not invent
    one.
  - **`scores`** holds the detail: `final_score`, `total_points`, `max_points`,
    `rating`, `worst_category`, `worst_category_percent`, `deployability`,
    `responsiveness`, `median_turn_ms`, `total_tokens`, `token_efficiency`, plus
    two arrays:
    - **`category_scores`** — the per-category breakdown. Each element:
      `{category, label, earned, max, percent, pass_count, partial_count,
      fail_count}` (e.g. `{"category":"A","label":"Tool Selection","earned":6,
      "max":6,"percent":100,...}`). `--short` yields **5 categories**; the full
      suite yields up to 14. Variable count — the `ToolEvalCategory` table handles
      it.
    - **`scenario_results`** — one entry per scenario, each carrying a large
      `raw_log` reasoning trace (the bulk of the file size). Not persisted as
      structured rows in v1.
- **stderr progress events** (subprocess mode):
  `{"event":"scenario_start",…}`, `{"event":"scenario_result",…,"points":…}`,
  `{"event":"benchmark_complete","json_file":"…"}`. These stream as log lines
  through the existing `onLog` path with no special handling.
- **Relevant flags:** `--short` (15 core scenarios), `--hardmode` (adds Category
  P hard tier), `--context-pressure R` (0.0–1.0), `--seed N` (deterministic),
  `--model`, `--base-url`, `--json-file`, `--output-dir`.

### Schema confirmed

The category shape was an open question during brainstorming; it is now
**resolved** by the captured sample (above). Pinned upstream commit:
`c3868bff099592c9a1045de2c9a3dc24abebb7fb` (tool-eval-bench `2.0.6`). The parser
and fixture are built against this real schema, so there is no remaining
soft spot. Note: a `--short` run takes ~25 min against an 8B Ollama model
(60–320 s per scenario); against a fast vLLM serve it is much quicker. The first
`uvx` invocation also git-clones + builds the tool.

## Architecture

### Component map (what changes)

```
packages/server/src/benchmarks/
  presets.ts          (edit)  + kind field, + ToolEvalConfig type, + 4 presets
  tool-eval-args.ts   (new)   buildToolEvalArgs(config, target) -> string[]
  tool-eval-parser.ts (new)   parseToolEvalResults(jsonText) -> ToolEvalSummary
  orchestrator.ts     (edit)  dispatch command + parse by kind (strategy)
  args.ts, parser.ts  (unchanged)
  endpoint.ts         (unchanged — reused as-is)

packages/server/src/routes/benchmarks.ts  (edit)
  POST / resolves preset.kind, dispatches to the right strategy,
  persists eval fields + ToolEvalCategory rows on completion.

prisma/schema.prisma  (edit)
  BenchmarkRun + kind + toolEval* fields + toolEvalCategories relation
  ToolEvalCategory (new model)

packages/dashboard/
  components/benchmark-form-modal.tsx     (edit)  group presets by kind
  components/tool-eval-result-card.tsx    (new)   score + categories + safety
  app/.../benchmark detail + list views   (edit)  branch render by run.kind
  lib/benchmarks.ts                       (edit)  types for kind + eval fields
```

The generic spawn / line-buffered log streaming / process-group cancel /
`result.json` read in `orchestrator.ts` is **kind-agnostic and stays shared**.
Only argv construction and result parsing branch.

### Data model

`BenchmarkRun` gains:

```prisma
kind          String  @default("throughput") // "throughput" | "tool-eval"
// --- tool-eval headline metrics (null on throughput runs) ---
toolEvalScore          Float?   // scores.final_score 0–100
toolEvalRating         String?  // rating star string, e.g. "★★★ Adequate"
toolEvalDeployability  Int?     // deployability
toolEvalResponsiveness Int?     // responsiveness (small star-style int, NOT 0–100)
toolEvalTotalScenarios Int?     // total_scenarios
toolEvalTotalPoints    Int?     // scores.total_points
toolEvalMaxPoints      Int?     // scores.max_points
toolEvalSafetyWarnings String?  // JSON-encoded string[] (safety_warnings)
toolEvalCategories     ToolEvalCategory[]
```

- `kind` defaults to `"throughput"`, so every existing row stays valid with no
  data migration.
- `config` (existing `String`) holds the JSON-encoded `ToolEvalConfig` for
  eval runs, exactly as it holds `BenchmarkConfig` for throughput runs.
- `rawOutput` (existing) stores the full tool-eval `--json` payload — the source
  of truth and the rendered-verbatim fallback if structured parsing fails.
- Throughput-only fields (`meanTps`, `meanTtfrMs`, `results`) stay null/empty on
  eval runs; the eval fields stay null on throughput runs.

New table (mirrors the `BenchmarkResult` run→rows relation):

```prisma
model ToolEvalCategory {
  id           String @id @default(cuid())
  runId        String
  code         String  // category_scores[].category, e.g. "A"
  label        String  // category_scores[].label, e.g. "Tool Selection"
  percent      Float   // category_scores[].percent (0–100)
  earned       Int     // category_scores[].earned
  maxPoints    Int     // category_scores[].max
  passCount    Int     // category_scores[].pass_count
  partialCount Int     // category_scores[].partial_count
  failCount    Int     // category_scores[].fail_count
  run          BenchmarkRun @relation(fields: [runId], references: [id], onDelete: Cascade)
  @@index([runId])
}
```

Applied via `npm run db:push` — additive (new nullable columns + new table),
non-destructive. No `--force-reset`.

**`rawOutput` size:** the full `--json` payload includes a large `raw_log`
reasoning trace per scenario (~91 KB for 15 scenarios; the 63-scenario suite
will be several hundred KB). v1 stores the payload verbatim in `rawOutput`
(SQLite TEXT handles it) and the dashboard renders the *structured* fields, not
the raw blob. If row size becomes a concern, an optional follow-up is to strip
`scores.scenario_results[].raw_log` before persisting — not done in v1.

### Presets

`BenchmarkPreset` gains `kind: "throughput" | "tool-eval"` (defaulting/treated
as `"throughput"` for the five existing presets). A new config type:

```ts
export type ToolEvalConfig = {
  short: boolean;            // --short
  hardmode: boolean;         // --hardmode
  contextPressure: number | null; // --context-pressure R (null = omit)
  seed: number;              // --seed N, for reproducibility
};
```

`BenchmarkPreset.config` becomes `BenchmarkConfig | ToolEvalConfig`,
discriminated by `kind`. Four presets are appended to `BENCHMARK_PRESETS`:

| id                   | label                      | config |
|----------------------|----------------------------|--------|
| `tool-eval-quick`    | Tool eval — quick (15)     | `{ short: true,  hardmode: false, contextPressure: null, seed: 42 }` |
| `tool-eval-full`     | Tool eval — full (63)      | `{ short: false, hardmode: false, contextPressure: null, seed: 42 }` |
| `tool-eval-hardmode` | Tool eval — hard mode      | `{ short: false, hardmode: true,  contextPressure: null, seed: 42 }` |
| `tool-eval-pressure` | Tool eval — context pressure | `{ short: false, hardmode: false, contextPressure: 0.75, seed: 42 }` |

`getPreset` / `listPresets` are unchanged in signature; the `/api/benchmarks/presets`
endpoint returns all nine presets, each carrying its `kind`.

### Args builder (`tool-eval-args.ts`)

```ts
export type ToolEvalTarget = { baseUrl: string; modelName: string; outputPath: string };

export function buildToolEvalArgs(config: ToolEvalConfig, target: ToolEvalTarget): string[];
```

Produces argv (the `--from <git-spec> tool-eval-bench` prefix is added by the
orchestrator, matching how `buildBenchyArgs` output is prefixed today):

```
--base-url <baseUrl>          # already includes /v1, same as llama-benchy path
--model <modelName>           # explicit, skips the interactive picker
--json-file <outputPath>      # implies --json; writes result.json
--seed <seed>
[--short]                     # if config.short
[--hardmode]                  # if config.hardmode
[--context-pressure <R>]      # if config.contextPressure != null
```

Unit/property tested the same way as `args.test.ts`: each boolean flag appears
iff set; `--context-pressure` is present iff non-null and followed by exactly
one value token.

### Parser (`tool-eval-parser.ts`)

```ts
export type ToolEvalCategoryInput = {
  code: string; label: string; percent: number;
  earned: number; maxPoints: number;
  passCount: number; partialCount: number; failCount: number;
};
export type ToolEvalSummary = {
  finalScore: number;       // top-level final_score (== scores.final_score)
  rating: string;           // rating
  deployability: number | null;
  responsiveness: number | null;
  totalScenarios: number;
  totalPoints: number | null;  // scores.total_points
  maxPoints: number | null;    // scores.max_points
  safetyWarnings: string[];
  categories: ToolEvalCategoryInput[]; // from scores.category_scores
};
export function parseToolEvalResults(jsonText: string): ToolEvalSummary;
```

- **Fail-fast** (per Principle 3): throws with a clear message if `final_score`,
  `rating`, `total_scenarios`, or `scores.category_scores` is missing or the
  wrong type — no silent defaulting. Mirrors the `num()`/`nestedNum()` guards in
  `parser.ts`.
- `deployability` and `responsiveness` are optional → `null` if absent.
- `total_points` / `max_points` read from `scores`; `null` if absent.
- `safety_warnings` defaults to `[]` if absent.
- `categories` maps `scores.category_scores[]` directly to
  `ToolEvalCategoryInput` (1:1 field rename). Variable length (5 for `--short`,
  up to 14 for full). The fixture `tool-eval-result.json` (real captured sample,
  raw_logs trimmed) pins the test.

### Orchestrator dispatch

`runBenchmark` currently hardcodes the `uvx --from llama-benchy …` prefix and
calls `parseBenchyResults` on exit. Refactor so the kind-specific bits are
selected by a `kind` (or a passed-in strategy object) while the spawn / log
stream / process-group cancel / file read stay shared:

- **command prefix:** throughput → `["--from", LLAMA_BENCHY_SPEC, "llama-benchy", …args]`;
  tool-eval → `["--from", TOOL_EVAL_SPEC, "tool-eval-bench", …args]` where
  `TOOL_EVAL_SPEC = "git+https://github.com/SeraphimSerapis/tool-eval-bench.git@c3868bff099592c9a1045de2c9a3dc24abebb7fb"`
  (overridable via `TOOL_EVAL_BENCH_REF` env, like `LLAMA_BENCHY_VERSION`).
- **parse on exit:** throughput → `parseBenchyResults` + `summarizeResults`;
  tool-eval → `parseToolEvalResults`. The return type widens to carry either a
  throughput summary or a tool-eval summary (a discriminated result), so the
  route writes the right columns.

`PYTHONUNBUFFERED=1`, `detached:true`, SIGTERM-to-process-group cancel, and the
`onLog` line splitting are unchanged and apply to tool-eval as-is.

### Server route

`POST /api/benchmarks` changes minimally:

- Resolve the preset, read `preset.kind`, store it on the new row's `kind`.
- `config` validation: for `kind === "tool-eval"`, the body must use a preset
  (no custom config path in v1) — reject custom `config` for tool-eval with 400.
- Endpoint URL derivation (`deploymentEndpointUrl(deployment) + "/v1"`) and the
  `servedModelName` snapshot are reused unchanged.
- Dispatch to the tool-eval args builder + strategy.
- On `exitCode === 0`: write `toolEvalScore`, `toolEvalRating`,
  `toolEvalDeployability`, `toolEvalResponsiveness`, `toolEvalTotalScenarios`,
  `toolEvalTotalPoints`, `toolEvalMaxPoints`, `toolEvalSafetyWarnings` (JSON),
  `rawOutput`, and `toolEvalCategories: { create: [...] }`. (Throughput runs
  continue to write `meanTps`/`meanTtfrMs`/`results` as today.)
- All existing guards (deploymentId required, deployment running, no concurrent
  in-flight run), SSE events (`benchmark:created`, `benchmark:status`,
  `benchmark:log`, `benchmark:deleted`), and the logs endpoint are unchanged and
  apply to both kinds.
- `GET /:id` includes `toolEvalCategories` (alongside `results`) so the detail
  view has everything in one fetch.

### Dashboard

- **Launcher modal** (`benchmark-form-modal.tsx`): presets already render
  dynamically from `/presets`; group them under two headers ("Throughput" /
  "Tool-calling eval") by `kind`. Custom-config controls render only for
  throughput presets.
- **Detail view:** branch on `run.kind`. Throughput → existing
  `benchmark-result-table` + `benchmark-chart`. Tool-eval → new
  `tool-eval-result-card`: final score + star rating, deployability/quality/
  responsiveness, horizontal per-category bars, and a callout for any
  `safetyWarnings`. The raw log panel and `rawOutput` fallback are shared.
- **List view:** add a `kind` badge; the summary column shows the eval score for
  tool-eval runs and `meanTps` for throughput runs.
- **`lib/benchmarks.ts`:** extend the run/preset/result TS types with `kind`,
  the `toolEval*` fields, and `toolEvalCategories`.

## Error handling

- Non-zero exit → row `status: "failed"`, `error: "tool-eval-bench exited with
  code N"` (parallel to the existing llama-benchy message). Partial/garbage
  `result.json` → parser throws, caught by the existing `.catch`, row marked
  failed with the parser message. (Same shape as today's `[parser]` path.)
- Cancel mid-run → SIGTERM to the process group; if the row was already flipped
  to `"canceled"`, the completion handler leaves it alone (existing logic).
- Interactive-picker hang is prevented structurally by always passing `--model`.

## Testing (medium/high tier per CLAUDE.md)

- **Unit/property — `tool-eval-args.test.ts`:** correct argv for each variant;
  boolean flags appear iff set; `--context-pressure` present iff non-null with
  exactly one value token (property test mirroring `args.test.ts`).
- **Unit — `tool-eval-parser.test.ts`:** maps headline fields + categories from a
  fixture `result.json`; **throws** on missing `final_score` / `rating` /
  `total_scenarios`; tolerates absent optional fields (deployability, warnings).
- **Orchestrator — `orchestrator.test.ts`:** dispatches by kind; spawns
  `tool-eval-bench` with the pinned `--from` spec, `--model`, `--json-file`,
  `detached:true`, and the variant flags.
- **Integration — `benchmarks.routes.test.ts`:** `/presets` includes the four
  tool-eval presets each with `kind:"tool-eval"`; `POST` with a tool-eval preset
  creates a `kind:"tool-eval"` run; a completed run persists the eval headline
  fields + `ToolEvalCategory` rows (drive via a stubbed orchestrator result, as
  the existing route tests do).
- **Fixture:** `benchmarks.fixtures/tool-eval-result.json` — captured from a real
  `--json` run during implementation; pins the category schema.
- **No agent code changes → no agent version bump** (orchestration is
  server-side; `packages/agent/src/` is untouched).

`npm test` must be green before the work is claimed done.

## Out of scope (v1)

- `--perf` throughput sweep inside a tool-eval run.
- Custom tool-eval config UI (variants are preset-only).
- `--diff` / `--resume` / `--history` / `--trials` cross-run comparison features.
- Leaderboard / `config_fingerprint` grouping.

## Risks & mitigations

- **Category JSON shape unknown** → capture a real sample first; isolated to the
  parser + fixture; headline contract is documented and firm.
- **Unpinned upstream CLI** → pin a commit SHA in `TOOL_EVAL_SPEC`; override via
  env for upgrades.
- **First `uvx` install latency** (git clone + build) → acceptable; surfaces as
  log output during the "running" phase, same UX as llama-benchy's first run.
