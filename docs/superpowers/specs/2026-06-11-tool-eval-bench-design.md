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
  `uvx --from "git+https://github.com/SeraphimSerapis/tool-eval-bench.git@<pinned-sha>" tool-eval-bench …`
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
- **Documented JSON schema fields:** `schema_version`,
  `tool_eval_bench_version`, `final_score` (0–100), `rating` (star string),
  `safety_warnings` (list), `deployability` (int/None), `total_scenarios`. The
  forum post additionally describes `quality` (0–100), `responsiveness` (0–100),
  and a per-category breakdown with percentages.
- **stderr progress events** (subprocess mode):
  `{"event":"scenario_start",…}`, `{"event":"scenario_result",…,"points":…}`,
  `{"event":"benchmark_complete","json_file":"…"}`. These stream as log lines
  through the existing `onLog` path with no special handling.
- **Relevant flags:** `--short` (15 core scenarios), `--hardmode` (adds Category
  P hard tier), `--context-pressure R` (0.0–1.0), `--seed N` (deterministic),
  `--model`, `--base-url`, `--json-file`, `--output-dir`.

### Known unknown

The exact JSON shape of the **per-category breakdown** is not fully documented.
The headline fields above *are* documented, so the parser's required-field
contract (`final_score`, `rating`, `total_scenarios`) is firm. During
implementation we capture one real `--json` sample (run against a deployed
cluster model, e.g. a small model on the Spark) to pin the category-array
structure and the parser fixture. **If the category shape differs from the
assumption below, adjust the parser + fixture only — no schema or route change
is required**, because categories land in a dedicated table fed by the parser.

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
toolEvalScore          Float?   // final_score 0–100
toolEvalRating         String?  // star rating string
toolEvalDeployability  Int?
toolEvalQuality        Int?
toolEvalResponsiveness Int?
toolEvalTotalScenarios Int?
toolEvalSafetyWarnings String?  // JSON-encoded string[]
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
  id        String  @id @default(cuid())
  runId     String
  name      String          // e.g. "Tool selection"
  score     Float           // 0–100 percentage
  points    Int?            // achieved points, if reported
  maxPoints Int?            // possible points, if reported
  run       BenchmarkRun @relation(fields: [runId], references: [id], onDelete: Cascade)
  @@index([runId])
}
```

Applied via `npm run db:push` — additive (new nullable columns + new table),
non-destructive. No `--force-reset`.

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
export type ToolEvalCategoryInput = { name: string; score: number; points: number | null; maxPoints: number | null };
export type ToolEvalSummary = {
  finalScore: number;
  rating: string;
  deployability: number | null;
  quality: number | null;
  responsiveness: number | null;
  totalScenarios: number;
  safetyWarnings: string[];
  categories: ToolEvalCategoryInput[];
};
export function parseToolEvalResults(jsonText: string): ToolEvalSummary;
```

- **Fail-fast** (per Principle 3): throws with a clear message if `final_score`,
  `rating`, or `total_scenarios` is missing or the wrong type — no silent
  defaulting. Mirrors the `num()`/`nestedNum()` guards in `parser.ts`.
- `deployability`, `quality`, `responsiveness` are optional (the schema documents
  `deployability` as `int/None`); absent → `null`.
- `safety_warnings` defaults to `[]` if absent.
- `categories` is mapped from the breakdown array; its exact key names are
  confirmed against the captured sample during implementation. If the sample
  reveals a different structure, only this function + its fixture change.

### Orchestrator dispatch

`runBenchmark` currently hardcodes the `uvx --from llama-benchy …` prefix and
calls `parseBenchyResults` on exit. Refactor so the kind-specific bits are
selected by a `kind` (or a passed-in strategy object) while the spawn / log
stream / process-group cancel / file read stay shared:

- **command prefix:** throughput → `["--from", LLAMA_BENCHY_SPEC, "llama-benchy", …args]`;
  tool-eval → `["--from", TOOL_EVAL_SPEC, "tool-eval-bench", …args]` where
  `TOOL_EVAL_SPEC = "git+https://github.com/SeraphimSerapis/tool-eval-bench.git@<pinned-sha>"`
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
  `toolEvalDeployability`, `toolEvalQuality`, `toolEvalResponsiveness`,
  `toolEvalTotalScenarios`, `toolEvalSafetyWarnings` (JSON), `rawOutput`, and
  `toolEvalCategories: { create: [...] }`. (Throughput runs continue to write
  `meanTps`/`meanTtfrMs`/`results` as today.)
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
