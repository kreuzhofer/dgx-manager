# Accuracy-eval Benchmark Kind (lm-eval-harness) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third benchmark `kind` — `accuracy` — powered by lm-evaluation-harness (`lm-eval`), running curated generative academic benchmarks (IFEval, MMLU-Pro, GPQA-Diamond, GSM8K, BBH, MATH-hard) against a deployment's `/v1` endpoint, surfaced one-click in the dashboard alongside `throughput` and `tool-eval`.

**Architecture:** Server-side `uvx lm_eval` in API mode (`local-chat-completions`), spawned by the existing `orchestrator.ts` — no agent involvement. For reasoning models, a per-run ephemeral localhost HTTP proxy strips `<think>…</think>` from responses before lm-eval scores them, so stock lm-eval tasks run unmodified. Results persist as one headline `accuracyScore` column plus a JSON `accuracyMetrics` breakdown on `BenchmarkRun` — no new table.

**Tech Stack:** TypeScript (strict, ESM), Express 5, Prisma + SQLite, Vitest + fast-check + supertest, Next.js 15 dashboard, `uvx` for the Python benchmark CLI.

**Spec:** `docs/superpowers/specs/2026-07-09-accuracy-eval-lm-eval-benchmark-design.md`

## Global Constraints

- **TypeScript strict mode, ES modules.** All intra-package imports use the `.js` extension (e.g. `./presets.js`).
- **NO agent version bump.** This plan touches zero files under `packages/agent/src/` — do not run `scripts/bump-agent-version.sh`.
- **lm-eval version pin:** the `uvx` spec is `lm-eval[ifeval,math]`, overridable via the `LM_EVAL_VERSION` env var → `lm-eval[ifeval,math]==$LM_EVAL_VERSION`. Mirrors `LLAMA_BENCHY_VERSION` / `TOOL_EVAL_BENCH_REF` in `orchestrator.ts`.
- **lm-eval model type:** `local-chat-completions`; `base_url` must be the full `.../v1/chat/completions` path; always pass `tokenized_requests=False` and `num_concurrent=1`.
- **Six benchmarks × {quick, full} = 12 accuracy presets**, ids `acc-<bench>-{quick,full}`, all `kind: "accuracy"`, all `reasoning: true`, `applyChatTemplate: true`, `seed: 42`. quick sets `limit`; full sets `limit: null`.
- **Fail-fast, no silent fallback:** the parser throws on a missing primary metric; the run is then marked `failed` with that message.
- **Data model:** two nullable columns on `BenchmarkRun` (`accuracyScore Float?`, `accuracyMetrics String?`). No new table. Prisma schema at `prisma/schema.prisma`.
- **`npm test` must be green** before any task is considered done (run from repo root: `npm test`).
- **Prisma schema location:** `prisma/schema.prisma`. Apply changes with `npm run db:push` then `npm run db:generate` (root scripts).

## File Structure

**Server — new files (`packages/server/src/benchmarks/`):**
- `lm-eval-args.ts` — pure `buildLmEvalArgs(config, target)` → lm_eval argv.
- `reasoning.ts` — pure `stripReasoning(content)` → removes `<think>…</think>`.
- `reasoning-proxy.ts` — `startReasoningProxy(targetV1Url)` → ephemeral localhost strip proxy.
- `lm-eval-parser.ts` — `parseLmEvalResults(json, primaryTask, primaryMetric)` → `{ primaryScore, metrics }`.
- `lm-eval-result-file.ts` — `findLmEvalResultFile(outputDir)` → newest `results_*.json` path or null.
- Test files co-located: `lm-eval-args.test.ts`, `reasoning.test.ts`, `reasoning-proxy.test.ts`, `lm-eval-parser.test.ts`, `lm-eval-result-file.test.ts`.
- Fixture: `packages/server/src/__tests__/integration/benchmarks.fixtures/lm-eval-result.json`.

**Server — modified:**
- `prisma/schema.prisma` — two columns on `BenchmarkRun`.
- `packages/server/src/benchmarks/presets.ts` (+ `presets.test.ts`) — `AccuracyConfig`, `"accuracy"` kind, 12 presets.
- `packages/server/src/benchmarks/orchestrator.ts` (+ `orchestrator.test.ts`) — generalize `spawnTracked` result locator, add `runAccuracy`.
- `packages/server/src/routes/benchmarks.ts` (+ `__tests__/integration/benchmarks.routes.test.ts`) — accuracy POST branch + persistence.
- `packages/server/src/openapi.ts` — mention lm-eval in the Benchmarks tag description.

**Dashboard — modified/new:**
- `packages/dashboard/lib/benchmarks.ts` — types.
- `packages/dashboard/components/benchmark-form-modal.tsx` — accuracy preset section.
- `packages/dashboard/components/accuracy-result-card.tsx` — **new** result card.
- `packages/dashboard/app/benchmarks/[id]/page.tsx` — render the card.
- `packages/dashboard/app/benchmarks/page.tsx` — kind pill, score column, filter options.
- `packages/dashboard/app/benchmarks/compare/page.tsx` — accuracy score comparison.

---

## Task 1: Prisma schema — accuracy columns

**Files:**
- Modify: `prisma/schema.prisma` (model `BenchmarkRun`)

**Interfaces:**
- Produces: `BenchmarkRun.accuracyScore: Float?`, `BenchmarkRun.accuracyMetrics: String?` on the generated Prisma client (consumed by Tasks 8, 9).

- [ ] **Step 1: Add the columns.** In `prisma/schema.prisma`, inside `model BenchmarkRun`, immediately after the `toolEvalSafetyWarnings String?` line, add:

```prisma
  // --- accuracy / lm-eval headline + breakdown (null on other kinds) ---
  // Primary metric ×100 (0–100), surfaced in list/compare like meanTps/toolEvalScore.
  accuracyScore   Float?
  // JSON-encoded AccuracyMetricInput[]: per-task/per-subtask breakdown for the
  // detail page (analogous to the JSON-string toolEvalSafetyWarnings).
  accuracyMetrics String?
```

Also update the `kind` field comment to list the third value:

```prisma
  kind            String   @default("throughput") // throughput | tool-eval | accuracy
```

- [ ] **Step 2: Apply the schema to the dev DB and regenerate the client.**

Run: `npm run db:push && npm run db:generate`
Expected: `db push` reports the schema is in sync (adds two columns); `generate` writes the client without error.

- [ ] **Step 3: Verify the generated client has the fields.**

Run: `grep -n "accuracyScore\|accuracyMetrics" packages/server/src/generated/prisma/models/BenchmarkRun.ts`
Expected: both identifiers appear.

- [ ] **Step 4: Confirm the suite still compiles/passes.**

Run: `npm test`
Expected: PASS (no behavior change yet; this confirms the regenerated client is consistent).

- [ ] **Step 5: Commit.**

```bash
git add prisma/schema.prisma packages/server/src/generated/prisma
git commit -m "feat(benchmarks): add accuracyScore + accuracyMetrics columns to BenchmarkRun"
```

---

## Task 2: presets.ts — AccuracyConfig type + 12 accuracy presets

**Files:**
- Modify: `packages/server/src/benchmarks/presets.ts`
- Test: `packages/server/src/benchmarks/presets.test.ts`

**Interfaces:**
- Produces: `AccuracyConfig` type; `BenchmarkKind` widened to `"throughput" | "tool-eval" | "accuracy"`; 12 presets `acc-{ifeval,mmlu-pro,gpqa-diamond,gsm8k,bbh,math-hard}-{quick,full}`. Consumed by Tasks 3, 8, 9, 10.

```ts
export type AccuracyConfig = {
  tasks: string[];           // lm-eval task/group names
  primaryTask: string;       // headline task (∈ tasks)
  primaryMetric: string;     // metric key without ",<filter>", e.g. "exact_match"
  limit: number | null;      // --limit N (quick) or null (full)
  numFewshot: number | null; // --num_fewshot; null = task default
  maxGenToks: number;        // --gen_kwargs max_gen_toks=N
  applyChatTemplate: boolean;
  reasoning: boolean;
  seed: number;
};
```

- [ ] **Step 1: Write the failing tests.** In `presets.test.ts`, (a) replace the exhaustive id-list expectation in the first test with the full 21-id sorted array, (b) widen the `kind`-contains assertion, and (c) add an accuracy describe block. Apply these edits:

Replace the array literal in the `"exposes the five throughput presets plus four tool-eval presets by id"` test body with:

```ts
    expect(listPresets().map((p) => p.id).sort()).toEqual([
      "acc-bbh-full",
      "acc-bbh-quick",
      "acc-gpqa-diamond-full",
      "acc-gpqa-diamond-quick",
      "acc-gsm8k-full",
      "acc-gsm8k-quick",
      "acc-ifeval-full",
      "acc-ifeval-quick",
      "acc-math-hard-full",
      "acc-math-hard-quick",
      "acc-mmlu-pro-full",
      "acc-mmlu-pro-quick",
      "chat-long",
      "chat-short",
      "code-32k",
      "quick-smoke",
      "throughput",
      "tool-eval-full",
      "tool-eval-hardmode",
      "tool-eval-pressure",
      "tool-eval-quick",
    ]);
```

Replace the body of the `"every preset carries a kind field"` test with:

```ts
    for (const p of BENCHMARK_PRESETS) {
      expect(["throughput", "tool-eval", "accuracy"]).toContain(p.kind);
    }
```

Append a new describe block at the end of the file:

```ts
describe("accuracy presets", () => {
  const benches = ["ifeval", "mmlu-pro", "gpqa-diamond", "gsm8k", "bbh", "math-hard"];

  it("registers a quick and a full variant per benchmark, all kind 'accuracy'", () => {
    for (const b of benches) {
      const quick = getPreset(`acc-${b}-quick`);
      const full = getPreset(`acc-${b}-full`);
      expect(quick, `acc-${b}-quick should exist`).toBeDefined();
      expect(full, `acc-${b}-full should exist`).toBeDefined();
      expect(quick!.kind).toBe("accuracy");
      expect(full!.kind).toBe("accuracy");
    }
  });

  it("quick variants set a numeric limit; full variants set limit null", () => {
    for (const b of benches) {
      const quick = getPreset(`acc-${b}-quick`)!.config as import("./presets.js").AccuracyConfig;
      const full = getPreset(`acc-${b}-full`)!.config as import("./presets.js").AccuracyConfig;
      expect(typeof quick.limit).toBe("number");
      expect(full.limit).toBeNull();
    }
  });

  it("every accuracy preset has primaryTask ∈ tasks and a non-empty primaryMetric", () => {
    for (const p of listPresets().filter((p) => p.kind === "accuracy")) {
      const cfg = p.config as import("./presets.js").AccuracyConfig;
      expect(cfg.tasks.length).toBeGreaterThan(0);
      expect(cfg.tasks).toContain(cfg.primaryTask);
      expect(cfg.primaryMetric.length).toBeGreaterThan(0);
      expect(cfg.reasoning).toBe(true);
      expect(cfg.applyChatTemplate).toBe(true);
      expect(cfg.maxGenToks).toBeGreaterThan(0);
    }
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `npx vitest run packages/server/src/benchmarks/presets.test.ts`
Expected: FAIL — accuracy presets don't exist yet / `AccuracyConfig` not exported.

- [ ] **Step 3: Implement the presets.** In `presets.ts`:

Widen the kind union:

```ts
export type BenchmarkKind = "throughput" | "tool-eval" | "accuracy";
```

Add the `AccuracyConfig` type (after `ToolEvalConfig`):

```ts
export type AccuracyConfig = {
  tasks: string[];
  primaryTask: string;
  primaryMetric: string;
  limit: number | null;
  numFewshot: number | null;
  maxGenToks: number;
  applyChatTemplate: boolean;
  reasoning: boolean;
  seed: number;
};
```

Widen the preset config union:

```ts
export type BenchmarkPreset = {
  id: string;
  label: string;
  description: string;
  kind: BenchmarkKind;
  config: BenchmarkConfig | ToolEvalConfig | AccuracyConfig;
};
```

Add the generator just above the `export const BENCHMARK_PRESETS` declaration:

```ts
type AccuracyBench = {
  idBase: string;
  label: string;
  task: string;
  primaryMetric: string;
  quickLimit: number;
  maxGenToks: number;
  blurb: string;
};

// The v1 lineup: HF Open LLM Leaderboard v2 minus MuSR, all generative/CoT so
// they run over an OpenAI chat endpoint. Task ids are pinned against the
// LM_EVAL_VERSION set in orchestrator.ts — verify them when bumping that pin.
const ACCURACY_BENCHES: AccuracyBench[] = [
  { idBase: "ifeval", label: "IFEval", task: "ifeval", primaryMetric: "prompt_level_strict_acc", quickLimit: 100, maxGenToks: 2048, blurb: "Instruction-following adherence." },
  { idBase: "mmlu-pro", label: "MMLU-Pro (CoT)", task: "mmlu_pro", primaryMetric: "exact_match", quickLimit: 200, maxGenToks: 4096, blurb: "Knowledge/reasoning tail, chain-of-thought." },
  { idBase: "gpqa-diamond", label: "GPQA-Diamond (CoT)", task: "gpqa_diamond_cot_zeroshot", primaryMetric: "exact_match", quickLimit: 50, maxGenToks: 4096, blurb: "Hard graduate-level Q&A, chain-of-thought." },
  { idBase: "gsm8k", label: "GSM8K", task: "gsm8k_cot", primaryMetric: "exact_match", quickLimit: 200, maxGenToks: 2048, blurb: "Grade-school math word problems." },
  { idBase: "bbh", label: "BBH", task: "bbh_cot_zeroshot", primaryMetric: "exact_match", quickLimit: 40, maxGenToks: 4096, blurb: "Big-Bench-Hard reasoning suite, chain-of-thought." },
  { idBase: "math-hard", label: "MATH-hard", task: "leaderboard_math_hard", primaryMetric: "exact_match", quickLimit: 100, maxGenToks: 4096, blurb: "Competition-level MATH (level-5)." },
];

function accuracyPresets(): BenchmarkPreset[] {
  const out: BenchmarkPreset[] = [];
  for (const b of ACCURACY_BENCHES) {
    const base: AccuracyConfig = {
      tasks: [b.task],
      primaryTask: b.task,
      primaryMetric: b.primaryMetric,
      limit: null,
      numFewshot: null,
      maxGenToks: b.maxGenToks,
      applyChatTemplate: true,
      reasoning: true,
      seed: 42,
    };
    out.push({
      id: `acc-${b.idBase}-quick`,
      label: `${b.label} — quick (${b.quickLimit})`,
      description: `${b.blurb} Sampled to ${b.quickLimit} items for a fast quality probe.`,
      kind: "accuracy",
      config: { ...base, limit: b.quickLimit },
    });
    out.push({
      id: `acc-${b.idBase}-full`,
      label: `${b.label} — full`,
      description: `${b.blurb} Complete dataset — can run for hours on a slow endpoint.`,
      kind: "accuracy",
      config: { ...base },
    });
  }
  return out;
}
```

Append the accuracy presets to the exported array by adding `...accuracyPresets(),` as the last element of the `BENCHMARK_PRESETS` array literal (after the `tool-eval-pressure` preset object).

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `npx vitest run packages/server/src/benchmarks/presets.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add packages/server/src/benchmarks/presets.ts packages/server/src/benchmarks/presets.test.ts
git commit -m "feat(benchmarks): add AccuracyConfig + 12 lm-eval accuracy presets"
```

---

## Task 3: lm-eval-args.ts — buildLmEvalArgs

**Files:**
- Create: `packages/server/src/benchmarks/lm-eval-args.ts`
- Test: `packages/server/src/benchmarks/lm-eval-args.test.ts`

**Interfaces:**
- Consumes: `AccuracyConfig` from `./presets.js`.
- Produces:
```ts
export type LmEvalTarget = { baseUrl: string; modelName: string; outputDir: string };
export function buildLmEvalArgs(config: AccuracyConfig, target: LmEvalTarget): string[];
```
`baseUrl` is a `.../v1` base; the builder appends `/chat/completions`. Consumed by Task 8.

- [ ] **Step 1: Write the failing test.** Create `lm-eval-args.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { buildLmEvalArgs } from "./lm-eval-args.js";
import type { AccuracyConfig } from "./presets.js";

const base: AccuracyConfig = {
  tasks: ["ifeval"],
  primaryTask: "ifeval",
  primaryMetric: "prompt_level_strict_acc",
  limit: 100,
  numFewshot: null,
  maxGenToks: 2048,
  applyChatTemplate: true,
  reasoning: true,
  seed: 42,
};
const target = { baseUrl: "http://10.0.0.1:8000/v1", modelName: "m", outputDir: "/out" };

function valueAfter(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i < 0 ? undefined : args[i + 1];
}

describe("buildLmEvalArgs", () => {
  it("targets local-chat-completions with the /chat/completions base_url and headless args", () => {
    const args = buildLmEvalArgs(base, target);
    expect(valueAfter(args, "--model")).toBe("local-chat-completions");
    expect(valueAfter(args, "--model_args")).toBe(
      "base_url=http://10.0.0.1:8000/v1/chat/completions,model=m,num_concurrent=1,tokenized_requests=False",
    );
    expect(valueAfter(args, "--tasks")).toBe("ifeval");
    expect(valueAfter(args, "--gen_kwargs")).toBe("max_gen_toks=2048");
    expect(valueAfter(args, "--seed")).toBe("42");
    expect(valueAfter(args, "--output_path")).toBe("/out");
  });

  it("joins multiple tasks with commas", () => {
    const args = buildLmEvalArgs({ ...base, tasks: ["a", "b"] }, target);
    expect(valueAfter(args, "--tasks")).toBe("a,b");
  });

  it("includes --apply_chat_template only when applyChatTemplate is set", () => {
    expect(buildLmEvalArgs(base, target)).toContain("--apply_chat_template");
    expect(buildLmEvalArgs({ ...base, applyChatTemplate: false }, target)).not.toContain("--apply_chat_template");
  });

  it("includes --limit only when limit is non-null", () => {
    expect(valueAfter(buildLmEvalArgs(base, target), "--limit")).toBe("100");
    expect(buildLmEvalArgs({ ...base, limit: null }, target)).not.toContain("--limit");
  });

  it("includes --num_fewshot only when numFewshot is non-null (0 is valid)", () => {
    expect(buildLmEvalArgs({ ...base, numFewshot: null }, target)).not.toContain("--num_fewshot");
    expect(valueAfter(buildLmEvalArgs({ ...base, numFewshot: 0 }, target), "--num_fewshot")).toBe("0");
    expect(valueAfter(buildLmEvalArgs({ ...base, numFewshot: 5 }, target), "--num_fewshot")).toBe("5");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `npx vitest run packages/server/src/benchmarks/lm-eval-args.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `lm-eval-args.ts`.**

```ts
import type { AccuracyConfig } from "./presets.js";

export type LmEvalTarget = {
  baseUrl: string;   // OpenAI base including /v1 (deployment or strip proxy)
  modelName: string; // vLLM served model id
  outputDir: string; // lm-eval writes results_*.json under here
};

// lm-eval's local-chat-completions model wants base_url to be the FULL
// /v1/chat/completions path. tokenized_requests=False keeps it from loading a
// local HF tokenizer for the served model; num_concurrent=1 suits one slow
// endpoint. --gen_kwargs / --limit / --num_fewshot are single-token key=val or
// scalar flags (not nargs), unlike llama-benchy's list flags.
export function buildLmEvalArgs(config: AccuracyConfig, target: LmEvalTarget): string[] {
  const modelArgs = [
    `base_url=${target.baseUrl}/chat/completions`,
    `model=${target.modelName}`,
    "num_concurrent=1",
    "tokenized_requests=False",
  ].join(",");

  const args: string[] = [
    "--model", "local-chat-completions",
    "--model_args", modelArgs,
    "--tasks", config.tasks.join(","),
    "--gen_kwargs", `max_gen_toks=${config.maxGenToks}`,
    "--seed", String(config.seed),
    "--output_path", target.outputDir,
  ];
  if (config.applyChatTemplate) args.push("--apply_chat_template");
  if (config.limit !== null) args.push("--limit", String(config.limit));
  if (config.numFewshot !== null) args.push("--num_fewshot", String(config.numFewshot));
  return args;
}
```

- [ ] **Step 4: Run the test to verify it passes.**

Run: `npx vitest run packages/server/src/benchmarks/lm-eval-args.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add packages/server/src/benchmarks/lm-eval-args.ts packages/server/src/benchmarks/lm-eval-args.test.ts
git commit -m "feat(benchmarks): buildLmEvalArgs for lm-eval API-mode runs"
```

---

## Task 4: reasoning.ts — stripReasoning (pure)

**Files:**
- Create: `packages/server/src/benchmarks/reasoning.ts`
- Test: `packages/server/src/benchmarks/reasoning.test.ts`

**Interfaces:**
- Produces: `export function stripReasoning(content: string): string;` — consumed by Task 7.

- [ ] **Step 1: Write the failing tests.** Create `reasoning.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { stripReasoning } from "./reasoning.js";

describe("stripReasoning", () => {
  it("removes a complete <think>…</think> block and keeps the trailing answer", () => {
    expect(stripReasoning("<think>lots of thinking</think>The answer is B")).toBe("The answer is B");
  });

  it("handles a template-injected open tag (only the closing tag is echoed)", () => {
    expect(stripReasoning("reasoning text</think>\n\nThe answer is 42")).toBe("The answer is 42");
  });

  it("yields empty output when the model was cut off mid-think (no close tag)", () => {
    expect(stripReasoning("<think>never finished reasoning")).toBe("");
  });

  it("is a no-op (modulo trim) when there are no reasoning tags", () => {
    expect(stripReasoning("The answer is 7")).toBe("The answer is 7");
  });

  it("is case-insensitive on the tags", () => {
    expect(stripReasoning("<THINK>x</THINK>done")).toBe("done");
  });

  // Invariant: output never contains a think tag, for any input.
  test.prop([fc.string()])("output never contains a think tag", (s) => {
    const out = stripReasoning(s).toLowerCase();
    expect(out.includes("<think>")).toBe(false);
    expect(out.includes("</think>")).toBe(false);
  });

  // Invariant: idempotent.
  test.prop([fc.string()])("is idempotent", (s) => {
    expect(stripReasoning(stripReasoning(s))).toBe(stripReasoning(s));
  });

  // Invariant: a think block prepended to a tag-free answer is fully removed.
  test.prop([
    fc.stringMatching(/^[a-zA-Z0-9 .,!?]+$/),
    fc.stringMatching(/^[a-zA-Z0-9 .,!?]+$/),
  ])("strips a wrapped think block down to the answer", (noise, answer) => {
    expect(stripReasoning(`<think>${noise}</think>${answer}`)).toBe(answer.trim());
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `npx vitest run packages/server/src/benchmarks/reasoning.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `reasoning.ts`.**

```ts
// Strip a reasoning model's thinking from a chat-completion answer so lm-eval's
// answer-extraction filters see only the final answer. Handles: complete
// <think>…</think> blocks; a template-injected open tag where only </think> is
// echoed (keep text after the last close); and a truncated open tag with no
// close (no answer was produced → empty). No-op when there are no tags, so it is
// safe for non-reasoning models.
export function stripReasoning(content: string): string {
  if (!content) return content;
  let out = content.replace(/<think>[\s\S]*?<\/think>/gi, "");

  const lower = out.toLowerCase();
  const lastClose = lower.lastIndexOf("</think>");
  if (lastClose !== -1) {
    out = out.slice(lastClose + "</think>".length);
  }

  const openIdx = out.toLowerCase().indexOf("<think>");
  if (openIdx !== -1) {
    out = out.slice(0, openIdx);
  }
  return out.trim();
}
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `npx vitest run packages/server/src/benchmarks/reasoning.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add packages/server/src/benchmarks/reasoning.ts packages/server/src/benchmarks/reasoning.test.ts
git commit -m "feat(benchmarks): stripReasoning helper for reasoning-model answers"
```

---

## Task 5: lm-eval-parser.ts — parseLmEvalResults (+ fixture)

**Files:**
- Create: `packages/server/src/benchmarks/lm-eval-parser.ts`
- Create: `packages/server/src/__tests__/integration/benchmarks.fixtures/lm-eval-result.json`
- Test: `packages/server/src/benchmarks/lm-eval-parser.test.ts`

**Interfaces:**
- Produces:
```ts
export type AccuracyMetricInput = {
  task: string; metric: string; value: number;
  stderr: number | null; isGroup: boolean; nSamples: number | null;
};
export type LmEvalSummary = { primaryScore: number; metrics: AccuracyMetricInput[] };
export function parseLmEvalResults(jsonText: string, primaryTask: string, primaryMetric: string): LmEvalSummary;
```
Consumed by Tasks 8, 9.

- [ ] **Step 1: Create the fixture** `benchmarks.fixtures/lm-eval-result.json`:

```json
{
  "results": {
    "ifeval": {
      "alias": "ifeval",
      "prompt_level_strict_acc,none": 0.42,
      "prompt_level_strict_acc_stderr,none": 0.021,
      "inst_level_strict_acc,none": 0.55,
      "inst_level_strict_acc_stderr,none": "N/A",
      "prompt_level_loose_acc,none": 0.45,
      "prompt_level_loose_acc_stderr,none": 0.021
    },
    "bbh": {
      "alias": "bbh",
      "exact_match,none": 0.6,
      "exact_match_stderr,none": 0.015
    },
    "bbh_boolean_expressions": {
      "alias": " - boolean_expressions",
      "exact_match,none": 0.9,
      "exact_match_stderr,none": 0.03
    }
  },
  "groups": {
    "bbh": {
      "alias": "bbh",
      "exact_match,none": 0.6,
      "exact_match_stderr,none": 0.015
    }
  },
  "n-samples": {
    "ifeval": { "original": 541, "effective": 100 },
    "bbh": { "original": 6511, "effective": 40 },
    "bbh_boolean_expressions": { "original": 250, "effective": 40 }
  }
}
```

- [ ] **Step 2: Write the failing tests.** Create `lm-eval-parser.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseLmEvalResults } from "./lm-eval-parser.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixture = readFileSync(
  join(__dirname, "../__tests__/integration/benchmarks.fixtures/lm-eval-result.json"),
  "utf-8",
);

describe("parseLmEvalResults", () => {
  it("returns the primary metric ×100 as primaryScore", () => {
    const { primaryScore } = parseLmEvalResults(fixture, "ifeval", "prompt_level_strict_acc");
    expect(primaryScore).toBeCloseTo(42, 5);
  });

  it("emits a breakdown row per numeric metric with paired stderr and n-samples", () => {
    const { metrics } = parseLmEvalResults(fixture, "ifeval", "prompt_level_strict_acc");
    const row = metrics.find((m) => m.task === "ifeval" && m.metric === "prompt_level_strict_acc")!;
    expect(row).toMatchObject({ value: 0.42, stderr: 0.021, isGroup: false, nSamples: 100 });
  });

  it("treats a non-numeric stderr ('N/A') as null", () => {
    const { metrics } = parseLmEvalResults(fixture, "ifeval", "prompt_level_strict_acc");
    const row = metrics.find((m) => m.metric === "inst_level_strict_acc")!;
    expect(row.stderr).toBeNull();
  });

  it("flags group-level rows via the groups object", () => {
    const { metrics } = parseLmEvalResults(fixture, "bbh", "exact_match");
    expect(metrics.find((m) => m.task === "bbh")!.isGroup).toBe(true);
    expect(metrics.find((m) => m.task === "bbh_boolean_expressions")!.isGroup).toBe(false);
  });

  it("never emits a stderr key as its own metric row", () => {
    const { metrics } = parseLmEvalResults(fixture, "ifeval", "prompt_level_strict_acc");
    expect(metrics.some((m) => m.metric.endsWith("_stderr"))).toBe(false);
  });

  it("throws when the primary metric is absent", () => {
    expect(() => parseLmEvalResults(fixture, "ifeval", "nonexistent")).toThrow(/missing/i);
  });

  it("throws on malformed JSON", () => {
    expect(() => parseLmEvalResults("not json", "ifeval", "prompt_level_strict_acc")).toThrow(/parse/i);
  });

  it("throws when the results object is missing", () => {
    expect(() => parseLmEvalResults("{}", "ifeval", "prompt_level_strict_acc")).toThrow(/results/i);
  });
});
```

- [ ] **Step 3: Run the tests to verify they fail.**

Run: `npx vitest run packages/server/src/benchmarks/lm-eval-parser.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement `lm-eval-parser.ts`.**

```ts
export type AccuracyMetricInput = {
  task: string;
  metric: string;      // metric name without the ",<filter>" suffix
  value: number;       // raw lm-eval value (0–1 for accuracies)
  stderr: number | null;
  isGroup: boolean;
  nSamples: number | null;
};

export type LmEvalSummary = {
  primaryScore: number; // primary metric ×100 (0–100)
  metrics: AccuracyMetricInput[];
};

type Obj = Record<string, unknown>;

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// Parse an lm-evaluation-harness results JSON. Metric keys look like
// "<metric>,<filter>" (e.g. "exact_match,none") with a sibling
// "<metric>_stderr,<filter>". We split on the LAST comma to recover the filter,
// skip *_stderr keys as standalone metrics (they attach to their base metric),
// and skip the "alias" string. Group-level tasks are flagged via the top-level
// `groups` object. Fail-fast: the primary metric must be present and numeric.
export function parseLmEvalResults(
  jsonText: string,
  primaryTask: string,
  primaryMetric: string,
): LmEvalSummary {
  let parsed: Obj;
  try {
    parsed = JSON.parse(jsonText) as Obj;
  } catch (e) {
    throw new Error(`failed to parse lm-eval JSON: ${(e as Error).message}`);
  }

  const results = parsed.results;
  if (!results || typeof results !== "object") {
    throw new Error("lm-eval result missing required object: results");
  }
  const groups = (parsed.groups && typeof parsed.groups === "object" ? parsed.groups : {}) as Obj;
  const nSamplesAll = (parsed["n-samples"] && typeof parsed["n-samples"] === "object"
    ? parsed["n-samples"]
    : {}) as Obj;

  const metrics: AccuracyMetricInput[] = [];
  for (const [task, taskVal] of Object.entries(results as Obj)) {
    if (!taskVal || typeof taskVal !== "object") continue;
    const entry = taskVal as Obj;
    const isGroup = Object.prototype.hasOwnProperty.call(groups, task);
    const nInfo = nSamplesAll[task] as Obj | undefined;
    const nSamples = nInfo ? numOrNull(nInfo.effective) : null;

    for (const [key, value] of Object.entries(entry)) {
      if (key === "alias") continue;
      const comma = key.lastIndexOf(",");
      const metricName = comma === -1 ? key : key.slice(0, comma);
      if (metricName.endsWith("_stderr")) continue;
      const numeric = numOrNull(value);
      if (numeric === null) continue;

      const filter = comma === -1 ? "" : key.slice(comma);
      const stderrKey = `${metricName}_stderr${filter}`;
      metrics.push({
        task,
        metric: metricName,
        value: numeric,
        stderr: numOrNull(entry[stderrKey]),
        isGroup,
        nSamples,
      });
    }
  }

  const primaryEntry = (results as Obj)[primaryTask] as Obj | undefined;
  const primaryRaw = primaryEntry ? primaryEntry[`${primaryMetric},none`] : undefined;
  const primary = numOrNull(primaryRaw);
  if (primary === null) {
    throw new Error(
      `lm-eval result missing primary metric: ${primaryTask}/${primaryMetric}`,
    );
  }

  return { primaryScore: primary * 100, metrics };
}
```

- [ ] **Step 5: Run the tests to verify they pass.**

Run: `npx vitest run packages/server/src/benchmarks/lm-eval-parser.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add packages/server/src/benchmarks/lm-eval-parser.ts packages/server/src/benchmarks/lm-eval-parser.test.ts packages/server/src/__tests__/integration/benchmarks.fixtures/lm-eval-result.json
git commit -m "feat(benchmarks): parseLmEvalResults with per-task breakdown + fail-fast primary"
```

---

## Task 6: lm-eval-result-file.ts — findLmEvalResultFile

**Files:**
- Create: `packages/server/src/benchmarks/lm-eval-result-file.ts`
- Test: `packages/server/src/benchmarks/lm-eval-result-file.test.ts`

**Interfaces:**
- Produces: `export function findLmEvalResultFile(outputDir: string): string | null;` — returns the newest `results_*.json` under `outputDir` (recursively), or null. Consumed by Task 8.

- [ ] **Step 1: Write the failing tests.** Create `lm-eval-result-file.test.ts`:

```ts
import { describe, expect, it, afterAll } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { findLmEvalResultFile } from "./lm-eval-result-file.js";

const dirs: string[] = [];
function tmp(): string {
  const d = mkdtempSync(join(tmpdir(), "lm-eval-rf-"));
  dirs.push(d);
  return d;
}
afterAll(() => { for (const d of dirs) rmSync(d, { recursive: true, force: true }); });

describe("findLmEvalResultFile", () => {
  it("returns null when there is no results file", () => {
    expect(findLmEvalResultFile(tmp())).toBeNull();
  });

  it("finds a results file nested in a model subdirectory", () => {
    const root = tmp();
    const sub = join(root, "some__model");
    mkdirSync(sub);
    const f = join(sub, "results_2026-07-09T10-00-00.json");
    writeFileSync(f, "{}");
    expect(findLmEvalResultFile(root)).toBe(f);
  });

  it("returns the newest results file by mtime when several exist", () => {
    const root = tmp();
    const older = join(root, "results_2026-07-09T09-00-00.json");
    const newer = join(root, "results_2026-07-09T11-00-00.json");
    writeFileSync(older, "{}");
    writeFileSync(newer, "{}");
    utimesSync(older, new Date(1_000_000), new Date(1_000_000));
    utimesSync(newer, new Date(2_000_000), new Date(2_000_000));
    expect(findLmEvalResultFile(root)).toBe(newer);
  });

  it("ignores non-results json files", () => {
    const root = tmp();
    writeFileSync(join(root, "samples_ifeval.json"), "{}");
    expect(findLmEvalResultFile(root)).toBeNull();
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `npx vitest run packages/server/src/benchmarks/lm-eval-result-file.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `lm-eval-result-file.ts`.**

```ts
import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const RESULTS_RE = /^results_.*\.json$/;

// lm-eval writes results to <output_path>/<sanitized-model>/results_<ts>.json
// (nested), not a fixed path. Recursively find every results_*.json under
// outputDir and return the newest by mtime, or null if none exists.
export function findLmEvalResultFile(outputDir: string): string | null {
  let best: string | null = null;
  let bestMtime = -Infinity;

  const walk = (dir: string) => {
    let entries: import("node:fs").Dirent[];
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return; // dir vanished / unreadable — nothing to contribute
    }
    for (const e of entries) {
      const full = join(dir, e.name);
      if (e.isDirectory()) {
        walk(full);
      } else if (RESULTS_RE.test(e.name)) {
        const m = statSync(full).mtimeMs;
        if (m > bestMtime) {
          bestMtime = m;
          best = full;
        }
      }
    }
  };

  walk(outputDir);
  return best;
}
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `npx vitest run packages/server/src/benchmarks/lm-eval-result-file.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add packages/server/src/benchmarks/lm-eval-result-file.ts packages/server/src/benchmarks/lm-eval-result-file.test.ts
git commit -m "feat(benchmarks): locate newest lm-eval results_*.json under output dir"
```

---

## Task 7: reasoning-proxy.ts — startReasoningProxy

**Files:**
- Create: `packages/server/src/benchmarks/reasoning-proxy.ts`
- Test: `packages/server/src/benchmarks/reasoning-proxy.test.ts`

**Interfaces:**
- Consumes: `stripReasoning` from `./reasoning.js`.
- Produces:
```ts
export type ReasoningProxy = { url: string; close: () => Promise<void> };
export function startReasoningProxy(targetV1Url: string): Promise<ReasoningProxy>;
```
`url` is a `.../v1` base to hand to lm-eval. Consumed by Task 8.

- [ ] **Step 1: Write the failing tests.** Create `reasoning-proxy.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import http from "node:http";
import { startReasoningProxy } from "./reasoning-proxy.js";

// Spin a fake upstream that echoes a canned response for a given path.
function fakeUpstream(handler: (req: http.IncomingMessage, res: http.ServerResponse) => void) {
  return new Promise<{ v1Url: string; close: () => Promise<void> }>((resolve) => {
    const server = http.createServer(handler);
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address() as { port: number };
      resolve({
        v1Url: `http://127.0.0.1:${addr.port}/v1`,
        close: () => new Promise((r) => server.close(() => r())),
      });
    });
  });
}

describe("startReasoningProxy", () => {
  it("strips <think>…</think> from chat-completion responses", async () => {
    const upstream = await fakeUpstream((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({
        choices: [{ message: { role: "assistant", content: "<think>secret</think>Answer: B" } }],
      }));
    });
    const proxy = await startReasoningProxy(upstream.v1Url);

    const resp = await fetch(`${proxy.url}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [] }),
    });
    const body = await resp.json();
    expect(body.choices[0].message.content).toBe("Answer: B");

    await proxy.close();
    await upstream.close();
  });

  it("passes non-chat responses (e.g. /v1/models) through unchanged", async () => {
    const upstream = await fakeUpstream((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ data: [{ id: "served-model" }] }));
    });
    const proxy = await startReasoningProxy(upstream.v1Url);

    const resp = await fetch(`${proxy.url}/models`);
    const body = await resp.json();
    expect(body.data[0].id).toBe("served-model");

    await proxy.close();
    await upstream.close();
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `npx vitest run packages/server/src/benchmarks/reasoning-proxy.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `reasoning-proxy.ts`.**

```ts
import http from "node:http";
import { AddressInfo } from "node:net";
import { stripReasoning } from "./reasoning.js";

export type ReasoningProxy = {
  url: string;             // .../v1 base to hand to lm-eval
  close: () => Promise<void>;
};

function readBody(req: http.IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

// Rewrite a chat-completion JSON body, stripping reasoning from each choice's
// message content. Returns the original text unchanged if it isn't the expected
// shape (so /v1/models and errors pass through untouched).
function rewriteChatBody(text: string): string {
  try {
    const parsed = JSON.parse(text) as {
      choices?: Array<{ message?: { content?: unknown } }>;
    };
    if (!Array.isArray(parsed.choices)) return text;
    for (const c of parsed.choices) {
      if (c.message && typeof c.message.content === "string") {
        c.message.content = stripReasoning(c.message.content);
      }
    }
    return JSON.stringify(parsed);
  } catch {
    return text;
  }
}

// Localhost proxy in front of `targetV1Url` (a .../v1 base) that strips
// <think>…</think> from /v1/chat/completions responses before returning them, so
// lm-eval scores the final answer. Non-streaming only (lm-eval uses
// non-streaming completions).
export function startReasoningProxy(targetV1Url: string): Promise<ReasoningProxy> {
  return new Promise((resolve) => {
    const server = http.createServer(async (req, res) => {
      try {
        const suffix = (req.url ?? "").replace(/^\/v1/, "");
        const targetUrl = `${targetV1Url}${suffix}`;
        const body = await readBody(req);

        const headers: Record<string, string> = {};
        for (const [k, v] of Object.entries(req.headers)) {
          if (typeof v === "string" && k.toLowerCase() !== "host" && k.toLowerCase() !== "content-length") {
            headers[k] = v;
          }
        }

        const upstream = await fetch(targetUrl, {
          method: req.method,
          headers,
          body: req.method === "GET" || req.method === "HEAD" ? undefined : body,
        });

        const text = await upstream.text();
        const isChat = suffix.includes("/chat/completions");
        const out = isChat ? rewriteChatBody(text) : text;

        res.statusCode = upstream.status;
        const ct = upstream.headers.get("content-type");
        if (ct) res.setHeader("content-type", ct);
        res.end(out);
      } catch (e) {
        res.statusCode = 502;
        res.end(JSON.stringify({ error: `reasoning-proxy: ${(e as Error).message}` }));
      }
    });

    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        url: `http://127.0.0.1:${port}/v1`,
        close: () => new Promise((r) => server.close(() => r())),
      });
    });
  });
}
```

- [ ] **Step 4: Run the tests to verify they pass.**

Run: `npx vitest run packages/server/src/benchmarks/reasoning-proxy.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add packages/server/src/benchmarks/reasoning-proxy.ts packages/server/src/benchmarks/reasoning-proxy.test.ts
git commit -m "feat(benchmarks): ephemeral reasoning strip proxy for lm-eval"
```

---

## Task 8: orchestrator.ts — generalize spawnTracked + runAccuracy

**Files:**
- Modify: `packages/server/src/benchmarks/orchestrator.ts`
- Test: `packages/server/src/benchmarks/orchestrator.test.ts`

**Interfaces:**
- Consumes: `buildLmEvalArgs`/`LmEvalTarget` (Task 3), `startReasoningProxy`/`ReasoningProxy` (Task 7), `parseLmEvalResults`/`LmEvalSummary` (Task 5), `findLmEvalResultFile` (Task 6), `AccuracyConfig` (Task 2).
- Produces:
```ts
export type RunAccuracyOpts = {
  runId: string; config: AccuracyConfig; endpointV1Url: string;
  servedModel: string; outputDir: string; onLog: (line: string) => void;
};
export type RunAccuracyResult = { exitCode: number | null; summary: LmEvalSummary | null; rawOutput: string | null };
export async function runAccuracy(opts: RunAccuracyOpts): Promise<RunAccuracyResult>;
```
Consumed by Task 9. `cancelBenchmark` continues to work for accuracy runs (same `ACTIVE` map).

- [ ] **Step 1: Write the failing tests.** Append to `orchestrator.test.ts`. First, add module mocks for the proxy and result-file locator near the top-level mocks (after the existing `vi.mock("node:fs", …)` block):

```ts
const startProxyMock = vi.fn();
vi.mock("./reasoning-proxy.js", () => ({
  startReasoningProxy: (...a: unknown[]) => startProxyMock(...a),
}));
const findFileMock = vi.fn();
vi.mock("./lm-eval-result-file.js", () => ({
  findLmEvalResultFile: (...a: unknown[]) => findFileMock(...a),
}));
```

Update the import line to include `runAccuracy`:

```ts
import { runBenchmark, runToolEval, runAccuracy, cancelBenchmark } from "./orchestrator.js";
```

Append this describe block at the end of the file:

```ts
describe("runAccuracy", () => {
  const baseConfig = {
    tasks: ["ifeval"], primaryTask: "ifeval", primaryMetric: "prompt_level_strict_acc",
    limit: 100, numFewshot: null, maxGenToks: 2048,
    applyChatTemplate: true, reasoning: true, seed: 42,
  };

  beforeEach(() => {
    spawnMock.mockReset();
    readFileSyncMock.mockReset();
    mkdirSyncMock.mockReset();
    existsSyncMock.mockReset();
    startProxyMock.mockReset();
    findFileMock.mockReset();
  });

  it("starts the strip proxy, runs uvx lm_eval against it, parses, and closes the proxy", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    const closeMock = vi.fn().mockResolvedValue(undefined);
    startProxyMock.mockResolvedValue({ url: "http://127.0.0.1:5555/v1", close: closeMock });
    findFileMock.mockReturnValue("/o/model/results_x.json");
    existsSyncMock.mockReturnValue(true);
    readFileSyncMock.mockReturnValue(JSON.stringify({
      results: { ifeval: { "prompt_level_strict_acc,none": 0.5 } },
    }));

    const promise = runAccuracy({
      runId: "run_acc", config: baseConfig, endpointV1Url: "http://10.0.0.1:8000/v1",
      servedModel: "m", outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 0);
    const r = await promise;

    expect(startProxyMock).toHaveBeenCalledWith("http://10.0.0.1:8000/v1");
    const [cmd, argv] = spawnMock.mock.calls[0];
    expect(cmd).toBe("uvx");
    expect(argv[0]).toBe("--from");
    expect(argv[1]).toMatch(/^lm-eval\[.+\]/);
    expect(argv[2]).toBe("lm_eval");
    expect(argv).toContain("base_url=http://127.0.0.1:5555/v1/chat/completions,model=m,num_concurrent=1,tokenized_requests=False");
    expect(r.exitCode).toBe(0);
    expect(r.summary?.primaryScore).toBeCloseTo(50, 5);
    expect(closeMock).toHaveBeenCalledTimes(1);
  });

  it("skips the proxy and targets the endpoint directly when reasoning is false", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    findFileMock.mockReturnValue(null);
    existsSyncMock.mockReturnValue(false);

    const promise = runAccuracy({
      runId: "run_acc2", config: { ...baseConfig, reasoning: false },
      endpointV1Url: "http://10.0.0.1:8000/v1", servedModel: "m", outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 0);
    const r = await promise;

    expect(startProxyMock).not.toHaveBeenCalled();
    const [, argv] = spawnMock.mock.calls[0];
    expect(argv).toContain("base_url=http://10.0.0.1:8000/v1/chat/completions,model=m,num_concurrent=1,tokenized_requests=False");
    expect(r.summary).toBeNull(); // no result file → no summary
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `npx vitest run packages/server/src/benchmarks/orchestrator.test.ts`
Expected: FAIL — `runAccuracy` is not exported.

- [ ] **Step 3: Generalize `spawnTracked` and add `runAccuracy`.** In `orchestrator.ts`:

Add imports at the top (after the existing tool-eval-parser import):

```ts
import { buildLmEvalArgs } from "./lm-eval-args.js";
import { parseLmEvalResults, type LmEvalSummary } from "./lm-eval-parser.js";
import { findLmEvalResultFile } from "./lm-eval-result-file.js";
import { startReasoningProxy, type ReasoningProxy } from "./reasoning-proxy.js";
import type { AccuracyConfig } from "./presets.js";
```

Add the lm-eval spec constant next to the existing ones:

```ts
// lm-evaluation-harness with the IFEval + MATH scoring extras. Overridable pin.
const LM_EVAL_SPEC =
  process.env.LM_EVAL_VERSION
    ? `lm-eval[ifeval,math]==${process.env.LM_EVAL_VERSION}`
    : "lm-eval[ifeval,math]";
```

Extend `SpawnTrackedOpts` with an optional result-file locator:

```ts
type SpawnTrackedOpts = {
  runId: string;
  command: string;
  args: string[];
  outputDir: string;
  onLog: (line: string) => void;
  // Resolve the result file to read on exit. Defaults to outputDir/result.json
  // (llama-benchy / tool-eval-bench). lm-eval passes a locator for its nested
  // results_*.json.
  resultFile?: (outputDir: string) => string | null;
};
```

Replace the `child.on("close", …)` handler body's result-path resolution so it uses the locator:

```ts
    child.on("close", (code) => {
      ACTIVE.delete(opts.runId);
      const resultPath = opts.resultFile
        ? opts.resultFile(opts.outputDir)
        : join(opts.outputDir, "result.json");
      let rawOutput: string | null = null;
      if (code === 0 && resultPath && existsSync(resultPath)) {
        try {
          rawOutput = readFileSync(resultPath, "utf-8");
        } catch (e) {
          opts.onLog(`[read] ${(e as Error).message}`);
        }
      }
      resolve({ exitCode: code, rawOutput });
    });
```

Add `runAccuracy` after `runToolEval` (before `cancelBenchmark`):

```ts
export type RunAccuracyOpts = {
  runId: string;
  config: AccuracyConfig;
  endpointV1Url: string; // deployment .../v1
  servedModel: string;
  outputDir: string;
  onLog: (line: string) => void;
};

export type RunAccuracyResult = {
  exitCode: number | null;
  summary: LmEvalSummary | null;
  rawOutput: string | null;
};

export async function runAccuracy(opts: RunAccuracyOpts): Promise<RunAccuracyResult> {
  let proxy: ReasoningProxy | null = null;
  try {
    const baseUrl = opts.config.reasoning
      ? (proxy = await startReasoningProxy(opts.endpointV1Url)).url
      : opts.endpointV1Url;

    const args = buildLmEvalArgs(opts.config, {
      baseUrl,
      modelName: opts.servedModel,
      outputDir: opts.outputDir,
    });

    const { exitCode, rawOutput } = await spawnTracked({
      runId: opts.runId,
      command: "uvx",
      args: ["--from", LM_EVAL_SPEC, "lm_eval", ...args],
      outputDir: opts.outputDir,
      onLog: opts.onLog,
      resultFile: findLmEvalResultFile,
    });

    let summary: LmEvalSummary | null = null;
    if (exitCode === 0 && rawOutput !== null) {
      try {
        summary = parseLmEvalResults(rawOutput, opts.config.primaryTask, opts.config.primaryMetric);
      } catch (e) {
        opts.onLog(`[parser] ${(e as Error).message}`);
      }
    }
    return { exitCode, summary, rawOutput };
  } finally {
    if (proxy) await proxy.close();
  }
}
```

- [ ] **Step 4: Run the orchestrator tests to verify they pass.**

Run: `npx vitest run packages/server/src/benchmarks/orchestrator.test.ts`
Expected: PASS (new `runAccuracy` tests + existing `runBenchmark`/`runToolEval` tests unaffected by the locator change).

- [ ] **Step 5: Commit.**

```bash
git add packages/server/src/benchmarks/orchestrator.ts packages/server/src/benchmarks/orchestrator.test.ts
git commit -m "feat(benchmarks): runAccuracy orchestrator + pluggable result-file locator"
```

---

## Task 9: routes/benchmarks.ts — accuracy POST branch + persistence

**Files:**
- Modify: `packages/server/src/routes/benchmarks.ts`
- Modify: `packages/server/src/openapi.ts`
- Test: `packages/server/src/__tests__/integration/benchmarks.routes.test.ts`

**Interfaces:**
- Consumes: `runAccuracy` (Task 8), `AccuracyConfig` (Task 2), `accuracyScore`/`accuracyMetrics` columns (Task 1).

- [ ] **Step 1: Write the failing tests.** In `benchmarks.routes.test.ts`:

Add `runAccuracyMock` and wire it into the orchestrator mock. Change the mock block to:

```ts
const runMock = vi.fn();
const runToolEvalMock = vi.fn();
const runAccuracyMock = vi.fn();
const cancelMock = vi.fn();
vi.mock("../../benchmarks/orchestrator.js", () => ({
  runBenchmark: (...a: unknown[]) => runMock(...a),
  runToolEval: (...a: unknown[]) => runToolEvalMock(...a),
  runAccuracy: (...a: unknown[]) => runAccuracyMock(...a),
  cancelBenchmark: (...a: unknown[]) => cancelMock(...a),
}));
```

Add `runAccuracyMock.mockReset();` to the `beforeEach` resets.

Append a new describe block:

```ts
describe("POST /api/benchmarks (accuracy dispatch)", () => {
  it("creates a run with kind 'accuracy' for an accuracy preset", async () => {
    const d = await seedRunningDeployment();
    runAccuracyMock.mockReturnValue(new Promise(() => {}));
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "acc-ifeval-quick" });
    expect(res.status).toBe(201);
    expect(res.body.kind).toBe("accuracy");
    expect(res.body.presetId).toBe("acc-ifeval-quick");
    expect(runAccuracyMock).toHaveBeenCalledTimes(1);
    const call = runAccuracyMock.mock.calls[0][0];
    expect(call.endpointV1Url).toBe("http://10.0.0.1:8000/v1");
    expect(call.config.primaryTask).toBe("ifeval");
    expect(runMock).not.toHaveBeenCalled();
    expect(runToolEvalMock).not.toHaveBeenCalled();
  });

  it("persists accuracyScore and accuracyMetrics on completion", async () => {
    const d = await seedRunningDeployment();
    runAccuracyMock.mockResolvedValue({
      exitCode: 0,
      rawOutput: "{}",
      summary: {
        primaryScore: 42,
        metrics: [{ task: "ifeval", metric: "prompt_level_strict_acc", value: 0.42, stderr: 0.02, isGroup: false, nSamples: 100 }],
      },
    });

    const app = makeApp();
    const res = await request(app)
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "acc-ifeval-quick" });
    const runId = res.body.id;
    await new Promise((r) => setTimeout(r, 20));

    const detail = await request(app).get(`/api/benchmarks/${runId}`);
    expect(detail.body.status).toBe("completed");
    expect(detail.body.accuracyScore).toBe(42);
    const metrics = JSON.parse(detail.body.accuracyMetrics);
    expect(metrics[0].task).toBe("ifeval");
  });

  it("marks the run failed when lm-eval exits non-zero", async () => {
    const d = await seedRunningDeployment();
    runAccuracyMock.mockResolvedValue({ exitCode: 1, rawOutput: null, summary: null });

    const app = makeApp();
    const res = await request(app)
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "acc-ifeval-quick" });
    const runId = res.body.id;
    await new Promise((r) => setTimeout(r, 20));

    const detail = await request(app).get(`/api/benchmarks/${runId}`);
    expect(detail.body.status).toBe("failed");
    expect(detail.body.error).toMatch(/lm-eval exited with code 1/);
    expect(detail.body.accuracyScore).toBeNull();
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail.**

Run: `npx vitest run packages/server/src/__tests__/integration/benchmarks.routes.test.ts`
Expected: FAIL — route has no accuracy branch (kind resolves but `runAccuracy` never called; persistence absent).

- [ ] **Step 3: Implement the route branch.** In `routes/benchmarks.ts`:

Update imports:

```ts
import {
  BENCHMARK_PRESETS,
  getPreset,
  type BenchmarkConfig,
  type ToolEvalConfig,
  type AccuracyConfig,
} from "../benchmarks/presets.js";
```

```ts
import {
  runBenchmark,
  runToolEval,
  runAccuracy,
  cancelBenchmark,
} from "../benchmarks/orchestrator.js";
```

Widen the kind/config locals in the POST handler:

```ts
  let kind: "throughput" | "tool-eval" | "accuracy" = "throughput";
  let config: BenchmarkConfig | ToolEvalConfig | AccuracyConfig;
```

Add the accuracy branch. Change the dispatch tail from `if (kind === "tool-eval") { … } else { … }` to a three-way branch — insert this block before the existing `if (kind === "tool-eval")`:

```ts
  if (kind === "accuracy") {
    runAccuracy({
      runId: run.id,
      config: config as AccuracyConfig,
      endpointV1Url: endpointUrl,
      servedModel: servedModelName,
      outputDir,
      onLog,
    })
      .then(async (r) => {
        const current = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
        if (current?.status === "canceled") return;
        if (r.exitCode === 0 && r.summary) {
          await prisma.benchmarkRun.update({
            where: { id: run.id },
            data: {
              status: "completed",
              completedAt: new Date(),
              rawOutput: r.rawOutput,
              accuracyScore: r.summary.primaryScore,
              accuracyMetrics: JSON.stringify(r.summary.metrics),
            },
          });
          const final = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
          sseBroadcast({ type: "benchmark:status", payload: final });
        } else {
          await finishFailed(`lm-eval exited with code ${r.exitCode}`);
        }
      })
      .catch((e) => finishFailed((e as Error).message));
  } else if (kind === "tool-eval") {
```

(Leave the existing tool-eval `.then/.catch` body unchanged; it now follows `} else if (kind === "tool-eval") {`, and the throughput `else` stays last.)

- [ ] **Step 4: Update the OpenAPI tag description.** In `openapi.ts`, change the Benchmarks description line:

```ts
- **Benchmarks** — server-side benchmark runs (llama-benchy / tool-eval-bench / lm-eval) against a deployment.
```

- [ ] **Step 5: Run the route tests to verify they pass.**

Run: `npx vitest run packages/server/src/__tests__/integration/benchmarks.routes.test.ts`
Expected: PASS.

- [ ] **Step 6: Run the full server suite to catch cross-file regressions (presets/openapi tests).**

Run: `npm test`
Expected: PASS (all suites).

- [ ] **Step 7: Commit.**

```bash
git add packages/server/src/routes/benchmarks.ts packages/server/src/openapi.ts packages/server/src/__tests__/integration/benchmarks.routes.test.ts
git commit -m "feat(benchmarks): POST accuracy branch — dispatch lm-eval + persist score/breakdown"
```

---

## Task 10: Dashboard — types + accuracy section in the run form

**Files:**
- Modify: `packages/dashboard/lib/benchmarks.ts`
- Modify: `packages/dashboard/components/benchmark-form-modal.tsx`

**Interfaces:**
- Consumes: `/api/benchmarks/presets` (already returns accuracy presets after Task 2).
- Produces: `AccuracyConfig`, `AccuracyMetric` types; `accuracyScore`/`accuracyMetrics` on `BenchmarkRun`; a selectable "Accuracy (lm-eval)" preset group in the form. Consumed by Tasks 11, 12.

- [ ] **Step 1: Add the dashboard types.** In `lib/benchmarks.ts`:

Widen the kind union:

```ts
export type BenchmarkKind = "throughput" | "tool-eval" | "accuracy";
```

Add the config + metric types (after `ToolEvalConfig`):

```ts
export type AccuracyConfig = {
  tasks: string[];
  primaryTask: string;
  primaryMetric: string;
  limit: number | null;
  numFewshot: number | null;
  maxGenToks: number;
  applyChatTemplate: boolean;
  reasoning: boolean;
  seed: number;
};

export type AccuracyMetric = {
  task: string;
  metric: string;
  value: number;
  stderr: number | null;
  isGroup: boolean;
  nSamples: number | null;
};
```

Widen the preset config union:

```ts
export type BenchmarkPreset = {
  id: string;
  label: string;
  description: string;
  kind: BenchmarkKind;
  config: BenchmarkConfig | ToolEvalConfig | AccuracyConfig;
};
```

Add the two fields to `BenchmarkRun` (after `toolEvalCategories?: ToolEvalCategory[];`):

```ts
  accuracyScore: number | null;
  accuracyMetrics: string | null;
```

- [ ] **Step 2: Add the accuracy group to the form.** In `benchmark-form-modal.tsx`, replace the kind-group loop opener:

```tsx
            {(["throughput", "tool-eval", "accuracy"] as const).map((kind) => {
              const group = presets.filter((p) => p.kind === kind);
              if (group.length === 0) return null;
              const heading =
                kind === "throughput" ? "Throughput"
                : kind === "tool-eval" ? "Tool-calling eval"
                : "Accuracy (lm-eval)";
              return (
                <div key={kind} className="space-y-2">
                  <div className="text-xs uppercase tracking-wide text-gray-500">
                    {heading}
                  </div>
```

Remove the now-replaced inline `{kind === "throughput" ? "Throughput" : "Tool-calling eval"}` heading div that followed the old opener (it is superseded by the `heading` variable above).

Add an accuracy config hint next to the existing throughput hint block (immediately after the `p.kind === "throughput" && "pp" in p.config && (…)` block):

```tsx
                          {p.kind === "accuracy" && "tasks" in p.config && (
                            <div className="text-xs text-gray-500 mt-1">
                              tasks={p.config.tasks.join(",")}
                              {" "}limit={p.config.limit ?? "full"}
                              {" "}max_gen_toks={p.config.maxGenToks}
                            </div>
                          )}
```

- [ ] **Step 3: Verify the dashboard type-checks and builds.**

Run: `npm run build --workspace=packages/dashboard`
Expected: build succeeds (no TS errors from the new union members).

- [ ] **Step 4: Commit.**

```bash
git add packages/dashboard/lib/benchmarks.ts packages/dashboard/components/benchmark-form-modal.tsx
git commit -m "feat(dashboard): accuracy benchmark types + Accuracy (lm-eval) preset group"
```

---

## Task 11: Dashboard — accuracy result card + detail page

**Files:**
- Create: `packages/dashboard/components/accuracy-result-card.tsx`
- Modify: `packages/dashboard/app/benchmarks/[id]/page.tsx`

**Interfaces:**
- Consumes: `BenchmarkRun.accuracyScore`, `BenchmarkRun.accuracyMetrics`, `AccuracyMetric` (Task 10).

- [ ] **Step 1: Create the card.** `components/accuracy-result-card.tsx`:

```tsx
import type { AccuracyMetric, BenchmarkRun } from "@/lib/benchmarks";

// Defensive parse: accuracyMetrics is server-written JSON; a malformed value
// degrades to "no breakdown" rather than blanking the detail page.
function parseMetrics(raw: string | null): AccuracyMetric[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as AccuracyMetric[]) : [];
  } catch {
    return [];
  }
}

export function AccuracyResultCard({ run }: { run: BenchmarkRun }) {
  const metrics = parseMetrics(run.accuracyMetrics);

  return (
    <div className="space-y-6">
      <div>
        <div className="text-5xl font-semibold">
          {run.accuracyScore != null ? run.accuracyScore.toFixed(1) : "—"}
          <span className="text-xl text-gray-500">/100</span>
        </div>
        <div className="text-sm text-gray-400 mt-1">Primary metric</div>
      </div>

      <div className="space-y-1">
        <div className="text-sm font-medium text-gray-300">Per-task breakdown</div>
        {metrics.length === 0 && <div className="text-sm text-gray-500">No metric data.</div>}
        {metrics.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="text-left text-xs text-gray-500 uppercase tracking-wide border-b border-gray-800">
                <tr>
                  <th className="px-3 py-2">Task</th>
                  <th className="px-3 py-2">Metric</th>
                  <th className="px-3 py-2 text-right">Value</th>
                  <th className="px-3 py-2 text-right">± stderr</th>
                  <th className="px-3 py-2 text-right">n</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m, i) => (
                  <tr key={`${m.task}-${m.metric}-${i}`} className="border-b border-gray-800 last:border-b-0">
                    <td className={`px-3 py-2 ${m.isGroup ? "font-medium" : "text-gray-400 pl-6"}`}>{m.task}</td>
                    <td className="px-3 py-2 text-gray-400">{m.metric}</td>
                    <td className="px-3 py-2 text-right font-mono">{(m.value * 100).toFixed(1)}</td>
                    <td className="px-3 py-2 text-right font-mono text-gray-500">
                      {m.stderr != null ? (m.stderr * 100).toFixed(1) : "—"}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-gray-500">{m.nSamples ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire it into the detail page.** In `app/benchmarks/[id]/page.tsx`:

Add the import:

```tsx
import { AccuracyResultCard } from "@/components/accuracy-result-card";
```

Replace the result-rendering ternary (the `run.kind === "tool-eval" ? (…) : (…)` block) with a three-way branch:

```tsx
      {run.kind === "accuracy" ? (
        run.status === "completed" && <AccuracyResultCard run={run} />
      ) : run.kind === "tool-eval" ? (
        run.status === "completed" && <ToolEvalResultCard run={run} />
      ) : (
        run.results && run.results.length > 0 && (
          <>
            <BenchmarkChart series={series} metric="tps" />
            <BenchmarkChart series={series} metric="ttfrMs" />
            <BenchmarkResultTable rows={run.results} />
          </>
        )
      )}
```

Update the raw-JSON label to name lm-eval for accuracy runs:

```tsx
          <summary className="cursor-pointer text-gray-400">
            Raw {run.kind === "tool-eval" ? "tool-eval-bench" : run.kind === "accuracy" ? "lm-eval" : "llama-benchy"} JSON
          </summary>
```

- [ ] **Step 3: Verify the dashboard builds.**

Run: `npm run build --workspace=packages/dashboard`
Expected: build succeeds.

- [ ] **Step 4: Commit.**

```bash
git add packages/dashboard/components/accuracy-result-card.tsx packages/dashboard/app/benchmarks/[id]/page.tsx
git commit -m "feat(dashboard): accuracy result card on the benchmark detail page"
```

---

## Task 12: Dashboard — list kind pill/score + filter options + compare

**Files:**
- Modify: `packages/dashboard/app/benchmarks/page.tsx`
- Modify: `packages/dashboard/app/benchmarks/compare/page.tsx`

**Interfaces:**
- Consumes: `BenchmarkRun.kind`, `BenchmarkRun.accuracyScore` (Task 10).

- [ ] **Step 1: List page — kind pill.** In `app/benchmarks/page.tsx`, replace the kind-pill cell:

```tsx
                    <td className="px-4 py-3">
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        r.kind === "tool-eval" ? "bg-purple-900 text-purple-200"
                        : r.kind === "accuracy" ? "bg-emerald-900 text-emerald-200"
                        : "bg-sky-900 text-sky-200"
                      }`}>
                        {r.kind}
                      </span>
                    </td>
```

- [ ] **Step 2: List page — score column.** Replace the "Decode t/s" value cell to show the accuracy score for accuracy runs:

```tsx
                    <td className="px-4 py-3 text-right font-mono">
                      {r.kind === "tool-eval"
                        ? (r.toolEvalScore != null ? `${r.toolEvalScore}/100` : "—")
                        : r.kind === "accuracy"
                        ? (r.accuracyScore != null ? `${r.accuracyScore.toFixed(1)}/100` : "—")
                        : (r.meanTps != null ? r.meanTps.toFixed(1) : "—")}
                    </td>
```

- [ ] **Step 3: List page — preset filter options.** In the preset `<select>`, add the accuracy options after the `tool-eval-pressure` option:

```tsx
          <option value="acc-ifeval-quick">acc-ifeval-quick</option>
          <option value="acc-ifeval-full">acc-ifeval-full</option>
          <option value="acc-mmlu-pro-quick">acc-mmlu-pro-quick</option>
          <option value="acc-mmlu-pro-full">acc-mmlu-pro-full</option>
          <option value="acc-gpqa-diamond-quick">acc-gpqa-diamond-quick</option>
          <option value="acc-gpqa-diamond-full">acc-gpqa-diamond-full</option>
          <option value="acc-gsm8k-quick">acc-gsm8k-quick</option>
          <option value="acc-gsm8k-full">acc-gsm8k-full</option>
          <option value="acc-bbh-quick">acc-bbh-quick</option>
          <option value="acc-bbh-full">acc-bbh-full</option>
          <option value="acc-math-hard-quick">acc-math-hard-quick</option>
          <option value="acc-math-hard-full">acc-math-hard-full</option>
```

- [ ] **Step 4: Compare page — accuracy score bars.** In `app/benchmarks/compare/page.tsx`, add a bar comparison for accuracy runs. Insert this block immediately before the three `<BenchmarkChart …>` lines (inside the returned `<div className="p-6 space-y-6">`):

```tsx
      {runs.some((r) => r.kind === "accuracy") && (
        <div className="space-y-2">
          <h2 className="text-lg font-medium">Accuracy score</h2>
          {runs
            .filter((r) => r.kind === "accuracy")
            .map((r, i) => (
              <div key={r.id} className="text-sm">
                <div className="flex justify-between text-xs text-gray-400 mb-0.5">
                  <span>{r.deployment?.displayName ?? r.modelName} · {r.presetId ?? "custom"}</span>
                  <span>{r.accuracyScore != null ? `${r.accuracyScore.toFixed(1)}/100` : "—"}</span>
                </div>
                <div className="h-2 rounded bg-gray-800 overflow-hidden">
                  <div
                    className="h-full"
                    style={{
                      width: `${Math.max(0, Math.min(100, r.accuracyScore ?? 0))}%`,
                      background: PALETTE[i % PALETTE.length],
                    }}
                  />
                </div>
              </div>
            ))}
        </div>
      )}
```

- [ ] **Step 5: Verify the dashboard builds.**

Run: `npm run build --workspace=packages/dashboard`
Expected: build succeeds.

- [ ] **Step 6: Full suite green.**

Run: `npm test`
Expected: PASS (all server suites).

- [ ] **Step 7: Commit.**

```bash
git add packages/dashboard/app/benchmarks/page.tsx packages/dashboard/app/benchmarks/compare/page.tsx
git commit -m "feat(dashboard): accuracy kind pill, score column, filters + compare bars"
```

---

## Manual verification (post-implementation)

Automated tests never spawn a real `uvx lm_eval`. After the tasks are done, verify end-to-end against a live deployment (this is the environmental behavior noted in the spec):

1. Ensure `HF_TOKEN` with access to `Idavidrein/gpqa` is set in the server environment (only needed for the GPQA presets; the other five run without it).
2. With a GLM-5.2 deployment `running`, open **Deployments → Benchmark** on that row, pick **Accuracy (lm-eval) → IFEval — quick (100)**, and start it.
3. Watch the live log stream (first run downloads the IFEval dataset from HF), confirm the run reaches `completed`, and confirm the detail page shows a headline score + per-task breakdown.
4. Sanity-check the score is non-trivial (not ~0) — a near-zero score signals the reasoning strip / `max_gen_toks` budget needs tuning for this model.
5. Run a second accuracy benchmark on a different deployment (or the unpruned vs pruned GLM-5.2) and confirm the **Compare** view shows both accuracy bars — the prune-quality use case.

## Self-Review notes

- **Spec coverage:** execution model (Tasks 3, 8, 9), reasoning strip proxy (Tasks 4, 7, 8), presets/lineup + quick/full (Task 2), data model (Task 1), parser + fail-fast (Task 5), result-file locator (Task 6), route persistence (Task 9), dashboard form/detail/compare/list (Tasks 10–12), GPQA `HF_TOKEN` + first-run download caveats (Manual verification). All spec sections map to a task.
- **No agent version bump** (Global Constraints) — no `packages/agent/src` files touched.
- **Breaking-test guards:** Task 2 explicitly updates the exhaustive id list and the kind-contains assertion in `presets.test.ts`; Task 9 updates the orchestrator mock in the route integration test and runs the full suite.
