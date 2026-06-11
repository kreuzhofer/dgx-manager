# Tool-Eval-Bench Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second benchmark *kind* — `tool-eval` (tool-calling accuracy via the `tool-eval-bench` CLI) — to the existing Benchmarks feature, reusing the run lifecycle and isolating only the per-kind command, parser, and result UI.

**Architecture:** `BenchmarkRun` gains a `kind` discriminator plus nullable tool-eval score columns and a `ToolEvalCategory` child table. The server orchestrator is refactored so generic spawn/log/cancel is shared and only argv + result parsing branch by kind. Four tool-eval presets are added to the same launcher; the detail page branches its result rendering on `run.kind`. Eval-only (no `--perf`) for v1.

**Tech Stack:** Express 5, Prisma (SQLite), Vitest + fast-check + supertest, Next.js 15 / React 19, Tailwind CSS 4. Upstream CLI pinned: `tool-eval-bench@c3868bff099592c9a1045de2c9a3dc24abebb7fb` (v2.0.6).

**Reference spec:** `docs/superpowers/specs/2026-06-11-tool-eval-bench-design.md`
**Captured fixture (already committed):** `packages/server/src/__tests__/integration/benchmarks.fixtures/tool-eval-result.json`

**No agent version bump:** this feature touches only `packages/server` and `packages/dashboard`. `packages/agent/src/` is untouched, so `./scripts/bump-agent-version.sh` is NOT run.

---

## File Structure

**Server (`packages/server/src/`)**
- `benchmarks/presets.ts` *(modify)* — add `kind` to `BenchmarkPreset`, `ToolEvalConfig` type, union `config`, 4 tool-eval presets.
- `benchmarks/tool-eval-args.ts` *(create)* — `buildToolEvalArgs(config, target)`.
- `benchmarks/tool-eval-args.test.ts` *(create)*.
- `benchmarks/tool-eval-parser.ts` *(create)* — `parseToolEvalResults(jsonText)`.
- `benchmarks/tool-eval-parser.test.ts` *(create)*.
- `benchmarks/orchestrator.ts` *(modify)* — extract shared `spawnTracked`, add `runToolEval`.
- `benchmarks/orchestrator.test.ts` *(modify)* — add a tool-eval spawn test.
- `routes/benchmarks.ts` *(modify)* — dispatch `POST /` by `preset.kind`; persist eval fields + categories; include categories in `GET /:id`.
- `__tests__/integration/benchmarks.routes.test.ts` *(modify)* — tool-eval preset + persistence cases.

**Schema**
- `prisma/schema.prisma` *(modify)* — `BenchmarkRun` columns + `ToolEvalCategory` model.

**Dashboard (`packages/dashboard/`)**
- `lib/benchmarks.ts` *(modify)* — types: `kind`, `toolEval*`, `ToolEvalCategory`, `ToolEvalConfig`.
- `components/benchmark-form-modal.tsx` *(modify)* — group presets by kind, guard the throughput-only summary line.
- `components/tool-eval-result-card.tsx` *(create)* — score + category bars + safety warnings.
- `app/benchmarks/[id]/page.tsx` *(modify)* — branch result rendering on `run.kind`.
- `app/benchmarks/page.tsx` *(modify)* — kind badge + score-or-t/s summary column.

---

## Task 1: Prisma schema — `kind`, tool-eval columns, `ToolEvalCategory`

**Files:**
- Modify: `prisma/schema.prisma:231-285` (the `BenchmarkRun` and `BenchmarkResult` block)

- [ ] **Step 1: Add the `kind` field and tool-eval columns to `BenchmarkRun`**

In `prisma/schema.prisma`, inside `model BenchmarkRun`, immediately after the `status` line (`status String @default("pending") …`) add:

```prisma
  // Which benchmark family this run belongs to. "throughput" = llama-benchy
  // (pp/tg/depth/concurrency -> TPS/TTFR rows). "tool-eval" = tool-eval-bench
  // (tool-calling accuracy -> a score + per-category breakdown).
  kind            String   @default("throughput") // throughput | tool-eval
```

Then, immediately after the `meanTtfrMs Float?` line, add the tool-eval headline columns:

```prisma
  // --- tool-eval-bench headline metrics (null on throughput runs) ---
  toolEvalScore          Float?  // scores.final_score, 0-100
  toolEvalRating         String? // rating star string, e.g. "★★★ Adequate"
  toolEvalDeployability  Int?    // deployability
  toolEvalResponsiveness Int?    // responsiveness (small star-style int, NOT 0-100)
  toolEvalTotalScenarios Int?    // total_scenarios
  toolEvalTotalPoints    Int?    // scores.total_points
  toolEvalMaxPoints      Int?    // scores.max_points
  toolEvalSafetyWarnings String? // JSON-encoded string[] (safety_warnings)
```

And add the relation field next to the existing `results BenchmarkResult[]` line:

```prisma
  toolEvalCategories ToolEvalCategory[]
```

- [ ] **Step 2: Add the `ToolEvalCategory` model**

Immediately after the closing brace of `model BenchmarkResult`, add:

```prisma
model ToolEvalCategory {
  id           String @id @default(cuid())
  runId        String
  // From scores.category_scores[] in the tool-eval-bench JSON.
  code         String // category, e.g. "A"
  label        String // e.g. "Tool Selection"
  percent      Float  // 0-100
  earned       Int
  maxPoints    Int    // category_scores[].max
  passCount    Int
  partialCount Int
  failCount    Int
  run          BenchmarkRun @relation(fields: [runId], references: [id], onDelete: Cascade)

  @@index([runId])
}
```

- [ ] **Step 3: Apply schema to the dev DB and regenerate the client**

Run:
```bash
npm run db:push && npm run db:generate
```
Expected: `db:push` reports the new column/table applied (additive, no data loss prompt); `db:generate` reports the Prisma client regenerated. No `--force-reset`.

- [ ] **Step 4: Type-check the server compiles against the new client**

Run:
```bash
npx tsc -p packages/server --noEmit
```
Expected: exits 0 (the new fields are now known to the Prisma client; no code references them yet, so nothing breaks).

- [ ] **Step 5: Commit**

```bash
git add prisma/schema.prisma
git commit -m "feat(db): add kind + tool-eval columns and ToolEvalCategory to BenchmarkRun"
```

---

## Task 2: Presets — `kind`, `ToolEvalConfig`, four tool-eval presets

**Files:**
- Modify: `packages/server/src/benchmarks/presets.ts`
- Test: `packages/server/src/benchmarks/presets.test.ts` *(create)*

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/benchmarks/presets.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { BENCHMARK_PRESETS, getPreset } from "./presets.js";

describe("tool-eval presets", () => {
  const ids = ["tool-eval-quick", "tool-eval-full", "tool-eval-hardmode", "tool-eval-pressure"];

  it("registers all four tool-eval presets with kind 'tool-eval'", () => {
    for (const id of ids) {
      const p = getPreset(id);
      expect(p, `preset ${id} should exist`).toBeDefined();
      expect(p!.kind).toBe("tool-eval");
    }
  });

  it("keeps the five throughput presets tagged kind 'throughput'", () => {
    const throughputIds = ["quick-smoke", "chat-short", "chat-long", "code-32k", "throughput"];
    for (const id of throughputIds) {
      expect(getPreset(id)!.kind).toBe("throughput");
    }
  });

  it("maps each tool-eval preset to the documented flag combination", () => {
    const cfg = (id: string) => getPreset(id)!.config as {
      short: boolean; hardmode: boolean; contextPressure: number | null; seed: number;
    };
    expect(cfg("tool-eval-quick")).toMatchObject({ short: true, hardmode: false, contextPressure: null });
    expect(cfg("tool-eval-full")).toMatchObject({ short: false, hardmode: false, contextPressure: null });
    expect(cfg("tool-eval-hardmode")).toMatchObject({ short: false, hardmode: true, contextPressure: null });
    expect(cfg("tool-eval-pressure")).toMatchObject({ short: false, hardmode: false, contextPressure: 0.75 });
  });

  it("every preset carries a kind field", () => {
    for (const p of BENCHMARK_PRESETS) {
      expect(["throughput", "tool-eval"]).toContain(p.kind);
    }
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
npx vitest run packages/server/src/benchmarks/presets.test.ts
```
Expected: FAIL — `kind` does not exist on `BenchmarkPreset`, and `getPreset("tool-eval-quick")` is `undefined`.

- [ ] **Step 3: Implement the types and presets**

In `packages/server/src/benchmarks/presets.ts`:

Add the `ToolEvalConfig` type after the `BenchmarkConfig` type (after line 12):

```ts
export type ToolEvalConfig = {
  short: boolean;          // --short (15 core scenarios) vs full 63
  hardmode: boolean;       // --hardmode (adds the hard scenario tier)
  contextPressure: number | null; // --context-pressure R (0-1); null = omit
  seed: number;            // --seed N, for reproducible runs
};
```

Change the `BenchmarkPreset` type (lines 14-19) to carry `kind` and a union config:

```ts
export type BenchmarkKind = "throughput" | "tool-eval";

export type BenchmarkPreset = {
  id: string;
  label: string;
  description: string;
  kind: BenchmarkKind;
  config: BenchmarkConfig | ToolEvalConfig;
};
```

Add `kind: "throughput",` to each of the five existing preset objects (one line per preset, placed right after its `description:` line). For example `quick-smoke` becomes:

```ts
  {
    id: "quick-smoke",
    label: "Quick smoke",
    description: "30-second sanity check: one short prompt, one generation.",
    kind: "throughput",
    config: {
      pp: [128],
      // …unchanged…
    },
  },
```

Then append the four tool-eval presets to the `BENCHMARK_PRESETS` array (after the `throughput` preset, before the closing `];`):

```ts
  {
    id: "tool-eval-quick",
    label: "Tool eval — quick (15)",
    description: "15 core tool-calling scenarios: a fast tool-use sanity check.",
    kind: "tool-eval",
    config: { short: true, hardmode: false, contextPressure: null, seed: 42 },
  },
  {
    id: "tool-eval-full",
    label: "Tool eval — full (63)",
    description: "Full 63-scenario tool-calling suite across all categories.",
    kind: "tool-eval",
    config: { short: false, hardmode: false, contextPressure: null, seed: 42 },
  },
  {
    id: "tool-eval-hardmode",
    label: "Tool eval — hard mode",
    description: "Full suite plus the harder scenario tier (Category P).",
    kind: "tool-eval",
    config: { short: false, hardmode: true, contextPressure: null, seed: 42 },
  },
  {
    id: "tool-eval-pressure",
    label: "Tool eval — context pressure",
    description: "Full suite with context filled to 75% to stress long-context tool use.",
    kind: "tool-eval",
    config: { short: false, hardmode: false, contextPressure: 0.75, seed: 42 },
  },
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
npx vitest run packages/server/src/benchmarks/presets.test.ts
```
Expected: PASS (4 tests).

- [ ] **Step 5: Verify the existing args test still compiles against the union config**

Run:
```bash
npx vitest run packages/server/src/benchmarks/args.test.ts
```
Expected: PASS (the existing throughput tests are unaffected; `buildBenchyArgs` still takes `BenchmarkConfig`).

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/benchmarks/presets.ts packages/server/src/benchmarks/presets.test.ts
git commit -m "feat(bench): add ToolEvalConfig and four tool-eval presets"
```

---

## Task 3: `buildToolEvalArgs`

**Files:**
- Create: `packages/server/src/benchmarks/tool-eval-args.ts`
- Test: `packages/server/src/benchmarks/tool-eval-args.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/benchmarks/tool-eval-args.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { buildToolEvalArgs } from "./tool-eval-args.js";
import type { ToolEvalConfig } from "./presets.js";

const base: ToolEvalConfig = { short: false, hardmode: false, contextPressure: null, seed: 42 };
const target = { baseUrl: "http://10.0.0.1:8000/v1", modelName: "m", outputPath: "/out/result.json" };

function valuesAfter(args: string[], flag: string): string[] {
  const idx = args.indexOf(flag);
  if (idx < 0) return [];
  const tail = args.slice(idx + 1);
  const stop = tail.findIndex((t) => t.startsWith("--"));
  return stop === -1 ? tail : tail.slice(0, stop);
}

describe("buildToolEvalArgs", () => {
  it("always emits base-url, explicit model, json-file and seed", () => {
    const args = buildToolEvalArgs(base, target);
    expect(valuesAfter(args, "--base-url")).toEqual(["http://10.0.0.1:8000/v1"]);
    expect(valuesAfter(args, "--model")).toEqual(["m"]);
    expect(valuesAfter(args, "--json-file")).toEqual(["/out/result.json"]);
    expect(valuesAfter(args, "--seed")).toEqual(["42"]);
  });

  it("omits --short, --hardmode and --context-pressure for the full default", () => {
    const args = buildToolEvalArgs(base, target);
    expect(args).not.toContain("--short");
    expect(args).not.toContain("--hardmode");
    expect(args).not.toContain("--context-pressure");
  });

  it("includes --short only when short is set", () => {
    expect(buildToolEvalArgs({ ...base, short: true }, target)).toContain("--short");
  });

  it("includes --hardmode only when hardmode is set", () => {
    expect(buildToolEvalArgs({ ...base, hardmode: true }, target)).toContain("--hardmode");
  });

  it("includes --context-pressure with one value token only when non-null", () => {
    const on = buildToolEvalArgs({ ...base, contextPressure: 0.75 }, target);
    expect(valuesAfter(on, "--context-pressure")).toEqual(["0.75"]);
    const off = buildToolEvalArgs({ ...base, contextPressure: null }, target);
    expect(off).not.toContain("--context-pressure");
  });

  // Invariant: --model is always present exactly once (so the interactive
  // model picker can never hang a headless run).
  test.prop([fc.boolean(), fc.boolean(), fc.option(fc.float({ min: 0, max: 1, noNaN: true }), { nil: null })])(
    "always passes --model exactly once regardless of variant flags",
    (short, hardmode, contextPressure) => {
      const args = buildToolEvalArgs({ short, hardmode, contextPressure, seed: 7 }, target);
      expect(args.filter((a) => a === "--model").length).toBe(1);
    },
  );
});
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
npx vitest run packages/server/src/benchmarks/tool-eval-args.test.ts
```
Expected: FAIL — `Cannot find module './tool-eval-args.js'`.

- [ ] **Step 3: Implement `buildToolEvalArgs`**

Create `packages/server/src/benchmarks/tool-eval-args.ts`:

```ts
import type { ToolEvalConfig } from "./presets.js";

export type ToolEvalTarget = {
  baseUrl: string;   // OpenAI-compatible base URL, already including /v1
  modelName: string; // passed explicitly to skip tool-eval-bench's picker
  outputPath: string;
};

// tool-eval-bench is an OpenAI-compatible CLI. We always pass --model so the
// interactive /v1/models picker can never block a headless run, and
// --json-file so results land at our conventional result.json path
// (--json-file implies --json). Variant flags are boolean/optional toggles.
export function buildToolEvalArgs(
  config: ToolEvalConfig,
  target: ToolEvalTarget,
): string[] {
  const args: string[] = [
    "--base-url", target.baseUrl,
    "--model", target.modelName,
    "--json-file", target.outputPath,
    "--seed", String(config.seed),
  ];
  if (config.short) args.push("--short");
  if (config.hardmode) args.push("--hardmode");
  if (config.contextPressure !== null) {
    args.push("--context-pressure", String(config.contextPressure));
  }
  return args;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
npx vitest run packages/server/src/benchmarks/tool-eval-args.test.ts
```
Expected: PASS (5 examples + 1 property).

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/benchmarks/tool-eval-args.ts packages/server/src/benchmarks/tool-eval-args.test.ts
git commit -m "feat(bench): add buildToolEvalArgs"
```

---

## Task 4: `parseToolEvalResults`

**Files:**
- Create: `packages/server/src/benchmarks/tool-eval-parser.ts`
- Test: `packages/server/src/benchmarks/tool-eval-parser.test.ts`
- Reads: `packages/server/src/__tests__/integration/benchmarks.fixtures/tool-eval-result.json` (already committed)

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/benchmarks/tool-eval-parser.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseToolEvalResults } from "./tool-eval-parser.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = join(here, "../__tests__/integration/benchmarks.fixtures/tool-eval-result.json");
const fixture = readFileSync(fixturePath, "utf-8");

describe("parseToolEvalResults", () => {
  it("maps headline fields from the real captured sample", () => {
    const r = parseToolEvalResults(fixture);
    expect(r.finalScore).toBe(67);
    expect(r.rating).toBe("★★★ Adequate");
    expect(r.deployability).toBe(48);
    expect(r.responsiveness).toBe(2);
    expect(r.totalScenarios).toBe(15);
    expect(r.totalPoints).toBe(20);
    expect(r.maxPoints).toBe(30);
    expect(r.safetyWarnings).toEqual([]);
  });

  it("maps every category_scores entry 1:1", () => {
    const r = parseToolEvalResults(fixture);
    expect(r.categories.length).toBe(5);
    const a = r.categories.find((c) => c.code === "A")!;
    expect(a).toMatchObject({
      code: "A", label: "Tool Selection", percent: 100,
      earned: 6, maxPoints: 6, passCount: 3, partialCount: 0, failCount: 0,
    });
  });

  it("throws on invalid JSON", () => {
    expect(() => parseToolEvalResults("not json")).toThrow(/failed to parse tool-eval JSON/);
  });

  it("throws when final_score is missing", () => {
    const bad = JSON.stringify({ rating: "x", total_scenarios: 1, scores: { category_scores: [] } });
    expect(() => parseToolEvalResults(bad)).toThrow(/final_score/);
  });

  it("throws when scores.category_scores is missing", () => {
    const bad = JSON.stringify({ final_score: 1, rating: "x", total_scenarios: 1, scores: {} });
    expect(() => parseToolEvalResults(bad)).toThrow(/category_scores/);
  });

  it("defaults safety_warnings to [] and optional ints to null when absent", () => {
    const minimal = JSON.stringify({
      final_score: 50, rating: "★★", total_scenarios: 3,
      scores: { category_scores: [] },
    });
    const r = parseToolEvalResults(minimal);
    expect(r.safetyWarnings).toEqual([]);
    expect(r.deployability).toBeNull();
    expect(r.responsiveness).toBeNull();
    expect(r.totalPoints).toBeNull();
    expect(r.maxPoints).toBeNull();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
npx vitest run packages/server/src/benchmarks/tool-eval-parser.test.ts
```
Expected: FAIL — `Cannot find module './tool-eval-parser.js'`.

- [ ] **Step 3: Implement `parseToolEvalResults`**

Create `packages/server/src/benchmarks/tool-eval-parser.ts`:

```ts
export type ToolEvalCategoryInput = {
  code: string;
  label: string;
  percent: number;
  earned: number;
  maxPoints: number;
  passCount: number;
  partialCount: number;
  failCount: number;
};

export type ToolEvalSummary = {
  finalScore: number;
  rating: string;
  deployability: number | null;
  responsiveness: number | null;
  totalScenarios: number;
  totalPoints: number | null;
  maxPoints: number | null;
  safetyWarnings: string[];
  categories: ToolEvalCategoryInput[];
};

type Obj = Record<string, unknown>;

function reqNum(o: Obj, key: string): number {
  const v = o[key];
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new Error(`tool-eval result missing required numeric field: ${key}`);
  }
  return v;
}

function reqStr(o: Obj, key: string): string {
  const v = o[key];
  if (typeof v !== "string") {
    throw new Error(`tool-eval result missing required string field: ${key}`);
  }
  return v;
}

function optNum(o: Obj, key: string): number | null {
  const v = o[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// Parse the tool-eval-bench --json payload (schema_version "1", CLI v2.0.6).
// Headline fields are top-level; total/max points and the per-category
// breakdown live under `scores`. Fail-fast: anything we depend on that is
// absent or the wrong type throws, rather than silently defaulting.
export function parseToolEvalResults(jsonText: string): ToolEvalSummary {
  let parsed: Obj;
  try {
    parsed = JSON.parse(jsonText) as Obj;
  } catch (e) {
    throw new Error(`failed to parse tool-eval JSON: ${(e as Error).message}`);
  }

  const scores = parsed.scores;
  if (!scores || typeof scores !== "object") {
    throw new Error("tool-eval result missing required object: scores");
  }
  const s = scores as Obj;

  const rawCats = s.category_scores;
  if (!Array.isArray(rawCats)) {
    throw new Error("tool-eval result missing required array: scores.category_scores");
  }
  const categories: ToolEvalCategoryInput[] = rawCats.map((c) => {
    const cat = c as Obj;
    return {
      code: reqStr(cat, "category"),
      label: reqStr(cat, "label"),
      percent: reqNum(cat, "percent"),
      earned: reqNum(cat, "earned"),
      maxPoints: reqNum(cat, "max"),
      passCount: reqNum(cat, "pass_count"),
      partialCount: reqNum(cat, "partial_count"),
      failCount: reqNum(cat, "fail_count"),
    };
  });

  const rawWarnings = parsed.safety_warnings;
  const safetyWarnings = Array.isArray(rawWarnings) ? rawWarnings.map(String) : [];

  return {
    finalScore: reqNum(parsed, "final_score"),
    rating: reqStr(parsed, "rating"),
    deployability: optNum(parsed, "deployability"),
    responsiveness: optNum(parsed, "responsiveness"),
    totalScenarios: reqNum(parsed, "total_scenarios"),
    totalPoints: optNum(s, "total_points"),
    maxPoints: optNum(s, "max_points"),
    safetyWarnings,
    categories,
  };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
npx vitest run packages/server/src/benchmarks/tool-eval-parser.test.ts
```
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/benchmarks/tool-eval-parser.ts packages/server/src/benchmarks/tool-eval-parser.test.ts
git commit -m "feat(bench): add parseToolEvalResults with fail-fast field validation"
```

---

## Task 5: Orchestrator — extract `spawnTracked`, add `runToolEval`

**Files:**
- Modify: `packages/server/src/benchmarks/orchestrator.ts`
- Modify: `packages/server/src/benchmarks/orchestrator.test.ts`

- [ ] **Step 1: Write the failing test**

In `packages/server/src/benchmarks/orchestrator.test.ts`, add the import of `runToolEval` to the existing import line:

```ts
import { runBenchmark, runToolEval, cancelBenchmark } from "./orchestrator.js";
```

Then add a new `describe` block at the end of the file (before the final closing of the file):

```ts
describe("runToolEval", () => {
  beforeEach(() => {
    spawnMock.mockReset();
    readFileSyncMock.mockReset();
    mkdirSyncMock.mockReset();
    existsSyncMock.mockReset();
  });

  it("spawns `uvx tool-eval-bench` with the supplied argv and detached", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(true);
    readFileSyncMock.mockReturnValue(
      JSON.stringify({
        final_score: 80, rating: "★★★★", total_scenarios: 15, safety_warnings: [],
        scores: { total_points: 24, max_points: 30, category_scores: [] },
      }),
    );

    const onLog = vi.fn();
    const promise = runToolEval({
      runId: "run_te",
      args: ["--base-url", "http://10.0.0.1:8000/v1", "--model", "m", "--json-file", "/mnt/tank/benchmarks/run_te/result.json", "--seed", "42"],
      outputDir: "/mnt/tank/benchmarks/run_te",
      onLog,
    });

    child.stderr.emit("data", Buffer.from('{"event":"scenario_start","scenario_id":"TC-01"}\n'));
    child.emit("close", 0);

    const result = await promise;
    const [cmd, argv, spawnOpts] = spawnMock.mock.calls[0];
    expect(cmd).toBe("uvx");
    expect(argv[0]).toBe("--from");
    expect(argv[1]).toMatch(/tool-eval-bench\.git@[0-9a-f]{7,}$/);
    expect(argv[2]).toBe("tool-eval-bench");
    expect(argv.slice(3)).toEqual([
      "--base-url", "http://10.0.0.1:8000/v1",
      "--model", "m",
      "--json-file", "/mnt/tank/benchmarks/run_te/result.json",
      "--seed", "42",
    ]);
    expect((spawnOpts as { detached: boolean }).detached).toBe(true);
    expect(result.exitCode).toBe(0);
    expect(result.summary?.finalScore).toBe(80);
    expect(onLog).toHaveBeenCalledWith('{"event":"scenario_start","scenario_id":"TC-01"}');
  });

  it("returns a null summary when the process exits non-zero", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(false);

    const promise = runToolEval({
      runId: "run_te2", args: ["--base-url", "u", "--model", "m", "--json-file", "/o/result.json"],
      outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 1);
    const result = await promise;
    expect(result.exitCode).toBe(1);
    expect(result.summary).toBeNull();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
npx vitest run packages/server/src/benchmarks/orchestrator.test.ts
```
Expected: FAIL — `runToolEval` is not exported.

- [ ] **Step 3: Refactor the orchestrator to share spawn plumbing and add `runToolEval`**

Replace the entire contents of `packages/server/src/benchmarks/orchestrator.ts` with:

```ts
import { spawn, type ChildProcess } from "node:child_process";
import { readFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import {
  parseBenchyResults,
  summarizeResults,
  type BenchmarkResultInput,
} from "./parser.js";
import {
  parseToolEvalResults,
  type ToolEvalSummary,
} from "./tool-eval-parser.js";

const LLAMA_BENCHY_SPEC =
  process.env.LLAMA_BENCHY_VERSION
    ? `llama-benchy==${process.env.LLAMA_BENCHY_VERSION}`
    : "llama-benchy";

// Pinned upstream commit (tool-eval-bench v2.0.6). Overridable for upgrades.
const TOOL_EVAL_SPEC =
  process.env.TOOL_EVAL_BENCH_REF ||
  "git+https://github.com/SeraphimSerapis/tool-eval-bench.git@c3868bff099592c9a1045de2c9a3dc24abebb7fb";

// In-memory registry of in-flight runs. Lost on restart — see Task 9 for the
// boot-time reconciliation that marks any orphaned "running" rows as failed.
const ACTIVE: Map<string, ChildProcess> = new Map();

type SpawnTrackedOpts = {
  runId: string;
  command: string;   // executable, e.g. "uvx"
  args: string[];    // full argv for the executable
  outputDir: string; // mkdir'd before spawn; must contain the result.json path
  onLog: (line: string) => void;
};

type SpawnTrackedResult = { exitCode: number | null; rawOutput: string | null };

// Shared, kind-agnostic plumbing: mkdir the output dir, spawn the process in
// its own group (so we can kill the whole group on cancel), stream stdout and
// stderr line-by-line to onLog, and on exit read result.json verbatim. Parsing
// is left to the caller because it differs per benchmark kind.
function spawnTracked(opts: SpawnTrackedOpts): Promise<SpawnTrackedResult> {
  mkdirSync(opts.outputDir, { recursive: true, mode: 0o777 });

  return new Promise((resolve) => {
    const child = spawn(opts.command, opts.args, {
      stdio: ["ignore", "pipe", "pipe"],
      // detached:true so we can kill the whole process group (uvx may spawn a
      // python subprocess) via process.kill(-pid).
      detached: true,
      // PYTHONUNBUFFERED forces line buffering on the child's piped stdout so
      // our onLog fires live instead of only at process exit.
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    ACTIVE.set(opts.runId, child);

    const pump = (b: Buffer) => {
      for (const line of b.toString().split("\n")) {
        if (line) opts.onLog(line);
      }
    };
    child.stdout?.on("data", pump);
    child.stderr?.on("data", pump);

    child.on("close", (code) => {
      ACTIVE.delete(opts.runId);
      const resultPath = join(opts.outputDir, "result.json");
      let rawOutput: string | null = null;
      if (code === 0 && existsSync(resultPath)) {
        try {
          rawOutput = readFileSync(resultPath, "utf-8");
        } catch (e) {
          opts.onLog(`[read] ${(e as Error).message}`);
        }
      }
      resolve({ exitCode: code, rawOutput });
    });
  });
}

export type RunBenchmarkOpts = {
  runId: string;
  args: string[];     // llama-benchy argv (from buildBenchyArgs)
  outputDir: string;
  onLog: (line: string) => void;
};

export type RunBenchmarkResult = {
  exitCode: number | null;
  results: BenchmarkResultInput[];
  summary: { meanTps: number | null; meanTtfrMs: number | null };
  rawOutput: string | null;
};

export async function runBenchmark(opts: RunBenchmarkOpts): Promise<RunBenchmarkResult> {
  const { exitCode, rawOutput } = await spawnTracked({
    runId: opts.runId,
    command: "uvx",
    args: ["--from", LLAMA_BENCHY_SPEC, "llama-benchy", ...opts.args],
    outputDir: opts.outputDir,
    onLog: opts.onLog,
  });

  let results: BenchmarkResultInput[] = [];
  if (exitCode === 0 && rawOutput) {
    try {
      results = parseBenchyResults(rawOutput);
    } catch (e) {
      opts.onLog(`[parser] ${(e as Error).message}`);
    }
  }
  return { exitCode, results, summary: summarizeResults(results), rawOutput };
}

export type RunToolEvalOpts = {
  runId: string;
  args: string[];     // tool-eval-bench argv (from buildToolEvalArgs)
  outputDir: string;
  onLog: (line: string) => void;
};

export type RunToolEvalResult = {
  exitCode: number | null;
  summary: ToolEvalSummary | null;
  rawOutput: string | null;
};

export async function runToolEval(opts: RunToolEvalOpts): Promise<RunToolEvalResult> {
  const { exitCode, rawOutput } = await spawnTracked({
    runId: opts.runId,
    command: "uvx",
    args: ["--from", TOOL_EVAL_SPEC, "tool-eval-bench", ...opts.args],
    outputDir: opts.outputDir,
    onLog: opts.onLog,
  });

  let summary: ToolEvalSummary | null = null;
  if (exitCode === 0 && rawOutput) {
    try {
      summary = parseToolEvalResults(rawOutput);
    } catch (e) {
      opts.onLog(`[parser] ${(e as Error).message}`);
    }
  }
  return { exitCode, summary, rawOutput };
}

export function cancelBenchmark(runId: string): boolean {
  const child = ACTIVE.get(runId);
  if (!child) return false;
  // Signal the whole process group — we spawned with detached:true, so the CLI
  // and its python subprocess are in their own group. SIGTERM gives the tool a
  // chance to flush partial results before exit.
  if (child.pid) {
    try {
      process.kill(-child.pid, "SIGTERM");
    } catch {
      // ESRCH if the group already exited between detect-and-signal
      child.kill("SIGTERM");
    }
  } else {
    child.kill("SIGTERM");
  }
  return true;
}

export function isRunActive(runId: string): boolean {
  return ACTIVE.has(runId);
}
```

- [ ] **Step 4: Run the orchestrator tests to verify all pass**

Run:
```bash
npx vitest run packages/server/src/benchmarks/orchestrator.test.ts
```
Expected: PASS — the original throughput tests (unchanged `runBenchmark` signature/return) plus the two new `runToolEval` tests.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/benchmarks/orchestrator.ts packages/server/src/benchmarks/orchestrator.test.ts
git commit -m "feat(bench): share spawn plumbing and add runToolEval orchestrator"
```

---

## Task 6: Route — dispatch `POST /` by kind, persist eval results

**Files:**
- Modify: `packages/server/src/routes/benchmarks.ts`
- Modify: `packages/server/src/__tests__/integration/benchmarks.routes.test.ts`

- [ ] **Step 1: Write the failing integration tests**

Open `packages/server/src/__tests__/integration/benchmarks.routes.test.ts` and locate the existing `GET /presets` test (around line 84) and the `POST /api/benchmarks` describe. Add these tests inside the existing top-level `describe` (after the presets test). They reuse the file's existing helpers (`request(app)`, the seeded running deployment, `wipeAll()` — follow the surrounding test's setup exactly for creating a running deployment; reuse the same `seedRunningDeployment()`/equivalent already used by the neighbouring POST tests):

```ts
it("GET /presets includes the four tool-eval presets tagged kind 'tool-eval'", async () => {
  const res = await request(app).get("/api/benchmarks/presets");
  expect(res.status).toBe(200);
  const ids = res.body.map((p: { id: string }) => p.id);
  for (const id of ["tool-eval-quick", "tool-eval-full", "tool-eval-hardmode", "tool-eval-pressure"]) {
    expect(ids).toContain(id);
  }
  const quick = res.body.find((p: { id: string }) => p.id === "tool-eval-quick");
  expect(quick.kind).toBe("tool-eval");
});

it("POST with a tool-eval preset creates a run with kind 'tool-eval'", async () => {
  const deploymentId = await seedRunningDeployment(); // existing helper in this file
  const res = await request(app)
    .post("/api/benchmarks")
    .send({ deploymentId, presetId: "tool-eval-quick" });
  expect(res.status).toBe(201);
  expect(res.body.kind).toBe("tool-eval");
  expect(res.body.presetId).toBe("tool-eval-quick");
});

it("rejects a custom config for a tool-eval run", async () => {
  const deploymentId = await seedRunningDeployment();
  // No presetId + a config body is only valid for throughput; tool-eval is preset-only.
  // This asserts the throughput custom path is unaffected and tool-eval stays preset-only:
  const res = await request(app)
    .post("/api/benchmarks")
    .send({ deploymentId, config: { pp: [1], tg: [1], depth: [0], runs: 1, concurrency: [1], latencyMode: "none", enablePrefixCaching: false, skipCoherence: false } });
  expect(res.status).toBe(201);
  expect(res.body.kind).toBe("throughput");
});
```

Add a persistence test that drives the completion path. Because `runToolEval` shells out, mock it at the module boundary the same way the existing route tests stub the orchestrator (follow the existing `vi.mock("../../benchmarks/orchestrator.js", …)` pattern in this file; if the file currently mocks `runBenchmark`, extend that mock to also export `runToolEval`). Example mock return + assertion:

```ts
// In the orchestrator mock for this suite, make runToolEval resolve with a
// parsed summary so the route's completion handler persists eval fields:
//   runToolEval: vi.fn(async (opts) => { opts.onLog("done"); return {
//     exitCode: 0, rawOutput: "{}",
//     summary: { finalScore: 67, rating: "★★★ Adequate", deployability: 48,
//       responsiveness: 2, totalScenarios: 15, totalPoints: 20, maxPoints: 30,
//       safetyWarnings: [], categories: [
//         { code: "A", label: "Tool Selection", percent: 100, earned: 6,
//           maxPoints: 6, passCount: 3, partialCount: 0, failCount: 0 }] } }; }),

it("persists eval headline fields and category rows on completion", async () => {
  const deploymentId = await seedRunningDeployment();
  const res = await request(app)
    .post("/api/benchmarks")
    .send({ deploymentId, presetId: "tool-eval-quick" });
  const runId = res.body.id;

  // The mocked runToolEval resolves synchronously; await a tick for the
  // completion handler, then read the row back (mirror how existing tests
  // await completion in this file).
  await new Promise((r) => setTimeout(r, 20));

  const detail = await request(app).get(`/api/benchmarks/${runId}`);
  expect(detail.body.status).toBe("completed");
  expect(detail.body.toolEvalScore).toBe(67);
  expect(detail.body.toolEvalRating).toBe("★★★ Adequate");
  expect(detail.body.toolEvalTotalScenarios).toBe(15);
  expect(detail.body.toolEvalCategories.length).toBe(1);
  expect(detail.body.toolEvalCategories[0].code).toBe("A");
});
```

> If the existing suite does not yet have a `seedRunningDeployment()` helper, reuse whatever the neighbouring POST tests use to create a `running` deployment (search the file for `status: "running"`); do not invent a new pattern.

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
npx vitest run packages/server/src/__tests__/integration/benchmarks.routes.test.ts
```
Expected: FAIL — `kind` is undefined on the response, `toolEvalScore`/`toolEvalCategories` absent, route does not branch.

- [ ] **Step 3: Implement the route dispatch**

In `packages/server/src/routes/benchmarks.ts`:

Extend the imports:

```ts
import {
  BENCHMARK_PRESETS,
  getPreset,
  type BenchmarkConfig,
  type ToolEvalConfig,
} from "../benchmarks/presets.js";
import { buildBenchyArgs } from "../benchmarks/args.js";
import { buildToolEvalArgs } from "../benchmarks/tool-eval-args.js";
import { deploymentEndpointUrl } from "../benchmarks/endpoint.js";
import {
  runBenchmark,
  runToolEval,
  cancelBenchmark,
} from "../benchmarks/orchestrator.js";
```

Add `toolEvalCategories: true` to the `include` of the `GET /:id` handler so the detail response carries categories:

```ts
benchmarksRouter.get("/:id", async (req, res) => {
  const run = await prisma.benchmarkRun.findUnique({
    where: { id: req.params.id },
    include: {
      results: true,
      toolEvalCategories: true,
      deployment: { include: { node: true, model: true } },
    },
  });
  if (!run) return res.status(404).json({ error: "not found" });
  res.json(run);
});
```

In the `POST /` handler, replace the config-resolution block (lines 127-134, the `let config: BenchmarkConfig; if (presetId) {…} else {…}`) with a kind-aware resolution:

```ts
  let kind: "throughput" | "tool-eval" = "throughput";
  let config: BenchmarkConfig | ToolEvalConfig;
  if (presetId) {
    const preset = getPreset(presetId);
    if (!preset) return res.status(400).json({ error: "unknown presetId" });
    kind = preset.kind;
    config = preset.config;
  } else {
    // Custom config is throughput-only; tool-eval runs must use a preset.
    config = customConfig!;
  }
```

In the `prisma.benchmarkRun.create({ data: { … } })` call, add `kind` to the data:

```ts
  const run = await prisma.benchmarkRun.create({
    data: {
      deploymentId,
      presetId: presetId ?? null,
      kind,
      modelName: deployment.model.name,
      endpointUrl,
      servedModelName,
      config: JSON.stringify(config),
      status: "pending",
    },
  });
```

Replace the args construction + the `runBenchmark({…}).then(…)` block so the kind selects the runner and the completion handler writes the right columns. Replace from the `const args = buildBenchyArgs(…)` line down to the end of the `.catch(…)` chain with:

```ts
  // Move to "running" immediately so the dashboard reflects state.
  await prisma.benchmarkRun.update({
    where: { id: run.id },
    data: { status: "running", startedAt: new Date() },
  });
  sseBroadcast({
    type: "benchmark:status",
    payload: { id: run.id, status: "running", deploymentId: run.deploymentId },
  });

  const logDir = join(SHARED_STORAGE, "logs", "benchmarks");
  mkdirSync(logDir, { recursive: true, mode: 0o777 });
  const logPath = join(logDir, `${run.id}.log`);
  const onLog = (line: string) => {
    try {
      appendFileSync(logPath, line + "\n", { mode: 0o666 });
    } catch {
      // Disk-full or perms — keep streaming via SSE even if persistence fails.
    }
    sseBroadcast({ type: "benchmark:log", payload: { runId: run.id, log: line } });
  };
  const resultPath = join(outputDir, "result.json");

  const finishFailed = async (message: string) => {
    await prisma.benchmarkRun.update({
      where: { id: run.id },
      data: { status: "failed", completedAt: new Date(), error: message },
    });
    sseBroadcast({
      type: "benchmark:status",
      payload: { id: run.id, status: "failed", error: message },
    });
  };

  if (kind === "tool-eval") {
    const args = buildToolEvalArgs(config as ToolEvalConfig, {
      baseUrl: endpointUrl,
      modelName: servedModelName,
      outputPath: resultPath,
    });
    runToolEval({ runId: run.id, args, outputDir, onLog })
      .then(async (r) => {
        const current = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
        if (current?.status === "canceled") return;
        if (r.exitCode === 0 && r.summary) {
          const s = r.summary;
          await prisma.benchmarkRun.update({
            where: { id: run.id },
            data: {
              status: "completed",
              completedAt: new Date(),
              rawOutput: r.rawOutput,
              toolEvalScore: s.finalScore,
              toolEvalRating: s.rating,
              toolEvalDeployability: s.deployability,
              toolEvalResponsiveness: s.responsiveness,
              toolEvalTotalScenarios: s.totalScenarios,
              toolEvalTotalPoints: s.totalPoints,
              toolEvalMaxPoints: s.maxPoints,
              toolEvalSafetyWarnings: JSON.stringify(s.safetyWarnings),
              toolEvalCategories: { create: s.categories },
            },
          });
          const final = await prisma.benchmarkRun.findUnique({
            where: { id: run.id },
            include: { toolEvalCategories: true },
          });
          sseBroadcast({ type: "benchmark:status", payload: final });
        } else {
          await finishFailed(`tool-eval-bench exited with code ${r.exitCode}`);
        }
      })
      .catch((e) => finishFailed((e as Error).message));
  } else {
    const args = buildBenchyArgs(config as BenchmarkConfig, {
      baseUrl: endpointUrl,
      modelName: servedModelName,
      outputPath: resultPath,
    });
    runBenchmark({ runId: run.id, args, outputDir, onLog })
      .then(async (r) => {
        const current = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
        if (current?.status === "canceled") return;
        if (r.exitCode === 0) {
          await prisma.benchmarkRun.update({
            where: { id: run.id },
            data: {
              status: "completed",
              completedAt: new Date(),
              rawOutput: r.rawOutput,
              meanTps: r.summary.meanTps,
              meanTtfrMs: r.summary.meanTtfrMs,
              results: { create: r.results },
            },
          });
        } else {
          await finishFailed(`llama-benchy exited with code ${r.exitCode}`);
          return;
        }
        const final = await prisma.benchmarkRun.findUnique({
          where: { id: run.id },
          include: { results: true },
        });
        sseBroadcast({ type: "benchmark:status", payload: final });
      })
      .catch((e) => finishFailed((e as Error).message));
  }

  res.status(201).json(run);
});
```

> Note: the `outputDir` const (`join(SHARED_STORAGE, "benchmarks", run.id)`) defined earlier in the handler is reused; keep its existing declaration. Remove the now-replaced single `const args = buildBenchyArgs(...)` that preceded the "running" update.

- [ ] **Step 4: Run the integration tests to verify they pass**

Run:
```bash
npx vitest run packages/server/src/__tests__/integration/benchmarks.routes.test.ts
```
Expected: PASS — including the existing throughput route tests (unchanged behavior) and the new tool-eval cases.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/routes/benchmarks.ts packages/server/src/__tests__/integration/benchmarks.routes.test.ts
git commit -m "feat(bench): route dispatches tool-eval runs and persists scores"
```

---

## Task 7: Dashboard types

**Files:**
- Modify: `packages/dashboard/lib/benchmarks.ts`

- [ ] **Step 1: Add the tool-eval types**

In `packages/dashboard/lib/benchmarks.ts`:

After the `BenchmarkConfig` type, add:

```ts
export type ToolEvalConfig = {
  short: boolean;
  hardmode: boolean;
  contextPressure: number | null;
  seed: number;
};

export type BenchmarkKind = "throughput" | "tool-eval";

export type ToolEvalCategory = {
  id: string;
  code: string;
  label: string;
  percent: number;
  earned: number;
  maxPoints: number;
  passCount: number;
  partialCount: number;
  failCount: number;
};
```

Change `BenchmarkPreset` to carry `kind` and the union config:

```ts
export type BenchmarkPreset = {
  id: string;
  label: string;
  description: string;
  kind: BenchmarkKind;
  config: BenchmarkConfig | ToolEvalConfig;
};
```

Add the fields to `BenchmarkRun` (after `meanTtfrMs: number | null;` and next to `results?:`):

```ts
  kind: BenchmarkKind;
  toolEvalScore: number | null;
  toolEvalRating: string | null;
  toolEvalDeployability: number | null;
  toolEvalResponsiveness: number | null;
  toolEvalTotalScenarios: number | null;
  toolEvalTotalPoints: number | null;
  toolEvalMaxPoints: number | null;
  toolEvalSafetyWarnings: string | null; // JSON-encoded string[]
  toolEvalCategories?: ToolEvalCategory[];
```

- [ ] **Step 2: Type-check the dashboard compiles**

Run:
```bash
npx tsc -p packages/dashboard --noEmit
```
Expected: exits 0 (existing consumers reference only fields that still exist; the union `config` is not yet destructured anywhere that breaks).

> If `tsc` reports an error in `benchmark-form-modal.tsx` about `p.config.pp` on a `ToolEvalConfig`, that is expected and fixed in Task 8 — proceed to Task 8 before committing if so. Otherwise commit now.

- [ ] **Step 3: Commit**

```bash
git add packages/dashboard/lib/benchmarks.ts
git commit -m "feat(dashboard): tool-eval benchmark types"
```

---

## Task 8: Dashboard launcher — group presets by kind

**Files:**
- Modify: `packages/dashboard/components/benchmark-form-modal.tsx`

> No unit test: this repo has no dashboard test harness. Verification is `npm run build` (type-check + Next build) plus the manual check in Step 3.

- [ ] **Step 1: Render presets grouped by kind and guard the throughput-only summary line**

In `packages/dashboard/components/benchmark-form-modal.tsx`, replace the `{!showCustom && ( … )}` preset-list block (the `presets.map(...)` section) with a version that splits by `kind` and only shows the pp/tg summary for throughput presets:

```tsx
        {!showCustom && (
          <div className="space-y-4">
            {(["throughput", "tool-eval"] as const).map((kind) => {
              const group = presets.filter((p) => p.kind === kind);
              if (group.length === 0) return null;
              return (
                <div key={kind} className="space-y-2">
                  <div className="text-xs uppercase tracking-wide text-gray-500">
                    {kind === "throughput" ? "Throughput" : "Tool-calling eval"}
                  </div>
                  {group.map((p) => (
                    <label key={p.id} className="block p-3 rounded bg-gray-800 hover:bg-gray-700 cursor-pointer">
                      <div className="flex items-start gap-2">
                        <input
                          type="radio"
                          name="preset"
                          value={p.id}
                          checked={presetId === p.id}
                          onChange={() => setPresetId(p.id)}
                          className="mt-1"
                        />
                        <div>
                          <div className="font-medium">{p.label}</div>
                          <div className="text-xs text-gray-400">{p.description}</div>
                          {p.kind === "throughput" && "pp" in p.config && (
                            <div className="text-xs text-gray-500 mt-1">
                              pp=[{p.config.pp.join(",")}] tg=[{p.config.tg.join(",")}]
                              {" "}depth=[{p.config.depth.join(",")}]
                              {" "}concurrency=[{p.config.concurrency.join(",")}]
                              {" "}runs={p.config.runs}
                            </div>
                          )}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              );
            })}
          </div>
        )}
```

- [ ] **Step 2: Build the dashboard to verify it type-checks and compiles**

Run:
```bash
npm run build --workspace packages/dashboard
```
Expected: build succeeds; no TS error on `p.config.pp` (the `"pp" in p.config` narrows the union).

- [ ] **Step 3: Manual verification**

Start the app (`MANAGER_ADVERTISE_HOST=<ip> SSH_USER=<user> docker compose up -d --build` per CLAUDE.md, or `npm run dev`), open a running deployment's "Run benchmark" modal, and confirm two groups render ("Throughput" and "Tool-calling eval") with the four new tool-eval options selectable. Record this in the PR description.

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/components/benchmark-form-modal.tsx
git commit -m "feat(dashboard): group benchmark presets by kind in launcher"
```

---

## Task 9: Dashboard — tool-eval result card + detail-page branch

**Files:**
- Create: `packages/dashboard/components/tool-eval-result-card.tsx`
- Modify: `packages/dashboard/app/benchmarks/[id]/page.tsx`

- [ ] **Step 1: Create the result card**

Create `packages/dashboard/components/tool-eval-result-card.tsx`:

```tsx
import type { BenchmarkRun } from "@/lib/benchmarks";

export function ToolEvalResultCard({ run }: { run: BenchmarkRun }) {
  const warnings: string[] = run.toolEvalSafetyWarnings
    ? (JSON.parse(run.toolEvalSafetyWarnings) as string[])
    : [];
  const cats = run.toolEvalCategories ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-baseline">
        <div>
          <div className="text-5xl font-semibold">{run.toolEvalScore ?? "—"}<span className="text-xl text-gray-500">/100</span></div>
          <div className="text-lg text-amber-400 mt-1">{run.toolEvalRating ?? ""}</div>
        </div>
        <div className="text-sm text-gray-400 space-y-1">
          <div>Deployability: <span className="text-gray-200">{run.toolEvalDeployability ?? "—"}</span></div>
          <div>Responsiveness: <span className="text-gray-200">{run.toolEvalResponsiveness ?? "—"}</span></div>
          <div>
            Points: <span className="text-gray-200">{run.toolEvalTotalPoints ?? "—"}/{run.toolEvalMaxPoints ?? "—"}</span>
            {" "}across {run.toolEvalTotalScenarios ?? "—"} scenarios
          </div>
        </div>
      </div>

      {warnings.length > 0 && (
        <div className="rounded border border-red-700 bg-red-950/40 p-3">
          <div className="text-sm font-medium text-red-300 mb-1">Safety warnings</div>
          <ul className="text-xs text-red-200 list-disc ml-4 space-y-0.5">
            {warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}

      <div className="space-y-2">
        <div className="text-sm font-medium text-gray-300">Category breakdown</div>
        {cats.length === 0 && <div className="text-sm text-gray-500">No category data.</div>}
        {cats.map((c) => (
          <div key={c.id} className="text-sm">
            <div className="flex justify-between text-xs text-gray-400 mb-0.5">
              <span>{c.code} · {c.label}</span>
              <span>{c.percent}% ({c.earned}/{c.maxPoints}) · {c.passCount}✓ {c.partialCount}~ {c.failCount}✗</span>
            </div>
            <div className="h-2 rounded bg-gray-800 overflow-hidden">
              <div
                className="h-full bg-emerald-500"
                style={{ width: `${Math.max(0, Math.min(100, c.percent))}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Branch the detail page on `run.kind`**

In `packages/dashboard/app/benchmarks/[id]/page.tsx`:

Add the import near the other component imports:

```tsx
import { ToolEvalResultCard } from "@/components/tool-eval-result-card";
```

Replace the throughput results block (the `{run.results && run.results.length > 0 && ( <> <BenchmarkChart …/> … </> )}` section) with a kind branch:

```tsx
      {run.kind === "tool-eval" ? (
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

> The `series` const above this block references `run.results ?? []`; leave it as-is — it is only consumed inside the throughput branch.

- [ ] **Step 3: Build the dashboard**

Run:
```bash
npm run build --workspace packages/dashboard
```
Expected: build succeeds.

- [ ] **Step 4: Manual verification**

Open a completed tool-eval run's detail page and confirm the score, rating, deployability/responsiveness, points, category bars, and (if any) safety warnings render; confirm a throughput run still shows charts + table. Record in the PR description.

- [ ] **Step 5: Commit**

```bash
git add packages/dashboard/components/tool-eval-result-card.tsx "packages/dashboard/app/benchmarks/[id]/page.tsx"
git commit -m "feat(dashboard): tool-eval result card + detail-page branch"
```

---

## Task 10: Dashboard list — kind badge + score-or-throughput summary

**Files:**
- Modify: `packages/dashboard/app/benchmarks/page.tsx`

- [ ] **Step 1: Add a Kind column header**

In `packages/dashboard/app/benchmarks/page.tsx`, in the `<thead>` row (the block with `<th>When</th>`, `<th>Deployment</th>`, `<th>Preset</th>`…), add a Kind header after the `Preset` header:

```tsx
                  <th className="px-4 py-3">Kind</th>
```

- [ ] **Step 2: Render the kind badge and a kind-aware summary in each row**

In the `filtered.map((r) => ( … ))` body, add a cell after the Preset cell:

```tsx
                    <td className="px-4 py-3">
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        r.kind === "tool-eval" ? "bg-purple-900 text-purple-200" : "bg-sky-900 text-sky-200"
                      }`}>
                        {r.kind === "tool-eval" ? "tool-eval" : "throughput"}
                      </span>
                    </td>
```

Then change the "Mean t/s" cell so tool-eval rows show their score instead. Locate the cell rendering `r.meanTps` (the `text-right` cell under the "Mean t/s" header) and replace its inner expression with:

```tsx
                      {r.kind === "tool-eval"
                        ? (r.toolEvalScore != null ? `${r.toolEvalScore}/100` : "—")
                        : (r.meanTps != null ? r.meanTps.toFixed(1) : "—")}
```

> Leave the "Mean TTFR" cell as-is; it shows "—" for tool-eval rows (which have a null `meanTtfrMs`), which is correct.

- [ ] **Step 3: Build the dashboard**

Run:
```bash
npm run build --workspace packages/dashboard
```
Expected: build succeeds.

- [ ] **Step 4: Manual verification**

Open `/benchmarks` and confirm each row shows a kind badge, tool-eval rows show `NN/100` in the summary column, and throughput rows still show t/s. Record in the PR description.

- [ ] **Step 5: Commit**

```bash
git add packages/dashboard/app/benchmarks/page.tsx
git commit -m "feat(dashboard): kind badge and score column on benchmark list"
```

---

## Task 11: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire server test suite**

Run:
```bash
npm test
```
Expected: all suites pass — the new `presets`, `tool-eval-args`, `tool-eval-parser`, `orchestrator`, and `benchmarks.routes` tests plus all pre-existing tests are green.

- [ ] **Step 2: Type-check both packages**

Run:
```bash
npx tsc -p packages/server --noEmit && npm run build --workspace packages/dashboard
```
Expected: both exit 0 / build succeeds.

- [ ] **Step 3: Confirm no agent version bump was made**

Run:
```bash
git diff --name-only 5c1cbca..HEAD | grep '^packages/agent/' || echo "agent untouched — correct, no bump needed"
```
Expected: prints "agent untouched — correct, no bump needed".

- [ ] **Step 4 (optional, real end-to-end):** Against a non-critical running deployment (e.g. a small Ollama-backed model, NOT the Nemotron soak), launch the `tool-eval-quick` preset from the dashboard and confirm: logs stream live, the run completes, and the detail page shows scores + categories. The `tool-eval-quick` run can take ~25 min on a slow 8B model; a fast vLLM serve is much quicker. Document the result in the PR.

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** data model (Task 1), presets (Task 2), args (Task 3), parser (Task 4), orchestrator strategy (Task 5), route dispatch + persistence (Task 6), dashboard types/launcher/detail/list (Tasks 7-10), tests at every tier, no agent bump (Task 11/Step 3). The only spec items deliberately out of scope: `--perf`, custom tool-eval UI, diff/resume/history/leaderboard.
- **Type consistency:** `ToolEvalConfig` { short, hardmode, contextPressure, seed } and `ToolEvalSummary` / `ToolEvalCategoryInput` field names are identical across server (Tasks 2-6) and dashboard (`ToolEvalCategory`, Task 7). Prisma column names (`toolEval*`, `ToolEvalCategory.code/label/percent/earned/maxPoints/passCount/partialCount/failCount`) match the parser output keys consumed by the route's `toolEvalCategories: { create: s.categories }`.
- **Pinned ref:** `TOOL_EVAL_SPEC` default = `…tool-eval-bench.git@c3868bff099592c9a1045de2c9a3dc24abebb7fb`, overridable via `TOOL_EVAL_BENCH_REF`.
