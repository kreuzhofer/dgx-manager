# Benchmark Runner (llama-benchy) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an in-app benchmarking workflow that runs [`llama-benchy`](https://github.com/eugr/llama-benchy) in a Docker container against any running deployment's OpenAI-compatible endpoint, persists results, and presents a comparison UI so the operator can rank deployments by tokens/sec, time-to-first-response, and behavior under context depth/concurrency.

**Architecture:**
- The manager spawns `uvx llama-benchy …` as a background child process from the server container. `uvx` creates an isolated Python env on the fly — no Docker image, no per-deployment install, no GPU usage. The process hits the deployment's OpenAI endpoint over the existing management network.
- New Prisma models `BenchmarkRun` (header) + `BenchmarkResult` (per-row metrics). New REST routes under `/api/benchmarks`. Live progress/logs piped through the existing SSE channel using `benchmark:*` event types, mirroring the `deployment:*` pattern.
- The orchestrator keeps an in-memory `Map<runId, ChildProcess>` so cancel just calls `child.kill()` on the spawned tree. Server restarts orphan in-flight processes — those runs are reconciled to `failed` on boot (acceptable since they're short and re-runnable).
- Dashboard gets a **"Benchmark"** button on every running deployment row, a **`/benchmarks`** list page with filters, a per-run detail view with charts (Recharts), and a **`/benchmarks/compare`** page for side-by-side comparison.

**Tech Stack:** TypeScript, Express 5, Prisma + SQLite, Next.js 15 App Router, React 19, Tailwind 4, Recharts, Vitest + supertest + fast-check, `uv`/`uvx` (Astral) installed once in the server image.

---

## File Structure

**New files:**

| Path | Responsibility |
|------|---------------|
| `packages/server/src/benchmarks/presets.ts` | Built-in preset registry + `BenchmarkConfig` type |
| `packages/server/src/benchmarks/presets.test.ts` | Unit tests for the preset registry |
| `packages/server/src/benchmarks/args.ts` | Pure `buildBenchyArgs(config, target)` |
| `packages/server/src/benchmarks/args.test.ts` | Unit + property tests for `buildBenchyArgs` |
| `packages/server/src/benchmarks/parser.ts` | Pure `parseBenchyResults(json)` → `BenchmarkResultInput[]` |
| `packages/server/src/benchmarks/parser.test.ts` | Unit tests against fixtures |
| `packages/server/src/benchmarks/orchestrator.ts` | Side-effectful runner: spawn docker, stream stdout/stderr, persist results, broadcast SSE |
| `packages/server/src/benchmarks/endpoint.ts` | Pure `deploymentEndpointUrl(deployment)` helper |
| `packages/server/src/benchmarks/endpoint.test.ts` | Unit tests |
| `packages/server/src/routes/benchmarks.ts` | Express router: POST/GET/DELETE/cancel + presets |
| `packages/server/src/__tests__/integration/benchmarks.routes.test.ts` | supertest integration tests for the router |
| `packages/server/src/__tests__/integration/benchmarks.fixtures/result.json` | Sample llama-benchy JSON output |
| `packages/dashboard/lib/benchmarks.ts` | API client (`startBenchmark`, `listBenchmarks`, etc.) + types |
| `packages/dashboard/components/benchmark-form-modal.tsx` | Modal launched from the Deployments page |
| `packages/dashboard/components/benchmark-run-row.tsx` | Shared row component (used on list page + compare picker) |
| `packages/dashboard/components/benchmark-result-table.tsx` | Renders `BenchmarkResult[]` rows |
| `packages/dashboard/components/benchmark-chart.tsx` | Recharts wrapper for tps / ttfr visualization |
| `packages/dashboard/app/benchmarks/page.tsx` | List page with filters |
| `packages/dashboard/app/benchmarks/[id]/page.tsx` | Per-run detail page (table + charts + raw output) |
| `packages/dashboard/app/benchmarks/compare/page.tsx` | Side-by-side comparison |

**Modified files:**

| Path | Change |
|------|--------|
| `prisma/schema.prisma` | Add `BenchmarkRun`, `BenchmarkResult`; add `benchmarkRuns` relation to `Deployment` |
| `Dockerfile.server` | Install `uv` (which provides `uvx`) in the server image |
| `packages/server/src/index.ts` | Mount `benchmarksRouter` at `/api/benchmarks`; reconcile stale `running` rows to `failed` on boot |
| `packages/dashboard/app/deployments/page.tsx` | Add "Benchmark" button on running deployments + SSE handlers for `benchmark:*` |
| `packages/dashboard/components/top-nav.tsx` | Add "Benchmarks" nav link |
| `packages/dashboard/package.json` | Add `recharts` dependency |
| `README.md` | One short paragraph in the Features section pointing at `/benchmarks` |

---

## Task 1: Prisma models for benchmark runs and results

**Files:**
- Modify: `prisma/schema.prisma` (after the `Deployment` model, before `ClusterNode`)

**Why this task first:** Schema is the contract every later task references. We push it before writing any code that imports it.

- [ ] **Step 1: Add models to `prisma/schema.prisma`**

Append the two models at the bottom of the file (after `TrainingMetric`):

```prisma
model BenchmarkRun {
  id              String   @id @default(cuid())
  // Nullable + onDelete:SetNull so deleting a deployment doesn't erase its
  // benchmark history — runs are a permanent record we want to keep for
  // before/after comparisons.
  deploymentId    String?
  presetId        String?  // null when the run used a fully custom config
  // Snapshot of the target at run-time. We keep these locally so the row
  // remains interpretable after the deployment is gone.
  modelName       String
  endpointUrl     String
  servedModelName String
  // JSON-encoded BenchmarkConfig (pp, tg, depth, runs, concurrency, etc.)
  config          String
  status          String   @default("pending") // pending | running | completed | failed | canceled
  error           String?
  startedAt       DateTime?
  completedAt     DateTime?
  // Raw markdown table emitted by llama-benchy (--format markdown). Rendered
  // verbatim on the detail page as a fallback if structured parsing fails.
  rawOutput       String?
  // Aggregate stats surfaced in list/compare views so we don't recompute on
  // every page render.
  meanTps         Float?
  meanTtfrMs      Float?
  createdAt       DateTime @default(now())
  deployment      Deployment? @relation(fields: [deploymentId], references: [id], onDelete: SetNull)
  results         BenchmarkResult[]

  @@index([deploymentId, createdAt])
  @@index([status])
}

model BenchmarkResult {
  id          String   @id @default(cuid())
  runId       String
  // What kind of measurement this row represents: "pp" (prompt processing /
  // prefill) or "tg" (token generation / decode). Lets us bucket charts.
  opType      String
  pp          Int
  tg          Int
  depth       Int
  concurrency Int      @default(1)
  tps         Float
  peakTps     Float?
  ttfrMs      Float?
  estPptMs    Float?
  e2eTtftMs   Float?
  // Standard deviations from multi-run averaging — null when --runs=1.
  tpsStdev    Float?
  ttfrStdev   Float?
  run         BenchmarkRun @relation(fields: [runId], references: [id], onDelete: Cascade)

  @@index([runId])
}
```

Then add the reverse relation on the existing `Deployment` model. Find the `Deployment` model block (around line 68) and add `benchmarkRuns BenchmarkRun[]` next to the other relations:

```prisma
model Deployment {
  // ... existing fields unchanged ...
  lbEndpoints   LoadBalancerEndpoint[]
  clusterNodes  ClusterNode[]
  benchmarkRuns BenchmarkRun[]
}
```

- [ ] **Step 2: Apply the schema to the local SQLite**

Run:
```bash
PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION="I understand this is destructive and I have backups" npm run db:push
npm run db:generate
```
Expected: `db push` reports `Your database is now in sync with your Prisma schema.` followed by `prisma generate` regenerating the client.

- [ ] **Step 3: Commit**

```bash
git add prisma/schema.prisma
git commit -m "benchmarks: add BenchmarkRun + BenchmarkResult Prisma models"
```

---

## Task 2: Install `uv`/`uvx` in the server image

**Files:**
- Modify: `Dockerfile.server`

**Why:** The orchestrator (Task 7) shells out to `uvx llama-benchy …`. `uvx` ships as part of [`uv`](https://docs.astral.sh/uv/) (a single static binary) and creates an isolated Python env on first invocation, so we don't need Python or any package pre-installed. Caches land under `~/.cache/uv` inside the container — fine because the container's filesystem is persistent for the lifetime of the deployment, and the first run takes ~10s.

> **Note for local dev (`npm run dev`):** ensure `uvx` is on your `$PATH`. Install once with `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`).

- [ ] **Step 1: Add `uv` to the server image**

Open `Dockerfile.server`. Find the runtime stage's apt section (or add one near the top of the runtime stage after `FROM`). Add:

```dockerfile
# uv provides `uvx`, which we use to run `llama-benchy` on demand for the
# /api/benchmarks workflow. Pinning the uv version keeps benchmark runs
# reproducible across rebuilds.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -LsSf https://astral.sh/uv/0.5.0/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/uv \
 && mv /root/.local/bin/uvx /usr/local/bin/uvx \
 && rm -rf /var/lib/apt/lists/*
```

If the file already has an apt-install block, fold `curl` + `ca-certificates` into it and keep the rest of the `RUN` separate so a future package addition doesn't re-download `uv`.

- [ ] **Step 2: Pre-warm the llama-benchy env at build time (optional but recommended)**

Right after the `uv` install in the same Dockerfile, add:

```dockerfile
# Warm uvx's cache for llama-benchy so the first benchmark in a fresh
# container doesn't pay a 10–30s cold-start penalty.
ARG LLAMA_BENCHY_VERSION=0.5.0
RUN uvx --from "llama-benchy==${LLAMA_BENCHY_VERSION}" llama-benchy --help > /dev/null
```

> **Note on version pinning:** If `llama-benchy==0.5.0` is unavailable when this plan is executed, replace with the latest published version on PyPI and adjust the parser fixtures in Task 6 accordingly. The orchestrator in Task 7 uses the same `LLAMA_BENCHY_VERSION` value through an env var.

- [ ] **Step 3: Rebuild the server image and sanity-check uvx**

Run:
```bash
./scripts/build-agent-bundles.sh && \
  MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build server
docker compose exec server uvx --from llama-benchy llama-benchy --help | head -n 20
```
Expected: help output listing `--base-url`, `--model`, `--pp`, `--tg`, `--depth`, `--runs`, `--format`.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile.server
git commit -m "benchmarks: install uv/uvx in server image + warm llama-benchy cache"
```

---

## Task 3: Pure helper — `deploymentEndpointUrl`

**Files:**
- Create: `packages/server/src/benchmarks/endpoint.ts`
- Create: `packages/server/src/benchmarks/endpoint.test.ts`

**Why:** The benchmark target URL is derived from `Deployment.node.ipAddress` + `Deployment.port`. Several places will need this — extract it once, test it as a pure function.

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/benchmarks/endpoint.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { deploymentEndpointUrl } from "./endpoint.js";

describe("deploymentEndpointUrl", () => {
  it("returns http://<ip>:<port> for a running deployment", () => {
    expect(
      deploymentEndpointUrl({
        port: 8000,
        node: { ipAddress: "192.168.1.10" },
      }),
    ).toBe("http://192.168.1.10:8000");
  });

  it("throws when port is missing", () => {
    expect(() =>
      deploymentEndpointUrl({
        port: null,
        node: { ipAddress: "192.168.1.10" },
      }),
    ).toThrow(/port/);
  });

  it("throws when node ipAddress is missing", () => {
    expect(() =>
      deploymentEndpointUrl({
        port: 8000,
        node: { ipAddress: null },
      }),
    ).toThrow(/ipAddress/);
  });

  it("throws when node is null", () => {
    expect(() =>
      deploymentEndpointUrl({ port: 8000, node: null }),
    ).toThrow(/node/);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run packages/server/src/benchmarks/endpoint.test.ts`
Expected: FAIL with "Cannot find module './endpoint.js'".

- [ ] **Step 3: Implement the helper**

Create `packages/server/src/benchmarks/endpoint.ts`:

```ts
export type EndpointDeployment = {
  port: number | null;
  node: { ipAddress: string | null } | null;
};

export function deploymentEndpointUrl(d: EndpointDeployment): string {
  if (!d.node) throw new Error("deployment.node is required");
  if (!d.node.ipAddress) throw new Error("deployment.node.ipAddress is required");
  if (!d.port) throw new Error("deployment.port is required");
  return `http://${d.node.ipAddress}:${d.port}`;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/benchmarks/endpoint.test.ts`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/benchmarks/endpoint.ts packages/server/src/benchmarks/endpoint.test.ts
git commit -m "benchmarks: add deploymentEndpointUrl helper"
```

---

## Task 4: Built-in preset registry + `BenchmarkConfig` type

**Files:**
- Create: `packages/server/src/benchmarks/presets.ts`
- Create: `packages/server/src/benchmarks/presets.test.ts`

**Why:** Most users will pick a preset rather than fiddle with raw `--pp`/`--tg`/`--depth` lists. Defining presets server-side keeps the dashboard dumb and makes the comparison view meaningful (runs from the same preset are directly comparable).

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/benchmarks/presets.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { BENCHMARK_PRESETS, getPreset, listPresets } from "./presets.js";

describe("BENCHMARK_PRESETS", () => {
  it("exposes the five expected presets by id", () => {
    expect(listPresets().map((p) => p.id).sort()).toEqual([
      "chat-long",
      "chat-short",
      "code-32k",
      "quick-smoke",
      "throughput",
    ]);
  });

  it("every preset has at least one pp value, one tg value, and runs>=1", () => {
    for (const p of listPresets()) {
      expect(p.config.pp.length).toBeGreaterThan(0);
      expect(p.config.tg.length).toBeGreaterThan(0);
      expect(p.config.runs).toBeGreaterThanOrEqual(1);
    }
  });

  it("getPreset returns the preset by id", () => {
    const p = getPreset("quick-smoke");
    expect(p?.id).toBe("quick-smoke");
  });

  it("getPreset returns undefined for unknown ids", () => {
    expect(getPreset("does-not-exist")).toBeUndefined();
  });

  it("quick-smoke is small enough to finish in under a minute on a single GPU", () => {
    const p = getPreset("quick-smoke")!;
    // Heuristic: a single short prompt × generation × 1 run.
    expect(p.config.pp).toEqual([128]);
    expect(p.config.tg).toEqual([32]);
    expect(p.config.runs).toBe(1);
    expect(p.config.concurrency).toEqual([1]);
  });
});

// The presets themselves are kept narrow; if you add or rename one, update
// this list-level test too — that's intentional, presets are part of the
// product surface.
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run packages/server/src/benchmarks/presets.test.ts`
Expected: FAIL with "Cannot find module './presets.js'".

- [ ] **Step 3: Implement the presets registry**

Create `packages/server/src/benchmarks/presets.ts`:

```ts
export type LatencyMode = "api" | "generation" | "none";

export type BenchmarkConfig = {
  pp: number[];
  tg: number[];
  depth: number[];
  runs: number;
  concurrency: number[];
  latencyMode: LatencyMode;
  enablePrefixCaching: boolean;
  skipCoherence: boolean;
};

export type BenchmarkPreset = {
  id: string;
  label: string;
  description: string;
  config: BenchmarkConfig;
};

export const BENCHMARK_PRESETS: BenchmarkPreset[] = [
  {
    id: "quick-smoke",
    label: "Quick smoke",
    description: "30-second sanity check: one short prompt, one generation.",
    config: {
      pp: [128],
      tg: [32],
      depth: [0],
      runs: 1,
      concurrency: [1],
      latencyMode: "api",
      enablePrefixCaching: false,
      skipCoherence: false,
    },
  },
  {
    id: "chat-short",
    label: "Chat (short)",
    description: "Typical chatbot turn: 512-token prompt, 128 generated.",
    config: {
      pp: [512],
      tg: [128],
      depth: [0],
      runs: 3,
      concurrency: [1, 4],
      latencyMode: "api",
      enablePrefixCaching: false,
      skipCoherence: false,
    },
  },
  {
    id: "chat-long",
    label: "Chat (long context)",
    description: "Long conversation: 2k prompt, 128 generated, swept across 0/4k context.",
    config: {
      pp: [2048],
      tg: [128],
      depth: [0, 4096],
      runs: 3,
      concurrency: [1, 4],
      latencyMode: "api",
      enablePrefixCaching: false,
      skipCoherence: false,
    },
  },
  {
    id: "code-32k",
    label: "Code (32k context)",
    description: "Repo-scale codegen: 8k prompt, 512 generated, swept up to 32k context.",
    config: {
      pp: [8192],
      tg: [512],
      depth: [0, 16384, 32000],
      runs: 2,
      concurrency: [1],
      latencyMode: "api",
      enablePrefixCaching: false,
      skipCoherence: true,
    },
  },
  {
    id: "throughput",
    label: "Throughput sweep",
    description: "Concurrency ramp: same prompt at 1/4/16/32/64 in-flight requests.",
    config: {
      pp: [512],
      tg: [128],
      depth: [0],
      runs: 2,
      concurrency: [1, 4, 16, 32, 64],
      latencyMode: "none",
      enablePrefixCaching: false,
      skipCoherence: false,
    },
  },
];

export function listPresets(): BenchmarkPreset[] {
  return BENCHMARK_PRESETS;
}

export function getPreset(id: string): BenchmarkPreset | undefined {
  return BENCHMARK_PRESETS.find((p) => p.id === id);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/benchmarks/presets.test.ts`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/benchmarks/presets.ts packages/server/src/benchmarks/presets.test.ts
git commit -m "benchmarks: add built-in preset registry"
```

---

## Task 5: Pure helper — `buildBenchyArgs`

**Files:**
- Create: `packages/server/src/benchmarks/args.ts`
- Create: `packages/server/src/benchmarks/args.test.ts`

**Why:** Translating a `BenchmarkConfig` + target into the exact argv handed to `docker run` is the most error-prone step. Pure function ⇒ exhaustively testable, no spawned processes needed.

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/benchmarks/args.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { buildBenchyArgs } from "./args.js";
import type { BenchmarkConfig } from "./presets.js";

const baseConfig: BenchmarkConfig = {
  pp: [128, 512],
  tg: [32, 128],
  depth: [0, 4096],
  runs: 3,
  concurrency: [1, 4],
  latencyMode: "api",
  enablePrefixCaching: false,
  skipCoherence: false,
};

describe("buildBenchyArgs", () => {
  it("emits --base-url, --model and the JSON output path", () => {
    const args = buildBenchyArgs(baseConfig, {
      baseUrl: "http://10.0.0.1:8000",
      modelName: "llama-3.1-8b",
      outputPath: "/output/result.json",
    });
    expect(args).toContain("--base-url");
    expect(args).toContain("http://10.0.0.1:8000");
    expect(args).toContain("--model");
    expect(args).toContain("llama-3.1-8b");
    expect(args).toContain("--save-result");
    expect(args).toContain("/output/result.json");
    expect(args).toContain("--format");
    expect(args).toContain("json");
  });

  it("joins pp/tg/depth/concurrency with commas (llama-benchy CSV-style list flag)", () => {
    const args = buildBenchyArgs(baseConfig, {
      baseUrl: "http://10.0.0.1:8000",
      modelName: "m",
      outputPath: "/output/r.json",
    });
    const idx = (flag: string) => args.indexOf(flag);
    expect(args[idx("--pp") + 1]).toBe("128,512");
    expect(args[idx("--tg") + 1]).toBe("32,128");
    expect(args[idx("--depth") + 1]).toBe("0,4096");
    expect(args[idx("--concurrency") + 1]).toBe("1,4");
    expect(args[idx("--runs") + 1]).toBe("3");
  });

  it("includes --enable-prefix-caching only when enabled", () => {
    const off = buildBenchyArgs(
      { ...baseConfig, enablePrefixCaching: false },
      { baseUrl: "u", modelName: "m", outputPath: "/o" },
    );
    const on = buildBenchyArgs(
      { ...baseConfig, enablePrefixCaching: true },
      { baseUrl: "u", modelName: "m", outputPath: "/o" },
    );
    expect(off).not.toContain("--enable-prefix-caching");
    expect(on).toContain("--enable-prefix-caching");
  });

  it("includes --skip-coherence only when enabled", () => {
    const on = buildBenchyArgs(
      { ...baseConfig, skipCoherence: true },
      { baseUrl: "u", modelName: "m", outputPath: "/o" },
    );
    expect(on).toContain("--skip-coherence");
  });

  it("passes latency-mode through", () => {
    const args = buildBenchyArgs(
      { ...baseConfig, latencyMode: "generation" },
      { baseUrl: "u", modelName: "m", outputPath: "/o" },
    );
    const idx = args.indexOf("--latency-mode");
    expect(args[idx + 1]).toBe("generation");
  });

  // Invariant: every numeric list flag is emitted as one --flag followed by
  // one comma-joined value (never as repeated --flag tokens, never empty).
  test.prop([
    fc.array(fc.integer({ min: 1, max: 100000 }), { minLength: 1, maxLength: 5 }),
    fc.array(fc.integer({ min: 1, max: 100000 }), { minLength: 1, maxLength: 5 }),
    fc.array(fc.integer({ min: 0, max: 200000 }), { minLength: 1, maxLength: 5 }),
    fc.array(fc.integer({ min: 1, max: 256 }), { minLength: 1, maxLength: 5 }),
  ])("emits exactly one value token after each list flag", (pp, tg, depth, concurrency) => {
    const args = buildBenchyArgs(
      { ...baseConfig, pp, tg, depth, concurrency },
      { baseUrl: "u", modelName: "m", outputPath: "/o" },
    );
    for (const flag of ["--pp", "--tg", "--depth", "--concurrency"]) {
      const occurrences = args.filter((a) => a === flag);
      expect(occurrences.length).toBe(1);
      const idx = args.indexOf(flag);
      const value = args[idx + 1];
      expect(value).toMatch(/^[0-9]+(,[0-9]+)*$/);
    }
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run packages/server/src/benchmarks/args.test.ts`
Expected: FAIL with "Cannot find module './args.js'".

- [ ] **Step 3: Implement `buildBenchyArgs`**

Create `packages/server/src/benchmarks/args.ts`:

```ts
import type { BenchmarkConfig } from "./presets.js";

export type BenchyTarget = {
  baseUrl: string;
  modelName: string;
  outputPath: string;
};

export function buildBenchyArgs(
  config: BenchmarkConfig,
  target: BenchyTarget,
): string[] {
  const args: string[] = [
    "--base-url", target.baseUrl,
    "--model", target.modelName,
    "--format", "json",
    "--save-result", target.outputPath,
    "--pp", config.pp.join(","),
    "--tg", config.tg.join(","),
    "--depth", config.depth.join(","),
    "--concurrency", config.concurrency.join(","),
    "--runs", String(config.runs),
    "--latency-mode", config.latencyMode,
  ];
  if (config.enablePrefixCaching) args.push("--enable-prefix-caching");
  if (config.skipCoherence) args.push("--skip-coherence");
  return args;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/benchmarks/args.test.ts`
Expected: PASS, 6 tests (5 it + 1 property).

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/benchmarks/args.ts packages/server/src/benchmarks/args.test.ts
git commit -m "benchmarks: pure buildBenchyArgs helper with property tests"
```

---

## Task 6: Pure helper — `parseBenchyResults`

**Files:**
- Create: `packages/server/src/__tests__/integration/benchmarks.fixtures/result.json`
- Create: `packages/server/src/benchmarks/parser.ts`
- Create: `packages/server/src/benchmarks/parser.test.ts`

**Why:** llama-benchy writes a JSON file at the end of a run. We parse it once, end-to-end, into a list of `BenchmarkResult` rows ready to insert. Pure function so the orchestrator stays thin.

> **Tactic for the engineer:** If the real llama-benchy output schema differs from the fixture below, regenerate the fixture by running the image once against any deployment and copying `result.json` into the fixtures dir. The parser code is small; adjust field names and re-run the tests.

- [ ] **Step 1: Write the fixture**

Create `packages/server/src/__tests__/integration/benchmarks.fixtures/result.json`. (Path is shared between the parser test and the route test in Task 8.)

```json
{
  "meta": {
    "base_url": "http://10.0.0.1:8000",
    "model": "llama-3.1-8b",
    "runs": 3
  },
  "rows": [
    {
      "op": "pp",
      "pp": 512,
      "tg": 32,
      "depth": 0,
      "concurrency": 1,
      "t/s": 1840.4,
      "peak t/s": 1955.0,
      "ttfr (ms)": 142.3,
      "est_ppt (ms)": 278.0,
      "e2e_ttft (ms)": 420.1,
      "t/s_stdev": 18.2,
      "ttfr_stdev": 5.1
    },
    {
      "op": "tg",
      "pp": 512,
      "tg": 128,
      "depth": 0,
      "concurrency": 1,
      "t/s": 84.5,
      "peak t/s": 92.1,
      "ttfr (ms)": 142.3,
      "est_ppt (ms)": 278.0,
      "e2e_ttft (ms)": 420.1,
      "t/s_stdev": 0.9,
      "ttfr_stdev": 5.1
    },
    {
      "op": "tg",
      "pp": 512,
      "tg": 128,
      "depth": 4096,
      "concurrency": 4,
      "t/s": 220.3,
      "peak t/s": 240.0,
      "ttfr (ms)": 410.0,
      "est_ppt (ms)": 1100.0,
      "e2e_ttft (ms)": 1500.0,
      "t/s_stdev": 4.0,
      "ttfr_stdev": 15.0
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

Create `packages/server/src/benchmarks/parser.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseBenchyResults, summarizeResults } from "./parser.js";

const fixture = readFileSync(
  join(__dirname, "../__tests__/integration/benchmarks.fixtures/result.json"),
  "utf-8",
);

describe("parseBenchyResults", () => {
  it("parses the three rows from the fixture", () => {
    const rows = parseBenchyResults(fixture);
    expect(rows).toHaveLength(3);
  });

  it("maps llama-benchy field names to BenchmarkResult fields", () => {
    const rows = parseBenchyResults(fixture);
    expect(rows[0]).toEqual({
      opType: "pp",
      pp: 512,
      tg: 32,
      depth: 0,
      concurrency: 1,
      tps: 1840.4,
      peakTps: 1955.0,
      ttfrMs: 142.3,
      estPptMs: 278.0,
      e2eTtftMs: 420.1,
      tpsStdev: 18.2,
      ttfrStdev: 5.1,
    });
  });

  it("returns an empty array for empty input", () => {
    expect(parseBenchyResults('{"rows":[]}')).toEqual([]);
  });

  it("throws a descriptive error on malformed JSON", () => {
    expect(() => parseBenchyResults("not json")).toThrow(/parse/i);
  });

  it("throws when required fields are missing on a row", () => {
    const bad = JSON.stringify({ rows: [{ op: "pp", pp: 1 }] });
    expect(() => parseBenchyResults(bad)).toThrow(/missing/i);
  });
});

describe("summarizeResults", () => {
  it("computes mean tps and mean ttfr across all rows", () => {
    const rows = parseBenchyResults(fixture);
    const summary = summarizeResults(rows);
    // (1840.4 + 84.5 + 220.3) / 3 = 715.07
    expect(summary.meanTps).toBeCloseTo(715.07, 1);
    // (142.3 + 142.3 + 410.0) / 3 = 231.53
    expect(summary.meanTtfrMs).toBeCloseTo(231.53, 1);
  });

  it("returns nulls when given no rows", () => {
    expect(summarizeResults([])).toEqual({ meanTps: null, meanTtfrMs: null });
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `npx vitest run packages/server/src/benchmarks/parser.test.ts`
Expected: FAIL with "Cannot find module './parser.js'".

- [ ] **Step 4: Implement the parser**

Create `packages/server/src/benchmarks/parser.ts`:

```ts
export type BenchmarkResultInput = {
  opType: string;
  pp: number;
  tg: number;
  depth: number;
  concurrency: number;
  tps: number;
  peakTps: number | null;
  ttfrMs: number | null;
  estPptMs: number | null;
  e2eTtftMs: number | null;
  tpsStdev: number | null;
  ttfrStdev: number | null;
};

type RawRow = Record<string, unknown>;

function num(row: RawRow, key: string): number {
  const v = row[key];
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new Error(`benchmark row missing required numeric field: ${key}`);
  }
  return v;
}

function optNum(row: RawRow, key: string): number | null {
  const v = row[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function parseBenchyResults(jsonText: string): BenchmarkResultInput[] {
  let parsed: { rows?: RawRow[] };
  try {
    parsed = JSON.parse(jsonText);
  } catch (e) {
    throw new Error(`failed to parse llama-benchy JSON: ${(e as Error).message}`);
  }
  const rows = parsed.rows ?? [];
  return rows.map((r) => ({
    opType: String(r["op"] ?? "tg"),
    pp: num(r, "pp"),
    tg: num(r, "tg"),
    depth: num(r, "depth"),
    concurrency: num(r, "concurrency"),
    tps: num(r, "t/s"),
    peakTps: optNum(r, "peak t/s"),
    ttfrMs: optNum(r, "ttfr (ms)"),
    estPptMs: optNum(r, "est_ppt (ms)"),
    e2eTtftMs: optNum(r, "e2e_ttft (ms)"),
    tpsStdev: optNum(r, "t/s_stdev"),
    ttfrStdev: optNum(r, "ttfr_stdev"),
  }));
}

export function summarizeResults(rows: BenchmarkResultInput[]): {
  meanTps: number | null;
  meanTtfrMs: number | null;
} {
  if (rows.length === 0) return { meanTps: null, meanTtfrMs: null };
  const meanTps =
    rows.reduce((acc, r) => acc + r.tps, 0) / rows.length;
  const ttfrRows = rows.filter((r) => r.ttfrMs !== null) as Array<
    BenchmarkResultInput & { ttfrMs: number }
  >;
  const meanTtfrMs =
    ttfrRows.length === 0
      ? null
      : ttfrRows.reduce((acc, r) => acc + r.ttfrMs, 0) / ttfrRows.length;
  return { meanTps, meanTtfrMs };
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/benchmarks/parser.test.ts`
Expected: PASS, 7 tests.

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/__tests__/integration/benchmarks.fixtures \
        packages/server/src/benchmarks/parser.ts \
        packages/server/src/benchmarks/parser.test.ts
git commit -m "benchmarks: parseBenchyResults + summarizeResults with fixture"
```

---

## Task 7: Orchestrator — `runBenchmark` (spawns `uvx`, persists results)

**Files:**
- Create: `packages/server/src/benchmarks/orchestrator.ts`
- Create: `packages/server/src/benchmarks/orchestrator.test.ts`

**Why:** This is the only side-effectful module — it owns the spawned child, the SSE broadcasts via `onLog`, and the DB writes. The runner is just `uvx llama-benchy …` spawned as a background child process; nothing else needed. We keep an in-memory `Map<runId, ChildProcess>` so `cancelBenchmark(runId)` can `child.kill()` the live process.

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/benchmarks/orchestrator.test.ts`:

```ts
import { describe, expect, it, vi, beforeEach } from "vitest";
import { EventEmitter } from "node:events";

// Mock node:child_process so spawn hits our fake.
const spawnMock = vi.fn();
vi.mock("node:child_process", () => ({
  spawn: (...args: unknown[]) => spawnMock(...args),
}));

// Mock fs so we don't touch the shared storage path during unit tests.
const readFileSyncMock = vi.fn();
const mkdirSyncMock = vi.fn();
const existsSyncMock = vi.fn();
vi.mock("node:fs", async () => {
  const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
  return {
    ...actual,
    readFileSync: (...a: unknown[]) => readFileSyncMock(...a),
    mkdirSync: (...a: unknown[]) => mkdirSyncMock(...a),
    existsSync: (...a: unknown[]) => existsSyncMock(...a),
  };
});

import { runBenchmark, cancelBenchmark } from "./orchestrator.js";

function makeFakeChild(pid = 4242) {
  const child = new EventEmitter() as EventEmitter & {
    stdout: EventEmitter;
    stderr: EventEmitter;
    kill: ReturnType<typeof vi.fn>;
    pid: number;
  };
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.kill = vi.fn();
  child.pid = pid;
  return child;
}

describe("runBenchmark", () => {
  beforeEach(() => {
    spawnMock.mockReset();
    readFileSyncMock.mockReset();
    mkdirSyncMock.mockReset();
    existsSyncMock.mockReset();
  });

  it("spawns `uvx llama-benchy` with the supplied argv and mkdirs the output dir", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(true);
    readFileSyncMock.mockReturnValue('{"rows":[]}');

    const onLog = vi.fn();
    const promise = runBenchmark({
      runId: "run_abc",
      args: ["--base-url", "http://10.0.0.1:8000", "--model", "m", "--save-result", "/mnt/tank/benchmarks/run_abc/result.json"],
      outputDir: "/mnt/tank/benchmarks/run_abc",
      onLog,
    });

    child.stdout.emit("data", Buffer.from("running test 1/3...\n"));
    child.emit("close", 0);

    const result = await promise;
    expect(spawnMock).toHaveBeenCalledTimes(1);
    const [cmd, argv, spawnOpts] = spawnMock.mock.calls[0];
    expect(cmd).toBe("uvx");
    expect(argv[0]).toBe("--from");
    expect(argv[1]).toMatch(/^llama-benchy(==.+)?$/);
    expect(argv[2]).toBe("llama-benchy");
    expect(argv.slice(3)).toEqual([
      "--base-url", "http://10.0.0.1:8000",
      "--model", "m",
      "--save-result", "/mnt/tank/benchmarks/run_abc/result.json",
    ]);
    expect((spawnOpts as { detached: boolean }).detached).toBe(true);
    expect(result.exitCode).toBe(0);
    expect(onLog).toHaveBeenCalledWith("running test 1/3...");
    expect(mkdirSyncMock).toHaveBeenCalledWith(
      "/mnt/tank/benchmarks/run_abc",
      expect.objectContaining({ recursive: true }),
    );
  });

  it("returns parsed results when result.json exists after exit", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(true);
    readFileSyncMock.mockReturnValue(
      JSON.stringify({
        rows: [{
          op: "tg", pp: 1, tg: 2, depth: 0, concurrency: 1,
          "t/s": 50.5, "peak t/s": 60, "ttfr (ms)": 100,
          "est_ppt (ms)": 50, "e2e_ttft (ms)": 150,
          "t/s_stdev": 1, "ttfr_stdev": 2,
        }],
      }),
    );
    const promise = runBenchmark({
      runId: "r2", args: [], outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 0);
    const r = await promise;
    expect(r.results).toHaveLength(1);
    expect(r.results[0].tps).toBe(50.5);
  });

  it("returns exitCode and no results when the child exits non-zero", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(false);
    const promise = runBenchmark({
      runId: "r3", args: [], outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 137);
    const r = await promise;
    expect(r.exitCode).toBe(137);
    expect(r.results).toEqual([]);
    expect(readFileSyncMock).not.toHaveBeenCalled();
  });

  it("cancelBenchmark kills the process group of an in-flight run", async () => {
    const child = makeFakeChild(9999);
    spawnMock.mockReturnValue(child);
    const promise = runBenchmark({
      runId: "cancelme", args: [], outputDir: "/o", onLog: vi.fn(),
    });
    expect(cancelBenchmark("cancelme")).toBe(true);
    expect(child.kill).toHaveBeenCalledWith("SIGTERM");
    child.emit("close", 143);
    await promise;
  });

  it("cancelBenchmark returns false when the run is not active", () => {
    expect(cancelBenchmark("ghost")).toBe(false);
  });

  it("removes the run from the active map after the child exits", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(false);
    const promise = runBenchmark({
      runId: "cleanup", args: [], outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 0);
    await promise;
    expect(cancelBenchmark("cleanup")).toBe(false);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run packages/server/src/benchmarks/orchestrator.test.ts`
Expected: FAIL with "Cannot find module './orchestrator.js'".

- [ ] **Step 3: Implement the orchestrator**

Create `packages/server/src/benchmarks/orchestrator.ts`:

```ts
import { spawn, type ChildProcess } from "node:child_process";
import { readFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import {
  parseBenchyResults,
  summarizeResults,
  type BenchmarkResultInput,
} from "./parser.js";

const LLAMA_BENCHY_SPEC =
  process.env.LLAMA_BENCHY_VERSION
    ? `llama-benchy==${process.env.LLAMA_BENCHY_VERSION}`
    : "llama-benchy";

// In-memory registry of in-flight runs. Lost on restart — see Task 9 for the
// boot-time reconciliation that marks any orphaned "running" rows as failed.
const ACTIVE: Map<string, ChildProcess> = new Map();

export type RunBenchmarkOpts = {
  runId: string;
  args: string[];        // llama-benchy argv (from buildBenchyArgs)
  outputDir: string;     // host path; must contain the --save-result path the args point at
  onLog: (line: string) => void;
};

export type RunBenchmarkResult = {
  exitCode: number | null;
  results: BenchmarkResultInput[];
  summary: { meanTps: number | null; meanTtfrMs: number | null };
  rawOutput: string | null;
};

export function runBenchmark(opts: RunBenchmarkOpts): Promise<RunBenchmarkResult> {
  mkdirSync(opts.outputDir, { recursive: true, mode: 0o777 });

  const argv = ["--from", LLAMA_BENCHY_SPEC, "llama-benchy", ...opts.args];

  return new Promise((resolve) => {
    const child = spawn("uvx", argv, {
      stdio: ["ignore", "pipe", "pipe"],
      // detached:true so we can kill the whole process group (uvx may
      // spawn a python subprocess) via process.kill(-pid).
      detached: true,
    });
    ACTIVE.set(opts.runId, child);

    child.stdout?.on("data", (b: Buffer) => {
      for (const line of b.toString().split("\n")) {
        if (line) opts.onLog(line);
      }
    });
    child.stderr?.on("data", (b: Buffer) => {
      for (const line of b.toString().split("\n")) {
        if (line) opts.onLog(line);
      }
    });

    child.on("close", (code) => {
      ACTIVE.delete(opts.runId);
      const resultPath = join(opts.outputDir, "result.json");
      let results: BenchmarkResultInput[] = [];
      let rawOutput: string | null = null;
      if (code === 0 && existsSync(resultPath)) {
        try {
          rawOutput = readFileSync(resultPath, "utf-8");
          results = parseBenchyResults(rawOutput);
        } catch (e) {
          opts.onLog(`[parser] ${(e as Error).message}`);
        }
      }
      resolve({
        exitCode: code,
        results,
        summary: summarizeResults(results),
        rawOutput,
      });
    });
  });
}

export function cancelBenchmark(runId: string): boolean {
  const child = ACTIVE.get(runId);
  if (!child) return false;
  // SIGTERM gives llama-benchy a chance to flush partial results.
  child.kill("SIGTERM");
  return true;
}

export function isRunActive(runId: string): boolean {
  return ACTIVE.has(runId);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/benchmarks/orchestrator.test.ts`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/benchmarks/orchestrator.ts \
        packages/server/src/benchmarks/orchestrator.test.ts
git commit -m "benchmarks: orchestrator spawns uvx llama-benchy with cancel hook"
```

---

## Task 8: REST routes — `POST/GET/DELETE /api/benchmarks` + cancel + presets

**Files:**
- Create: `packages/server/src/routes/benchmarks.ts`
- Create: `packages/server/src/__tests__/integration/benchmarks.routes.test.ts`

**Why:** This is the surface the dashboard talks to. We integration-test it the same way the deployments route is tested (per-suite SQLite, stub the orchestrator).

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/__tests__/integration/benchmarks.routes.test.ts`:

```ts
import {
  describe, expect, it, beforeAll, afterAll, beforeEach, vi,
} from "vitest";
import { execSync } from "node:child_process";
import { mkdtempSync, rmSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-benchmark-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;
process.env.SHARED_STORAGE_PATH = TMP_DIR;

// Mock the orchestrator so the route test never spawns uvx.
const runMock = vi.fn();
const cancelMock = vi.fn();
vi.mock("../../benchmarks/orchestrator.js", () => ({
  runBenchmark: (...a: unknown[]) => runMock(...a),
  cancelBenchmark: (...a: unknown[]) => cancelMock(...a),
}));

let prisma: typeof import("../../prisma.js").prisma;
let benchmarksRouter: typeof import("../../routes/benchmarks.js").benchmarksRouter;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      DATABASE_URL: `file:${DB_PATH}`,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "I understand this is destructive and I have backups",
    },
    stdio: "pipe",
  });
  ({ prisma } = await import("../../prisma.js"));
  ({ benchmarksRouter } = await import("../../routes/benchmarks.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  runMock.mockReset();
  cancelMock.mockReset();
  // FK-ordered wipe
  await prisma.benchmarkResult.deleteMany();
  await prisma.benchmarkRun.deleteMany();
  await prisma.deployment.deleteMany();
  await prisma.model.deleteMany();
  await prisma.node.deleteMany();
});

function makeApp() {
  const app = express();
  app.use(express.json());
  app.use("/api/benchmarks", benchmarksRouter);
  return app;
}

async function seedRunningDeployment() {
  const node = await prisma.node.create({
    data: { name: "n1", ipAddress: "10.0.0.1", status: "online" },
  });
  const model = await prisma.model.create({
    data: { name: "llama-3.1-8b", runtime: "vllm" },
  });
  return prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: "running",
      port: 8000,
      displayName: "llama-prod",
    },
    include: { node: true, model: true },
  });
}

describe("GET /api/benchmarks/presets", () => {
  it("returns the built-in presets", async () => {
    const res = await request(makeApp()).get("/api/benchmarks/presets");
    expect(res.status).toBe(200);
    expect(res.body.map((p: { id: string }) => p.id)).toContain("quick-smoke");
  });
});

describe("POST /api/benchmarks", () => {
  it("creates a run, spawns the orchestrator, and returns the run id", async () => {
    const d = await seedRunningDeployment();
    runMock.mockResolvedValue({
      exitCode: 0,
      results: [{
        opType: "tg", pp: 128, tg: 32, depth: 0, concurrency: 1,
        tps: 50, peakTps: 60, ttfrMs: 100,
        estPptMs: 50, e2eTtftMs: 150, tpsStdev: 1, ttfrStdev: 2,
      }],
      summary: { meanTps: 50, meanTtfrMs: 100 },
      rawOutput: "{}",
    });

    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "quick-smoke" });

    expect(res.status).toBe(201);
    expect(res.body.id).toBeDefined();
    expect(res.body.status).toBe("pending");

    // Orchestrator was called with the right shape
    expect(runMock).toHaveBeenCalledTimes(1);
    const call = runMock.mock.calls[0][0];
    expect(call.runId).toBe(res.body.id);
    expect(call.args).toContain("--base-url");
    expect(call.args).toContain("http://10.0.0.1:8000");
    expect(call.outputDir).toBe(`${TMP_DIR}/benchmarks/${res.body.id}`);
    // The --save-result path passed to llama-benchy must live inside outputDir
    const idx = call.args.indexOf("--save-result");
    expect(call.args[idx + 1]).toBe(`${TMP_DIR}/benchmarks/${res.body.id}/result.json`);

    // Wait one microtask for the orchestrator's then() to land
    await new Promise((r) => setImmediate(r));

    const stored = await prisma.benchmarkRun.findUnique({
      where: { id: res.body.id },
      include: { results: true },
    });
    expect(stored?.status).toBe("completed");
    expect(stored?.results).toHaveLength(1);
    expect(stored?.meanTps).toBe(50);
  });

  it("returns 404 when the deployment does not exist", async () => {
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: "nope", presetId: "quick-smoke" });
    expect(res.status).toBe(404);
  });

  it("returns 409 when the deployment is not running", async () => {
    const node = await prisma.node.create({
      data: { name: "n", ipAddress: "10.0.0.1", status: "online" },
    });
    const model = await prisma.model.create({
      data: { name: "m", runtime: "vllm" },
    });
    const d = await prisma.deployment.create({
      data: { nodeId: node.id, modelId: model.id, status: "stopped", port: 8000 },
    });
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "quick-smoke" });
    expect(res.status).toBe(409);
    expect(res.body.error).toMatch(/running/i);
  });

  it("returns 409 if another benchmark is already running for that deployment", async () => {
    const d = await seedRunningDeployment();
    // Hold the first run open by returning a pending promise.
    runMock.mockReturnValue(new Promise(() => {}));
    const first = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "quick-smoke" });
    expect(first.status).toBe(201);

    const second = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "quick-smoke" });
    expect(second.status).toBe(409);
    expect(second.body.error).toMatch(/already/i);
  });

  it("400s when neither presetId nor a custom config is provided", async () => {
    const d = await seedRunningDeployment();
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id });
    expect(res.status).toBe(400);
  });

  it("accepts a fully custom config", async () => {
    const d = await seedRunningDeployment();
    runMock.mockResolvedValue({
      exitCode: 0, results: [],
      summary: { meanTps: null, meanTtfrMs: null }, rawOutput: null,
    });
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({
        deploymentId: d.id,
        config: {
          pp: [64], tg: [16], depth: [0], runs: 1,
          concurrency: [1], latencyMode: "api",
          enablePrefixCaching: false, skipCoherence: false,
        },
      });
    expect(res.status).toBe(201);
    expect(res.body.presetId).toBeNull();
  });
});

describe("GET /api/benchmarks", () => {
  it("returns runs filtered by deploymentId, newest first", async () => {
    const d = await seedRunningDeployment();
    await prisma.benchmarkRun.create({
      data: {
        deploymentId: d.id, modelName: "m", endpointUrl: "u",
        servedModelName: "m", config: "{}", status: "completed",
      },
    });
    await prisma.benchmarkRun.create({
      data: {
        deploymentId: d.id, modelName: "m", endpointUrl: "u",
        servedModelName: "m", config: "{}", status: "completed",
      },
    });
    const res = await request(makeApp())
      .get(`/api/benchmarks?deploymentId=${d.id}`);
    expect(res.status).toBe(200);
    expect(res.body).toHaveLength(2);
  });
});

describe("GET /api/benchmarks/:id", () => {
  it("returns a run with its results", async () => {
    const d = await seedRunningDeployment();
    const run = await prisma.benchmarkRun.create({
      data: {
        deploymentId: d.id, modelName: "m", endpointUrl: "u",
        servedModelName: "m", config: "{}", status: "completed",
        results: {
          create: [{
            opType: "tg", pp: 1, tg: 2, depth: 0, concurrency: 1, tps: 10,
          }],
        },
      },
    });
    const res = await request(makeApp()).get(`/api/benchmarks/${run.id}`);
    expect(res.status).toBe(200);
    expect(res.body.results).toHaveLength(1);
  });

  it("404 for unknown id", async () => {
    const res = await request(makeApp()).get("/api/benchmarks/missing");
    expect(res.status).toBe(404);
  });
});

describe("DELETE /api/benchmarks/:id", () => {
  it("removes the run and cascades to results", async () => {
    const d = await seedRunningDeployment();
    const run = await prisma.benchmarkRun.create({
      data: {
        deploymentId: d.id, modelName: "m", endpointUrl: "u",
        servedModelName: "m", config: "{}", status: "completed",
        results: {
          create: [{
            opType: "tg", pp: 1, tg: 2, depth: 0, concurrency: 1, tps: 10,
          }],
        },
      },
    });
    const res = await request(makeApp()).delete(`/api/benchmarks/${run.id}`);
    expect(res.status).toBe(204);
    expect(await prisma.benchmarkRun.findUnique({ where: { id: run.id } })).toBeNull();
    expect(await prisma.benchmarkResult.findMany({ where: { runId: run.id } })).toHaveLength(0);
  });
});

describe("POST /api/benchmarks/:id/cancel", () => {
  it("kills the child and marks the run canceled", async () => {
    const d = await seedRunningDeployment();
    const run = await prisma.benchmarkRun.create({
      data: {
        deploymentId: d.id, modelName: "m", endpointUrl: "u",
        servedModelName: "m", config: "{}", status: "running",
      },
    });
    const res = await request(makeApp()).post(`/api/benchmarks/${run.id}/cancel`);
    expect(res.status).toBe(200);
    expect(cancelMock).toHaveBeenCalledWith(run.id);
    const after = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
    expect(after?.status).toBe("canceled");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/benchmarks.routes.test.ts`
Expected: FAIL with "Cannot find module '../../routes/benchmarks.js'".

- [ ] **Step 3: Implement the route**

Create `packages/server/src/routes/benchmarks.ts`:

```ts
import express, { type Request, type Response } from "express";
import { join } from "node:path";
import { prisma } from "../prisma.js";
import { broadcast as sseBroadcast } from "../sse.js";
import {
  BENCHMARK_PRESETS,
  getPreset,
  type BenchmarkConfig,
} from "../benchmarks/presets.js";
import { buildBenchyArgs } from "../benchmarks/args.js";
import { deploymentEndpointUrl } from "../benchmarks/endpoint.js";
import {
  runBenchmark,
  cancelBenchmark,
} from "../benchmarks/orchestrator.js";

const SHARED_STORAGE =
  process.env.SHARED_STORAGE_PATH || "/mnt/tank";

export const benchmarksRouter = express.Router();

benchmarksRouter.get("/presets", (_req, res) => {
  res.json(BENCHMARK_PRESETS);
});

benchmarksRouter.get("/", async (req, res) => {
  const { deploymentId } = req.query as { deploymentId?: string };
  const runs = await prisma.benchmarkRun.findMany({
    where: deploymentId ? { deploymentId } : undefined,
    orderBy: { createdAt: "desc" },
    include: { deployment: { include: { node: true, model: true } } },
  });
  res.json(runs);
});

benchmarksRouter.get("/:id", async (req, res) => {
  const run = await prisma.benchmarkRun.findUnique({
    where: { id: req.params.id },
    include: {
      results: true,
      deployment: { include: { node: true, model: true } },
    },
  });
  if (!run) return res.status(404).json({ error: "not found" });
  res.json(run);
});

benchmarksRouter.delete("/:id", async (req, res) => {
  const existing = await prisma.benchmarkRun.findUnique({
    where: { id: req.params.id },
  });
  if (!existing) return res.status(404).end();
  await prisma.benchmarkRun.delete({ where: { id: req.params.id } });
  sseBroadcast({ type: "benchmark:deleted", payload: { id: req.params.id } });
  res.status(204).end();
});

benchmarksRouter.post("/:id/cancel", async (req, res) => {
  const run = await prisma.benchmarkRun.findUnique({
    where: { id: req.params.id },
  });
  if (!run) return res.status(404).end();
  cancelBenchmark(run.id);
  const updated = await prisma.benchmarkRun.update({
    where: { id: run.id },
    data: { status: "canceled", completedAt: new Date() },
  });
  sseBroadcast({ type: "benchmark:status", payload: updated });
  res.json(updated);
});

type StartBody = {
  deploymentId?: string;
  presetId?: string;
  config?: BenchmarkConfig;
};

benchmarksRouter.post("/", async (req: Request, res: Response) => {
  const { deploymentId, presetId, config: customConfig } = req.body as StartBody;

  if (!deploymentId) {
    return res.status(400).json({ error: "deploymentId is required" });
  }
  if (!presetId && !customConfig) {
    return res.status(400).json({ error: "presetId or config is required" });
  }

  const deployment = await prisma.deployment.findUnique({
    where: { id: deploymentId },
    include: { node: true, model: true },
  });
  if (!deployment) {
    return res.status(404).json({ error: "deployment not found" });
  }
  if (deployment.status !== "running") {
    return res
      .status(409)
      .json({ error: "deployment is not running" });
  }

  const inflight = await prisma.benchmarkRun.findFirst({
    where: {
      deploymentId,
      status: { in: ["pending", "running"] },
    },
  });
  if (inflight) {
    return res
      .status(409)
      .json({ error: "a benchmark is already running for this deployment" });
  }

  let config: BenchmarkConfig;
  if (presetId) {
    const preset = getPreset(presetId);
    if (!preset) return res.status(400).json({ error: "unknown presetId" });
    config = preset.config;
  } else {
    config = customConfig!;
  }

  let endpointUrl: string;
  try {
    endpointUrl = deploymentEndpointUrl(deployment);
  } catch (e) {
    return res.status(409).json({ error: (e as Error).message });
  }
  const servedModelName = deployment.displayName ?? deployment.model.name;

  const run = await prisma.benchmarkRun.create({
    data: {
      deploymentId,
      presetId: presetId ?? null,
      modelName: deployment.model.name,
      endpointUrl,
      servedModelName,
      config: JSON.stringify(config),
      status: "pending",
    },
  });
  sseBroadcast({ type: "benchmark:created", payload: run });

  const outputDir = join(SHARED_STORAGE, "benchmarks", run.id);
  const args = buildBenchyArgs(config, {
    baseUrl: endpointUrl,
    modelName: servedModelName,
    outputPath: join(outputDir, "result.json"),
  });

  // Move to "running" immediately so the dashboard reflects state.
  await prisma.benchmarkRun.update({
    where: { id: run.id },
    data: { status: "running", startedAt: new Date() },
  });
  sseBroadcast({
    type: "benchmark:status",
    payload: { id: run.id, status: "running" },
  });

  runBenchmark({
    runId: run.id,
    args,
    outputDir,
    onLog: (line) => {
      sseBroadcast({
        type: "benchmark:log",
        payload: { runId: run.id, log: line },
      });
    },
  })
    .then(async (r) => {
      // SIGTERM from cancel exits the child non-zero; if the row was already
      // flipped to "canceled" by the cancel route, leave it alone.
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
        await prisma.benchmarkRun.update({
          where: { id: run.id },
          data: {
            status: "failed",
            completedAt: new Date(),
            error: `llama-benchy exited with code ${r.exitCode}`,
          },
        });
      }
      const final = await prisma.benchmarkRun.findUnique({
        where: { id: run.id },
        include: { results: true },
      });
      sseBroadcast({ type: "benchmark:status", payload: final });
    })
    .catch(async (e) => {
      await prisma.benchmarkRun.update({
        where: { id: run.id },
        data: {
          status: "failed",
          completedAt: new Date(),
          error: (e as Error).message,
        },
      });
      sseBroadcast({
        type: "benchmark:status",
        payload: { id: run.id, status: "failed", error: (e as Error).message },
      });
    });

  res.status(201).json(run);
});
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/benchmarks.routes.test.ts`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/routes/benchmarks.ts \
        packages/server/src/__tests__/integration/benchmarks.routes.test.ts
git commit -m "benchmarks: REST routes for runs, presets, cancel, delete"
```

---

## Task 9: Mount the router + reconcile stale runs in the server entrypoint

**Files:**
- Modify: `packages/server/src/index.ts`

**Why:** Server restarts orphan in-flight runs (the in-memory `ACTIVE` map is gone, but the DB row still says `running`). On boot we mark any such rows as `failed` so the dashboard reflects reality.

- [ ] **Step 1: Add the import, mount, and reconciliation**

Edit `packages/server/src/index.ts`. Add an import near the other router imports:

```ts
import { benchmarksRouter } from "./routes/benchmarks.js";
import { prisma } from "./prisma.js";
```

Add a `use` line in the routes block (right after `datasetsRouter`):

```ts
app.use("/api/benchmarks", benchmarksRouter);
```

And add a boot-time reconciliation block right before `server.listen(…)`:

```ts
// Any benchmark row left in "pending"/"running" across a restart is orphaned
// (the in-memory ACTIVE map was lost). Mark them failed so the dashboard
// doesn't show stale spinners forever.
await prisma.benchmarkRun.updateMany({
  where: { status: { in: ["pending", "running"] } },
  data: { status: "failed", error: "server restarted before run completed", completedAt: new Date() },
});
```

If `server.listen` isn't already inside an async wrapper, wrap the file's top-level execution in an `async function main() { … }` and call `main()` at the bottom — same pattern other server-side scripts already use.

- [ ] **Step 2: Run the full server build to confirm it type-checks**

Run: `npm run build`
Expected: build succeeds with no TypeScript errors.

- [ ] **Step 3: Run the full test suite**

Run: `npm test`
Expected: every test passes (including the new orchestrator + routes tests).

- [ ] **Step 4: Commit**

```bash
git add packages/server/src/index.ts
git commit -m "benchmarks: mount router + reconcile orphaned runs on boot"
```

---

## Task 10: Dashboard API client + types

**Files:**
- Create: `packages/dashboard/lib/benchmarks.ts`

**Why:** Single typed module to call the new endpoints, mirroring the existing `lib/api.ts` style. Used by every benchmark UI surface.

- [ ] **Step 1: Inspect the existing `lib/api.ts` to match its conventions**

Run: `cat packages/dashboard/lib/api.ts | head -n 40`
Expected: a small module that wraps `fetch` with a base URL from `NEXT_PUBLIC_API_URL`. Reuse whatever helper it exports.

- [ ] **Step 2: Write `lib/benchmarks.ts`**

Create `packages/dashboard/lib/benchmarks.ts`:

```ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

export type BenchmarkConfig = {
  pp: number[];
  tg: number[];
  depth: number[];
  runs: number;
  concurrency: number[];
  latencyMode: "api" | "generation" | "none";
  enablePrefixCaching: boolean;
  skipCoherence: boolean;
};

export type BenchmarkPreset = {
  id: string;
  label: string;
  description: string;
  config: BenchmarkConfig;
};

export type BenchmarkResult = {
  id: string;
  opType: string;
  pp: number;
  tg: number;
  depth: number;
  concurrency: number;
  tps: number;
  peakTps: number | null;
  ttfrMs: number | null;
  estPptMs: number | null;
  e2eTtftMs: number | null;
  tpsStdev: number | null;
  ttfrStdev: number | null;
};

export type BenchmarkRun = {
  id: string;
  deploymentId: string | null;
  presetId: string | null;
  modelName: string;
  endpointUrl: string;
  servedModelName: string;
  config: string;
  status: "pending" | "running" | "completed" | "failed" | "canceled";
  error: string | null;
  startedAt: string | null;
  completedAt: string | null;
  meanTps: number | null;
  meanTtfrMs: number | null;
  rawOutput: string | null;
  createdAt: string;
  results?: BenchmarkResult[];
  deployment?: {
    id: string;
    displayName: string | null;
    node: { id: string; name: string; ipAddress: string | null };
    model: { id: string; name: string; runtime: string };
  } | null;
};

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function listPresets(): Promise<BenchmarkPreset[]> {
  return jsonOrThrow(await fetch(`${API_BASE}/api/benchmarks/presets`));
}

export async function listBenchmarks(opts?: { deploymentId?: string }): Promise<BenchmarkRun[]> {
  const qs = opts?.deploymentId ? `?deploymentId=${opts.deploymentId}` : "";
  return jsonOrThrow(await fetch(`${API_BASE}/api/benchmarks${qs}`));
}

export async function getBenchmark(id: string): Promise<BenchmarkRun> {
  return jsonOrThrow(await fetch(`${API_BASE}/api/benchmarks/${id}`));
}

export async function startBenchmark(body: {
  deploymentId: string;
  presetId?: string;
  config?: BenchmarkConfig;
}): Promise<BenchmarkRun> {
  return jsonOrThrow(
    await fetch(`${API_BASE}/api/benchmarks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function cancelBenchmark(id: string): Promise<BenchmarkRun> {
  return jsonOrThrow(
    await fetch(`${API_BASE}/api/benchmarks/${id}/cancel`, { method: "POST" }),
  );
}

export async function deleteBenchmark(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/benchmarks/${id}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 204) {
    throw new Error(`delete failed: ${res.status}`);
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add packages/dashboard/lib/benchmarks.ts
git commit -m "dashboard: API client + types for /api/benchmarks"
```

---

## Task 11: Install Recharts

**Files:**
- Modify: `packages/dashboard/package.json`
- Modify: `package-lock.json`

- [ ] **Step 1: Add the dependency**

Run:
```bash
npm install --workspace packages/dashboard recharts@^2.13.0
```
Expected: lockfile updated, `recharts` added to `packages/dashboard/package.json` dependencies.

- [ ] **Step 2: Commit**

```bash
git add packages/dashboard/package.json package-lock.json
git commit -m "dashboard: add recharts dependency for benchmark charts"
```

---

## Task 12: `BenchmarkFormModal` component

**Files:**
- Create: `packages/dashboard/components/benchmark-form-modal.tsx`

**Why:** Reusable modal opened from the Deployments page (and later from the per-deployment detail page). Lets the user pick a preset (default) or expand "Custom" to override `pp`/`tg`/`depth`/etc.

- [ ] **Step 1: Write the modal component**

Create `packages/dashboard/components/benchmark-form-modal.tsx`:

```tsx
"use client";

import { useEffect, useState } from "react";
import {
  listPresets,
  startBenchmark,
  type BenchmarkPreset,
  type BenchmarkConfig,
} from "@/lib/benchmarks";

type Props = {
  deploymentId: string;
  deploymentLabel: string;
  onClose: () => void;
  onStarted: () => void;
};

export function BenchmarkFormModal({
  deploymentId, deploymentLabel, onClose, onStarted,
}: Props) {
  const [presets, setPresets] = useState<BenchmarkPreset[]>([]);
  const [presetId, setPresetId] = useState<string>("quick-smoke");
  const [showCustom, setShowCustom] = useState(false);
  const [custom, setCustom] = useState<BenchmarkConfig>({
    pp: [128], tg: [32], depth: [0], runs: 1,
    concurrency: [1], latencyMode: "api",
    enablePrefixCaching: false, skipCoherence: false,
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { listPresets().then(setPresets).catch(() => {}); }, []);

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await startBenchmark(
        showCustom
          ? { deploymentId, config: custom }
          : { deploymentId, presetId },
      );
      onStarted();
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const parseIntList = (s: string): number[] =>
    s.split(",").map((x) => parseInt(x.trim(), 10)).filter((n) => Number.isFinite(n) && n > 0);

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-lg p-6 w-[640px] max-h-[90vh] overflow-y-auto">
        <h2 className="text-xl font-semibold mb-1">Run benchmark</h2>
        <p className="text-sm text-gray-400 mb-4">Target: {deploymentLabel}</p>

        {!showCustom && (
          <div className="space-y-2">
            {presets.map((p) => (
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
                    <div className="text-xs text-gray-500 mt-1">
                      pp=[{p.config.pp.join(",")}] tg=[{p.config.tg.join(",")}]
                      {" "}depth=[{p.config.depth.join(",")}]
                      {" "}concurrency=[{p.config.concurrency.join(",")}]
                      {" "}runs={p.config.runs}
                    </div>
                  </div>
                </div>
              </label>
            ))}
          </div>
        )}

        <button
          type="button"
          className="text-xs text-blue-400 hover:underline mt-3"
          onClick={() => setShowCustom((v) => !v)}
        >
          {showCustom ? "Use a preset instead" : "Custom configuration…"}
        </button>

        {showCustom && (
          <div className="space-y-3 mt-3 p-3 bg-gray-800 rounded">
            <Field label="Prompt tokens (comma-separated)"
              value={custom.pp.join(",")}
              onChange={(v) => setCustom({ ...custom, pp: parseIntList(v) })} />
            <Field label="Generated tokens (comma-separated)"
              value={custom.tg.join(",")}
              onChange={(v) => setCustom({ ...custom, tg: parseIntList(v) })} />
            <Field label="Context depths (comma-separated)"
              value={custom.depth.join(",")}
              onChange={(v) => setCustom({ ...custom, depth: parseIntList(v) })} />
            <Field label="Concurrency levels (comma-separated)"
              value={custom.concurrency.join(",")}
              onChange={(v) => setCustom({ ...custom, concurrency: parseIntList(v) })} />
            <Field label="Runs per cell"
              value={String(custom.runs)}
              onChange={(v) => setCustom({ ...custom, runs: parseInt(v, 10) || 1 })} />
            <label className="block text-sm">
              Latency mode
              <select
                className="block mt-1 bg-gray-900 border border-gray-700 rounded px-2 py-1"
                value={custom.latencyMode}
                onChange={(e) => setCustom({ ...custom, latencyMode: e.target.value as BenchmarkConfig["latencyMode"] })}>
                <option value="api">api</option>
                <option value="generation">generation</option>
                <option value="none">none</option>
              </select>
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox"
                checked={custom.enablePrefixCaching}
                onChange={(e) => setCustom({ ...custom, enablePrefixCaching: e.target.checked })} />
              Enable prefix caching measurement
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox"
                checked={custom.skipCoherence}
                onChange={(e) => setCustom({ ...custom, skipCoherence: e.target.checked })} />
              Skip coherence check
            </label>
          </div>
        )}

        {error && <div className="text-red-400 text-sm mt-3">{error}</div>}

        <div className="flex justify-end gap-2 mt-6">
          <button
            type="button"
            className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600"
            onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            disabled={submitting}
            className="px-3 py-1 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50"
            onClick={submit}>
            {submitting ? "Starting…" : "Start benchmark"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Field({
  label, value, onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block text-sm">
      {label}
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="block mt-1 w-full bg-gray-900 border border-gray-700 rounded px-2 py-1"
      />
    </label>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/dashboard/components/benchmark-form-modal.tsx
git commit -m "dashboard: BenchmarkFormModal (preset picker + custom config)"
```

---

## Task 13: Add "Benchmark" button on the Deployments page

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx`

**Why:** The user-facing entry point. The button only shows on running deployments. Clicking opens `BenchmarkFormModal`; the page also listens for `benchmark:status` SSE events to flash a small status pill on the row.

- [ ] **Step 1: Add the import + modal state at the top of the component**

Open `packages/dashboard/app/deployments/page.tsx`. Add to the imports:

```tsx
import { BenchmarkFormModal } from "@/components/benchmark-form-modal";
```

Add inside the component body (next to the other `useState` hooks):

```tsx
const [benchmarkTarget, setBenchmarkTarget] = useState<
  { id: string; label: string } | null
>(null);
const [latestBenchmarkStatus, setLatestBenchmarkStatus] = useState<
  Record<string, { status: string; runId: string }>
>({});
```

- [ ] **Step 2: Handle `benchmark:*` SSE events**

Find the existing `handleSSE` (or equivalent) callback. Add cases for the new event types alongside the `deployment:*` cases:

```tsx
if (event.type === "benchmark:created" || event.type === "benchmark:status") {
  const payload = event.payload as { id: string; deploymentId?: string | null; status: string };
  if (payload.deploymentId) {
    setLatestBenchmarkStatus((prev) => ({
      ...prev,
      [payload.deploymentId!]: { status: payload.status, runId: payload.id },
    }));
  }
}
```

- [ ] **Step 3: Add the "Benchmark" button to each running deployment row**

Find the row of action buttons rendered for each deployment (near the existing "API" link at around `page.tsx:1189`). Add right after it:

```tsx
{d.status === "running" && d.port && (
  <button
    type="button"
    className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-purple-300 transition-colors"
    onClick={() => setBenchmarkTarget({
      id: d.id,
      label: `${d.displayName ?? d.model.name} @ ${d.node?.name ?? "?"}`,
    })}
  >
    Benchmark
  </button>
)}
{latestBenchmarkStatus[d.id] && (
  <a
    href={`/benchmarks/${latestBenchmarkStatus[d.id].runId}`}
    className="text-xs px-2 py-1 rounded bg-purple-900/40 hover:bg-purple-900/60 text-purple-200"
  >
    {latestBenchmarkStatus[d.id].status}
  </a>
)}
```

- [ ] **Step 4: Render the modal**

Near the end of the component's returned JSX (sibling of the rest of the page content):

```tsx
{benchmarkTarget && (
  <BenchmarkFormModal
    deploymentId={benchmarkTarget.id}
    deploymentLabel={benchmarkTarget.label}
    onClose={() => setBenchmarkTarget(null)}
    onStarted={() => {/* SSE will populate latestBenchmarkStatus */}}
  />
)}
```

- [ ] **Step 5: Smoke-test in the browser**

Run:
```bash
./scripts/build-agent-bundles.sh && \
  MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build
```
Open `http://192.168.44.36:3000/deployments`. On a running deployment, click **Benchmark**. The modal should open and list five presets. Picking "Quick smoke" and clicking **Start benchmark** should return immediately, and the row should grow a "running" pill within a second.

- [ ] **Step 6: Commit**

```bash
git add packages/dashboard/app/deployments/page.tsx
git commit -m "dashboard: Benchmark button + status pill on Deployments page"
```

---

## Task 14: `BenchmarkResultTable` + `BenchmarkChart` components

**Files:**
- Create: `packages/dashboard/components/benchmark-result-table.tsx`
- Create: `packages/dashboard/components/benchmark-chart.tsx`

**Why:** Shared building blocks used by both the detail page and the compare page.

- [ ] **Step 1: Write the table component**

Create `packages/dashboard/components/benchmark-result-table.tsx`:

```tsx
"use client";

import type { BenchmarkResult } from "@/lib/benchmarks";

export function BenchmarkResultTable({ rows }: { rows: BenchmarkResult[] }) {
  if (rows.length === 0) {
    return <div className="text-sm text-gray-500">No results.</div>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead className="text-left text-gray-400 border-b border-gray-800">
          <tr>
            <Th>Op</Th><Th>pp</Th><Th>tg</Th><Th>depth</Th><Th>conc</Th>
            <Th>t/s</Th><Th>peak t/s</Th><Th>ttfr (ms)</Th>
            <Th>est_ppt (ms)</Th><Th>e2e_ttft (ms)</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-b border-gray-900 hover:bg-gray-900/40">
              <Td>{r.opType}</Td>
              <Td>{r.pp}</Td>
              <Td>{r.tg}</Td>
              <Td>{r.depth}</Td>
              <Td>{r.concurrency}</Td>
              <Td>{r.tps.toFixed(1)}{r.tpsStdev != null && <span className="text-gray-500"> ±{r.tpsStdev.toFixed(1)}</span>}</Td>
              <Td>{r.peakTps?.toFixed(1) ?? "—"}</Td>
              <Td>{r.ttfrMs?.toFixed(1) ?? "—"}{r.ttfrStdev != null && <span className="text-gray-500"> ±{r.ttfrStdev.toFixed(1)}</span>}</Td>
              <Td>{r.estPptMs?.toFixed(1) ?? "—"}</Td>
              <Td>{r.e2eTtftMs?.toFixed(1) ?? "—"}</Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const Th = ({ children }: { children: React.ReactNode }) => (
  <th className="px-2 py-1 font-medium">{children}</th>
);
const Td = ({ children }: { children: React.ReactNode }) => (
  <td className="px-2 py-1">{children}</td>
);
```

- [ ] **Step 2: Write the chart component**

Create `packages/dashboard/components/benchmark-chart.tsx`:

```tsx
"use client";

import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { BenchmarkResult } from "@/lib/benchmarks";

export type ChartSeries = {
  label: string;
  color: string;
  rows: BenchmarkResult[];
};

type Metric = "tps" | "ttfrMs" | "e2eTtftMs";

const METRIC_LABEL: Record<Metric, string> = {
  tps: "Tokens / second",
  ttfrMs: "Time to first response (ms)",
  e2eTtftMs: "End-to-end TTFT (ms)",
};

// Buckets rows by (op, pp, tg, depth, concurrency) so each x-axis tick
// represents one comparable workload across all series.
function bucketKey(r: BenchmarkResult): string {
  return `${r.opType}/pp${r.pp}/tg${r.tg}/d${r.depth}/c${r.concurrency}`;
}

export function BenchmarkChart({
  series, metric,
}: {
  series: ChartSeries[];
  metric: Metric;
}) {
  const allKeys = Array.from(
    new Set(series.flatMap((s) => s.rows.map(bucketKey))),
  ).sort();

  const data = allKeys.map((key) => {
    const row: Record<string, number | string> = { workload: key };
    for (const s of series) {
      const match = s.rows.find((r) => bucketKey(r) === key);
      if (match) {
        const v = match[metric];
        if (typeof v === "number") row[s.label] = v;
      }
    }
    return row;
  });

  return (
    <div className="h-72">
      <div className="text-sm text-gray-400 mb-1">{METRIC_LABEL[metric]}</div>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="workload" stroke="#9ca3af" tick={{ fontSize: 10 }} />
          <YAxis stroke="#9ca3af" />
          <Tooltip contentStyle={{ background: "#111827", border: "1px solid #374151" }} />
          <Legend />
          {series.map((s) => (
            <Bar key={s.label} dataKey={s.label} fill={s.color} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add packages/dashboard/components/benchmark-result-table.tsx \
        packages/dashboard/components/benchmark-chart.tsx
git commit -m "dashboard: BenchmarkResultTable + BenchmarkChart components"
```

---

## Task 15: `/benchmarks` list page

**Files:**
- Create: `packages/dashboard/app/benchmarks/page.tsx`
- Modify: `packages/dashboard/components/top-nav.tsx`

**Why:** Top-level inventory of every run, with filters and a route into the compare page.

- [ ] **Step 1: Add "Benchmarks" to the top nav**

Open `packages/dashboard/components/top-nav.tsx`. Find the existing nav-link list (Deployments, Nodes, etc.) and add an entry for `/benchmarks` with the label "Benchmarks", styled like the others.

- [ ] **Step 2: Write the list page**

Create `packages/dashboard/app/benchmarks/page.tsx`:

```tsx
"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  deleteBenchmark, listBenchmarks, type BenchmarkRun,
} from "@/lib/benchmarks";
import { useSSE } from "@/lib/sse";

export default function BenchmarksPage() {
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [filter, setFilter] = useState({
    deploymentName: "",
    presetId: "",
    status: "",
  });
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const refresh = useCallback(() => {
    listBenchmarks().then(setRuns).catch(() => {});
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useSSE((event) => {
    if (
      event.type === "benchmark:created" ||
      event.type === "benchmark:status" ||
      event.type === "benchmark:deleted"
    ) {
      refresh();
    }
  });

  const filtered = useMemo(() => runs.filter((r) => {
    const depName = r.deployment?.displayName ?? r.modelName ?? "";
    if (filter.deploymentName && !depName.toLowerCase().includes(filter.deploymentName.toLowerCase())) return false;
    if (filter.presetId && r.presetId !== filter.presetId) return false;
    if (filter.status && r.status !== filter.status) return false;
    return true;
  }), [runs, filter]);

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const compareHref =
    selected.size >= 2
      ? `/benchmarks/compare?ids=${Array.from(selected).join(",")}`
      : null;

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Benchmarks</h1>
        <div className="flex gap-2">
          {compareHref && (
            <Link
              href={compareHref}
              className="px-3 py-1 rounded bg-purple-700 hover:bg-purple-600 text-sm"
            >
              Compare ({selected.size})
            </Link>
          )}
        </div>
      </div>

      <div className="flex gap-2">
        <input
          placeholder="Filter by deployment/model…"
          value={filter.deploymentName}
          onChange={(e) => setFilter({ ...filter, deploymentName: e.target.value })}
          className="bg-gray-900 border border-gray-800 rounded px-2 py-1 text-sm"
        />
        <select
          value={filter.presetId}
          onChange={(e) => setFilter({ ...filter, presetId: e.target.value })}
          className="bg-gray-900 border border-gray-800 rounded px-2 py-1 text-sm">
          <option value="">All presets</option>
          <option value="quick-smoke">quick-smoke</option>
          <option value="chat-short">chat-short</option>
          <option value="chat-long">chat-long</option>
          <option value="code-32k">code-32k</option>
          <option value="throughput">throughput</option>
        </select>
        <select
          value={filter.status}
          onChange={(e) => setFilter({ ...filter, status: e.target.value })}
          className="bg-gray-900 border border-gray-800 rounded px-2 py-1 text-sm">
          <option value="">All statuses</option>
          <option value="pending">pending</option>
          <option value="running">running</option>
          <option value="completed">completed</option>
          <option value="failed">failed</option>
          <option value="canceled">canceled</option>
        </select>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="text-left text-gray-400 border-b border-gray-800">
            <tr>
              <th></th>
              <th className="px-2 py-1">When</th>
              <th className="px-2 py-1">Deployment</th>
              <th className="px-2 py-1">Model</th>
              <th className="px-2 py-1">Preset</th>
              <th className="px-2 py-1">Status</th>
              <th className="px-2 py-1">Mean t/s</th>
              <th className="px-2 py-1">Mean ttfr (ms)</th>
              <th className="px-2 py-1"></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={r.id} className="border-b border-gray-900 hover:bg-gray-900/40">
                <td className="px-2 py-1">
                  <input
                    type="checkbox"
                    checked={selected.has(r.id)}
                    onChange={() => toggle(r.id)}
                  />
                </td>
                <td className="px-2 py-1 text-gray-300">
                  {new Date(r.createdAt).toLocaleString()}
                </td>
                <td className="px-2 py-1">
                  {r.deployment?.displayName ?? r.deployment?.model.name ?? <span className="text-gray-500">(deleted)</span>}
                </td>
                <td className="px-2 py-1">{r.modelName}</td>
                <td className="px-2 py-1">{r.presetId ?? <span className="text-gray-500">custom</span>}</td>
                <td className="px-2 py-1">
                  <StatusPill status={r.status} />
                </td>
                <td className="px-2 py-1">{r.meanTps?.toFixed(1) ?? "—"}</td>
                <td className="px-2 py-1">{r.meanTtfrMs?.toFixed(1) ?? "—"}</td>
                <td className="px-2 py-1 flex gap-2">
                  <Link href={`/benchmarks/${r.id}`} className="text-blue-400 hover:underline">
                    View
                  </Link>
                  <button
                    type="button"
                    className="text-red-400 hover:underline"
                    onClick={async () => {
                      if (confirm("Delete this benchmark run?")) {
                        await deleteBenchmark(r.id);
                        refresh();
                      }
                    }}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const color = {
    completed: "bg-green-900/40 text-green-300",
    running:   "bg-blue-900/40 text-blue-300",
    pending:   "bg-gray-800 text-gray-300",
    failed:    "bg-red-900/40 text-red-300",
    canceled:  "bg-yellow-900/40 text-yellow-300",
  }[status] ?? "bg-gray-800 text-gray-300";
  return <span className={`px-2 py-0.5 rounded text-xs ${color}`}>{status}</span>;
}
```

- [ ] **Step 3: Smoke-test in the browser**

After the dashboard is running, visit `http://<host>:3000/benchmarks`. The page should render, the filters should narrow the list, checking two completed runs should reveal the **Compare** button.

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/app/benchmarks/page.tsx \
        packages/dashboard/components/top-nav.tsx
git commit -m "dashboard: /benchmarks list page with filters + multi-select"
```

---

## Task 16: `/benchmarks/[id]` detail page

**Files:**
- Create: `packages/dashboard/app/benchmarks/[id]/page.tsx`

**Why:** Single run drill-down: status header, chart of metrics, full result table, log tail.

- [ ] **Step 1: Write the page**

Create `packages/dashboard/app/benchmarks/[id]/page.tsx`:

```tsx
"use client";

import { use, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  cancelBenchmark, getBenchmark, type BenchmarkRun,
} from "@/lib/benchmarks";
import { BenchmarkResultTable } from "@/components/benchmark-result-table";
import { BenchmarkChart } from "@/components/benchmark-chart";
import { useSSE } from "@/lib/sse";

export default function BenchmarkDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [run, setRun] = useState<BenchmarkRun | null>(null);
  const [log, setLog] = useState<string>("");

  const refresh = useCallback(() => {
    getBenchmark(id).then(setRun).catch(() => {});
  }, [id]);

  useEffect(() => { refresh(); }, [refresh]);

  useSSE((event) => {
    if (event.type === "benchmark:status") {
      const p = event.payload as { id: string };
      if (p.id === id) refresh();
    }
    if (event.type === "benchmark:log") {
      const p = event.payload as { runId: string; log: string };
      if (p.runId === id) setLog((prev) => (prev + p.log + "\n").slice(-50_000));
    }
  });

  if (!run) return <div className="p-6 text-gray-400">Loading…</div>;

  const series = [
    { label: run.deployment?.displayName ?? run.modelName, color: "#a78bfa", rows: run.results ?? [] },
  ];

  return (
    <div className="p-6 space-y-6">
      <div>
        <Link href="/benchmarks" className="text-sm text-blue-400 hover:underline">
          ← All benchmarks
        </Link>
        <h1 className="text-2xl font-semibold mt-1">
          {run.deployment?.displayName ?? run.modelName}{" "}
          <span className="text-sm text-gray-500">({run.presetId ?? "custom"})</span>
        </h1>
        <div className="text-sm text-gray-400 mt-1">
          Endpoint: <code>{run.endpointUrl}</code> · Served as <code>{run.servedModelName}</code>
        </div>
        <div className="text-sm text-gray-400">
          Started: {run.startedAt ? new Date(run.startedAt).toLocaleString() : "—"} ·
          Finished: {run.completedAt ? new Date(run.completedAt).toLocaleString() : "—"}
        </div>
        <div className="mt-2 flex gap-2 items-center">
          <span className="text-sm">Status:</span> <code>{run.status}</code>
          {run.status === "running" && (
            <button
              type="button"
              className="text-xs px-2 py-1 rounded bg-yellow-700 hover:bg-yellow-600"
              onClick={() => cancelBenchmark(id).then(refresh)}
            >
              Cancel
            </button>
          )}
        </div>
        {run.error && <div className="text-red-400 mt-2">Error: {run.error}</div>}
      </div>

      {run.results && run.results.length > 0 && (
        <>
          <BenchmarkChart series={series} metric="tps" />
          <BenchmarkChart series={series} metric="ttfrMs" />
          <BenchmarkResultTable rows={run.results} />
        </>
      )}

      <details className="text-sm">
        <summary className="cursor-pointer text-gray-400">Live log</summary>
        <pre className="mt-2 p-3 bg-black rounded text-xs overflow-x-auto max-h-96">{log || "(no log)"}</pre>
      </details>

      {run.rawOutput && (
        <details className="text-sm">
          <summary className="cursor-pointer text-gray-400">Raw llama-benchy JSON</summary>
          <pre className="mt-2 p-3 bg-black rounded text-xs overflow-x-auto max-h-96">{run.rawOutput}</pre>
        </details>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/dashboard/app/benchmarks/[id]/page.tsx
git commit -m "dashboard: /benchmarks/[id] detail page with charts + live log"
```

---

## Task 17: `/benchmarks/compare` page

**Files:**
- Create: `packages/dashboard/app/benchmarks/compare/page.tsx`

**Why:** Picking two or more runs from the list yields a side-by-side bar chart on the metrics that matter (tps, ttfr, e2e), aligned on identical workload buckets.

- [ ] **Step 1: Write the page**

Create `packages/dashboard/app/benchmarks/compare/page.tsx`:

```tsx
"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { getBenchmark, type BenchmarkRun } from "@/lib/benchmarks";
import { BenchmarkChart, type ChartSeries } from "@/components/benchmark-chart";

const PALETTE = ["#a78bfa", "#34d399", "#f59e0b", "#60a5fa", "#f472b6", "#f87171"];

export default function ComparePage() {
  const search = useSearchParams();
  const ids = (search.get("ids") ?? "").split(",").filter(Boolean);
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);

  useEffect(() => {
    Promise.all(ids.map(getBenchmark))
      .then(setRuns)
      .catch(() => {});
    // ids is derived from the URL; reload when it changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search.get("ids")]);

  if (runs.length === 0) {
    return (
      <div className="p-6 text-gray-400">
        Pick at least two runs from the{" "}
        <Link href="/benchmarks" className="text-blue-400 hover:underline">benchmarks list</Link>.
      </div>
    );
  }

  const series: ChartSeries[] = runs.map((r, i) => ({
    label: `${r.deployment?.displayName ?? r.modelName} (${r.presetId ?? "custom"})`,
    color: PALETTE[i % PALETTE.length],
    rows: r.results ?? [],
  }));

  return (
    <div className="p-6 space-y-6">
      <div>
        <Link href="/benchmarks" className="text-sm text-blue-400 hover:underline">
          ← All benchmarks
        </Link>
        <h1 className="text-2xl font-semibold mt-1">Compare</h1>
        <ul className="text-sm text-gray-400 mt-2 space-y-1">
          {runs.map((r, i) => (
            <li key={r.id}>
              <span className="inline-block w-3 h-3 rounded-sm mr-2"
                    style={{ background: PALETTE[i % PALETTE.length] }} />
              <Link href={`/benchmarks/${r.id}`} className="hover:underline">
                {r.deployment?.displayName ?? r.modelName}
              </Link>{" "}
              · {r.presetId ?? "custom"} · {new Date(r.createdAt).toLocaleString()}
            </li>
          ))}
        </ul>
      </div>

      <BenchmarkChart series={series} metric="tps" />
      <BenchmarkChart series={series} metric="ttfrMs" />
      <BenchmarkChart series={series} metric="e2eTtftMs" />
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/dashboard/app/benchmarks/compare/page.tsx
git commit -m "dashboard: /benchmarks/compare page (side-by-side charts)"
```

---

## Task 18: End-to-end smoke test against a real deployment

**Files:** (no code; manual verification)

**Why:** Type-checking, unit tests, and integration tests don't exercise `uvx llama-benchy` against a real vLLM endpoint. This catches issues the test suite can't (uvx install problems, network reachability, real llama-benchy JSON schema mismatches).

- [ ] **Step 1: Confirm the environment**

```bash
docker compose ps
docker compose exec server uvx --from llama-benchy llama-benchy --help | head -n 5
```
Expected: `server` and `dashboard` containers are `running`; the `--help` output prints without errors.

- [ ] **Step 2: Pick a running deployment from the UI**

Visit `/deployments`. Identify one row in the `running` state with a known endpoint URL.

- [ ] **Step 3: Run "Quick smoke"**

Click **Benchmark** → "Quick smoke" → **Start benchmark**. Within ~30 seconds, the status pill should transition `pending` → `running` → `completed`.

- [ ] **Step 4: Verify the detail page**

Visit `/benchmarks/<id>`. Confirm:
- Charts render with at least one bar
- The result table has at least one row
- The "Raw llama-benchy JSON" section opens and the JSON parses

- [ ] **Step 5: Run a second benchmark with a different preset, then compare**

Repeat with "Chat (short)". On `/benchmarks`, check both runs and click **Compare**. Confirm both series appear in the chart.

- [ ] **Step 6: Test cancel**

Start a "Code (32k context)" run. Before it finishes, click **Cancel** on the detail page. Confirm:
- Status flips to `canceled` within a few seconds
- `docker compose exec server pgrep -af llama-benchy` returns no rows for the cancelled run

- [ ] **Step 7: Test deletion**

From the list page, delete one of the completed runs. Confirm the row disappears and `prisma studio` shows no orphaned `BenchmarkResult` rows for that `runId`.

- [ ] **Step 8: If anything failed, debug and fix**

The most likely failure mode is that the real `llama-benchy --format json` schema differs from the fixture in Task 6. If so:
1. Copy the real `result.json` from `$SHARED_STORAGE/benchmarks/<runId>/result.json` into `packages/server/src/__tests__/integration/benchmarks.fixtures/result.json` (overwriting the synthetic one)
2. Update `parser.ts` field names to match
3. Re-run `npx vitest run packages/server/src/benchmarks/parser.test.ts`
4. Commit: `git commit -am "benchmarks: align parser with real llama-benchy v<X.Y.Z> schema"`

- [ ] **Step 9: Final commit (if any fixes were made)**

If everything passed without fixes, skip this step. Otherwise, commit any small adjustments.

---

## Future Ideas (Out of Scope)

These were considered but explicitly deferred to keep this plan tight and shippable:

- **Scheduled benchmarks**: cron-style recurring runs (e.g. "run `chat-short` against every running deployment every Monday at 02:00") so we get a longitudinal performance record without operator action.
- **Auto-benchmark on deploy**: a per-recipe checkbox that triggers `quick-smoke` automatically after a deployment reaches `running` — surfaces regressions the moment a new model lands.
- **CSV export from the compare page** for sharing in Slack/docs.
- **Cost/efficiency view**: combine `meanTps` with the deployment's `vramActual` to compute t/s-per-GB-VRAM, useful for choosing between quantization formats.
- **Latency percentiles** (p50/p95/p99) from `llama-benchy`'s `--save-all-throughput-timeseries` output — requires us to also parse the timeseries CSV.
- **Side-by-side run with a load-balancer rule** as the target (benchmark the LB itself, not a single deployment) — `Deployment` would become optional on `BenchmarkRun` and we'd add `lbRuleId String?`.
- **Slack alerts on regression**: if `meanTps` for a deployment drops more than 10% vs. the previous run of the same preset, post to a configured channel.
- **Result retention policy** (e.g. delete `BenchmarkResult` rows older than 90 days while keeping `BenchmarkRun` headers) so the SQLite file doesn't grow unboundedly.

---

## Plan complete

Saved to `docs/superpowers/plans/2026-05-17-benchmark-runner-llama-benchy.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
