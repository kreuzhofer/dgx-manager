# Sparkrun Deploy Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DGX Manager's eugr `spark-vllm-docker` / `run-recipe.sh` deployment path with [sparkrun](https://github.com/spark-arena/sparkrun) as the inference launch backend, keeping the existing WS/status/metrics/admission plumbing (Design B, agent-side on the head node).

**Architecture:** The head-node agent runs `uvx sparkrun run` instead of `run-recipe.sh`. The recipe catalog comes from `sparkrun list`/`show` (registries) instead of scanning an NFS repo. Custom recipes launch by a `SHARED_STORAGE`-relative path through an API field. The SSH provisioner installs + sets up sparkrun. vLLM-shaped status/metrics scraping is preserved; non-vLLM runtimes launch but get no special metrics (YAGNI).

**Tech Stack:** TypeScript (ESM, strict), Node child_process, Vitest + @fast-check/vitest + supertest, Prisma/SQLite, WebSocket hubs.

**Spec:** `docs/superpowers/specs/2026-06-12-sparkrun-deploy-backend-design.md` (decisions D1–D6, verification items V1–V5).
**Discovery findings:** `docs/superpowers/specs/2026-06-12-sparkrun-discovery-findings.md` (sparkrun **0.2.38**; fixtures captured).

## Phase 0 Results & Required Plan Adjustments (apply as you reach each task)

The discovery spike (Task 0) is **complete** — V1/V2/V5 from safe commands, V3/V4 from a live
qwen3-1.7b run (sparkrun 0.2.38) on a freed node, then cleanly stopped. Concrete corrections to
the task code below:

- **Version pin (V5):** `SPARKRUN_PKG = "sparkrun==0.2.38"` everywhere (Tasks 3, 6, 11).
- **`list` fields (Task 1):** real keys are `name` (the `@registry/file` ref), `file`, `model`,
  `description`, `runtime`, `min_nodes`, `tp`, `gpu_mem`, `registry`. `tp`/`gpu_mem` may be `""`
  — tolerate `"" | number`. Map `ref = name`. Test loads `__fixtures__/sparkrun-list.json`.
- **`show` has NO `--json` (Task 2):** replace the `show --json` approach with **`export recipe
  <name>`** (YAML) for `defaults`/`metadata`/`runtime`/`container`/`command`, and parse the
  `show` **text** block only for the VRAM estimate (`Per-GPU total: N GB`, `DGX Spark fit: YES`).
  There is **no `metadata.model_vram`** — get VRAM from `show` text or keep DGX Manager's own
  estimate. Fixtures: `sparkrun-export-recipe.yaml`, `sparkrun-show.txt`.
- **Liveness via `cluster check-job` (Tasks 5, 6, 8):** prefer **`sparkrun cluster check-job
  <clusterId|recipe> -H <hosts>`** (exit 0 = running) over parsing `status` text — simpler and
  pinned by the live run. Keep a thin `status` text parser only if a richer listing is needed
  (`status` is keyed by a short cluster ID `[<hex>]`, container `sparkrun_<hex>_solo`).
  **Capture the cluster ID from `run` output** (`Cluster: sparkrun_<hex>`) and store
  `{ deploymentId -> clusterId, hosts, tp }`. `stop`/`status`/`check-job` **require `-H/--hosts`**
  (no default cluster). `stop <clusterId> -H <host>` confirmed clean (container removed, port freed).
- **Container naming is SAFE (collision risk retired):** sparkrun names containers
  `sparkrun_<hex>_solo`, **not** `vllm_node` — it coexists with eugr's container, so no
  force-removal hazard. Multi-deployment-per-node is fine.
- **vLLM metrics carry `{labels}` (Task 7):** regex must allow an optional `\{[^}]*\}` between
  metric name and value; this vLLM build uses `vllm:kv_cache_usage_perc`. Fixture:
  `sparkrun-vllm-metrics.txt`.
- **`setup` is non-interactive subcommands (Task 11):** use `setup ssh`, `setup earlyoom`,
  `setup docker-group` (and `setup cx7` if CX7 present) with `-H/--hosts` — **not** a
  `--non-interactive` flag. Avoid `setup wizard` (interactive).
- **NEW Task 11b — pre-warm the image during provisioning:** first deploy triggers a **~15-min
  from-source `docker build`** of `sparkrun-eugr-vllm-tf5` (compiles torch/FlashInfer for sm121).
  Provisioning must pre-build it (e.g. a warm-up `run --dry-run` won't build; trigger the actual
  build once, or call sparkrun's build path) so first real deploy isn't blocked for 15+ min.
  Also: the agent's deploy "building" phase (Tasks 6/8) must tolerate a multi-minute build
  without timing out and surface a distinct "building" status.

**Conventions reused:**
- Pure helpers tested next to source as `<name>.test.ts` (model on `packages/server/src/benchmarks/args.test.ts`).
- Property tests via `it.prop` from `@fast-check/vitest` with a plain-English invariant doc comment.
- Integration HTTP tests via supertest against an Express app mounting only the router under test, with a stub `agentHub` injected via `app.set("agentHub", …)` (model on `packages/server/src/__tests__/integration/deployments.vram-admission.test.ts`).
- Subprocess spawning modeled on `packages/server/src/benchmarks/orchestrator.ts` `spawnTracked` (detached, process-group kill).
- **Agent version bump is MANDATORY** after any `packages/agent/src/**` edit — done once in the final task.

---

## Phase 0 — Discovery Spike (resolves V1–V5, produces test fixtures)

> This phase runs sparkrun on a real node and **captures actual output into fixture files** that later TDD tasks consume. No production code yet. Its findings are written to a findings doc; if observed formats differ from what later tasks assume, update the type + parser in that task — the captured fixture is the source of truth.

### Task 0: Capture sparkrun behavior + output fixtures

**Files:**
- Create: `docs/superpowers/specs/2026-06-12-sparkrun-discovery-findings.md`
- Create: `packages/agent/src/runtime/__fixtures__/sparkrun-list.txt`
- Create: `packages/agent/src/runtime/__fixtures__/sparkrun-show.txt`
- Create: `packages/agent/src/runtime/__fixtures__/sparkrun-status.txt`

- [ ] **Step 1: Install + inspect setup for a non-interactive path (V1)**

On a DGX Spark node (or the manager host with SSH to one), run and record:
```bash
uvx sparkrun --version
uvx sparkrun setup --help        # look for non-interactive flags / config-file input
uvx sparkrun run --help
uvx sparkrun list --help
uvx sparkrun status --help
```
Record in the findings doc: the pinned version string, whether `setup` accepts flags / a config file for non-interactive use (V1), and which subcommands accept `--json` or similar (V2).

- [ ] **Step 2: Capture `list` / `show` output (V2)**

```bash
uvx sparkrun list --json > /tmp/list.txt 2>&1 || uvx sparkrun list > /tmp/list.txt 2>&1
uvx sparkrun show qwen3-1.7b-vllm --json > /tmp/show.txt 2>&1 || uvx sparkrun show qwen3-1.7b-vllm > /tmp/show.txt 2>&1
```
Copy the real captured output into the three `__fixtures__/*.txt` files (use whatever format sparkrun actually emits — JSON if available, else text). Note in findings whether output is JSON or text, and the field names present.

- [ ] **Step 3: Capture a deploy lifecycle + status + metrics (V3, V4)**

```bash
uvx sparkrun run qwen3-1.7b-vllm --no-follow      # detached
uvx sparkrun status > /tmp/status.txt 2>&1         # capture -> __fixtures__/sparkrun-status.txt
curl -s http://localhost:8000/metrics | head -40   # confirm vLLM metrics present (V4)
# restart-survival check (V3):
pkill -f 'sparkrun run' ; uvx sparkrun status       # is the workload still listed after the launcher dies?
uvx sparkrun stop qwen3-1.7b-vllm
```
Record in findings: status output format + the stable key that identifies a workload (recipe name? id?), whether `--tp` must be passed to `status`/`stop` (V3), and whether `/metrics` is reachable on `localhost:{port}` (V4).

- [ ] **Step 4: Write the findings doc + pin the version (V5)**

Fill `sparkrun-discovery-findings.md` with answers to V1–V5 and the exact version to pin. Define a module constant value to use later: `SPARKRUN_VERSION` (e.g. `"sparkrun==0.0.16"`).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-06-12-sparkrun-discovery-findings.md \
        packages/agent/src/runtime/__fixtures__/
git commit -m "docs(sparkrun): discovery spike findings + captured CLI fixtures"
```

---

## Phase 1 — Recipe catalog (sparkrun list/show → dashboard)

### Task 1: `sparkrun list` parser

**Files:**
- Create: `packages/agent/src/runtime/sparkrun-parse.ts`
- Test: `packages/agent/src/runtime/sparkrun-parse.test.ts`

- [ ] **Step 1: Write the failing test** (load the Phase 0 fixture; assert the shape the dashboard needs)

```ts
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseSparkrunList, type SparkrunRecipeSummary } from "./sparkrun-parse.js";

const fixture = readFileSync(
  join(__dirname, "__fixtures__/sparkrun-list.json"),  // captured in Phase 0 (48 recipes)
  "utf8",
);

describe("parseSparkrunList", () => {
  it("returns one summary per recipe with name + registry", () => {
    const recipes: SparkrunRecipeSummary[] = parseSparkrunList(fixture);
    expect(recipes.length).toBeGreaterThan(0);
    for (const r of recipes) {
      expect(typeof r.ref).toBe("string");
      expect(r.ref.length).toBeGreaterThan(0);
    }
    // at least one official recipe is present in the captured fixture
    expect(recipes.some((r) => r.ref.includes("qwen") || r.registry)).toBe(true);
  });

  it("never throws on empty input", () => {
    expect(parseSparkrunList("")).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/sparkrun-parse.test.ts`
Expected: FAIL — `parseSparkrunList` is not defined.

- [ ] **Step 3: Write minimal implementation** (reconcile field access against the real fixture format from Phase 0 — JSON branch if sparkrun emits JSON, text branch otherwise)

```ts
/** A recipe as the dashboard picker needs it: a launchable ref + display info. */
export interface SparkrunRecipeSummary {
  ref: string;          // what `sparkrun run <ref>` accepts, e.g. "qwen3-1.7b-vllm"
  name: string;         // display name (falls back to ref)
  description?: string;
  runtime?: string;     // declared runtime; opaque per D6
  registry?: string;    // e.g. "official" | "community" | custom name
}

/**
 * Parse `sparkrun list` output. If sparkrun emits JSON (confirmed in Phase 0),
 * the JSON branch is used; otherwise the tabular text branch parses columns.
 * Both paths are exercised by fixture tests.
 */
export function parseSparkrunList(raw: string): SparkrunRecipeSummary[] {
  const text = raw.trim();
  if (!text) return [];
  // JSON branch
  if (text.startsWith("[") || text.startsWith("{")) {
    const data = JSON.parse(text);
    const arr: any[] = Array.isArray(data) ? data : (data.recipes ?? []);
    return arr.map((r) => ({
      ref: String(r.ref ?? r.id ?? r.name),
      name: String(r.name ?? r.ref ?? r.id),
      description: r.description,
      runtime: r.runtime,
      registry: r.registry,
    }));
  }
  // Text branch: one recipe per non-header line, first token is the ref.
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !/^name\b/i.test(l) && !/^-+$/.test(l))
    .map((l) => {
      const [ref, ...rest] = l.split(/\s{2,}|\t/);
      return { ref: ref.trim(), name: ref.trim(), description: rest.join(" ").trim() || undefined };
    });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/runtime/sparkrun-parse.test.ts`
Expected: PASS. (If it fails because the real fixture differs, adjust the field access in Step 3 to match the captured fixture — the fixture is authoritative.)

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/sparkrun-parse.ts packages/agent/src/runtime/sparkrun-parse.test.ts
git commit -m "feat(agent): parse sparkrun list output into recipe summaries"
```

### Task 2: Carry deploy-relevant defaults on the list summary (REVISED per Phase 0)

**Rationale (controller decision):** The original Task 2 planned a separate `sparkrun show --json`
parser for defaults + VRAM. Phase 0 showed (a) `show` has no `--json` (text only) and `export
recipe` is YAML — both fragile/slow to parse per recipe; (b) **`list --json` already carries the
deploy defaults** (`model`, `min_nodes`, `tp`, `gpu_mem`); and (c) **sparkrun does its own VRAM
admission at run time** (`DGX Spark fit: YES/NO`). So a separate show/export parser is redundant
(YAGNI). Instead, extend the list summary with the fields already present in `list --json`.

**Files:**
- Modify: `packages/agent/src/runtime/sparkrun-parse.ts`
- Test: `packages/agent/src/runtime/sparkrun-parse.test.ts`

- [ ] **Step 1: Write the failing test** (extend the existing fixture test)

```ts
describe("parseSparkrunList — deploy defaults", () => {
  it("carries model, minNodes, and tolerates empty tp/gpu_mem", () => {
    const recipes = parseSparkrunList(fixture);
    // every recipe has a model string and minNodes >= 1
    for (const r of recipes) {
      expect(typeof r.model).toBe("string");
      expect(r.minNodes).toBeGreaterThanOrEqual(1);
      // tp/gpuMem are number | undefined (fixture has some "" — must become undefined, never NaN)
      if (r.tpDefault !== undefined) expect(Number.isFinite(r.tpDefault)).toBe(true);
      if (r.gpuMemDefault !== undefined) expect(Number.isFinite(r.gpuMemDefault)).toBe(true);
    }
    // a known multi-node recipe in the fixture reports minNodes > 1
    expect(recipes.some((r) => r.minNodes >= 2)).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/sparkrun-parse.test.ts -t "deploy defaults"`
Expected: FAIL — `model`/`minNodes`/`tpDefault`/`gpuMemDefault` not on the type/result.

- [ ] **Step 3: Extend the interface + mapping** in `sparkrun-parse.ts`

```ts
// add to SparkrunRecipeSummary:
//   model?: string;       // HF model id / path
//   minNodes: number;     // from min_nodes (default 1)
//   tpDefault?: number;   // from tp; "" -> undefined
//   gpuMemDefault?: number; // from gpu_mem; "" -> undefined

// helper: sparkrun emits "" for unset numeric fields — coerce to undefined, never NaN
function numOrUndef(v: unknown): number | undefined {
  if (v === "" || v == null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

// in the JSON branch mapping, add:
//   model: r.model ? String(r.model) : undefined,
//   minNodes: Number(r.min_nodes ?? 1),
//   tpDefault: numOrUndef(r.tp),
//   gpuMemDefault: numOrUndef(r.gpu_mem),
// in the text-fallback branch, set minNodes: 1 and leave the rest undefined.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/runtime/sparkrun-parse.test.ts`
Expected: PASS (all parseSparkrunList tests).

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/sparkrun-parse.ts packages/agent/src/runtime/sparkrun-parse.test.ts
git commit -m "feat(agent): carry model/minNodes/tp/gpu_mem defaults on list summary"
```

### Task 3: Replace agent recipe discovery with sparkrun list

**Files:**
- Modify: `packages/agent/src/recipes.ts` (replace `discoverRecipes` body; remove `spark-vllm-docker` clone/scan + `VLLM_REPO_URL`)
- Modify: `packages/agent/src/index.ts` (the `agent:recipes` send + `cmd:rescan-recipes` handler now call the new discovery)

- [ ] **Step 1: Write the failing test** — `packages/agent/src/recipes.test.ts`

```ts
import { describe, it, expect, vi } from "vitest";

vi.mock("node:child_process", () => ({
  execFileSync: vi.fn(() => "[]"),
}));

import { discoverRecipes } from "./recipes.js";

describe("discoverRecipes", () => {
  it("returns summaries from `sparkrun list` (no NFS scan)", () => {
    const recipes = discoverRecipes();
    expect(Array.isArray(recipes)).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/recipes.test.ts`
Expected: FAIL — current `discoverRecipes` scans the filesystem and ignores the mock.

- [ ] **Step 3: Write minimal implementation** — replace the body of `recipes.ts`

**CONTRACT NOTE (must not break):** the `Recipe` shape the agent emits over `agent:recipes` is
consumed verbatim by the server (`recipe.file === recipeFile` matching in `deployments.ts`;
`recipe.defaults.{tensor_parallel,pipeline_parallel,gpu_memory_utilization}`) and the dashboard
(deploy-form pre-fill from `defaults.{port,max_model_len,tensor_parallel,pipeline_parallel,
gpu_memory_utilization}`, `cluster_only` for cluster UI, `name`/`description`/`model` display,
and the form POSTs `recipeFile = recipe.file`). So **keep the `Recipe` interface unchanged** and
MAP sparkrun summaries into it. `file` MUST be the sparkrun launch ref (what later flows to
`sparkrun run <ref>`).

```ts
import { execFileSync } from "node:child_process";
import { parseSparkrunList, type SparkrunRecipeSummary } from "./runtime/sparkrun-parse.js";

/** Pinned in Phase 0 findings; keep agent + provisioner in sync. */
export const SPARKRUN_PKG = "sparkrun==0.2.38";

// KEEP the existing Recipe interface (file/name/description/model/container/
// cluster_only/solo_only/defaults) — downstream depends on it. Do NOT alias it
// to SparkrunRecipeSummary.

/** Map a sparkrun list summary into the wire Recipe shape the dashboard/server expect. */
function toRecipe(s: SparkrunRecipeSummary): Recipe {
  return {
    file: s.ref,                          // launch ref; flows back as recipeFile -> sparkrun run
    name: s.name,
    description: s.description,
    model: s.model,
    container: "sparkrun",                // not consumed for vLLM deploy; sparkrun resolves the image
    cluster_only: s.minNodes > 1 ? true : undefined,
    solo_only: undefined,                 // list --json has no max_nodes; leave unset
    defaults: {
      tensor_parallel: s.tpDefault ?? 1,
      pipeline_parallel: 1,
      gpu_memory_utilization: s.gpuMemDefault ?? 0.85,
      port: 8000,
      max_model_len: "",
    },
  };
}

/** Enumerate recipes from sparkrun's configured registries. */
export function discoverRecipes(): Recipe[] {
  try {
    const out = execFileSync(
      "uvx",
      ["--from", SPARKRUN_PKG, "sparkrun", "list", "--json"],
      { encoding: "utf8", timeout: 30_000 },
    );
    return parseSparkrunList(out).map(toRecipe);
  } catch (err) {
    console.error("sparkrun list failed:", err);
    return [];
  }
}
```

(Delete the old `VLLM_REPO_URL`, clone logic, and custom YAML parser from this file. Keep the
exported `Recipe` interface exactly as it is today.)

- [ ] **Step 3b: Strengthen the test** — assert the MAPPING, not just array-ness:

```ts
import { describe, it, expect, vi } from "vitest";
vi.mock("node:child_process", () => ({
  execFileSync: vi.fn(() => JSON.stringify([
    { name: "@reg/qwen3-1.7b-vllm", file: "qwen3-1.7b-vllm", model: "Qwen/Qwen3-1.7B",
      description: "", runtime: "vllm-distributed", min_nodes: 1, tp: 1, gpu_mem: 0.3, registry: "reg" },
    { name: "@reg/big", file: "big", model: "X", runtime: "vllm", min_nodes: 2, tp: 2, gpu_mem: "", registry: "reg" },
  ])),
}));
import { discoverRecipes } from "./recipes.js";

describe("discoverRecipes", () => {
  it("maps sparkrun summaries to the wire Recipe shape", () => {
    const r = discoverRecipes();
    expect(r).toHaveLength(2);
    expect(r[0].file).toBe("@reg/qwen3-1.7b-vllm");          // file === launch ref
    expect(r[0].defaults.tensor_parallel).toBe(1);
    expect(r[0].defaults.gpu_memory_utilization).toBe(0.3);
    expect(r[0].cluster_only).toBeUndefined();
    expect(r[1].cluster_only).toBe(true);                     // min_nodes 2 -> cluster_only
    expect(r[1].defaults.gpu_memory_utilization).toBe(0.85);  // empty gpu_mem -> default
  });
});
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/recipes.test.ts`
Expected: PASS.

- [ ] **Step 5: Update `index.ts` callers + run agent build**

In `packages/agent/src/index.ts`, ensure the `agent:recipes` payload and the `cmd:rescan-recipes` handler call `discoverRecipes()` (unchanged call site if it already does). Then:
Run: `npm run build -w packages/agent`
Expected: no type errors.

- [ ] **Step 6: Commit**

```bash
git add packages/agent/src/recipes.ts packages/agent/src/recipes.test.ts packages/agent/src/index.ts
git commit -m "feat(agent): source recipe catalog from sparkrun list, drop NFS scan"
```

---

## Phase 2 — Launch path (replace run-recipe.sh)

### Task 4: `buildSparkrunArgs` pure helper (property-tested)

**Files:**
- Create: `packages/agent/src/runtime/sparkrun-args.ts`
- Test: `packages/agent/src/runtime/sparkrun-args.test.ts`

- [ ] **Step 1: Write the failing test** (unit + property; invariants from spec §7)

```ts
import { describe, it, expect } from "vitest";
import { it as itProp, fc } from "@fast-check/vitest";
import { buildSparkrunArgs } from "./sparkrun-args.js";

describe("buildSparkrunArgs", () => {
  it("solo deploy: ref, --no-follow, port forwarded; no host flag", () => {
    const args = buildSparkrunArgs({
      recipeRef: "qwen3-1.7b-vllm",
      hosts: ["10.0.0.1"],
      port: 8000,
    });
    expect(args).toContain("run");
    expect(args).toContain("qwen3-1.7b-vllm");
    expect(args).toContain("--no-follow");
    expect(args.join(" ")).toContain("--port 8000");
    // single host => no -H needed
    expect(args).not.toContain("-H");
  });

  it("cluster deploy: -H lists head first, --tp equals host count", () => {
    const args = buildSparkrunArgs({
      recipeRef: "big-model",
      hosts: ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
    });
    const hIdx = args.indexOf("-H");
    expect(hIdx).toBeGreaterThanOrEqual(0);
    expect(args[hIdx + 1]).toBe("10.0.0.1,10.0.0.2,10.0.0.3");
    const tpIdx = args.indexOf("--tp");
    expect(args[tpIdx + 1]).toBe("3");
  });

  it("forwards -o overrides verbatim", () => {
    const args = buildSparkrunArgs({
      recipeRef: "r",
      hosts: ["h"],
      options: { max_model_len: 8192, foo: "bar" },
    });
    expect(args).toContain("-o");
    expect(args.join(" ")).toContain("max_model_len=8192");
    expect(args.join(" ")).toContain("foo=bar");
  });

  it("never emits eugr run-recipe.sh flags", () => {
    const args = buildSparkrunArgs({ recipeRef: "r", hosts: ["a", "b"] });
    expect(args).not.toContain("--eth-if");
    expect(args).not.toContain("--ib-if");
    expect(args).not.toContain("--setup");
  });

  /** Invariant: for any non-empty host list, --tp (when not explicitly overridden)
   * equals the number of hosts, because each DGX Spark contributes exactly one GPU. */
  itProp.prop([fc.array(fc.string({ minLength: 1 }), { minLength: 1, maxLength: 8 })])(
    "tp defaults to host count",
    (hosts) => {
      const args = buildSparkrunArgs({ recipeRef: "r", hosts });
      const tpIdx = args.indexOf("--tp");
      expect(args[tpIdx + 1]).toBe(String(hosts.length));
    },
  );
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/sparkrun-args.test.ts`
Expected: FAIL — `buildSparkrunArgs` not defined.

- [ ] **Step 3: Write minimal implementation**

```ts
export interface SparkrunLaunchOptions {
  recipeRef: string;             // registry name | @spark-arena/id | URL | absolute path
  hosts: string[];               // head first; length drives --tp by default
  tp?: number;                   // explicit tensor-parallel override
  pp?: number;
  port?: number;
  gpuMem?: number;
  maxModelLen?: number;
  servedModelName?: string;
  options?: Record<string, string | number>;  // -o key=value passthrough
}

/** Build the argv for `uvx --from <pkg> sparkrun <argv>`. Pure + deterministic. */
export function buildSparkrunArgs(o: SparkrunLaunchOptions): string[] {
  const args: string[] = ["run", o.recipeRef, "--no-follow"];
  if (o.hosts.length > 1) args.push("-H", o.hosts.join(","));
  const tp = o.tp ?? o.hosts.length;
  args.push("--tp", String(tp));
  if (o.pp != null) args.push("--pp", String(o.pp));
  if (o.port != null) args.push("--port", String(o.port));
  if (o.gpuMem != null) args.push("--gpu-mem", String(o.gpuMem));
  if (o.maxModelLen != null) args.push("--max-model-len", String(o.maxModelLen));
  if (o.servedModelName) args.push("--served-model-name", o.servedModelName);
  for (const [k, v] of Object.entries(o.options ?? {})) {
    args.push("-o", `${k}=${v}`);
  }
  return args;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/runtime/sparkrun-args.test.ts`
Expected: PASS (5 cases incl. property).

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/sparkrun-args.ts packages/agent/src/runtime/sparkrun-args.test.ts
git commit -m "feat(agent): buildSparkrunArgs pure helper for sparkrun run argv"
```

### Task 5: `parseClusterId` (REVISED per Phase 0 — drop brittle status-list parser)

**Rationale:** Phase 0 showed `status` is human text and the clean liveness probe is
`sparkrun cluster check-job <clusterId|recipe> -H <hosts>` (exit 0 = running) — see Task 6.
Reconciliation works off the **stored cluster ID per deployment** + check-job, so a full
status-list parser is unnecessary (YAGNI). The one pure thing we need from sparkrun output is to
**capture the cluster ID** that `run` prints (`Cluster: sparkrun_<hex>`), to store for later
`stop`/`check-job`.

**Files:**
- Modify: `packages/agent/src/runtime/sparkrun-parse.ts`
- Test: `packages/agent/src/runtime/sparkrun-parse.test.ts`

- [ ] **Step 1: Write the failing test** (fixture `sparkrun-run-dryrun.txt` contains a real
  `Cluster:   sparkrun_2cf3f3031766` line)

```ts
import { parseClusterId } from "./sparkrun-parse.js";

const runFixture = readFileSync(
  join(__dirname, "__fixtures__/sparkrun-run-dryrun.txt"),
  "utf8",
);

describe("parseClusterId", () => {
  it("extracts the sparkrun_<hex> cluster id from run output", () => {
    expect(parseClusterId(runFixture)).toBe("sparkrun_2cf3f3031766");
  });
  it("returns undefined when absent", () => {
    expect(parseClusterId("no cluster line here")).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/sparkrun-parse.test.ts -t parseClusterId`
Expected: FAIL — not defined.

- [ ] **Step 3: Write minimal implementation** (append to `sparkrun-parse.ts`)

```ts
/** Extract the `Cluster: sparkrun_<hex>` workload id printed by `sparkrun run`. */
export function parseClusterId(runOutput: string): string | undefined {
  const m = runOutput.match(/\bsparkrun_[0-9a-f]+\b/);
  return m ? m[0] : undefined;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/runtime/sparkrun-parse.test.ts -t parseClusterId`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/sparkrun-parse.ts packages/agent/src/runtime/sparkrun-parse.test.ts
git commit -m "feat(agent): parseClusterId extracts sparkrun workload id from run output"
```

### Task 6: `launchSparkrun` / `stopSparkrun` (spawn + lifecycle)

**Files:**
- Create: `packages/agent/src/runtime/sparkrun.ts`
- Test: `packages/agent/src/runtime/sparkrun.test.ts` (unit: spawn argv assembly with a mocked `spawn`)

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

const spawnMock = vi.fn(() => ({
  pid: 1234,
  stdout: { on: vi.fn() },
  stderr: { on: vi.fn() },
  on: vi.fn(),
  unref: vi.fn(),
}));
const execFileSyncMock = vi.fn(() => "");
vi.mock("node:child_process", () => ({ spawn: spawnMock, execFileSync: execFileSyncMock }));
// avoid real ~/.dgx-agent writes during the launch test
vi.mock("./deployment-store.js", () => ({ saveDeployment: vi.fn(), removeDeployment: vi.fn() }));

import { launchSparkrun, stopSparkrun, isWorkloadRunning } from "./sparkrun.js";

beforeEach(() => { spawnMock.mockClear(); execFileSyncMock.mockReset(); execFileSyncMock.mockReturnValue(""); });

describe("launchSparkrun", () => {
  it("spawns uvx with sparkrun run argv for the given recipe + hosts", () => {
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, () => {});
    expect(spawnMock).toHaveBeenCalledOnce();
    const [cmd, argv] = spawnMock.mock.calls[0];
    expect(cmd).toBe("uvx");
    expect(argv).toContain("sparkrun");
    expect(argv).toContain("run");
    expect(argv).toContain("qwen3-1.7b-vllm");
    expect(argv).toContain("--no-follow");
  });
});

describe("stopSparkrun", () => {
  it("calls sparkrun stop with target, -H hosts, and --tp", () => {
    stopSparkrun("dep-1", "sparkrun_abc123", ["10.0.0.1", "10.0.0.2"], 2);
    const [cmd, argv] = execFileSyncMock.mock.calls[0];
    expect(cmd).toBe("uvx");
    expect(argv).toEqual(expect.arrayContaining(["stop", "sparkrun_abc123", "-H", "10.0.0.1,10.0.0.2", "--tp", "2"]));
  });
});

describe("isWorkloadRunning", () => {
  it("true when check-job exits 0, false when it throws", () => {
    execFileSyncMock.mockReturnValueOnce("");
    expect(isWorkloadRunning("sparkrun_abc", ["10.0.0.1"])).toBe(true);
    execFileSyncMock.mockImplementationOnce(() => { throw new Error("exit 1"); });
    expect(isWorkloadRunning("sparkrun_abc", ["10.0.0.1"])).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/sparkrun.test.ts`
Expected: FAIL — `launchSparkrun` not defined.

- [ ] **Step 3: Write minimal implementation** (REVISED per Phase 0 — `parseClusterId` capture + `cluster check-job` liveness, no status-list parser)

First extend `deployment-store.ts`'s `TrackedDeployment` interface with two optional fields:
`clusterId?: string;` and `tp?: number;` (additive — existing `recipeFile`/`recipeName`/`port`/
`clusterNodes` already cover the rest). No other store change.

```ts
// packages/agent/src/runtime/sparkrun.ts
import { spawn, execFileSync } from "node:child_process";
import { SPARKRUN_PKG } from "../recipes.js";
import { buildSparkrunArgs, type SparkrunLaunchOptions } from "./sparkrun-args.js";
import { parseClusterId } from "./sparkrun-parse.js";
import { saveDeployment, removeDeployment } from "./deployment-store.js";

export type Opts = Omit<SparkrunLaunchOptions, "recipeRef"> & { recipeName?: string };

/** Launch a recipe via sparkrun on the head node (this agent). Detached.
 *  Captures the `Cluster: sparkrun_<hex>` id from output and persists it for stop/check-job. */
export function launchSparkrun(
  deploymentId: string,
  recipeRef: string,
  opts: Opts,
  onLog: (line: string) => void,
  onExit: (code: number | null) => void,
): void {
  const argv = ["--from", SPARKRUN_PKG, "sparkrun", ...buildSparkrunArgs({ recipeRef, ...opts })];
  const child = spawn("uvx", argv, { detached: true });
  const hosts = opts.hosts;
  const tp = opts.tp ?? hosts.length;
  let buf = "";
  let clusterId: string | undefined;
  const persist = () =>
    saveDeployment({
      deploymentId,
      recipeFile: recipeRef,
      recipeName: opts.recipeName ?? recipeRef,
      port: opts.port ?? 8000,
      startedAt: new Date().toISOString(),
      clusterNodes: hosts,
      clusterId,
      tp,
    });
  const onData = (b: Buffer) => {
    const s = b.toString();
    onLog(s);
    if (!clusterId) {
      buf += s;
      const id = parseClusterId(buf);
      if (id) { clusterId = id; persist(); }   // re-persist once we know the id
    }
  };
  child.stdout?.on("data", onData);
  child.stderr?.on("data", onData);
  child.on("exit", (code) => onExit(code));
  persist();   // initial record (clusterId fills in when sparkrun prints it)
}

/** Stop a sparkrun workload. target = stored clusterId (preferred) or recipe ref.
 *  `-H hosts` and `--tp` are required to match the run-time cluster resolution (Phase 0). */
export function stopSparkrun(deploymentId: string, target: string, hosts: string[], tp?: number): void {
  const args = ["--from", SPARKRUN_PKG, "sparkrun", "stop", target, "-H", hosts.join(",")];
  if (tp != null) args.push("--tp", String(tp));
  try {
    execFileSync("uvx", args, { timeout: 120_000 });
  } finally {
    removeDeployment(deploymentId);
  }
}

/** Liveness probe via `sparkrun cluster check-job` (exit 0 = running). target = clusterId|ref. */
export function isWorkloadRunning(target: string, hosts: string[]): boolean {
  try {
    execFileSync(
      "uvx",
      ["--from", SPARKRUN_PKG, "sparkrun", "cluster", "check-job", target, "-H", hosts.join(",")],
      { timeout: 30_000, stdio: "ignore" },
    );
    return true;
  } catch {
    return false;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/runtime/sparkrun.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/sparkrun.ts packages/agent/src/runtime/sparkrun.test.ts packages/agent/src/runtime/deployment-store.ts
git commit -m "feat(agent): launchSparkrun/stopSparkrun lifecycle + workload listing"
```

### Task 7: vLLM metrics scrape via sparkrun workloads (`checkDeployments`)

**Files:**
- Create: `packages/agent/src/runtime/sparkrun-metrics.ts` (move vLLM `/metrics` parsing out of `vllm.ts`)
- Test: `packages/agent/src/runtime/sparkrun-metrics.test.ts`

- [ ] **Step 1: Write the failing test** (the metrics text parser is pure; reuse the exact vLLM keys from `vllm.ts`)

```ts
import { describe, it, expect } from "vitest";
import { parseVllmMetrics } from "./sparkrun-metrics.js";

// Real Phase 0 shape: metric names carry {labels}; this vLLM uses kv_cache_usage_perc.
const sample = `
vllm:num_requests_running{engine="0",model_name="qwen3-1.7b"} 2.0
vllm:num_requests_waiting{engine="0",model_name="qwen3-1.7b"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="qwen3-1.7b"} 0.37
`;

describe("parseVllmMetrics", () => {
  it("extracts requests running and kv-cache usage from labeled metrics", () => {
    const m = parseVllmMetrics(sample);
    expect(m.numRequestsRunning).toBe(2);
    expect(m.kvCacheUsagePerc).toBeCloseTo(0.37);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/sparkrun-metrics.test.ts`
Expected: FAIL — not defined.

- [ ] **Step 3: Write minimal implementation** (copy the existing regexes from `vllm.ts checkDeployments`; reconcile metric names against the Phase 0 `/metrics` capture)

```ts
export interface VllmMetrics {
  numRequestsRunning?: number;
  kvCacheUsagePerc?: number;
}

function num(text: string, key: string): number | undefined {
  // vLLM metric lines carry Prometheus {labels} (confirmed in Phase 0), e.g.
  //   vllm:num_requests_running{engine="0",model_name="qwen3-1.7b"} 0.0
  // so the optional `\{[^}]*\}` between name and value is REQUIRED.
  const esc = key.replace(/[:]/g, "\\:");
  const m = text.match(new RegExp(`^${esc}(\\{[^}]*\\})?\\s+([0-9.eE+-]+)`, "m"));
  return m ? Number(m[2]) : undefined;
}

export function parseVllmMetrics(text: string): VllmMetrics {
  return {
    numRequestsRunning: num(text, "vllm:num_requests_running"),
    kvCacheUsagePerc:
      num(text, "vllm:gpu_cache_usage_perc") ?? num(text, "vllm:kv_cache_usage_perc"),
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/runtime/sparkrun-metrics.test.ts`
Expected: PASS.

- [ ] **Step 5: Wire `checkDeployments`** — update the agent's deployment-status loop to: (a) for each tracked deployment, probe liveness via `isWorkloadRunning(clusterId ?? recipeFile, clusterNodes)` (from Task 6's `sparkrun.ts`), (b) `fetch http://localhost:{port}/metrics` and `parseVllmMetrics`. Keep the `VllmStatus` output shape so `index.ts`/`agent-hub` are unchanged. Run:
Run: `npm run build -w packages/agent`
Expected: no type errors.

- [ ] **Step 6: Commit**

```bash
git add packages/agent/src/runtime/sparkrun-metrics.ts packages/agent/src/runtime/sparkrun-metrics.test.ts packages/agent/src/runtime/vllm.ts
git commit -m "feat(agent): scrape vLLM metrics for sparkrun workloads"
```

### Task 8: Wire `cmd:deploy` / `cmd:undeploy` to sparkrun + reconcile on reconnect

**Files:**
- Modify: `packages/agent/src/index.ts` (`cmd:deploy` → `launchSparkrun`; `cmd:undeploy` → `stopSparkrun`; reconnect reconciliation uses `listSparkrunWorkloads()`)

- [ ] **Step 1: Write the failing test** — `packages/agent/src/runtime/deploy-status.test.ts` (extend the existing pure status-mapping test)

```ts
import { describe, it, expect } from "vitest";
import { reconcileDeployStatus } from "./deploy-status.js";

describe("reconcileDeployStatus", () => {
  it("running when sparkrun lists the ref, failed when absent and launcher dead", () => {
    expect(reconcileDeployStatus({ ref: "r", launcherAlive: false, listed: true })).toBe("running");
    expect(reconcileDeployStatus({ ref: "r", launcherAlive: false, listed: false })).toBe("failed");
    expect(reconcileDeployStatus({ ref: "r", launcherAlive: true, listed: false })).toBe("deploying");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/deploy-status.test.ts -t reconcileDeployStatus`
Expected: FAIL — `reconcileDeployStatus` not defined.

- [ ] **Step 3: Write minimal implementation** — add to `deploy-status.ts`

```ts
export function reconcileDeployStatus(s: {
  ref: string;
  launcherAlive: boolean;
  listed: boolean;
}): "running" | "deploying" | "failed" {
  if (s.listed) return "running";
  return s.launcherAlive ? "deploying" : "failed";
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/runtime/deploy-status.test.ts -t reconcileDeployStatus`
Expected: PASS.

- [ ] **Step 5: Wire `index.ts`** — in the `cmd:deploy` handler, build `Opts` from the payload (`hosts` = `clusterNodes` head-first, or `[localIp]` solo; `port`, `gpuMem`, etc. from `config`) and call `launchSparkrun(deploymentId, recipeRef, opts, onLog, onExit)`. In `cmd:undeploy`, look up the stored deployment and call `stopSparkrun(deploymentId, storedClusterId ?? recipeFile, clusterNodes, tp)`. On reconnect, for each stored deployment compute `reconcileDeployStatus({ ref, launcherAlive, listed: isWorkloadRunning(clusterId ?? recipeFile, clusterNodes) })`.

- [ ] **Step 5b: Handle `inlineRecipeYaml` (D7)** — in `cmd:deploy`, if the payload carries `inlineRecipeYaml`, write it to a transient file before launching and use that path as `recipeRef`:

```ts
// in cmd:deploy handler, before launchSparkrun:
import { writeFileSync, mkdirSync, rmSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

let recipeRef = payload.recipeRef as string | undefined;
if (payload.inlineRecipeYaml) {
  const dir = join(homedir(), ".dgx-agent", "adhoc");
  mkdirSync(dir, { recursive: true });
  recipeRef = join(dir, `${payload.deploymentId}.yaml`);
  writeFileSync(recipeRef, payload.inlineRecipeYaml as string, "utf8");
}
// ... launchSparkrun(deploymentId, recipeRef!, opts, ...)
```

On `cmd:undeploy`/cleanup, `rmSync(join(homedir(), ".dgx-agent", "adhoc", \`${deploymentId}.yaml\`), { force: true })` for the temp file. Add a unit test for a tiny `writeInlineRecipe(deploymentId, yaml, dir): string` helper (extract the write+path logic so it's testable without the WS handler): asserts the file is written under `dir` and the returned path ends with `<deploymentId>.yaml`.

- [ ] **Step 6: Run build + commit**

```bash
npm run build -w packages/agent
git add packages/agent/src/index.ts packages/agent/src/runtime/deploy-status.ts packages/agent/src/runtime/deploy-status.test.ts packages/agent/src/runtime/sparkrun.ts packages/agent/src/runtime/sparkrun.test.ts
git commit -m "feat(agent): route cmd:deploy/undeploy through sparkrun + inline YAML + reconnect reconcile"
```

---

## Phase 3 — Custom recipe launch-by-path (API)

### Task 9: `recipePath` validation helper (security boundary)

**Files:**
- Create: `packages/server/src/deployments/recipe-path.ts`
- Test: `packages/server/src/deployments/recipe-path.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from "vitest";
import { resolveRecipePath } from "./recipe-path.js";

const ROOT = "/mnt/tank";

describe("resolveRecipePath", () => {
  it("accepts an in-tree relative path", () => {
    expect(resolveRecipePath("recipes/dev/my.yaml", ROOT)).toBe("/mnt/tank/recipes/dev/my.yaml");
  });
  it("rejects parent-traversal", () => {
    expect(() => resolveRecipePath("../etc/passwd", ROOT)).toThrow(/outside shared storage/i);
  });
  it("rejects absolute escape", () => {
    expect(() => resolveRecipePath("/etc/passwd", ROOT)).toThrow(/outside shared storage/i);
  });
  it("rejects sneaky traversal that re-enters", () => {
    expect(() => resolveRecipePath("recipes/../../etc/passwd", ROOT)).toThrow(/outside shared storage/i);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/deployments/recipe-path.test.ts`
Expected: FAIL — not defined.

- [ ] **Step 3: Write minimal implementation**

```ts
import { resolve, sep } from "node:path";

/**
 * Resolve a user-supplied recipe path against SHARED_STORAGE, rejecting any
 * path that escapes it. Fail-fast: throws on violation (spec §4.4 security boundary).
 */
export function resolveRecipePath(rel: string, root: string): string {
  const base = resolve(root);
  const full = resolve(base, rel);
  if (full !== base && !full.startsWith(base + sep)) {
    throw new Error(`recipePath resolves outside shared storage: ${rel}`);
  }
  return full;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/deployments/recipe-path.test.ts`
Expected: PASS (4 cases).

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/deployments/recipe-path.ts packages/server/src/deployments/recipe-path.test.ts
git commit -m "feat(server): validate custom recipePath stays within shared storage"
```

### Task 9b: `validateInlineRecipe` helper (inline YAML body — D7)

**Files:**
- Create: `packages/server/src/deployments/recipe-inline.ts`
- Test: `packages/server/src/deployments/recipe-inline.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from "vitest";
import { validateInlineRecipe, MAX_INLINE_RECIPE_BYTES } from "./recipe-inline.js";

describe("validateInlineRecipe", () => {
  it("accepts a plausible sparkrun recipe", () => {
    expect(() => validateInlineRecipe("model: Qwen/Qwen3-1.7B\nruntime: vllm\n")).not.toThrow();
  });
  it("rejects empty / whitespace", () => {
    expect(() => validateInlineRecipe("   ")).toThrow(/empty/i);
  });
  it("rejects content that doesn't look like a recipe", () => {
    expect(() => validateInlineRecipe("hello world")).toThrow(/recipe/i);
  });
  it("rejects oversized input", () => {
    expect(() => validateInlineRecipe("model:\n" + "x".repeat(MAX_INLINE_RECIPE_BYTES))).toThrow(/too large/i);
  });
});
```

- [ ] **Step 2: Run → FAIL.** `npx vitest run packages/server/src/deployments/recipe-inline.test.ts`

- [ ] **Step 3: Implement**

```ts
/** Cap on an inline recipe body (defensive — a recipe is a few KB at most). */
export const MAX_INLINE_RECIPE_BYTES = 512 * 1024;

/**
 * Validate an inline sparkrun recipe YAML sent over the API (D7). Fail-fast: throws on
 * empty, oversized, or content that doesn't look like a recipe. NOTE: an accepted recipe's
 * `command:` runs in a container — this is an arbitrary-execution surface; callers must audit-log.
 */
export function validateInlineRecipe(yaml: string): void {
  if (!yaml || !yaml.trim()) throw new Error("recipeYaml is empty");
  if (Buffer.byteLength(yaml, "utf8") > MAX_INLINE_RECIPE_BYTES) {
    throw new Error(`recipeYaml too large (> ${MAX_INLINE_RECIPE_BYTES} bytes)`);
  }
  // Minimal shape check: a sparkrun recipe declares at least one of these top-level keys.
  if (!/^\s*(model|command|runtime)\s*:/m.test(yaml)) {
    throw new Error("recipeYaml does not look like a sparkrun recipe (needs model:/command:/runtime:)");
  }
}
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit**
```bash
git add packages/server/src/deployments/recipe-inline.ts packages/server/src/deployments/recipe-inline.test.ts
git commit -m "feat(server): validate inline recipe YAML body for API deploy (D7)"
```

### Task 10: Accept `recipePath` AND `recipeYaml` in the deploy route (D5 + D7)

**Files:**
- Modify: `packages/server/src/routes/deployments.ts` (accept `recipePath` and `recipeYaml` as alternatives to `recipeFile`; validate; carry into the `cmd:deploy` payload — `recipeRef` for file/path, `inlineRecipeYaml` for the YAML body)
- Test: `packages/server/src/__tests__/integration/deployments.sparkrun.test.ts`

> NOTE: the agent side of `inlineRecipeYaml` (write to `~/.dgx-agent/adhoc/<id>.yaml`, run sparkrun against it) is implemented in **Task 8 Step 5b**. This task only adds the server route handling: validate the body and put `inlineRecipeYaml` in the `cmd:deploy` payload.

- [ ] **Step 1: Write the failing test** (supertest; stub `agentHub`; model on `deployments.vram-admission.test.ts` setup)

```ts
import { describe, it, expect, beforeAll } from "vitest";
import request from "supertest";
import express from "express";
import { deploymentsRouter } from "../../routes/deployments.js";

function appWith(agentHub: any) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", agentHub);
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

describe("POST /api/deployments with recipePath", () => {
  it("rejects a traversal recipePath with 400 and emits no deploy", async () => {
    const sent: any[] = [];
    const app = appWith({
      getRecipes: () => [],
      sendToAgent: (_id: string, msg: any) => sent.push(msg),
      onlineNodeIds: () => ["node-a"],
    });
    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "node-a", recipePath: "../../etc/passwd" });
    expect(res.status).toBe(400);
    expect(sent.filter((m) => m.type === "cmd:deploy")).toHaveLength(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/deployments.sparkrun.test.ts`
Expected: FAIL — route does not yet handle `recipePath` (likely 400 "recipeFile required" but for the wrong reason, or 500).

- [ ] **Step 3: Write minimal implementation** — in `deployments.ts`, after destructuring the body (incl. `recipePath`, `recipeYaml`) add:

```ts
import { resolveRecipePath } from "../deployments/recipe-path.js";
import { validateInlineRecipe } from "../deployments/recipe-inline.js";
import { SHARED_STORAGE } from "../env.js";

// ...inside the POST handler, before building the deploy payload:
let recipeRef: string | undefined;          // registry ref or validated path
let inlineRecipeYaml: string | undefined;   // raw YAML body (D7)
if (recipeYaml) {
  try {
    validateInlineRecipe(recipeYaml);
  } catch (e) {
    return res.status(400).json({ error: (e as Error).message });
  }
  inlineRecipeYaml = recipeYaml;
  // audit: inline recipe = arbitrary container command; log who/when/size
  console.log(`[deploy] inline recipeYaml (${Buffer.byteLength(recipeYaml, "utf8")} bytes) from ${req.ip}`);
} else if (recipePath) {
  try {
    recipeRef = resolveRecipePath(recipePath, SHARED_STORAGE);
  } catch (e) {
    return res.status(400).json({ error: (e as Error).message });
  }
} else if (recipeFile) {
  recipeRef = recipeFile; // registry ref / name from `sparkrun list`
} else if (!isOllama) {
  return res.status(400).json({ error: "recipeFile, recipePath, or recipeYaml required for vLLM deployments" });
}
```

Then include both `recipeRef` and `inlineRecipeYaml` in the `cmd:deploy` payload the agent reads
(the agent prefers `inlineRecipeYaml` when present — Task 8 Step 5b). Add a `recipeYaml`
happy-path test asserting the emitted `cmd:deploy` carries `inlineRecipeYaml` and that an
oversized/empty body is rejected 400 with no deploy emitted.

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/deployments.sparkrun.test.ts`
Expected: PASS.

- [ ] **Step 5: Run the full deployments integration suite (no regressions)**

Run: `npx vitest run packages/server/src/__tests__/integration/deployments.vram-admission.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/routes/deployments.ts packages/server/src/__tests__/integration/deployments.sparkrun.test.ts
git commit -m "feat(server): accept validated recipePath; pass sparkrun ref to agent"
```

---

## Phase 4 — Provisioning (install + setup sparkrun on nodes)

> Gated by **V1** (non-interactive setup). If Phase 0 found sparkrun flags/config for non-interactive setup, use them; otherwise this task replicates the underlying steps (SSH mesh — keys already deployed for agents — plus sudoers + earlyoom) as documented in the findings doc.

### Task 11: Add sparkrun to the prerequisite audit + provision

**Files:**
- Modify: `packages/server/src/ssh/provisioner.ts` (add a sparkrun `PrereqCheck` to `auditNode`; add install/setup to `provisionNode`)
- Test: `packages/server/src/ssh/provisioner.sparkrun.test.ts` (unit: the audit check command + the provision command strings are pure-ish; assert them)

- [ ] **Step 1: Write the failing test** (extract the command builders as pure functions so they're testable without SSH)

```ts
import { describe, it, expect } from "vitest";
import { sparkrunAuditCmd, sparkrunInstallCmd } from "./provisioner.js";

describe("sparkrun provisioning commands", () => {
  it("audit checks sparkrun is runnable via uvx", () => {
    expect(sparkrunAuditCmd()).toContain("sparkrun");
    expect(sparkrunAuditCmd()).toContain("uvx");
  });
  it("install runs setup non-interactively (per findings)", () => {
    const cmd = sparkrunInstallCmd();
    expect(cmd).toContain("sparkrun");
    // setup invoked with the non-interactive flag/config resolved in Phase 0
    expect(cmd).toMatch(/setup/);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/ssh/provisioner.sparkrun.test.ts`
Expected: FAIL — builders not defined.

- [ ] **Step 3: Write minimal implementation** — add to `provisioner.ts` (fill the exact setup flag from Phase 0 findings):

```ts
/** Audit: sparkrun resolvable via uvx (uvx itself is already provisioned). */
export function sparkrunAuditCmd(): string {
  return `uvx --from sparkrun sparkrun --version >/dev/null 2>&1 && echo installed`;
}

/** Install + non-interactive setup. Replace SETUP_FLAGS per Phase 0 findings. */
export function sparkrunInstallCmd(): string {
  const SETUP_FLAGS = "--non-interactive"; // <- exact flag/config from findings (V1)
  return `uvx --from sparkrun sparkrun setup ${SETUP_FLAGS}`;
}
```

Then add a `PrereqCheck` in `auditNode` using `sparkrunAuditCmd()` (status green if `installed`), and a branch in `provisionNode` that runs `sparkrunInstallCmd()` over SSH (model on the existing uvx/ollama install branches).

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/ssh/provisioner.sparkrun.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/ssh/provisioner.ts packages/server/src/ssh/provisioner.sparkrun.test.ts
git commit -m "feat(server): provision sparkrun (install + non-interactive setup) on nodes"
```

---

## Phase 5 — Fine-tune deploy reroute

### Task 12: Generate a sparkrun recipe for merged models + launch via path

**Files:**
- Modify: `packages/agent/src/index.ts` (`cmd:finetune:deploy` writes a sparkrun-format recipe to NFS, then `launchSparkrun(<path>)`)
- Modify: `packages/agent/src/runtime/inference-template.ts` (emit sparkrun-shaped YAML: add `runtime: vllm`)
- Test: `packages/agent/src/runtime/inference-template.test.ts` (extend)

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from "vitest";
import { renderSparkrunFinetuneRecipe } from "./inference-template.js";

describe("renderSparkrunFinetuneRecipe", () => {
  it("emits a sparkrun recipe pointing at the merged model with runtime vllm", () => {
    const yaml = renderSparkrunFinetuneRecipe({
      mergedModelPath: "/workspace/outputs/abc/merged",
      servedModelName: "My-Model",
      container: "vllm-node",
    });
    expect(yaml).toContain("runtime: vllm");
    expect(yaml).toContain("/workspace/outputs/abc/merged");
    expect(yaml).toContain("My-Model");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/inference-template.test.ts -t renderSparkrunFinetuneRecipe`
Expected: FAIL — not defined.

- [ ] **Step 3: Write minimal implementation** — add to `inference-template.ts`

```ts
export function renderSparkrunFinetuneRecipe(p: {
  mergedModelPath: string;
  servedModelName: string;
  container: string;
  maxModelLen?: number;
  gpuMem?: number;
}): string {
  return [
    `model: ${p.mergedModelPath}`,
    `runtime: vllm`,
    `container: ${p.container}`,
    `defaults:`,
    `  port: 8000`,
    `  host: 0.0.0.0`,
    `  tensor_parallel: 1`,
    `  gpu_memory_utilization: ${p.gpuMem ?? 0.85}`,
    `  max_model_len: ${p.maxModelLen ?? 4096}`,
    `  served_model_name: ${p.servedModelName}`,
    `command: |`,
    `  vllm serve ${p.mergedModelPath} --host {host} --port {port} \\`,
    `    --max-model-len {max_model_len} --gpu-memory-utilization {gpu_memory_utilization} \\`,
    `    -tp {tensor_parallel} --served-model-name {served_model_name} \\`,
    `    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --dtype auto`,
    ``,
  ].join("\n");
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/runtime/inference-template.test.ts -t renderSparkrunFinetuneRecipe`
Expected: PASS.

- [ ] **Step 5: Wire `cmd:finetune:deploy`** in `index.ts` — write the rendered YAML to `${SHARED_STORAGE}/recipes/finetune-${jobId.slice(0,12)}.yaml`, then `launchSparkrun(deploymentId, <that path>, { hosts, ... })`. (When a training recipe ships an `inference*.yaml` template, prefer it as today; otherwise use `renderSparkrunFinetuneRecipe`.) Run:
Run: `npm run build -w packages/agent`
Expected: no type errors.

- [ ] **Step 6: Commit**

```bash
git add packages/agent/src/runtime/inference-template.ts packages/agent/src/runtime/inference-template.test.ts packages/agent/src/index.ts
git commit -m "feat(agent): deploy fine-tuned models via sparkrun recipe-by-path"
```

---

## Phase 7 — OpenAPI spec + Swagger UI (D8)

> Execute after Phase 5; **Phase 6 (retire + version bump) runs last.** Goal: an agent or human
> can discover and understand the whole API from `GET /api/openapi.json` / `GET /api/docs`
> without reading source.

### Task 15: OpenAPI infrastructure + system overview

**Files:**
- Create: `packages/server/src/openapi.ts` (build the spec; rich `info.description` + tag descriptions)
- Modify: `packages/server/src/index.ts` (serve `/api/openapi.json` + `/api/docs`)
- Modify: `packages/server/package.json` (add deps)
- Test: `packages/server/src/__tests__/integration/openapi.routes.test.ts`

- [ ] **Step 1: Add deps**

Run: `npm i -w packages/server swagger-jsdoc swagger-ui-express && npm i -w packages/server -D @types/swagger-jsdoc @types/swagger-ui-express`

- [ ] **Step 2: Write the failing test**

```ts
import { describe, it, expect } from "vitest";
import request from "supertest";
import express from "express";
import { mountOpenApi } from "../../openapi.js";

function app() { const a = express(); mountOpenApi(a); return a; }

describe("OpenAPI", () => {
  it("serves a 3.x spec with system overview + domain tags", async () => {
    const res = await request(app()).get("/api/openapi.json");
    expect(res.status).toBe(200);
    expect(res.body.openapi).toMatch(/^3\./);
    expect(res.body.info.title).toMatch(/DGX Manager/i);
    expect(res.body.info.description.length).toBeGreaterThan(200); // system overview present
    const tagNames = (res.body.tags ?? []).map((t: any) => t.name);
    expect(tagNames).toEqual(expect.arrayContaining(["Deployments", "Nodes", "Recipes"]));
  });
  it("serves Swagger UI HTML", async () => {
    const res = await request(app()).get("/api/docs/");
    expect(res.status).toBe(200);
    expect(res.text).toMatch(/swagger-ui/i);
  });
});
```

- [ ] **Step 3: Run → FAIL.** `npx vitest run packages/server/src/__tests__/integration/openapi.routes.test.ts`

- [ ] **Step 4: Implement `openapi.ts`**

```ts
import swaggerJsdoc from "swagger-jsdoc";
import swaggerUi from "swagger-ui-express";
import type { Express } from "express";

const SYSTEM_OVERVIEW = `
# DGX Manager API

Full-stack control plane for a DGX Spark cluster. Domains and how they relate:

- **Nodes** — register/provision DGX Spark nodes over SSH; agents on each node report GPU metrics.
- **Recipes** — inference recipes discovered from sparkrun registries; the catalog the deploy form uses.
- **Deployments** — launch a recipe on one or more nodes via **sparkrun** (the head-node agent runs
  \`sparkrun run\`). Accepts a registry recipe (\`recipeFile\`), an NFS path (\`recipePath\`), or an
  **inline recipe body** (\`recipeYaml\`) for remote recipe development. Stopped via DELETE.
- **Load Balancer** — round-robin inference proxy routing to running deployments.
- **Fine-tune** — training jobs (train → merge → quantize → deploy the merged model via sparkrun).
- **Benchmarks** — server-side benchmark runs (llama-benchy / tool-eval-bench) against a deployment.
- **Datasets / Models / Settings / Tokens / Agent bundle** — supporting resources.

Typical flow: register a Node → it provisions (incl. sparkrun) → pick a Recipe → create a
Deployment → route traffic through the Load Balancer → optionally Benchmark it.
`;

export function buildOpenApiSpec() {
  return swaggerJsdoc({
    definition: {
      openapi: "3.0.3",
      info: { title: "DGX Manager API", version: "1.0.0", description: SYSTEM_OVERVIEW },
      tags: [
        { name: "Nodes", description: "Register, audit, and provision DGX Spark nodes." },
        { name: "Recipes", description: "Inference recipe catalog from sparkrun registries." },
        { name: "Deployments", description: "Launch/stop inference workloads via sparkrun (registry ref, NFS path, or inline YAML)." },
        { name: "Load Balancer", description: "Round-robin inference proxy over running deployments." },
        { name: "Fine-tune", description: "Training jobs and deploying fine-tuned models." },
        { name: "Benchmarks", description: "Server-side benchmark runs against deployments." },
        { name: "Datasets", description: "Training/eval dataset management." },
        { name: "Models", description: "Model records." },
        { name: "Settings", description: "Server settings." },
        { name: "Tokens", description: "HF / API tokens." },
        { name: "Agent bundle", description: "Per-arch agent tarballs + install script." },
      ],
    },
    // JSDoc @openapi annotations live in the route files (Task 16)
    apis: ["packages/server/src/routes/*.ts", "packages/server/src/routes/*.js"],
  });
}

/** Mount GET /api/openapi.json (spec) and GET /api/docs (Swagger UI). */
export function mountOpenApi(app: Express) {
  const spec = buildOpenApiSpec();
  app.get("/api/openapi.json", (_req, res) => res.json(spec));
  app.use("/api/docs", swaggerUi.serve, swaggerUi.setup(spec));
}
```

Then in `packages/server/src/index.ts` call `mountOpenApi(app)` after the routers are mounted.

- [ ] **Step 5: Run test → PASS; full `npm test`; commit**
```bash
git add packages/server/src/openapi.ts packages/server/src/index.ts packages/server/package.json package-lock.json packages/server/src/__tests__/integration/openapi.routes.test.ts
git commit -m "feat(server): OpenAPI spec + Swagger UI with system overview (D8)"
```

### Task 16: Annotate every route with comprehensive `@openapi` JSDoc

**Files (modify — add JSDoc `@openapi` blocks above each handler):** all 13 routers —
`nodes.ts`, `models.ts`, `deployments.ts`, `finetune.ts`, `loadbalancer.ts`, `recipes.ts`,
`training-recipes.ts`, `tokens.ts`, `settings.ts`, `ollama-catalog.ts`, `agent-bundle.ts`,
`datasets.ts`, `benchmarks.ts`.
**Test:** extend `openapi.routes.test.ts`.

- [ ] **Step 1: Extend the failing test** — assert real operations are documented with descriptions

```ts
it("documents key operations with comprehensive descriptions + tags", async () => {
  const res = await request(app()).get("/api/openapi.json");
  const paths = res.body.paths ?? {};
  // deploy, list recipes, list nodes are all present and tagged + described
  const deploy = paths["/api/deployments"]?.post;
  expect(deploy).toBeTruthy();
  expect(deploy.tags).toContain("Deployments");
  expect((deploy.description ?? "").length).toBeGreaterThan(80); // comprehensive, not a stub
  expect(paths["/api/recipes"]?.get?.tags).toContain("Recipes");
  expect(paths["/api/nodes"]?.get?.tags).toContain("Nodes");
  // the inline-recipe body field is documented on the deploy operation
  expect(JSON.stringify(deploy.requestBody ?? {})).toMatch(/recipeYaml/);
});
```

- [ ] **Step 2: Run → FAIL** (operations not yet annotated).

- [ ] **Step 3: Annotate each route handler.** For every endpoint add a JSDoc block, e.g. above the deployments POST:

```ts
/**
 * @openapi
 * /api/deployments:
 *   post:
 *     tags: [Deployments]
 *     summary: Launch an inference deployment via sparkrun
 *     description: >
 *       Launches a recipe on one or more DGX Spark nodes. The head-node agent runs
 *       `sparkrun run`, which distributes the image+model and starts containers. Provide
 *       exactly one recipe source: `recipeFile` (a registry recipe from GET /api/recipes),
 *       `recipePath` (a YAML already staged under shared storage), or `recipeYaml` (an inline
 *       recipe body — used by remote recipe-development machines that never touch the cluster
 *       filesystem). VRAM admission runs before launch; returns the created Deployment record.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               nodeId: { type: string, description: "Single-node deploy target (or 'auto')." }
 *               nodeIds: { type: array, items: { type: string }, description: "Cluster node ids, head first." }
 *               recipeFile: { type: string, description: "Registry recipe ref from GET /api/recipes." }
 *               recipePath: { type: string, description: "Path under shared storage to a recipe YAML." }
 *               recipeYaml: { type: string, description: "Inline sparkrun recipe YAML body (remote dev)." }
 *               config: { type: object, description: "Overrides: port, gpuMem, tensorParallel, maxModelLen…" }
 *               displayName: { type: string }
 *     responses:
 *       200: { description: Created deployment record }
 *       400: { description: Invalid request (bad recipe source / VRAM shortfall) }
 */
```

Do the same for every operation in all 13 routers: a `tags`, a one-line `summary`, a multi-line
`description` explaining the endpoint's role in the dgx-manager workflow, and request/response
shapes. Keep descriptions comprehensive (an agent should understand the endpoint without source).

- [ ] **Step 4: Run the OpenAPI test → PASS; full `npm test`; commit**
```bash
git add packages/server/src/routes/ packages/server/src/__tests__/integration/openapi.routes.test.ts
git commit -m "docs(server): comprehensive @openapi annotations on all routes"
```

## Phase 6 — Retire eugr code + finalize

### Task 13: Remove dead eugr launch code

**Files:**
- Modify: `packages/agent/src/runtime/vllm.ts` (delete `launchRecipe`, `buildLaunchArgs`, `syncContainerImage`, `generateLocalModelRecipe`, `stopRecipe`/`reattachLogs` run-recipe.sh wiring no longer referenced)
- Modify: `packages/agent/src/runtime/vllm.test.ts` (remove tests for deleted functions; keep any still-valid metrics tests or move them to `sparkrun-metrics.test.ts`)

- [ ] **Step 1: Find remaining references**

Run: `grep -rn "run-recipe.sh\|launchRecipe\|buildLaunchArgs\|syncContainerImage\|generateLocalModelRecipe\|spark-vllm-docker\|VLLM_REPO" packages/`
Expected: only definitions (no live callers) remain — confirm before deleting.

- [ ] **Step 2: Delete the dead functions + their tests**

Remove the functions listed above and any now-unused imports/constants. Delete `vllm.test.ts` cases that exercised them.

- [ ] **Step 3: Build + full test run**

Run: `npm run build && npm test`
Expected: build clean; all tests green.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(agent): remove eugr run-recipe.sh launch path"
```

### Task 14: Bump agent version + final verification

**Files:**
- Modify: `packages/agent/package.json` (via script)

- [ ] **Step 1: Bump the agent version (MANDATORY — agent/src changed)**

Run: `./scripts/bump-agent-version.sh`
Expected: patch version incremented (e.g. 0.5.0 → 0.5.1).

- [ ] **Step 2: Full test suite green**

Run: `npm test`
Expected: PASS (all suites).

- [ ] **Step 3: Manual / real-DGX verification (environmental — document in PR)**

On the cluster: provision a node (confirm sparkrun installs + sets up), deploy a registry recipe solo, deploy a 2-node TP recipe, confirm dashboard shows metrics, `DELETE` to stop, restart the agent and confirm reconciliation marks the still-running workload `running`. Record results + sparkrun version in the PR description (per CLAUDE.md: environmental behavior that can't be unit-tested must be manually verified and described).

- [ ] **Step 4: Commit + open PR**

```bash
git add packages/agent/package.json
git commit -m "chore(agent): bump version for sparkrun deploy backend"
```

---

## Self-Review

**Spec coverage:**
- D1 replace eugr → Task 13 (retire) + Phase 2 (replacement). ✓
- D2 agent-side head-node launch → Tasks 4–8. ✓
- D3 provisioning installs sparkrun → Task 11. ✓
- D4 catalog from `sparkrun list`/`show` → Tasks 1–3. ✓
- D5 custom recipe by path, API-only, validated → Tasks 9–10. ✓
- D6 vLLM parity, runtime opaque → Task 7 (vLLM-shaped metrics), `runtime` carried but not branched. ✓
- V1–V5 → Phase 0 spike (Task 0) feeds fixtures into Tasks 1, 2, 5, 7 and the setup flag into Task 11. ✓
- Fine-tune deploy reroute (spec §4.5) → Task 12; training untouched. ✓
- Testing tiers (spec §7): property test (Task 4), path-validation unit (Task 9), parser fixtures (Tasks 1/2/5/7), integration happy/error (Task 10), admission regression (Task 10 step 5), manual (Task 14). ✓

**Placeholder scan:** The only intentionally-deferred concrete values are the sparkrun CLI field names (resolved by the Phase 0 fixtures, which the tests load) and the non-interactive setup flag in `sparkrunInstallCmd` (resolved by V1 findings). These are flagged at their use sites with the fixture/findings as the authority — not silent TODOs.

**Type consistency:** `SparkrunRecipeSummary` (Task 1) reused as `Recipe` (Task 3); `SparkrunLaunchOptions`/`buildSparkrunArgs` (Task 4) consumed by `launchSparkrun` (Task 6); `SparkrunWorkload` (Task 5) consumed by `listSparkrunWorkloads` (Task 6) and `reconcileDeployStatus` (Task 8); `resolveRecipePath` (Task 9) consumed by the route (Task 10); `SPARKRUN_PKG` defined in Task 3, used in Tasks 6 & 11. Consistent.
