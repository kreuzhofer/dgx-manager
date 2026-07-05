# In-repo `@dgxrun` Recipe Catalog — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make dgxrun recipes first-class and in-repo — served under an `@dgxrun/` namespace, grouped separately in the deploy dropdown, and deployable straight from the dropdown (selecting `@dgxrun/*` routes to the dgxrun runner).

**Architecture:** In-repo `recipes/dgxrun/*.yaml`; a pure server loader maps them to the `Recipe` wire shape with `source:"dgxrun"`; the recipes route merges them into `GET /api/recipes`; the deploy route resolves an `@dgxrun/` `recipeFile` to its in-repo YAML and feeds the existing `resolveDgxrunRecipe`; the dashboard groups the dropdown by `source`.

**Tech Stack:** TypeScript (strict, ESM), Node 22, Vitest + supertest, `yaml`, Next.js dashboard.

## Global Constraints

- TypeScript strict + ESM; `.js` import extensions in TS source.
- **No `packages/agent/src/` edits** — so **no agent version bump**. The `Recipe` type is imported **type-only** from the agent package (`import type { Recipe }`); the server defines `CatalogRecipe = Recipe & { source: "dgxrun" }`. Sparkrun recipes carry **no** `source` field; the dashboard treats absent `source` as the sparkrun group. (This is a deliberate refinement of the spec's "add `source` to the wire shape" — kept server-side to avoid an agent bump.)
- Pure functions take injected `readDir`/`readFile` (default real `fs`) so they unit-test without touching the real dir — same pattern as `dgxrun-args.ts` / `dgxrun-recipe.ts`.
- `resolveDgxrunRecipe(yamlText)` (existing, `deployments/dgxrun-recipe.ts`) is reused unchanged.
- Commit after every task, prefix `feat(dgxrun-catalog):`. `npm test` green + `npx tsc --noEmit -p packages/server/tsconfig.json` clean before each commit.
- Recipes dir at runtime: `DGXRUN_RECIPES_DIR` env, default `join(process.cwd(), "recipes/dgxrun")` (cwd is the repo root in dev and `/app` in the container).

## File structure

- `recipes/dgxrun/glm-5.2-awq-15pct.yaml` — migrated GLM-5.2 dgxrun recipe (first catalog entry).
- `packages/server/src/deployments/dgxrun-catalog.ts` — pure `loadDgxrunCatalog` + `resolveDgxrunRecipeFile`, plus a cached `getDgxrunCatalog`/`refreshDgxrunCatalog`.
- `packages/server/src/deployments/dgxrun-catalog.test.ts` — unit tests.
- `packages/server/src/routes/recipes.ts` — merge catalog into `GET /` + refresh on `POST /refresh`.
- `packages/server/src/routes/deployments.ts` — `@dgxrun/` recipeFile branch.
- `packages/server/src/__tests__/integration/dgxrun-catalog.test.ts` — HTTP integration.
- `Dockerfile.server` — `COPY recipes/ ./recipes/` so the catalog ships in the image.
- `packages/dashboard/app/page.tsx` (or the deploy form component) — group the recipe `<select>` by `source`.

---

### Task 1: Pure catalog loader — `loadDgxrunCatalog`

**Files:**
- Create: `packages/server/src/deployments/dgxrun-catalog.ts`
- Test: `packages/server/src/deployments/dgxrun-catalog.test.ts`

**Interfaces:**
- Consumes: `import type { Recipe } from "../../../agent/src/recipes.js"` (type-only; confirm the exact relative path resolves from `deployments/` — adjust if the server already re-exports `Recipe`).
- Produces: `export type CatalogRecipe = Recipe & { source: "dgxrun" }` and
  `export function loadDgxrunCatalog(dir: string, deps?: { readDir?: (d: string) => string[]; readFile?: (p: string) => string }): CatalogRecipe[]`.
  Maps each `*.yaml` in `dir` (parsed via `yaml`) to a `CatalogRecipe`: `file: "@dgxrun/<basename-no-ext>"`, `name` (recipe `name` or the basename), `model`, `description`, `container: "dgxrun"`, `source: "dgxrun"`, `arch: "arm64"`, `cluster_only: true`, `defaults: { tensor_parallel, gpu_memory_utilization, port, max_model_len }` pulled from the recipe's `defaults` (fallbacks: tp 4, gpumem 0.85, port 8000, max_model_len ""). A file that fails to parse or lacks `runner: dgxrun` is **skipped with `console.warn`**; other files still load. Missing dir → `[]`.

- [ ] **Step 1: Write the failing test**
```ts
import { describe, it, expect } from "vitest";
import { loadDgxrunCatalog } from "./dgxrun-catalog.js";

const VALID = `runner: dgxrun
model: CosmicRaisins/GLM-5.2-AWQ-INT4-15pct
container: vllm-node-tf5-glm52-b12x:probe
cluster_only: true
defaults:
  tensor_parallel: 4
  gpu_memory_utilization: 0.88
  port: 8000
  max_model_len: 87040
command: vllm serve {model}`;

describe("loadDgxrunCatalog", () => {
  const deps = (files: Record<string, string>) => ({
    readDir: () => Object.keys(files),
    readFile: (p: string) => files[p.split("/").pop()!],
  });
  it("maps a valid dgxrun yaml to a CatalogRecipe under @dgxrun/", () => {
    const r = loadDgxrunCatalog("/recipes/dgxrun", deps({ "glm-5.2-awq-15pct.yaml": VALID }));
    expect(r).toHaveLength(1);
    expect(r[0].file).toBe("@dgxrun/glm-5.2-awq-15pct");
    expect(r[0].source).toBe("dgxrun");
    expect(r[0].container).toBe("dgxrun");
    expect(r[0].cluster_only).toBe(true);
    expect(r[0].defaults.tensor_parallel).toBe(4);
    expect(r[0].defaults.max_model_len).toBe(87040);
  });
  it("skips a malformed file but keeps the good ones", () => {
    const r = loadDgxrunCatalog("/d", deps({ "bad.yaml": ": not: yaml:", "ok.yaml": VALID }));
    expect(r.map((x) => x.file)).toEqual(["@dgxrun/ok"]);
  });
  it("skips a yaml without runner: dgxrun", () => {
    const r = loadDgxrunCatalog("/d", deps({ "spark.yaml": "container: foo\ncommand: bar" }));
    expect(r).toEqual([]);
  });
  it("missing dir -> []", () => {
    const r = loadDgxrunCatalog("/nope", { readDir: () => { throw new Error("ENOENT"); }, readFile: () => "" });
    expect(r).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**
Run: `npx vitest run packages/server/src/deployments/dgxrun-catalog.test.ts`
Expected: FAIL — `loadDgxrunCatalog` not exported.

- [ ] **Step 3: Write minimal implementation**
```ts
import { readdirSync, readFileSync } from "fs";
import { join } from "path";
import { parse } from "yaml";
import type { Recipe } from "../../../agent/src/recipes.js";

export type CatalogRecipe = Recipe & { source: "dgxrun" };

interface CatalogDeps { readDir?: (d: string) => string[]; readFile?: (p: string) => string; }

export function loadDgxrunCatalog(dir: string, deps: CatalogDeps = {}): CatalogRecipe[] {
  const readDir = deps.readDir ?? ((d) => readdirSync(d));
  const readFile = deps.readFile ?? ((p) => readFileSync(p, "utf-8"));
  let files: string[];
  try { files = readDir(dir).filter((f) => f.endsWith(".yaml") || f.endsWith(".yml")); }
  catch { return []; }
  const out: CatalogRecipe[] = [];
  for (const f of files.sort()) {
    const base = f.replace(/\.ya?ml$/, "");
    let doc: unknown;
    try { doc = parse(readFile(join(dir, f))); }
    catch (e) { console.warn(`[dgxrun-catalog] skip ${f}: parse error ${(e as Error).message}`); continue; }
    if (doc == null || typeof doc !== "object" || Array.isArray(doc)) { console.warn(`[dgxrun-catalog] skip ${f}: not a mapping`); continue; }
    const o = doc as Record<string, unknown>;
    if (o.runner !== "dgxrun") { console.warn(`[dgxrun-catalog] skip ${f}: not runner:dgxrun`); continue; }
    const d = (o.defaults && typeof o.defaults === "object" ? o.defaults : {}) as Record<string, unknown>;
    out.push({
      file: `@dgxrun/${base}`,
      name: typeof o.name === "string" ? o.name : base,
      description: typeof o.description === "string" ? o.description : undefined,
      model: typeof o.model === "string" ? o.model : undefined,
      container: "dgxrun",
      source: "dgxrun",
      arch: "arm64",
      cluster_only: true,
      defaults: {
        tensor_parallel: d.tensor_parallel ?? 4,
        gpu_memory_utilization: d.gpu_memory_utilization ?? 0.85,
        port: d.port ?? 8000,
        max_model_len: d.max_model_len ?? "",
      },
    });
  }
  return out;
}
```
Note: if `import type { Recipe }` from that relative path doesn't resolve under the server tsconfig, define a local structural `Recipe`-compatible interface in this file instead (still no agent edit).

- [ ] **Step 4: Run test to verify it passes**
Run: `npx vitest run packages/server/src/deployments/dgxrun-catalog.test.ts` — Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add packages/server/src/deployments/dgxrun-catalog.ts packages/server/src/deployments/dgxrun-catalog.test.ts
git commit -m "feat(dgxrun-catalog): pure loadDgxrunCatalog dir -> CatalogRecipe[]"
```

---

### Task 2: `@dgxrun/` recipeFile → in-repo path resolver (with traversal rejection)

**Files:**
- Modify: `packages/server/src/deployments/dgxrun-catalog.ts`
- Test: `packages/server/src/deployments/dgxrun-catalog.test.ts`

**Interfaces:**
- Produces: `export function resolveDgxrunRecipeFile(recipeFile: string, dir: string): string | null`.
  Returns the absolute path `join(dir, "<name>.yaml")` for a `recipeFile` of the form `@dgxrun/<name>`, or `null` if it doesn't start with `@dgxrun/` or `<name>` contains a path separator / `..` / is empty (security boundary — mirrors `resolveRecipePath`).

- [ ] **Step 1: Write the failing test (append)**
```ts
import { resolveDgxrunRecipeFile } from "./dgxrun-catalog.js";
describe("resolveDgxrunRecipeFile", () => {
  it("maps @dgxrun/<name> to <dir>/<name>.yaml", () => {
    expect(resolveDgxrunRecipeFile("@dgxrun/glm-5.2-awq-15pct", "/app/recipes/dgxrun"))
      .toBe("/app/recipes/dgxrun/glm-5.2-awq-15pct.yaml");
  });
  it("rejects non-@dgxrun refs", () => {
    expect(resolveDgxrunRecipeFile("@community/foo", "/d")).toBeNull();
    expect(resolveDgxrunRecipeFile("plain", "/d")).toBeNull();
  });
  it("rejects path traversal / separators", () => {
    expect(resolveDgxrunRecipeFile("@dgxrun/../../etc/passwd", "/d")).toBeNull();
    expect(resolveDgxrunRecipeFile("@dgxrun/sub/evil", "/d")).toBeNull();
    expect(resolveDgxrunRecipeFile("@dgxrun/", "/d")).toBeNull();
  });
});
```

- [ ] **Step 2: Run** — FAIL (`resolveDgxrunRecipeFile` undefined).

- [ ] **Step 3: Implement (append to dgxrun-catalog.ts)**
```ts
const DGXRUN_PREFIX = "@dgxrun/";
export function resolveDgxrunRecipeFile(recipeFile: string, dir: string): string | null {
  if (typeof recipeFile !== "string" || !recipeFile.startsWith(DGXRUN_PREFIX)) return null;
  const name = recipeFile.slice(DGXRUN_PREFIX.length);
  if (!name || name.includes("/") || name.includes("\\") || name.includes("..")) return null;
  return join(dir, `${name}.yaml`);
}
```

- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(dgxrun-catalog): resolveDgxrunRecipeFile with traversal rejection"`

---

### Task 3: Migrate GLM-5.2 recipe + serve the catalog in `GET /api/recipes`

**Files:**
- Create: `recipes/dgxrun/glm-5.2-awq-15pct.yaml`
- Modify: `packages/server/src/deployments/dgxrun-catalog.ts` (add cached `getDgxrunCatalog`/`refreshDgxrunCatalog` + `DGXRUN_RECIPES_DIR`)
- Modify: `packages/server/src/routes/recipes.ts` (merge + refresh)
- Modify: `Dockerfile.server` (`COPY recipes/ ./recipes/`)
- Test: `packages/server/src/__tests__/integration/dgxrun-catalog.test.ts`

**Interfaces:**
- Consumes: `loadDgxrunCatalog` (Task 1).
- Produces: `export const DGXRUN_RECIPES_DIR = process.env.DGXRUN_RECIPES_DIR || join(process.cwd(), "recipes/dgxrun")`, `export function getDgxrunCatalog(): CatalogRecipe[]` (memoized), `export function refreshDgxrunCatalog(): void` (clears the memo).

- [ ] **Step 1: Create the recipe file** `recipes/dgxrun/glm-5.2-awq-15pct.yaml` — copy the validated dgxrun recipe verbatim from `/tmp/claude-1000/-home-daniel-src-github-kreuzhofer-dgx-manager/25f6b78a-e34f-4e00-b896-5a0f8a4f48fd/scratchpad/glm52-dgxrun.yaml` (the full 71-line body: `runner: dgxrun`, `maxoutmem: true`, `model`, `container: vllm-node-tf5-glm52-b12x:probe`, `cluster_only: true`, the `defaults` block with `max_model_len: 87040` / `gpu_memory_utilization: 0.88`, the full `env` NCCL/RDMA + HF-offline stanza, and the `command:` vLLM serve line). Add a top comment: `# @dgxrun/glm-5.2-awq-15pct — in-repo dgxrun catalog (supersedes the community-registry copy for dgxrun deploys).`

- [ ] **Step 2: Add cached accessors to `dgxrun-catalog.ts`**
```ts
export const DGXRUN_RECIPES_DIR = process.env.DGXRUN_RECIPES_DIR || join(process.cwd(), "recipes/dgxrun");
let _cache: CatalogRecipe[] | null = null;
export function getDgxrunCatalog(): CatalogRecipe[] {
  if (_cache == null) _cache = loadDgxrunCatalog(DGXRUN_RECIPES_DIR);
  return _cache;
}
export function refreshDgxrunCatalog(): void { _cache = null; }
```

- [ ] **Step 3: Write the failing integration test**
```ts
// mirror packages/server/src/__tests__/integration/deployments.dgxrun.test.ts bootstrap
// (per-suite sqlite via mkdtempSync + DATABASE_URL before importing prisma, prisma db push)
import request from "supertest";
// ... set DGXRUN_RECIPES_DIR to a temp dir containing glm-5.2-awq-15pct.yaml BEFORE importing the app,
//     OR point it at the repo's real recipes/dgxrun. Mount recipesRouter with a stub agentHub whose
//     getRecipes() returns [] so only the catalog shows.
describe("GET /api/recipes includes @dgxrun catalog", () => {
  it("lists the @dgxrun/glm-5.2-awq-15pct recipe with source dgxrun", async () => {
    const res = await request(app).get("/api/recipes");
    expect(res.status).toBe(200);
    const hit = res.body.find((r: any) => r.file === "@dgxrun/glm-5.2-awq-15pct");
    expect(hit).toBeTruthy();
    expect(hit.source).toBe("dgxrun");
  });
});
```

- [ ] **Step 4: Run** — FAIL (route doesn't merge the catalog).

- [ ] **Step 5: Merge into `recipes.ts`** — in `GET /`, build `const recipes = [...getDgxrunCatalog(), ...agentHub.getRecipes()];` (dgxrun first) and keep the existing arch-filter logic operating on `recipes`. In `POST /refresh`, call `refreshDgxrunCatalog()` before/alongside the agent rescan.

- [ ] **Step 6: `Dockerfile.server`** — add `COPY recipes/ ./recipes/` (near the other COPY lines, before the CMD) so `recipes/dgxrun/*.yaml` ships in the image at `/app/recipes/dgxrun`.

- [ ] **Step 7: Run** — `npx vitest run packages/server/src/__tests__/integration/dgxrun-catalog.test.ts` PASS; `npm test` green; `tsc --noEmit -p packages/server/tsconfig.json` clean.

- [ ] **Step 8: Commit** — `git add -A && git commit -m "feat(dgxrun-catalog): migrate GLM-5.2 + serve @dgxrun in /api/recipes"`

---

### Task 4: Deploy route — `@dgxrun/` recipeFile routes to dgxrun

**Files:**
- Modify: `packages/server/src/routes/deployments.ts` (the "Resolve dgxrun runner" block, ~lines 177-192)
- Test: `packages/server/src/__tests__/integration/dgxrun-catalog.test.ts` (add a case)

**Interfaces:**
- Consumes: `resolveDgxrunRecipeFile`, `DGXRUN_RECIPES_DIR` (Tasks 2-3); existing `resolveDgxrunRecipe`, `readFileSync`.
- Produces: no new export — a `POST /api/deployments { recipeFile: "@dgxrun/<name>" }` now resolves `isDgxrun=true` from the in-repo YAML.

- [ ] **Step 1: Write the failing integration test (add to the suite)**
```ts
describe("POST /api/deployments with @dgxrun recipeFile", () => {
  it("routes an @dgxrun/ recipeFile to the dgxrun runner", async () => {
    // seed two+ nodes for a cluster deploy; stub agentHub.sendToAgent to capture the fan-out
    const res = await request(app).post("/api/deployments")
      .send({ recipeFile: "@dgxrun/glm-5.2-awq-15pct", nodeIds: [nodeA, nodeB, nodeC, nodeD] });
    expect([200, 201]).toContain(res.status);
    // assert dgxrun path was taken (e.g. deployment.runner/kind === dgxrun, or the captured
    // cmd:deploy payloads carry rank/master-addr from buildDgxrunDeploys)
    expect(res.body.deployment?.isDgxrun ?? res.body.isDgxrun ?? true).toBeTruthy();
  });
});
```
(Adjust the assertion to whatever the existing `deployments.dgxrun.test.ts` checks for a dgxrun deploy — reuse its harness + assertions.)

- [ ] **Step 2: Run** — FAIL (a `@dgxrun/` recipeFile currently falls through to the sparkrun path; `isDgxrun` stays false).

- [ ] **Step 3: Implement** — in the dgxrun-resolution block of `deployments.ts`, extend how `recipeText` is obtained:
```ts
    let recipeText: string | undefined;
    if (inlineRecipeYaml) {
      recipeText = inlineRecipeYaml;
    } else if (recipePath && recipeRef) {
      try { recipeText = readFileSync(recipeRef, "utf8"); } catch { /* agent will read the file */ }
    } else if (recipeFile) {
      const p = resolveDgxrunRecipeFile(recipeFile, DGXRUN_RECIPES_DIR);
      if (p) {
        try { recipeText = readFileSync(p, "utf8"); }
        catch { return res.status(404).json({ error: `dgxrun recipe not found: ${recipeFile}` }); }
      }
    }
```
Add the imports: `import { resolveDgxrunRecipeFile, DGXRUN_RECIPES_DIR } from "../deployments/dgxrun-catalog.js";`. A non-`@dgxrun/` `recipeFile` yields `p === null` → `recipeText` stays undefined → sparkrun path unchanged.

- [ ] **Step 4: Run** — PASS; `npm test` green; `tsc` clean.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(dgxrun-catalog): route @dgxrun/ recipeFile to the dgxrun runner"`

---

### Task 5: Dashboard — group the deploy dropdown by `source`

**Files:**
- Modify: the deploy form recipe `<select>` (in `packages/dashboard/app/page.tsx` or its deploy component — grep for the recipe select that maps `/api/recipes`).

**Interfaces:**
- Consumes: `GET /api/recipes` entries; `r.source === "dgxrun"` marks dgxrun recipes (absent → sparkrun).

- [ ] **Step 1: Locate the select** — `grep -rn "recipes.map\|<option" packages/dashboard/app packages/dashboard/components | grep -i recipe`. Confirm the array is the `/api/recipes` result.

- [ ] **Step 2: Render two `<optgroup>`s** — split the recipes into `dgxrun = recipes.filter(r => r.source === "dgxrun")` and `spark = recipes.filter(r => r.source !== "dgxrun")`, then render:
```tsx
{dgxrun.length > 0 && (
  <optgroup label="dgxrun (multi-node)">
    {dgxrun.map((r) => <option key={r.file} value={r.file}>{r.name}</option>)}
  </optgroup>
)}
<optgroup label="sparkrun">
  {spark.map((r) => <option key={r.file} value={r.file}>{r.name}</option>)}
</optgroup>
```
Match the existing option rendering (label field, key, any arch/disabled logic already present — preserve it inside each group).

- [ ] **Step 3: Verify** — `npm run build` (dashboard) compiles; the selected value is still the recipe `file` (`@dgxrun/...`), so no change to the POST body. Manual note: dgxrun group appears above sparkrun.

- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat(dgxrun-catalog): group deploy dropdown into dgxrun + sparkrun"`

---

## Self-review (author checklist — completed)

- **Spec coverage:** in-repo dir + migrated GLM (T3) ✓; pure loader with injected fs (T1) ✓; `@dgxrun/` routing + traversal rejection (T2, T4) ✓; merge into `/api/recipes` + refresh (T3) ✓; dropdown grouping (T5) ✓; reuse `resolveDgxrunRecipe`/dgxrun dispatch unchanged (T4) ✓; Dockerfile ships the catalog (T3) ✓.
- **`source` decision:** kept server-side (`CatalogRecipe = Recipe & {source}`), agent untouched → no agent bump (documented in Global Constraints). Sparkrun recipes omit `source`; dashboard defaults absent→sparkrun (T5 filter uses `!== "dgxrun"`).
- **Placeholder scan:** all steps carry real code/paths; the one external artifact (the 71-line GLM YAML) is referenced by exact scratchpad path to copy verbatim.
- **Type consistency:** `CatalogRecipe`/`loadDgxrunCatalog` (T1) → `getDgxrunCatalog` (T3) → recipes route; `resolveDgxrunRecipeFile`/`DGXRUN_RECIPES_DIR` (T2-3) → deploy route (T4). Names align.
