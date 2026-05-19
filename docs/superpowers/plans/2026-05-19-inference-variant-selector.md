# Inference Variant Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let operators pick which inference template a fine-tune deployment uses, populated dynamically from the recipe directory (one entry per `inference*.yaml` file) — replacing the hardcoded bf16/fp8 binary.

**Architecture:** Pure file-system convention on the recipe side: any `inference-<id>.yaml` next to `recipe.yaml` becomes a selectable variant, with `inference.yaml` mapping to the special id `default`. The agent enumerates the files and forwards a `{id, name, description}` list to the manager via the existing training-recipes channel; the manager stores the chosen id verbatim in `Deployment.config.artifactVariant`; the dashboard renders a `<select>` on both the deploy and edit-restart forms. Back-compat: legacy `"bf16"` and `"fp8"` ids still resolve to the correct templates.

**Tech Stack:** TypeScript (Node 22 agent + Express 5 server + Next.js 15 dashboard), Vitest + supertest + fast-check, Tailwind 4, Prisma + SQLite (no schema change).

---

## Decisions locked in during brainstorming

- **Variant id = filename slug.** Strip `inference-` prefix and `.yaml` suffix. Bare `inference.yaml` → `default`. (`inference-fp8.yaml` → `fp8`, `inference-int4.yaml` → `int4`, etc.)
- **Label + description come from the YAML's own `name:` and `description:` head fields.** No new metadata on the recipe side. Both qwen3.6-27b templates already have them.
- **Auto-select when only one variant exists.** No "Default" picker noise for that case.
- **Manager doesn't bake precision names anywhere.** No `"bf16"|"fp8"` literal types; variants are opaque strings derived from filenames.
- **Legacy values are aliases.** `"bf16"` is treated identically to `"default"`; `"fp8"` stays `"fp8"`. Pre-fix deployments keep working without a data migration.

---

## File Structure

**Created files (none — the work is additive to existing modules):**

(no new source files)

**Created test files:**

| Path | Responsibility |
|---|---|
| `packages/agent/src/runtime/inference-variants.test.ts` | Unit tests for `inferenceVariantIdFromFilename`, `inferenceFilenameForId`, `listInferenceVariants` |
| `packages/server/src/__tests__/integration/deployments.restart-variant.test.ts` | Integration test: restart with `artifactVariant: "fp8"` (and a custom id) propagates correctly |

**Modified files:**

| Path | Change |
|---|---|
| `packages/agent/src/runtime/inference-template.ts` | Add `inferenceVariantIdFromFilename`, `inferenceFilenameForId`, `listInferenceVariants` helpers. Make `findInferenceTemplate` accept arbitrary variant id strings (back-compat with `"bf16"`/`"fp8"` literals). |
| `packages/agent/src/runtime/inference-template.test.ts` | Update the existing variant-arg tests to reflect the relaxed type + add a `"default"` case. |
| `packages/agent/src/training-recipes.ts` | Extend `TrainingRecipe` interface with `inferenceVariants?: InferenceVariant[]`. Populate in `discoverTrainingRecipes()` by calling `listInferenceVariants(recipeDir)`. |
| `packages/server/src/ws/agent-hub.ts` | Mirror the `TrainingRecipe.inferenceVariants` field on the server-side type (kept in sync with the agent type by convention). |
| `packages/server/src/routes/finetune.ts` | Relax `artifactVariant` type from `"bf16" \| "fp8"` to `string` in the POST `/deploy` body; basic slug validation. |
| `packages/server/src/routes/deployments.ts` | Relax `artifactVariant` type read in the restart route. No behavior change (already passes through to the agent). |
| `packages/agent/package.json` | Patch version bump via `./scripts/bump-agent-version.sh`. |
| `packages/dashboard/app/deployments/page.tsx` | Update the `Recipe` TS interface to include `inferenceVariants`. Add a `<select>` to both the fine-tune deploy form and the edit-restart form. Auto-select the only variant when exactly one is present. |

---

## Type definitions used across tasks

The two interfaces are declared in Task 1 and reused everywhere after. Repeated here for orientation:

```ts
// packages/agent/src/runtime/inference-template.ts
export interface InferenceVariant {
  /** Slug derived from filename. `inference.yaml` → "default";
   *  `inference-fp8.yaml` → "fp8". Used as the wire id everywhere. */
  id: string;
  /** Filename relative to the recipe dir. */
  filename: string;
  /** From the YAML's top-level `name:` field. */
  name: string;
  /** From the YAML's top-level `description:` field. Optional. */
  description?: string;
}
```

```ts
// packages/agent/src/training-recipes.ts (added to existing TrainingRecipe)
inferenceVariants?: InferenceVariant[];
```

---

## Task 1: Pure helpers — variant id ↔ filename mapping

**Files:**
- Modify: `packages/agent/src/runtime/inference-template.ts`
- Create: `packages/agent/src/runtime/inference-variants.test.ts`

**Why:** The id↔filename mapping is the foundation everything else builds on. Pure functions ⇒ exhaustively testable, no fs needed.

- [ ] **Step 1: Write the failing test**

Create `packages/agent/src/runtime/inference-variants.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import {
  inferenceVariantIdFromFilename,
  inferenceFilenameForId,
} from "./inference-template.js";

describe("inferenceVariantIdFromFilename", () => {
  it("maps the bare inference.yaml to the special id 'default'", () => {
    expect(inferenceVariantIdFromFilename("inference.yaml")).toBe("default");
  });

  it("strips the inference- prefix and .yaml suffix", () => {
    expect(inferenceVariantIdFromFilename("inference-fp8.yaml")).toBe("fp8");
    expect(inferenceVariantIdFromFilename("inference-int4.yaml")).toBe("int4");
    expect(inferenceVariantIdFromFilename("inference-low-ctx.yaml")).toBe("low-ctx");
  });

  it("returns null for filenames that don't fit the convention", () => {
    expect(inferenceVariantIdFromFilename("recipe.yaml")).toBeNull();
    expect(inferenceVariantIdFromFilename("inference.yml")).toBeNull(); // .yml not .yaml
    expect(inferenceVariantIdFromFilename("not-inference-fp8.yaml")).toBeNull();
    expect(inferenceVariantIdFromFilename("inference-.yaml")).toBeNull(); // empty slug
  });
});

describe("inferenceFilenameForId", () => {
  it("maps 'default' back to inference.yaml", () => {
    expect(inferenceFilenameForId("default")).toBe("inference.yaml");
  });

  it("maps legacy 'bf16' back to inference.yaml for back-compat", () => {
    // Saved deployments before this feature stored "bf16" — keep them working.
    expect(inferenceFilenameForId("bf16")).toBe("inference.yaml");
  });

  it("maps any other id to inference-<id>.yaml", () => {
    expect(inferenceFilenameForId("fp8")).toBe("inference-fp8.yaml");
    expect(inferenceFilenameForId("int4")).toBe("inference-int4.yaml");
    expect(inferenceFilenameForId("low-ctx")).toBe("inference-low-ctx.yaml");
  });

  // Invariant: for any slug we'd derive from a real filename, the round-trip
  // back through inferenceFilenameForId returns the original filename.
  test.prop([
    fc.stringMatching(/^[a-z0-9][a-z0-9-]{0,30}$/),
  ])("round-trips filename → id → filename for any plausible slug", (slug) => {
    const filename = slug === "default" ? "inference.yaml" : `inference-${slug}.yaml`;
    const id = inferenceVariantIdFromFilename(filename);
    expect(id).toBe(slug);
    expect(inferenceFilenameForId(id!)).toBe(filename);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/inference-variants.test.ts`
Expected: FAIL with import errors — `inferenceVariantIdFromFilename` and `inferenceFilenameForId` not exported.

- [ ] **Step 3: Implement the helpers**

In `packages/agent/src/runtime/inference-template.ts`, add **above** the existing `findInferenceTemplate` export (and after `applyFinetuneSubstitutions`):

```ts
/**
 * Convert an inference filename into its variant id slug.
 *
 *   "inference.yaml"       → "default"
 *   "inference-fp8.yaml"   → "fp8"
 *   "inference-low-ctx.yaml" → "low-ctx"
 *
 * Returns null for filenames that don't match the convention (so callers can
 * skip non-inference files during a directory scan).
 */
export function inferenceVariantIdFromFilename(filename: string): string | null {
  if (filename === "inference.yaml") return "default";
  const m = filename.match(/^inference-([a-z0-9][a-z0-9-]*)\.yaml$/);
  return m ? m[1] : null;
}

/**
 * Convert a variant id back into the filename to look up in a recipe dir.
 * Legacy back-compat: "bf16" is treated as an alias for "default" so saved
 * deployments from before this feature keep resolving to inference.yaml.
 */
export function inferenceFilenameForId(id: string): string {
  if (id === "default" || id === "bf16") return "inference.yaml";
  return `inference-${id}.yaml`;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/runtime/inference-variants.test.ts`
Expected: PASS — 3 unit cases + 1 property.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/inference-template.ts \
        packages/agent/src/runtime/inference-variants.test.ts
git commit -m "agent: pure helpers for inference variant id ↔ filename"
```

---

## Task 2: `listInferenceVariants(recipeDir)` directory scan

**Files:**
- Modify: `packages/agent/src/runtime/inference-template.ts`
- Modify: `packages/agent/src/runtime/inference-variants.test.ts`

**Why:** Pull together filesystem enumeration + YAML head parsing. Each `inference*.yaml` file becomes an `InferenceVariant` with id, filename, name, description.

- [ ] **Step 1: Add the failing test**

Append to `packages/agent/src/runtime/inference-variants.test.ts`:

```ts
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { listInferenceVariants } from "./inference-template.js";

describe("listInferenceVariants", () => {
  it("returns [] for a recipe dir with no inference templates", () => {
    const dir = mkdtempSync(join(tmpdir(), "variants-"));
    try {
      writeFileSync(join(dir, "recipe.yaml"), "name: x\n");
      expect(listInferenceVariants(dir)).toEqual([]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("returns a single variant when only inference.yaml exists", () => {
    const dir = mkdtempSync(join(tmpdir(), "variants-"));
    try {
      writeFileSync(join(dir, "inference.yaml"),
        `name: my-recipe-bf16\ndescription: Default BF16 serve.\nmodel: x\n`);
      const out = listInferenceVariants(dir);
      expect(out).toEqual([{
        id: "default",
        filename: "inference.yaml",
        name: "my-recipe-bf16",
        description: "Default BF16 serve.",
      }]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("enumerates multiple variants sorted with 'default' first then alphabetical", () => {
    const dir = mkdtempSync(join(tmpdir(), "variants-"));
    try {
      writeFileSync(join(dir, "inference.yaml"),         `name: r-bf16\ndescription: bf16.\n`);
      writeFileSync(join(dir, "inference-fp8.yaml"),     `name: r-fp8\ndescription: fp8 on-load.\n`);
      writeFileSync(join(dir, "inference-int4.yaml"),    `name: r-int4\ndescription: int4 awq.\n`);
      writeFileSync(join(dir, "recipe.yaml"),            `name: r-train\n`);
      writeFileSync(join(dir, "not-an-inference.txt"),   `noise`);

      const out = listInferenceVariants(dir);
      expect(out.map((v) => v.id)).toEqual(["default", "fp8", "int4"]);
      expect(out[1].filename).toBe("inference-fp8.yaml");
      expect(out[1].name).toBe("r-fp8");
      expect(out[2].description).toBe("int4 awq.");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("omits description when the YAML doesn't declare one", () => {
    const dir = mkdtempSync(join(tmpdir(), "variants-"));
    try {
      writeFileSync(join(dir, "inference.yaml"), `name: bare\n`);
      const out = listInferenceVariants(dir);
      expect(out[0].name).toBe("bare");
      expect(out[0].description).toBeUndefined();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("falls back to the variant id when name: is missing from the YAML", () => {
    // A malformed template still appears in the list; name defaults to the id
    // so the UI can render something instead of crashing.
    const dir = mkdtempSync(join(tmpdir(), "variants-"));
    try {
      writeFileSync(join(dir, "inference-fp8.yaml"), `model: x\n`);
      const out = listInferenceVariants(dir);
      expect(out).toEqual([{
        id: "fp8",
        filename: "inference-fp8.yaml",
        name: "fp8",
        description: undefined,
      }]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/inference-variants.test.ts`
Expected: FAIL — `listInferenceVariants` not exported, and the `InferenceVariant` interface not defined.

- [ ] **Step 3: Implement the helper**

Append to `packages/agent/src/runtime/inference-template.ts`:

```ts
import { readdirSync, readFileSync } from "fs";

export interface InferenceVariant {
  /** Slug derived from filename. `inference.yaml` → "default";
   *  `inference-fp8.yaml` → "fp8". Used as the wire id everywhere. */
  id: string;
  /** Filename relative to the recipe dir. */
  filename: string;
  /** From the YAML's top-level `name:` field. Falls back to the id when
   *  the file doesn't declare one (malformed templates still appear). */
  name: string;
  /** From the YAML's top-level `description:` field. Optional. */
  description?: string;
}

/**
 * Enumerate every inference template in a training-recipe dir. Each
 * `inference*.yaml` file becomes one entry. Reads `name:` and
 * `description:` from each YAML's top of file using a lightweight
 * regex — we don't import a YAML parser for two fields. The list is
 * sorted with "default" first, then alphabetical by id, so the UI
 * order is deterministic without the dashboard having to re-sort.
 */
export function listInferenceVariants(recipeDir: string): InferenceVariant[] {
  let entries: string[];
  try {
    entries = readdirSync(recipeDir);
  } catch {
    return [];
  }
  const out: InferenceVariant[] = [];
  for (const filename of entries) {
    const id = inferenceVariantIdFromFilename(filename);
    if (!id) continue;
    const full = join(recipeDir, filename);
    let text = "";
    try { text = readFileSync(full, "utf-8"); } catch { /* fall through */ }
    const nameMatch = text.match(/^name:\s*(.+?)\s*$/m);
    const descMatch = text.match(/^description:\s*(.+?)\s*$/m);
    out.push({
      id,
      filename,
      name: nameMatch ? stripYamlQuotes(nameMatch[1]) : id,
      description: descMatch ? stripYamlQuotes(descMatch[1]) : undefined,
    });
  }
  out.sort((a, b) => {
    if (a.id === "default") return -1;
    if (b.id === "default") return 1;
    return a.id.localeCompare(b.id);
  });
  return out;
}

function stripYamlQuotes(v: string): string {
  if ((v.startsWith('"') && v.endsWith('"')) ||
      (v.startsWith("'") && v.endsWith("'"))) {
    return v.slice(1, -1);
  }
  return v;
}
```

- [ ] **Step 4: Run to verify all variant tests pass**

Run: `npx vitest run packages/agent/src/runtime/inference-variants.test.ts`
Expected: PASS — 5 listInferenceVariants cases + the 4 from Task 1.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/inference-template.ts \
        packages/agent/src/runtime/inference-variants.test.ts
git commit -m "agent: listInferenceVariants enumerates inference templates per recipe"
```

---

## Task 3: Relax `findInferenceTemplate` to accept any variant id

**Files:**
- Modify: `packages/agent/src/runtime/inference-template.ts:62-69`
- Modify: `packages/agent/src/runtime/inference-template.test.ts`

**Why:** The existing function is typed `variant: "bf16" | "fp8" = "bf16"`. Now that variant ids are open-ended strings, the type must relax — but the runtime must keep accepting legacy "bf16" and "fp8" so old saved configs don't break.

- [ ] **Step 1: Write the failing test**

Append to `packages/agent/src/runtime/inference-template.test.ts`:

```ts
describe("findInferenceTemplate — variant id back-compat", () => {
  it("'default' resolves to inference.yaml (new canonical id)", () => {
    const dir = mkdtempSync(join(tmpdir(), "find-"));
    try {
      writeFileSync(join(dir, "inference.yaml"), "name: x");
      expect(findInferenceTemplate(dir, "default")).toBe(join(dir, "inference.yaml"));
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  it("'bf16' still resolves to inference.yaml (legacy alias)", () => {
    const dir = mkdtempSync(join(tmpdir(), "find-"));
    try {
      writeFileSync(join(dir, "inference.yaml"), "name: x");
      expect(findInferenceTemplate(dir, "bf16")).toBe(join(dir, "inference.yaml"));
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  it("arbitrary slug resolves to inference-<slug>.yaml", () => {
    const dir = mkdtempSync(join(tmpdir(), "find-"));
    try {
      writeFileSync(join(dir, "inference-int4.yaml"), "name: x");
      expect(findInferenceTemplate(dir, "int4")).toBe(join(dir, "inference-int4.yaml"));
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/inference-template.test.ts`
Expected: FAIL — TypeScript rejects `"default"` and `"int4"` because the parameter type is `"bf16" | "fp8"`.

- [ ] **Step 3: Implement — replace the existing function body**

In `packages/agent/src/runtime/inference-template.ts`, replace the `findInferenceTemplate` function (currently lines 62–69) with:

```ts
/**
 * Return the absolute path to the inference template for a given variant
 * id, or null if no template exists for it. The id is an opaque slug —
 * typically derived from the filename via `inferenceVariantIdFromFilename`.
 * Legacy ids `"bf16"` and `"fp8"` keep resolving so pre-feature deployments
 * still launch correctly after a restart.
 */
export function findInferenceTemplate(
  recipeDir: string,
  variant: string = "default",
): string | null {
  const filename = inferenceFilenameForId(variant);
  const candidate = join(recipeDir, filename);
  return existsSync(candidate) ? candidate : null;
}
```

- [ ] **Step 4: Run to verify it passes (including pre-existing tests)**

Run: `npx vitest run packages/agent/src/runtime/inference-template.test.ts`
Expected: PASS — pre-existing variant tests still green + 3 new cases.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/inference-template.ts \
        packages/agent/src/runtime/inference-template.test.ts
git commit -m "agent: findInferenceTemplate accepts any variant id, keeps legacy aliases"
```

---

## Task 4: Populate `TrainingRecipe.inferenceVariants` during discovery

**Files:**
- Modify: `packages/agent/src/training-recipes.ts:13-36, 135-200`

**Why:** The agent already ships training-recipe metadata to the server on connect (`agent:training-recipes` message). Adding `inferenceVariants` to that payload is the cheapest way to get the list to the dashboard.

- [ ] **Step 1: Write the failing test**

Create `packages/agent/src/training-recipes.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { discoverTrainingRecipes } from "./training-recipes.js";

describe("discoverTrainingRecipes — inference variants", () => {
  it("populates inferenceVariants for each recipe dir with inference*.yaml files", () => {
    const repoRoot = mkdtempSync(join(tmpdir(), "training-recipes-"));
    process.env.TRAINING_REPO_PATH = repoRoot;
    const recipeDir = join(repoRoot, "recipes", "demo");
    try {
      mkdirSync(recipeDir, { recursive: true });
      writeFileSync(join(recipeDir, "recipe.yaml"),
        `name: Demo\nbase_model: x/y\nframework: deepspeed\nmethod: lora\n` +
        `container:\n  image: img\n  name: demo\n` +
        `scripts:\n  entrypoint: e.sh\n  train: t.py\n  launch: l.sh\n` +
        `defaults: {}\nhardware:\n  min_nodes: 1\n  gpus_per_node: 1\n  vram_estimate_mb: 0\n`);
      writeFileSync(join(recipeDir, "inference.yaml"),
        `name: demo-bf16\ndescription: Default serve.\n`);
      writeFileSync(join(recipeDir, "inference-fp8.yaml"),
        `name: demo-fp8\ndescription: On-load FP8.\n`);

      const recipes = discoverTrainingRecipes();
      const demo = recipes.find((r) => r.file === "recipes/demo");
      expect(demo).toBeDefined();
      expect(demo!.inferenceVariants).toEqual([
        { id: "default", filename: "inference.yaml",     name: "demo-bf16", description: "Default serve." },
        { id: "fp8",     filename: "inference-fp8.yaml", name: "demo-fp8",  description: "On-load FP8." },
      ]);
    } finally {
      rmSync(repoRoot, { recursive: true, force: true });
      delete process.env.TRAINING_REPO_PATH;
    }
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run packages/agent/src/training-recipes.test.ts`
Expected: FAIL — `inferenceVariants` is undefined on the returned recipe.

- [ ] **Step 3: Extend the type + populate during discovery**

In `packages/agent/src/training-recipes.ts`:

Add this import line near the top (after the existing `from "fs"` import):

```ts
import { listInferenceVariants, type InferenceVariant } from "./runtime/inference-template.js";
```

Extend the `TrainingRecipe` interface — add one field after the existing `deploy?:` line (around line 35):

```ts
  /** One entry per `inference*.yaml` file in the recipe dir. Empty
   *  array when the recipe has no inference templates (older recipes,
   *  pre-inference-template feature). */
  inferenceVariants?: InferenceVariant[];
```

In `discoverTrainingRecipes`, inside the `recipes.push({...})` object (right before the closing `});` near line 191), add the field:

```ts
        inferenceVariants: listInferenceVariants(join(recipesDir, entry.name)),
```

- [ ] **Step 4: Run to verify it passes + check the existing training-recipes test stays green**

Run: `npx vitest run packages/agent/src/training-recipes.test.ts packages/agent/src/runtime/inference-variants.test.ts`
Expected: PASS — new test + Task 1/2 tests all green.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/training-recipes.ts \
        packages/agent/src/training-recipes.test.ts
git commit -m "agent: discoverTrainingRecipes populates inferenceVariants per recipe"
```

---

## Task 5: Bump agent version

**Files:**
- Modify: `packages/agent/package.json` (via script)

**Why:** Project rule (CLAUDE.md): any change under `packages/agent/src/` requires a patch-version bump so the dashboard's outdated-agent detector fires.

- [ ] **Step 1: Bump**

Run:
```bash
./scripts/bump-agent-version.sh
```

Expected output: the script prints the new version. The new patch number lands in `packages/agent/package.json`.

- [ ] **Step 2: Verify**

Run:
```bash
git diff packages/agent/package.json
```

Expected: exactly one line changed — the `"version"` field went up by one patch.

- [ ] **Step 3: Commit**

```bash
git add packages/agent/package.json
git commit -m "agent: bump version for inferenceVariants discovery"
```

---

## Task 6: Mirror `inferenceVariants` on the server-side TrainingRecipe type

**Files:**
- Modify: `packages/server/src/ws/agent-hub.ts` (the `TrainingRecipe` interface around line 25)

**Why:** The server has its own `TrainingRecipe` type that mirrors the agent's. Without the new field on the server side, the dashboard's API response won't carry `inferenceVariants` through TypeScript even though the agent sends it.

- [ ] **Step 1: Inspect the current server-side type**

Run:
```bash
grep -n -A 30 "^export interface TrainingRecipe" packages/server/src/ws/agent-hub.ts
```

Expected: see a `TrainingRecipe` interface mirroring the agent's, with `file`, `name`, `description`, `base_model`, `framework`, `method`, `defaults`, `hardware`, `deploy?:`, etc.

- [ ] **Step 2: Add the mirror field**

Find the `TrainingRecipe` interface in `packages/server/src/ws/agent-hub.ts`. Add one field (inside the interface, after the existing `deploy?:` line):

```ts
  /** Mirror of agent's TrainingRecipe.inferenceVariants — one entry per
   *  inference*.yaml file in the recipe dir. Empty/undefined when the
   *  recipe doesn't ship inference templates. */
  inferenceVariants?: {
    id: string;
    filename: string;
    name: string;
    description?: string;
  }[];
```

(We inline the shape rather than importing `InferenceVariant` from the agent package because the agent's runtime/ folder isn't a published export — the server consumes the type by structural duck-typing across the WS boundary, same as the rest of `TrainingRecipe`.)

- [ ] **Step 3: Run the full test suite to confirm nothing broke**

Run:
```bash
npm test
```

Expected: pre-existing test count + the new tests from Tasks 1, 2, 4 all green.

- [ ] **Step 4: Commit**

```bash
git add packages/server/src/ws/agent-hub.ts
git commit -m "server: mirror TrainingRecipe.inferenceVariants type from agent"
```

---

## Task 7: Relax `artifactVariant` validation in routes

**Files:**
- Modify: `packages/server/src/routes/finetune.ts:685-691` (the POST `/deploy` body type)
- Modify: `packages/server/src/routes/deployments.ts:457-470` (the restart-route variant read)
- Create: `packages/server/src/__tests__/integration/deployments.restart-variant.test.ts`

**Why:** Both routes currently type `artifactVariant?: "bf16" | "fp8"`. Now that variants are open-ended slugs, the literal type must go. We add a simple slug-shape validation so a typo'd or hostile value can't escape into a filesystem path.

- [ ] **Step 1: Write the failing integration test**

Create `packages/server/src/__tests__/integration/deployments.restart-variant.test.ts`:

```ts
/**
 * Variant-id propagation through the restart route. Complements the
 * existing deployments.restart-finetune suite by testing arbitrary
 * variant slugs (not just the legacy bf16/fp8 pair) and rejection of
 * malformed slugs at the route layer.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-restart-variant-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      DATABASE_URL: `file:${DB_PATH}`,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "User consented to db push --force-reset against per-suite SQLite test databases in /tmp on 2026-05-03 (option #1)",
    },
    stdio: "pipe",
  });
  ({ prisma } = await import("../../prisma.js"));
  ({ deploymentsRouter } = await import("../../routes/deployments.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  await prisma.loadBalancerEndpoint.deleteMany();
  await prisma.clusterNode.deleteMany();
  await prisma.deployment.deleteMany();
  await prisma.fineTuneJob.deleteMany();
  await prisma.model.deleteMany();
  await prisma.node.deleteMany();
});

const TRAINING_RECIPE = {
  file: "recipes/test-training",
  name: "Test Training Recipe",
  deploy: { container: "vllm-node-custom" },
};

function makeStubHub() {
  const sent: { nodeId: string; message: { type: string; payload: Record<string, unknown> } }[] = [];
  return {
    hub: {
      getRecipes: () => [],
      getTrainingRecipes: () => [TRAINING_RECIPE],
      getOllamaModels: () => [],
      sendToAgent: (nodeId: string, message: { type: string; payload: Record<string, unknown> }) => {
        sent.push({ nodeId, message });
      },
    },
    sent,
  };
}

function makeApp(hub: unknown) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

async function seedFineTuneDeployment(artifactVariant?: string) {
  const node = await prisma.node.create({
    data: { name: "n1", status: "online", vramTotal: 128000, ipAddress: "10.0.0.10" },
  });
  const job = await prisma.fineTuneJob.create({
    data: {
      nodeId: node.id, baseModel: "meta-llama/Llama-3.1-8B", method: "lora",
      dataset: "test", recipeFile: "recipes/test-training",
      status: "completed", mergeStatus: "completed",
      outputDir: "/mnt/tank/outputs/job1", mergedPath: "/mnt/tank/outputs/job1/merged",
    },
  });
  const model = await prisma.model.create({
    data: { name: "finetune-job1", runtime: "vllm", finetuneJobId: job.id },
  });
  const cfg: Record<string, unknown> = {
    port: 8000, gpuMem: 0.8, maxModelLen: 8192,
    localModelPath: "/mnt/tank/outputs/job1/merged",
  };
  if (artifactVariant) cfg.artifactVariant = artifactVariant;
  return prisma.deployment.create({
    data: {
      nodeId: node.id, modelId: model.id, status: "failed", port: 8000,
      config: JSON.stringify(cfg),
    },
  });
}

describe("POST /api/deployments/:id/restart — arbitrary variant ids", () => {
  it("accepts a custom variant slug and forwards it as artifactVariant", async () => {
    const dep = await seedFineTuneDeployment();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post(`/api/deployments/${dep.id}/restart`)
      .send({ config: { artifactVariant: "int4" } });

    expect(res.status).toBe(200);
    expect(sent[0].message.type).toBe("cmd:finetune:deploy");
    expect(sent[0].message.payload.artifactVariant).toBe("int4");

    // And persisted in saved config for the next restart cycle
    const saved = JSON.parse((await prisma.deployment.findUnique({ where: { id: dep.id } }))!.config!);
    expect(saved.artifactVariant).toBe("int4");
  });

  it("rejects malformed variant slugs with 400", async () => {
    const dep = await seedFineTuneDeployment();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post(`/api/deployments/${dep.id}/restart`)
      .send({ config: { artifactVariant: "../etc/passwd" } });

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/artifactVariant/);
  });

  it("preserves the legacy 'bf16' slug through restart (no auto-migration)", async () => {
    // Pre-feature deployments stored "bf16" — the route shouldn't rewrite
    // it on restart. Storage-side back-compat is the agent's job.
    const dep = await seedFineTuneDeployment("bf16");
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app).post(`/api/deployments/${dep.id}/restart`).send({});

    expect(res.status).toBe(200);
    expect(sent[0].message.payload.artifactVariant).toBe("bf16");
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/deployments.restart-variant.test.ts`
Expected: FAIL — the "rejects malformed slugs" test fails because no validation is in place yet; the "accepts int4" test fails at the type-checker level if we're strict (or at runtime depending on whether TypeScript stripped it).

- [ ] **Step 3: Add the validator + relax the types**

In `packages/server/src/routes/finetune.ts` near the top of the file (alongside other helpers), add:

```ts
/** Variant ids are filename slugs — lowercase alphanumerics + hyphens, must
 *  start with an alphanumeric, max ~32 chars. Tight enough to keep `..` and
 *  path separators out of the filename interpolation in
 *  `inferenceFilenameForId`. */
const VARIANT_SLUG_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;

export function isValidVariantSlug(v: unknown): v is string {
  return typeof v === "string" && VARIANT_SLUG_RE.test(v);
}
```

In the same file, find the destructure at line ~685 (`const { nodeId, nodeIds, config, artifactVariant, displayName: rawDisplayName } = req.body as { ... }`). Change the type annotation on `artifactVariant`:

```ts
  const { nodeId, nodeIds, config, artifactVariant, displayName: rawDisplayName } = req.body as {
    nodeId?: string;
    nodeIds?: string[];
    config?: Record<string, unknown>;
    artifactVariant?: string;        // was: "bf16" | "fp8"
    displayName?: string | null;
  };
```

Right after the destructure, add the validation guard:

```ts
  if (artifactVariant !== undefined && !isValidVariantSlug(artifactVariant)) {
    return res.status(400).json({
      error: `Invalid artifactVariant: must match ${VARIANT_SLUG_RE} (got ${JSON.stringify(artifactVariant)})`,
    });
  }
```

Right below that, replace the existing variant normalization line:

```ts
// Old:
//   const variant: "bf16" | "fp8" = artifactVariant === "fp8" ? "fp8" : "bf16";
// New:
  const variant: string = artifactVariant ?? "default";
```

In `packages/server/src/routes/deployments.ts`, find the restart route's variant read (around line 457):

```ts
// Old:
//   const artifactVariant = (config.artifactVariant as "bf16" | "fp8" | undefined) ?? "bf16";
// New:
  const artifactVariant: string = typeof config.artifactVariant === "string" && isValidVariantSlug(config.artifactVariant)
    ? config.artifactVariant
    : "default";
```

And right after merging `overrides` into `config` (around line 387, after the line `const config = { ...savedConfig, ...overrides };`), add the validation guard for caller-supplied overrides:

```ts
  if (typeof overrides.artifactVariant !== "undefined" && !isValidVariantSlug(overrides.artifactVariant)) {
    return res.status(400).json({
      error: `Invalid artifactVariant: must match /^[a-z0-9][a-z0-9-]{0,31}$/ (got ${JSON.stringify(overrides.artifactVariant)})`,
    });
  }
```

You'll need to import `isValidVariantSlug` from finetune.ts. At the top of `deployments.ts`:

```ts
import { isValidVariantSlug } from "./finetune.js";
```

- [ ] **Step 4: Run all routes tests to verify**

Run: `npx vitest run packages/server/src/__tests__/integration/deployments.restart-variant.test.ts packages/server/src/__tests__/integration/deployments.restart-finetune.test.ts`
Expected: PASS — 3 new tests + 7 existing restart-finetune tests.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/routes/finetune.ts \
        packages/server/src/routes/deployments.ts \
        packages/server/src/__tests__/integration/deployments.restart-variant.test.ts
git commit -m "server: accept arbitrary variant slugs on deploy + restart with slug validation"
```

---

## Task 8: Update the dashboard's `Recipe` TypeScript interface

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx:10-24`

**Why:** The Recipe interface on the dashboard mirrors the server's TrainingRecipe shape (re-used for both vLLM and training recipes today). Adding `inferenceVariants` here makes the field visible to the deploy + restart form code coming in Tasks 9 and 10.

- [ ] **Step 1: Add the field**

In `packages/dashboard/app/deployments/page.tsx`, find the `interface Recipe {` block (line ~10). Add this field at the bottom of the interface (right before the closing `}`):

```ts
  /** One entry per `inference*.yaml` in the recipe dir. Surfaced only on
   *  training recipes; absent / undefined for plain vLLM serve recipes. */
  inferenceVariants?: {
    id: string;
    filename: string;
    name: string;
    description?: string;
  }[];
```

- [ ] **Step 2: Run the dashboard typecheck**

Run:
```bash
npx tsc --noEmit -p packages/dashboard/tsconfig.json
```

Expected: clean — no errors. (The field is additive, no existing usage breaks.)

- [ ] **Step 3: Commit**

```bash
git add packages/dashboard/app/deployments/page.tsx
git commit -m "dashboard: extend Recipe type with inferenceVariants"
```

---

## Task 9: Add variant selector to the fine-tune deploy form

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx` (the fine-tune section of the deploy form, around lines 940–1010)

**Why:** The deploy form currently doesn't expose variant choice — fine-tune deploys silently default to bf16. New behavior: surface a `<select>` populated from the training recipe's `inferenceVariants`; auto-select when there's exactly one; persist the chosen id via the existing `finetuneArtifactVariant` state which the POST body already includes.

- [ ] **Step 1: Inspect the current fine-tune section**

Run:
```bash
sed -n '940,1010p' packages/dashboard/app/deployments/page.tsx
```

You should see the fine-tune branch of the deploy form, including the existing TP/PP/max_model_len/gpu_mem inputs sourced from `finetuneRecipeData?.deploy`.

- [ ] **Step 2: Add the auto-select effect + the `<select>`**

Add a `useEffect` that auto-selects when there's exactly one variant. Find the existing `useEffect` near line 240 that prefills from `finetuneRecipeData.deploy` (`if (!finetuneRecipeData?.deploy || prefilledFromFinetuneRecipe.current) return;`). Right after that effect, add:

```tsx
  // When the training recipe exposes inference variants, auto-select the
  // only one when there's exactly one. Multi-variant case: leave it to
  // the user — show "default" as the initial selection if it exists, else
  // the first sorted entry from listInferenceVariants.
  useEffect(() => {
    if (runtimeMode !== "finetune") return;
    if (finetuneArtifactVariant) return; // user already chose
    const vs = finetuneRecipeData?.inferenceVariants ?? [];
    if (vs.length === 0) return;
    const auto = vs.length === 1 ? vs[0].id : (vs.find((v) => v.id === "default")?.id ?? vs[0].id);
    setFinetuneArtifactVariant(auto);
  }, [runtimeMode, finetuneRecipeData, finetuneArtifactVariant]);
```

In the JSX where the fine-tune deploy form shows the recipe info (look for the existing `{finetuneRecipeData && (` block around line 947), insert a new control **right after** the "Recipe:" line and **before** the TP/PP/max_model_len/gpu_mem grid:

```tsx
            {finetuneRecipeData?.inferenceVariants && finetuneRecipeData.inferenceVariants.length > 0 && (
              <div className="mt-3">
                <label className="block text-xs text-gray-400 mb-1">
                  Inference variant
                  {finetuneRecipeData.inferenceVariants.length === 1 && (
                    <span className="ml-2 text-gray-500">(only one available — auto-selected)</span>
                  )}
                </label>
                <select
                  value={finetuneArtifactVariant ?? ""}
                  onChange={(e) => setFinetuneArtifactVariant(e.target.value || null)}
                  disabled={finetuneRecipeData.inferenceVariants.length === 1}
                  className="w-full bg-gray-950 border border-gray-800 rounded px-2 py-1.5 text-sm focus:outline-none focus:border-gray-600 disabled:opacity-60"
                >
                  {finetuneRecipeData.inferenceVariants.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.id} — {v.name}
                    </option>
                  ))}
                </select>
                {(() => {
                  const sel = finetuneRecipeData!.inferenceVariants!.find((v) => v.id === finetuneArtifactVariant);
                  return sel?.description ? (
                    <p className="mt-1 text-xs text-gray-500">{sel.description}</p>
                  ) : null;
                })()}
              </div>
            )}
```

- [ ] **Step 3: Run dashboard typecheck**

Run:
```bash
npx tsc --noEmit -p packages/dashboard/tsconfig.json
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/app/deployments/page.tsx
git commit -m "dashboard: inference-variant <select> on fine-tune deploy form"
```

---

## Task 10: Add variant selector to the edit-restart form

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx` (the `beginEditRestart` prefill + the restart-form JSX around lines 1278–1352)

**Why:** Same affordance for restarting a stopped fine-tune. Pre-fill from the saved `artifactVariant` (so a previously-fp8 deploy stays fp8 after restart); allow the user to switch.

- [ ] **Step 1: Extend the editingRestart state shape**

Find the `editingRestart` state declaration (around line 138). It declares per-id fields `port`, `maxModelLen`, `tensorParallel`, `pipelineParallel`, `gpuMem`. Add `artifactVariant`:

```ts
  const [editingRestart, setEditingRestart] = useState<Record<string, {
    port?: string;
    maxModelLen?: string;
    tensorParallel?: string;
    pipelineParallel?: string;
    gpuMem?: string;
    artifactVariant?: string;
  }>>({});
```

(Adjust to the precise existing fields — just append `artifactVariant?: string;` inside the inline record type.)

- [ ] **Step 2: Pre-fill artifactVariant in beginEditRestart**

In `beginEditRestart` (around line 523), inside the `setEditingRestart((prev) => ({ ... }))` call, add `artifactVariant` next to the other fields:

```ts
    setEditingRestart((prev) => ({
      ...prev,
      [id]: {
        port: pick(cfg.port, recipeDefaults.port),
        maxModelLen: pick(cfg.maxModelLen, recipeDefaults.max_model_len),
        tensorParallel: pick(cfg.tensorParallel, recipeDefaults.tensor_parallel),
        pipelineParallel: pick(cfg.pipelineParallel, recipeDefaults.pipeline_parallel),
        gpuMem: pick(cfg.gpuMem, recipeDefaults.gpu_memory_utilization),
        artifactVariant: typeof cfg.artifactVariant === "string" ? cfg.artifactVariant : "default",
      },
    }));
```

- [ ] **Step 3: Forward artifactVariant in submitEditRestart**

In `submitEditRestart` (around line 556), after the existing `if (fields.gpuMem)` line, add:

```ts
    if (fields.artifactVariant) overrides.artifactVariant = fields.artifactVariant;
```

- [ ] **Step 4: Find the deployment's training recipe in the form's scope**

In the per-deployment render block where `isFinetune` and `recipeDefaultsForRestart` are computed (around line 1115–1119), add:

```ts
            const ftRecipeFile = d.model?.finetuneJob?.recipeFile;
            const ftRecipe = ftRecipeFile
              ? trainingRecipes.find((r) => r.file === ftRecipeFile)
              : undefined;
            const ftVariants = ftRecipe?.inferenceVariants ?? [];
```

- [ ] **Step 5: Render the variant select inside the edit form (fine-tunes only)**

Inside the `{!isOllama && editingRestart[d.id] && (` block, at the **end of the `grid grid-cols-2 md:grid-cols-5 gap-3` div** (right before the closing `</div>` of that grid, which sits above the Cancel/Restart buttons around line 1336), insert a 6th cell for the variant select — but only when this is a fine-tune AND the recipe exposes variants:

```tsx
                      {isFinetune && ftVariants.length > 0 && (
                        <div className="col-span-2 md:col-span-5">
                          <label className="block text-[10px] text-gray-500 mb-0.5">
                            Inference variant
                            {ftVariants.length === 1 && (
                              <span className="ml-2 text-gray-500">(only one available)</span>
                            )}
                          </label>
                          <select
                            value={editingRestart[d.id].artifactVariant ?? "default"}
                            onChange={(e) => setEditingRestart((p) => ({
                              ...p,
                              [d.id]: { ...p[d.id], artifactVariant: e.target.value },
                            }))}
                            disabled={ftVariants.length === 1}
                            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500 disabled:opacity-60"
                          >
                            {ftVariants.map((v) => (
                              <option key={v.id} value={v.id}>
                                {v.id} — {v.name}
                              </option>
                            ))}
                          </select>
                          {(() => {
                            const sel = ftVariants.find((v) => v.id === (editingRestart[d.id].artifactVariant ?? "default"));
                            return sel?.description ? (
                              <p className="mt-1 text-[10px] text-gray-500">{sel.description}</p>
                            ) : null;
                          })()}
                        </div>
                      )}
```

- [ ] **Step 6: Run dashboard typecheck**

Run:
```bash
npx tsc --noEmit -p packages/dashboard/tsconfig.json
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add packages/dashboard/app/deployments/page.tsx
git commit -m "dashboard: inference-variant <select> on edit-restart form"
```

---

## Task 11: Full build + test sweep

**Files:** (no source changes — just verification)

**Why:** The feature spans agent + server + dashboard. Make sure the whole thing builds and every test stays green before declaring done.

- [ ] **Step 1: Run the full build**

Run:
```bash
npm run build
```

Expected: all three packages compile clean; the Next.js build of the dashboard prints the route table including `/deployments` with a slightly larger bundle than before (the select adds ~1 KB).

- [ ] **Step 2: Run the full test suite**

Run:
```bash
npm test
```

Expected: every test passes. New cases since `main`: ~12 new ones (Task 1 = 4, Task 2 = 5, Task 3 = 3, Task 4 = 1, Task 7 = 3) — actual count will depend on how property tests are counted.

- [ ] **Step 3: If any tests fail, stop here and fix forward**

Tests should all pass after Task 7. If they don't:
- Pre-existing `deployments.restart-finetune.test.ts`: the `artifactVariant=bf16` default-back-compat test now reads "default" if you accidentally normalized in the route — should still be "bf16" for legacy rows. Revisit Task 7 Step 3 — the route must NOT auto-rewrite legacy values.
- Pre-existing `inference-template.test.ts`: if you changed the parameter type but the old tests fail, the old tests probably passed `"bf16"`/`"fp8"` as the variant — those still work after Task 3, so they should stay green.

No commit at this step — the previous task commits already cover the changes. This is verification only.

---

## Task 12: Manual smoke test on the running 4-node deployment

**Files:** (no code; manual verification)

**Why:** This whole feature exists so the user can switch the live `chat3d-build123d-02-synthetic-16k:ma` deployment from bf16 to fp8 via the UI instead of curl. Verify end-to-end.

- [ ] **Step 1: Deploy the new images**

Run:
```bash
MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build
```

Expected: server + dashboard rebuild and replace cleanly. The agent on each node also picks up the new bundle on next reconnect (the version bump from Task 5 triggers the dashboard's outdated-agent indicator).

- [ ] **Step 2: Verify the API surfaces inferenceVariants**

Run:
```bash
curl -s http://localhost:4000/api/training-recipes \
  | python3 -c "import json,sys; r=[x for x in json.load(sys.stdin) if x['file']=='recipes/qwen3.6-27b-base-lora-attn-mlp'][0]; print(json.dumps(r.get('inferenceVariants'), indent=2))"
```

Expected: a JSON array with two entries — `default` (inference.yaml) and `fp8` (inference-fp8.yaml), each with the recipe's `name:` and `description:`.

- [ ] **Step 3: Open the Deployments page**

Visit `http://<host>:3000/deployments` and look at the `chat3d-build123d-02-synthetic-16k:ma` deployment.

If it's still `running`: click **Stop** first.

After it shows `stopped` or `failed`, click **Restart**. The edit-restart form should appear and now include the **Inference variant** dropdown at the bottom with two options:
- `default — qwen3.6-27b-base-lora-attn-mlp-bf16`
- `fp8 — qwen3.6-27b-base-lora-attn-mlp-fp8`

The current value should be pre-selected ("default" since this row's saved `artifactVariant` is missing → restart route fallback).

- [ ] **Step 4: Switch to fp8 and restart**

Pick **fp8** from the dropdown, leave the rest of the fields, click **Restart**.

Expected: the deployment goes through `restarting → starting → launching → running`. Watch the live deployment log on the deployments page — the vLLM launch command should include `--quantization fp8`.

- [ ] **Step 5: Confirm via /v1/models**

Run:
```bash
curl -s http://192.168.44.36:8000/v1/models | python3 -c "import json,sys; m=json.load(sys.stdin)['data'][0]; print('model id:', m['id'])"
```

Expected: the served model id matches the deployment's `displayName` (`chat3d-build123d-02-synthetic-16k:ma`).

- [ ] **Step 6: Verify saved config persists the new variant**

Run:
```bash
docker run --rm -v dgx-manager_dgx-data:/data alpine sh -c \
  "apk add -q sqlite && sqlite3 /data/dev.db \"SELECT json_extract(config, '\$.artifactVariant') FROM Deployment WHERE id='cmpa33s9l00lp36qk3svbsxib';\""
```

Expected: `fp8`. If it's empty, the dashboard form didn't forward the field — revisit Task 10 Step 3.

- [ ] **Step 7: Quick benchmark to confirm the speedup**

From the Deployments page, click **Benchmark** on the now-fp8 deployment, pick **Quick smoke**, and start. Compare the resulting mean t/s on `/benchmarks` against the previously-recorded bf16 throughput run for the same model (`cmpa8c7ji00dw36o4ampkbtr5`).

Expected: fp8 mean tg t/s ≥ ~1.3× the bf16 mean. If not, the launch may not have applied `--quantization fp8` — confirm in the live log.

- [ ] **Step 8: If everything passed, no commit needed**

This task is verification only.

---

## Self-review

**Spec coverage:**
- ✅ Variant id derivation: Task 1
- ✅ Filename → variant list with name + description: Task 2
- ✅ Back-compat aliases (bf16, fp8): Tasks 1, 3
- ✅ Auto-select when only one variant: Tasks 9, 10
- ✅ Surface info per recipe via the API: Task 4 + 6 (agent → server → dashboard)
- ✅ Open-string `artifactVariant` validation: Task 7
- ✅ Deploy + restart UI: Tasks 9, 10
- ✅ Manual smoke covers the live use case: Task 12

**No placeholders found.** Every code block is concrete; commands have expected outputs; file paths are specific.

**Type consistency check:**
- `InferenceVariant` shape: `{ id, filename, name, description? }` — used identically in Tasks 1, 2, 4, 6, 8 (agent + server + dashboard).
- `artifactVariant`/`variant` parameter shape: open string everywhere. The type was loosened at the routes in Task 7 and at `findInferenceTemplate` in Task 3.
- `inferenceVariantIdFromFilename` / `inferenceFilenameForId` / `listInferenceVariants` / `findInferenceTemplate` — all referenced by the same names in later tasks.

---

## Plan complete

Saved to `docs/superpowers/plans/2026-05-19-inference-variant-selector.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
