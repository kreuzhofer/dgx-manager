# In-repo `@dgxrun` recipe catalog (design)

**Status:** design / approved to plan
**Date:** 2026-07-05

## Motivation

dgxrun is *our* multi-node mp vLLM runner, so its recipes should be owned by this repo and first-class in the UI — not borrowed from external sparkrun registries. Today there is a gap: a dgxrun recipe can only be deployed via inline `recipeYaml` or a `recipePath`. The GLM-5.2 recipe currently lives in the external `community-recipe-registry`, so it *appears* in the deploy dropdown as `@community-kreuzhofer/glm-5.2-…`, but selecting it there routes through the **sparkrun/Ray** path (`recipeFile`), which is exactly what's broken for GLM-5.2 — the deploy fails at Ray init. `deployments.ts` documents this explicitly (lines 174-176): "Registry-ref (recipeFile) dgxrun is a v1 follow-up."

This spec makes dgxrun recipes in-repo, catalogued under a dedicated `@dgxrun/…` namespace, kept separate from sparkrun recipes, and **dropdown-deployable** — selecting an `@dgxrun/…` recipe routes to the dgxrun runner automatically.

## Goals

- dgxrun recipes live **in-repo** at `recipes/dgxrun/*.yaml`, versioned with the code.
- The **server** serves them at `GET /api/recipes` under `file: "@dgxrun/<name>"` with a `source: "dgxrun"` marker, merged with (but distinguishable from) the agent-discovered sparkrun recipes.
- Selecting an `@dgxrun/…` recipe → `POST /api/deployments { recipeFile: "@dgxrun/…" }` resolves to the **dgxrun runner** automatically (closes the v1 `recipeFile`→dgxrun gap), reusing the existing `resolveDgxrunRecipe` + `buildDgxrunDeploys`.
- The deploy dropdown renders dgxrun recipes as a **separate group** ("dgxrun") from sparkrun recipes.
- Migrate GLM-5.2 into the catalog as the first `@dgxrun` recipe.

## Non-goals

No change to sparkrun discovery/deploy. No new dgxrun runtime behavior (dispatch/fan-out unchanged). No auth/permissions. Not touching the external community-recipe-registry (its GLM copy stays for sparkrun users; we mark it superseded in a comment, no deletion).

## Architecture

### 1. In-repo catalog — `recipes/dgxrun/*.yaml`
Each file is a full dgxrun recipe YAML (the validated shape: `model`, `container` image, vLLM args, `env` incl. `HF_HOME=/cache/huggingface` + `HF_HUB_OFFLINE=1`, `runner: dgxrun`, cluster/master settings). This dir is the single source of truth for dgxrun recipes.

### 2. Server catalog loader — `packages/server/src/deployments/dgxrun-catalog.ts` (pure)
- `loadDgxrunCatalog(dir, readDir?, readFile?): Recipe[]` — reads `*.yaml` in `dir`, parses each, maps to the existing `Recipe` wire shape:
  - `file: "@dgxrun/<basename-without-ext>"`, `name`, `model`, `description`, `container: "dgxrun"`, `source: "dgxrun"`, `arch: "arm64"` (DGX; derived or defaulted), `cluster_only: true`, `defaults` (tensor_parallel, gpu_memory_utilization, port, max_model_len from the recipe).
  - `readDir`/`readFile` injectable (default real fs) for unit tests — same pattern as `dgxrun-args.ts`.
- A malformed file is **skipped with a logged warning** (fail-safe: one bad recipe never blanks the catalog), consistent with fail-fast-at-boundary + observability.
- Loaded once at server startup and cached on the `AgentHub` (or a module singleton); `POST /api/recipes/refresh` re-reads the dir.

### 3. Serve — merge into `GET /api/recipes`
`recipes.ts` currently returns `agentHub.getRecipes()` (arch-filtered). Change: return `[...dgxrunCatalog, ...agentHub.getRecipes()]` (dgxrun first) so the existing arch filter and node-scoping apply uniformly. The `Recipe` interface gains `source?: "sparkrun" | "dgxrun"` (default treat missing as `"sparkrun"`).

### 4. Route — `@dgxrun/` prefix → dgxrun (the gap-closer)
In `deployments.ts`, in the recipe-source resolution block (currently: inline `recipeYaml` → `recipePath` → `recipeFile`): when `recipeFile` starts with `@dgxrun/`, resolve `<name>` → read `recipes/dgxrun/<name>.yaml` → feed its text to the **existing** `resolveDgxrunRecipe(text)`; on match, `isDgxrun = true` and proceed through the existing dgxrun dispatch. A non-`@dgxrun/` `recipeFile` is unchanged (sparkrun path). Validate the resolved path stays inside `recipes/dgxrun/` (reject `..`/absolute — security boundary, mirrors `resolveRecipePath`).

### 5. Dashboard — group the dropdown by `source`
The deploy form maps `GET /api/recipes` into the recipe `<select>`. Render two `<optgroup>`s: **"dgxrun"** (source === "dgxrun") and **"sparkrun"** (everything else), dgxrun first. No behavior change beyond grouping; the selected `file` already carries the `@dgxrun/` prefix that drives routing.

### 6. Migration
Add `recipes/dgxrun/glm-5.2-awq-15pct.yaml` (from the validated deploy recipe: `runner: dgxrun`, HF offline env, mp executor, MTP, RDMA/NCCL env, 85K@0.88). It becomes `@dgxrun/glm-5.2-awq-15pct`, deployable from the dropdown.

## Testing (risk-scaled — new catalog source + a new deploy routing branch)

- **Unit** (`dgxrun-catalog.test.ts`): `loadDgxrunCatalog` with a stub `readDir`/`readFile` — maps a valid YAML to the `Recipe` shape with `file:"@dgxrun/<name>"`, `source:"dgxrun"`; a malformed file is skipped (others still load); empty dir → `[]`.
- **Unit**: the `@dgxrun/` recipeFile → path resolution helper (given `"@dgxrun/glm-5.2-awq-15pct"` → `recipes/dgxrun/glm-5.2-awq-15pct.yaml`; rejects traversal like `@dgxrun/../../etc/passwd`).
- **Integration** (`agent-v2`-style, stub agentHub): `GET /api/recipes` includes the `@dgxrun/glm-5.2-awq-15pct` recipe with `source:"dgxrun"`; `POST /api/deployments { recipeFile:"@dgxrun/glm-5.2-awq-15pct", nodeIds:[…] }` resolves `isDgxrun=true` and builds the head-first fan-out (assert the dispatch shape, no real agents).
- `npm test` green. No agent code changes ⇒ no agent version bump.

## Isolation / boundaries
- `dgxrun-catalog.ts` — pure dir→`Recipe[]` mapper, injectable fs, no HTTP/DB. Testable in isolation.
- Deploy-route change — a single added branch in the existing source-resolution ladder; reuses `resolveDgxrunRecipe`/`buildDgxrunDeploys` unchanged.
- Recipes route — one-line merge.
- Dashboard — presentational grouping only.

## References
- `resolveDgxrunRecipe` (`packages/server/src/deployments/dgxrun-recipe.ts`), `buildDgxrunDeploys` (`dgxrun-dispatch.ts`), the `Recipe` wire shape (`packages/agent/src/recipes.ts`).
- dgxrun spec/plan: `docs/superpowers/plans/2026-07-04-dgxrun-runner.md`; memory `glm52-shm-and-jit-findings`.
