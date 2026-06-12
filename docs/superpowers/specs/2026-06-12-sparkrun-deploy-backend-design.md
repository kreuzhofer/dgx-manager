# Sparkrun Deploy Backend — Design

**Date:** 2026-06-12
**Status:** Approved design, pre-implementation
**Topic:** Replace the eugr `spark-vllm-docker` deployment path with [sparkrun](https://github.com/spark-arena/sparkrun) as DGX Manager's inference launch backend.

## 1. Background & Goal

The DGX Spark community has moved from eugr's `spark-vllm-docker` (custom `run-recipe.sh` /
`launch-cluster.sh` / `build-and-copy.sh` scripts) to **sparkrun** (`spark-arena/sparkrun`,
invoked as `uvx sparkrun`). Sparkrun is a single CLI that resolves a recipe, distributes
container images + models over SSH, configures networking, and launches inference containers
across one or more DGX Spark nodes — no Slurm/K8s. It supports git-based recipe **registries**
(`@spark-arena/<id>`, community, custom), multiple runtimes (`vllm`, `vllm-distributed`,
`vllm-ray`, `sglang`, `llama-cpp`, `trtllm`) selected via a `runtime:` field, and native VRAM
estimation.

**Goal: fully replace** the eugr launch path with sparkrun, retiring DGX Manager's bespoke
cluster-orchestration and image-sync code. This is inference-deploy only — fine-tune *training*
orchestration is unaffected; fine-tune *deploy* is rerouted through sparkrun.

### Key compatibility finding

The sparkrun recipe schema is a **superset** of the eugr format DGX Manager already parses:
same `model`, `container`, `command`, and `defaults` (`port`, `host`, `tensor_parallel`,
`gpu_memory_utilization`, `max_model_len`, `served_model_name`). Sparkrun adds `runtime`,
`min_nodes`/`max_nodes`, `env`, `mods`, `metadata` (incl. `model_vram`), `model_revision`. The
registry even references `@eugr/mods/...`. So this is an evolution, not a rewrite of recipe
semantics.

## 2. Decisions (locked during brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Scope vs eugr | **Replace eugr fully.** Retire `run-recipe.sh` path + `spark-vllm-docker` dependency. |
| D2 | Execution seam — who runs `sparkrun` | **Agent-side, on the head node** (Design B). The head-node agent runs `sparkrun run` in place of `run-recipe.sh`; the WS deploy/status/metrics/admission plumbing is preserved. |
| D3 | Provisioning state | **Nodes are bare.** The SSH provisioner installs sparkrun and runs its setup as part of bringing a node online. (`uvx` is already installed by the provisioner today.) |
| D4 | Recipe catalog source | **`sparkrun list` registries** (primary), reported to the dashboard over WS. Drop the NFS-clone scan of `spark-vllm-docker/recipes`. |
| D5 | Custom/dev recipes | **API-only, launch-by-path.** Drop a YAML on the NFS shared dir; launch via an API option that passes a `SHARED_STORAGE`-relative path to `sparkrun run <path>`. No upload UI, no dashboard picker for these. Path is validated to stay within `SHARED_STORAGE`. |
| D6 | Runtimes surfaced | **vLLM parity first.** Treat the recipe's `runtime:` as opaque — whatever a recipe declares runs. Status/metrics scraping stays vLLM-shaped (as today). Non-vLLM recipes launch but get no runtime-specific metrics until an adapter is added later (explicit YAGNI deferral). |

## 3. Architecture

Unchanged transport: Dashboard ⇄ WS ⇄ Server ⇄ WS ⇄ Agent (per node). Only the agent's launch
mechanism and the recipe-catalog source change. Agents keep reporting GPU metrics via
`nvidia-smi`.

### 3.1 Deploy flow (target)

```
Dashboard  POST /api/deployments { recipeRef | recipePath, nodeIds, config }
   │
Server     admission (VRAM, unchanged) → create Deployment(pending)
   │        pick head node (first of nodeIds), build cmd:deploy payload
   │  WS cmd:deploy → head-node agent
   ▼
Agent      launchSparkrun():  uvx sparkrun run <ref|path> \
   │           -H head,worker1,worker2 --tp N --port P \
   │           --gpu-mem G --max-model-len L --served-model-name S \
   │           -o key=value ...   (detached; --no-follow)
   │
sparkrun   SSHes to peers, syncs image+model, starts containers on all hosts
   ▼
Agent      checkDeployments(): `sparkrun status` (source of truth for what's
   │        running) + curl head localhost:port/metrics (vLLM metrics)
   │  WS agent:deployment:status → Server → SSE → Dashboard
   ▼
Inference proxy / load balancer routes to head-host:port from Deployment record
```

Stop: `DELETE /api/deployments/:id` → WS `cmd:undeploy` → agent `sparkrun stop <ref> [--tp N]`.

### 3.2 Recipe catalog flow (target)

```
Agent  on connect / on refresh:  `sparkrun list`  → recipe summaries
       (name, runtime, registry, description)  → WS agent:recipes → Server cache
Server GET /api/recipes → dashboard picker (unchanged shape to the dashboard)
Server POST /api/recipes/refresh → WS cmd:rescan-recipes → agent re-runs sparkrun list
Deploy-form details + admission inputs:  `sparkrun show <recipe>`
       (defaults, node range, VRAM estimate)  — replaces local YAML parsing
```

The dashboard's recipe-picker contract (`GET /api/recipes` returning a list the dropdown
renders) is **preserved**; only the agent-side producer changes from YAML-scan to `sparkrun
list`/`show`.

## 4. Component Changes

### 4.1 Agent — launch path (replaces eugr)

- **NEW `packages/agent/src/runtime/sparkrun.ts`** — the launcher. Mirrors the role of
  `vllm.ts` but builds a `sparkrun run` argv instead of a `run-recipe.sh` argv:
  - `buildSparkrunArgs({ recipeRef, hosts, tp, pp, port, gpuMem, maxModelLen, servedModelName, options })`
    → pure function returning the argv array. **Unit/property-tested.**
  - `launchSparkrun(deploymentId, recipeRef, options, onLog, onExit)` — spawns
    `uvx sparkrun run …` detached (`--no-follow`), streams logs to the same phase-detection
    pipeline used today, persists tracking.
  - `stopSparkrun(deploymentId, recipeRef, tp)` — `uvx sparkrun stop <ref> [--tp N]`.
    (Per sparkrun docs `--tp` must be threaded to `stop`/`status` for cluster resolution.)
  - `checkSparkrunDeployments()` — queries `sparkrun status` for liveness, then scrapes
    `http://localhost:{port}/metrics` on the head for vLLM metrics (reuses the existing
    `vllm:num_requests_running` / `vllm:kv_cache_usage_perc` parsing).
- **`packages/agent/src/runtime/vllm.ts`** — **retire** `launchRecipe`, `buildLaunchArgs`,
  `syncContainerImage` (sparkrun does image/model sync), `generateLocalModelRecipe`,
  `run-recipe.sh` wiring. **Keep & relocate** the vLLM `/metrics` parsing into `sparkrun.ts`
  (or a shared `vllm-metrics.ts`). The `VllmStatus` shape stays.
- **`packages/agent/src/recipes.ts`** — **retire** the custom YAML parser + `spark-vllm-docker`
  clone/scan. Replace with thin wrappers: `listRecipes()` → parse `sparkrun list`,
  `showRecipe(ref)` → parse `sparkrun show`. Recipe summary shape adapts to sparkrun fields
  (add `runtime`, `min_nodes`/`max_nodes`, `metadata.model_vram`).
- **`packages/agent/src/index.ts`** — `cmd:deploy` calls `launchSparkrun`; `cmd:undeploy` calls
  `stopSparkrun`; `cmd:rescan-recipes` re-runs `listRecipes`. Reconnect reconciliation
  (deployment-store) re-points to `sparkrun status` as the source of truth for "what is
  actually running" after an agent restart.
- **`packages/agent/src/runtime/deployment-store.ts`** — keep; reconciliation now cross-checks
  `sparkrun status` instead of a live `run-recipe.sh` subprocess.
- **Agent version bump** (`./scripts/bump-agent-version.sh`) — mandatory, since `agent/src/*`
  changes. Done once at end of implementation.

### 4.2 Agent bundle / provisioning

- **`packages/server/src/ssh/provisioner.ts`** — extend `auditNode`/`provisionNode` with a
  **sparkrun prerequisite**: install via `uvx` (already present) and run sparkrun's setup so
  the cluster mesh/sudoers/earlyoom/networking that `sparkrun run` depends on exists.
  - ⚠️ **Verification item V1:** `sparkrun setup` is documented as an *interactive* wizard. The
    provisioner needs a **non-interactive** path — either sparkrun flags / a pre-written
    sparkrun config, or DGX Manager replicates the underlying steps (SSH mesh — the manager
    already deploys agent SSH keys — plus sudoers + earlyoom). Resolve V1 before implementing
    this sub-task; the rest of the design does not depend on its outcome.

### 4.3 Server

- **`packages/server/src/routes/deployments.ts`** — payload gains an optional `recipePath`
  (SHARED_STORAGE-relative) alternative to `recipeFile`/`recipeRef`. `cmd:deploy` payload
  carries the resolved `recipeRef` (registry name | validated path) and host list. Head-node
  selection (first node) unchanged. Admission unchanged.
- **`packages/server/src/routes/recipes.ts`** — `/refresh` keeps broadcasting
  `cmd:rescan-recipes`; semantics now "re-run `sparkrun list`". `GET /api/recipes` shape
  preserved.
- **`packages/server/src/admission/vram.ts`** — keep `computeVramShortfall` /
  `checkVllmVramAdmission` driving off agent-reported free VRAM; the recipe's VRAM estimate
  now maps from sparkrun `metadata.model_vram` / `sparkrun show`. Pure-helper split preserved
  (still property-testable).
- **`packages/server/src/ws/agent-hub.ts`** — recipe cache now holds sparkrun summaries; deploy
  command paths unchanged in shape. Metrics path untouched.
- **Inference proxy / load balancer** — endpoint = head host + port from the `Deployment`
  record; effectively unchanged.

### 4.4 Custom recipe by path (D5)

- New request field `recipePath` on `POST /api/deployments`. **Validation (security boundary):**
  resolve against `SHARED_STORAGE`, reject if the resolved real path escapes `SHARED_STORAGE`
  (no `..`, no absolute escape, must exist, must be a file). On the agent, `sparkrun run` is
  given the validated absolute path. Fail-fast with a 400 on any violation. **Unit-tested** with
  traversal/escape cases.

### 4.5 Fine-tune deploy (rerouted, not redesigned)

- `cmd:finetune:deploy` currently calls `generateLocalModelRecipe` + `launchRecipe`. It instead
  writes a **sparkrun-format** recipe (`model: /workspace/outputs/{jobId}/merged`,
  `runtime: vllm`, `container`, `command`, `defaults`) to the NFS shared dir and calls
  `launchSparkrun(<that path>)`. Reuses the existing inference-template substitution
  (`inference-template.ts`). Fine-tune **training** (`cmd:finetune:start`) is untouched.

## 5. What gets retired

- `agent/src/runtime/vllm.ts`: `launchRecipe`, `buildLaunchArgs`, `syncContainerImage`,
  `generateLocalModelRecipe`, all `run-recipe.sh` wiring (~majority of the file).
- `agent/src/recipes.ts`: custom YAML parser + `spark-vllm-docker` clone/scan + `VLLM_REPO_URL`.
- Runtime dependence on the `spark-vllm-docker` NFS repo for deployment.

## 6. Risks & Verification Items

These are flagged to resolve **during** implementation; none invalidate the design, but each
shapes a sub-task. Fail-fast: if a verification fails, surface it explicitly rather than
falling back silently.

- **V1 — Non-interactive `sparkrun setup`** (mesh/sudoers/earlyoom/CX7). Flags/config vs.
  manual replication. Gates §4.2 only.
- **V2 — Machine-readable CLI output.** Do `sparkrun list` / `show` / `status` emit JSON (or
  stably parseable text), with keys we can map to recipes/deployments? Determines parsing
  robustness in `recipes.ts` / `checkSparkrunDeployments`. If no JSON, write tested text
  parsers and pin the sparkrun version.
- **V3 — Detached lifecycle + reconnect.** Confirm a detached `sparkrun run` survives an agent
  restart and that `sparkrun status` (with `--tp`) reliably re-identifies it for
  reconciliation and `stop`.
- **V4 — vLLM `/metrics` reachable** on `localhost:{port}` of the head node under
  sparkrun-launched containers (parity with today's scrape).
- **V5 — sparkrun version pinning.** Pin the `uvx sparkrun` version used by agents +
  provisioner so CLI-output parsing and flag names don't drift under us.

## 7. Testing Plan (per CLAUDE.md risk tiers)

This touches multi-node coordination + deployment → **high-risk tier**.

- **Property test** — `buildSparkrunArgs`: invariants — `--tp` equals host count for cluster
  deploys; port/served-model overrides always forwarded; `-o k=v` round-trips each override;
  never emits eugr `run-recipe.sh` flags.
- **Unit test** — `recipePath` validation: traversal (`../`), absolute-escape, non-existent,
  non-file all rejected; in-tree path accepted.
- **Unit test** — parsers for `sparkrun list` / `show` / `status` against captured fixture
  output (happy + malformed).
- **Integration test** (supertest, stub `agentHub`) — `POST /api/deployments` happy path emits
  the expected `cmd:deploy` with a sparkrun `recipeRef`; error path (missing recipe / VRAM
  shortfall / bad `recipePath`) returns the right 4xx and creates no Deployment.
- **Integration test** — VRAM admission unchanged behavior against a per-suite SQLite (existing
  `deployments.vram-admission.test.ts` pattern).
- **Manual / real-DGX** — actual `sparkrun run` solo + 2-node TP, `sparkrun stop`, reconnect
  reconciliation, and metrics scrape. These are environmental; documented in the PR with the
  manual steps performed (per CLAUDE.md "can't add an automated test → say so").

## 8. Out of Scope (YAGNI)

- Runtime-specific metrics/health adapters for sglang / llama.cpp / trtllm (deferred until a
  real recipe needs one — D6).
- Dashboard UI for authoring/uploading custom recipes (API-by-path only — D5).
- Server-side (`uvx`-on-manager) orchestration / removing agents from the deploy path
  (Design A — explicitly not chosen).
- Changes to fine-tune training orchestration.

## 9. Rollout

Behind the existing deploy flow — no schema migration required (Deployment/recipe records keep
their shape; `recipeFile` semantics widen to "recipe ref or path"). Reversible: the eugr path is
deleted in the same change, so rollback = revert the branch. Land provisioning (§4.2) and the
launcher (§4.1) together so a freshly provisioned node can actually deploy.
