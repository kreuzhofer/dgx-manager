# amd64 / RTX 5090 Recipe Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the manager able to deploy a curated set of small LLMs on the amd64 RTX 5090 node (`aihost01`, 32 GB), since every existing sparkrun recipe targets arm64 + GB10 and none will run there — and establish a reusable, arch-aware recipe strategy so we don't rebuild the library from scratch.

**Architecture:** Keep the entire sparkrun launch/lifecycle machinery untouched. Add an **arch dimension** to recipes end-to-end: (1) a small custom sparkrun registry of amd64 recipes built on the multi-arch upstream `vllm/vllm-openai` image, registered automatically on amd64 nodes by the agent; (2) arch tagging + per-node filtering of the recipe catalog in the manager so the dashboard only offers compatible recipes; (3) Ollama as the immediate zero-friction fallback path. The DGX-Spark recipes stay arm64-only and are simply hidden from amd64 nodes.

**Tech Stack:** Node.js agent (`packages/agent`), Express 5 server (`packages/server`), Prisma/SQLite, sparkrun (uvx), vLLM (`vllm/vllm-openai` amd64), Ollama, Vitest + fast-check + supertest.

---

## ⚠️ Coordination note (concurrent work on this branch)

Another process is actively working on `feat/hf-cache-management`. **Do not edit shared agent/server files until that work lands or this moves to its own branch.** Recommended: execute this plan in a fresh worktree/branch (`feat/amd64-recipes`) created from `main` once the HF-cache work is merged. The new files (custom registry, arch helper) are additive and low-conflict; the touch-points in `recipes.ts`, `agent-hub.ts`, and `deployments.ts` are the conflict risk — rebase those last.

---

## Findings (evidence behind this plan)

**Current state — verified live (2026-06-13):**
- `aihost01` = node `cmqb6l6ws1q8i36o0gqjeut5e`, IP `192.168.44.30`, **amd64**, **RTX 5090 / 32607 MiB**, online, agent `v0.5.563`, all prereqs green (Docker 29.5.3, nvidia-container-toolkit 1.19.1, Node 22, **Ollama v0.20.7**, NFS `/mnt/tank` from `192.168.44.22:/tank`). Currently **idle, zero deployments**.
- The agent already detects arch (`process.arch` → `amd64`/`arm64`, `index.ts:27`) and reports it on register; the server persists `Node.arch` (`agent-hub.ts:151`). **The arch signal already exists end-to-end — it is just not used for recipe selection.**
- aihost01 reports the **full 48 vLLM recipe catalog** to the manager, identical to the arm64 Sparks. **Every one of those 48 would fail on the 5090** (see below). This is the core bug to fix: the catalog is not arch-filtered.

**DGX-Spark / arm64 dependencies (the blockers — flagged):**
1. **Container images are arm64 + GB10 only.** `ghcr.io/spark-arena/dgx-vllm-eugr-nightly[-tf5]`, `avarok/atlas-gb10`, `scitrera/dgx-spark-*` — all arm64 GB10 builds. Confirmed `vllm/vllm-openai` *is* multi-arch (has an `amd64` manifest); the DGX images are not pullable/runnable on amd64.
2. **`sm_121a` / `compute_121` build target.** eugr Dockerfile builds NCCL/cutlass/FlashInfer/vLLM for `sm_121a` (GB10). The RTX 5090 is `sm_120` (consumer Blackwell) — different target; the prebuilt GB10 kernels do not apply.
3. **NVFP4 dominant quantization (39 recipes).** Gated behind sm_121a kernels + Marlin/cutlass-moe backends + `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` + `--load-format fastsafetensors|instanttensor`. Not portable to amd64/sm_120 as-is.
4. **Driver 580.x requirement** (590.x has a GB10 unified-memory CUDAGraph deadlock). aihost01 runs a generic Ubuntu 24.04 driver — different constraint.
5. **GB10-gated clustering.** eugr `autodiscover.sh` only adds peers whose `nvidia-smi` name == `NVIDIA GB10`. Multi-node is GB10-only by construction; amd64 is **solo-only**.
6. **Unified-memory assumption.** Grace-Blackwell coherent memory lets recipes assume huge effective memory. The 5090 has **discrete 32 GB** — `--max-model-len` and model size must be sized for 32 GB, not unified memory.

**Conclusion the user anticipated:** Yes, we need our own amd64 recipes — but **not from scratch**. We reuse 100% of sparkrun's launcher/parser/lifecycle and the upstream multi-arch `vllm/vllm-openai` image; only the recipe YAMLs (≈6–10 small models) and arch-routing glue are new.

---

## Strategy (two tracks, both recommended)

**Track A — Ollama (immediate, low risk).** Ollama v0.20.7 is already installed and working on aihost01, and the agent already has an Ollama runtime (`runtime/ollama.ts`) + catalog selection. Small GGUF models run today with near-zero code change. This is the fastest "run a similar set of models on amd64" win. Use for: quick demos, models where vLLM parity isn't required.

**Track B — Custom amd64 vLLM sparkrun registry (parity).** Author a tiny git registry (`dgx-manager-amd64-recipes`) of v2 sparkrun recipes that set `container: vllm/vllm-openai:<cuda12.8 tag>` + a standard vLLM serve `command`, sized for 32 GB, using fp8/awq/gguf (no nvfp4, no GB10 backends). Register it on amd64 nodes; the agent's existing `sparkrun list --json` picks it up unchanged. This gives OpenAI-compatible high-throughput serving matching the DGX experience, with full reuse of the existing deploy path.

**Arch routing (both tracks):** Tag every recipe with a derived `arch` (`arm64` for the DGX registries, `amd64` for the new registry / ollama), store it, and filter the per-node catalog so the dashboard only offers recipes whose arch matches the node. This fixes the current "48 incompatible recipes shown on the 5090" bug regardless of track.

**Initial amd64 model lineup (fits 32 GB with margin):**
| Model | Quant | ~VRAM | Runtime |
|---|---|---|---|
| Qwen3-1.7B | bf16 | ~4 GB | vllm |
| Qwen3-4B | bf16 | ~9 GB | vllm |
| Qwen3-8B | fp8 | ~9 GB | vllm |
| Llama-3.1-8B-Instruct | fp8 | ~9 GB | vllm |
| Qwen2.5-14B-Instruct | AWQ-int4 | ~10 GB | vllm |
| (fallback) Qwen3-14B Q4_K_M | gguf | ~9 GB | ollama |

(Exact list finalized in Task 6 after the Phase 0 probe confirms what vLLM-amd64 actually serves on sm_120.)

---

## Phase 0 — Empirical validation (do this FIRST, before any code)

This phase is manual/CLI verification on the live cluster; it produces the evidence that the code changes are built on. No code changes here.

### Task 0.1: Confirm an existing recipe fails on aihost01 (negative control)

**Files:** none (live API).

- [ ] **Step 1: Pick the smallest existing recipe and attempt a deploy to aihost01.**

```bash
# smallest single-node vllm recipe in the catalog
curl -s -X POST http://localhost:4000/api/deployments \
  -H 'content-type: application/json' \
  -d '{"nodeId":"cmqb6l6ws1q8i36o0gqjeut5e","recipePath":"@sparkrun-transitional/qwen3-1.7b-vllm","runtime":"vllm"}'
```

- [ ] **Step 2: Watch the agent logs and capture the failure.**

Run: `docker compose logs server -f | grep -i aihost` and on the node, the sparkrun log.
Expected: failure pulling/running the arm64 GB10 container on amd64 (manifest/arch error). **Record the exact error** — it is the justification for arch-filtering.

- [ ] **Step 3: Clean up the failed deployment record** (per project rule — never leave a trail).

```bash
curl -s -X DELETE http://localhost:4000/api/deployments/<id>
```

### Task 0.2: Prove the amd64 vLLM path works (positive control)

**Files:** none (live, hand-rolled inline recipe).

- [ ] **Step 1: Hand-author a minimal amd64 inline recipe and deploy via the inline path.**

```bash
curl -s -X POST http://localhost:4000/api/deployments \
  -H 'content-type: application/json' \
  -d '{"nodeId":"cmqb6l6ws1q8i36o0gqjeut5e","runtime":"vllm","recipeYaml":"name: qwen3-1_7b-amd64\nrecipe_version: 2\nruntime: vllm\ncontainer: vllm/vllm-openai:latest\nmodel: Qwen/Qwen3-1.7B\ncommand: --model Qwen/Qwen3-1.7B --gpu-memory-utilization 0.85 --max-model-len 8192 --port {port}\n"}'
```

- [ ] **Step 2: Confirm it serves.** Hit `/lb/` or the node's vLLM port with an OpenAI `/v1/models` request. Expected: model listed, a completion returns.
  - If `vllm/vllm-openai:latest` lacks sm_120 (RTX 5090 Blackwell) support, pin a CUDA-12.8 tag known to support Blackwell and retry. **Record the working tag** — it becomes the `container:` value for every Track-B recipe.

- [ ] **Step 3: Stop + delete the probe deployment** (cleanup).

- [ ] **Step 4: Decision gate.** If Step 2 succeeds → Track B is viable, proceed. If it fails for sm_120 reasons → fall back to Track A (Ollama) as primary and note the vLLM blocker in the plan before proceeding.

---

## Phase 1 — Arch-aware recipe catalog (manager + agent)

This is the structural fix and is independent of which track ships recipes. TDD throughout.

### Task 1: Pure helper — derive recipe arch from registry/runtime

**Files:**
- Create: `packages/agent/src/runtime/recipe-arch.ts`
- Test: `packages/agent/src/runtime/recipe-arch.test.ts`

- [ ] **Step 1: Write the failing test.**

```ts
import { describe, it, expect } from "vitest";
import { deriveRecipeArch } from "./recipe-arch.js";

describe("deriveRecipeArch", () => {
  it("tags the amd64 registry as amd64", () => {
    expect(deriveRecipeArch("@dgx-amd64/qwen3-1.7b-vllm")).toBe("amd64");
  });
  it("tags ollama recipes as any", () => {
    expect(deriveRecipeArch("ollama:qwen3:8b")).toBe("any");
  });
  it("defaults DGX/sparkrun registries to arm64", () => {
    expect(deriveRecipeArch("@sparkrun-transitional/qwen3-1.7b-vllm")).toBe("arm64");
    expect(deriveRecipeArch("@official/qwen3.6-27b-fp8-vllm")).toBe("arm64");
  });
});
```

- [ ] **Step 2: Run to verify it fails.** Run: `npx vitest run packages/agent/src/runtime/recipe-arch.test.ts` → FAIL (module not found).

- [ ] **Step 3: Implement.**

```ts
export type RecipeArch = "amd64" | "arm64" | "any";

const AMD64_REGISTRIES = ["@dgx-amd64"];

/** Derive the target CPU arch of a recipe from its ref.
 *  amd64 = our custom upstream-vLLM registry; any = ollama (multi-arch);
 *  everything else is a DGX-Spark (GB10/arm64) registry. */
export function deriveRecipeArch(ref: string): RecipeArch {
  if (ref.startsWith("ollama:")) return "any";
  if (AMD64_REGISTRIES.some((r) => ref.startsWith(r + "/"))) return "amd64";
  return "arm64";
}
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit.** `git add packages/agent/src/runtime/recipe-arch.ts packages/agent/src/runtime/recipe-arch.test.ts && git commit -m "feat(agent): derive recipe target arch from ref"`

### Task 2: Carry `arch` on the wire Recipe shape

**Files:**
- Modify: `packages/agent/src/recipes.ts` (the `toRecipe` mapper + `Recipe` type)
- Test: `packages/agent/src/recipes.test.ts` (add a case; create if absent)

- [ ] **Step 1: Write the failing test** asserting `toRecipe` populates `arch` via `deriveRecipeArch`.

```ts
import { describe, it, expect } from "vitest";
import { toRecipe } from "./recipes.js";

it("stamps arch onto the wire recipe", () => {
  const r = toRecipe({ ref: "@sparkrun-transitional/qwen3-1.7b-vllm", name: "x", minNodes: 1 });
  expect(r.arch).toBe("arm64");
});
```

- [ ] **Step 2: Run → FAIL** (`arch` missing).

- [ ] **Step 3: Implement.** Add `arch: RecipeArch` to the `Recipe` type and set `arch: deriveRecipeArch(summary.ref)` in `toRecipe`. Import from `./runtime/recipe-arch.js`.

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit.** `git commit -am "feat(agent): include arch on wire Recipe"`

### Task 3: Server persists + exposes recipe arch; filters per node

**Files:**
- Modify: `packages/server/src/ws/agent-hub.ts` (recipe ingest — keep `arch` field)
- Modify: `packages/server/src/routes/recipes.ts` (accept `?nodeId=` filter)
- Test: `packages/server/src/routes/recipes.test.ts` (supertest, stub agentHub)

- [ ] **Step 1: Write the failing supertest** — `GET /api/recipes?nodeId=<amd64 node>` returns only `arch in {amd64, any}`; without `nodeId` returns all.

```ts
// mount only the recipes router; inject a stub agentHub exposing a fixed catalog
// catalog: [{ref:"@dgx-amd64/a", arch:"amd64"}, {ref:"@official/b", arch:"arm64"}, {ref:"ollama:c", arch:"any"}]
// node lookup stub: amd64 node -> {arch:"amd64"}
it("filters recipes to the node's arch", async () => {
  const res = await request(app).get("/api/recipes?nodeId=amd64node");
  const refs = res.body.map((r:any)=>r.ref);
  expect(refs).toEqual(expect.arrayContaining(["@dgx-amd64/a","ollama:c"]));
  expect(refs).not.toContain("@official/b");
});
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** the `?nodeId=` filter: look up `node.arch`; keep recipes where `recipe.arch === node.arch || recipe.arch === "any"`. No `nodeId` → unfiltered (back-compat). Ensure agent-hub stores `arch` on ingested recipes (default `"arm64"` for legacy agents that don't send it — explicit, observable fallback).

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit.** `git commit -am "feat(server): arch-filter recipe catalog by node"`

### Task 4: Admission guard — reject arch-mismatched deploys (fail fast)

**Files:**
- Modify: `packages/server/src/routes/deployments.ts` (deploy validation)
- Test: `packages/server/src/__tests__/integration/deployments.arch-admission.test.ts`

- [ ] **Step 1: Write the failing integration test** — deploying an `arm64` recipe ref to an `amd64` node returns 4xx with a clear arch-mismatch message; matching arch (or inline/ollama) is allowed.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** a pure `checkRecipeArchAdmission(recipeArch, nodeArch)` helper (so it's property-testable) + wire it into the registry-ref deploy branch. Inline recipes (`recipeYaml`) and ollama bypass (user takes responsibility / arch-agnostic). Mirror the existing vram-admission split pattern (`admission/vram.ts`).

- [ ] **Step 4: Run → PASS.** Add a property test for the pure helper (symmetry/total over arch pairs).

- [ ] **Step 5: Commit.** `git commit -am "feat(server): fail-fast on recipe/node arch mismatch"`

### Task 5: Dashboard — only offer compatible recipes

**Files:**
- Modify: dashboard deploy UI recipe selector (`packages/dashboard/...` — locate the deploy form that lists recipes) + `lib/api.ts` recipe fetch to pass the selected `nodeId`.

- [ ] **Step 1:** Change the recipe fetch in the deploy flow to call `/api/recipes?nodeId=<selected node>` so the dropdown is pre-filtered. (User-visible change first, per project principle 2.)
- [ ] **Step 2:** Manually verify in the browser: selecting aihost01 shows only amd64/ollama recipes; selecting a Spark shows the arm64 set.
- [ ] **Step 3: Commit.** `git commit -am "feat(dashboard): filter deploy recipe list by node arch"`

---

## Phase 2 — Track A: Ollama recipes on amd64 (immediate win)

### Task 6a: Verify + register the Ollama lineup for amd64

**Files:**
- Modify (if needed): `packages/agent/src/runtime/ollama.ts` / ollama catalog source.

- [ ] **Step 1:** Confirm the existing ollama deploy path works on aihost01 by deploying `qwen3:8b` (or similar) via the API to node `cmqb6l6ws...`. Verify it serves on the ollama port. (Ollama already runs an embedding deploy elsewhere — the path is proven; this confirms it on amd64.)
- [ ] **Step 2:** Curate the small-model ollama catalog entries (1.7B–14B Q4) so they surface as `arch: any` recipes.
- [ ] **Step 3:** Clean up the test deploy. Commit any catalog change.

---

## Phase 3 — Track B: Custom amd64 vLLM sparkrun registry

### Task 6: Author the `dgx-manager-amd64-recipes` registry

**Files:**
- Create: new git repo / directory of v2 sparkrun recipe YAMLs (one per model in the lineup). Each:

```yaml
name: qwen3-8b-fp8-vllm
recipe_version: 2
runtime: vllm
container: vllm/vllm-openai:<tag-confirmed-in-Task-0.2>
model: Qwen/Qwen3-8B
metadata:
  description: Qwen3 8B (fp8) for amd64 + consumer Blackwell (RTX 5090, 32GB)
  quantization: fp8
  category: chat
min_nodes: 1
max_nodes: 1
defaults:
  gpu_mem: 0.85
  max_model_len: 16384
command: >-
  --model Qwen/Qwen3-8B --quantization fp8
  --gpu-memory-utilization {gpu_mem} --max-model-len {max_model_len} --port {port}
```

- [ ] **Step 1:** Author one recipe (Qwen3-1.7B bf16) and validate it deploys + serves on aihost01 via the registry path (not inline). Iterate on flags until clean.
- [ ] **Step 2:** Author the rest of the lineup (Task 0/strategy table), each validated to fit 32 GB and serve.
- [ ] **Step 3:** Document the registry URL and pin in the repo README.

### Task 7: Agent auto-registers the amd64 registry on amd64 nodes

**Files:**
- Modify: `packages/agent/src/recipes.ts` (or a new `runtime/registry-setup.ts`) — on startup, if `AGENT_ARCH === "amd64"`, ensure the `@dgx-amd64` registry is present in the node's sparkrun config (write/merge `~/.config/sparkrun/registries.yaml` or call `sparkrun registry add`) before `discoverRecipes()`.
- Test: `packages/agent/src/runtime/registry-setup.test.ts` (pure config-merge function — test that it adds the registry idempotently without clobbering existing ones).

- [ ] **Step 1: Write the failing test** for an idempotent `mergeRegistry(existingYaml, entry)` pure function.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the pure merge + the arch-gated call site.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5:** Bump agent version (`./scripts/bump-agent-version.sh` — MANDATORY after agent src edits) and commit.

### Task 8: End-to-end validation on the live 5090

- [ ] **Step 1:** Rebuild agent bundles (`./scripts/build-agent-bundles.sh`) + redeploy manager (`docker compose up -d --build`); trigger agent update on aihost01.
- [ ] **Step 2:** Confirm aihost01 now reports the amd64 + ollama recipes (and the dashboard hides arm64 ones).
- [ ] **Step 3:** Deploy 2–3 models from the registry, confirm OpenAI-compatible serving through `/lb/`, record VRAM actuals.
- [ ] **Step 4:** Clean up all validation deployments.

---

## Self-review notes / open decisions

- **`vllm/vllm-openai` sm_120 support is the single biggest risk** — Phase 0 Task 0.2 is the gate. If the stock image doesn't support consumer Blackwell, options: (a) pin a newer tag, (b) build a thin amd64 vLLM image, or (c) lean on Track A (Ollama) as primary. Decision deferred to Task 0.2 evidence.
- **Registry hosting:** a public/private git repo is simplest (sparkrun clones registries). Alternatively ship the YAMLs inside the agent bundle and register a local path — avoids a network dependency. Recommend local-path-in-bundle for hermeticity; revisit if recipes need to change without an agent redeploy.
- **Cleanup discipline:** every probe/validation deployment above must be `DELETE`d (project rule). No failed/scratch records left behind.
- **No contract breaks:** `/api/recipes` without `nodeId` stays unfiltered; legacy agents (no `arch` on recipes) default to `arm64` — explicit and observable.

## Execution handoff

Recommend executing **after** the concurrent `feat/hf-cache-management` work merges, in a fresh `feat/amd64-recipes` worktree off `main`. Phase 0 (live probe) gates everything and needs no code.
