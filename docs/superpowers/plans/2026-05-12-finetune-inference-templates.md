# Fine-tune Inference Templates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each fine-tune recipe in `dgx-manager-fine-tune-recipes` owns its own `inference.yaml` (colocated with `recipe.yaml`). The manager materializes it into `spark-vllm-docker/recipes/finetune-<jobId-12>.yaml` at deploy time — substituting model path and served name — instead of auto-generating a one-size-fits-all command from hardcoded flags.

**Architecture:** Recipe-side colocation: the training repo owns the full lifecycle (train → merge → serve). The manager's `generateLocalModelRecipe` becomes a thin orchestrator that prefers `inference.yaml` if present and falls back to the existing minimal auto-gen if not. No new cross-repo protocol; `spark-vllm-docker` stays focused on base-model recipes.

**Tech Stack:** TypeScript (Node.js agent + Express server), Vitest, fs/path stdlib, YAML-as-string templates (no parser dependency — pure text substitution on well-known fields).

---

## File Structure

**New files:**

| Path | Purpose |
|---|---|
| `packages/agent/src/runtime/inference-template.ts` | Pure helpers: `findInferenceTemplate(recipeDir)`, `applyFinetuneSubstitutions(yaml, params)`, `MERGED_PATH_PLACEHOLDER` constant |
| `packages/agent/src/runtime/inference-template.test.ts` | Unit tests for both helpers — pure, no fs mocks beyond `mkdtempSync` |
| `<fine-tune-recipes>/recipes/qwen3.6-27b-base-lora-attn-mlp/inference.yaml` | First concrete inference template; mirrors `spark-vllm-docker/recipes/qwen3.6-27b-bf16.yaml` with the `qwen3_xml` parser override and `{{MERGED_MODEL_PATH}}` placeholder. **Lives in the dgx-manager-fine-tune-recipes repo, not in dgx-manager.** |

**Modified files:**

| Path | Change |
|---|---|
| `packages/agent/src/runtime/vllm.ts` | `generateLocalModelRecipe` gets a new branch: if `recipeDir` arg points to a dir containing `inference.yaml`, materialize from that template; else keep current minimal auto-gen path |
| `packages/agent/src/index.ts` | `cmd:finetune:deploy` handler accepts a new `recipeFile` payload field and passes it as `recipeDir` to `generateLocalModelRecipe` |
| `packages/server/src/routes/finetune.ts` | Deploy route adds `recipeFile: job.recipeFile` to the `cmd:finetune:deploy` payload (~3 line change) |
| `packages/agent/package.json` | Patch bump (e.g. 0.5.343 → 0.5.344) — Task 9 |

**Test files:** Unit tests live next to source as `inference-template.test.ts`. No integration test needed — the deploy path is exercised end-to-end by the existing manual smoke flow (Task 10).

**Out of scope for this plan:** any `extends:` mechanism for sharing inference.yaml across sibling recipes. We'll revisit if 3+ recipes accumulate near-identical templates.

---

## Task 1: Failing unit test for `applyFinetuneSubstitutions`

**Files:**
- Create: `packages/agent/src/runtime/inference-template.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect } from "vitest";
import { applyFinetuneSubstitutions, MERGED_PATH_PLACEHOLDER } from "./inference-template.js";

describe("applyFinetuneSubstitutions", () => {
  it("replaces {{MERGED_MODEL_PATH}} placeholder and injects served_model_name", () => {
    const input = `recipe_version: "1"
name: qwen3.6-27b-bf16
model: ${MERGED_PATH_PLACEHOLDER}
container: vllm-node

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 1
  max_model_len: 32768

command: |
  vllm serve ${MERGED_PATH_PLACEHOLDER} \\
    --host {host} \\
    --port {port} \\
    --max-model-len {max_model_len} \\
    -tp {tensor_parallel}
`;

    const out = applyFinetuneSubstitutions(input, {
      modelPath: "/workspace/outputs/cmp073lno00mn36p0bhffd2q4/merged",
      servedModelName: "chat3d-build123d-01",
    });

    // Placeholder is replaced wherever it appears
    expect(out).not.toContain(MERGED_PATH_PLACEHOLDER);
    expect(out).toContain("model: /workspace/outputs/cmp073lno00mn36p0bhffd2q4/merged");
    expect(out).toContain("vllm serve /workspace/outputs/cmp073lno00mn36p0bhffd2q4/merged");
    // served_model_name added to defaults
    expect(out).toMatch(/^defaults:[\s\S]*?served_model_name: chat3d-build123d-01/m);
    // Other content preserved verbatim
    expect(out).toContain("tensor_parallel: 1");
    expect(out).toContain("port: 8000");
  });

  it("is idempotent if served_model_name is already declared", () => {
    const input = `defaults:
  port: 8000
  served_model_name: existing-name

command: |
  vllm serve ${MERGED_PATH_PLACEHOLDER}
`;
    const out = applyFinetuneSubstitutions(input, {
      modelPath: "/path/to/merged",
      servedModelName: "new-name",
    });
    // Existing served_model_name wins — author intent is preserved
    expect(out).toContain("served_model_name: existing-name");
    expect(out).not.toContain("served_model_name: new-name");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/inference-template.test.ts`
Expected: FAIL with `Cannot find module './inference-template.js'`

---

## Task 2: Implement `applyFinetuneSubstitutions`

**Files:**
- Create: `packages/agent/src/runtime/inference-template.ts`

- [ ] **Step 1: Write the minimal implementation**

```typescript
/**
 * Materialize a fine-tune's inference template by substituting the merged-model
 * path placeholder and injecting served_model_name into the defaults block.
 *
 * Inference templates live next to training recipes (e.g.
 * `<fine-tune-recipes>/recipes/<name>/inference.yaml`). They look like a
 * regular spark-vllm-docker recipe but use the literal placeholder
 * {{MERGED_MODEL_PATH}} wherever the local model path needs to land.
 *
 * Substitution is plain text — we don't parse YAML — to avoid imposing a
 * specific YAML library on the agent and to keep round-trips byte-exact
 * for hand-tuned comments and whitespace.
 */
export const MERGED_PATH_PLACEHOLDER = "{{MERGED_MODEL_PATH}}";

export interface SubstitutionParams {
  modelPath: string;        // absolute path inside the container, e.g. /workspace/outputs/<jobId>/merged
  servedModelName: string;  // friendly name to report via /v1/models
}

export function applyFinetuneSubstitutions(
  yaml: string,
  params: SubstitutionParams,
): string {
  // 1. Replace every occurrence of {{MERGED_MODEL_PATH}} with the merged path.
  let out = yaml.split(MERGED_PATH_PLACEHOLDER).join(params.modelPath);

  // 2. Inject served_model_name into defaults: block — unless author already
  // declared one (then we preserve their intent; see test 2).
  const hasServedName = /^\s*served_model_name:\s*\S/m.test(out);
  if (!hasServedName) {
    out = out.replace(
      /^defaults:\s*\n/m,
      `defaults:\n  served_model_name: ${params.servedModelName}\n`,
    );
  }

  return out;
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/runtime/inference-template.test.ts`
Expected: PASS — both `it()` blocks green

- [ ] **Step 3: Commit**

```bash
git add packages/agent/src/runtime/inference-template.ts packages/agent/src/runtime/inference-template.test.ts
git commit -m "agent: pure substitution helper for fine-tune inference templates"
```

---

## Task 3: Failing test for `findInferenceTemplate`

**Files:**
- Modify: `packages/agent/src/runtime/inference-template.test.ts`

- [ ] **Step 1: Append the failing test**

Add this block at the bottom of `packages/agent/src/runtime/inference-template.test.ts`:

```typescript
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { findInferenceTemplate } from "./inference-template.js";

describe("findInferenceTemplate", () => {
  it("returns the path when inference.yaml exists in the recipe dir", () => {
    const tmp = mkdtempSync(join(tmpdir(), "inftpl-"));
    try {
      const recipeDir = join(tmp, "recipes", "qwen3.6-27b-base-lora-attn-mlp");
      mkdirSync(recipeDir, { recursive: true });
      const target = join(recipeDir, "inference.yaml");
      writeFileSync(target, "recipe_version: \"1\"\nname: test\n");

      expect(findInferenceTemplate(recipeDir)).toBe(target);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("returns null when the recipe dir has no inference.yaml", () => {
    const tmp = mkdtempSync(join(tmpdir(), "inftpl-"));
    try {
      const recipeDir = join(tmp, "recipes", "qwen3.6-27b-base-lora-attn-only");
      mkdirSync(recipeDir, { recursive: true });
      // Note: no inference.yaml written
      expect(findInferenceTemplate(recipeDir)).toBeNull();
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("returns null when the recipe dir does not exist at all", () => {
    expect(findInferenceTemplate("/nonexistent/path/never/created")).toBeNull();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/runtime/inference-template.test.ts`
Expected: FAIL with `findInferenceTemplate is not a function` (or similar import error)

---

## Task 4: Implement `findInferenceTemplate`

**Files:**
- Modify: `packages/agent/src/runtime/inference-template.ts`

- [ ] **Step 1: Append the implementation**

Add at the end of `packages/agent/src/runtime/inference-template.ts`:

```typescript
import { existsSync } from "fs";
import { join } from "path";

/**
 * Return the absolute path to `<recipeDir>/inference.yaml` if it exists,
 * else null. Used by the deploy path to decide whether to inherit a
 * hand-authored serve config (Task 5 caller) or fall through to the
 * minimal auto-gen.
 */
export function findInferenceTemplate(recipeDir: string): string | null {
  const candidate = join(recipeDir, "inference.yaml");
  return existsSync(candidate) ? candidate : null;
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/runtime/inference-template.test.ts`
Expected: PASS — all 5 `it()` blocks green (2 from Task 1 + 3 from Task 3)

- [ ] **Step 3: Commit**

```bash
git add packages/agent/src/runtime/inference-template.ts packages/agent/src/runtime/inference-template.test.ts
git commit -m "agent: locate inference.yaml sibling of a fine-tune recipe"
```

---

## Task 5: `generateLocalModelRecipe` prefers inference.yaml when present

**Files:**
- Modify: `packages/agent/src/runtime/vllm.ts:49-124` (the `generateLocalModelRecipe` function)

The current function takes ~14 params and hardcodes a YAML template. We're adding one new param (`recipeDir?: string`) and an early-return branch.

- [ ] **Step 1: Add the imports**

At the top of `packages/agent/src/runtime/vllm.ts` (around line 5), add:

```typescript
import { findInferenceTemplate, applyFinetuneSubstitutions } from "./inference-template.js";
```

- [ ] **Step 2: Extend the `generateLocalModelRecipe` signature**

In `packages/agent/src/runtime/vllm.ts`, find the params block of `generateLocalModelRecipe` (currently lines 49-68). Add a new optional field at the bottom of the param object:

```typescript
  // Absolute filesystem path to the training recipe's directory (e.g.
  // `<fine-tune-recipes>/recipes/qwen3.6-27b-base-lora-attn-mlp`). If this
  // directory contains an `inference.yaml`, that file is used verbatim as
  // the serve template — only `{{MERGED_MODEL_PATH}}` is substituted and
  // `served_model_name` is injected into the defaults block. If absent or
  // the file doesn't exist, the existing minimal auto-gen path runs.
  recipeDir?: string;
```

- [ ] **Step 3: Add the early-return branch**

In `generateLocalModelRecipe`, immediately after line 72 (`const fullPath = join(VLLM_REPO_PATH, recipeFile);`) and BEFORE the `containerModelPath` line, add:

```typescript
  // Prefer a hand-authored inference.yaml colocated with the training
  // recipe. Lets each fine-tune family own its full lifecycle (train +
  // merge + serve) with the right vLLM flags for its weights.
  if (params.recipeDir) {
    const templatePath = findInferenceTemplate(params.recipeDir);
    if (templatePath) {
      const tmpl = readFileSync(templatePath, "utf-8");
      const materialized = applyFinetuneSubstitutions(tmpl, {
        modelPath: params.modelPath.replace(`${SHARED_STORAGE}/`, `${WORKSPACE}/`),
        servedModelName: params.servedModelName || recipeName,
      });
      mkdirSync(join(VLLM_REPO_PATH, "recipes"), { recursive: true });
      writeFileSync(fullPath, materialized, "utf-8");
      console.log(
        `Generated vLLM recipe from template: ${fullPath} ` +
        `(source=${templatePath})`
      );
      return recipeFile;
    }
  }
```

- [ ] **Step 4: Run the agent test suite**

Run: `npm test -- packages/agent`
Expected: PASS — all 82+ tests green, no regressions

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/vllm.ts
git commit -m "agent: prefer fine-tune recipe's inference.yaml in generateLocalModelRecipe"
```

---

## Task 6: Server passes `recipeFile` in `cmd:finetune:deploy`

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (around lines 736-756 where the payload is constructed)

- [ ] **Step 1: Add the recipeFile field**

In `packages/server/src/routes/finetune.ts`, find the `agentHub.sendToAgent(headNodeId, { type: "cmd:finetune:deploy", payload: { ... } })` block. Inside the `payload:` object, add the new field next to `modelName`:

```typescript
      // Relative path of the training recipe (e.g.
      // "recipes/qwen3.6-27b-base-lora-attn-mlp"). The agent resolves this
      // to an absolute dir and looks for a sibling inference.yaml to use
      // as the vLLM serve template. If absent, deploy falls back to the
      // legacy minimal auto-gen.
      recipeFile: job.recipeFile,
```

- [ ] **Step 2: Verify the server still compiles**

Run: `npm run build -- --workspace packages/server`
Expected: tsc completes with 0 new errors (any pre-existing errors in unrelated test files are not introduced by this change)

- [ ] **Step 3: Commit**

```bash
git add packages/server/src/routes/finetune.ts
git commit -m "server: send recipeFile path in cmd:finetune:deploy payload"
```

---

## Task 7: Agent unpacks `recipeFile` and resolves the recipe directory

**Files:**
- Modify: `packages/agent/src/index.ts` (the `cmd:finetune:deploy` handler block, around lines 781-825)

The agent currently destructures `{ jobId, deploymentId, modelPath, deployContainer, config, clusterNodes, clusterNodeFastIps, modelName }` from the payload. Add `recipeFile`.

- [ ] **Step 1: Find the SHARED_STORAGE + recipes-repo path constant**

Open `packages/agent/src/index.ts` and search for an existing reference to the fine-tune-recipes repo path. It's typically derived as `${SHARED_STORAGE}/src/github/dgx-manager-fine-tune-recipes`. Confirm what variable holds it (likely `process.env.SHARED_STORAGE`-derived inside `packages/agent/src/training-recipes.ts`).

If no such constant exists in `index.ts`, add it near the top of the file (after the existing const declarations around line 23):

```typescript
const FINETUNE_RECIPES_REPO = process.env.FINETUNE_RECIPES_REPO
  || `${process.env.SHARED_STORAGE || "/mnt/tank"}/src/github/dgx-manager-fine-tune-recipes`;
```

- [ ] **Step 2: Extend the destructure**

In the `cmd:finetune:deploy` handler, change:

```typescript
      const {
        jobId, deploymentId, modelPath, deployContainer, config,
        clusterNodes, clusterNodeFastIps, modelName,
      } = msg.payload as {
        jobId: string;
        deploymentId: string;
        modelPath: string;
        deployContainer?: string;
        config?: Record<string, unknown>;
        clusterNodes?: string[];
        clusterNodeFastIps?: (string | null)[];
        modelName?: string;
      };
```

to:

```typescript
      const {
        jobId, deploymentId, modelPath, deployContainer, config,
        clusterNodes, clusterNodeFastIps, modelName, recipeFile,
      } = msg.payload as {
        jobId: string;
        deploymentId: string;
        modelPath: string;
        deployContainer?: string;
        config?: Record<string, unknown>;
        clusterNodes?: string[];
        clusterNodeFastIps?: (string | null)[];
        modelName?: string;
        recipeFile?: string;
      };
```

- [ ] **Step 3: Pass `recipeDir` to `generateLocalModelRecipe`**

In the same handler, find the `generateLocalModelRecipe({...})` call. Add `recipeDir` to the params, derived from `recipeFile` if present:

```typescript
        const recipeFileRecipeDir = recipeFile
          ? join(FINETUNE_RECIPES_REPO, recipeFile)
          : undefined;

        const recipeFileGenerated = generateLocalModelRecipe({
          jobId,
          modelPath,
          container: deployContainer || "vllm-node",
          port,
          gpuMemoryUtilization: gpuMem,
          maxModelLen,
          isCluster,
          tensorParallel: tensorParallel ?? 1,
          pipelineParallel: pipelineParallel ?? 1,
          servedModelName: modelName,
          recipeDir: recipeFileRecipeDir,
        });
```

Note: the existing variable name in this handler is `recipeFile` (the result of `generateLocalModelRecipe`). Avoid colliding — rename the local variable that captures the function's return value to `recipeFileGenerated` (or similar), and adjust references below it accordingly.

- [ ] **Step 4: Verify the agent test suite**

Run: `npm test -- packages/agent`
Expected: PASS — no regressions; new path is exercised only when `recipeFile` and `inference.yaml` both exist (no test fixture for that here — Task 10 is the e2e smoke).

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/index.ts
git commit -m "agent: wire recipeFile from deploy payload into generateLocalModelRecipe"
```

---

## Task 8: Author inference.yaml for `qwen3.6-27b-base-lora-attn-mlp`

This file lives in the **other** repo (`dgx-manager-fine-tune-recipes`), not in `dgx-manager`. Path on the mounted filesystem is `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora-attn-mlp/inference.yaml`.

**Files:**
- Create: `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora-attn-mlp/inference.yaml`

- [ ] **Step 1: Write the inference template**

```yaml
# Recipe: Qwen3.6-27B fine-tune (LoRA attn+MLP merged → bf16)
# Generated from this template when the manager deploys the fine-tune.
# {{MERGED_MODEL_PATH}} is substituted with the container-visible path
# to the merged weights at deploy time; served_model_name is injected
# into defaults: from the fine-tune job's friendly name.
#
# Mirrors spark-vllm-docker/recipes/qwen3.6-27b-bf16.yaml with two
# differences:
#   * --tool-call-parser qwen3_xml (this fine-tune's chat template emits
#     <tool_call><function=name><parameter=...> XML, not the base
#     model's qwen3_coder shape)
#   * model path is a placeholder, not an HF id

recipe_version: "1"
name: qwen3.6-27b-base-lora-attn-mlp-bf16
description: vLLM serving the LoRA-merged Qwen3.6-27B fine-tune in BF16

model: {{MERGED_MODEL_PATH}}
container: vllm-node

mods:
  - mods/fix-qwen3-coder-next

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 2
  gpu_memory_utilization: 0.85
  max_model_len: 128000
  max_num_batched_tokens: 8192

env:
  VLLM_MARLIN_USE_ATOMIC_ADD: 1
  VLLM_DISTRIBUTED_EXECUTOR_CONFIG: '{"placement_group_options":{"strategy":"SPREAD"}}'

command: |
  vllm serve {{MERGED_MODEL_PATH}} \
    --host {host} \
    --port {port} \
    --max-model-len {max_model_len} \
    --max-num-batched-tokens {max_num_batched_tokens} \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --trust-remote-code \
    -tp {tensor_parallel} \
    --distributed-executor-backend ray
```

- [ ] **Step 2: Commit in the recipes repo**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
git add recipes/qwen3.6-27b-base-lora-attn-mlp/inference.yaml
git commit -m "qwen3.6-27b attn-mlp: add inference.yaml for fine-tune deploys"
cd -
```

(This commit lives in the `dgx-manager-fine-tune-recipes` repo, not `dgx-manager`. Each fine-tune's serve template lives next to its training recipe by design.)

---

## Task 9: Bump agent version

**Files:**
- Modify: `packages/agent/package.json` via the bump script

- [ ] **Step 1: Run the bump script**

```bash
./scripts/bump-agent-version.sh
```

Expected: stdout shows `Agent version bumped: 0.5.X → 0.5.X+1`

- [ ] **Step 2: Run the full test suite**

Run: `npm test`
Expected: 82+ tests pass (or whatever the current baseline is); no new failures.

- [ ] **Step 3: Commit**

```bash
git add packages/agent/package.json
git commit -m "agent: bump version (inference templates)"
```

---

## Task 10: End-to-end smoke test (manual)

This task validates the full path on the live cluster. No automated coverage — the deploy flow touches the real DGX nodes.

- [ ] **Step 1: Rebuild bundles + server**

```bash
./scripts/build-agent-bundles.sh
MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build server
```

Wait for `curl -s http://192.168.44.36:4000/api/agent/version` to return the new version.

- [ ] **Step 2: Update agents on all 4 nodes**

```bash
for id in $(curl -s http://192.168.44.36:4000/api/nodes | jq -r '.[].id'); do
  curl -s -X POST "http://192.168.44.36:4000/api/nodes/$id/update-agent"
done
```

Wait until `curl -s http://192.168.44.36:4000/api/nodes | jq '[.[] | .agentVersion]'` shows all 4 on the new version.

- [ ] **Step 3: Stop the existing chat3d-build123d-01 deployment**

```bash
DEPL_ID=$(curl -s http://192.168.44.36:4000/api/deployments \
  | jq -r '.[] | select(.model.name == "chat3d-build123d-01" and .status == "running") | .id' | head -1)
curl -s -X DELETE "http://192.168.44.36:4000/api/deployments/$DEPL_ID"
```

Wait ~10s for the container to fully stop.

- [ ] **Step 4: Relaunch via the finetune deploy API**

```bash
curl -s -X POST http://192.168.44.36:4000/api/finetune/cmp073lno00mn36p0bhffd2q4/deploy \
  -H "Content-Type: application/json" \
  -d '{"nodeIds":["cmno92lcz006s36o3k3yijvbp","cmoo6lc8e00nq36r6zihfjfz7"],"config":{"port":8000,"maxModelLen":128000,"tensorParallel":2}}'
```

Expected: 200 with a new deployment id.

- [ ] **Step 5: Verify the materialized YAML came from the template**

```bash
cat /mnt/tank/src/github/spark-vllm-docker/recipes/finetune-cmp073lno00m.yaml
```

Expected:
- `--max-num-batched-tokens`, `--kv-cache-dtype fp8`, `--attention-backend flashinfer`, `--enable-prefix-caching`, `--trust-remote-code` all present (proving inference.yaml was the source, NOT the minimal auto-gen)
- `served_model_name: chat3d-build123d-01` in defaults
- `model:` resolves to `/workspace/outputs/cmp073lno00mn36p0bhffd2q4/merged` (placeholder substituted)

- [ ] **Step 6: Wait for the endpoint to come up and probe tool_choice + throughput**

```bash
until curl -s --max-time 5 http://192.168.44.37:8000/v1/models 2>/dev/null \
  | jq -e '.data[0].id == "chat3d-build123d-01"' >/dev/null 2>&1; do sleep 15; done

# Verify tool_choice=auto still works (parser fix preserved)
curl -s --max-time 60 -X POST http://192.168.44.37:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"chat3d-build123d-01","messages":[{"role":"user","content":"hi"}],"tool_choice":"auto","tools":[{"type":"function","function":{"name":"noop","parameters":{"type":"object"}}}],"max_tokens":4}' \
  | jq '.choices[0].message.tool_calls'
```

Expected: 200 with the right model id; the tool_choice probe returns either `null` (model chose not to call) or a properly-shaped `tool_calls[]` — not a 400.

- [ ] **Step 7: Compare throughput vs the previous bf16-without-flags run**

Run a short throughput probe:

```bash
curl -s --max-time 120 -X POST http://192.168.44.37:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"chat3d-build123d-01","messages":[{"role":"user","content":"Write a 50x30x5mm baseplate with two 5mm-diameter through-holes spaced 30mm apart in build123d. Just code."}],"max_tokens":400,"temperature":0}' \
  | jq '{completion_tokens: .usage.completion_tokens, tps: (.usage.completion_tokens / 1)}'
```

Expected: a meaningful uplift from the previously-observed ~6 tps (target ≥10 tps for a single request; bigger gain expected at concurrency >1 thanks to prefix caching + flashinfer).

- [ ] **Step 8: Post to #agent-room confirming the inference-template path works**

(Done via the existing /loop webhook flow — no manual posting needed if a cron is armed. Otherwise: a short `[dgx-claude] proposes: ...` reply citing the materialized YAML and the throughput numbers.)

---

## Self-Review

**Spec coverage:**
- ✅ Colocate inference.yaml with training recipe → Tasks 5+8
- ✅ Manager prefers template if present, falls back if not → Task 5 (the `if (templatePath)` branch + the existing untouched code below it)
- ✅ Substitute model path and served name → Tasks 1+2 (covered by tests)
- ✅ Server sends recipe path → Task 6
- ✅ Agent looks up sibling inference.yaml → Task 7
- ✅ First concrete template ships with this plan → Task 8
- ✅ Backward compat (existing fine-tunes without inference.yaml keep working) → Task 5's fall-through behavior
- ✅ Validation that the path is exercised end-to-end → Task 10

**Placeholder scan:** All steps have concrete code or exact commands. No "TODO" / "implement later" / "similar to Task N". Method names (`applyFinetuneSubstitutions`, `findInferenceTemplate`, `MERGED_PATH_PLACEHOLDER`) are consistent across all tasks.

**Type consistency:**
- `applyFinetuneSubstitutions(yaml: string, params: SubstitutionParams): string` — defined Task 2, called Task 5
- `findInferenceTemplate(recipeDir: string): string | null` — defined Task 4, called Task 5
- `recipeDir?: string` on `generateLocalModelRecipe` — added Task 5, populated Task 7
- `recipeFile?: string` payload field — added Task 6 (server), unpacked Task 7 (agent)
