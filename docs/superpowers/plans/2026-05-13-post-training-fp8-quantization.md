# Post-Training FP8 Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in post-merge FP8 quantization step to the dgx-manager fine-tune pipeline so operators can produce a vLLM-servable FP8 variant of any merged LoRA fine-tune and deploy either BF16 or FP8 from a single dashboard.

**Architecture:** Mirror the existing merge pipeline (`POST /api/finetune/:id/merge` → `cmd:finetune:merge` → `mergeLoraAdapter()` in agent runtime) one-for-one for quantization. State machine extends `merged → quantizing → quantized | failed` on `FineTuneJob`. Recipe repo gains a shared `scripts/quantize_fp8.py` (llmcompressor FP8_DYNAMIC W8A8 wrapper) and per-recipe `inference-fp8.yaml` template. Deploy endpoint and template-resolver gain an `artifactVariant: "bf16" | "fp8"` parameter to pick the right artifact dir and inference template.

**Tech Stack:** TypeScript / Node 22 / Express 5 / Prisma 7 (SQLite) / Next.js 15 / React 19 server, vitest + supertest + fast-check tests, Python (llmcompressor) inside `nvcr.io/nvidia/pytorch:25.11-py3` container on agent side, vLLM with `--quantization fp8` at serve time.

**Repos touched:**
- `dgx-manager` (this repo): server, agent, dashboard, prisma, tests
- `dgx-manager-fine-tune-recipes` at `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/`: shared `scripts/quantize_fp8.py`, per-recipe `inference-fp8.yaml`, `recipe.yaml` schema additions. **Commits in this repo are separate from dgx-manager commits.**

**Conventions to honour:**
- DRY/YAGNI/TDD/frequent commits per CLAUDE.md.
- Every change to `packages/agent/src/**` MUST be followed by `./scripts/bump-agent-version.sh` before commit (mandatory dashboard signal).
- Integration tests follow the pattern in `packages/server/src/__tests__/integration/deployments.vram-admission.test.ts`: per-suite SQLite via `mkdtempSync` + `DATABASE_URL=file:/tmp/...` set BEFORE importing prisma, schema applied via `npx prisma db push --force-reset` with `PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION` set per-suite (NEVER globally).
- Pre-commit hooks must run; do not pass `--no-verify`.

**Scope guardrails (out of scope here):**
- AWQ / INT4 / NVFP4 / other quant formats (FP8_DYNAMIC W8A8 only).
- KV-cache dtype changes (already FP8 in current template; unchanged).
- Multi-node quantization (single-GB10, single-container is sufficient for 27B).
- Re-quantization triggers / dataset-change detection. One-shot per merged artifact.
- Auto-quantize-after-merge (it stays opt-in for now; operator presses a button).

**State machine (extends current `mergeStatus`):**

```
status: completed → mergeStatus: running → mergeStatus: completed
                                              ↓
                            quantizationStatus: pending  (default after migration)
                                              ↓ POST /:id/quantize
                                          quantizing
                                              ↓ agent:finetune:quantize-complete
                                         quantized | failed
```

`quantizationStatus` defaults to `pending` for any merged job (existing rows backfill to `pending` via migration). `quantized` is the only state that lets `POST /:id/deploy` with `artifactVariant: "fp8"` succeed. Other variants always succeed (BF16 path is unchanged).

---

## File Structure

**New files (dgx-manager repo):**

- `packages/agent/src/runtime/finetune-quantize.ts` — quantize runtime (extracted from finetune.ts to keep that file from growing further; the existing merge code stays where it is for now, since moving it is out of scope and risky)
- `packages/server/src/__tests__/integration/finetune.quantize.test.ts` — endpoint integration tests
- `packages/agent/src/runtime/finetune-quantize.test.ts` — unit tests for the pure helpers (container arg builder, log-line progress detector)

**New files (recipe repo):**

- `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/scripts/quantize_fp8.py` — llmcompressor FP8_DYNAMIC wrapper, shared across all recipes
- `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora-attn-mlp/inference-fp8.yaml` — FP8 inference template for the canonical recipe (other recipes get this added later as needed)

**Modified files (dgx-manager repo):**

- `prisma/schema.prisma` — FineTuneJob gets 4 new columns
- `packages/server/src/routes/finetune.ts` — add `POST /:id/quantize`, extend `POST /:id/deploy` with `artifactVariant`
- `packages/agent/src/training-recipes.ts` — parser picks up `scripts.quantize_fp8`
- `packages/server/src/agent-hub.ts` — broadcast `agent:finetune:quantize-progress|complete` to dashboards as `finetune:quantize-progress|status`
- `packages/agent/src/index.ts` — handle `cmd:finetune:quantize`
- `packages/agent/src/runtime/inference-template.ts` — `findInferenceTemplate(recipeDir, variant)` chooses bf16 vs fp8 file
- `packages/agent/src/runtime/finetune.ts` — `inference-fp8.yaml` selection at deploy time (caller passes `artifactVariant`)
- `packages/dashboard/app/finetune/page.tsx` — Quantize button + artifact selector in deploy modal

**Modified files (recipe repo):**

- `recipes/qwen3.6-27b-base-lora-attn-mlp/recipe.yaml` — add `scripts.quantize_fp8: scripts/quantize_fp8.py`

---

## Task 1: Add the shared quantize_fp8.py script (recipe repo)

**Repo:** `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/` — distinct from dgx-manager.

**Files:**
- Create: `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/scripts/quantize_fp8.py`

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""
FP8 quantization wrapper for merged fine-tune artifacts.

Runs llmcompressor's FP8_DYNAMIC W8A8 scheme against a merged BF16
model directory, writing the quantized weights to a sibling directory.
Designed to be called from the dgx-manager agent quantize runtime.

Usage:
    python quantize_fp8.py --model-dir /path/to/merged \\
                           --output-dir /path/to/merged-fp8

Exit codes:
    0 — success, output dir contains a vLLM-loadable FP8 model
    2 — bad arguments
    3 — model load failed
    4 — quantization failed
    5 — save failed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Path to merged BF16 model dir")
    p.add_argument("--output-dir", required=True, help="Where to write FP8 weights")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    if not model_dir.is_dir():
        print(f"ERROR: model dir does not exist: {model_dir}", file=sys.stderr)
        return 2
    if not (model_dir / "config.json").exists():
        print(f"ERROR: missing config.json in {model_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[quantize_fp8] Loading model from {model_dir}", flush=True)
    t0 = time.time()
    try:
        # Local imports keep cold-start fast for argparse failures.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmcompressor.transformers import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: import failed: {e}", file=sys.stderr)
        return 3

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: load failed: {e}", file=sys.stderr)
        return 3
    print(f"[quantize_fp8] Model loaded in {time.time() - t0:.1f}s", flush=True)

    print("[quantize_fp8] Applying FP8_DYNAMIC W8A8 quantization", flush=True)
    t1 = time.time()
    try:
        recipe = QuantizationModifier(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=["lm_head"],
        )
        oneshot(model=model, recipe=recipe)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: quantization failed: {e}", file=sys.stderr)
        return 4
    print(f"[quantize_fp8] Quantization complete in {time.time() - t1:.1f}s", flush=True)

    print(f"[quantize_fp8] Saving FP8 model to {output_dir}", flush=True)
    t2 = time.time()
    try:
        model.save_pretrained(str(output_dir), save_compressed=True)
        tokenizer.save_pretrained(str(output_dir))
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: save failed: {e}", file=sys.stderr)
        return 5
    print(f"[quantize_fp8] FP8 model saved in {time.time() - t2:.1f}s", flush=True)

    # Sanity: vLLM looks at config.json's `quantization_config` block.
    cfg = output_dir / "config.json"
    if not cfg.exists():
        print(f"ERROR: missing config.json after save: {cfg}", file=sys.stderr)
        return 5

    print("[quantize_fp8] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make executable + sanity-check syntax**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
chmod +x scripts/quantize_fp8.py
python3 -c "import ast; ast.parse(open('scripts/quantize_fp8.py').read())"
```

Expected: no output (parse success).

- [ ] **Step 3: --help works**

```bash
python3 /mnt/tank/src/github/dgx-manager-fine-tune-recipes/scripts/quantize_fp8.py --help
```

Expected: argparse usage text with `--model-dir` and `--output-dir`.

- [ ] **Step 4: Commit (recipe repo)**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
git add scripts/quantize_fp8.py
git commit -m "scripts: add quantize_fp8.py (llmcompressor FP8_DYNAMIC W8A8 wrapper)

Shared post-merge quantization step for fine-tunes. Loads a merged BF16
model dir, applies FP8_DYNAMIC quantization (Linear layers, lm_head
ignored), saves to output dir as a vLLM-loadable artifact."
```

---

## Task 2: Add inference-fp8.yaml for qwen3.6-27b-base-lora-attn-mlp (recipe repo)

**Repo:** `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/`

**Files:**
- Create: `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora-attn-mlp/inference-fp8.yaml`

- [ ] **Step 1: Create the FP8 inference template**

```yaml
# Recipe: Qwen3.6-27B fine-tune (LoRA attn+MLP merged → FP8 quantized)
# Generated from this template when the manager deploys the fine-tune.
# {{MERGED_MODEL_PATH}} is substituted with the container-visible path
# to the FP8-quantized weights at deploy time (i.e. .../merged-fp8/).
#
# Mirrors inference.yaml but:
#   * model path is the FP8 quantized artifact, not the BF16 merged dir
#   * command gains --quantization fp8 (vLLM needs this when serving
#     FP8 weights that are stored with FP8_DYNAMIC scales)
#   * Weights are ~27 GB (vs ~54 GB BF16), so tensor_parallel could
#     drop to 1; kept at 2 here to match the BF16 sibling so behavior
#     is identical except for precision. Override at deploy time if
#     you want single-GPU serving.

recipe_version: "1"
name: qwen3.6-27b-base-lora-attn-mlp-fp8
description: vLLM serving the LoRA-merged Qwen3.6-27B fine-tune in FP8 (FP8_DYNAMIC W8A8)

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
    --served-model-name {served_model_name} \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --trust-remote-code \
    -tp {tensor_parallel} \
    --distributed-executor-backend ray
```

- [ ] **Step 2: Diff against BF16 sibling to confirm only intended changes**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
diff recipes/qwen3.6-27b-base-lora-attn-mlp/inference.yaml \
     recipes/qwen3.6-27b-base-lora-attn-mlp/inference-fp8.yaml
```

Expected diff: header comment differs, `name:` differs, `description:` differs, command gains `--quantization fp8` line. No other changes.

- [ ] **Step 3: Commit (recipe repo)**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
git add recipes/qwen3.6-27b-base-lora-attn-mlp/inference-fp8.yaml
git commit -m "qwen3.6-27b attn-mlp: add inference-fp8.yaml for FP8-quantized deploys

Sibling to inference.yaml. Used by the dgx-manager deploy path when the
operator picks artifactVariant=fp8 — points at merged-fp8/ instead of
merged/ and tells vLLM the weights are FP8_DYNAMIC quantized."
```

---

## Task 3: Add scripts.quantize_fp8 to recipe.yaml (recipe repo)

**Repo:** `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/`

**Files:**
- Modify: `recipes/qwen3.6-27b-base-lora-attn-mlp/recipe.yaml`

- [ ] **Step 1: Add the scripts.quantize_fp8 field**

The current `scripts:` block ends with the `merge:` line. Add the new field directly below. Find this in `recipes/qwen3.6-27b-base-lora-attn-mlp/recipe.yaml`:

```yaml
scripts:
  entrypoint: entrypoint.sh
  train: train.py
  launch: launch.sh
  ds_config: ds_config.json
  # Reuses the 35B-A3B merge script: ...
  merge: scripts/merge_qwen3moe.py
```

Replace with:

```yaml
scripts:
  entrypoint: entrypoint.sh
  train: train.py
  launch: launch.sh
  ds_config: ds_config.json
  # Reuses the 35B-A3B merge script: ...
  merge: scripts/merge_qwen3moe.py
  # FP8 quantization wrapper. Shared script — same input contract
  # (--model-dir, --output-dir) so the agent quantize runtime is
  # recipe-agnostic.
  quantize_fp8: scripts/quantize_fp8.py
```

- [ ] **Step 2: Validate the YAML still parses**

```bash
python3 -c "import yaml; yaml.safe_load(open('/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora-attn-mlp/recipe.yaml'))" && echo OK
```

Expected: `OK`

- [ ] **Step 3: Commit (recipe repo)**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
git add recipes/qwen3.6-27b-base-lora-attn-mlp/recipe.yaml
git commit -m "qwen3.6-27b attn-mlp: wire scripts.quantize_fp8 into recipe.yaml

Points to the new shared scripts/quantize_fp8.py. Picked up by the
dgx-manager agent's training-recipes parser and surfaced to the server
deploy/quantize endpoints."
```

---

## Task 4: Prisma migration — add quantization columns to FineTuneJob

**Repo:** `dgx-manager`

**Files:**
- Modify: `prisma/schema.prisma`

- [ ] **Step 1: Add the four new columns**

Open `prisma/schema.prisma` and find the `model FineTuneJob {` block. Locate the line `mergedPath   String?` and the line `deploymentId String?` (they are consecutive). Insert the four new columns between `mergedPath` and `deploymentId` so the BF16 / FP8 artifact fields cluster together:

Before:

```prisma
  mergeStatus  String?
  mergedPath   String?
  deploymentId String?
```

After:

```prisma
  mergeStatus          String?
  mergedPath           String?
  // FP8 quantization (post-merge). Defaults to "pending" once mergeStatus
  // is "completed"; transitions to "quantizing"→"quantized"|"failed" via
  // POST /api/finetune/:id/quantize. Independent of mergeStatus so the
  // BF16 deploy path can ignore it entirely.
  quantizationStatus   String?  // null | "pending" | "quantizing" | "quantized" | "failed"
  quantizedPath        String?
  quantizationLog      String?
  quantizedAt          DateTime?
  deploymentId         String?
```

- [ ] **Step 2: Generate Prisma client + push schema**

```bash
cd /home/daniel/src/github/dgx-manager
npm run db:generate
PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION="dev schema iteration on local SQLite" \
  npx prisma db push
```

Expected: `Your database is now in sync with your Prisma schema.`

- [ ] **Step 3: Confirm the columns exist**

```bash
sqlite3 /home/daniel/src/github/dgx-manager/dev.db ".schema FineTuneJob" | grep -E 'quantization|quantized'
```

Expected: four lines matching `quantizationStatus`, `quantizedPath`, `quantizationLog`, `quantizedAt`.

- [ ] **Step 4: Commit**

```bash
cd /home/daniel/src/github/dgx-manager
git add prisma/schema.prisma
git commit -m "prisma: add FP8 quantization columns to FineTuneJob

Additive migration: quantizationStatus, quantizedPath, quantizationLog,
quantizedAt. All nullable; existing rows are unaffected and continue to
deploy BF16 as before. Migration is additive — no breaking change."
```

---

## Task 5: Training-recipes parser picks up scripts.quantize_fp8

**Repo:** `dgx-manager`

**Files:**
- Modify: `packages/agent/src/training-recipes.ts:174`
- Modify: same file's `TrainingRecipe` interface

- [ ] **Step 1: Write the failing unit test**

Create `packages/agent/src/training-recipes.test.ts` if it does not already exist (check first with `ls packages/agent/src/training-recipes.test.ts`). If it does, append; otherwise create:

```typescript
import { describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { discoverTrainingRecipes } from "./training-recipes.js";

describe("training-recipes parser", () => {
  it("picks up scripts.quantize_fp8 when present", () => {
    const root = mkdtempSync(join(tmpdir(), "recipes-"));
    const recipeDir = join(root, "recipes", "test-recipe");
    mkdirSync(recipeDir, { recursive: true });
    writeFileSync(
      join(recipeDir, "recipe.yaml"),
      [
        "recipe_version: \"1\"",
        "name: test",
        "base_model: foo/bar",
        "framework: deepspeed",
        "method: lora",
        "scripts:",
        "  entrypoint: entrypoint.sh",
        "  train: train.py",
        "  launch: launch.sh",
        "  merge: scripts/merge.py",
        "  quantize_fp8: scripts/quantize_fp8.py",
        "",
      ].join("\n"),
    );
    const recipes = discoverTrainingRecipes(root);
    expect(recipes).toHaveLength(1);
    expect(recipes[0]!.scripts.quantize_fp8).toBe("scripts/quantize_fp8.py");
    rmSync(root, { recursive: true, force: true });
  });

  it("leaves quantize_fp8 undefined when recipe omits it", () => {
    const root = mkdtempSync(join(tmpdir(), "recipes-"));
    const recipeDir = join(root, "recipes", "test-recipe");
    mkdirSync(recipeDir, { recursive: true });
    writeFileSync(
      join(recipeDir, "recipe.yaml"),
      "recipe_version: \"1\"\nname: test\nbase_model: foo/bar\n",
    );
    const recipes = discoverTrainingRecipes(root);
    expect(recipes[0]!.scripts.quantize_fp8).toBeUndefined();
    rmSync(root, { recursive: true, force: true });
  });
});
```

If `discoverTrainingRecipes` is not exported, that's the failure mode of Step 2 below.

- [ ] **Step 2: Run the test — expect FAIL**

```bash
cd /home/daniel/src/github/dgx-manager
npx vitest run packages/agent/src/training-recipes.test.ts
```

Expected: test FAILS because `scripts.quantize_fp8` is not part of the parsed shape (still typed `merge` only).

- [ ] **Step 3: Extend the parser**

In `packages/agent/src/training-recipes.ts`, find the `TrainingRecipe` interface (search for `interface TrainingRecipe` or `export interface TrainingRecipe`). Find the `scripts:` block in that type, currently containing `entrypoint`, `train`, `launch`, `ds_config`, `merge`. Add a `quantize_fp8?: string` field immediately after `merge`:

```typescript
  scripts: {
    entrypoint: string;
    train: string;
    launch: string;
    ds_config?: string;
    merge?: string;
    /** Post-merge FP8 quantization wrapper (shared scripts/quantize_fp8.py).
     *  Repo-relative. When absent, POST /api/finetune/:id/quantize returns
     *  501 for that recipe — opt-in per recipe. */
    quantize_fp8?: string;
  };
```

Then in the parser body around line 174 (where `merge: scripts.merge as string | undefined,` is), add a sibling line:

```typescript
        scripts: {
          entrypoint: (scripts.entrypoint as string) || "entrypoint.sh",
          train: (scripts.train as string) || "train.py",
          launch: (scripts.launch as string) || "launch.sh",
          ds_config: scripts.ds_config as string | undefined,
          merge: scripts.merge as string | undefined,
          quantize_fp8: scripts.quantize_fp8 as string | undefined,
        },
```

- [ ] **Step 4: Re-run the test — expect PASS**

```bash
npx vitest run packages/agent/src/training-recipes.test.ts
```

Expected: 2/2 tests pass.

- [ ] **Step 5: Bump agent version**

```bash
cd /home/daniel/src/github/dgx-manager
./scripts/bump-agent-version.sh
```

Expected: prints the new version (e.g. `0.5.362`).

- [ ] **Step 6: Commit**

```bash
git add packages/agent/src/training-recipes.ts packages/agent/src/training-recipes.test.ts packages/agent/package.json
git commit -m "agent: training-recipes parser picks up scripts.quantize_fp8

New optional field on the scripts block. When absent the recipe is not
quantizable and POST /api/finetune/:id/quantize will surface 501 to the
caller. When present it's the repo-relative path to the quantization
wrapper script (mirrors scripts.merge). Bumped agent version."
```

---

## Task 6: Server endpoint — POST /api/finetune/:id/quantize

**Repo:** `dgx-manager`

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (insert new endpoint after the existing `POST /:id/merge` block ending at line 623)

- [ ] **Step 1: Write the failing integration test**

Create `packages/server/src/__tests__/integration/finetune.quantize.test.ts`:

```typescript
/**
 * Integration tests for POST /api/finetune/:id/quantize.
 *
 * Same pattern as deployments.vram-admission.test.ts: per-suite SQLite,
 * stub agent hub, supertest against an Express app that mounts only
 * the finetune router.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-quantize-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let finetuneRouter: typeof import("../../routes/finetune.js").finetuneRouter;

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
  ({ finetuneRouter } = await import("../../routes/finetune.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

async function wipeAll() {
  await prisma.fineTuneClusterNode.deleteMany();
  await prisma.deployment.deleteMany();
  await prisma.model.deleteMany();
  await prisma.fineTuneJob.deleteMany();
  await prisma.node.deleteMany();
}

beforeEach(async () => { await wipeAll(); });

function makeStubHub(recipe: { file: string; scripts: { quantize_fp8?: string; merge?: string } }) {
  const sent: { nodeId: string; message: unknown }[] = [];
  return {
    hub: {
      getTrainingRecipes: () => [recipe],
      sendToAgent: (nodeId: string, message: unknown) => sent.push({ nodeId, message }),
    },
    sent,
  };
}

function makeApp(hub: { getTrainingRecipes: () => unknown[]; sendToAgent: (...a: unknown[]) => void }) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.use("/api/finetune", finetuneRouter);
  return app;
}

async function seedMergedJob(opts: { quantizationStatus?: string | null; recipeFile?: string } = {}) {
  const node = await prisma.node.create({
    data: { id: "n1", name: "n1", ipAddress: "10.0.0.1", agentPort: 8089, status: "online", vramTotal: 122000 },
  });
  return prisma.fineTuneJob.create({
    data: {
      nodeId: node.id,
      recipeFile: opts.recipeFile ?? "recipes/test-recipe",
      baseModel: "Qwen/Qwen3.6-27B",
      method: "lora",
      dataset: "/tmp/ds.jsonl",
      status: "completed",
      mergeStatus: "completed",
      mergedPath: "/mnt/tank/outputs/job-1/merged",
      outputDir: "/mnt/tank/outputs/job-1",
      quantizationStatus: opts.quantizationStatus ?? null,
    },
  });
}

describe("POST /api/finetune/:id/quantize", () => {
  it("happy path: kicks the agent and transitions to quantizing", async () => {
    const { hub, sent } = makeStubHub({
      file: "recipes/test-recipe",
      scripts: { quantize_fp8: "scripts/quantize_fp8.py", merge: "scripts/merge.py" },
    });
    const app = makeApp(hub);
    const job = await seedMergedJob();

    const res = await request(app).post(`/api/finetune/${job.id}/quantize`).send({});
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("quantizing");
    expect(res.body.quantizedPath).toBe("/mnt/tank/outputs/job-1/merged-fp8");

    const updated = await prisma.fineTuneJob.findUnique({ where: { id: job.id } });
    expect(updated?.quantizationStatus).toBe("quantizing");

    expect(sent).toHaveLength(1);
    expect(sent[0]!.nodeId).toBe("n1");
    const msg = sent[0]!.message as { type: string; payload: { jobId: string; quantizeScript: string } };
    expect(msg.type).toBe("cmd:finetune:quantize");
    expect(msg.payload.jobId).toBe(job.id);
    expect(msg.payload.quantizeScript).toBe("scripts/quantize_fp8.py");
  });

  it("idempotent: already-quantized returns 200 with existing path, does not re-send", async () => {
    const { hub, sent } = makeStubHub({
      file: "recipes/test-recipe",
      scripts: { quantize_fp8: "scripts/quantize_fp8.py" },
    });
    const app = makeApp(hub);
    const job = await seedMergedJob({ quantizationStatus: "quantized" });
    await prisma.fineTuneJob.update({
      where: { id: job.id },
      data: { quantizedPath: "/mnt/tank/outputs/job-1/merged-fp8" },
    });

    const res = await request(app).post(`/api/finetune/${job.id}/quantize`).send({});
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("quantized");
    expect(res.body.quantizedPath).toBe("/mnt/tank/outputs/job-1/merged-fp8");
    expect(sent).toHaveLength(0);
  });

  it("returns 400 when mergeStatus is not completed", async () => {
    const { hub } = makeStubHub({ file: "recipes/test-recipe", scripts: { quantize_fp8: "scripts/quantize_fp8.py" } });
    const app = makeApp(hub);
    const job = await seedMergedJob();
    await prisma.fineTuneJob.update({ where: { id: job.id }, data: { mergeStatus: "running" } });

    const res = await request(app).post(`/api/finetune/${job.id}/quantize`).send({});
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/merge/i);
  });

  it("returns 501 when recipe lacks scripts.quantize_fp8", async () => {
    const { hub } = makeStubHub({ file: "recipes/test-recipe", scripts: { merge: "scripts/merge.py" } });
    const app = makeApp(hub);
    const job = await seedMergedJob();

    const res = await request(app).post(`/api/finetune/${job.id}/quantize`).send({});
    expect(res.status).toBe(501);
    expect(res.body.error).toMatch(/quantize/i);
  });

  it("returns 409 when quantization already in progress", async () => {
    const { hub, sent } = makeStubHub({ file: "recipes/test-recipe", scripts: { quantize_fp8: "scripts/quantize_fp8.py" } });
    const app = makeApp(hub);
    const job = await seedMergedJob({ quantizationStatus: "quantizing" });

    const res = await request(app).post(`/api/finetune/${job.id}/quantize`).send({});
    expect(res.status).toBe(409);
    expect(sent).toHaveLength(0);
  });
});
```

- [ ] **Step 2: Run the test — expect FAIL**

```bash
cd /home/daniel/src/github/dgx-manager
npx vitest run packages/server/src/__tests__/integration/finetune.quantize.test.ts
```

Expected: all five `describe.it` cases FAIL — endpoint doesn't exist yet (404 from supertest).

- [ ] **Step 3: Implement the endpoint**

In `packages/server/src/routes/finetune.ts`, insert this block immediately after the existing `POST /:id/merge` handler (currently ending around line 623, before `POST /:id/deploy` at line 625):

```typescript
finetuneRouter.post("/:id/quantize", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });

  // Pre-flight: merge must be done.
  if (job.mergeStatus !== "completed" || !job.mergedPath) {
    return res.status(400).json({ error: "Job must be merged before quantizing. Call POST /merge first." });
  }

  // Recipe must support quantization. Mirrors how scripts.merge is required
  // for the merge endpoint.
  const agentHub: AgentHub = req.app.get("agentHub");
  const recipe = job.recipeFile
    ? agentHub.getTrainingRecipes().find((r) => r.file === job.recipeFile)
    : undefined;
  const quantizeScript = recipe?.scripts.quantize_fp8;
  if (!quantizeScript) {
    return res.status(501).json({
      error: `Recipe ${job.recipeFile} does not declare scripts.quantize_fp8 — quantization not supported for this recipe.`,
    });
  }

  // Idempotency: already quantized → return existing artifact.
  if (job.quantizationStatus === "quantized" && job.quantizedPath) {
    return res.json({ status: "quantized", quantizedPath: job.quantizedPath });
  }

  // In-flight: refuse to re-kick.
  if (job.quantizationStatus === "quantizing") {
    return res.status(409).json({ error: "Quantization already in progress for this job." });
  }

  const quantizedOutputDir = `${job.outputDir ?? `${SHARED_STORAGE}/outputs/${job.id}`}/merged-fp8`;

  agentHub.sendToAgent(job.nodeId, {
    type: "cmd:finetune:quantize",
    payload: {
      jobId: job.id,
      mergedPath: job.mergedPath,
      quantizedOutputDir,
      quantizeScript,
    },
  });

  await prisma.fineTuneJob.update({
    where: { id: job.id },
    data: { quantizationStatus: "quantizing", quantizedPath: quantizedOutputDir },
  });

  res.json({ status: "quantizing", quantizedPath: quantizedOutputDir });
});
```

- [ ] **Step 4: Re-run the test — expect PASS**

```bash
npx vitest run packages/server/src/__tests__/integration/finetune.quantize.test.ts
```

Expected: 5/5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/routes/finetune.ts \
        packages/server/src/__tests__/integration/finetune.quantize.test.ts
git commit -m "server: POST /api/finetune/:id/quantize endpoint

Mirrors POST /:id/merge: validates pre-conditions (merge done, recipe
supports quantize_fp8, not already running), fires a cmd:finetune:quantize
WS message at the head node's agent, transitions FineTuneJob.quantization\\
Status to \"quantizing\". Idempotent on quantized state. Integration
tests cover happy path, idempotency, pre-merge guard, missing-recipe-script,
and in-flight 409."
```

---

## Task 7: Agent runtime — `quantizeMergedToFp8()` + `cmd:finetune:quantize` handler

**Repo:** `dgx-manager`

**Files:**
- Create: `packages/agent/src/runtime/finetune-quantize.ts`
- Create: `packages/agent/src/runtime/finetune-quantize.test.ts`
- Modify: `packages/agent/src/index.ts` (insert new case after `cmd:finetune:merge` at line 796)

- [ ] **Step 1: Write the failing unit test for the pure progress-detector helper**

Create `packages/agent/src/runtime/finetune-quantize.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { detectQuantizeProgress } from "./finetune-quantize.js";

describe("detectQuantizeProgress", () => {
  it("emits load progress for 'Loading model'", () => {
    expect(detectQuantizeProgress("[quantize_fp8] Loading model from /foo"))
      .toEqual({ phase: "loading", progress: 0.1 });
  });
  it("emits quantize progress for 'Applying FP8_DYNAMIC'", () => {
    expect(detectQuantizeProgress("[quantize_fp8] Applying FP8_DYNAMIC W8A8 quantization"))
      .toEqual({ phase: "quantizing", progress: 0.5 });
  });
  it("emits saving progress for 'Saving FP8 model'", () => {
    expect(detectQuantizeProgress("[quantize_fp8] Saving FP8 model to /foo"))
      .toEqual({ phase: "saving", progress: 0.85 });
  });
  it("emits final progress for 'OK'", () => {
    expect(detectQuantizeProgress("[quantize_fp8] OK"))
      .toEqual({ phase: "saving", progress: 1.0 });
  });
  it("returns null for unrelated lines", () => {
    expect(detectQuantizeProgress("some unrelated log line")).toBeNull();
    expect(detectQuantizeProgress("")).toBeNull();
  });
});
```

- [ ] **Step 2: Run the test — expect FAIL**

```bash
cd /home/daniel/src/github/dgx-manager
npx vitest run packages/agent/src/runtime/finetune-quantize.test.ts
```

Expected: import error (`detectQuantizeProgress` doesn't exist).

- [ ] **Step 3: Create the runtime module**

Create `packages/agent/src/runtime/finetune-quantize.ts`:

```typescript
/**
 * FP8 quantization runner for merged fine-tune artifacts.
 *
 * Mirrors mergeLoraAdapter() in runtime/finetune.ts but runs the
 * recipe's quantize_fp8 script (defaults to scripts/quantize_fp8.py)
 * against the merged BF16 dir, producing a sibling merged-fp8/ dir.
 *
 * Single-GB10 single-container — no multi-node coordination needed
 * for 27B-class models (output is ~27 GB, comfortably fits one
 * GB10's 122 GB unified pool with headroom for transient activations).
 */

import { spawn, execSync, type ChildProcess } from "child_process";
import { createWriteStream, mkdirSync } from "fs";
import { dirname, join } from "path";
import { SHARED_STORAGE, WORKSPACE, toContainerPath } from "../env.js";

export interface QuantizeCallbacks {
  onLog: (line: string) => void;
  onProgress: (phase: "loading" | "quantizing" | "saving", progress: number) => void;
  onComplete: (status: "completed" | "failed", outputPath?: string, error?: string) => void;
}

interface QuantizeProgressUpdate {
  phase: "loading" | "quantizing" | "saving";
  progress: number;
}

/**
 * Pure helper: classify a stdout line from quantize_fp8.py into a
 * progress update, or null if the line isn't a recognized milestone.
 * Exported so unit tests can exercise it without a docker container.
 */
export function detectQuantizeProgress(line: string): QuantizeProgressUpdate | null {
  const l = line.toLowerCase();
  if (l.includes("[quantize_fp8] loading model")) return { phase: "loading", progress: 0.1 };
  if (l.includes("[quantize_fp8] model loaded")) return { phase: "loading", progress: 0.3 };
  if (l.includes("[quantize_fp8] applying fp8_dynamic")) return { phase: "quantizing", progress: 0.5 };
  if (l.includes("[quantize_fp8] quantization complete")) return { phase: "quantizing", progress: 0.8 };
  if (l.includes("[quantize_fp8] saving fp8 model")) return { phase: "saving", progress: 0.85 };
  if (l.includes("[quantize_fp8] fp8 model saved")) return { phase: "saving", progress: 0.95 };
  if (l.includes("[quantize_fp8] ok")) return { phase: "saving", progress: 1.0 };
  return null;
}

/**
 * Quantize a merged BF16 model to FP8_DYNAMIC W8A8 using the recipe's
 * quantize_fp8 script. Writes to {mergedDir}/../merged-fp8/.
 */
export async function quantizeMergedToFp8(
  jobId: string,
  mergedPath: string,
  quantizedOutputDir: string,
  callbacks: QuantizeCallbacks,
  quantizeScriptRelative: string,
): Promise<void> {
  const containerName = `dgx-quantize-${jobId.slice(0, 12)}`;
  const containerImage = "nvcr.io/nvidia/pytorch:25.11-py3";
  const quantizeScript = `${WORKSPACE}/src/github/dgx-manager-fine-tune-recipes/${quantizeScriptRelative}`;

  const containerMergedPath = toContainerPath(mergedPath);
  const containerOutputDir = toContainerPath(quantizedOutputDir);

  const quantizeLogPath = join(dirname(quantizedOutputDir), "quantize.log");
  try { mkdirSync(dirname(quantizeLogPath), { recursive: true }); } catch { /* */ }
  const logStream = createWriteStream(quantizeLogPath, { flags: "a" });
  const tee = (s: string) => { callbacks.onLog(s); try { logStream.write(s); } catch { /* */ } };

  try {
    try { execSync(`docker rm -f ${containerName}`, { timeout: 15_000, stdio: "ignore" }); } catch { /* */ }

    tee(`[agent] Starting quantize container (script=${quantizeScriptRelative})\n`);
    callbacks.onProgress("loading", 0);

    const dockerArgs = [
      "run", "-d",
      "--name", containerName,
      "--gpus", "all",
      "--network", "host",
      "--ipc", "host",
      "--privileged",
      "--ulimit", "memlock=-1",
      "--shm-size=1g",
      "--user", "root",
      "-e", "CUDA_VISIBLE_DEVICES=0",
      "-e", "PYTHONUNBUFFERED=1",
      "-e", `HF_HOME=${WORKSPACE}/models`,
      "-v", `${SHARED_STORAGE}:${WORKSPACE}`,
      "--entrypoint", "sleep",
      containerImage,
      "infinity",
    ];

    execSync(`docker ${dockerArgs.join(" ")}`, { timeout: 120_000 });

    tee("[agent] Installing quantization dependencies...\n");
    try {
      execSync(
        `docker exec ${containerName} pip install -q llmcompressor transformers accelerate safetensors`,
        { timeout: 600_000, stdio: "ignore" },
      );
    } catch (err) {
      tee(`[agent] Failed to install quantization deps: ${err}\n`);
      callbacks.onComplete("failed", undefined, `Failed to install quantization deps: ${err}`);
      try { logStream.end(); } catch { /* */ }
      try { execSync(`docker rm -f ${containerName}`, { timeout: 15_000, stdio: "ignore" }); } catch { /* */ }
      return;
    }
    tee("[agent] Dependencies installed.\n");

    tee(`[agent] Running quantize: ${containerMergedPath} -> ${containerOutputDir}\n`);
    tee(`[agent] Script: ${quantizeScript}\n\n`);

    const quantProc: ChildProcess = spawn("docker", [
      "exec", containerName,
      "python", quantizeScript,
      "--model-dir", containerMergedPath,
      "--output-dir", containerOutputDir,
    ], { stdio: ["ignore", "pipe", "pipe"] });

    quantProc.stdout?.on("data", (data: Buffer) => {
      const text = data.toString();
      tee(text);
      for (const line of text.split("\n")) {
        const update = detectQuantizeProgress(line);
        if (update) callbacks.onProgress(update.phase, update.progress);
      }
    });
    quantProc.stderr?.on("data", (data: Buffer) => tee(data.toString()));

    await new Promise<void>((resolve) => {
      quantProc.on("exit", (code) => {
        tee(`\n[agent] Quantize process exited with code ${code}\n`);
        try {
          execSync(`docker exec ${containerName} chmod -R a+rw ${containerOutputDir}`, { timeout: 30_000, stdio: "ignore" });
        } catch { /* best effort */ }

        if (code === 0) {
          tee(`[agent] FP8 model saved to ${quantizedOutputDir}\n`);
          callbacks.onComplete("completed", quantizedOutputDir);
        } else {
          callbacks.onComplete("failed", undefined, `Quantize failed with exit code ${code}`);
        }
        try { logStream.end(); } catch { /* */ }
        resolve();
      });
    });
  } finally {
    try { execSync(`docker rm -f ${containerName}`, { timeout: 30_000, stdio: "ignore" }); } catch { /* */ }
  }
}
```

- [ ] **Step 4: Run the unit test — expect PASS**

```bash
npx vitest run packages/agent/src/runtime/finetune-quantize.test.ts
```

Expected: 5/5 pass.

- [ ] **Step 5: Wire the cmd handler in agent/src/index.ts**

In `packages/agent/src/index.ts`, find the `case "cmd:finetune:merge":` block ending at line 796 (the line `break;` right after the inner braces). Insert this new case immediately below:

```typescript
    case "cmd:finetune:quantize": {
      const { jobId, mergedPath, quantizedOutputDir, quantizeScript } = msg.payload as {
        jobId: string; mergedPath: string; quantizedOutputDir: string; quantizeScript: string;
      };

      console.log(`[finetune] Quantizing job ${jobId}: ${mergedPath} -> ${quantizedOutputDir} (script=${quantizeScript})`);
      const { quantizeMergedToFp8 } = await import("./runtime/finetune-quantize.js");
      quantizeMergedToFp8(jobId, mergedPath, quantizedOutputDir, {
        onLog: (line) => sendMsg("agent:finetune:quantize-progress", { jobId, log: line }),
        onProgress: (phase, phaseProgress) => sendMsg("agent:finetune:quantize-progress", { jobId, phase, phaseProgress }),
        onComplete: (status, outputPath, error) => {
          console.log(`[finetune] Quantize ${jobId} ${status}${error ? `: ${error}` : ""}`);
          sendMsg("agent:finetune:quantize-complete", {
            jobId, status, quantizedPath: outputPath ?? null, error: error ?? undefined,
          });
        },
      }, quantizeScript);
      break;
    }
```

- [ ] **Step 6: Confirm TypeScript compiles**

```bash
npm run build --workspace packages/agent
```

Expected: build succeeds, no errors.

- [ ] **Step 7: Bump agent version**

```bash
./scripts/bump-agent-version.sh
```

- [ ] **Step 8: Commit**

```bash
git add packages/agent/src/runtime/finetune-quantize.ts \
        packages/agent/src/runtime/finetune-quantize.test.ts \
        packages/agent/src/index.ts \
        packages/agent/package.json
git commit -m "agent: cmd:finetune:quantize handler + quantize runtime

quantizeMergedToFp8() mirrors mergeLoraAdapter(): spins a vllm-node image
container, installs llmcompressor + transformers, runs the recipe's
quantize_fp8 script against the merged BF16 dir, writes the FP8 weights
to a sibling merged-fp8 dir. Streams stdout via agent:finetune:quantize-
progress/complete. Unit tests cover the pure log-line progress classifier.
Bumped agent version."
```

---

## Task 8: Server broadcast — finetune:quantize-progress / finetune:quantize-status

**Repo:** `dgx-manager`

**Files:**
- Modify: `packages/server/src/ws/agent-hub.ts` (find the existing `agent:finetune:merge-progress` / `agent:finetune:merge-complete` handlers and add the quantize equivalents next to them)

- [ ] **Step 1: Locate the existing merge handlers**

```bash
grep -nE 'agent:finetune:merge-progress|agent:finetune:merge-complete' packages/server/src/ws/agent-hub.ts
```

Note the line numbers; the new handlers belong immediately below them.

- [ ] **Step 2: Add the quantize-progress handler**

In `packages/server/src/ws/agent-hub.ts`, find the existing `agent:finetune:merge-progress` handler (it forwards an SSE/dashboardHub broadcast as `finetune:merge-progress`). Directly below it add:

```typescript
      case "agent:finetune:quantize-progress": {
        const { jobId } = msg.payload as { jobId: string };
        this.dashboardHub.broadcast({ type: "finetune:quantize-progress", payload: msg.payload });
        if (process.env.DEBUG_FINETUNE) {
          console.log(`[hub] quantize-progress for job ${jobId}`);
        }
        break;
      }
```

- [ ] **Step 3: Add the quantize-complete handler**

Below the merge-complete handler add:

```typescript
      case "agent:finetune:quantize-complete": {
        const { jobId, status, quantizedPath, error } = msg.payload as {
          jobId: string;
          status: "completed" | "failed";
          quantizedPath: string | null;
          error?: string;
        };

        await this.prisma.fineTuneJob.update({
          where: { id: jobId },
          data: {
            quantizationStatus: status === "completed" ? "quantized" : "failed",
            quantizedPath: status === "completed" ? quantizedPath : null,
            quantizationLog: error ?? null,
            quantizedAt: status === "completed" ? new Date() : null,
          },
        });
        this.dashboardHub.broadcast({
          type: "finetune:quantize-status",
          payload: { jobId, status, quantizedPath, error },
        });
        break;
      }
```

- [ ] **Step 4: Write the failing integration test**

Append to `packages/server/src/__tests__/integration/finetune.quantize.test.ts`:

```typescript
describe("AgentHub: quantize-complete persists state", () => {
  it("transitions to quantized and stores quantizedPath", async () => {
    const node = await prisma.node.create({
      data: { id: "n2", name: "n2", ipAddress: "10.0.0.2", agentPort: 8089, status: "online", vramTotal: 122000 },
    });
    const job = await prisma.fineTuneJob.create({
      data: {
        nodeId: node.id, recipeFile: "recipes/test", baseModel: "Qwen/Qwen3.6-27B",
        method: "lora", dataset: "/tmp/ds.jsonl",
        status: "completed", mergeStatus: "completed",
        mergedPath: "/mnt/tank/outputs/job-2/merged",
        outputDir: "/mnt/tank/outputs/job-2",
        quantizationStatus: "quantizing",
        quantizedPath: "/mnt/tank/outputs/job-2/merged-fp8",
      },
    });

    const { AgentHub } = await import("../../ws/agent-hub.js");
    const broadcasts: { type: string; payload: unknown }[] = [];
    const hub = new AgentHub({
      prisma, dashboardHub: { broadcast: (e: { type: string; payload: unknown }) => broadcasts.push(e) },
    } as never);
    await (hub as unknown as {
      handleAgentMessage: (msg: { type: string; payload: unknown }) => Promise<void>;
    }).handleAgentMessage({
      type: "agent:finetune:quantize-complete",
      payload: { jobId: job.id, status: "completed", quantizedPath: "/mnt/tank/outputs/job-2/merged-fp8" },
    });

    const updated = await prisma.fineTuneJob.findUnique({ where: { id: job.id } });
    expect(updated?.quantizationStatus).toBe("quantized");
    expect(updated?.quantizedPath).toBe("/mnt/tank/outputs/job-2/merged-fp8");
    expect(updated?.quantizedAt).toBeTruthy();
    expect(broadcasts.find((b) => b.type === "finetune:quantize-status")).toBeTruthy();
  });

  it("failed status clears quantizedPath and stores error in quantizationLog", async () => {
    const node = await prisma.node.create({
      data: { id: "n3", name: "n3", ipAddress: "10.0.0.3", agentPort: 8089, status: "online", vramTotal: 122000 },
    });
    const job = await prisma.fineTuneJob.create({
      data: {
        nodeId: node.id, recipeFile: "recipes/test", baseModel: "Qwen/Qwen3.6-27B",
        method: "lora", dataset: "/tmp/ds.jsonl",
        status: "completed", mergeStatus: "completed",
        mergedPath: "/mnt/tank/outputs/job-3/merged",
        outputDir: "/mnt/tank/outputs/job-3",
        quantizationStatus: "quantizing",
        quantizedPath: "/mnt/tank/outputs/job-3/merged-fp8",
      },
    });

    const { AgentHub } = await import("../../ws/agent-hub.js");
    const hub = new AgentHub({
      prisma, dashboardHub: { broadcast: () => {} },
    } as never);
    await (hub as unknown as {
      handleAgentMessage: (msg: { type: string; payload: unknown }) => Promise<void>;
    }).handleAgentMessage({
      type: "agent:finetune:quantize-complete",
      payload: { jobId: job.id, status: "failed", quantizedPath: null, error: "OOM at FP8 cast" },
    });

    const updated = await prisma.fineTuneJob.findUnique({ where: { id: job.id } });
    expect(updated?.quantizationStatus).toBe("failed");
    expect(updated?.quantizedPath).toBeNull();
    expect(updated?.quantizationLog).toBe("OOM at FP8 cast");
  });
});
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
npx vitest run packages/server/src/__tests__/integration/finetune.quantize.test.ts
```

Expected: 7/7 pass (5 from Task 6 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/ws/agent-hub.ts \
        packages/server/src/__tests__/integration/finetune.quantize.test.ts
git commit -m "server: agent-hub handles quantize-progress/complete events

quantize-progress forwards to dashboard-hub for live UI updates.
quantize-complete persists the new quantizationStatus/quantizedPath/
quantizationLog/quantizedAt columns on FineTuneJob and broadcasts a
finetune:quantize-status event. Integration tests cover both success
and failure transitions."
```

---

## Task 9: Deploy endpoint — `artifactVariant` parameter + inference-fp8 template selection

**Repo:** `dgx-manager`

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (the existing `POST /:id/deploy` handler at line 625)
- Modify: `packages/agent/src/runtime/inference-template.ts` (extend `findInferenceTemplate` to take a variant)
- Modify: `packages/agent/src/runtime/finetune.ts` (caller passes the variant through)

- [ ] **Step 1: Write the failing unit test for findInferenceTemplate**

Append to `packages/agent/src/runtime/inference-template.test.ts` (file already exists per the codebase):

```typescript
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { describe, expect, it } from "vitest";
import { findInferenceTemplate } from "./inference-template.js";

describe("findInferenceTemplate(variant)", () => {
  it("returns inference.yaml when variant is bf16 (or omitted)", () => {
    const dir = mkdtempSync(join(tmpdir(), "tpl-"));
    writeFileSync(join(dir, "inference.yaml"), "name: bf16\n");
    expect(findInferenceTemplate(dir, "bf16")).toBe(join(dir, "inference.yaml"));
    expect(findInferenceTemplate(dir)).toBe(join(dir, "inference.yaml"));
    rmSync(dir, { recursive: true, force: true });
  });

  it("returns inference-fp8.yaml when variant is fp8 and file exists", () => {
    const dir = mkdtempSync(join(tmpdir(), "tpl-"));
    writeFileSync(join(dir, "inference.yaml"), "name: bf16\n");
    writeFileSync(join(dir, "inference-fp8.yaml"), "name: fp8\n");
    expect(findInferenceTemplate(dir, "fp8")).toBe(join(dir, "inference-fp8.yaml"));
    rmSync(dir, { recursive: true, force: true });
  });

  it("returns null when variant is fp8 but inference-fp8.yaml is missing", () => {
    const dir = mkdtempSync(join(tmpdir(), "tpl-"));
    writeFileSync(join(dir, "inference.yaml"), "name: bf16\n");
    expect(findInferenceTemplate(dir, "fp8")).toBeNull();
    rmSync(dir, { recursive: true, force: true });
  });

  it("returns null when neither variant exists", () => {
    const dir = mkdtempSync(join(tmpdir(), "tpl-"));
    expect(findInferenceTemplate(dir, "bf16")).toBeNull();
    expect(findInferenceTemplate(dir, "fp8")).toBeNull();
    rmSync(dir, { recursive: true, force: true });
  });
});
```

- [ ] **Step 2: Run the test — expect FAIL**

```bash
npx vitest run packages/agent/src/runtime/inference-template.test.ts
```

Expected: FAIL — current `findInferenceTemplate(recipeDir)` takes one arg.

- [ ] **Step 3: Extend findInferenceTemplate**

Open `packages/agent/src/runtime/inference-template.ts`. Replace the existing `findInferenceTemplate` (around line 58) with:

```typescript
/**
 * Return the absolute path to the inference template for a given
 * artifact variant, or null if no template exists for it.
 *
 *   bf16 (default) → <recipeDir>/inference.yaml
 *   fp8            → <recipeDir>/inference-fp8.yaml
 *
 * Used by the deploy path to decide whether to inherit a hand-authored
 * serve config or fall through to the minimal auto-gen.
 */
export function findInferenceTemplate(
  recipeDir: string,
  variant: "bf16" | "fp8" = "bf16",
): string | null {
  const filename = variant === "fp8" ? "inference-fp8.yaml" : "inference.yaml";
  const candidate = join(recipeDir, filename);
  return existsSync(candidate) ? candidate : null;
}
```

- [ ] **Step 4: Re-run unit tests — expect PASS**

```bash
npx vitest run packages/agent/src/runtime/inference-template.test.ts
```

Expected: all four cases pass (plus any pre-existing cases that were already passing).

- [ ] **Step 5: Thread the variant through deploy in agent/runtime/finetune.ts**

In `packages/agent/src/runtime/finetune.ts`, find every call to `findInferenceTemplate(recipeDir)` and update each call site to take a new `artifactVariant` parameter that comes in via the deploy payload. Two places need plumbing:

(a) The `cmd:finetune:deploy` handler payload in `packages/agent/src/index.ts` (around line 798): add `artifactVariant` to the destructured payload and pass it through to the runtime function. The relevant block currently does:

```typescript
    case "cmd:finetune:deploy": {
      const {
        jobId, deploymentId, modelPath, deployContainer, config,
        clusterNodes, clusterNodeFastIps, modelName, recipeFile,
      } = msg.payload as {
        ...
```

Change to:

```typescript
    case "cmd:finetune:deploy": {
      const {
        jobId, deploymentId, modelPath, deployContainer, config,
        clusterNodes, clusterNodeFastIps, modelName, recipeFile, artifactVariant,
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
        artifactVariant?: "bf16" | "fp8";
      };
```

(b) Wherever the runtime deploy function reads `findInferenceTemplate(recipeDir)` in `packages/agent/src/runtime/finetune.ts`, change the call to `findInferenceTemplate(recipeDir, artifactVariant ?? "bf16")` and add `artifactVariant?: "bf16" | "fp8"` to the deploy-function signature.

Use the following to locate the call sites:

```bash
grep -nE 'findInferenceTemplate' packages/agent/src/runtime/finetune.ts
```

- [ ] **Step 6: Write the failing integration test for the deploy endpoint**

Append to `packages/server/src/__tests__/integration/finetune.quantize.test.ts`:

```typescript
describe("POST /api/finetune/:id/deploy with artifactVariant", () => {
  async function seedQuantizedJob() {
    const node = await prisma.node.create({
      data: { id: "nd1", name: "nd1", ipAddress: "10.0.1.1", agentPort: 8089, status: "online", vramTotal: 122000 },
    });
    return prisma.fineTuneJob.create({
      data: {
        nodeId: node.id, recipeFile: "recipes/test", baseModel: "Qwen/Qwen3.6-27B",
        method: "lora", dataset: "/tmp/ds.jsonl", displayName: "test-deploy",
        status: "completed", mergeStatus: "completed",
        mergedPath: "/mnt/tank/outputs/job-d/merged",
        outputDir: "/mnt/tank/outputs/job-d",
        quantizationStatus: "quantized",
        quantizedPath: "/mnt/tank/outputs/job-d/merged-fp8",
        quantizedAt: new Date(),
      },
    });
  }

  it("defaults to bf16 (merged path) when artifactVariant is omitted", async () => {
    const { hub, sent } = makeStubHub({ file: "recipes/test", scripts: { quantize_fp8: "scripts/quantize_fp8.py" } });
    const app = makeApp(hub);
    const job = await seedQuantizedJob();

    const res = await request(app).post(`/api/finetune/${job.id}/deploy`).send({ nodeId: job.nodeId });
    expect(res.status).toBe(201);
    const cmd = sent.find((s) => (s.message as { type: string }).type === "cmd:finetune:deploy");
    expect(cmd).toBeTruthy();
    const payload = (cmd!.message as { payload: { artifactVariant?: string; modelPath: string } }).payload;
    expect(payload.artifactVariant ?? "bf16").toBe("bf16");
    expect(payload.modelPath).toBe("/mnt/tank/outputs/job-d/merged");
  });

  it("uses fp8 path when artifactVariant=fp8 and quantizedPath exists", async () => {
    const { hub, sent } = makeStubHub({ file: "recipes/test", scripts: { quantize_fp8: "scripts/quantize_fp8.py" } });
    const app = makeApp(hub);
    const job = await seedQuantizedJob();

    const res = await request(app).post(`/api/finetune/${job.id}/deploy`).send({ nodeId: job.nodeId, artifactVariant: "fp8" });
    expect(res.status).toBe(201);
    const cmd = sent.find((s) => (s.message as { type: string }).type === "cmd:finetune:deploy");
    const payload = (cmd!.message as { payload: { artifactVariant: string; modelPath: string } }).payload;
    expect(payload.artifactVariant).toBe("fp8");
    expect(payload.modelPath).toBe("/mnt/tank/outputs/job-d/merged-fp8");
  });

  it("returns 400 when artifactVariant=fp8 but quantizedPath is missing", async () => {
    const { hub } = makeStubHub({ file: "recipes/test", scripts: { quantize_fp8: "scripts/quantize_fp8.py" } });
    const app = makeApp(hub);
    const node = await prisma.node.create({
      data: { id: "nd2", name: "nd2", ipAddress: "10.0.1.2", agentPort: 8089, status: "online", vramTotal: 122000 },
    });
    const job = await prisma.fineTuneJob.create({
      data: {
        nodeId: node.id, recipeFile: "recipes/test", baseModel: "Qwen/Qwen3.6-27B",
        method: "lora", dataset: "/tmp/ds.jsonl",
        status: "completed", mergeStatus: "completed",
        mergedPath: "/mnt/tank/outputs/job-nq/merged",
        outputDir: "/mnt/tank/outputs/job-nq",
      },
    });

    const res = await request(app).post(`/api/finetune/${job.id}/deploy`).send({ nodeId: job.nodeId, artifactVariant: "fp8" });
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/quantiz/i);
  });
});
```

- [ ] **Step 7: Run the test — expect FAIL**

```bash
npx vitest run packages/server/src/__tests__/integration/finetune.quantize.test.ts
```

Expected: the three new cases FAIL — deploy endpoint doesn't read `artifactVariant` yet.

- [ ] **Step 8: Extend the deploy endpoint**

Open `packages/server/src/routes/finetune.ts` and locate the `POST /:id/deploy` handler at line 625. Find the existing block (around line 632 onward) that computes `modelPath`:

```typescript
  const { nodeId, nodeIds, config } = req.body;
  ...
  // Determine model path — use merged if available, otherwise adapter
  const modelPath = job.mergedPath || (job.outputDir ? `${job.outputDir}/merged` : null);
  if (!modelPath || job.mergeStatus !== "completed") {
    return res.status(400).json({ error: "Model must be merged before deployment. Call POST /merge first." });
  }
```

Change the body destructure to:

```typescript
  const { nodeId, nodeIds, config, artifactVariant } = req.body as {
    nodeId?: string;
    nodeIds?: string[];
    config?: Record<string, unknown>;
    artifactVariant?: "bf16" | "fp8";
  };
```

Replace the model-path resolution + guard with:

```typescript
  const variant: "bf16" | "fp8" = artifactVariant === "fp8" ? "fp8" : "bf16";

  let modelPath: string | null;
  if (variant === "fp8") {
    if (job.quantizationStatus !== "quantized" || !job.quantizedPath) {
      return res.status(400).json({
        error: "FP8 deploy requested but no quantized artifact exists. Call POST /quantize first.",
      });
    }
    modelPath = job.quantizedPath;
  } else {
    modelPath = job.mergedPath || (job.outputDir ? `${job.outputDir}/merged` : null);
    if (!modelPath || job.mergeStatus !== "completed") {
      return res.status(400).json({ error: "Model must be merged before deployment. Call POST /merge first." });
    }
  }
```

Then find where the `cmd:finetune:deploy` WS message is built (search for `type: "cmd:finetune:deploy"` in the same file) and add `artifactVariant: variant` to the payload:

```typescript
  agentHub.sendToAgent(headNodeId, {
    type: "cmd:finetune:deploy",
    payload: {
      jobId: job.id,
      deploymentId: deployment.id,
      modelPath,
      ...
      recipeFile: job.recipeFile,
      artifactVariant: variant,
    },
  });
```

- [ ] **Step 9: Re-run tests — expect PASS**

```bash
npx vitest run packages/server/src/__tests__/integration/finetune.quantize.test.ts
```

Expected: 10/10 pass.

- [ ] **Step 10: Bump agent version (because index.ts changed in step 5)**

```bash
./scripts/bump-agent-version.sh
```

- [ ] **Step 11: Commit**

```bash
git add packages/server/src/routes/finetune.ts \
        packages/agent/src/runtime/inference-template.ts \
        packages/agent/src/runtime/inference-template.test.ts \
        packages/agent/src/runtime/finetune.ts \
        packages/agent/src/index.ts \
        packages/agent/package.json \
        packages/server/src/__tests__/integration/finetune.quantize.test.ts
git commit -m "deploy: artifactVariant selects bf16 (merged) vs fp8 (merged-fp8)

POST /api/finetune/:id/deploy gains an artifactVariant param ('bf16' |
'fp8', defaults to bf16). FP8 requires quantizationStatus=quantized
and a non-null quantizedPath; otherwise 400. The cmd:finetune:deploy
WS message carries the variant so the agent picks the right inference
template (inference.yaml vs inference-fp8.yaml). Bumped agent version."
```

---

## Task 10: Dashboard — Quantize button + artifact selector in deploy modal

**Repo:** `dgx-manager`

**Files:**
- Modify: `packages/dashboard/app/finetune/page.tsx`

- [ ] **Step 1: Add the FineTuneJob type fields**

Open `packages/dashboard/app/finetune/page.tsx`. Find the `FineTuneJob` type definition (around line 30–55; contains `mergeStatus`, `mergedPath` fields). Add the four quantization fields after `mergedPath: string | null;`:

```typescript
  mergeStatus: string | null;
  mergedPath: string | null;
  quantizationStatus: string | null; // null | pending | quantizing | quantized | failed
  quantizedPath: string | null;
  quantizationLog: string | null;
  quantizedAt: string | null;
```

- [ ] **Step 2: Add quantizeJob() handler**

Find the existing `mergeJob` handler (around line 376):

```typescript
  const mergeJob = async (id: string) => {
    try {
      await apiFetch(`/api/finetune/${id}/merge`, { method: "POST" });
      setJobs((prev) =>
        prev.map((j) => j.id === id ? { ...j, mergeStatus: "running" } : j)
      );
    } catch (err) {
      toast.error("Merge failed", { description: err instanceof Error ? err.message : String(err) });
    }
  };
```

Directly below it, add:

```typescript
  const quantizeJob = async (id: string) => {
    try {
      await apiFetch(`/api/finetune/${id}/quantize`, { method: "POST" });
      setJobs((prev) =>
        prev.map((j) => j.id === id ? { ...j, quantizationStatus: "quantizing" } : j)
      );
    } catch (err) {
      toast.error("Quantize failed", { description: err instanceof Error ? err.message : String(err) });
    }
  };
```

- [ ] **Step 3: Subscribe to quantize WS events**

Find the existing WS event handlers for `finetune:merge-progress` and `finetune:merge-status` (around line 251–266). Add two sibling handlers immediately below:

```typescript
    if (event.type === "finetune:quantize-progress") {
      // No-op for now — we don't render quantize progress live, just status.
      return;
    }
    if (event.type === "finetune:quantize-status") {
      const { jobId, status, quantizedPath } = event.payload as {
        jobId: string; status: string; quantizedPath?: string;
      };
      setJobs((prev) =>
        prev.map((j) => j.id === jobId
          ? { ...j, quantizationStatus: status === "completed" ? "quantized" : "failed",
              quantizedPath: quantizedPath ?? j.quantizedPath }
          : j),
      );
      return;
    }
```

- [ ] **Step 4: Add the Quantize button next to the existing Merge button**

Find the row that renders the Merge button (around line 904):

```typescript
{job.status === "completed" && (!job.mergeStatus || job.mergeStatus === "failed") && (
  <Button onClick={() => mergeJob(job.id)} ...>Merge Model</Button>
)}
{job.mergeStatus === "running" && (...)}
{job.mergeStatus === "completed" && (...)}
```

After the `mergeStatus === "completed"` block, insert:

```typescript
{job.mergeStatus === "completed" && (!job.quantizationStatus || job.quantizationStatus === "failed") && (
  <Button
    size="sm"
    variant="outline"
    onClick={() => quantizeJob(job.id)}
    className="border-purple-500/40 text-purple-300 hover:bg-purple-500/10"
  >
    Quantize to FP8
  </Button>
)}
{job.quantizationStatus === "quantizing" && (
  <span className="text-xs text-purple-300">Quantizing FP8…</span>
)}
{job.quantizationStatus === "quantized" && (
  <span className="text-xs text-purple-300">FP8 ready ✓</span>
)}
{job.quantizationStatus === "failed" && (
  <span className="text-xs text-red-400">FP8 quantize failed</span>
)}
```

- [ ] **Step 5: Add artifact-variant selector in the deploy modal**

Find the deploy modal / deploy button handler (search for `deployJob` or `artifactVariant` candidates around line 425–460). The existing deploy flow looks like:

```typescript
const modelPath = job.mergedPath || `${job.outputDir}/merged`;
// ...calls deploy endpoint with { nodeId } or similar...
```

Extend the local component state to track which variant the operator picks (BF16 default, FP8 only enabled when `quantizationStatus === "quantized"`). Use a small radio group right above the existing deploy submit button in the modal:

```tsx
<div className="mt-3 flex flex-col gap-1">
  <label className="text-xs text-gray-400">Deploy artifact</label>
  <div className="flex gap-3">
    <label className="flex items-center gap-1 text-sm">
      <input type="radio" name="artifactVariant" value="bf16"
        checked={selectedVariant === "bf16"}
        onChange={() => setSelectedVariant("bf16")} />
      BF16 (merged)
    </label>
    <label className={`flex items-center gap-1 text-sm ${job.quantizationStatus === "quantized" ? "" : "opacity-50"}`}>
      <input type="radio" name="artifactVariant" value="fp8"
        disabled={job.quantizationStatus !== "quantized"}
        checked={selectedVariant === "fp8"}
        onChange={() => setSelectedVariant("fp8")} />
      FP8 (merged-fp8) {job.quantizationStatus !== "quantized" && <span className="text-xs text-gray-500">(not quantized)</span>}
    </label>
  </div>
</div>
```

Add `const [selectedVariant, setSelectedVariant] = useState<"bf16" | "fp8">("bf16");` at the appropriate component-state declaration block (top of the deploy modal component).

Update the deploy POST to include the variant:

```typescript
await apiFetch(`/api/finetune/${job.id}/deploy`, {
  method: "POST",
  body: JSON.stringify({ nodeId, artifactVariant: selectedVariant }),
});
```

- [ ] **Step 6: Confirm the dashboard builds**

```bash
cd /home/daniel/src/github/dgx-manager
npm run build --workspace packages/dashboard
```

Expected: build succeeds; no TS errors.

- [ ] **Step 7: Commit**

```bash
git add packages/dashboard/app/finetune/page.tsx
git commit -m "dashboard: Quantize to FP8 button + artifact selector on deploy

Adds a 'Quantize to FP8' button that appears after merge completes,
a status badge for quantizing/quantized/failed states, and a BF16/FP8
radio selector in the deploy modal. FP8 option is disabled until the
job has been successfully quantized."
```

---

## Task 11: End-to-end smoke against the real recipe

**Repo:** `dgx-manager`

This task is a manual verification, not a code change. It catches integration drift between the recipe repo and the manager (e.g. missing `quantize_fp8` field in the parsed recipe).

- [ ] **Step 1: Confirm the recipe surfaces quantize_fp8**

```bash
curl -sS http://localhost:4000/api/training-recipes 2>/dev/null \
  | python3 -c "
import json, sys
recipes = json.load(sys.stdin)
match = [r for r in recipes if r['file'] == 'recipes/qwen3.6-27b-base-lora-attn-mlp']
print(json.dumps(match[0]['scripts'], indent=2)) if match else print('NOT FOUND')
"
```

Expected: a `scripts` block containing `"quantize_fp8": "scripts/quantize_fp8.py"`.

- [ ] **Step 2: Pick a completed-and-merged fine-tune job and hit POST /quantize**

```bash
JOB_ID=<an existing merged job id>
curl -sS -X POST "http://localhost:4000/api/finetune/${JOB_ID}/quantize" -H 'Content-Type: application/json' -d '{}' | jq
```

Expected: `{ "status": "quantizing", "quantizedPath": ".../merged-fp8" }`.

- [ ] **Step 3: Tail the quantize log on shared storage**

```bash
tail -f /mnt/tank/outputs/${JOB_ID}/quantize.log
```

Expected: sees `[agent] Starting quantize container`, then `[quantize_fp8] Loading model from ...`, then `[quantize_fp8] Applying FP8_DYNAMIC ...`, then `[quantize_fp8] OK`. End-to-end takes ~5 min for a 27B model on one GB10.

- [ ] **Step 4: Confirm DB transitioned**

```bash
sqlite3 /home/daniel/src/github/dgx-manager/dev.db \
  "SELECT quantizationStatus, quantizedPath, quantizedAt FROM FineTuneJob WHERE id = '${JOB_ID}';"
```

Expected: `quantizationStatus=quantized`, non-null `quantizedPath` and `quantizedAt`.

- [ ] **Step 5: Deploy FP8 variant**

```bash
curl -sS -X POST "http://localhost:4000/api/finetune/${JOB_ID}/deploy" \
  -H 'Content-Type: application/json' \
  -d '{"nodeId":"<node>","artifactVariant":"fp8"}' | jq
```

Expected: 201 Created with a deployment record pointing at the FP8 path.

- [ ] **Step 6: Confirm vLLM uses --quantization fp8**

```bash
ssh <node-ip> 'docker inspect $(docker ps -q --filter "name=dgx-vllm-") | grep -A2 Cmd'
```

Expected: the rendered Cmd array contains `--quantization fp8` and the model path ends in `/merged-fp8`.

- [ ] **Step 7: Smoke test the FP8 deployment**

```bash
curl -sS http://<node-ip>:8000/v1/models | jq
curl -sS http://<node-ip>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served-model-name>","messages":[{"role":"user","content":"hello"}]}' | jq
```

Expected: model listed; chat completion returns sensible output.

---

## Self-Review

**Spec coverage check:**
- ✅ Recipe-level: `scripts/quantize_fp8.py` (Task 1), per-recipe `inference-fp8.yaml` (Task 2), `recipe.yaml` field (Task 3).
- ✅ DB: additive migration with `quantizationStatus`, `quantizedPath`, `quantizationLog`, `quantizedAt` (Task 4).
- ✅ Server: `POST /:id/quantize` (Task 6), agent-hub broadcasts (Task 8), deploy `artifactVariant` (Task 9).
- ✅ Agent: `cmd:finetune:quantize` handler + runtime (Task 7), `findInferenceTemplate(variant)` (Task 9), version bumps after every agent change (Tasks 5, 7, 9).
- ✅ Dashboard: Quantize button + status badge + artifact selector (Task 10).
- ✅ Tests: recipe parser unit test (Task 5), endpoint integration tests for happy path / idempotency / pre-merge guard / missing-script-501 / in-flight-409 / quantize-complete persistence success+failure (Tasks 6 + 8), `findInferenceTemplate(variant)` unit tests (Task 9), deploy endpoint integration tests for bf16-default / fp8-selection / fp8-without-quant-400 (Task 9).
- ✅ Constraints: idempotency (Task 6), additive migration (Task 4), no breaking schema change (Task 4), opt-in not auto (Task 10 — button-triggered).
- ✅ End-to-end smoke (Task 11) closes the loop on recipe-repo ↔ manager coupling.

**Placeholder scan:** No "TBD"/"TODO"/"add appropriate error handling"/"similar to Task N" — every step has the actual code or command.

**Type consistency:** `quantizationStatus` / `quantizedPath` / `quantizationLog` / `quantizedAt` are spelled identically across schema (Task 4), endpoint (Task 6), agent-hub (Task 8), and dashboard (Task 10). `artifactVariant` is the same name in deploy endpoint (Task 9), `findInferenceTemplate` signature (Task 9), agent `cmd:finetune:deploy` payload (Task 9), and dashboard radio group state (Task 10). `cmd:finetune:quantize` / `agent:finetune:quantize-progress` / `agent:finetune:quantize-complete` are consistent between server (Task 6, 8) and agent (Task 7).

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-13-post-training-fp8-quantization.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
