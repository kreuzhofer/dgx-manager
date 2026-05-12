# Fine-Tune Naming + Stale-Model Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two coupled improvements to the fine-tune model lifecycle:
1. Auto-clean stale `Model` rows so the deployable-model list doesn't accumulate orphans from deleted fine-tunes.
2. Let users name a fine-tune at creation time and rename it later, with the name flowing into the deployable `Model` row.

**Architecture:** Add `displayName` to `FineTuneJob` (nullable, falls back to job-id-derived name). Add `finetuneJobId` FK on `Model` with `onDelete: SetNull` so the existing Cascade-on-deploy logic still works AND deleting a fine-tune nukes its standalone Model row when no Deployment references it. The deploy route uses the FK + display name. Dashboard wires a name input into the launch dialog and an inline rename action onto job rows.

**Tech Stack:** Prisma (SQLite), Express 5 + supertest + vitest, Next.js 15 App Router (React 19), Tailwind CSS 4.

---

## File Structure

**Created:**
- `packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts` — integration coverage for the rename + cleanup lifecycle (POST sets, PATCH renames, DELETE cascades Model cleanup, orphans get purged).

**Modified:**
- `prisma/schema.prisma` — add `FineTuneJob.displayName String?`, add `Model.finetuneJobId String? @unique` (one-to-one with `FineTuneJob`), `onDelete: Cascade` for the Model→FineTuneJob relation (so deleting a job removes its standalone Model row, but deleting a Model doesn't touch the job).
- `packages/server/src/routes/finetune.ts` — accept `displayName` in POST; add PATCH `/api/finetune/:id`; update POST `/:id/deploy` to (a) use displayName when present (fallback to `finetune-<id-prefix>`) and (b) set Model.finetuneJobId.
- `packages/server/src/routes/finetune.ts` — add POST `/api/finetune/cleanup-orphan-models` (one-shot admin endpoint to purge existing stale rows).
- `packages/dashboard/app/finetune/page.tsx` — name input in the launch dialog; inline rename action in the job-row action group; render `displayName` (with fallback) in job cards.

**Not touched (deliberate):**
- `Deployment` model — its FK to `Model` stays as-is; if a user deletes a fine-tune whose Model is currently deployed, Prisma's onDelete: Cascade on `Model.finetuneJobId` would set the deployment's modelId... actually no, we use `onDelete: Cascade` on `Model.finetuneJobId → FineTuneJob`, meaning when FineTuneJob is deleted, Model is deleted, and Deployment.modelId then references a missing Model. This is acceptable because the existing delete-job flow already blocks/stops active deployments; we add a guard to the cleanup logic to refuse deletion when a Deployment exists.

---

## Task 1: Schema — add `displayName` and `Model.finetuneJobId` FK

**Files:**
- Modify: `prisma/schema.prisma`

- [ ] **Step 1: Read current schema for FineTuneJob and Model**

Run: `grep -nE "^model (FineTuneJob|Model) " prisma/schema.prisma`

Expected: two matches with line numbers. Note them — you'll edit both blocks.

- [ ] **Step 2: Add `displayName` to FineTuneJob**

In `prisma/schema.prisma`, locate `model FineTuneJob {` and add this field right after `nodeId`:

```prisma
  // User-facing name for the fine-tune. Settable at creation and via PATCH.
  // When null, the dashboard falls back to a derived label (recipe name +
  // job-id prefix). Mirrors into the associated Model.name on deploy so the
  // deployable-models list shows the user's chosen label.
  displayName  String?
```

- [ ] **Step 3: Add `finetuneJobId` FK + relation to Model**

In `prisma/schema.prisma`, locate `model Model {` and add this field right after `parameters`:

```prisma
  // FK to the FineTuneJob this Model row represents. Null for non-finetune
  // models (hand-created, Ollama, etc.). Enables (a) cascade-delete so a
  // deleted job doesn't leave orphan Model rows in the deployable list,
  // and (b) explicit linkage instead of name-prefix string magic.
  finetuneJobId String?      @unique
  finetuneJob   FineTuneJob? @relation(fields: [finetuneJobId], references: [id], onDelete: Cascade)
```

- [ ] **Step 4: Add inverse relation on FineTuneJob**

Inside `model FineTuneJob { ... }`, add this line near the existing `clusterNodes` relation:

```prisma
  model        Model?
```

- [ ] **Step 5: Apply schema + regenerate client**

Run:
```
DATABASE_URL=file:./dev.db npx prisma db push
DATABASE_URL=file:./dev.db npx prisma generate
```

Expected: "Your database is now in sync with your Prisma schema" + Prisma client regen success.

- [ ] **Step 6: Commit**

```bash
git add prisma/schema.prisma
git commit -m "schema: FineTuneJob.displayName + Model.finetuneJobId FK"
```

---

## Task 2: POST `/api/finetune` accepts `displayName`

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (POST handler around line 88)
- Test: `packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts` (NEW)

- [ ] **Step 1: Bootstrap the test file**

Create `packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts`:

```typescript
/**
 * Integration coverage for the displayName + Model-cleanup lifecycle on
 * /api/finetune.
 *
 * Follows the same per-suite SQLite + supertest + stub-hub pattern as
 * finetune.cluster-persistence.test.ts.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
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

const RECIPE = {
  file: "recipes/test-attn-mlp",
  name: "Test Recipe",
  base_model: "Qwen/Qwen3.6-27B",
  method: "lora",
  defaults: {},
  scripts: { merge: "scripts/merge.py" },
};

function makeStubHub() {
  const sentMessages: { nodeId: string; message: unknown }[] = [];
  return {
    hub: {
      getTrainingRecipes: () => [RECIPE],
      sendToAgent: (nodeId: string, message: unknown) => {
        sentMessages.push({ nodeId, message });
      },
    },
    sentMessages,
  };
}

function makeApp(hub: unknown) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.use("/api/finetune", finetuneRouter);
  return app;
}

async function wipeAll() {
  await prisma.fineTuneClusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.fineTuneJob.deleteMany({});
  await prisma.node.deleteMany({});
}

async function seedNode() {
  return prisma.node.create({
    data: { id: "node-1", name: "dgx-spark-01", ipAddress: "192.168.44.36", status: "online" },
  });
}

describe("finetune displayName + Model cleanup", () => {
  it("POST without displayName leaves it null", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/finetune")
      .send({
        nodeId: "node-1",
        recipeFile: RECIPE.file,
        dataset: "/tmp/fake.jsonl",
      });

    expect(res.status).toBe(201);
    expect(res.body.displayName).toBeNull();
  });
});
```

- [ ] **Step 2: Add the POST-accepts-displayName test**

Append this test to the `describe` block in `finetune.naming-and-cleanup.test.ts`:

```typescript
  it("POST with displayName persists it on the job", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/finetune")
      .send({
        nodeId: "node-1",
        recipeFile: RECIPE.file,
        dataset: "/tmp/fake.jsonl",
        displayName: "build123d-v1",
      });

    expect(res.status).toBe(201);
    expect(res.body.displayName).toBe("build123d-v1");

    const row = await prisma.fineTuneJob.findUnique({ where: { id: res.body.id } });
    expect(row?.displayName).toBe("build123d-v1");
  });
```

- [ ] **Step 3: Run tests, verify both fail**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts`

Expected: tests fail because (a) `displayName` is currently `undefined` (not present in the schema response) and (b) the second test's POST doesn't persist `displayName`.

- [ ] **Step 4: Add `displayName` to POST body destructuring + write**

In `packages/server/src/routes/finetune.ts`, locate the POST handler (around line 88) and change the destructure line from:

```typescript
  const { nodeId, nodeIds, recipeFile, dataset, config, resumeFromJobId } = req.body;
```

to:

```typescript
  const { nodeId, nodeIds, recipeFile, dataset, config, resumeFromJobId, displayName } = req.body;
```

Then in the `prisma.fineTuneJob.create` call (the `data: {...}` block, around line 124), add `displayName` right after `method`:

```typescript
  const job = await prisma.fineTuneJob.create({
    data: {
      nodeId: headNodeId,
      recipeFile: effectiveRecipeFile,
      baseModel,
      method,
      displayName: typeof displayName === "string" && displayName.trim() ? displayName.trim() : null,
      dataset: effectiveDataset,
      config: Object.keys(mergedConfig).length ? JSON.stringify(mergedConfig) : null,
      status: "pending",
    },
  });
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts`

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/routes/finetune.ts packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts
git commit -m "finetune: accept displayName in POST"
```

---

## Task 3: PATCH `/api/finetune/:id` for rename

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (add PATCH between existing handlers)
- Test: `packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts`

- [ ] **Step 1: Write the failing tests**

Append to the `describe` block in `finetune.naming-and-cleanup.test.ts`:

```typescript
  it("PATCH /:id can set displayName on an existing job", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl" });
    expect(create.body.displayName).toBeNull();

    const patch = await request(app)
      .patch(`/api/finetune/${create.body.id}`)
      .send({ displayName: "renamed-via-patch" });
    expect(patch.status).toBe(200);
    expect(patch.body.displayName).toBe("renamed-via-patch");

    const get = await request(app).get(`/api/finetune/${create.body.id}`);
    expect(get.body.displayName).toBe("renamed-via-patch");
  });

  it("PATCH /:id trims whitespace and treats empty string as clearing", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl",
              displayName: "initial" });
    expect(create.body.displayName).toBe("initial");

    const clear = await request(app)
      .patch(`/api/finetune/${create.body.id}`)
      .send({ displayName: "   " });
    expect(clear.status).toBe(200);
    expect(clear.body.displayName).toBeNull();
  });

  it("PATCH /:id returns 404 when the job doesn't exist", async () => {
    await wipeAll();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .patch("/api/finetune/does-not-exist")
      .send({ displayName: "x" });
    expect(res.status).toBe(404);
  });
```

- [ ] **Step 2: Run tests, verify all three fail**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts -t "PATCH"`

Expected: 3 tests fail (route doesn't exist yet).

- [ ] **Step 3: Add the PATCH handler**

In `packages/server/src/routes/finetune.ts`, add this handler after the existing GET `/:id/metrics` handler (around line 86) and before the POST handler:

```typescript
// PATCH /:id — mutate user-editable fields on the job. Currently scoped to
// displayName; extend the allowed fields here as future rename-able
// attributes are added. Trims whitespace; empty/whitespace-only strings
// clear the name (back to null → dashboard falls back to derived label).
finetuneRouter.patch("/:id", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });

  const updates: { displayName?: string | null } = {};
  if ("displayName" in req.body) {
    const raw = req.body.displayName;
    if (raw === null) {
      updates.displayName = null;
    } else if (typeof raw === "string") {
      const trimmed = raw.trim();
      updates.displayName = trimmed.length ? trimmed : null;
    } else {
      return res.status(400).json({ error: "displayName must be a string or null" });
    }
  }

  if (Object.keys(updates).length === 0) {
    return res.status(400).json({ error: "no allowed fields provided" });
  }

  const updated = await prisma.fineTuneJob.update({
    where: { id: req.params.id },
    data: updates,
    include: {
      node: true,
      clusterNodes: { include: { node: true }, orderBy: { role: "asc" } },
    },
  });
  res.json(updated);
});
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts -t "PATCH"`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/routes/finetune.ts packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts
git commit -m "finetune: PATCH /:id for rename (displayName only)"
```

---

## Task 4: Deploy route links Model to FineTuneJob + uses displayName

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (POST `/:id/deploy` around line 306)
- Test: `packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts`

- [ ] **Step 1: Write the failing tests**

Append to the `describe` block:

```typescript
  it("POST /:id/deploy upserts a Model row whose finetuneJobId is set", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl" });
    // Simulate merge having completed so the deploy route accepts it.
    await prisma.fineTuneJob.update({
      where: { id: create.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged" },
    });

    const dep = await request(app)
      .post(`/api/finetune/${create.body.id}/deploy`)
      .send({ config: {} });
    expect(dep.status).toBe(200);

    // Find the upserted Model and confirm the FK is set
    const model = await prisma.model.findFirst({
      where: { finetuneJobId: create.body.id },
    });
    expect(model).not.toBeNull();
    expect(model?.runtime).toBe("vllm");
  });

  it("POST /:id/deploy uses displayName as Model.name when available", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({
        nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl",
        displayName: "build123d-v1",
      });
    await prisma.fineTuneJob.update({
      where: { id: create.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged" },
    });

    await request(app)
      .post(`/api/finetune/${create.body.id}/deploy`)
      .send({ config: {} });

    const model = await prisma.model.findFirst({
      where: { finetuneJobId: create.body.id },
    });
    expect(model?.name).toBe("build123d-v1");
  });

  it("POST /:id/deploy falls back to finetune-<prefix> when displayName is null", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl" });
    await prisma.fineTuneJob.update({
      where: { id: create.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged" },
    });

    await request(app)
      .post(`/api/finetune/${create.body.id}/deploy`)
      .send({ config: {} });

    const model = await prisma.model.findFirst({
      where: { finetuneJobId: create.body.id },
    });
    expect(model?.name).toBe(`finetune-${create.body.id.slice(0, 8)}`);
  });
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts -t "/deploy"`

Expected: 3 tests fail (the current deploy route uses the old upsert pattern without the FK and without displayName).

- [ ] **Step 3: Update the deploy route to use displayName + FK**

In `packages/server/src/routes/finetune.ts`, locate the existing `prisma.model.upsert` call inside the deploy handler (around line 349). Replace it with:

```typescript
  // Deployable Model name: prefer the user-set displayName, fall back to a
  // stable id-derived label. The finetuneJobId FK is what makes the row
  // get cleaned up automatically when the job is deleted (onDelete: Cascade
  // in schema.prisma).
  const stableName = `finetune-${job.id.slice(0, 8)}`;
  const modelName = job.displayName?.trim() || stableName;

  const model = await prisma.model.upsert({
    where: { finetuneJobId: job.id },
    create: { name: modelName, runtime: "vllm", finetuneJobId: job.id },
    update: { name: modelName },
  });
```

Note: This replaces the previous `where: { name: ... }` upsert key with `where: { finetuneJobId: ... }` (which is `@unique` per Task 1). The FK is the canonical key now.

- [ ] **Step 4: Run tests, verify they pass**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts -t "/deploy"`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/routes/finetune.ts packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts
git commit -m "finetune: deploy links Model.finetuneJobId + honors displayName"
```

---

## Task 5: Rename propagates to the deployable Model row

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (PATCH handler from Task 3)
- Test: `packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts`

- [ ] **Step 1: Write the failing test**

Append:

```typescript
  it("PATCH /:id renames the associated Model row if one exists", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({
        nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl",
        displayName: "original-name",
      });
    await prisma.fineTuneJob.update({
      where: { id: create.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged" },
    });
    await request(app)
      .post(`/api/finetune/${create.body.id}/deploy`)
      .send({ config: {} });

    // Confirm pre-condition: Model.name == 'original-name'
    let model = await prisma.model.findFirst({ where: { finetuneJobId: create.body.id } });
    expect(model?.name).toBe("original-name");

    // Rename
    await request(app)
      .patch(`/api/finetune/${create.body.id}`)
      .send({ displayName: "renamed-name" });

    model = await prisma.model.findFirst({ where: { finetuneJobId: create.body.id } });
    expect(model?.name).toBe("renamed-name");
  });

  it("PATCH /:id clearing displayName resets Model.name to the stable fallback", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({
        nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl",
        displayName: "to-be-cleared",
      });
    await prisma.fineTuneJob.update({
      where: { id: create.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged" },
    });
    await request(app)
      .post(`/api/finetune/${create.body.id}/deploy`)
      .send({ config: {} });

    await request(app)
      .patch(`/api/finetune/${create.body.id}`)
      .send({ displayName: null });

    const model = await prisma.model.findFirst({ where: { finetuneJobId: create.body.id } });
    expect(model?.name).toBe(`finetune-${create.body.id.slice(0, 8)}`);
  });
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts -t "PATCH"`

Expected: the two new rename-propagation tests fail (Model.name doesn't follow the rename yet); existing PATCH tests still pass.

- [ ] **Step 3: Extend PATCH to update the associated Model**

In `packages/server/src/routes/finetune.ts`, modify the PATCH handler from Task 3. Replace the final `prisma.fineTuneJob.update(...)` + `res.json(updated)` block with:

```typescript
  const updated = await prisma.fineTuneJob.update({
    where: { id: req.params.id },
    data: updates,
    include: {
      node: true,
      clusterNodes: { include: { node: true }, orderBy: { role: "asc" } },
      model: true,
    },
  });

  // Propagate displayName to the deployable Model row if one exists. The
  // Model is linked via FK (finetuneJobId @unique), created by POST /:id/deploy.
  // Pre-merge / pre-deploy jobs have no Model row yet — nothing to do.
  if ("displayName" in updates && updated.model) {
    const stableName = `finetune-${updated.id.slice(0, 8)}`;
    const modelName = updates.displayName?.trim() || stableName;
    if (updated.model.name !== modelName) {
      await prisma.model.update({
        where: { id: updated.model.id },
        data: { name: modelName },
      });
    }
  }

  res.json(updated);
```

- [ ] **Step 4: Run tests, verify all PATCH tests pass**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts -t "PATCH"`

Expected: all 5 PATCH-related tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/routes/finetune.ts packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts
git commit -m "finetune: PATCH propagates displayName to associated Model.name"
```

---

## Task 6: DELETE cascades to Model row (via FK)

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (DELETE handler around line 183) — guard against deleting when an active Deployment exists.
- Test: `packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts`

- [ ] **Step 1: Write the failing tests**

Append:

```typescript
  it("DELETE /:id removes the associated Model row via cascade", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl" });
    await prisma.fineTuneJob.update({
      where: { id: create.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged" },
    });
    await request(app)
      .post(`/api/finetune/${create.body.id}/deploy`)
      .send({ config: {} });

    expect(await prisma.model.count({ where: { finetuneJobId: create.body.id } })).toBe(1);

    const del = await request(app).delete(`/api/finetune/${create.body.id}`);
    expect(del.status).toBe(200);

    expect(await prisma.fineTuneJob.count({ where: { id: create.body.id } })).toBe(0);
    expect(await prisma.model.count({ where: { finetuneJobId: create.body.id } })).toBe(0);
  });

  it("DELETE /:id refuses when the Model has an active Deployment", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl" });
    await prisma.fineTuneJob.update({
      where: { id: create.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged" },
    });
    await request(app)
      .post(`/api/finetune/${create.body.id}/deploy`)
      .send({ config: {} });

    const model = await prisma.model.findFirstOrThrow({ where: { finetuneJobId: create.body.id } });
    // Insert an active deployment that references this Model
    await prisma.deployment.create({
      data: {
        nodeId: "node-1",
        modelId: model.id,
        status: "running",
      },
    });

    const del = await request(app).delete(`/api/finetune/${create.body.id}`);
    expect(del.status).toBe(409);
    expect(del.body.error).toMatch(/active deployment/i);

    // Job + Model are still there
    expect(await prisma.fineTuneJob.count({ where: { id: create.body.id } })).toBe(1);
    expect(await prisma.model.count({ where: { finetuneJobId: create.body.id } })).toBe(1);
  });
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts -t "DELETE"`

Expected: the cascade test may already pass if Prisma's onDelete: Cascade from Task 1 is wired correctly; the "refuses when active Deployment" test will fail.

- [ ] **Step 3: Guard DELETE against active deployments**

In `packages/server/src/routes/finetune.ts`, locate the DELETE handler (around line 183). At the top of the handler, after the `if (!job)` 404 check, insert:

```typescript
  // If a Deployment is currently using this fine-tune's Model, refuse the
  // delete — the user must stop+remove the deployment first. Without this
  // guard, the Cascade onDelete on Model.finetuneJobId would nuke the
  // Model and leave Deployment.modelId pointing at a missing row.
  const linkedModel = await prisma.model.findUnique({
    where: { finetuneJobId: job.id },
    include: { deployments: true },
  });
  if (linkedModel) {
    const active = linkedModel.deployments.filter(
      (d) => !["stopped", "failed", "removed"].includes(d.status),
    );
    if (active.length > 0) {
      return res.status(409).json({
        error: `Cannot delete: ${active.length} active deployment(s) reference this model. ` +
               `Stop the deployment(s) first.`,
        deploymentIds: active.map((d) => d.id),
      });
    }
  }
```

- [ ] **Step 4: Run tests, verify both pass**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts -t "DELETE"`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/routes/finetune.ts packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts
git commit -m "finetune: DELETE refuses on active deployments + cascades Model cleanup"
```

---

## Task 7: One-shot cleanup endpoint for existing orphans

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (add `POST /cleanup-orphan-models`)
- Test: `packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts`

Background: in production, the schema migration in Task 1 leaves existing `Model` rows with `finetuneJobId = NULL` even when their name matches the legacy `finetune-<id>` pattern. We need a one-shot endpoint that:
- Finds Models named `finetune-<8 hex>` with `finetuneJobId IS NULL`
- For each, looks up the matching FineTuneJob by id-prefix
- If found: sets the FK (back-links the existing row)
- If not found AND no active deployments reference it: deletes the row

- [ ] **Step 1: Write the failing test**

Append:

```typescript
  it("POST /cleanup-orphan-models back-links legacy Model rows + deletes truly orphaned ones", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    // Case A: legacy Model row whose name matches an existing job's id-prefix
    // (created before this migration). FK is null. Expected: back-linked.
    const aliveJob = await prisma.fineTuneJob.create({
      data: {
        nodeId: "node-1", baseModel: "B", method: "lora", dataset: "/tmp/x",
        mergeStatus: "completed", mergedPath: "/tmp/m",
      },
    });
    await prisma.model.create({
      data: { name: `finetune-${aliveJob.id.slice(0, 8)}`, runtime: "vllm" },
    });

    // Case B: legacy Model row for a job that no longer exists (orphan).
    // FK is null, no Deployment references it. Expected: deleted.
    await prisma.model.create({
      data: { name: "finetune-deadbeef", runtime: "vllm" },
    });

    // Case C: legacy orphan WITH a still-active Deployment (rare but possible).
    // Expected: kept (user must remove the deployment first).
    const stuckModel = await prisma.model.create({
      data: { name: "finetune-feedface", runtime: "vllm" },
    });
    await prisma.deployment.create({
      data: { nodeId: "node-1", modelId: stuckModel.id, status: "running" },
    });

    const res = await request(app)
      .post("/api/finetune/cleanup-orphan-models")
      .send({});
    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({
      backlinked: 1,
      deleted: 1,
      kept_due_to_deployment: 1,
    });

    // A: now linked
    const a = await prisma.model.findFirstOrThrow({ where: { finetuneJobId: aliveJob.id } });
    expect(a.name).toBe(`finetune-${aliveJob.id.slice(0, 8)}`);

    // B: gone
    expect(await prisma.model.count({ where: { name: "finetune-deadbeef" } })).toBe(0);

    // C: still there
    expect(await prisma.model.count({ where: { name: "finetune-feedface" } })).toBe(1);
  });
```

- [ ] **Step 2: Run test, verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts -t "cleanup-orphan"`

Expected: 404 because the route doesn't exist yet.

- [ ] **Step 3: Add the cleanup endpoint**

In `packages/server/src/routes/finetune.ts`, add at the end of the file (before the closing `export` if any — append after the last `finetuneRouter.post(...)`):

```typescript
// POST /cleanup-orphan-models — one-shot maintenance op. Walks every Model
// row whose name matches the legacy "finetune-<8hex>" pattern AND has
// finetuneJobId = NULL, then either:
//   - back-links the FK if the prefix matches an existing FineTuneJob, OR
//   - deletes the row if no matching job AND no active Deployment uses it.
//
// Returns counts: { backlinked, deleted, kept_due_to_deployment }.
finetuneRouter.post("/cleanup-orphan-models", async (_req, res) => {
  const legacyPattern = /^finetune-([0-9a-z]{8})$/;

  const candidates = await prisma.model.findMany({
    where: { finetuneJobId: null, name: { startsWith: "finetune-" } },
    include: { deployments: true },
  });

  let backlinked = 0;
  let deleted = 0;
  let kept_due_to_deployment = 0;

  for (const m of candidates) {
    const match = legacyPattern.exec(m.name);
    if (!match) continue;
    const prefix = match[1];

    // Cheapest way to find a job whose id STARTS WITH prefix in SQLite:
    // it's not indexed but Model count is tiny in practice so a scan is fine.
    const job = await prisma.fineTuneJob.findFirst({
      where: { id: { startsWith: prefix } },
    });

    if (job) {
      await prisma.model.update({
        where: { id: m.id },
        data: { finetuneJobId: job.id },
      });
      backlinked++;
      continue;
    }

    const active = m.deployments.filter(
      (d) => !["stopped", "failed", "removed"].includes(d.status),
    );
    if (active.length > 0) {
      kept_due_to_deployment++;
      continue;
    }

    await prisma.model.delete({ where: { id: m.id } });
    deleted++;
  }

  res.json({ backlinked, deleted, kept_due_to_deployment });
});
```

- [ ] **Step 4: Run test, verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts -t "cleanup-orphan"`

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/routes/finetune.ts packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts
git commit -m "finetune: one-shot cleanup-orphan-models endpoint"
```

---

## Task 8: GET surfaces displayName everywhere

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (GET handlers around lines 10 and 18)

The earlier integration tests in Task 2 already assert `displayName` shows up on POST/GET responses, but a sanity-check pass to be sure every GET handler `include`s the right fields and doesn't accidentally hide `displayName` via a select clause is worth a focused diff.

- [ ] **Step 1: Run all tests so far to catch any regression**

Run: `npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts`

Expected: every test added so far passes.

- [ ] **Step 2: Skim the GET handlers for selects that could drop the field**

Run: `grep -nE "fineTuneJob\.(findMany|findUnique)" packages/server/src/routes/finetune.ts`

For each match, verify there is no `select: {...}` that omits `displayName`. The current code uses `include` rather than `select`, so `displayName` is returned by default — no change required.

- [ ] **Step 3: Commit only if anything changed**

If a handler turned out to use `select`, fix it and commit:

```bash
git add packages/server/src/routes/finetune.ts
git commit -m "finetune: ensure GET handlers don't drop displayName"
```

Otherwise no commit.

---

## Task 9: Dashboard — type updates for `displayName` + `Model.name`

**Files:**
- Modify: `packages/dashboard/app/finetune/page.tsx`

- [ ] **Step 1: Locate the `FineTuneJob` interface**

Run: `grep -n "interface FineTuneJob" packages/dashboard/app/finetune/page.tsx`

Expected: one match. The interface block follows it.

- [ ] **Step 2: Add `displayName` to the interface**

In `packages/dashboard/app/finetune/page.tsx`, inside the `interface FineTuneJob {` block, add:

```typescript
  displayName: string | null;
```

(Anywhere in the block; alongside the other top-level fields is fine.)

- [ ] **Step 3: TypeScript-check**

Run: `cd packages/dashboard && npx tsc --noEmit`

Expected: no new errors.

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/app/finetune/page.tsx
git commit -m "dashboard: typed displayName on FineTuneJob"
```

---

## Task 10: Dashboard — name input in the launch dialog

**Files:**
- Modify: `packages/dashboard/app/finetune/page.tsx` (launch form + submit handler around line 290–310)

- [ ] **Step 1: Find the launch form state declarations**

Run: `grep -nE "useState.*dataset|useState.*selectedRecipe" packages/dashboard/app/finetune/page.tsx | head`

Note: the dataset state is your anchor — add `displayName` state right next to it.

- [ ] **Step 2: Add `displayName` state**

Near the other launch-form `useState` calls (e.g. `const [dataset, setDataset] = useState("");`), add:

```typescript
  const [newJobName, setNewJobName] = useState("");
```

- [ ] **Step 3: Add the input to the form JSX**

Find the launch dialog form (the same block that contains the dataset selector). Add a name input above the recipe picker:

```tsx
  <div className="mb-3">
    <label className="block text-xs text-gray-400 mb-1">
      Name <span className="text-gray-600">(optional — defaults to recipe + job id)</span>
    </label>
    <input
      type="text"
      value={newJobName}
      onChange={(e) => setNewJobName(e.target.value)}
      placeholder="e.g. build123d-v1"
      className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
      maxLength={80}
    />
  </div>
```

- [ ] **Step 4: Wire the name into the POST body**

In the submit handler (around line 307), find the `body` object construction:

```typescript
  const body: Record<string, unknown> = { recipeFile: selectedRecipe, dataset, config };
```

Change to:

```typescript
  const body: Record<string, unknown> = { recipeFile: selectedRecipe, dataset, config };
  if (newJobName.trim()) body.displayName = newJobName.trim();
```

After the successful POST (the `setJobs(...)` line), also clear the input:

```typescript
  setDataset("");
  setNewJobName("");
```

- [ ] **Step 5: Rebuild dashboard + smoke-test in browser**

Run from repo root:
```
MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build dashboard
```

Then open http://192.168.44.36:3000/finetune, fill in a name, submit, verify the name appears on the new job row. If you can't easily exercise it (no live cluster), at least verify the build succeeded and the input renders by curling the page.

- [ ] **Step 6: Commit**

```bash
git add packages/dashboard/app/finetune/page.tsx
git commit -m "dashboard: name input in fine-tune launch dialog"
```

---

## Task 11: Dashboard — inline rename action on job rows

**Files:**
- Modify: `packages/dashboard/app/finetune/page.tsx` (job row rendering around line 820–850)

- [ ] **Step 1: Add rename state**

Near the other top-of-component `useState` calls, add:

```typescript
  // Map of jobId -> current draft display name while the user is editing it.
  // Presence of the key means "editing"; absence means "not editing".
  const [renameDraft, setRenameDraft] = useState<Record<string, string>>({});
```

- [ ] **Step 2: Add the rename handler**

Above the `return` of the component, add:

```typescript
  const startRename = (job: FineTuneJob) => {
    setRenameDraft((prev) => ({ ...prev, [job.id]: job.displayName ?? "" }));
  };

  const cancelRename = (id: string) => {
    setRenameDraft((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  };

  const submitRename = async (id: string) => {
    const draft = renameDraft[id] ?? "";
    try {
      const updated = await apiFetch<FineTuneJob>(`/api/finetune/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ displayName: draft.trim() || null }),
      });
      setJobs((prev) => prev.map((j) => (j.id === id ? { ...j, displayName: updated.displayName } : j)));
      cancelRename(id);
    } catch (err) {
      alert(String(err));
    }
  };
```

- [ ] **Step 3: Add the Rename button next to Delete**

Find the job row's action group (the `<button>` block that includes Deploy/Resume/Delete, around line 820–850). Add a Rename button next to the Delete:

```tsx
  {!isActive && !isStopping && (
    <button
      onClick={() => startRename(job)}
      className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 transition-colors"
    >
      Rename
    </button>
  )}
```

- [ ] **Step 4: Render the inline rename form when editing**

Find where the job's display label is rendered (search for `job.baseModel` or `job.recipeFile` in the row's left-hand cell). Wrap it conditionally:

```tsx
  {renameDraft[job.id] !== undefined ? (
    <div className="flex items-center gap-2">
      <input
        type="text"
        value={renameDraft[job.id]}
        onChange={(e) => setRenameDraft((prev) => ({ ...prev, [job.id]: e.target.value }))}
        placeholder="(no name)"
        className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white w-48"
        autoFocus
        onKeyDown={(e) => {
          if (e.key === "Enter") submitRename(job.id);
          if (e.key === "Escape") cancelRename(job.id);
        }}
      />
      <button
        onClick={() => submitRename(job.id)}
        className="text-xs px-2 py-1 rounded bg-green-900/50 hover:bg-green-800 text-green-300"
      >Save</button>
      <button
        onClick={() => cancelRename(job.id)}
        className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-400"
      >Cancel</button>
    </div>
  ) : (
    <span className="text-sm font-medium">
      {job.displayName || `${job.recipeFile?.split("/").pop() || "job"}-${job.id.slice(0, 8)}`}
    </span>
  )}
```

The fallback label is `<recipe-name>-<8hex>` so an un-named job still has a readable identifier.

- [ ] **Step 5: TypeScript-check**

Run: `cd packages/dashboard && npx tsc --noEmit`

Expected: no new errors.

- [ ] **Step 6: Rebuild + manual smoke**

Run: `MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build dashboard`

Verify in browser: click Rename on a completed job, type a new name, press Enter, confirm the row updates and the new name persists across page reloads.

- [ ] **Step 7: Commit**

```bash
git add packages/dashboard/app/finetune/page.tsx
git commit -m "dashboard: inline rename action on fine-tune job rows"
```

---

## Task 12: Existing-data cleanup against the live DB

**Files:** none modified. Run-only task.

Once the manager is deployed with the new code, hit the cleanup endpoint once to purge whatever orphans accumulated before the FK existed.

- [ ] **Step 1: Snapshot the Model table before**

Run from a node that can reach the manager:

```
curl -s http://192.168.44.36:4000/api/models | jq 'length, (map(.name) | sort)' > /tmp/models-before.json
cat /tmp/models-before.json
```

- [ ] **Step 2: Run cleanup**

Run:

```
curl -s -X POST http://192.168.44.36:4000/api/finetune/cleanup-orphan-models | jq
```

Expected output shape:

```
{
  "backlinked": <int>,
  "deleted": <int>,
  "kept_due_to_deployment": <int>
}
```

- [ ] **Step 3: Snapshot after + diff**

Run:

```
curl -s http://192.168.44.36:4000/api/models | jq 'length, (map(.name) | sort)' > /tmp/models-after.json
diff /tmp/models-before.json /tmp/models-after.json
```

Expected: the diff shows the deleted Model names removed and the remaining list now matches what's actually deployable.

- [ ] **Step 4: No commit needed** (this task only mutates runtime state).

---

## Final pass

- [ ] **Step 1: Full suite green**

Run: `npm test`

Expected: all tests pass. The pre-existing `deployments.vram-admission.test.ts` flake (test-isolation issue when running the whole suite) is unrelated and may persist — note it in the commit log if needed.

- [ ] **Step 2: TypeScript clean across server + dashboard**

Run:
```
cd packages/server && npx tsc --noEmit
cd packages/dashboard && npx tsc --noEmit
```

Expected: no new errors in either.

- [ ] **Step 3: Rebuild both containers**

Run: `MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build`

- [ ] **Step 4: End-to-end smoke**

In the dashboard:
1. Launch a tiny test fine-tune with a `displayName` set in the dialog → confirm the job row shows the name.
2. Rename the job via the inline editor → confirm the row updates and the API returns the new name.
3. After merge completes, hit Deploy → confirm the resulting Model row's `name` matches the displayName.
4. Delete the job (after stopping any deployment) → confirm the Model row disappears from `GET /api/models`.

- [ ] **Step 5: PR / push**

```bash
git push origin claude/setup-dev-environment-gSVCI
```

If you'd open a PR for this branch (this repo's default branch IS the working branch — see the existing repo's setup), do so via `gh pr create` per the standard flow.
