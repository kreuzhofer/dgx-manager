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
  // FK-safe deletion order: leaf tables first, then parents.
  // trainingMetric → fineTuneClusterNode → deployment → model →
  // fineTuneJob → metricSnapshot → node
  await prisma.trainingMetric.deleteMany({});
  await prisma.fineTuneClusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.fineTuneJob.deleteMany({});
  await prisma.metricSnapshot.deleteMany({});
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
    expect(dep.status).toBe(201);

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

  it("POST /:id/deploy returns 409 when another Model has the same name", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    // Pre-seed a Model with the name the new deploy will want.
    await prisma.model.create({
      data: { name: "collision-name", runtime: "vllm" },
    });

    // Create a job that wants the same name.
    const create = await request(app)
      .post("/api/finetune")
      .send({
        nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl",
        displayName: "collision-name",
      });
    await prisma.fineTuneJob.update({
      where: { id: create.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged" },
    });

    const dep = await request(app)
      .post(`/api/finetune/${create.body.id}/deploy`)
      .send({ config: {} });

    expect(dep.status).toBe(409);
    expect(dep.body.error).toMatch(/already exists/i);
    expect(dep.body.error).toMatch(/collision-name/);
  });

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

  it("PATCH /:id returns 409 when the new displayName collides with another Model.name", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    // Job A: deployed with name "first"
    const a = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl",
              displayName: "first" });
    await prisma.fineTuneJob.update({
      where: { id: a.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged-a" },
    });
    await request(app).post(`/api/finetune/${a.body.id}/deploy`).send({ config: {} });

    // Job B: deployed with name "second"
    const b = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl",
              displayName: "second" });
    await prisma.fineTuneJob.update({
      where: { id: b.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged-b" },
    });
    await request(app).post(`/api/finetune/${b.body.id}/deploy`).send({ config: {} });

    // Try to rename A to "second" — should 409.
    const patch = await request(app)
      .patch(`/api/finetune/${a.body.id}`)
      .send({ displayName: "second" });
    expect(patch.status).toBe(409);
    expect(patch.body.error).toMatch(/already exists/i);

    // Both Model rows still have their original names.
    const aModel = await prisma.model.findFirst({ where: { finetuneJobId: a.body.id } });
    const bModel = await prisma.model.findFirst({ where: { finetuneJobId: b.body.id } });
    expect(aModel?.name).toBe("first");
    expect(bModel?.name).toBe("second");

    // Atomicity: FineTuneJob A's displayName must NOT have been updated
    // since the transactional rename rolled back.
    const aJob = await prisma.fineTuneJob.findUnique({ where: { id: a.body.id } });
    expect(aJob?.displayName).toBe("first");
  });

  it("DELETE /:id removes the associated Model row via cascade", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl" });

    // Create the Model row directly (bypass POST /deploy so we don't also
    // get a pending Deployment row that would block the delete). This
    // isolates the FK cascade behavior under test.
    await prisma.model.create({
      data: {
        name: `finetune-${create.body.id.slice(0, 8)}`,
        runtime: "vllm",
        finetuneJobId: create.body.id,
      },
    });

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

  it("DELETE /:id refuses when the Model has a pending Deployment (agent in-flight)", async () => {
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
    // This creates the Model AND a `pending` Deployment via the stub hub.
    await request(app).post(`/api/finetune/${create.body.id}/deploy`).send({ config: {} });

    const del = await request(app).delete(`/api/finetune/${create.body.id}`);
    expect(del.status).toBe(409);
    expect(del.body.error).toMatch(/active deployment/i);

    // Job + Model + pending Deployment are all still there
    expect(await prisma.fineTuneJob.count({ where: { id: create.body.id } })).toBe(1);
    expect(await prisma.model.count({ where: { finetuneJobId: create.body.id } })).toBe(1);
    expect(await prisma.deployment.count({ where: { modelId: { not: undefined } } })).toBeGreaterThanOrEqual(1);
  });

  it("DELETE /:id sweeps terminal Deployment rows then cascades the Model", async () => {
    await wipeAll();
    await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl" });

    // Set up Model + a terminal Deployment directly via Prisma.
    const model = await prisma.model.create({
      data: {
        name: `finetune-${create.body.id.slice(0, 8)}`,
        runtime: "vllm",
        finetuneJobId: create.body.id,
      },
    });
    const stoppedDep = await prisma.deployment.create({
      data: { nodeId: "node-1", modelId: model.id, status: "stopped" },
    });

    const del = await request(app).delete(`/api/finetune/${create.body.id}`);
    expect(del.status).toBe(200);

    // Cascade chain: FineTuneJob gone, Model gone (via cascade), stopped
    // Deployment gone (via explicit sweep).
    expect(await prisma.fineTuneJob.count({ where: { id: create.body.id } })).toBe(0);
    expect(await prisma.model.count({ where: { id: model.id } })).toBe(0);
    expect(await prisma.deployment.count({ where: { id: stoppedDep.id } })).toBe(0);
  });

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
});
