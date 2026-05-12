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
});
