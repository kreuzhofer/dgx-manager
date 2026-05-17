/**
 * Integration coverage for Deployment.displayName on the stock vLLM and
 * finetune deploy routes, plus the restart preservation/override path.
 *
 * Pattern matches finetune.naming-and-cleanup.test.ts: per-suite SQLite,
 * supertest, stub agentHub that captures outgoing messages.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-deploy-displayname-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;
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
  ({ deploymentsRouter } = await import("../../routes/deployments.js"));
  ({ finetuneRouter } = await import("../../routes/finetune.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

const RECIPE = {
  file: "recipes/test-recipe.yaml",
  name: "Test Recipe",
  defaults: { gpu_memory_utilization: 0.5 },
};

function makeStubHub() {
  const sent: { nodeId: string; message: { type: string; payload: Record<string, unknown> } }[] = [];
  return {
    hub: {
      getRecipes: () => [RECIPE],
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

async function wipeAll() {
  // FK-safe deletion order: leaf tables first. fineTuneJob is removed
  // between deployment and model because Model.finetuneJobId is a
  // (nullable) FK to FineTuneJob — without this, Task 5's seeds break
  // when this `beforeEach` runs.
  await prisma.loadBalancerEndpoint.deleteMany();
  await prisma.clusterNode.deleteMany();
  await prisma.deployment.deleteMany();
  await prisma.fineTuneJob.deleteMany();
  await prisma.model.deleteMany();
  await prisma.metricSnapshot.deleteMany();
  await prisma.node.deleteMany();
}

async function seedNode(name = "n1") {
  return prisma.node.create({
    data: {
      name,
      status: "online",
      vramTotal: 128000,
      ipAddress: "10.0.0.10",
    },
  });
}

beforeEach(wipeAll);

describe("POST /api/deployments with displayName", () => {
  it("persists displayName when supplied and forwards it as servedModelName", async () => {
    const node = await seedNode();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "chat3d-prod",
        config: { port: 8000 },
      });

    expect(res.status).toBe(201);
    expect(res.body.displayName).toBe("chat3d-prod");

    const row = await prisma.deployment.findUnique({ where: { id: res.body.id } });
    expect(row?.displayName).toBe("chat3d-prod");

    expect(sent).toHaveLength(1);
    expect(sent[0].message.type).toBe("cmd:deploy");
    expect(sent[0].message.payload.servedModelName).toBe("chat3d-prod");
  });

  it("leaves displayName null when omitted, and does not send servedModelName", async () => {
    const node = await seedNode();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        config: { port: 8000 },
      });

    expect(res.status).toBe(201);
    expect(res.body.displayName).toBeNull();
    expect(sent[0].message.payload.servedModelName).toBeUndefined();
  });

  it("rejects illegal characters with 400", async () => {
    const node = await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "chat 3d", // space rejected
        config: { port: 8000 },
      });

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/letters, digits/);
  });

  it("rejects duplicate displayName among running deployments with 409", async () => {
    const node = await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    // First deploy claims the name.
    await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "chat3d-prod",
        config: { port: 8000 },
      });

    // Mark it running so it counts toward uniqueness (handler creates as pending).
    await prisma.deployment.updateMany({
      where: { displayName: "chat3d-prod" },
      data: { status: "running" },
    });

    // Second deploy with same name → conflict.
    const node2 = await seedNode("n2");
    const res = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node2.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "chat3d-prod",
        config: { port: 8000 },
      });

    expect(res.status).toBe(409);
    expect(res.body.error).toMatch(/already in use/);
  });

  it("rejects displayName for Ollama deployments with 400", async () => {
    const node = await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        runtime: "ollama",
        modelName: "llama3.1:8b",
        displayName: "my-ollama-name",
      });

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/not supported for Ollama/);
  });
});

describe("POST /api/deployments/:id/restart with displayName", () => {
  it("preserves the existing displayName when body has no override", async () => {
    const node = await seedNode();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const created = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "preserved-name",
        config: { port: 8000 },
      });
    sent.length = 0; // clear send log

    const restart = await request(app)
      .post(`/api/deployments/${created.body.id}/restart`)
      .send({});

    expect(restart.status).toBe(200);
    const row = await prisma.deployment.findUnique({ where: { id: created.body.id } });
    expect(row?.displayName).toBe("preserved-name");
    expect(sent[0].message.payload.servedModelName).toBe("preserved-name");
  });

  it("accepts a displayName override and re-validates uniqueness (excluding self)", async () => {
    const node = await seedNode();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const created = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "original",
        config: { port: 8000 },
      });
    await prisma.deployment.update({
      where: { id: created.body.id },
      data: { status: "running" },
    });
    sent.length = 0;

    const restart = await request(app)
      .post(`/api/deployments/${created.body.id}/restart`)
      .send({ displayName: "renamed" });

    expect(restart.status).toBe(200);
    const row = await prisma.deployment.findUnique({ where: { id: created.body.id } });
    expect(row?.displayName).toBe("renamed");
    expect(sent[0].message.payload.servedModelName).toBe("renamed");
  });

  it("clears displayName when restart body sets it to null explicitly", async () => {
    const node = await seedNode();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const created = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "to-be-cleared",
        config: { port: 8000 },
      });
    sent.length = 0;

    const restart = await request(app)
      .post(`/api/deployments/${created.body.id}/restart`)
      .send({ displayName: null });

    expect(restart.status).toBe(200);
    const row = await prisma.deployment.findUnique({ where: { id: created.body.id } });
    expect(row?.displayName).toBeNull();
    expect(sent[0].message.payload.servedModelName).toBeUndefined();
  });
});

describe("POST /api/finetune/:id/deploy with displayName override", () => {
  // Local app helper that mounts BOTH routers (uniqueness check spans both
  // routes — finetune deploys need to see active deployments created via
  // POST /api/deployments and vice versa).
  function makeFtApp(hub: unknown) {
    const app = express();
    app.use(express.json());
    app.set("agentHub", hub);
    app.use("/api/finetune", finetuneRouter);
    app.use("/api/deployments", deploymentsRouter);
    return app;
  }

  function makeFtStubHub() {
    const sent: { nodeId: string; message: { type: string; payload: Record<string, unknown> } }[] = [];
    return {
      hub: {
        getRecipes: () => [],
        getOllamaModels: () => [],
        getTrainingRecipes: () => [
          {
            file: "recipes/test-attn-mlp",
            name: "Test FT Recipe",
            base_model: "Qwen/Qwen3.6-27B",
            method: "lora",
            defaults: {},
            scripts: { merge: "scripts/merge.py" },
            deploy: { gpu_memory_utilization: 0.5 },
          },
        ],
        sendToAgent: (nodeId: string, message: { type: string; payload: Record<string, unknown> }) => {
          sent.push({ nodeId, message });
        },
      },
      sent,
    };
  }

  async function seedCompletedJob(node: Awaited<ReturnType<typeof seedNode>>) {
    return prisma.fineTuneJob.create({
      data: {
        nodeId: node.id,
        baseModel: "Qwen/Qwen3.6-27B",
        method: "lora",
        dataset: "/tmp/ds.jsonl",
        recipeFile: "recipes/test-attn-mlp",
        status: "completed",
        mergeStatus: "completed",
        mergedPath: "/tmp/merged",
        displayName: "chat3d-build123d-01",
      },
    });
  }

  it("uses the FineTuneJob.displayName when no per-deploy displayName is supplied", async () => {
    const node = await seedNode();
    const job = await seedCompletedJob(node);
    const { hub, sent } = makeFtStubHub();
    const app = makeFtApp(hub);

    const res = await request(app)
      .post(`/api/finetune/${job.id}/deploy`)
      .send({ nodeId: node.id, config: { port: 8000 } });

    expect(res.status).toBe(201);
    expect(res.body.displayName).toBeNull();
    expect(sent[0].message.type).toBe("cmd:finetune:deploy");
    // Falls back to Model.name, which is the FineTuneJob.displayName.
    expect(sent[0].message.payload.modelName).toBe("chat3d-build123d-01");
  });

  it("uses the per-deploy displayName when supplied (overrides FT name for this deploy)", async () => {
    const node = await seedNode();
    const job = await seedCompletedJob(node);
    const { hub, sent } = makeFtStubHub();
    const app = makeFtApp(hub);

    const res = await request(app)
      .post(`/api/finetune/${job.id}/deploy`)
      .send({
        nodeId: node.id,
        displayName: "chat3d-prod-variant-a",
        config: { port: 8000 },
      });

    expect(res.status).toBe(201);
    expect(res.body.displayName).toBe("chat3d-prod-variant-a");
    expect(sent[0].message.payload.modelName).toBe("chat3d-prod-variant-a");
  });

  it("rejects 409 when the per-deploy displayName conflicts with an active deployment", async () => {
    const node = await seedNode();
    const job = await seedCompletedJob(node);
    const { hub } = makeFtStubHub();
    const app = makeFtApp(hub);

    // Seed an existing running deployment with the contested name.
    await prisma.deployment.create({
      data: {
        nodeId: node.id,
        modelId: (await prisma.model.create({ data: { name: "other", runtime: "vllm" } })).id,
        status: "running",
        displayName: "taken-name",
      },
    });

    const res = await request(app)
      .post(`/api/finetune/${job.id}/deploy`)
      .send({
        nodeId: node.id,
        displayName: "taken-name",
        config: { port: 8000 },
      });

    expect(res.status).toBe(409);
    expect(res.body.error).toMatch(/already in use/);
  });
});
