/**
 * The restart route used to always send cmd:deploy with config.recipeFile,
 * but fine-tune deployments don't have a recipeFile in their saved config
 * (it lives on the FineTuneJob row). The agent rejected those restarts
 * with "No recipeFile specified" — the deployment went terminal-failed
 * with no useful surface.
 *
 * These tests pin the fix: restart of a fine-tune deployment must send
 * cmd:finetune:deploy with the same payload shape the original finetune
 * deploy route uses, drawing the recipe + base model from the linked
 * FineTuneJob and the artifactVariant from saved config (default bf16).
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-deploy-restart-ft-test-"));
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

type SentMessage = { nodeId: string; message: { type: string; payload: Record<string, unknown> } };

const TRAINING_RECIPE = {
  file: "recipes/test-training",
  name: "Test Training Recipe",
  deploy: {
    container: "vllm-node-custom",
    gpu_memory_utilization: 0.85,
    max_model_len: 8192,
  },
};

function makeStubHub() {
  const sent: SentMessage[] = [];
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

async function wipeAll() {
  await prisma.loadBalancerEndpoint.deleteMany();
  await prisma.clusterNode.deleteMany();
  await prisma.deployment.deleteMany();
  await prisma.fineTuneJob.deleteMany();
  await prisma.model.deleteMany();
  await prisma.metricSnapshot.deleteMany();
  await prisma.node.deleteMany();
}

beforeEach(wipeAll);

async function seedFineTuneDeployment(opts: { artifactVariant?: string; status?: string } = {}) {
  const node = await prisma.node.create({
    data: { name: "n1", status: "online", vramTotal: 128000, ipAddress: "10.0.0.10" },
  });
  const job = await prisma.fineTuneJob.create({
    data: {
      nodeId: node.id,
      baseModel: "meta-llama/Llama-3.1-8B",
      method: "lora",
      dataset: "test-dataset",
      recipeFile: "recipes/test-training",
      status: "completed",
      mergeStatus: "completed",
      outputDir: "/mnt/tank/outputs/job1",
      mergedPath: "/mnt/tank/outputs/job1/merged",
    },
  });
  const model = await prisma.model.create({
    data: { name: "finetune-job1", runtime: "vllm", finetuneJobId: job.id },
  });
  const config: Record<string, unknown> = {
    port: 8000,
    gpuMem: 0.8,
    maxModelLen: 128000,
    tensorParallel: 4,
    localModelPath: "/mnt/tank/outputs/job1/merged",
  };
  if (opts.artifactVariant) config.artifactVariant = opts.artifactVariant;
  const deployment = await prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: opts.status ?? "failed",
      port: 8000,
      displayName: "chat3d-prod",
      config: JSON.stringify(config),
    },
  });
  return { node, job, model, deployment };
}

describe("POST /api/deployments/:id/restart for fine-tune deployments", () => {
  it("sends cmd:finetune:deploy (not cmd:deploy) when the model has a finetuneJobId", async () => {
    const { deployment, job } = await seedFineTuneDeployment();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(res.status).toBe(200);
    expect(sent).toHaveLength(1);
    expect(sent[0].message.type).toBe("cmd:finetune:deploy");
    // The agent's cmd:finetune:deploy handler keys on these fields:
    expect(sent[0].message.payload.jobId).toBe(job.id);
    expect(sent[0].message.payload.deploymentId).toBe(deployment.id);
    expect(sent[0].message.payload.modelPath).toBe("/mnt/tank/outputs/job1/merged");
    expect(sent[0].message.payload.baseModel).toBe("meta-llama/Llama-3.1-8B");
    expect(sent[0].message.payload.recipeFile).toBe("recipes/test-training");
    expect(sent[0].message.payload.modelName).toBe("chat3d-prod");
  });

  it("defaults artifactVariant to bf16 when the saved config doesn't have it (back-compat for pre-fix deployments)", async () => {
    const { deployment } = await seedFineTuneDeployment(); // no artifactVariant set
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(res.status).toBe(200);
    expect(sent[0].message.payload.artifactVariant).toBe("bf16");
  });

  it("preserves artifactVariant=fp8 from saved config", async () => {
    const { deployment } = await seedFineTuneDeployment({ artifactVariant: "fp8" });
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(res.status).toBe(200);
    expect(sent[0].message.payload.artifactVariant).toBe("fp8");
  });

  it("uses the training recipe's deploy.container when available", async () => {
    const { deployment } = await seedFineTuneDeployment();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(sent[0].message.payload.deployContainer).toBe("vllm-node-custom");
  });

  it("falls back to vllm-node when the training recipe is no longer registered", async () => {
    const { deployment } = await seedFineTuneDeployment();
    const hub = {
      getRecipes: () => [],
      getTrainingRecipes: () => [],
      getOllamaModels: () => [],
      sendToAgent: (_n: string, _m: unknown) => { /* noop */ },
    };
    // Re-build hub with capture
    const sent: SentMessage[] = [];
    const capturingHub = {
      ...hub,
      sendToAgent: (nodeId: string, message: { type: string; payload: Record<string, unknown> }) => {
        sent.push({ nodeId, message });
      },
    };
    const app = makeApp(capturingHub);

    await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(sent[0].message.payload.deployContainer).toBe("vllm-node");
  });

  it("merges caller-supplied config overrides (e.g. lower maxModelLen) into the payload AND persists them", async () => {
    const { deployment } = await seedFineTuneDeployment();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post(`/api/deployments/${deployment.id}/restart`)
      .send({ config: { maxModelLen: 32768 } });

    expect(res.status).toBe(200);
    const payloadConfig = sent[0].message.payload.config as Record<string, unknown>;
    expect(payloadConfig.maxModelLen).toBe(32768);
    // Original fields should be retained
    expect(payloadConfig.gpuMem).toBe(0.8);
    expect(payloadConfig.tensorParallel).toBe(4);

    // Saved config should be updated to reflect the override
    const after = await prisma.deployment.findUnique({ where: { id: deployment.id } });
    const saved = JSON.parse(after!.config!);
    expect(saved.maxModelLen).toBe(32768);
  });

  it("still uses cmd:deploy for non-finetune deployments (regression guard)", async () => {
    const node = await prisma.node.create({
      data: { name: "n1", status: "online", vramTotal: 128000, ipAddress: "10.0.0.10" },
    });
    const model = await prisma.model.create({ data: { name: "plain-model", runtime: "vllm" } });
    const deployment = await prisma.deployment.create({
      data: {
        nodeId: node.id,
        modelId: model.id,
        status: "failed",
        port: 8000,
        config: JSON.stringify({ port: 8000, recipeFile: "recipes/plain.yaml" }),
      },
    });
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(sent[0].message.type).toBe("cmd:deploy");
    expect(sent[0].message.payload.recipeFile).toBe("recipes/plain.yaml");
  });
});
