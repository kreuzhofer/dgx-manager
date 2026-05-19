/**
 * Variant-id propagation through the restart route. Complements the
 * existing deployments.restart-finetune suite by testing arbitrary
 * variant slugs (not just the legacy bf16/fp8 pair) and rejection of
 * malformed slugs at the route layer.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-restart-variant-test-"));
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

beforeEach(async () => {
  await prisma.loadBalancerEndpoint.deleteMany();
  await prisma.clusterNode.deleteMany();
  await prisma.deployment.deleteMany();
  await prisma.fineTuneJob.deleteMany();
  await prisma.model.deleteMany();
  await prisma.node.deleteMany();
});

const TRAINING_RECIPE = {
  file: "recipes/test-training",
  name: "Test Training Recipe",
  deploy: { container: "vllm-node-custom" },
};

function makeStubHub() {
  const sent: { nodeId: string; message: { type: string; payload: Record<string, unknown> } }[] = [];
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

async function seedFineTuneDeployment(artifactVariant?: string) {
  const node = await prisma.node.create({
    data: { name: "n1", status: "online", vramTotal: 128000, ipAddress: "10.0.0.10" },
  });
  const job = await prisma.fineTuneJob.create({
    data: {
      nodeId: node.id, baseModel: "meta-llama/Llama-3.1-8B", method: "lora",
      dataset: "test", recipeFile: "recipes/test-training",
      status: "completed", mergeStatus: "completed",
      outputDir: "/mnt/tank/outputs/job1", mergedPath: "/mnt/tank/outputs/job1/merged",
    },
  });
  const model = await prisma.model.create({
    data: { name: "finetune-job1", runtime: "vllm", finetuneJobId: job.id },
  });
  const cfg: Record<string, unknown> = {
    port: 8000, gpuMem: 0.8, maxModelLen: 8192,
    localModelPath: "/mnt/tank/outputs/job1/merged",
  };
  if (artifactVariant) cfg.artifactVariant = artifactVariant;
  return prisma.deployment.create({
    data: {
      nodeId: node.id, modelId: model.id, status: "failed", port: 8000,
      config: JSON.stringify(cfg),
    },
  });
}

describe("POST /api/deployments/:id/restart — arbitrary variant ids", () => {
  it("accepts a custom variant slug and forwards it as artifactVariant", async () => {
    const dep = await seedFineTuneDeployment();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post(`/api/deployments/${dep.id}/restart`)
      .send({ config: { artifactVariant: "int4" } });

    expect(res.status).toBe(200);
    expect(sent[0].message.type).toBe("cmd:finetune:deploy");
    expect(sent[0].message.payload.artifactVariant).toBe("int4");

    // And persisted in saved config for the next restart cycle
    const saved = JSON.parse((await prisma.deployment.findUnique({ where: { id: dep.id } }))!.config!);
    expect(saved.artifactVariant).toBe("int4");
  });

  it("rejects malformed variant slugs with 400", async () => {
    const dep = await seedFineTuneDeployment();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post(`/api/deployments/${dep.id}/restart`)
      .send({ config: { artifactVariant: "../etc/passwd" } });

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/artifactVariant/);
  });

  it("preserves the legacy 'bf16' slug through restart (no auto-migration)", async () => {
    // Pre-feature deployments stored "bf16" — the route shouldn't rewrite
    // it on restart. Storage-side back-compat is the agent's job.
    const dep = await seedFineTuneDeployment("bf16");
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app).post(`/api/deployments/${dep.id}/restart`).send({});

    expect(res.status).toBe(200);
    expect(sent[0].message.payload.artifactVariant).toBe("bf16");
  });
});
