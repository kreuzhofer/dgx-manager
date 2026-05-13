/**
 * Integration tests for POST /api/finetune/:id/quantize.
 *
 * Same pattern as deployments.vram-admission.test.ts: per-suite SQLite,
 * stub agent hub, supertest against an Express app that mounts only
 * the finetune router.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

// Capture SSE broadcasts instead of writing to real HTTP response streams.
const broadcasts: { type: string; payload: unknown }[] = [];
vi.mock("../../sse.js", () => ({
  broadcast: (event: { type: string; payload: unknown }) => {
    broadcasts.push(event);
  },
  sseHandler: vi.fn(),
}));

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

describe("AgentHub: quantize-complete persists state", () => {
  let AgentHub: typeof import("../../ws/agent-hub.js").AgentHub;

  beforeAll(async () => {
    ({ AgentHub } = await import("../../ws/agent-hub.js"));
  });

  it("transitions to quantized and stores quantizedPath", async () => {
    broadcasts.length = 0;
    const node = await prisma.node.create({
      data: { id: "n2", name: "n2", ipAddress: "10.0.0.2", agentPort: 8089, status: "online", vramTotal: 122000 },
    });
    const job = await prisma.fineTuneJob.create({
      data: {
        nodeId: node.id,
        recipeFile: "recipes/test",
        baseModel: "Qwen/Qwen3.6-27B",
        method: "lora",
        dataset: "/tmp/ds.jsonl",
        status: "completed",
        mergeStatus: "completed",
        mergedPath: "/mnt/tank/outputs/job-2/merged",
        outputDir: "/mnt/tank/outputs/job-2",
        quantizationStatus: "quantizing",
        quantizedPath: null,
      },
    });

    const hub = new AgentHub();
    await hub.handleAgentMessage({
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
    broadcasts.length = 0;
    const node = await prisma.node.create({
      data: { id: "n3", name: "n3", ipAddress: "10.0.0.3", agentPort: 8089, status: "online", vramTotal: 122000 },
    });
    const job = await prisma.fineTuneJob.create({
      data: {
        nodeId: node.id,
        recipeFile: "recipes/test",
        baseModel: "Qwen/Qwen3.6-27B",
        method: "lora",
        dataset: "/tmp/ds.jsonl",
        status: "completed",
        mergeStatus: "completed",
        mergedPath: "/mnt/tank/outputs/job-3/merged",
        outputDir: "/mnt/tank/outputs/job-3",
        quantizationStatus: "quantizing",
        quantizedPath: "/mnt/tank/outputs/job-3/merged-fp8",
      },
    });

    const hub = new AgentHub();
    await hub.handleAgentMessage({
      type: "agent:finetune:quantize-complete",
      payload: { jobId: job.id, status: "failed", quantizedPath: null, error: "OOM at FP8 cast" },
    });

    const updated = await prisma.fineTuneJob.findUnique({ where: { id: job.id } });
    expect(updated?.quantizationStatus).toBe("failed");
    expect(updated?.quantizedPath).toBeNull();
    expect(updated?.quantizationLog).toBe("OOM at FP8 cast");
  });
});
