/**
 * Integration test for the recipe/node arch-mismatch 4xx path on
 * POST /api/deployments. Mirrors deployments.vram-admission.test.ts's harness.
 *
 * Verifies:
 *   - An arm64 registry recipe deployed to an amd64 node → 400 arch-mismatch,
 *     nothing dispatched.
 *   - The same recipe on a matching arm64 node → admitted (cmd:deploy sent).
 *   - An arch-agnostic ("any") recipe is admitted on any node.
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

function makeStubHub(
  recipes: { file: string; arch: string; defaults: Record<string, unknown> }[],
) {
  const sentMessages: { nodeId: string; message: unknown }[] = [];
  return {
    hub: {
      getRecipes: () => recipes,
      getOllamaModels: () => [],
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
  // Hermetic: stub sshExec so the maxoutmem reclaim never opens a real SSH
  // connection to a fake node IP. Flag reads as absent (stdout "false").
  app.set("sshExec", async () => ({ code: 0, stdout: "false", stderr: "" }));
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

async function wipeAll() {
  await prisma.loadBalancerEndpoint.deleteMany({});
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.metricSnapshot.deleteMany({});
  await prisma.fineTuneJob.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
}

const ARM_RECIPE = {
  file: "@official/qwen3-vllm",
  arch: "arm64",
  defaults: { tensor_parallel: 1, gpu_memory_utilization: 0.5 },
};
const ANY_RECIPE = {
  file: "@official/any-vllm",
  arch: "any",
  defaults: { tensor_parallel: 1, gpu_memory_utilization: 0.5 },
};

describe("POST /api/deployments — arch admission", () => {
  it("rejects an arm64 registry recipe deployed to an amd64 node (400, nothing dispatched)", async () => {
    await wipeAll();
    await prisma.node.create({
      data: {
        id: "amd-node",
        name: "aihost01",
        ipAddress: "192.168.44.50",
        arch: "amd64",
        vramTotal: 32_000,
        status: "online",
      },
    });

    const { hub, sentMessages } = makeStubHub([ARM_RECIPE]);
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "amd-node", recipeFile: ARM_RECIPE.file, config: {} });

    expect(res.status).toBe(400);
    expect(res.body.error).toContain("mismatch");
    expect(res.body.recipeArch).toBe("arm64");
    expect(res.body.nodeArch).toBe("amd64");
    expect(sentMessages).toHaveLength(0);
  });

  it("admits an arm64 recipe on a matching arm64 node (dispatches cmd:deploy)", async () => {
    await wipeAll();
    await prisma.node.create({
      data: {
        id: "arm-node",
        name: "spark-01",
        ipAddress: "192.168.44.36",
        arch: "arm64",
        vramTotal: 122_502,
        status: "online",
      },
    });

    const { hub, sentMessages } = makeStubHub([ARM_RECIPE]);
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "arm-node", recipeFile: ARM_RECIPE.file, config: {} });

    expect(res.status).toBe(201);
    expect(sentMessages).toHaveLength(1);
    expect(sentMessages[0].nodeId).toBe("arm-node");
    expect((sentMessages[0].message as { type: string }).type).toBe("cmd:deploy");
  });

  it("admits an arch-agnostic 'any' recipe on an amd64 node", async () => {
    await wipeAll();
    await prisma.node.create({
      data: {
        id: "amd-node",
        name: "aihost01",
        ipAddress: "192.168.44.50",
        arch: "amd64",
        vramTotal: 32_000,
        status: "online",
      },
    });

    const { hub, sentMessages } = makeStubHub([ANY_RECIPE]);
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "amd-node", recipeFile: ANY_RECIPE.file, config: {} });

    expect(res.status).toBe(201);
    expect(sentMessages).toHaveLength(1);
    expect((sentMessages[0].message as { type: string }).type).toBe("cmd:deploy");
  });
});
