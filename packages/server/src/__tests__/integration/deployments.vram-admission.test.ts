/**
 * Integration test for the VRAM-admission 409 path on POST /api/deployments.
 *
 * Pattern this file establishes for the repo:
 *   - Integration tests live under `packages/<pkg>/src/__tests__/integration/`
 *     (separate folder so they can be excluded later if they get slow).
 *   - They get a per-suite SQLite via `DATABASE_URL=file:.tmp-<rand>.db`
 *     set in the env BEFORE importing prisma. `prisma db push --force-reset`
 *     creates the schema. Cleanup deletes the file at the end.
 *   - The Express app is built ad-hoc — only the router under test is
 *     mounted, with a stub AgentHub injected via `app.set("agentHub", …)`.
 *     No WebSocket, no agent processes.
 *   - HTTP exercises go through `supertest(app)` so the test never binds
 *     a port.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

// Per-suite SQLite. Must be set before any module that imports prisma.
const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

// Dynamic imports so the env var above is in place before prisma loads.
let prisma: typeof import("../../prisma.js").prisma;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;

beforeAll(async () => {
  // Apply schema to the fresh DB. The Prisma 7 CLI reads DATABASE_URL via
  // prisma.config.ts; --force-reset is safe because the DB is per-suite.
  //
  // PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION: Prisma 7's CLI refuses
  // destructive operations when invoked by an AI agent unless this env
  // var carries an explicit user consent record. The user opted in
  // ("#1" → option 1: grant consent for tests, on 2026-05-03) on the
  // explicit understanding that DATABASE_URL here always points at a
  // freshly-mkdtemp'd SQLite file in /tmp — there is no way for the
  // command to touch any real database.
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

/**
 * Stub AgentHub with just the surface the deployments route reads:
 *   - getRecipes() — recipe metadata (we provide our test recipe inline)
 *   - getOllamaModels() — empty for these tests
 *   - sendToAgent() — no-op; we're testing the admission gate, not the launch
 */
function makeStubHub(recipe: { file: string; defaults: Record<string, unknown> }) {
  const sentMessages: { nodeId: string; message: unknown }[] = [];
  return {
    hub: {
      getRecipes: () => [recipe],
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
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

/**
 * Wipe all tables in FK-dependency order so the next test starts clean.
 * Children before parents: LBEndpoint → ClusterNode → Deployment → ...
 */
async function wipeAll() {
  await prisma.loadBalancerEndpoint.deleteMany({});
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.metricSnapshot.deleteMany({});
  await prisma.fineTuneJob.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
}

const RECIPE = {
  file: "recipes/test-tp4.yaml",
  name: "Test 4-node",
  defaults: { tensor_parallel: 4, gpu_memory_utilization: 0.85 },
};

describe("POST /api/deployments — VRAM admission", () => {
  it("returns 409 with conflict when a cluster node is short on VRAM", async () => {
    // Seed: one Ollama model + deployment occupying 15 GB on what will be
    // node-3, plus 4 nodes with realistic Spark VRAM totals.
    await wipeAll();

    const nodes = await Promise.all(
      [1, 2, 3, 4].map((i) =>
        prisma.node.create({
          data: {
            id: `node-${i}`,
            name: `dgx-spark-0${i}`,
            ipAddress: `192.168.44.${35 + i}`,
            vramTotal: 122_502, // 119.69 GiB, matches GB10
            status: "online",
          },
        }),
      ),
    );
    // Latest metric for node-3: 15 GB used (Ollama model loaded).
    await prisma.metricSnapshot.create({
      data: { nodeId: "node-3", vramUsed: 15_360, gpuUtil: 0, timestamp: new Date() },
    });

    const ollamaModel = await prisma.model.create({
      data: { name: "qwen3-embedding:8b", runtime: "ollama" },
    });
    await prisma.deployment.create({
      data: {
        nodeId: "node-3",
        modelId: ollamaModel.id,
        status: "running",
        port: 11434,
        config: JSON.stringify({ runtime: "ollama", modelName: "qwen3-embedding:8b" }),
        vramActual: 15_360,
      },
    });

    const { hub, sentMessages } = makeStubHub(RECIPE);
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({
        nodeIds: nodes.map((n) => n.id),
        recipeFile: RECIPE.file,
        config: { tensorParallel: 4 },
      });

    expect(res.status).toBe(409);
    expect(res.body.error).toContain("dgx-spark-03");
    expect(res.body.error).toContain("safety margin");
    expect(res.body.shortfalls).toHaveLength(1);
    expect(res.body.shortfalls[0].nodeName).toBe("dgx-spark-03");
    expect(res.body.shortfalls[0].conflicts.map((c: { name: string }) => c.name))
      .toContain("qwen3-embedding:8b");
    // Critical: nothing was launched.
    expect(sentMessages).toHaveLength(0);
  });

  it("returns 201 (and dispatches cmd:deploy) once the conflict is removed", async () => {
    // Reset and seed without any active deployment.
    await wipeAll();

    const nodes = await Promise.all(
      [1, 2, 3, 4].map((i) =>
        prisma.node.create({
          data: {
            id: `node-${i}`,
            name: `dgx-spark-0${i}`,
            ipAddress: `192.168.44.${35 + i}`,
            vramTotal: 122_502,
            status: "online",
          },
        }),
      ),
    );
    // No metrics → vramUsed defaults to 0 in the admission helper.

    const { hub, sentMessages } = makeStubHub(RECIPE);
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({
        nodeIds: nodes.map((n) => n.id),
        recipeFile: RECIPE.file,
        config: { tensorParallel: 4 },
      });

    expect(res.status).toBe(201);
    expect(res.body.id).toBeDefined();
    expect(res.body.clusterMode).toBe(true);
    // The route dispatches a single cmd:deploy to the head node (node-1).
    expect(sentMessages).toHaveLength(1);
    expect(sentMessages[0].nodeId).toBe("node-1");
    expect((sentMessages[0].message as { type: string }).type).toBe("cmd:deploy");
  });
});
