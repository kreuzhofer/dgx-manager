/**
 * Integration test for FineTuneClusterNode persistence on the
 * POST/GET/DELETE finetune routes.
 *
 * Background: before this feature, multi-node finetune launches lost
 * the worker-node list after the start command fired (only the head
 * was kept on the FineTuneJob row). This test pins the contract that
 * (a) POST creates one FineTuneClusterNode row per participating node
 * with role="head" / "worker", (b) GET returns them embedded, and
 * (c) DELETE cascades them away.
 *
 * Follows the same per-suite SQLite + supertest + stub-hub pattern as
 * deployments.vram-admission.test.ts.
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
  // PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION: same standing consent
  // as deployments.vram-admission.test.ts — DATABASE_URL is a freshly
  // mkdtemp'd SQLite file in /tmp, so --force-reset cannot touch any
  // real database.
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
  file: "recipes/test-qwen3.6-attn-mlp",
  name: "Test Recipe",
  base_model: "Qwen/Qwen3.6-27B",
  method: "lora",
  defaults: {},
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
  await prisma.trainingMetric.deleteMany({});
  await prisma.fineTuneClusterNode.deleteMany({});
  await prisma.fineTuneJob.deleteMany({});
  await prisma.metricSnapshot.deleteMany({});
  await prisma.node.deleteMany({});
}

async function seedNodes() {
  return Promise.all(
    [1, 2, 3, 4].map((i) =>
      prisma.node.create({
        data: {
          id: `node-${i}`,
          name: `dgx-spark-0${i}`,
          ipAddress: `192.168.44.${35 + i}`,
          status: "online",
        },
      }),
    ),
  );
}

describe("finetune cluster persistence", () => {
  it("POST single-node leaves clusterNodes empty", async () => {
    await wipeAll();
    await seedNodes();
    const { hub, sentMessages } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/finetune")
      .send({
        nodeId: "node-2",
        recipeFile: RECIPE.file,
        dataset: "/tmp/fake.jsonl",
        config: { max_steps: 5 },
      });

    expect(res.status).toBe(201);
    expect(res.body.nodeId).toBe("node-2");
    // Single-node: no cluster rows persisted; head is implicit in nodeId.
    expect(res.body.clusterNodes).toEqual([]);
    // Sanity: the start command fired against the head.
    expect(sentMessages).toHaveLength(1);
    expect(sentMessages[0].nodeId).toBe("node-2");

    const rows = await prisma.fineTuneClusterNode.findMany({
      where: { jobId: res.body.id },
    });
    expect(rows).toHaveLength(0);
  });

  it("POST multi-node persists head + workers with correct roles", async () => {
    await wipeAll();
    await seedNodes();
    const { hub, sentMessages } = makeStubHub();
    const app = makeApp(hub);

    const nodeIds = ["node-2", "node-3", "node-4", "node-1"];
    const res = await request(app)
      .post("/api/finetune")
      .send({
        nodeIds,
        recipeFile: RECIPE.file,
        dataset: "/tmp/fake.jsonl",
        config: { max_steps: 5 },
      });

    expect(res.status).toBe(201);
    // Head is nodeIds[0]
    expect(res.body.nodeId).toBe("node-2");
    // Cluster includes all 4 with one head and three workers.
    expect(res.body.clusterNodes).toHaveLength(4);

    const rolesByNode: Record<string, string> = {};
    for (const cn of res.body.clusterNodes) {
      rolesByNode[cn.nodeId] = cn.role;
      // Embedded node is included so the dashboard can render names.
      expect(cn.node).toBeDefined();
      expect(cn.node.name).toMatch(/^dgx-spark-0\d$/);
    }
    expect(rolesByNode["node-2"]).toBe("head");
    expect(rolesByNode["node-3"]).toBe("worker");
    expect(rolesByNode["node-4"]).toBe("worker");
    expect(rolesByNode["node-1"]).toBe("worker");

    // Start command went to head only (the head orchestrates torchrun).
    expect(sentMessages).toHaveLength(1);
    expect(sentMessages[0].nodeId).toBe("node-2");

    // DB confirms persistence (not just an in-memory artifact of the response).
    const rows = await prisma.fineTuneClusterNode.findMany({
      where: { jobId: res.body.id },
      orderBy: { role: "asc" },
    });
    expect(rows).toHaveLength(4);
    expect(rows.filter((r) => r.role === "head")).toHaveLength(1);
    expect(rows.filter((r) => r.role === "worker")).toHaveLength(3);
  });

  it("GET /:id and GET / both surface clusterNodes with embedded node info", async () => {
    await wipeAll();
    await seedNodes();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const created = await request(app)
      .post("/api/finetune")
      .send({
        nodeIds: ["node-3", "node-4"],
        recipeFile: RECIPE.file,
        dataset: "/tmp/fake.jsonl",
      });
    expect(created.status).toBe(201);

    // GET /:id
    const byId = await request(app).get(`/api/finetune/${created.body.id}`);
    expect(byId.status).toBe(200);
    expect(byId.body.clusterNodes).toHaveLength(2);
    expect(byId.body.clusterNodes.map((cn: { node: { name: string } }) => cn.node.name).sort())
      .toEqual(["dgx-spark-03", "dgx-spark-04"]);

    // GET / (list)
    const list = await request(app).get("/api/finetune");
    expect(list.status).toBe(200);
    expect(list.body).toHaveLength(1);
    expect(list.body[0].clusterNodes).toHaveLength(2);
  });

  it("DELETE /:id cascades and removes FineTuneClusterNode rows", async () => {
    await wipeAll();
    await seedNodes();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const created = await request(app)
      .post("/api/finetune")
      .send({
        nodeIds: ["node-2", "node-3", "node-4"],
        recipeFile: RECIPE.file,
        dataset: "/tmp/fake.jsonl",
      });
    expect(created.status).toBe(201);
    const jobId = created.body.id;

    // Sanity: rows exist.
    expect(
      await prisma.fineTuneClusterNode.count({ where: { jobId } }),
    ).toBe(3);

    const del = await request(app).delete(`/api/finetune/${jobId}`);
    expect(del.status).toBe(200);
    expect(del.body.deleted).toBe(true);

    // Cascade removed both the job and its cluster rows.
    expect(await prisma.fineTuneJob.count({ where: { id: jobId } })).toBe(0);
    expect(await prisma.fineTuneClusterNode.count({ where: { jobId } })).toBe(0);
  });

  it("(jobId, nodeId) is unique — same node can't be persisted twice in one job", async () => {
    await wipeAll();
    await seedNodes();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    // Real launch path goes through nodeIds without dupes; this test just
    // confirms the schema-level guard so a future bug can't double-record.
    const created = await request(app)
      .post("/api/finetune")
      .send({
        nodeIds: ["node-2", "node-3"],
        recipeFile: RECIPE.file,
        dataset: "/tmp/fake.jsonl",
      });
    expect(created.status).toBe(201);

    await expect(
      prisma.fineTuneClusterNode.create({
        data: { jobId: created.body.id, nodeId: "node-2", role: "worker" },
      }),
    ).rejects.toThrow();
  });
});
