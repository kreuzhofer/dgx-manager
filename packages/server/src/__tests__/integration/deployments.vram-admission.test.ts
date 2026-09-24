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
  app.set("sshExec", async () => ({ code: 0, stdout: "false", stderr: "" }));
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

/**
 * Wipe all tables in FK-dependency order so the next test starts clean.
 * Children before parents: ClusterNode → Deployment → ...
 */
async function wipeAll() {
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.metricSnapshot.deleteMany({});
  await prisma.fineTuneClusterNode.deleteMany({});
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

describe("POST /api/deployments/:id/restart — VRAM admission self-exclusion (kreuzhofer/dgx-manager#1)", () => {
  it("admits a resident cluster deploy restarting at a higher gpu_memory_utilization (its own VRAM is reclaimable)", async () => {
    await wipeAll();

    // Two GB10 nodes, each measuring ~95 GB used — held by the very
    // deployment we are about to restart (gpt-oss-120b TP=2).
    await Promise.all(
      [1, 2].map((i) =>
        prisma.node.create({
          data: {
            id: `node-${i}`,
            name: `dgx-spark-0${i}`,
            ipAddress: `192.168.44.${36 + i}`,
            vramTotal: 124_546,
            status: "online",
          },
        }),
      ),
    );
    for (const nid of ["node-1", "node-2"]) {
      await prisma.metricSnapshot.create({
        data: { nodeId: nid, vramUsed: 95_000, gpuUtil: 0, timestamp: new Date() },
      });
    }

    const model = await prisma.model.create({
      data: { name: "openai/gpt-oss-120b", runtime: "vllm" },
    });
    const dep = await prisma.deployment.create({
      data: {
        nodeId: "node-1",
        modelId: model.id,
        status: "running",
        port: 8000,
        clusterMode: true,
        config: JSON.stringify({ tensorParallel: 2, port: 8000 }),
        vramActual: 190_000,
      },
    });
    await prisma.clusterNode.createMany({
      data: [
        { deploymentId: dep.id, nodeId: "node-1", role: "head", status: "running" },
        { deploymentId: dep.id, nodeId: "node-2", role: "worker", status: "running" },
      ],
    });

    const { hub, sentMessages } = makeStubHub(RECIPE);
    const app = makeApp(hub);

    // Without the self-reclaim fix this 409s: the node's own resident model is
    // double-counted against the 0.90 reservation.
    const res = await request(app)
      .post(`/api/deployments/${dep.id}/restart`)
      .send({ config: { maxModelLen: 65535, gpuMem: 0.9 } });

    expect(res.status).toBe(200);
    expect(res.body.status).toBe("restarting");
    // The relaunch was dispatched to the head node.
    expect(sentMessages).toHaveLength(1);
    expect(sentMessages[0].nodeId).toBe("node-1");
    expect((sentMessages[0].message as { type: string }).type).toBe("cmd:deploy");
  });

  it("still 409s when the resident VRAM is held by a DIFFERENT deployment (reclaim is capped by the restart's own authorised share)", async () => {
    await wipeAll();

    await prisma.node.create({
      data: {
        id: "node-1",
        name: "dgx-spark-01",
        ipAddress: "192.168.44.36",
        vramTotal: 124_546,
        status: "online",
      },
    });
    await prisma.metricSnapshot.create({
      data: { nodeId: "node-1", vramUsed: 95_000, gpuUtil: 0, timestamp: new Date() },
    });

    // A big OTHER deployment holding ~90 GB on the same node.
    const otherModel = await prisma.model.create({
      data: { name: "big-resident", runtime: "vllm" },
    });
    await prisma.deployment.create({
      data: {
        nodeId: "node-1",
        modelId: otherModel.id,
        status: "running",
        port: 8001,
        config: JSON.stringify({}),
        vramActual: 90_000,
      },
    });

    // The small deployment we restart is a tiny embedding model, authorised for
    // 0.05 of the node — ~6 GB.
    //
    // This test used to bound the reclaim by subtracting the other deployment's
    // recorded memory. That subtraction is gone (#118, ADR 0004 Decision 3): the
    // column it summed holds the whole NODE's reading for vLLM and dgxrun, so
    // co-resident deployments over-counted each other into a spurious refusal.
    // The protection it provided now comes from the authorised share instead —
    // which is why the share has to be a real one here. A deployment authorised
    // for most of the node would be credited most of the node, and admitted;
    // that over-credit is bounded and accepted (ADR 0004 Decision 4), where the
    // old rule's was unbounded.
    const model = await prisma.model.create({
      data: { name: "small-model", runtime: "vllm" },
    });
    const small = await prisma.deployment.create({
      data: {
        nodeId: "node-1",
        modelId: model.id,
        status: "running",
        port: 8000,
        config: JSON.stringify({ port: 8000, gpuMem: 0.05, authorisedGpuMem: 0.05 }),
        vramActual: 4_000,
      },
    });

    const { hub, sentMessages } = makeStubHub(RECIPE);
    const app = makeApp(hub);

    const res = await request(app)
      .post(`/api/deployments/${small.id}/restart`)
      .send({ config: { gpuMem: 0.9 } });

    expect(res.status).toBe(409);
    expect(res.body.shortfalls).toHaveLength(1);
    // Credited 6 GB — its own share — not the 95 GB the node is reading. The
    // other ~90 GB stays counted, so the node is still short, and the conflict
    // list names the OTHER deployment rather than the one being restarted.
    expect(res.body.shortfalls[0].vramUsedMB).toBe(95_000 - Math.round(124_546 * 0.05));
    expect(
      res.body.shortfalls[0].conflicts.map((c: { name: string }) => c.name),
    ).toContain("big-resident");
    expect(sentMessages).toHaveLength(0);
  });
});

/**
 * kreuzhofer/dgx-manager#118 — a restart used to be credited with ALL
 * unattributed memory on its node, so a node busy with a training run read as
 * empty and the restart was admitted however full the node was.
 *
 * The rule now: a restart is credited at most its **authorised share** of the
 * node (`round(vramTotal × its saved gpu_memory_utilization)`). Everything else
 * in the node reading counts against it. See ADR 0004 and CONTEXT.md § Node
 * memory.
 *
 * Every node below is 120 GB so the arithmetic is readable: the safety margin
 * is 5% = 6 GB, and a share of 0.25 is 30 GB.
 */
const NODE_TOTAL_MB = 120_000;
const MARGIN_MB = 6_000;

async function seedNode(id: string, name: string, vramUsedMB: number | null) {
  await prisma.node.create({
    data: { id, name, ipAddress: `192.168.44.${36 + Number(id.slice(-1))}`, vramTotal: NODE_TOTAL_MB, status: "online" },
  });
  if (vramUsedMB !== null) {
    await prisma.metricSnapshot.create({
      data: { nodeId: id, vramUsed: vramUsedMB, gpuUtil: 0, timestamp: new Date() },
    });
  }
}

/** A training job on `nodeId`. Holding columns are the caller's to set. */
async function seedFineTuneJob(
  nodeId: string,
  displayName: string,
  holding: { status?: string; mergeStatus?: string | null; quantizationStatus?: string | null },
) {
  return prisma.fineTuneJob.create({
    data: {
      nodeId,
      displayName,
      baseModel: "Qwen/Qwen3.8-27B",
      method: "lora",
      dataset: "b-mc2/sql-create-context",
      status: holding.status ?? "running",
      mergeStatus: holding.mergeStatus ?? null,
      quantizationStatus: holding.quantizationStatus ?? null,
    },
  });
}

/** A vLLM deployment on `nodeId` carrying `authorisedGpuMem` in its config blob. */
async function seedDeployment(opts: {
  nodeId: string;
  modelName: string;
  status?: string;
  config: Record<string, unknown>;
  vramActual?: number | null;
}) {
  const model = await prisma.model.create({
    data: { name: opts.modelName, runtime: "vllm" },
  });
  return prisma.deployment.create({
    data: {
      nodeId: opts.nodeId,
      modelId: model.id,
      status: opts.status ?? "running",
      port: 8000,
      config: JSON.stringify(opts.config),
      vramActual: opts.vramActual ?? null,
    },
  });
}

type ConflictBody = { id: string; name: string | null; status: string; kind: string };

describe("POST /api/deployments/:id/restart — unattributed memory counts against the restart (#118)", () => {
  it("refuses a restart on a node a running fine-tune job has filled", async () => {
    await wipeAll();
    // spark-01 reads 118 GB used. None of it is a deployment: a LoRA run is
    // training there. The manager has a FineTuneJob row for it but, before
    // this fix, no admission path looked at one.
    await seedNode("node-1", "dgx-spark-01", 118_000);
    await seedFineTuneJob("node-1", "sql-lora-27b", { status: "running" });

    // The deployment being restarted is stopped and holds nothing. It was
    // authorised for 0.25 of the node — 30 GB — so that is the most it can be
    // credited with, leaving 88 GB counted against it: 32 GB free, 36 GB
    // needed. Under the old rule the whole 118 GB was credited to it, the node
    // computed as empty, and this restart was admitted.
    const dep = await seedDeployment({
      nodeId: "node-1",
      modelName: "small-served-model",
      status: "stopped",
      config: { port: 8000, gpuMem: 0.25, authorisedGpuMem: 0.25 },
    });

    const { hub, sentMessages } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub)).post(`/api/deployments/${dep.id}/restart`).send({});

    expect(res.status).toBe(409);
    expect(res.body.shortfalls).toHaveLength(1);
    expect(res.body.shortfalls[0].vramUsedMB).toBe(118_000 - 30_000);
    expect(res.body.shortfalls[0].vramThresholdMB).toBe(30_000 + MARGIN_MB);
    // The refusal names the training run, and says it is one.
    const conflicts = res.body.shortfalls[0].conflicts as ConflictBody[];
    expect(conflicts.map((c) => c.name)).toEqual(["sql-lora-27b"]);
    expect(conflicts[0].kind).toBe("finetune");
    expect(conflicts[0].status).toBe("running");
    expect(res.body.error).toContain("sql-lora-27b");
    expect(res.body.error).toContain("fine-tune");
    // Nothing was launched.
    expect(sentMessages).toHaveLength(0);
  });

  it("counts a merging job as a holder — merging loads the base model", async () => {
    await wipeAll();
    await seedNode("node-1", "dgx-spark-01", 118_000);
    await seedFineTuneJob("node-1", "merge-in-flight", {
      status: "completed",
      mergeStatus: "running",
    });
    const dep = await seedDeployment({
      nodeId: "node-1",
      modelName: "small-served-model",
      status: "stopped",
      config: { port: 8000, gpuMem: 0.25, authorisedGpuMem: 0.25 },
    });

    const { hub } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub)).post(`/api/deployments/${dep.id}/restart`).send({});

    expect(res.status).toBe(409);
    const conflicts = res.body.shortfalls[0].conflicts as ConflictBody[];
    expect(conflicts).toHaveLength(1);
    expect(conflicts[0].name).toBe("merge-in-flight");
    // The rendered status is the activity holding the memory, not the job's
    // `status` column, which reads "completed" here.
    expect(conflicts[0].status).toBe("merging");
  });

  it("counts a quantizing job as a holder", async () => {
    await wipeAll();
    await seedNode("node-1", "dgx-spark-01", 118_000);
    await seedFineTuneJob("node-1", "fp8-quantize", {
      status: "completed",
      mergeStatus: "completed",
      quantizationStatus: "quantizing",
    });
    const dep = await seedDeployment({
      nodeId: "node-1",
      modelName: "small-served-model",
      status: "stopped",
      config: { port: 8000, gpuMem: 0.25, authorisedGpuMem: 0.25 },
    });

    const { hub } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub)).post(`/api/deployments/${dep.id}/restart`).send({});

    expect(res.status).toBe(409);
    const conflicts = res.body.shortfalls[0].conflicts as ConflictBody[];
    expect(conflicts).toHaveLength(1);
    expect(conflicts[0].name).toBe("fp8-quantize");
    expect(conflicts[0].status).toBe("quantizing");
  });

  it("does not name a finished job as a holder", async () => {
    await wipeAll();
    await seedNode("node-1", "dgx-spark-01", 118_000);
    // Nothing in flight: trained, merged, quantized. Whatever is holding the
    // node's 118 GB, it is not this job — so it must not be blamed.
    await seedFineTuneJob("node-1", "all-done", {
      status: "completed",
      mergeStatus: "completed",
      quantizationStatus: "quantized",
    });
    const dep = await seedDeployment({
      nodeId: "node-1",
      modelName: "small-served-model",
      status: "stopped",
      config: { port: 8000, gpuMem: 0.25, authorisedGpuMem: 0.25 },
    });

    const { hub } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub)).post(`/api/deployments/${dep.id}/restart`).send({});

    // Still refused — the memory is still there, unattributed — but with no
    // holder named rather than the wrong one.
    expect(res.status).toBe(409);
    expect(res.body.shortfalls[0].conflicts).toEqual([]);
  });

  it("counts a multi-node training run on the worker node it occupies", async () => {
    await wipeAll();
    await seedNode("node-1", "dgx-spark-01", 118_000);
    await seedNode("node-2", "dgx-spark-02", 0);
    // Head is node-2; node-1 is a worker. A worker node is held by the job
    // just as the head is.
    const job = await seedFineTuneJob("node-2", "tp2-training", { status: "running" });
    await prisma.fineTuneClusterNode.createMany({
      data: [
        { jobId: job.id, nodeId: "node-2", role: "head" },
        { jobId: job.id, nodeId: "node-1", role: "worker" },
      ],
    });
    const dep = await seedDeployment({
      nodeId: "node-1",
      modelName: "small-served-model",
      status: "stopped",
      config: { port: 8000, gpuMem: 0.25, authorisedGpuMem: 0.25 },
    });

    const { hub } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub)).post(`/api/deployments/${dep.id}/restart`).send({});

    expect(res.status).toBe(409);
    const conflicts = res.body.shortfalls[0].conflicts as ConflictBody[];
    expect(conflicts.map((c) => c.name)).toEqual(["tp2-training"]);
    expect(conflicts[0].kind).toBe("finetune");
  });

  it("credits a restart only for its SAVED share — a higher gpuMem override sizes the request, not the reclaim", async () => {
    await wipeAll();
    // The deployment is resident and the node reads 100 GB. It was authorised
    // for 0.5 (60 GB); the caller restarts it at 0.9 (108 GB + 6 GB margin).
    await seedNode("node-1", "dgx-spark-01", 100_000);
    const dep = await seedDeployment({
      nodeId: "node-1",
      modelName: "grown-model",
      config: { port: 8000, gpuMem: 0.5, authorisedGpuMem: 0.5 },
    });

    const { hub, sentMessages } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub))
      .post(`/api/deployments/${dep.id}/restart`)
      .send({ config: { gpuMem: 0.9 } });

    // Reclaim is 60 GB, not 100 GB: 40 GB still counted, 80 GB free, 114 GB
    // needed. Had the override sized the reclaim too, the whole 100 GB would
    // have been credited and the node would have read empty.
    expect(res.status).toBe(409);
    expect(res.body.shortfalls[0].vramUsedMB).toBe(100_000 - 60_000);
    expect(res.body.shortfalls[0].vramThresholdMB).toBe(108_000 + MARGIN_MB);
    expect(res.body.gpuMemoryUtilization).toBe(0.9);
    // The refusal points at the lever that gets a disagreeing user through it.
    expect(res.body.error).toContain("gpuMem");
    expect(sentMessages).toHaveLength(0);
  });

  it("does not spuriously refuse a restart when two deployments on one node both record the node-wide reading", async () => {
    await wipeAll();
    // vLLM and dgxrun deployments record the whole node's reading in
    // `vramActual`, so two of them on one node each claim all 95 GB. The
    // subtraction that used to bound the reclaim summed those, clamped the
    // reclaim to zero, and refused a restart for memory that WAS its own —
    // #1's symptom from the opposite direction.
    await seedNode("node-1", "dgx-spark-01", 95_000);
    const a = await seedDeployment({
      nodeId: "node-1",
      modelName: "co-resident-a",
      config: { port: 8001, gpuMem: 0.8, authorisedGpuMem: 0.8 },
      vramActual: 95_000,
    });
    const b = await seedDeployment({
      nodeId: "node-1",
      modelName: "co-resident-b",
      config: { port: 8000, gpuMem: 0.8, authorisedGpuMem: 0.8 },
      vramActual: 95_000,
    });

    // Either of them, not just one: the defect was symmetric.
    for (const dep of [a, b]) {
      const { hub, sentMessages } = makeStubHub(RECIPE);
      const res = await request(makeApp(hub)).post(`/api/deployments/${dep.id}/restart`).send({});
      expect(res.status).toBe(200);
      expect(res.body.status).toBe("restarting");
      expect(sentMessages).toHaveLength(1);
    }
  });

  it("leaves a fresh deploy's arithmetic alone, but lets it name the training run holding the node", async () => {
    await wipeAll();
    // A fresh deploy reclaims nothing, so the node reading is used as-is and
    // the over-commit is refused exactly as before. What changes is that the
    // 409 can now say what holds the memory instead of arriving with an empty
    // conflict list (#101).
    await seedNode("node-1", "dgx-spark-01", 60_000);
    await seedFineTuneJob("node-1", "sql-lora-27b", { status: "running" });

    const { hub, sentMessages } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub))
      .post("/api/deployments")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file });

    expect(res.status).toBe(409);
    expect(res.body.shortfalls[0].vramUsedMB).toBe(60_000);
    const conflicts = res.body.shortfalls[0].conflicts as ConflictBody[];
    expect(conflicts.map((c) => c.name)).toEqual(["sql-lora-27b"]);
    expect(conflicts[0].kind).toBe("finetune");
    expect(sentMessages).toHaveLength(0);
  });
});

/**
 * #122 made `gpuMem` the lever a refusal points at, and the number admission
 * checks against. An unreadable one used to reach the arithmetic and produce a
 * 409 whose every figure serialised as `null` — a refusal nobody can act on.
 * Validated at the boundary now, the way the route already validates
 * `artifactVariant`.
 */
describe("POST /api/deployments/:id/restart — the gpuMem override is validated at the boundary", () => {
  it.each([
    ["a non-numeric string", "abc"],
    ["a numeric string", "0.9"],
    ["more than a whole node", 1.5],
    ["zero", 0],
    ["a negative share", -0.5],
    ["null", null],
  ])("refuses %s with a 400 rather than admitting on a nonsense number", async (_why, gpuMem) => {
    await wipeAll();
    await seedNode("node-1", "dgx-spark-01", 0);
    const dep = await seedDeployment({
      nodeId: "node-1",
      modelName: "some-model",
      config: { port: 8000, gpuMem: 0.5, authorisedGpuMem: 0.5 },
    });

    const { hub, sentMessages } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub))
      .post(`/api/deployments/${dep.id}/restart`)
      .send({ config: { gpuMem } });

    expect(res.status).toBe(400);
    expect(res.body.error).toContain("gpuMem");
    expect(sentMessages).toHaveLength(0);
    // And the row's authorised share is untouched by a rejected request.
    const row = await prisma.deployment.findUniqueOrThrow({ where: { id: dep.id } });
    expect(JSON.parse(row.config!).authorisedGpuMem).toBe(0.5);
  });

  // A share that is readable passes the boundary and is then judged on the merits.
  // 1 is a legitimate share and still cannot be admitted — a whole node plus the
  // safety margin exceeds the node — which is a 409, not a 400.
  it.each([
    [0.5, 200],
    [0.88, 200],
    [1, 409],
  ])("accepts %s and lets admission decide it (%i)", async (gpuMem, expected) => {
    await wipeAll();
    await seedNode("node-1", "dgx-spark-01", 0);
    const dep = await seedDeployment({
      nodeId: "node-1",
      modelName: "some-model",
      config: { port: 8000, gpuMem: 0.5, authorisedGpuMem: 0.5 },
    });

    const { hub } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub))
      .post(`/api/deployments/${dep.id}/restart`)
      .send({ config: { gpuMem } });

    expect(res.status).toBe(expected);
  });
});

describe("POST /api/deployments — the authorised share is persisted, not implicit (#118)", () => {
  it("records the share admission resolved for the deployment", async () => {
    await wipeAll();
    await seedNode("node-1", "dgx-spark-01", 0);

    const { hub } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub))
      .post("/api/deployments")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file });

    expect(res.status).toBe(201);
    const row = await prisma.deployment.findUniqueOrThrow({ where: { id: res.body.id } });
    // The recipe default, resolved at admission time — the deployment's claim
    // is now explicit on the row instead of being re-derived from a catalog
    // that refreshes on its own schedule.
    expect(JSON.parse(row.config!).authorisedGpuMem).toBe(0.85);
  });

  it("records the caller's override when one sizes the request", async () => {
    await wipeAll();
    await seedNode("node-1", "dgx-spark-01", 0);

    const { hub } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub))
      .post("/api/deployments")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, config: { gpuMem: 0.5 } });

    expect(res.status).toBe(201);
    const row = await prisma.deployment.findUniqueOrThrow({ where: { id: res.body.id } });
    expect(JSON.parse(row.config!).authorisedGpuMem).toBe(0.5);
  });

  it("ignores an authorisedGpuMem supplied in the request body", async () => {
    await wipeAll();
    await seedNode("node-1", "dgx-spark-01", 0);

    const { hub } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub))
      .post("/api/deployments")
      .send({
        nodeId: "node-1",
        recipeFile: RECIPE.file,
        // A caller who could set this would be writing their own reclaim
        // allowance — the admission check must be the only author of it.
        config: { gpuMem: 0.5, authorisedGpuMem: 0.99 },
      });

    expect(res.status).toBe(201);
    const row = await prisma.deployment.findUniqueOrThrow({ where: { id: res.body.id } });
    expect(JSON.parse(row.config!).authorisedGpuMem).toBe(0.5);
  });

  it("ignores an authorisedGpuMem supplied to a restart", async () => {
    await wipeAll();
    await seedNode("node-1", "dgx-spark-01", 20_000);
    const dep = await seedDeployment({
      nodeId: "node-1",
      modelName: "grown-model",
      config: { port: 8000, gpuMem: 0.5, authorisedGpuMem: 0.5 },
    });

    const { hub } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub))
      .post(`/api/deployments/${dep.id}/restart`)
      // A caller who could raise this would widen the reclaim every subsequent
      // restart is granted — the check's own bound, set by the caller.
      .send({ config: { authorisedGpuMem: 0.99 } });

    expect(res.status).toBe(200);
    const row = await prisma.deployment.findUniqueOrThrow({ where: { id: dep.id } });
    expect(JSON.parse(row.config!).authorisedGpuMem).toBe(0.5);
  });

  it("carries the share forward when a restart is admitted at a new one", async () => {
    await wipeAll();
    await seedNode("node-1", "dgx-spark-01", 20_000);
    const dep = await seedDeployment({
      nodeId: "node-1",
      modelName: "grown-model",
      config: { port: 8000, gpuMem: 0.5, authorisedGpuMem: 0.5 },
    });

    const { hub } = makeStubHub(RECIPE);
    const res = await request(makeApp(hub))
      .post(`/api/deployments/${dep.id}/restart`)
      .send({ config: { gpuMem: 0.7 } });

    expect(res.status).toBe(200);
    const row = await prisma.deployment.findUniqueOrThrow({ where: { id: dep.id } });
    // Admitted at 0.7, so 0.7 is what it is authorised for from now on —
    // otherwise the NEXT restart would be under-credited and refused for
    // memory that is its own.
    expect(JSON.parse(row.config!).authorisedGpuMem).toBe(0.7);
  });
});
