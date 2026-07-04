/**
 * Integration tests for the dgxrun runner path (manager dispatch, phase 2).
 *
 * Covers:
 *   - a `runner: dgxrun` cluster deploy fans one cmd:deploy to EVERY node with
 *     correct per-rank payloads (head=rank0, masterAddr=head mgmt IP, headless
 *     for workers, full resolved recipe attached).
 *   - a rank failure triggers coordinated teardown: cmd:undeploy to every node.
 *
 * Mirrors deployments.sparkrun.test.ts (supertest + stub agentHub).
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-dgxrun-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;
let coordinatedDgxrunTeardown: typeof import("../../deployments/dgxrun-teardown.js").coordinatedDgxrunTeardown;

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
  ({ coordinatedDgxrunTeardown } = await import("../../deployments/dgxrun-teardown.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

async function wipeAll() {
  await prisma.loadBalancerEndpoint.deleteMany({});
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.metricSnapshot.deleteMany({});
  await prisma.fineTuneJob.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
}

type SentMessage = { nodeId: string; message: { type: string; payload: Record<string, unknown> } };

function makeStubHub() {
  const sent: SentMessage[] = [];
  const hub = {
    getRecipes: () => [],
    getTrainingRecipes: () => [],
    getOllamaModels: () => [],
    isAgentOnline: (_id: string) => true,
    onlineNodeIds: () => [] as string[],
    sendToAgent: (nodeId: string, message: unknown) => sent.push({ nodeId, message: message as SentMessage["message"] }),
  };
  return { hub, sent };
}

function makeApp(hub: unknown) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.set("sshExec", async () => ({ code: 0, stdout: "false", stderr: "" }));
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

const DGXRUN_YAML = [
  "runner: dgxrun",
  "model: org/glm-5.2",
  "container: my-vllm-image:probe",
  "cluster_only: true",
  "defaults:",
  "  port: 8000",
  "  host: 0.0.0.0",
  "  tensor_parallel: 4",
  "  gpu_memory_utilization: 0.88",
  "  served_model_name: glm-5.2",
  "env:",
  "  NCCL_NET: IB",
  "command: |",
  "  vllm serve {model} --served-model-name {served_model_name} --host {host} --port {port} -tp {tensor_parallel} --distributed-executor-backend mp",
].join("\n");

async function seedCluster(n: number) {
  const ids: string[] = [];
  for (let i = 0; i < n; i++) {
    const id = `dgx-node-${i}`;
    await prisma.node.create({
      data: {
        id, name: `dgx-spark-${i}`,
        ipAddress: `192.168.44.${36 + i}`,
        vramTotal: 122_502, status: "online",
      },
    });
    ids.push(id);
  }
  return ids;
}

describe("POST /api/deployments — runner:dgxrun fan-out", () => {
  it("fans one cmd:deploy per node with correct per-rank payloads", async () => {
    await wipeAll();
    const ids = await seedCluster(4);

    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeIds: ids, recipeYaml: DGXRUN_YAML });

    expect(res.status).toBe(201);

    // One deploy per node.
    expect(sent).toHaveLength(4);
    expect(sent.map((s) => s.nodeId)).toEqual(ids);

    // Every message is a dgxrun cmd:deploy with the full resolved recipe.
    for (const s of sent) {
      expect(s.message.type).toBe("cmd:deploy");
      expect(s.message.payload.kind).toBe("dgxrun");
      const recipe = s.message.payload.recipe as Record<string, unknown>;
      expect(recipe.container).toBe("my-vllm-image:probe");
      expect(recipe.command).toContain("vllm serve {model}");
      expect((recipe.env as Record<string, unknown>).NCCL_NET).toBe("IB");
      expect(s.message.payload.nnodes).toBe(4);
      // masterAddr = head node's mgmt IP on EVERY rank.
      expect(s.message.payload.masterAddr).toBe("192.168.44.36");
    }

    // Ranks assigned head-first; headless only for workers.
    expect(sent.map((s) => s.message.payload.rank)).toEqual([0, 1, 2, 3]);
    expect(sent.map((s) => s.message.payload.headless)).toEqual([false, true, true, true]);

    // Deployment persisted as a dgxrun cluster with cluster-node records.
    const created = await prisma.deployment.findUnique({
      where: { id: res.body.id }, include: { clusterNodes: true },
    });
    expect(created?.clusterMode).toBe(true);
    expect(created?.clusterNodes).toHaveLength(4);
    expect(JSON.parse(created!.config!).runner).toBe("dgxrun");
  });

  it("rejects a dgxrun recipe missing container with 400 and no sends", async () => {
    await wipeAll();
    const ids = await seedCluster(2);
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeIds: ids, recipeYaml: "runner: dgxrun\ncommand: vllm serve x\n" });

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/container/i);
    expect(sent).toHaveLength(0);
  });
});

describe("dgxrun coordinated teardown", () => {
  it("a rank failure tears down every rank (cmd:undeploy to all nodes)", async () => {
    await wipeAll();
    const ids = await seedCluster(4);

    // Deploy first so the deployment + cluster-node records exist.
    const { hub: deployHub } = makeStubHub();
    const app = makeApp(deployHub);
    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeIds: ids, recipeYaml: DGXRUN_YAML });
    expect(res.status).toBe(201);
    const deploymentId = res.body.id as string;

    // Now simulate a rank reporting failure → coordinated teardown fans
    // cmd:undeploy to EVERY cluster node (mp has no partial recovery).
    const { hub: teardownHub, sent } = makeStubHub();
    const targets = await coordinatedDgxrunTeardown(teardownHub, deploymentId);

    expect(targets.sort()).toEqual([...ids].sort());
    expect(sent).toHaveLength(4);
    for (const s of sent) {
      expect(s.message.type).toBe("cmd:undeploy");
      expect(s.message.payload.deploymentId).toBe(deploymentId);
      expect(s.message.payload.kind).toBe("dgxrun");
    }
  });

  it("is a no-op for a non-dgxrun deployment", async () => {
    await wipeAll();
    await prisma.node.create({
      data: { id: "solo", name: "solo", ipAddress: "192.168.44.50", vramTotal: 122_502, status: "online" },
    });
    const model = await prisma.model.create({ data: { name: "m", runtime: "vllm" } });
    const dep = await prisma.deployment.create({
      data: { nodeId: "solo", modelId: model.id, status: "failed", config: JSON.stringify({ recipeFile: "x" }) },
    });
    const { hub, sent } = makeStubHub();
    const targets = await coordinatedDgxrunTeardown(hub, dep.id);
    expect(targets).toEqual([]);
    expect(sent).toHaveLength(0);
  });
});
