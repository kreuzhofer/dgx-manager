/**
 * Integration tests for DELETE /api/nodes/:id — the offboarding/deletion flow.
 *
 * Covers:
 *  (1) The FK-cleanup fix: a node that participated in a multi-node fine-tune
 *      (FineTuneClusterNode rows referencing its nodeId) and has a deployment
 *      must delete without a Prisma foreign-key violation. This used to 500.
 *  (2) The graceful-with-timeout path: when the agent stays "online" past the
 *      deadline, the node is NOT deleted and the response is { timedOut: true };
 *      a subsequent force delete removes it.
 *  (3) The agent-offline fast path and the graceful offboard (agent goes offline).
 *
 * Follows the per-suite SQLite + supertest + stub-hub pattern. The offboard
 * deadline is injected small via app.set("offboardDeadlineMs") so tests don't
 * actually wait 30s.
 */
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "nodes-offboard-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let nodesRouter: typeof import("../../routes/nodes.js").nodesRouter;

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
  ({ nodesRouter } = await import("../../routes/nodes.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

// Wipe in FK-dependency order between tests.
afterEach(async () => {
  await prisma.fineTuneClusterNode.deleteMany();
  await prisma.trainingMetric.deleteMany();
  await prisma.loadBalancerEndpoint.deleteMany();
  await prisma.clusterNode.deleteMany();
  await prisma.metricSnapshot.deleteMany();
  await prisma.deployment.deleteMany();
  await prisma.model.deleteMany();
  await prisma.fineTuneJob.deleteMany();
  await prisma.node.deleteMany();
});

function makeApp(opts: {
  isAgentOnline?: (id: string) => boolean;
  sendToAgent?: any;
  deadlineMs?: number;
}) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", {
    isAgentOnline: opts.isAgentOnline ?? (() => false),
    sendToAgent: opts.sendToAgent ?? vi.fn(),
  });
  if (opts.deadlineMs != null) app.set("offboardDeadlineMs", opts.deadlineMs);
  app.use("/api/nodes", nodesRouter);
  return app;
}

/**
 * Build a node that is a WORKER in a multi-node fine-tune headed by ANOTHER
 * node, plus a deployment of its own with a cluster membership + LB endpoint.
 * This is the exact shape that produced the FK violation on node.delete().
 */
async function seedNodeWithForeignKeys() {
  const head = await prisma.node.create({ data: { name: "head-node" } });
  const worker = await prisma.node.create({ data: { name: "worker-node" } });

  // A fine-tune job owned by the head node, with the worker as a participant.
  const job = await prisma.fineTuneJob.create({
    data: {
      nodeId: head.id,
      baseModel: "Qwen/Qwen3-8B",
      method: "lora",
      dataset: "ds",
      status: "running",
    },
  });
  await prisma.fineTuneClusterNode.create({
    data: { jobId: job.id, nodeId: worker.id, role: "worker" },
  });

  // The worker also has its own deployment with a cluster node + LB endpoint.
  const model = await prisma.model.create({
    data: { name: "m1", runtime: "vllm" },
  });
  const deployment = await prisma.deployment.create({
    data: { nodeId: worker.id, modelId: model.id, status: "running" },
  });
  await prisma.clusterNode.create({
    data: { deploymentId: deployment.id, nodeId: worker.id, role: "head" },
  });
  const rule = await prisma.loadBalancerRule.create({
    data: { name: "r1", modelName: "m1" },
  });
  await prisma.loadBalancerEndpoint.create({
    data: { ruleId: rule.id, deploymentId: deployment.id },
  });
  await prisma.metricSnapshot.create({
    data: { nodeId: worker.id, gpuUtil: 10, vramUsed: 100 },
  });

  return { head, worker, job, deployment };
}

describe("DELETE /api/nodes/:id — FK cleanup", () => {
  it("force=true deletes a node that is a fine-tune worker + has a deployment (no FK 500)", async () => {
    const { worker } = await seedNodeWithForeignKeys();
    const app = makeApp({});

    const res = await request(app).delete(`/api/nodes/${worker.id}?force=true`);

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ deleted: true, forced: true });
    expect(await prisma.node.findUnique({ where: { id: worker.id } })).toBeNull();
    // The worker's FineTuneClusterNode membership (in the head's job) is gone.
    expect(await prisma.fineTuneClusterNode.count({ where: { nodeId: worker.id } })).toBe(0);
    // The parent job (on the head node) is untouched.
    expect(await prisma.fineTuneJob.count()).toBe(1);
    expect(await prisma.deployment.count({ where: { nodeId: worker.id } })).toBe(0);
    expect(await prisma.loadBalancerEndpoint.count()).toBe(0);
  });

  it("force via body { force: true } also deletes", async () => {
    const node = await prisma.node.create({ data: { name: "solo" } });
    const app = makeApp({});
    const res = await request(app)
      .delete(`/api/nodes/${node.id}`)
      .send({ force: true });
    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ deleted: true, forced: true });
    expect(await prisma.node.findUnique({ where: { id: node.id } })).toBeNull();
  });
});

describe("DELETE /api/nodes/:id — graceful offboarding", () => {
  it("agent offline: deletes immediately with reason=agent-offline", async () => {
    const node = await prisma.node.create({ data: { name: "offline-node" } });
    const sendToAgent = vi.fn();
    const app = makeApp({ isAgentOnline: () => false, sendToAgent });

    const res = await request(app).delete(`/api/nodes/${node.id}`);

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ deleted: true, offboarded: false, reason: "agent-offline" });
    expect(await prisma.node.findUnique({ where: { id: node.id } })).toBeNull();
    // Offline agent is never messaged.
    expect(sendToAgent).not.toHaveBeenCalled();
  });

  it("agent stays online past deadline: returns timedOut and does NOT delete; then force removes it", async () => {
    const node = await prisma.node.create({ data: { name: "stuck-node" } });
    const sendToAgent = vi.fn();
    // Always online → never goes offline → must time out.
    const app = makeApp({ isAgentOnline: () => true, sendToAgent, deadlineMs: 60 });

    const res = await request(app).delete(`/api/nodes/${node.id}`);

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ deleted: false, offboarded: false, timedOut: true });
    // Node still present.
    expect(await prisma.node.findUnique({ where: { id: node.id } })).not.toBeNull();
    // Agent was asked to deprovision.
    expect(sendToAgent).toHaveBeenCalledWith(node.id, { type: "cmd:deprovision", payload: {} });

    // Now force it.
    const forced = await request(app).delete(`/api/nodes/${node.id}?force=true`);
    expect(forced.status).toBe(200);
    expect(forced.body).toMatchObject({ deleted: true, forced: true });
    expect(await prisma.node.findUnique({ where: { id: node.id } })).toBeNull();
  });

  it("agent goes offline within deadline: offboards and deletes", async () => {
    const node = await prisma.node.create({ data: { name: "graceful-node" } });
    // Online for the first couple of checks, then offline (agent tore itself down).
    let calls = 0;
    const isAgentOnline = () => {
      calls += 1;
      return calls <= 2;
    };
    const sendToAgent = vi.fn();
    const app = makeApp({ isAgentOnline, sendToAgent, deadlineMs: 2000 });

    const res = await request(app).delete(`/api/nodes/${node.id}`);

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ deleted: true, offboarded: true });
    expect(await prisma.node.findUnique({ where: { id: node.id } })).toBeNull();
    expect(sendToAgent).toHaveBeenCalledWith(node.id, { type: "cmd:deprovision", payload: {} });
  });

  it("a throwing sendToAgent (dead socket) does not block the offboard", async () => {
    const node = await prisma.node.create({ data: { name: "dead-socket-node" } });
    const sendToAgent = vi.fn(() => {
      throw new Error("socket is not open");
    });
    // Report online once, then offline so the wait resolves quickly.
    let calls = 0;
    const isAgentOnline = () => {
      calls += 1;
      return calls <= 1;
    };
    const app = makeApp({ isAgentOnline, sendToAgent, deadlineMs: 2000 });

    const res = await request(app).delete(`/api/nodes/${node.id}`);

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ deleted: true, offboarded: true });
    expect(await prisma.node.findUnique({ where: { id: node.id } })).toBeNull();
  });
});
