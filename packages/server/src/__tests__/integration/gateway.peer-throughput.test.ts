/**
 * Integration test for the peer-throughput verdict on GET /api/gateway (#88).
 *
 * The pure comparison is property-tested in health/peer-throughput.test.ts.
 * What this suite covers is the half that needs a DB: loading each member's
 * window out of MetricSnapshot, counting the deployments on its node, and
 * shaping the verdict onto the pool view.
 *
 * Same harness as the other route suites: only the router under test mounted,
 * a per-test SQLite, and a stub agentHub supplying the liveness signal.
 */
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import request from "supertest";
import express from "express";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let gatewayViewRouter: typeof import("../../routes/gateway.js").gatewayViewRouter;
let resetOutstanding: typeof import("../../gateway/inflight.js").resetOutstanding;

beforeAll(async () => {
  // PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION: see deployments.vram-admission.test.ts.
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
  ({ gatewayViewRouter } = await import("../../routes/gateway.js"));
  ({ resetOutstanding } = await import("../../gateway/inflight.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

afterEach(async () => {
  await prisma.metricSnapshot.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
  resetOutstanding();
});

function makeApp() {
  const app = express();
  app.set("agentHub", { isAgentOnline: () => true });
  app.use("/api/gateway", gatewayViewRouter);
  return app;
}

let seq = 0;

/** A catalog model shared by every member of a pool, unless a case wants drift. */
async function seedModel(name?: string) {
  return prisma.model.create({ data: { name: name ?? `catalog-${++seq}`, runtime: "vllm" } });
}

/**
 * One pool member: a node, a running deployment publishing `publishedName`,
 * and `sampleCount` metric snapshots inside the window holding `rate` tok/s.
 */
async function seedMember(opts: {
  nodeName: string;
  publishedName: string;
  modelId: string;
  rate: number | null;
  sampleCount?: number;
  ageMinutes?: number;
}) {
  const n = ++seq;
  const node = await prisma.node.create({
    data: { name: opts.nodeName, ipAddress: `10.0.0.${n}`, status: "online" },
  });
  const deployment = await prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: opts.modelId,
      status: "running",
      port: 8000,
      publishedName: opts.publishedName,
    },
  });
  const count = opts.sampleCount ?? 120;
  const ageMs = (opts.ageMinutes ?? 0) * 60_000;
  await prisma.metricSnapshot.createMany({
    data: Array.from({ length: count }, (_, i) => ({
      nodeId: node.id,
      gpuUtil: 95,
      vramUsed: 100_000,
      tps: opts.rate,
      timestamp: new Date(Date.now() - ageMs - i * 5_000),
    })),
  });
  return { node, deployment };
}

/**
 * The throughput verdict for one member, out of the pool view. Keyed by
 * deployment id, not node: a node can host members of two different pools.
 */
async function throughputFor(deploymentId: string) {
  const res = await request(makeApp()).get("/api/gateway");
  expect(res.status).toBe(200);
  const members = res.body.pools.flatMap((p: { members: unknown[] }) => p.members);
  const member = members.find((m: { deploymentId: string }) => m.deploymentId === deploymentId) as {
    node: string;
    throughput: {
      state: string;
      rate: number | null;
      ratio: number | null;
      peerRates: { node: string; rate: number }[];
      windowMinutes: number;
      reason?: string;
    };
  };
  expect(member, `no member for deployment ${deploymentId}`).toBeDefined();
  return member;
}

describe("GET /api/gateway — peer throughput", () => {
  it("flags the pool member sustaining half its peers' rate", async () => {
    const model = await seedModel();
    const slowMember = await seedMember({ nodeName: "spark-02", publishedName: "pool-a", modelId: model.id, rate: 23 });
    await seedMember({ nodeName: "spark-03", publishedName: "pool-a", modelId: model.id, rate: 55 });
    await seedMember({ nodeName: "spark-04", publishedName: "pool-a", modelId: model.id, rate: 56 });

    const slow = await throughputFor(slowMember.deployment.id);

    expect(slow.throughput.state).toBe("suspect");
    expect(slow.throughput.rate).toBeCloseTo(23, 5);
    expect(slow.throughput.ratio!).toBeLessThan(0.8);
    expect(slow.throughput.peerRates.map((p) => p.node).sort()).toEqual(["spark-03", "spark-04"]);
    expect(slow.throughput.windowMinutes).toBe(60);
  });

  it("leaves healthy peers alone", async () => {
    const model = await seedModel();
    const a = await seedMember({ nodeName: "spark-03", publishedName: "pool-a", modelId: model.id, rate: 48.8 });
    const b = await seedMember({ nodeName: "spark-04", publishedName: "pool-a", modelId: model.id, rate: 56.5 });

    expect((await throughputFor(a.deployment.id)).throughput.state).toBe("ok");
    expect((await throughputFor(b.deployment.id)).throughput.state).toBe("ok");
  });

  it("reports a pool of one as not-comparable", async () => {
    const model = await seedModel();
    const solo = await seedMember({ nodeName: "spark-01", publishedName: "solo", modelId: model.id, rate: 30 });

    const only = await throughputFor(solo.deployment.id);

    expect(only.throughput.state).toBe("not-comparable");
    expect(only.throughput.reason).toBe("single-member");
  });

  it("reports a member that served nothing as idle rather than degraded", async () => {
    const model = await seedModel();
    const quiet = await seedMember({ nodeName: "spark-03", publishedName: "pool-a", modelId: model.id, rate: null });
    await seedMember({ nodeName: "spark-04", publishedName: "pool-a", modelId: model.id, rate: 56 });

    const idle = await throughputFor(quiet.deployment.id);

    expect(idle.throughput.state).toBe("not-comparable");
    expect(idle.throughput.reason).toBe("idle");
  });

  it("refuses to compare members of a pool serving different models", async () => {
    const a = await seedModel("glm-5.3-flash");
    const b = await seedModel("qwen3.8-27b");
    const one = await seedMember({ nodeName: "spark-03", publishedName: "pool-a", modelId: a.id, rate: 23 });
    const two = await seedMember({ nodeName: "spark-04", publishedName: "pool-a", modelId: b.id, rate: 56 });

    expect((await throughputFor(one.deployment.id)).throughput.reason).toBe("model-mismatch");
    expect((await throughputFor(two.deployment.id)).throughput.reason).toBe("model-mismatch");
  });

  // The node's rate is a sum over everything running there, so a second
  // deployment makes it unattributable — even though this member looks slow.
  it("refuses to compare a member whose node runs a second deployment", async () => {
    const model = await seedModel();
    const shared = await seedMember({
      nodeName: "spark-02", publishedName: "pool-a", modelId: model.id, rate: 23,
    });
    await prisma.deployment.create({
      data: { nodeId: shared.node.id, modelId: model.id, status: "running", port: 8001, publishedName: "other-pool" },
    });
    await seedMember({ nodeName: "spark-03", publishedName: "pool-a", modelId: model.id, rate: 55 });
    await seedMember({ nodeName: "spark-04", publishedName: "pool-a", modelId: model.id, rate: 56 });

    const verdict = await throughputFor(shared.deployment.id);

    expect(verdict.throughput.state).toBe("not-comparable");
    expect(verdict.throughput.reason).toBe("multi-deployment-node");
  });

  it("ignores samples from outside the window", async () => {
    const model = await seedModel();
    // Plenty of samples, but all of them two hours old: as far as the last
    // hour is concerned this member served nothing.
    const old = await seedMember({
      nodeName: "spark-03", publishedName: "pool-a", modelId: model.id, rate: 23, ageMinutes: 120,
    });
    await seedMember({ nodeName: "spark-04", publishedName: "pool-a", modelId: model.id, rate: 56 });

    const stale = await throughputFor(old.deployment.id);

    expect(stale.throughput.state).toBe("not-comparable");
    expect(stale.throughput.reason).toBe("idle");
  });

  // Two pools of the same model, serving very different workloads. Compared
  // within their own pool everyone is healthy; lumped together, the slower
  // pool's members would both read as degraded.
  it("compares members only against their own published name", async () => {
    const model = await seedModel();
    await seedMember({ nodeName: "spark-01", publishedName: "fast-pool", modelId: model.id, rate: 55 });
    await seedMember({ nodeName: "spark-02", publishedName: "fast-pool", modelId: model.id, rate: 56 });
    const slowA = await seedMember({ nodeName: "spark-03", publishedName: "slow-pool", modelId: model.id, rate: 20 });
    const slowB = await seedMember({ nodeName: "spark-04", publishedName: "slow-pool", modelId: model.id, rate: 21 });

    expect((await throughputFor(slowA.deployment.id)).throughput.state).toBe("ok");
    expect((await throughputFor(slowB.deployment.id)).throughput.state).toBe("ok");
  });
});
