/**
 * Integration test for GET /api/gateway — the management view the dashboard's
 * Gateway page renders.
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
let acquire: typeof import("../../gateway/inflight.js").acquire;
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
  ({ acquire, resetOutstanding } = await import("../../gateway/inflight.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

afterEach(async () => {
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
  resetOutstanding();
});

function makeApp(agentHub: unknown = { isAgentOnline: () => true }) {
  const app = express();
  app.set("agentHub", agentHub);
  app.use("/api/gateway", gatewayViewRouter);
  return app;
}

let seq = 0;
async function seed(opts: {
  publishedName: string | null;
  status?: string;
  runtime?: string;
  port?: number | null;
  nodeName?: string;
}) {
  const n = ++seq;
  const node = await prisma.node.create({
    data: { name: opts.nodeName ?? `node-${n}`, ipAddress: `10.0.0.${n}`, status: "online" },
  });
  const model = await prisma.model.create({
    data: { name: `catalog-${n}`, runtime: opts.runtime ?? "vllm" },
  });
  const deployment = await prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: opts.status ?? "running",
      port: opts.port === undefined ? 8000 : opts.port,
      publishedName: opts.publishedName,
    },
  });
  return { node, deployment };
}

describe("GET /api/gateway", () => {
  it("reports nothing when the cluster publishes nothing", async () => {
    const res = await request(makeApp()).get("/api/gateway");
    expect(res.status).toBe(200);
    expect(res.body.pools).toEqual([]);
  });

  // This is the string an operator copies and pastes onto another machine, so
  // it has to be the manager's real address — not a build-time default that
  // says localhost outside compose.
  describe("the base URL a client should use", () => {
    const original = process.env.MANAGER_ADVERTISE_HOST;
    afterEach(() => {
      if (original === undefined) delete process.env.MANAGER_ADVERTISE_HOST;
      else process.env.MANAGER_ADVERTISE_HOST = original;
    });

    it("uses the advertised host when one is configured", async () => {
      process.env.MANAGER_ADVERTISE_HOST = "192.168.44.14";
      const res = await request(makeApp()).get("/api/gateway");
      expect(res.body.baseUrl).toBe(`http://192.168.44.14:${process.env.PORT || "4000"}/v1`);
    });

    it("falls back to the host the request arrived on", async () => {
      delete process.env.MANAGER_ADVERTISE_HOST;
      const res = await request(makeApp()).get("/api/gateway");
      expect(res.body.baseUrl).toMatch(/^http:\/\/[^/]+\/v1$/);
      expect(res.body.baseUrl).not.toContain("undefined");
    });
  });

  it("reports a pool with its member's node, runtime and load", async () => {
    const { deployment } = await seed({
      publishedName: "glm-5.2",
      nodeName: "dgx-spark-01",
      runtime: "vllm",
    });
    acquire(deployment.id);
    acquire(deployment.id);

    const res = await request(makeApp()).get("/api/gateway");

    expect(res.body.pools).toHaveLength(1);
    expect(res.body.pools[0]).toMatchObject({ publishedName: "glm-5.2", servingCount: 1 });
    expect(res.body.pools[0].members[0]).toMatchObject({
      node: "dgx-spark-01",
      runtime: "vllm",
      port: 8000,
      inflight: 2,
      serving: true,
    });
  });

  // The reason this view exists: a pool is invisible in a per-deployment table.
  it("groups deployments sharing a name into one pool", async () => {
    await seed({ publishedName: "qwen3-embedding:8b", nodeName: "agenthost", runtime: "ollama" });
    await seed({ publishedName: "qwen3-embedding:8b", nodeName: "aihost01", runtime: "ollama" });

    const res = await request(makeApp()).get("/api/gateway");

    expect(res.body.pools).toHaveLength(1);
    expect(res.body.pools[0].members.map((m: { node: string }) => m.node).sort()).toEqual([
      "agenthost",
      "aihost01",
    ]);
    expect(res.body.pools[0].servingCount).toBe(2);
  });

  it("shows a member that cannot serve, with the reason", async () => {
    await seed({ publishedName: "glm-5.2", nodeName: "gone" });

    const res = await request(makeApp({ isAgentOnline: () => false })).get("/api/gateway");

    expect(res.body.pools[0].servingCount).toBe(0);
    expect(res.body.pools[0].members[0]).toMatchObject({
      node: "gone",
      serving: false,
      reason: "agent-offline",
    });
  });

  it("omits a deployment that has no published name", async () => {
    await seed({ publishedName: null });
    const res = await request(makeApp()).get("/api/gateway");
    expect(res.body.pools).toEqual([]);
  });

  it("orders pools by name so the page does not reshuffle between polls", async () => {
    await seed({ publishedName: "zeta" });
    await seed({ publishedName: "alpha" });

    const res = await request(makeApp()).get("/api/gateway");

    expect(res.body.pools.map((p: { publishedName: string }) => p.publishedName)).toEqual([
      "alpha",
      "zeta",
    ]);
  });
});
