/**
 * GET /api/nodes should return the latest sample for each node from the
 * in-memory metricsBuffer, not from the MetricSnapshot table. This keeps
 * the endpoint at sub-ms latency regardless of how many DB rows exist,
 * which used to be the source of a ~10s page load.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
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
let nodesRouter: typeof import("../../routes/nodes.js").nodesRouter;
let metricsBuffer: typeof import("../../metrics-buffer.js").metricsBuffer;

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
  ({ metricsBuffer } = await import("../../metrics-buffer.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

function makeApp() {
  const app = express();
  app.use(express.json());
  app.use("/api/nodes", nodesRouter);
  return app;
}

beforeEach(async () => {
  await prisma.metricSnapshot.deleteMany({});
  await prisma.node.deleteMany({});
  metricsBuffer.remove("node-a");
  metricsBuffer.remove("node-b");
});

describe("GET /api/nodes", () => {
  it("returns the latest sample from metricsBuffer for each node", async () => {
    await prisma.node.createMany({
      data: [
        { id: "node-a", name: "node-a", ipAddress: "10.0.0.1" },
        { id: "node-b", name: "node-b", ipAddress: "10.0.0.2" },
      ],
    });
    metricsBuffer.push("node-a", {
      timestamp: 1_700_000_000_000,
      gpuUtil: 42,
      vramUsed: 1234,
      temperature: 55,
      tps: null,
      activeRequests: null,
    });
    // node-b deliberately has no buffered sample.

    const res = await request(makeApp()).get("/api/nodes").expect(200);

    const a = res.body.find((n: { id: string }) => n.id === "node-a");
    const b = res.body.find((n: { id: string }) => n.id === "node-b");
    expect(a.metrics).toHaveLength(1);
    expect(a.metrics[0].gpuUtil).toBe(42);
    expect(a.metrics[0].vramUsed).toBe(1234);
    expect(b.metrics).toEqual([]);
  });

  it("ignores the MetricSnapshot table entirely (returns [] when only DB rows exist)", async () => {
    await prisma.node.create({
      data: { id: "node-a", name: "node-a", ipAddress: "10.0.0.1" },
    });
    // Persisted, but NOT in the buffer.
    await prisma.metricSnapshot.create({
      data: { nodeId: "node-a", gpuUtil: 99, vramUsed: 9999, timestamp: new Date() },
    });

    const res = await request(makeApp()).get("/api/nodes").expect(200);

    const a = res.body.find((n: { id: string }) => n.id === "node-a");
    expect(a.metrics).toEqual([]);
  });
});
