import {
  describe, expect, it, beforeAll, afterAll, beforeEach, vi,
} from "vitest";
import { execSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-benchmark-remote-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;
process.env.SHARED_STORAGE_PATH = TMP_DIR;

// Mock the orchestrator so the route test never spawns uvx.
const runMock = vi.fn();
const runToolEvalMock = vi.fn();
const runAccuracyMock = vi.fn();
const cancelMock = vi.fn();
vi.mock("../../benchmarks/orchestrator.js", () => ({
  runBenchmark: (...a: unknown[]) => runMock(...a),
  runToolEval: (...a: unknown[]) => runToolEvalMock(...a),
  runAccuracy: (...a: unknown[]) => runAccuracyMock(...a),
  cancelBenchmark: (...a: unknown[]) => cancelMock(...a),
}));

let prisma: typeof import("../../prisma.js").prisma;
let benchmarksRouter: typeof import("../../routes/benchmarks.js").benchmarksRouter;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      DATABASE_URL: `file:${DB_PATH}`,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "I understand this is destructive and I have backups",
    },
    stdio: "pipe",
  });
  ({ prisma } = await import("../../prisma.js"));
  ({ benchmarksRouter } = await import("../../routes/benchmarks.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  runMock.mockReset();
  runToolEvalMock.mockReset();
  runAccuracyMock.mockReset();
  cancelMock.mockReset();
  // FK-ordered wipe
  await prisma.benchmarkResult.deleteMany();
  await prisma.toolEvalCategory.deleteMany();
  await prisma.benchmarkRun.deleteMany();
  await prisma.deployment.deleteMany();
  await prisma.model.deleteMany();
  await prisma.node.deleteMany();
});

function makeApp() {
  const app = express();
  app.use(express.json());
  app.set("benchRunner", "remote");
  app.use("/api/benchmarks", benchmarksRouter);
  return app;
}

async function seedRunningDeployment() {
  const node = await prisma.node.create({
    data: { name: "n1", ipAddress: "10.0.0.1", status: "online" },
  });
  const model = await prisma.model.create({
    data: { name: "llama-3.1-8b", runtime: "vllm" },
  });
  return prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: "running",
      port: 8000,
      displayName: "llama-prod",
    },
    include: { node: true, model: true },
  });
}

describe("POST /api/benchmarks — remote runner resolution", () => {
  async function seedEvalNode() {
    return prisma.node.create({
      data: { name: "agenthost", ipAddress: "192.168.44.15", status: "online", role: "eval" },
    });
  }

  it("503 and creates NO run when there is no online eval node", async () => {
    const dep = await seedRunningDeployment(); // gpu node only, no eval node
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: dep.id, presetId: "quick-smoke" });
    expect(res.status).toBe(503);
    expect(res.body.error).toMatch(/eval runner unavailable/i);
    expect(await prisma.benchmarkRun.count()).toBe(0);
  });

  it("201 and persists runnerNodeId when an eval node is online", async () => {
    const evalNode = await seedEvalNode();
    const dep = await seedRunningDeployment();
    // The route always chains .then()/.catch() off runBenchmark's return value,
    // so the mock needs a Promise even when the test doesn't care about completion.
    runMock.mockReturnValue(new Promise(() => {}));
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: dep.id, presetId: "quick-smoke" });
    expect(res.status).toBe(201);
    const run = await prisma.benchmarkRun.findFirst();
    expect(run?.runnerNodeId).toBe(evalNode.id);
  });

  it("still 409s a second run against a busy deployment (existing guard, remote mode)", async () => {
    await seedEvalNode();
    const dep = await seedRunningDeployment();
    runMock.mockReturnValue(new Promise(() => {}));
    const app = makeApp();
    const first = await request(app).post("/api/benchmarks").send({ deploymentId: dep.id, presetId: "quick-smoke" });
    expect(first.status).toBe(201);
    const second = await request(app).post("/api/benchmarks").send({ deploymentId: dep.id, presetId: "quick-smoke" });
    expect(second.status).toBe(409);
  });
});
