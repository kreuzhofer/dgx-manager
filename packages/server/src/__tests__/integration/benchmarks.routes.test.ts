import {
  describe, expect, it, beforeAll, afterAll, beforeEach, vi,
} from "vitest";
import { execSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-benchmark-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;
process.env.SHARED_STORAGE_PATH = TMP_DIR;

// Mock the orchestrator so the route test never spawns uvx.
const runMock = vi.fn();
const runToolEvalMock = vi.fn();
const cancelMock = vi.fn();
vi.mock("../../benchmarks/orchestrator.js", () => ({
  runBenchmark: (...a: unknown[]) => runMock(...a),
  runToolEval: (...a: unknown[]) => runToolEvalMock(...a),
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

describe("GET /api/benchmarks/presets", () => {
  it("returns the built-in presets", async () => {
    const res = await request(makeApp()).get("/api/benchmarks/presets");
    expect(res.status).toBe(200);
    expect(res.body.map((p: { id: string }) => p.id)).toContain("quick-smoke");
  });

  it("includes the four tool-eval presets tagged kind 'tool-eval'", async () => {
    const res = await request(makeApp()).get("/api/benchmarks/presets");
    expect(res.status).toBe(200);
    const ids = res.body.map((p: { id: string }) => p.id);
    for (const id of [
      "tool-eval-quick",
      "tool-eval-full",
      "tool-eval-hardmode",
      "tool-eval-pressure",
    ]) {
      expect(ids).toContain(id);
    }
    const quick = res.body.find((p: { id: string }) => p.id === "tool-eval-quick");
    expect(quick.kind).toBe("tool-eval");
  });
});

describe("POST /api/benchmarks (tool-eval dispatch)", () => {
  it("creates a run with kind 'tool-eval' for a tool-eval preset", async () => {
    const d = await seedRunningDeployment();
    runToolEvalMock.mockReturnValue(new Promise(() => {}));
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "tool-eval-quick" });
    expect(res.status).toBe(201);
    expect(res.body.kind).toBe("tool-eval");
    expect(res.body.presetId).toBe("tool-eval-quick");
    expect(runToolEvalMock).toHaveBeenCalledTimes(1);
    expect(runMock).not.toHaveBeenCalled();
  });

  it("creates a throughput run for a custom config (tool-eval is preset-only)", async () => {
    const d = await seedRunningDeployment();
    runMock.mockReturnValue(new Promise(() => {}));
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({
        deploymentId: d.id,
        config: {
          pp: [1], tg: [1], depth: [0], runs: 1, concurrency: [1],
          latencyMode: "none", enablePrefixCaching: false, skipCoherence: false,
        },
      });
    expect(res.status).toBe(201);
    expect(res.body.kind).toBe("throughput");
    expect(runMock).toHaveBeenCalledTimes(1);
    expect(runToolEvalMock).not.toHaveBeenCalled();
  });

  it("persists eval headline fields and category rows on completion", async () => {
    const d = await seedRunningDeployment();
    runToolEvalMock.mockImplementation(async (opts: { onLog: (l: string) => void }) => {
      opts.onLog("benchmark_complete");
      return {
        exitCode: 0,
        rawOutput: "{}",
        summary: {
          finalScore: 67,
          rating: "★★★ Adequate",
          deployability: 48,
          responsiveness: 2,
          totalScenarios: 15,
          totalPoints: 20,
          maxPoints: 30,
          safetyWarnings: [],
          categories: [{
            code: "A", label: "Tool Selection", percent: 100,
            earned: 6, maxPoints: 6, passCount: 3, partialCount: 0, failCount: 0,
          }],
        },
      };
    });

    const app = makeApp();
    const res = await request(app)
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "tool-eval-quick" });
    const runId = res.body.id;

    // Let the floating completion promise settle.
    await new Promise((r) => setTimeout(r, 20));

    const detail = await request(app).get(`/api/benchmarks/${runId}`);
    expect(detail.body.status).toBe("completed");
    expect(detail.body.toolEvalScore).toBe(67);
    expect(detail.body.toolEvalRating).toBe("★★★ Adequate");
    expect(detail.body.toolEvalTotalScenarios).toBe(15);
    expect(detail.body.toolEvalCategories.length).toBe(1);
    expect(detail.body.toolEvalCategories[0].code).toBe("A");
  });

  it("marks the run failed when tool-eval-bench exits non-zero", async () => {
    const d = await seedRunningDeployment();
    runToolEvalMock.mockResolvedValue({ exitCode: 1, rawOutput: null, summary: null });

    const app = makeApp();
    const res = await request(app)
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "tool-eval-quick" });
    const runId = res.body.id;

    // Let the floating completion promise settle.
    await new Promise((r) => setTimeout(r, 20));

    const detail = await request(app).get(`/api/benchmarks/${runId}`);
    expect(detail.body.status).toBe("failed");
    expect(detail.body.error).toMatch(/tool-eval-bench exited with code 1/);
    expect(detail.body.toolEvalScore).toBeNull();
    expect(detail.body.toolEvalCategories.length).toBe(0);
  });
});

describe("POST /api/benchmarks", () => {
  it("creates a run, spawns the orchestrator, and returns the run id", async () => {
    const d = await seedRunningDeployment();
    runMock.mockResolvedValue({
      exitCode: 0,
      results: [{
        opType: "tg", pp: 128, tg: 32, depth: 0, concurrency: 1,
        tps: 50, peakTps: 60, ttfrMs: 100,
        estPptMs: 50, e2eTtftMs: 150, tpsStdev: 1, ttfrStdev: 2,
      }],
      summary: { meanTps: 50, meanTtfrMs: 100 },
      rawOutput: "{}",
    });

    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "quick-smoke" });

    expect(res.status).toBe(201);
    expect(res.body.id).toBeDefined();
    expect(res.body.status).toBe("pending");

    // Orchestrator was called with the right shape
    expect(runMock).toHaveBeenCalledTimes(1);
    const call = runMock.mock.calls[0][0];
    expect(call.runId).toBe(res.body.id);
    expect(call.args).toContain("--base-url");
    // llama-benchy needs the /v1 suffix on base-url (OpenAI client convention)
    expect(call.args).toContain("http://10.0.0.1:8000/v1");
    expect(call.outputDir).toBe(`${TMP_DIR}/benchmarks/${res.body.id}`);
    // The --save-result path passed to llama-benchy must live inside outputDir
    const idx = call.args.indexOf("--save-result");
    expect(call.args[idx + 1]).toBe(`${TMP_DIR}/benchmarks/${res.body.id}/result.json`);

    // Wait one microtask for the orchestrator's then() to land
    await new Promise((r) => setImmediate(r));

    const stored = await prisma.benchmarkRun.findUnique({
      where: { id: res.body.id },
      include: { results: true },
    });
    expect(stored?.status).toBe("completed");
    expect(stored?.results).toHaveLength(1);
    expect(stored?.meanTps).toBe(50);
  });

  it("returns 404 when the deployment does not exist", async () => {
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: "nope", presetId: "quick-smoke" });
    expect(res.status).toBe(404);
  });

  it("returns 409 when the deployment is not running", async () => {
    const node = await prisma.node.create({
      data: { name: "n", ipAddress: "10.0.0.1", status: "online" },
    });
    const model = await prisma.model.create({
      data: { name: "m", runtime: "vllm" },
    });
    const d = await prisma.deployment.create({
      data: { nodeId: node.id, modelId: model.id, status: "stopped", port: 8000 },
    });
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "quick-smoke" });
    expect(res.status).toBe(409);
    expect(res.body.error).toMatch(/running/i);
  });

  it("returns 409 if another benchmark is already running for that deployment", async () => {
    const d = await seedRunningDeployment();
    // Hold the first run open by returning a pending promise.
    runMock.mockReturnValue(new Promise(() => {}));
    const first = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "quick-smoke" });
    expect(first.status).toBe(201);

    const second = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id, presetId: "quick-smoke" });
    expect(second.status).toBe(409);
    expect(second.body.error).toMatch(/already/i);
  });

  it("400s when neither presetId nor a custom config is provided", async () => {
    const d = await seedRunningDeployment();
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({ deploymentId: d.id });
    expect(res.status).toBe(400);
  });

  it("accepts a fully custom config", async () => {
    const d = await seedRunningDeployment();
    runMock.mockResolvedValue({
      exitCode: 0, results: [],
      summary: { meanTps: null, meanTtfrMs: null }, rawOutput: null,
    });
    const res = await request(makeApp())
      .post("/api/benchmarks")
      .send({
        deploymentId: d.id,
        config: {
          pp: [64], tg: [16], depth: [0], runs: 1,
          concurrency: [1], latencyMode: "api",
          enablePrefixCaching: false, skipCoherence: false,
        },
      });
    expect(res.status).toBe(201);
    expect(res.body.presetId).toBeNull();
  });
});

describe("GET /api/benchmarks", () => {
  it("returns runs filtered by deploymentId, newest first", async () => {
    const d = await seedRunningDeployment();
    await prisma.benchmarkRun.create({
      data: {
        deploymentId: d.id, modelName: "m", endpointUrl: "u",
        servedModelName: "m", config: "{}", status: "completed",
      },
    });
    await prisma.benchmarkRun.create({
      data: {
        deploymentId: d.id, modelName: "m", endpointUrl: "u",
        servedModelName: "m", config: "{}", status: "completed",
      },
    });
    const res = await request(makeApp())
      .get(`/api/benchmarks?deploymentId=${d.id}`);
    expect(res.status).toBe(200);
    expect(res.body).toHaveLength(2);
  });
});

describe("GET /api/benchmarks/:id", () => {
  it("returns a run with its results", async () => {
    const d = await seedRunningDeployment();
    const run = await prisma.benchmarkRun.create({
      data: {
        deploymentId: d.id, modelName: "m", endpointUrl: "u",
        servedModelName: "m", config: "{}", status: "completed",
        results: {
          create: [{
            opType: "tg", pp: 1, tg: 2, depth: 0, concurrency: 1, tps: 10,
          }],
        },
      },
    });
    const res = await request(makeApp()).get(`/api/benchmarks/${run.id}`);
    expect(res.status).toBe(200);
    expect(res.body.results).toHaveLength(1);
  });

  it("404 for unknown id", async () => {
    const res = await request(makeApp()).get("/api/benchmarks/missing");
    expect(res.status).toBe(404);
  });
});

describe("DELETE /api/benchmarks/:id", () => {
  it("removes the run and cascades to results", async () => {
    const d = await seedRunningDeployment();
    const run = await prisma.benchmarkRun.create({
      data: {
        deploymentId: d.id, modelName: "m", endpointUrl: "u",
        servedModelName: "m", config: "{}", status: "completed",
        results: {
          create: [{
            opType: "tg", pp: 1, tg: 2, depth: 0, concurrency: 1, tps: 10,
          }],
        },
      },
    });
    const res = await request(makeApp()).delete(`/api/benchmarks/${run.id}`);
    expect(res.status).toBe(204);
    expect(await prisma.benchmarkRun.findUnique({ where: { id: run.id } })).toBeNull();
    expect(await prisma.benchmarkResult.findMany({ where: { runId: run.id } })).toHaveLength(0);
  });
});

describe("POST /api/benchmarks/:id/cancel", () => {
  it("kills the child and marks the run canceled", async () => {
    const d = await seedRunningDeployment();
    const run = await prisma.benchmarkRun.create({
      data: {
        deploymentId: d.id, modelName: "m", endpointUrl: "u",
        servedModelName: "m", config: "{}", status: "running",
      },
    });
    const res = await request(makeApp()).post(`/api/benchmarks/${run.id}/cancel`);
    expect(res.status).toBe(200);
    expect(cancelMock).toHaveBeenCalledWith(run.id);
    const after = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
    expect(after?.status).toBe("canceled");
  });
});
