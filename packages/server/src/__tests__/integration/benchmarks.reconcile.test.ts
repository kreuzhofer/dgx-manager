import {
  describe, expect, it, beforeAll, afterAll, beforeEach, vi,
} from "vitest";
import { execSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-benchmark-reconcile-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;
process.env.SHARED_STORAGE_PATH = TMP_DIR;
// Deliberately NOT mocking the orchestrator — reconcileStaleRuns/executeRun run
// for real, dispatching through the real remote-runner. Every network-shaped
// side effect (job.start/job.status/job.logs/job.result) goes through the stub
// `invoke` passed in per test, so nothing ever calls out to a real agent.
//
// This means the test needs orchestrator.ts's module-level `RUNNER` const to
// resolve to "remote" (the production default). Under vitest.config.ts's
// singleFork pool, that const is captured once per process from whatever
// BENCH_RUNNER happens to be at that moment, and it can already be "local"
// here: benchmarks/orchestrator.test.ts forces BENCH_RUNNER="local" via
// vi.hoisted() for its own unit tests, and hoisting means that assignment
// runs before that file's own "restore" logic captures the pre-existing
// value, so the restore ends up reasserting "local" instead of undoing it.
// Force the real default back before boot-reconcile.js pulls in a fresh
// copy of orchestrator.js (vitest's per-file module reset — see setup.ts —
// makes it re-evaluate the module-level const from this value).
process.env.BENCH_RUNNER = "remote";

let prisma: typeof import("../../prisma.js").prisma;
let reconcileStaleRuns: typeof import("../../benchmarks/boot-reconcile.js").reconcileStaleRuns;

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
  ({ reconcileStaleRuns } = await import("../../benchmarks/boot-reconcile.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  // FK-ordered wipe
  await prisma.benchmarkResult.deleteMany();
  await prisma.toolEvalCategory.deleteMany();
  await prisma.benchmarkRun.deleteMany();
  await prisma.deployment.deleteMany();
  await prisma.model.deleteMany();
  await prisma.node.deleteMany();
});

const THROUGHPUT_CONFIG = {
  pp: [128], tg: [32], depth: [0], runs: 1, concurrency: [1],
  latencyMode: "api", enablePrefixCaching: false, skipCoherence: false,
};

function seedRun(data: Partial<Parameters<typeof prisma.benchmarkRun.create>[0]["data"]> = {}) {
  return prisma.benchmarkRun.create({
    data: {
      kind: "throughput",
      modelName: "m",
      endpointUrl: "http://10.0.0.1:8000/v1",
      servedModelName: "m",
      config: JSON.stringify(THROUGHPUT_CONFIG),
      status: "running",
      runnerNodeId: null,
      ...data,
    },
  });
}

describe("reconcileStaleRuns", () => {
  it("fails a legacy (local) run with the preserved message", async () => {
    const run = await seedRun({ runnerNodeId: null });
    const invoke = vi.fn(() => {
      throw new Error("a legacy run must not query the agent");
    });

    await reconcileStaleRuns(invoke);

    const after = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
    expect(after?.status).toBe("failed");
    expect(after?.error).toBe("server restarted before run completed");
    expect(invoke).not.toHaveBeenCalled();
  });

  it("fails a remote run whose job is missing with no result", async () => {
    const run = await seedRun({ runnerNodeId: "n1" });
    const invoke = vi.fn(async (_nodeId: string, name: string) => {
      if (name === "job.status") return { ok: true, data: { kind: "missing" } };
      if (name === "job.result") return { ok: true, data: { raw: null } };
      throw new Error(`unexpected cap call: ${name}`);
    });

    await reconcileStaleRuns(invoke);

    const after = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
    expect(after?.status).toBe("failed");
    expect(after?.error).toMatch(/vanished/);
  });

  it("leaves an active remote run running (resumes, does not fail it)", async () => {
    const run = await seedRun({ runnerNodeId: "n1" });
    const invoke = vi.fn(async (_nodeId: string, name: string) => {
      if (name === "job.status") return { ok: true, data: { kind: "active" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
      throw new Error(`unexpected cap call: ${name}`);
    });

    // reconcileStaleRuns awaits only enough to decide the action and kick off
    // executeRun (fire-and-forget); executeRun's poll loop sleeps ~3s before
    // its next tick, so it cannot have finished by the time we get here.
    await reconcileStaleRuns(invoke);

    const after = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
    expect(after?.status).toBe("running");
  });

  it("does not query the agent for a legacy run", async () => {
    await seedRun({ runnerNodeId: null });
    let callCount = 0;
    const invoke = vi.fn(async () => {
      callCount++;
      return { ok: true, data: {} };
    });

    await reconcileStaleRuns(invoke);

    expect(callCount).toBe(0);
    expect(invoke).not.toHaveBeenCalled();
  });
});
