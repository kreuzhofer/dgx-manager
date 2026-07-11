import {
  describe, expect, it, beforeAll, afterAll, beforeEach, vi,
} from "vitest";
import { execSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// C1 regression coverage: a REMOTE run's tool output path must be relative to
// the eval node's job dir (the agent wrapper `cd`s to jobDir and `find`s
// results under jobDir/out — see packages/agent/src/jobs/job-spec.ts). Before
// the fix, executeRun always used the manager's absolute SHARED_STORAGE path
// (e.g. /mnt/tank/benchmarks/<id>/...), which does not exist on the eval
// node: the tool either fails to write there, or the wrapper's `find` scans
// an empty jobDir/out and the run silently completes with no result. Both
// sides of that boundary were previously stubbed in unit tests, so this
// exercises executeRun for real (only the network-shaped `invoke` cap calls
// are stubbed) with a real per-suite SQLite DB.
const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-benchmark-execute-output-path-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;
process.env.SHARED_STORAGE_PATH = TMP_DIR;
// See benchmarks.reconcile.test.ts for why this must be forced before
// execute.js (which transitively imports orchestrator.js) is loaded: the
// shared singleFork process may already have BENCH_RUNNER="local" cached
// from another suite, and "remote" (unset env, the production default) is
// what actually exercises the runnerNodeId-gated path this test verifies.
process.env.BENCH_RUNNER = "remote";

let prisma: typeof import("../../prisma.js").prisma;
let executeRun: typeof import("../../benchmarks/execute.js").executeRun;

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
  ({ executeRun } = await import("../../benchmarks/execute.js"));
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

const ACCURACY_CONFIG = {
  tasks: ["ifeval"], primaryTask: "ifeval", primaryMetric: "prompt_level_strict_acc",
  limit: 1, numFewshot: null, maxGenToks: 16, applyChatTemplate: false,
  reasoning: false, // skip the reasoning proxy — irrelevant to output-path routing
  seed: 1,
};

const THROUGHPUT_CONFIG = {
  pp: [128], tg: [32], depth: [0], runs: 1, concurrency: [1],
  latencyMode: "api", enablePrefixCaching: false, skipCoherence: false,
};

function seedRun(kind: string, config: unknown, runnerNodeId: string | null) {
  return prisma.benchmarkRun.create({
    data: {
      kind,
      modelName: "m",
      endpointUrl: "http://10.0.0.1:8000/v1",
      servedModelName: "m",
      config: JSON.stringify(config),
      status: "running",
      runnerNodeId,
    },
  });
}

/** A stub cap invoker that finishes the job on the very first status poll. */
function stubInvoke(onStart: (argv: string[], resultGlob: string | undefined) => void) {
  return vi.fn(async (_nodeId: string, name: string, input: unknown) => {
    if (name === "job.start") {
      const { argv, resultGlob } = input as { argv: string[]; resultGlob?: string };
      onStart(argv, resultGlob);
      return { ok: true, data: {} };
    }
    if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
    if (name === "job.status") return { ok: true, data: { kind: "exited", code: 0 } };
    if (name === "job.result") return { ok: true, data: { raw: null } };
    throw new Error(`unexpected cap call: ${name}`);
  });
}

describe("executeRun — remote output paths", () => {
  it("dispatches a REMOTE accuracy run with a job-dir-relative --output_path, not an absolute SHARED_STORAGE path", async () => {
    const run = await seedRun("accuracy", ACCURACY_CONFIG, "eval-node-1");
    let startArgv: string[] | undefined;
    const invoke = stubInvoke((argv) => { startArgv = argv; });

    executeRun(run, invoke, false);

    await vi.waitFor(() => {
      expect(startArgv).toBeDefined();
    });

    const idx = startArgv!.indexOf("--output_path");
    expect(idx).toBeGreaterThan(-1);
    expect(startArgv![idx + 1]).toBe("out");
    expect(startArgv!.some((a) => a.includes(TMP_DIR))).toBe(false);
    expect(startArgv!.some((a) => a.startsWith("/mnt/tank"))).toBe(false);

    // Let the run settle so it doesn't leak into the next test's DB wipe.
    await vi.waitFor(async () => {
      const after = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
      expect(after?.status).not.toBe("running");
    });
  });

  it("dispatches a REMOTE throughput run with a job-dir-relative --save-result path", async () => {
    const run = await seedRun("throughput", THROUGHPUT_CONFIG, "eval-node-1");
    let startArgv: string[] | undefined;
    const invoke = stubInvoke((argv) => { startArgv = argv; });

    executeRun(run, invoke, false);

    await vi.waitFor(() => {
      expect(startArgv).toBeDefined();
    });

    const idx = startArgv!.indexOf("--save-result");
    expect(idx).toBeGreaterThan(-1);
    expect(startArgv![idx + 1]).toBe("out/result.json");
    expect(startArgv!.some((a) => a.includes(TMP_DIR))).toBe(false);

    await vi.waitFor(async () => {
      const after = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
      expect(after?.status).not.toBe("running");
    });
  });
});

// Local (non-remote) absolute-path behavior is already covered by
// benchmarks.routes.test.ts (`expect(call.outputDir).toBe(...SHARED_STORAGE...)`
// with app.set("benchRunner", "local")), so it isn't duplicated here — this
// file spawns no real child process and stays focused on the remote seam.
