import { describe, expect, it, vi, beforeEach, afterAll } from "vitest";
import { EventEmitter } from "node:events";

// dispatch()'s module-level RUNNER const (in orchestrator.ts) is read once,
// at import time, from BENCH_RUNNER — defaulting to "remote". Every test
// below except the last predates the eval-node dispatch and exercises the
// *local* spawnTracked path (uvx argv construction, mkdirSync, detached
// SIGTERM group-kill) via the spawnMock/fs mocks in this file, so force
// local mode for orchestrator.js's one static import. Plain top-level code
// would run too late — ESM hoists `import` declarations above ordinary
// statements regardless of source order — so this has to go through
// vi.hoisted to actually land before "./orchestrator.js" evaluates.
// Restored in afterAll since vitest.config.ts runs the whole suite in one
// shared fork (poolOptions.forks.singleFork): an unrestored env var here
// would leak into whichever test file imports orchestrator.js next.
const PRE_BENCH_RUNNER = process.env.BENCH_RUNNER;
vi.hoisted(() => {
  process.env.BENCH_RUNNER = "local";
});

// Mock node:child_process so spawn hits our fake.
const spawnMock = vi.fn();
vi.mock("node:child_process", () => ({
  spawn: (...args: unknown[]) => spawnMock(...args),
}));

// Mock fs so we don't touch the shared storage path during unit tests.
const readFileSyncMock = vi.fn();
const mkdirSyncMock = vi.fn();
const existsSyncMock = vi.fn();
vi.mock("node:fs", async () => {
  const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
  return {
    ...actual,
    readFileSync: (...a: unknown[]) => readFileSyncMock(...a),
    mkdirSync: (...a: unknown[]) => mkdirSyncMock(...a),
    existsSync: (...a: unknown[]) => existsSyncMock(...a),
  };
});

const startProxyMock = vi.fn();
vi.mock("./reasoning-proxy.js", () => ({
  startReasoningProxy: (...a: unknown[]) => startProxyMock(...a),
}));
const findFileMock = vi.fn();
vi.mock("./lm-eval-result-file.js", () => ({
  findLmEvalResultFile: (...a: unknown[]) => findFileMock(...a),
}));

import { runBenchmark, runToolEval, runAccuracy, cancelBenchmark } from "./orchestrator.js";

afterAll(() => {
  if (PRE_BENCH_RUNNER === undefined) delete process.env.BENCH_RUNNER;
  else process.env.BENCH_RUNNER = PRE_BENCH_RUNNER;
});

function makeFakeChild(pid = 4242) {
  const child = new EventEmitter() as EventEmitter & {
    stdout: EventEmitter;
    stderr: EventEmitter;
    kill: ReturnType<typeof vi.fn>;
    pid: number;
  };
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.kill = vi.fn();
  child.pid = pid;
  return child;
}

describe("runBenchmark", () => {
  beforeEach(() => {
    spawnMock.mockReset();
    readFileSyncMock.mockReset();
    mkdirSyncMock.mockReset();
    existsSyncMock.mockReset();
  });

  it("spawns `uvx llama-benchy` with the supplied argv and mkdirs the output dir", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(true);
    readFileSyncMock.mockReturnValue('{"rows":[]}');

    const onLog = vi.fn();
    const promise = runBenchmark({
      runId: "run_abc",
      args: ["--base-url", "http://10.0.0.1:8000", "--model", "m", "--save-result", "/mnt/tank/benchmarks/run_abc/result.json"],
      outputDir: "/mnt/tank/benchmarks/run_abc",
      onLog,
    });

    child.stdout.emit("data", Buffer.from("running test 1/3...\n"));
    child.emit("close", 0);

    const result = await promise;
    expect(spawnMock).toHaveBeenCalledTimes(1);
    const [cmd, argv, spawnOpts] = spawnMock.mock.calls[0];
    expect(cmd).toBe("uvx");
    expect(argv[0]).toBe("--from");
    expect(argv[1]).toMatch(/^llama-benchy(==.+)?$/);
    expect(argv[2]).toBe("llama-benchy");
    expect(argv.slice(3)).toEqual([
      "--base-url", "http://10.0.0.1:8000",
      "--model", "m",
      "--save-result", "/mnt/tank/benchmarks/run_abc/result.json",
    ]);
    expect((spawnOpts as { detached: boolean }).detached).toBe(true);
    expect(result.exitCode).toBe(0);
    expect(onLog).toHaveBeenCalledWith("running test 1/3...");
    expect(mkdirSyncMock).toHaveBeenCalledWith(
      "/mnt/tank/benchmarks/run_abc",
      expect.objectContaining({ recursive: true }),
    );
  });

  it("returns parsed results when result.json exists after exit", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(true);
    readFileSyncMock.mockReturnValue(
      JSON.stringify({
        benchmarks: [{
          concurrency: 1, context_size: 0, prompt_size: 1, response_size: 2,
          pp_throughput: { mean: 100.0, std: 1, values: [100] },
          tg_throughput: { mean: 50.5, std: 0.5, values: [50.5] },
          peak_throughput: { mean: 60, std: 0, values: [60] },
          ttfr: { mean: 100, std: 2, values: [100] },
          est_ppt: { mean: 50, std: 0, values: [50] },
          e2e_ttft: { mean: 150, std: 0, values: [150] },
        }],
      }),
    );
    const promise = runBenchmark({
      runId: "r2", args: [], outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 0);
    const r = await promise;
    // One benchmarks entry → 2 rows (pp + tg)
    expect(r.results).toHaveLength(2);
    expect(r.results.find((x) => x.opType === "tg")?.tps).toBe(50.5);
  });

  it("returns exitCode and no results when the child exits non-zero", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(false);
    const promise = runBenchmark({
      runId: "r3", args: [], outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 137);
    const r = await promise;
    expect(r.exitCode).toBe(137);
    expect(r.results).toEqual([]);
    expect(readFileSyncMock).not.toHaveBeenCalled();
  });

  it("cancelBenchmark sends SIGTERM to the process group", async () => {
    const child = makeFakeChild(9999);
    spawnMock.mockReturnValue(child);
    const processKillSpy = vi.spyOn(process, "kill").mockImplementation(() => true);

    const promise = runBenchmark({
      runId: "cancelme", args: [], outputDir: "/o", onLog: vi.fn(),
    });
    expect(cancelBenchmark("cancelme")).toBe(true);
    expect(processKillSpy).toHaveBeenCalledWith(-9999, "SIGTERM");
    child.emit("close", 143);
    await promise;
    processKillSpy.mockRestore();
  });

  it("cancelBenchmark returns false when the run is not active", () => {
    expect(cancelBenchmark("ghost")).toBe(false);
  });

  it("removes the run from the active map after the child exits", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(false);
    const promise = runBenchmark({
      runId: "cleanup", args: [], outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 0);
    await promise;
    expect(cancelBenchmark("cleanup")).toBe(false);
  });
});

describe("runToolEval", () => {
  beforeEach(() => {
    spawnMock.mockReset();
    readFileSyncMock.mockReset();
    mkdirSyncMock.mockReset();
    existsSyncMock.mockReset();
  });

  it("spawns `uvx tool-eval-bench` with the supplied argv and detached", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(true);
    readFileSyncMock.mockReturnValue(
      JSON.stringify({
        final_score: 80, rating: "★★★★", total_scenarios: 15, safety_warnings: [],
        scores: { total_points: 24, max_points: 30, category_scores: [] },
      }),
    );

    const onLog = vi.fn();
    const promise = runToolEval({
      runId: "run_te",
      args: ["--base-url", "http://10.0.0.1:8000/v1", "--model", "m", "--json-file", "/mnt/tank/benchmarks/run_te/result.json", "--seed", "42"],
      outputDir: "/mnt/tank/benchmarks/run_te",
      onLog,
    });

    child.stderr.emit("data", Buffer.from('{"event":"scenario_start","scenario_id":"TC-01"}\n'));
    child.emit("close", 0);

    const result = await promise;
    const [cmd, argv, spawnOpts] = spawnMock.mock.calls[0];
    expect(cmd).toBe("uvx");
    expect(argv[0]).toBe("--from");
    expect(argv[1]).toMatch(/tool-eval-bench\.git@[0-9a-f]{7,}$/);
    expect(argv[2]).toBe("tool-eval-bench");
    expect(argv.slice(3)).toEqual([
      "--base-url", "http://10.0.0.1:8000/v1",
      "--model", "m",
      "--json-file", "/mnt/tank/benchmarks/run_te/result.json",
      "--seed", "42",
    ]);
    expect((spawnOpts as { detached: boolean }).detached).toBe(true);
    expect(result.exitCode).toBe(0);
    expect(result.summary?.finalScore).toBe(80);
    expect(onLog).toHaveBeenCalledWith('{"event":"scenario_start","scenario_id":"TC-01"}');
  });

  it("returns a null summary when the process exits non-zero", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    existsSyncMock.mockReturnValue(false);

    const promise = runToolEval({
      runId: "run_te2", args: ["--base-url", "u", "--model", "m", "--json-file", "/o/result.json"],
      outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 1);
    const result = await promise;
    expect(result.exitCode).toBe(1);
    expect(result.summary).toBeNull();
  });
});

describe("runAccuracy", () => {
  const baseConfig = {
    tasks: ["ifeval"], primaryTask: "ifeval", primaryMetric: "prompt_level_strict_acc",
    limit: 100, numFewshot: null, maxGenToks: 2048,
    applyChatTemplate: true, reasoning: true, seed: 42,
  };

  beforeEach(() => {
    spawnMock.mockReset();
    readFileSyncMock.mockReset();
    mkdirSyncMock.mockReset();
    existsSyncMock.mockReset();
    startProxyMock.mockReset();
    findFileMock.mockReset();
  });

  it("starts the strip proxy, runs uvx lm_eval against it, parses, and closes the proxy", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    const closeMock = vi.fn().mockResolvedValue(undefined);
    startProxyMock.mockResolvedValue({ url: "http://127.0.0.1:5555/v1", close: closeMock });
    findFileMock.mockReturnValue("/o/model/results_x.json");
    existsSyncMock.mockReturnValue(true);
    readFileSyncMock.mockReturnValue(JSON.stringify({
      results: { ifeval: { "prompt_level_strict_acc,none": 0.5 } },
    }));

    const promise = runAccuracy({
      runId: "run_acc", config: baseConfig, endpointV1Url: "http://10.0.0.1:8000/v1",
      servedModel: "m", outputDir: "/o", onLog: vi.fn(),
    });
    // runAccuracy awaits startReasoningProxy before spawning, so the child's
    // "close" listener isn't attached synchronously (unlike runBenchmark /
    // runToolEval, which spawn with no prior await). Wait for spawn to have
    // been called — i.e. for the listener to be attached — before emitting.
    await vi.waitFor(() => expect(spawnMock).toHaveBeenCalled());
    child.emit("close", 0);
    const r = await promise;

    expect(startProxyMock).toHaveBeenCalledWith("http://10.0.0.1:8000/v1");
    const [cmd, argv] = spawnMock.mock.calls[0];
    expect(cmd).toBe("uvx");
    expect(argv[0]).toBe("--from");
    // The extras are load-bearing, not cosmetic. `api` supplies tenacity/aiohttp for
    // lm-eval's local-chat-completions model — without it the run dies with
    // "Attempted to use an API model, but the required packages ['tenacity'] are not
    // installed". Only a real `uvx lm_eval` run caught this (2026-07-10); the old
    // /^lm-eval\[.+\]/ assertion was too loose to notice. `ifeval` = IFEval scorers,
    // `math` = sympy for MATH-hard.
    expect(argv[1]).toMatch(/^lm-eval\[[^\]]+\]/);
    for (const extra of ["api", "ifeval", "math"]) {
      expect(argv[1]).toContain(extra);
    }
    expect(argv[2]).toBe("lm_eval");
    expect(argv).toContain("base_url=http://127.0.0.1:5555/v1/chat/completions,model=m,num_concurrent=1,tokenized_requests=False");
    expect(r.exitCode).toBe(0);
    expect(r.summary?.primaryScore).toBeCloseTo(50, 5);
    expect(closeMock).toHaveBeenCalledTimes(1);
  });

  it("skips the proxy and targets the endpoint directly when reasoning is false", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    findFileMock.mockReturnValue(null);
    existsSyncMock.mockReturnValue(false);

    const promise = runAccuracy({
      runId: "run_acc2", config: { ...baseConfig, reasoning: false },
      endpointV1Url: "http://10.0.0.1:8000/v1", servedModel: "m", outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 0);
    const r = await promise;

    expect(startProxyMock).not.toHaveBeenCalled();
    const [, argv] = spawnMock.mock.calls[0];
    expect(argv).toContain("base_url=http://10.0.0.1:8000/v1/chat/completions,model=m,num_concurrent=1,tokenized_requests=False");
    expect(r.summary).toBeNull(); // no result file → no summary
  });

  it("returns the parser error when lm-eval exits 0 but results are unparseable", async () => {
    const child = makeFakeChild();
    spawnMock.mockReturnValue(child);
    findFileMock.mockReturnValue("/o/model/results_x.json");
    existsSyncMock.mockReturnValue(true);
    readFileSyncMock.mockReturnValue("not json");

    const promise = runAccuracy({
      runId: "run_acc_err", config: { ...baseConfig, reasoning: false },
      endpointV1Url: "http://10.0.0.1:8000/v1", servedModel: "m", outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 0);
    const r = await promise;
    expect(r.summary).toBeNull();
    expect(r.error).toMatch(/parse/i);
  });

  it("asks the remote wrapper to resolve lm-eval's nested result file", async () => {
    // Every other test in this file runs against the module instance loaded
    // at the top with BENCH_RUNNER forced to "local" (see the vi.hoisted()
    // block above), so its dispatch() always takes the local spawnTracked
    // branch regardless of runnerNodeId/invoke. To exercise the *remote*
    // branch — where dispatch() defers to runTrackedRemote() and never
    // touches node:child_process — reset the module registry and reimport
    // orchestrator.js with BENCH_RUNNER=remote, which is production's actual
    // default (unset env). Flip the env back to "local" immediately after
    // the fresh import captures it, so it doesn't leak into any later test.
    process.env.BENCH_RUNNER = "remote";
    vi.resetModules();
    const { runAccuracy: runAccuracyRemote } = await import("./orchestrator.js");
    process.env.BENCH_RUNNER = "local";

    const invoke = vi.fn(async (_nodeId: string, name: string, _input: unknown) => {
      if (name === "job.start") return { ok: true, data: {} };
      if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
      if (name === "job.status") return { ok: true, data: { kind: "exited", code: 0 } };
      if (name === "job.result") {
        return {
          ok: true,
          data: { raw: JSON.stringify({ results: { ifeval: { "prompt_level_strict_acc,none": 0.5 } } }) },
        };
      }
      return { ok: false, error: `unexpected cap ${name}` };
    });

    const r = await runAccuracyRemote({
      runId: "run_remote", config: { ...baseConfig, reasoning: false },
      endpointV1Url: "http://10.0.0.1:8000/v1", servedModel: "m", outputDir: "/o",
      onLog: vi.fn(), runnerNodeId: "eval-node-1", invoke,
    });

    expect(spawnMock).not.toHaveBeenCalled();
    const startCall = invoke.mock.calls.find(([, name]) => name === "job.start");
    expect(startCall?.[2]).toMatchObject({ resultGlob: "results_*.json" });
    expect(r.exitCode).toBe(0);
    expect(r.summary?.primaryScore).toBeCloseTo(50, 5);
  });
});
