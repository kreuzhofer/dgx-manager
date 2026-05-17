import { describe, expect, it, vi, beforeEach } from "vitest";
import { EventEmitter } from "node:events";

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

import { runBenchmark, cancelBenchmark } from "./orchestrator.js";

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
        rows: [{
          op: "tg", pp: 1, tg: 2, depth: 0, concurrency: 1,
          "t/s": 50.5, "peak t/s": 60, "ttfr (ms)": 100,
          "est_ppt (ms)": 50, "e2e_ttft (ms)": 150,
          "t/s_stdev": 1, "ttfr_stdev": 2,
        }],
      }),
    );
    const promise = runBenchmark({
      runId: "r2", args: [], outputDir: "/o", onLog: vi.fn(),
    });
    child.emit("close", 0);
    const r = await promise;
    expect(r.results).toHaveLength(1);
    expect(r.results[0].tps).toBe(50.5);
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

  it("cancelBenchmark kills the process group of an in-flight run", async () => {
    const child = makeFakeChild(9999);
    spawnMock.mockReturnValue(child);
    const promise = runBenchmark({
      runId: "cancelme", args: [], outputDir: "/o", onLog: vi.fn(),
    });
    expect(cancelBenchmark("cancelme")).toBe(true);
    expect(child.kill).toHaveBeenCalledWith("SIGTERM");
    child.emit("close", 143);
    await promise;
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
