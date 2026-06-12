import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

// ---------------------------------------------------------------------------
// Controllable child-process factory
// ---------------------------------------------------------------------------

function makeChild() {
  const h: Record<string, ((b: any) => void)[]> = {};
  return {
    pid: 4242,
    unref: vi.fn(),
    kill: vi.fn(),
    stdout: { on: (e: string, cb: any) => { (h["stdout." + e] ||= []).push(cb); } },
    stderr: { on: (e: string, cb: any) => { (h["stderr." + e] ||= []).push(cb); } },
    on: (e: string, cb: any) => { (h[e] ||= []).push(cb); },
    /** Emit a string on stdout.data listeners */
    __emit: (s: string) => (h["stdout.data"] || []).forEach((cb) => cb(Buffer.from(s))),
    /** Fire the registered exit handler(s) with the given code/signal */
    __exit: (code: number | null) => (h["exit"] || []).forEach((cb) => cb(code)),
  };
}

const children: ReturnType<typeof makeChild>[] = [];

const { spawnMock, execFileSyncMock, spawnSyncMock } = vi.hoisted(() => {
  const spawnMock = vi.fn();
  const execFileSyncMock = vi.fn(() => "");
  const spawnSyncMock = vi.fn(() => ({ stdout: "", stderr: "" }));
  return { spawnMock, execFileSyncMock, spawnSyncMock };
});

vi.mock("node:child_process", () => ({ spawn: spawnMock, execFileSync: execFileSyncMock, spawnSync: spawnSyncMock }));
vi.mock("./deployment-store.js", () => ({ saveDeployment: vi.fn(), removeDeployment: vi.fn() }));

import { launchSparkrun, stopSparkrun, isWorkloadRunning, writeInlineRecipe, removeInlineRecipe, inspectSparkrunContainer, snapshotContainerLogs, captureCrashedContainerLogs, resolveHfHome } from "./sparkrun.js";

beforeEach(() => {
  children.length = 0;
  spawnMock.mockClear();
  spawnMock.mockImplementation(() => {
    const c = makeChild();
    children.push(c);
    return c;
  });
  execFileSyncMock.mockReset();
  execFileSyncMock.mockReturnValue("");
  spawnSyncMock.mockReset();
  spawnSyncMock.mockReturnValue({ stdout: "", stderr: "" });
  // Spied so stopSparkrun's process-group kill never signals real processes.
  // Asserted on via vi.mocked(process.kill) in the launcher-kill tests.
  vi.spyOn(process, "kill").mockImplementation(() => true);
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// launchSparkrun — core behaviour
// ---------------------------------------------------------------------------

describe("launchSparkrun", () => {
  it("spawns uvx with sparkrun run argv for the given recipe + hosts", () => {
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, () => {});
    // The first spawn is the run-launcher
    const call = spawnMock.mock.calls[0] as [string, string[], ...unknown[]];
    const [cmd, argv] = call;
    expect(cmd).toBe("uvx");
    expect(argv).toContain("sparkrun");
    expect(argv).toContain("run");
    expect(argv).toContain("qwen3-1.7b-vllm");
    expect(argv).toContain("--no-follow");
  });

  it("pins HF_HOME onto shared storage when the agent env has none (no local-disk fill)", () => {
    const prev = process.env.HF_HOME;
    delete process.env.HF_HOME;
    try {
      launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, () => {});
      const opts = spawnMock.mock.calls[0][2] as { detached?: boolean; env?: Record<string, string> };
      expect(opts.detached).toBe(true);
      // SHARED_STORAGE defaults to /mnt/tank → HF cache lands on the NFS mount
      expect(opts.env?.HF_HOME).toBe("/mnt/tank/models");
    } finally {
      if (prev === undefined) delete process.env.HF_HOME; else process.env.HF_HOME = prev;
    }
  });

  it("respects an explicit HF_HOME already set in the agent env", () => {
    const prev = process.env.HF_HOME;
    process.env.HF_HOME = "/custom/hf";
    try {
      launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, () => {});
      const opts = spawnMock.mock.calls[0][2] as { env?: Record<string, string> };
      expect(opts.env?.HF_HOME).toBe("/custom/hf");
    } finally {
      if (prev === undefined) delete process.env.HF_HOME; else process.env.HF_HOME = prev;
    }
  });

  // -------------------------------------------------------------------------
  // Log-follower tests
  // -------------------------------------------------------------------------

  it("spawns a sparkrun logs follower after the cluster id appears", () => {
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, () => {});

    // children[0] is the run-launcher
    expect(spawnMock.mock.calls[0][1]).toContain("run");

    // Emit the cluster-id line on the launcher's stdout
    children[0].__emit("[5/6] Launching\nCluster:   sparkrun_abc123\n");

    // A second spawn (the follower) should have been issued
    const followerCall = spawnMock.mock.calls.find((c) => (c[1] as string[]).includes("logs"));
    expect(followerCall).toBeDefined();
    const followerArgv: string[] = followerCall![1];
    expect(followerArgv).toContain("logs");
    expect(followerArgv).toContain("sparkrun_abc123");
    expect(followerArgv).toContain("-H");
    expect(followerArgv).toContain("10.0.0.1");
    expect(followerArgv).toContain("--tail");
    expect(followerArgv).toContain("1000");
  });

  it("forwards follower output to onLog", () => {
    const onLog = vi.fn();
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, onLog, () => {});

    // Trigger cluster-id capture
    children[0].__emit("[5/6] Launching\nCluster:   sparkrun_abc123\n");

    // children[1] is the follower — emit a container log line from it
    children[1].__emit("Loading weights...\n");

    expect(onLog).toHaveBeenCalledWith("Loading weights...\n");
  });

  it("only spawns one follower even if more data arrives after cluster-id capture", () => {
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, () => {});
    children[0].__emit("Cluster:   sparkrun_abc123\n");
    children[0].__emit("some more output\n");

    const followerCalls = spawnMock.mock.calls.filter((c) => (c[1] as string[]).includes("logs"));
    expect(followerCalls).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// stopSparkrun
// ---------------------------------------------------------------------------

describe("stopSparkrun", () => {
  it("calls sparkrun stop with target, -H hosts, and --tp", () => {
    stopSparkrun("dep-1", "sparkrun_abc123", ["10.0.0.1", "10.0.0.2"], 2);
    const call = execFileSyncMock.mock.calls[0] as unknown as [string, string[], ...unknown[]];
    const [cmd, argv] = call;
    expect(cmd).toBe("uvx");
    expect(argv).toEqual(expect.arrayContaining(["stop", "sparkrun_abc123", "-H", "10.0.0.1,10.0.0.2", "--tp", "2"]));
  });

  it("kills the log follower when stop is called", () => {
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, () => {});

    // Trigger follower spawn
    children[0].__emit("Cluster:   sparkrun_abc123\n");
    expect(children[1]).toBeDefined();

    stopSparkrun("dep-1", "sparkrun_abc123", ["10.0.0.1"], 1);
    expect(children[1].kill).toHaveBeenCalled();
  });

  it("does not throw when there is no follower to kill", () => {
    // stopSparkrun called without a prior launch → no follower in map
    expect(() => stopSparkrun("dep-never-launched", "sparkrun_xyz", ["10.0.0.1"])).not.toThrow();
  });

  it("kills the in-flight launcher process group (negative pid) when stop is called mid-download", () => {
    // Launch but DO NOT emit a cluster id → still in the download/launch phase,
    // no container, launcher process group is what holds the running download.
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, () => {});
    stopSparkrun("dep-1", "qwen3-1.7b-vllm", ["10.0.0.1"], 1);
    // children[0] is the launcher (pid 4242) → group kill targets -4242
    expect(vi.mocked(process.kill)).toHaveBeenCalledWith(-4242, "SIGTERM");
  });

  it("does not call process.kill when no launcher is tracked", () => {
    stopSparkrun("dep-never-launched", "sparkrun_xyz", ["10.0.0.1"]);
    expect(vi.mocked(process.kill)).not.toHaveBeenCalled();
  });

  it("suppresses onExit for an intentionally-stopped launcher (no spurious 'failed')", () => {
    const onExit = vi.fn();
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, onExit);
    stopSparkrun("dep-1", "qwen3-1.7b-vllm", ["10.0.0.1"], 1);
    // The kill makes the launcher exit with a signal (code null) afterwards
    children[0].__exit(null);
    expect(onExit).not.toHaveBeenCalled();
  });

  it("still reports onExit when the launcher dies on its own (genuine launch failure)", () => {
    const onExit = vi.fn();
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, onExit);
    // No stop — launcher exits non-zero by itself
    children[0].__exit(1);
    expect(onExit).toHaveBeenCalledWith(1);
  });
});

// ---------------------------------------------------------------------------
// resolveHfHome
// ---------------------------------------------------------------------------

describe("resolveHfHome", () => {
  it("returns an explicit HF_HOME from the env unchanged", () => {
    expect(resolveHfHome({ HF_HOME: "/custom/hf" } as NodeJS.ProcessEnv)).toBe("/custom/hf");
  });

  it("falls back to ${SHARED_STORAGE}/models when HF_HOME is absent", () => {
    // SHARED_STORAGE defaults to /mnt/tank in this test env
    expect(resolveHfHome({} as NodeJS.ProcessEnv)).toBe("/mnt/tank/models");
  });
});

// ---------------------------------------------------------------------------
// isWorkloadRunning
// ---------------------------------------------------------------------------

describe("isWorkloadRunning", () => {
  it("true when check-job exits 0, false when it throws", () => {
    execFileSyncMock.mockReturnValueOnce("");
    expect(isWorkloadRunning("sparkrun_abc", ["10.0.0.1"])).toBe(true);
    execFileSyncMock.mockImplementationOnce(() => { throw new Error("exit 1"); });
    expect(isWorkloadRunning("sparkrun_abc", ["10.0.0.1"])).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// writeInlineRecipe / removeInlineRecipe
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// inspectSparkrunContainer
// ---------------------------------------------------------------------------

describe("inspectSparkrunContainer", () => {
  it("returns null when clusterId is undefined", () => {
    expect(inspectSparkrunContainer(undefined)).toBeNull();
  });

  it("returns null when docker ps finds no container", () => {
    // docker ps returns empty → containerNameFor returns null
    spawnSyncMock.mockReturnValue({ stdout: "", stderr: "" });
    expect(inspectSparkrunContainer("sparkrun_abc123")).toBeNull();
  });

  it("parses state and restartCount from docker inspect output", () => {
    // First call: docker ps to find name; second call: docker inspect
    spawnSyncMock
      .mockReturnValueOnce({ stdout: "sparkrun_abc123_solo\n", stderr: "" })   // ps
      .mockReturnValueOnce({ stdout: "exited 5\n", stderr: "" });              // inspect
    const result = inspectSparkrunContainer("sparkrun_abc123");
    expect(result).not.toBeNull();
    expect(result!.name).toBe("sparkrun_abc123_solo");
    expect(result!.state).toBe("exited");
    expect(result!.restartCount).toBe(5);
  });

  it("parses restarting state with restartCount 3", () => {
    spawnSyncMock
      .mockReturnValueOnce({ stdout: "sparkrun_abc456_solo\n", stderr: "" })
      .mockReturnValueOnce({ stdout: "restarting 3\n", stderr: "" });
    const result = inspectSparkrunContainer("sparkrun_abc456");
    expect(result!.state).toBe("restarting");
    expect(result!.restartCount).toBe(3);
  });

  it("returns null when docker inspect returns empty output", () => {
    spawnSyncMock
      .mockReturnValueOnce({ stdout: "sparkrun_abc789_solo\n", stderr: "" })
      .mockReturnValueOnce({ stdout: "", stderr: "" });
    expect(inspectSparkrunContainer("sparkrun_abc789")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// snapshotContainerLogs
// ---------------------------------------------------------------------------

describe("snapshotContainerLogs", () => {
  it("returns empty string when clusterId is undefined", () => {
    expect(snapshotContainerLogs(undefined)).toBe("");
  });

  it("returns empty string when no container found", () => {
    spawnSyncMock.mockReturnValue({ stdout: "", stderr: "" });
    expect(snapshotContainerLogs("sparkrun_abc123")).toBe("");
  });

  it("captures full stdout+stderr, stderr (errors) first so the root crash leads", () => {
    spawnSyncMock
      .mockReturnValueOnce({ stdout: "sparkrun_abc123_solo\n", stderr: "" })   // ps
      .mockReturnValueOnce({                                                    // logs (no --tail)
        stdout: "Starting vllm...\n",
        stderr: "vllm serve: error: argument --compilation-config: Invalid JSON\n",
      });
    const result = snapshotContainerLogs("sparkrun_abc123");
    expect(result).toContain("Starting vllm...");
    expect(result).toContain("vllm serve: error: argument --compilation-config: Invalid JSON");
    // stderr (the error) must come before stdout so firstErrorLine finds the root
    expect(result.indexOf("Invalid JSON")).toBeLessThan(result.indexOf("Starting vllm"));
    // full capture: docker logs called WITHOUT --tail
    const logsCall = spawnSyncMock.mock.calls.find((c: any) => c[1]?.includes("logs")) as any[] | undefined;
    expect(logsCall).toBeTruthy();
    expect(logsCall?.[1]).not.toContain("--tail");
  });

  it("trims the combined output", () => {
    spawnSyncMock
      .mockReturnValueOnce({ stdout: "sparkrun_abc123_solo\n", stderr: "" })
      .mockReturnValueOnce({ stdout: "  log line  \n  ", stderr: "\n  " });
    const result = snapshotContainerLogs("sparkrun_abc123");
    expect(result).toBe("log line");
  });
});

// ---------------------------------------------------------------------------
// captureCrashedContainerLogs
// ---------------------------------------------------------------------------

describe("captureCrashedContainerLogs", () => {
  it("returns empty string when clusterId is undefined", () => {
    expect(captureCrashedContainerLogs(undefined)).toBe("");
    // spawnSync must not have been called at all
    expect(spawnSyncMock).not.toHaveBeenCalled();
  });

  it("returns empty string when no container is found for the clusterId", () => {
    // docker ps returns nothing → containerNameFor returns null → early return
    spawnSyncMock.mockReturnValue({ stdout: "", stderr: "" });
    expect(captureCrashedContainerLogs("sparkrun_abc123")).toBe("");
  });

  it("calls docker stop before reading logs (stops the restart loop first)", () => {
    // Three spawnSync calls in order: ps (find name), stop, ps again (inside snapshotContainerLogs), logs
    spawnSyncMock
      .mockReturnValueOnce({ stdout: "sparkrun_abc123_solo\n", stderr: "" })  // ps for containerNameFor (stop phase)
      .mockReturnValueOnce({ stdout: "", stderr: "" })                         // docker stop
      .mockReturnValueOnce({ stdout: "sparkrun_abc123_solo\n", stderr: "" })  // ps for containerNameFor (logs phase)
      .mockReturnValueOnce({ stdout: "startup output\n", stderr: "root crash error\n" }); // docker logs

    const result = captureCrashedContainerLogs("sparkrun_abc123");

    // Verify docker stop was called with the right arguments
    const stopCall = spawnSyncMock.mock.calls.find(
      (c: any) => Array.isArray(c[1]) && c[1].includes("stop"),
    ) as any[] | undefined;
    expect(stopCall).toBeDefined();
    expect(stopCall?.[1]).toContain("-t");
    expect(stopCall?.[1]).toContain("3");
    expect(stopCall?.[1]).toContain("sparkrun_abc123_solo");

    // The returned log must include content from the subsequent docker logs call
    expect(result).toContain("root crash error");
  });

  it("docker stop is called BEFORE docker logs (stop precedes log read)", () => {
    spawnSyncMock
      .mockReturnValueOnce({ stdout: "sparkrun_abc123_solo\n", stderr: "" })  // ps (stop phase)
      .mockReturnValueOnce({ stdout: "", stderr: "" })                         // stop
      .mockReturnValueOnce({ stdout: "sparkrun_abc123_solo\n", stderr: "" })  // ps (logs phase)
      .mockReturnValueOnce({ stdout: "output\n", stderr: "err\n" });           // logs

    captureCrashedContainerLogs("sparkrun_abc123");

    const calls = spawnSyncMock.mock.calls as any[][];
    const stopIdx = calls.findIndex((c) => c[1]?.includes("stop"));
    const logsIdx = calls.findIndex((c) => c[1]?.includes("logs"));
    expect(stopIdx).toBeGreaterThanOrEqual(0);
    expect(logsIdx).toBeGreaterThanOrEqual(0);
    expect(stopIdx).toBeLessThan(logsIdx);
  });
});

// ---------------------------------------------------------------------------
// writeInlineRecipe / removeInlineRecipe
// ---------------------------------------------------------------------------

describe("writeInlineRecipe / removeInlineRecipe", () => {
  it("creates the YAML file under <dir>/<deploymentId>.yaml and returns its path", () => {
    const dir = mkdtempSync(join(tmpdir(), "sparkrun-test-"));
    const yaml = "model: meta-llama/Llama-3.1-8B-Instruct\n";
    const result = writeInlineRecipe("dep-xyz", yaml, dir);
    expect(result.endsWith("dep-xyz.yaml")).toBe(true);
    expect(readFileSync(result, "utf-8")).toBe(yaml);
  });

  it("removeInlineRecipe deletes the file (no-op when already absent)", () => {
    const dir = mkdtempSync(join(tmpdir(), "sparkrun-test-"));
    writeInlineRecipe("dep-abc", "model: foo\n", dir);
    removeInlineRecipe("dep-abc", dir);
    expect(existsSync(join(dir, "dep-abc.yaml"))).toBe(false);
    // second call must not throw
    expect(() => removeInlineRecipe("dep-abc", dir)).not.toThrow();
  });

  it("creates missing parent directories", () => {
    const base = mkdtempSync(join(tmpdir(), "sparkrun-test-"));
    const nested = join(base, "a", "b", "c");
    const result = writeInlineRecipe("dep-nested", "model: bar\n", nested);
    expect(existsSync(result)).toBe(true);
  });
});
