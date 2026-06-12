import { describe, it, expect, vi, beforeEach } from "vitest";
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

// ---------------------------------------------------------------------------
// Controllable child-process factory
// ---------------------------------------------------------------------------

function makeChild() {
  const h: Record<string, ((b: any) => void)[]> = {};
  return {
    pid: 1,
    unref: vi.fn(),
    kill: vi.fn(),
    stdout: { on: (e: string, cb: any) => { (h["stdout." + e] ||= []).push(cb); } },
    stderr: { on: (e: string, cb: any) => { (h["stderr." + e] ||= []).push(cb); } },
    on: (e: string, cb: any) => { (h[e] ||= []).push(cb); },
    /** Emit a string on stdout.data listeners */
    __emit: (s: string) => (h["stdout.data"] || []).forEach((cb) => cb(Buffer.from(s))),
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

import { launchSparkrun, stopSparkrun, isWorkloadRunning, writeInlineRecipe, removeInlineRecipe, inspectSparkrunContainer, snapshotContainerLogs } from "./sparkrun.js";

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
