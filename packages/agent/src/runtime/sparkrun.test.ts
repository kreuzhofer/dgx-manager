import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fc, it as fcIt } from "@fast-check/vitest";
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

const { spawnMock, execFileSyncMock, spawnSyncMock, execCaptureMock } = vi.hoisted(() => {
  const spawnMock = vi.fn();
  const execFileSyncMock = vi.fn(() => "");
  const spawnSyncMock = vi.fn(() => ({ stdout: "", stderr: "" }));
  const execCaptureMock = vi.fn();
  return { spawnMock, execFileSyncMock, spawnSyncMock, execCaptureMock };
});

/** Build an ExecCapture result; defaults are "exited 0, said nothing". */
function cap(o: { status?: number | null; stdout?: string; stderr?: string; error?: Error }) {
  // NB: `?? 0` would turn an explicit `status: null` (a timeout) into a clean
  // exit — the exact confusion this ticket is about. Only `undefined` defaults.
  return {
    status: o.status === undefined ? 0 : o.status,
    stdout: o.stdout ?? "", stderr: o.stderr ?? "", error: o.error,
  };
}

vi.mock("node:child_process", () => ({ spawn: spawnMock, execFileSync: execFileSyncMock, spawnSync: spawnSyncMock }));
vi.mock("./exec-capture.js", () => ({ execCapture: execCaptureMock }));
vi.mock("./deployment-store.js", () => ({ saveDeployment: vi.fn(), removeDeployment: vi.fn() }));

import { launchSparkrun, stopSparkrun, isWorkloadRunning, writeInlineRecipe, removeInlineRecipe, inspectSparkrunContainer, snapshotContainerLogs, captureCrashedContainerLogs, resolveHfHome, isHfHomeExplicit } from "./sparkrun.js";

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
  execCaptureMock.mockReset();
  execCaptureMock.mockResolvedValue(cap({}));
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
  it("calls sparkrun stop with target, -H hosts, and --tp", async () => {
    await stopSparkrun("dep-1", "sparkrun_abc123", ["10.0.0.1", "10.0.0.2"], 2);
    const call = execCaptureMock.mock.calls[0] as unknown as [string, string[], ...unknown[]];
    const [cmd, argv] = call;
    expect(cmd).toBe("uvx");
    expect(argv).toEqual(expect.arrayContaining(["stop", "sparkrun_abc123", "-H", "10.0.0.1,10.0.0.2", "--tp", "2"]));
  });

  // The old execFileSync threw when `sparkrun stop` failed, and the undeploy
  // handler logs "stop error (continuing)" off that throw. The async form must
  // keep throwing rather than reporting a clean teardown that never happened.
  it("throws when sparkrun stop exits non-zero", async () => {
    execCaptureMock.mockResolvedValueOnce(cap({ status: 1, stderr: "no such cluster" }));
    await expect(stopSparkrun("dep-1", "sparkrun_abc123", ["10.0.0.1"], 1))
      .rejects.toThrow("no such cluster");
  });

  // ...and the local record still has to go, or the health tick keeps reporting
  // a deployment we have already disowned.
  it("drops the local record even when sparkrun stop fails", async () => {
    const { removeDeployment } = await import("./deployment-store.js");
    execCaptureMock.mockResolvedValueOnce(cap({ status: null, error: new Error("uvx timed out") }));
    await expect(stopSparkrun("dep-1", "sparkrun_abc123", ["10.0.0.1"], 1)).rejects.toThrow();
    expect(removeDeployment).toHaveBeenCalledWith("dep-1");
  });

  it("kills the log follower when stop is called", async () => {
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, () => {});

    // Trigger follower spawn
    children[0].__emit("Cluster:   sparkrun_abc123\n");
    expect(children[1]).toBeDefined();

    await stopSparkrun("dep-1", "sparkrun_abc123", ["10.0.0.1"], 1);
    expect(children[1].kill).toHaveBeenCalled();
  });

  it("does not throw when there is no follower to kill", async () => {
    // stopSparkrun called without a prior launch → no follower in map
    await expect(stopSparkrun("dep-never-launched", "sparkrun_xyz", ["10.0.0.1"])).resolves.toBeUndefined();
  });

  it("kills the in-flight launcher process group (negative pid) when stop is called mid-download", async () => {
    // Launch but DO NOT emit a cluster id → still in the download/launch phase,
    // no container, launcher process group is what holds the running download.
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, () => {});
    await stopSparkrun("dep-1", "qwen3-1.7b-vllm", ["10.0.0.1"], 1);
    // children[0] is the launcher (pid 4242) → group kill targets -4242
    expect(vi.mocked(process.kill)).toHaveBeenCalledWith(-4242, "SIGTERM");
  });

  it("does not call process.kill when no launcher is tracked", async () => {
    await stopSparkrun("dep-never-launched", "sparkrun_xyz", ["10.0.0.1"]);
    expect(vi.mocked(process.kill)).not.toHaveBeenCalled();
  });

  it("suppresses onExit for an intentionally-stopped launcher (no spurious 'failed')", async () => {
    const onExit = vi.fn();
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, onExit);
    await stopSparkrun("dep-1", "qwen3-1.7b-vllm", ["10.0.0.1"], 1);
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

describe("isHfHomeExplicit", () => {
  it("is true when the env sets HF_HOME", () => {
    expect(isHfHomeExplicit({ HF_HOME: "/custom/hf" } as NodeJS.ProcessEnv)).toBe(true);
  });

  it("is false when HF_HOME is absent and the shared-storage default applies", () => {
    expect(isHfHomeExplicit({} as NodeJS.ProcessEnv)).toBe(false);
  });

  /** Invariant: "explicit" means exactly "resolveHfHome did NOT fall back to
   *  the shared-storage default". Any HF_HOME the env supplies is explicit —
   *  including one that happens to equal the default path. */
  fcIt.prop([fc.string()])("treats any env-supplied HF_HOME as explicit", (hfHome) => {
    const env = { HF_HOME: hfHome } as NodeJS.ProcessEnv;
    expect(isHfHomeExplicit(env)).toBe(true);
    expect(resolveHfHome(env)).toBe(hfHome);
  });
});

// ---------------------------------------------------------------------------
// isWorkloadRunning
// ---------------------------------------------------------------------------

describe("isWorkloadRunning", () => {
  it("true when check-job exits 0, false on a non-zero exit", async () => {
    execCaptureMock.mockResolvedValueOnce(cap({ status: 0 }));
    await expect(isWorkloadRunning("sparkrun_abc", ["10.0.0.1"])).resolves.toBe(true);
    execCaptureMock.mockResolvedValueOnce(cap({ status: 1, stderr: "no such job" }));
    await expect(isWorkloadRunning("sparkrun_abc", ["10.0.0.1"])).resolves.toBe(false);
  });

  // The old execFileSync threw on timeout and the catch returned false. The
  // async form reports a timeout as status null; it must still read as "not
  // running" rather than rejecting and killing the whole health tick.
  it("false when the check times out", async () => {
    execCaptureMock.mockResolvedValueOnce(
      cap({ status: null, error: new Error("uvx timed out after 30000ms") }),
    );
    await expect(isWorkloadRunning("sparkrun_abc", ["10.0.0.1"])).resolves.toBe(false);
  });
});

// ---------------------------------------------------------------------------
// inspectSparkrunContainer
// ---------------------------------------------------------------------------

describe("inspectSparkrunContainer", () => {
  it("returns null when clusterId is undefined", async () => {
    await expect(inspectSparkrunContainer(undefined)).resolves.toBeNull();
    expect(execCaptureMock).not.toHaveBeenCalled();
  });

  it("returns null when docker ps finds no container", async () => {
    execCaptureMock.mockResolvedValue(cap({}));
    await expect(inspectSparkrunContainer("sparkrun_abc123")).resolves.toBeNull();
  });

  it("parses state and restartCount from docker inspect output", async () => {
    execCaptureMock
      .mockResolvedValueOnce(cap({ stdout: "sparkrun_abc123_solo\n" }))  // ps
      .mockResolvedValueOnce(cap({ stdout: "exited 5\n" }));             // inspect
    const result = await inspectSparkrunContainer("sparkrun_abc123");
    expect(result).not.toBeNull();
    expect(result!.name).toBe("sparkrun_abc123_solo");
    expect(result!.state).toBe("exited");
    expect(result!.restartCount).toBe(5);
  });

  it("parses restarting state with restartCount 3", async () => {
    execCaptureMock
      .mockResolvedValueOnce(cap({ stdout: "sparkrun_abc456_solo\n" }))
      .mockResolvedValueOnce(cap({ stdout: "restarting 3\n" }));
    const result = await inspectSparkrunContainer("sparkrun_abc456");
    expect(result!.state).toBe("restarting");
    expect(result!.restartCount).toBe(3);
  });

  it("returns null when docker inspect returns empty output", async () => {
    execCaptureMock
      .mockResolvedValueOnce(cap({ stdout: "sparkrun_abc789_solo\n" }))
      .mockResolvedValueOnce(cap({}));
    await expect(inspectSparkrunContainer("sparkrun_abc789")).resolves.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// snapshotContainerLogs
// ---------------------------------------------------------------------------

describe("snapshotContainerLogs", () => {
  it("returns empty string when clusterId is undefined", async () => {
    await expect(snapshotContainerLogs(undefined)).resolves.toBe("");
  });

  it("returns empty string when no container found", async () => {
    execCaptureMock.mockResolvedValue(cap({}));
    await expect(snapshotContainerLogs("sparkrun_abc123")).resolves.toBe("");
  });

  it("captures full stdout+stderr, stderr (errors) first so the root crash leads", async () => {
    execCaptureMock
      .mockResolvedValueOnce(cap({ stdout: "sparkrun_abc123_solo\n" }))   // ps
      .mockResolvedValueOnce(cap({                                          // logs (no --tail)
        stdout: "Starting vllm...\n",
        stderr: "vllm serve: error: argument --compilation-config: Invalid JSON\n",
      }));
    const result = await snapshotContainerLogs("sparkrun_abc123");
    expect(result).toContain("Starting vllm...");
    expect(result).toContain("vllm serve: error: argument --compilation-config: Invalid JSON");
    // stderr (the error) must come before stdout so firstErrorLine finds the root
    expect(result.indexOf("Invalid JSON")).toBeLessThan(result.indexOf("Starting vllm"));
    // full capture: docker logs called WITHOUT --tail
    const logsCall = execCaptureMock.mock.calls.find((c: any) => c[1]?.includes("logs")) as any[] | undefined;
    expect(logsCall).toBeTruthy();
    expect(logsCall?.[1]).not.toContain("--tail");
  });

  it("trims the combined output", async () => {
    execCaptureMock
      .mockResolvedValueOnce(cap({ stdout: "sparkrun_abc123_solo\n" }))
      .mockResolvedValueOnce(cap({ stdout: "  log line  \n  ", stderr: "\n  " }));
    await expect(snapshotContainerLogs("sparkrun_abc123")).resolves.toBe("log line");
  });
});

// ---------------------------------------------------------------------------
// captureCrashedContainerLogs
// ---------------------------------------------------------------------------

describe("captureCrashedContainerLogs", () => {
  it("returns empty string when clusterId is undefined", async () => {
    await expect(captureCrashedContainerLogs(undefined)).resolves.toBe("");
    // nothing may be shelled out to at all
    expect(execCaptureMock).not.toHaveBeenCalled();
  });

  it("returns empty string when no container is found for the clusterId", async () => {
    // docker ps returns nothing → containerNameFor returns null → early return
    execCaptureMock.mockResolvedValue(cap({}));
    await expect(captureCrashedContainerLogs("sparkrun_abc123")).resolves.toBe("");
  });

  it("calls docker stop before reading logs (stops the restart loop first)", async () => {
    // Four calls in order: ps (find name), stop, ps again (inside snapshotContainerLogs), logs
    execCaptureMock
      .mockResolvedValueOnce(cap({ stdout: "sparkrun_abc123_solo\n" }))  // ps for containerNameFor (stop phase)
      .mockResolvedValueOnce(cap({}))                                      // docker stop
      .mockResolvedValueOnce(cap({ stdout: "sparkrun_abc123_solo\n" }))  // ps for containerNameFor (logs phase)
      .mockResolvedValueOnce(cap({ stdout: "startup output\n", stderr: "root crash error\n" })); // docker logs

    const result = await captureCrashedContainerLogs("sparkrun_abc123");

    // Verify docker stop was called with the right arguments
    const stopCall = execCaptureMock.mock.calls.find(
      (c: any) => Array.isArray(c[1]) && c[1].includes("stop"),
    ) as any[] | undefined;
    expect(stopCall).toBeDefined();
    expect(stopCall?.[1]).toContain("-t");
    expect(stopCall?.[1]).toContain("3");
    expect(stopCall?.[1]).toContain("sparkrun_abc123_solo");

    // The returned log must include content from the subsequent docker logs call
    expect(result).toContain("root crash error");
  });

  it("docker stop is called BEFORE docker logs (stop precedes log read)", async () => {
    execCaptureMock
      .mockResolvedValueOnce(cap({ stdout: "sparkrun_abc123_solo\n" }))  // ps (stop phase)
      .mockResolvedValueOnce(cap({}))                                      // stop
      .mockResolvedValueOnce(cap({ stdout: "sparkrun_abc123_solo\n" }))  // ps (logs phase)
      .mockResolvedValueOnce(cap({ stdout: "output\n", stderr: "err\n" })); // logs

    await captureCrashedContainerLogs("sparkrun_abc123");

    const calls = execCaptureMock.mock.calls as any[][];
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
