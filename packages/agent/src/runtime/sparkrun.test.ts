import { describe, it, expect, vi, beforeEach } from "vitest";

const { spawnMock, execFileSyncMock } = vi.hoisted(() => {
  const spawnMock = vi.fn(() => ({
    pid: 1234, stdout: { on: vi.fn() }, stderr: { on: vi.fn() }, on: vi.fn(), unref: vi.fn(),
  }));
  const execFileSyncMock = vi.fn(() => "");
  return { spawnMock, execFileSyncMock };
});

vi.mock("node:child_process", () => ({ spawn: spawnMock, execFileSync: execFileSyncMock }));
vi.mock("./deployment-store.js", () => ({ saveDeployment: vi.fn(), removeDeployment: vi.fn() }));

import { launchSparkrun, stopSparkrun, isWorkloadRunning } from "./sparkrun.js";

beforeEach(() => { spawnMock.mockClear(); execFileSyncMock.mockReset(); execFileSyncMock.mockReturnValue(""); });

describe("launchSparkrun", () => {
  it("spawns uvx with sparkrun run argv for the given recipe + hosts", () => {
    launchSparkrun("dep-1", "qwen3-1.7b-vllm", { hosts: ["10.0.0.1"], port: 8000 }, () => {}, () => {});
    expect(spawnMock).toHaveBeenCalledOnce();
    const call = spawnMock.mock.calls[0] as unknown as [string, string[], ...unknown[]];
    const [cmd, argv] = call;
    expect(cmd).toBe("uvx");
    expect(argv).toContain("sparkrun");
    expect(argv).toContain("run");
    expect(argv).toContain("qwen3-1.7b-vllm");
    expect(argv).toContain("--no-follow");
  });
});

describe("stopSparkrun", () => {
  it("calls sparkrun stop with target, -H hosts, and --tp", () => {
    stopSparkrun("dep-1", "sparkrun_abc123", ["10.0.0.1", "10.0.0.2"], 2);
    const call = execFileSyncMock.mock.calls[0] as unknown as [string, string[], ...unknown[]];
    const [cmd, argv] = call;
    expect(cmd).toBe("uvx");
    expect(argv).toEqual(expect.arrayContaining(["stop", "sparkrun_abc123", "-H", "10.0.0.1,10.0.0.2", "--tp", "2"]));
  });
});

describe("isWorkloadRunning", () => {
  it("true when check-job exits 0, false when it throws", () => {
    execFileSyncMock.mockReturnValueOnce("");
    expect(isWorkloadRunning("sparkrun_abc", ["10.0.0.1"])).toBe(true);
    execFileSyncMock.mockImplementationOnce(() => { throw new Error("exit 1"); });
    expect(isWorkloadRunning("sparkrun_abc", ["10.0.0.1"])).toBe(false);
  });
});
