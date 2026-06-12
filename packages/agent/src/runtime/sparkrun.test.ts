import { describe, it, expect, vi, beforeEach } from "vitest";
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const { spawnMock, execFileSyncMock } = vi.hoisted(() => {
  const spawnMock = vi.fn(() => ({
    pid: 1234, stdout: { on: vi.fn() }, stderr: { on: vi.fn() }, on: vi.fn(), unref: vi.fn(),
  }));
  const execFileSyncMock = vi.fn(() => "");
  return { spawnMock, execFileSyncMock };
});

vi.mock("node:child_process", () => ({ spawn: spawnMock, execFileSync: execFileSyncMock }));
vi.mock("./deployment-store.js", () => ({ saveDeployment: vi.fn(), removeDeployment: vi.fn() }));

import { launchSparkrun, stopSparkrun, isWorkloadRunning, writeInlineRecipe, removeInlineRecipe } from "./sparkrun.js";

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
