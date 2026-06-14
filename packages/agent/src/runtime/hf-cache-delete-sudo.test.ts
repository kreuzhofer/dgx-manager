import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock fs so we can force rmSync to fail, and child_process so we can observe
// the sudo fallback without actually shelling out. vi.hoisted so the mock
// factories (hoisted above imports) can reference these.
const { rmSync, execFileSync } = vi.hoisted(() => ({ rmSync: vi.fn(), execFileSync: vi.fn() }));
vi.mock("node:fs", async (orig) => {
  const actual = (await orig()) as Record<string, unknown>;
  return { ...actual, existsSync: () => true, rmSync };
});
vi.mock("node:child_process", () => ({ execFileSync }));

import { deleteCachedRepo } from "./hf-cache.js";

describe("deleteCachedRepo — root-owned cache fallback", () => {
  beforeEach(() => {
    rmSync.mockReset();
    execFileSync.mockReset();
  });

  /** Repos pulled by a root-running container are root-owned on the (NFS) cache;
   *  the agent's unprivileged rmSync then fails EACCES. We must escalate with
   *  `sudo rm -rf` on the already-validated target path. */
  it("falls back to `sudo rm -rf` on EACCES", () => {
    rmSync.mockImplementation(() => {
      const e = new Error("permission denied") as NodeJS.ErrnoException;
      e.code = "EACCES";
      throw e;
    });
    deleteCachedRepo("/tmp/hf", "model", "google/gemma-4-26B-A4B");
    expect(execFileSync).toHaveBeenCalledTimes(1);
    const [cmd, args] = execFileSync.mock.calls[0];
    expect(cmd).toBe("sudo");
    expect(args.slice(0, 3)).toEqual(["rm", "-rf", "--"]);
    // target must be the validated hub-relative repo dir, never raw repoId
    expect(args[3]).toContain("hub/models--google--gemma-4-26B-A4B");
  });

  it("also escalates on EPERM", () => {
    rmSync.mockImplementation(() => {
      const e = new Error("op not permitted") as NodeJS.ErrnoException;
      e.code = "EPERM";
      throw e;
    });
    deleteCachedRepo("/tmp/hf", "model", "Qwen/Qwen3-8B");
    expect(execFileSync).toHaveBeenCalledTimes(1);
  });

  /** A non-permission failure (e.g. EIO) is a real error — never silently
   *  escalate to sudo; rethrow so the caller sees it. */
  it("rethrows non-permission errors without sudo", () => {
    rmSync.mockImplementation(() => {
      const e = new Error("i/o error") as NodeJS.ErrnoException;
      e.code = "EIO";
      throw e;
    });
    expect(() => deleteCachedRepo("/tmp/hf", "model", "Qwen/Qwen3-8B")).toThrow(/i\/o error/);
    expect(execFileSync).not.toHaveBeenCalled();
  });

  it("does not shell out when rmSync succeeds", () => {
    rmSync.mockImplementation(() => undefined);
    deleteCachedRepo("/tmp/hf", "model", "Qwen/Qwen3-8B");
    expect(execFileSync).not.toHaveBeenCalled();
  });
});
