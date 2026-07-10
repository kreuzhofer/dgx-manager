import { describe, expect, it, vi } from "vitest";
import { EventEmitter } from "events";
import { makeJobCaps, type JobCapDeps } from "./job-cap.js";
import type { Capability } from "./registry.js";

const noopCtx = { emitChunk: () => {} };

function capsByName(caps: Capability[]): Record<string, Capability> {
  return Object.fromEntries(caps.map((c) => [c.name, c]));
}

/** A spawn stub that records argv and reports the given exit code + stdout. */
function fakeSpawn(result: { code: number; stdout?: string; stderr?: string }, calls: string[][]) {
  return ((cmd: string, args: string[]) => {
    calls.push([cmd, ...args]);
    const child: any = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.kill = () => {};
    setTimeout(() => {
      if (result.stdout) child.stdout.emit("data", Buffer.from(result.stdout));
      if (result.stderr) child.stderr.emit("data", Buffer.from(result.stderr));
      child.emit("close", result.code);
    }, 0);
    return child;
  }) as unknown as JobCapDeps["spawnFn"];
}

function deps(over: Partial<JobCapDeps> = {}): JobCapDeps {
  return {
    home: "/home/daniel",
    user: "daniel",
    spawnFn: fakeSpawn({ code: 0 }, []),
    writeFile: vi.fn(),
    mkdir: vi.fn(),
    readFileSlice: vi.fn(() => ({ chunk: "", size: 0 })),
    readTextFile: vi.fn(() => null),
    now: () => 1_000_000_000_000,
    listJobDirs: () => [],
    removeDir: vi.fn(),
    ...over,
  };
}

describe("job.start", () => {
  it("writes the wrapper script then launches a transient unit", async () => {
    const calls: string[][] = [];
    const writeFile = vi.fn();
    const caps = capsByName(makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 0 }, calls), writeFile })));

    const out = (await caps["job.start"].handle(
      { runId: "r1", argv: ["uvx", "lm_eval"], resultGlob: "results_*.json" },
      noopCtx,
    )) as { unit: string; jobDir: string };

    expect(out.unit).toBe("dgxbench-r1");
    expect(out.jobDir).toBe("/home/daniel/.dgx-agent/jobs/r1");
    // Script written before spawn.
    expect(writeFile).toHaveBeenCalled();
    const [scriptPath, script] = writeFile.mock.calls[0];
    expect(scriptPath).toBe("/home/daniel/.dgx-agent/jobs/r1/cmd.sh");
    expect(script).toContain("'lm_eval'");
    // Launched via sudo -n systemd-run.
    expect(calls[0].slice(0, 3)).toEqual(["sudo", "-n", "systemd-run"]);
    expect(calls[0]).toContain("--unit=dgxbench-r1");
  });

  it("rejects an unsafe runId before touching the shell", async () => {
    const calls: string[][] = [];
    const caps = capsByName(makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 0 }, calls) })));
    await expect(caps["job.start"].handle({ runId: "a;id", argv: ["x"] }, noopCtx)).rejects.toThrow(/unsafe/i);
    expect(calls).toHaveLength(0);
  });

  it("fails when systemd-run cannot start the unit", async () => {
    const caps = capsByName(
      makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 1, stderr: "sudo: a password is required" }, []) })),
    );
    await expect(caps["job.start"].handle({ runId: "r1", argv: ["x"] }, noopCtx)).rejects.toThrow(/password is required/);
  });
});

describe("job.status", () => {
  const showActive = "LoadState=loaded\nActiveState=active\nExecMainStatus=0\n";
  const showGone = "LoadState=not-found\nActiveState=inactive\nExecMainStatus=0\n";

  it("reports a live unit as active", async () => {
    const caps = capsByName(makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 0, stdout: showActive }, []) })));
    expect(await caps["job.status"].handle({ runId: "r1" }, noopCtx)).toEqual({ kind: "active" });
  });

  // A unit garbage-collected by systemd, but the wrapper left an exit file: the
  // job DID finish, and its code is authoritative.
  it("prefers the exit file when the unit is gone", async () => {
    const caps = capsByName(
      makeJobCaps(deps({
        spawnFn: fakeSpawn({ code: 0, stdout: showGone }, []),
        readTextFile: (p: string) => (p.endsWith("/exit") ? "0" : null),
      })),
    );
    expect(await caps["job.status"].handle({ runId: "r1" }, noopCtx)).toEqual({ kind: "exited", code: 0 });
  });

  it("reports missing only when the unit is gone AND there is no exit file", async () => {
    const caps = capsByName(
      makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 0, stdout: showGone }, []), readTextFile: () => null })),
    );
    expect(await caps["job.status"].handle({ runId: "r1" }, noopCtx)).toEqual({ kind: "missing" });
  });

  /**
   * THE invariant. systemctl could not answer. The job may well be running. We
   * must say "unknown" — never "missing", never "exited". The manager skips the
   * tick; anything else would kill an 80-minute eval on one slow poll.
   */
  it("returns unknown when systemctl fails, even if an exit file is absent", async () => {
    const caps = capsByName(
      makeJobCaps(deps({
        spawnFn: fakeSpawn({ code: 1, stderr: "Failed to connect to bus" }, []),
        readTextFile: () => null,
      })),
    );
    const r = (await caps["job.status"].handle({ runId: "r1" }, noopCtx)) as { kind: string };
    expect(r.kind).toBe("unknown");
  });
});

describe("job.logs", () => {
  it("returns the tail from the caller's offset", async () => {
    const caps = capsByName(
      makeJobCaps(deps({ readFileSlice: (_p, from, to) => ({ chunk: "world".slice(0, to - from), size: 10 }) })),
    );
    const r = (await caps["job.logs"].handle({ runId: "r1", offset: 5 }, noopCtx)) as {
      chunk: string; nextOffset: number; truncated: boolean;
    };
    expect(r.nextOffset).toBe(10);
    expect(r.truncated).toBe(false);
  });

  it("restarts at zero when the log shrank", async () => {
    const caps = capsByName(makeJobCaps(deps({ readFileSlice: () => ({ chunk: "abc", size: 3 }) })));
    const r = (await caps["job.logs"].handle({ runId: "r1", offset: 99 }, noopCtx)) as { truncated: boolean; nextOffset: number };
    expect(r.truncated).toBe(true);
    expect(r.nextOffset).toBe(3);
  });
});

describe("job.cancel", () => {
  it("stops the unit", async () => {
    const calls: string[][] = [];
    const caps = capsByName(makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 0 }, calls) })));
    await caps["job.cancel"].handle({ runId: "r1" }, noopCtx);
    expect(calls[0]).toEqual(["sudo", "-n", "systemctl", "stop", "dgxbench-r1"]);
  });

  // Cancelling an already-finished job is a no-op, not an error.
  it("is idempotent when the unit is already gone", async () => {
    const caps = capsByName(
      makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 5, stderr: "not loaded" }, []) })),
    );
    await expect(caps["job.cancel"].handle({ runId: "r1" }, noopCtx)).resolves.toEqual({ stopped: true });
  });
});

describe("job.result", () => {
  it("returns the raw result file contents", async () => {
    const caps = capsByName(makeJobCaps(deps({ readTextFile: () => '{"x":1}' })));
    await expect(caps["job.result"].handle({ runId: "r1" }, noopCtx)).resolves.toEqual({ raw: '{"x":1}' });
  });

  it("returns null when the result file is absent", async () => {
    const caps = capsByName(makeJobCaps(deps({ readTextFile: () => null })));
    await expect(caps["job.result"].handle({ runId: "r1" }, noopCtx)).resolves.toEqual({ raw: null });
  });
});

describe("job.start pruning", () => {
  it("removes job dirs older than the retention window and keeps recent ones", async () => {
    const removed: string[] = [];
    const now = 1_000_000_000_000;
    const day = 86_400_000;
    const caps = capsByName(makeJobCaps(deps({
      now: () => now,
      listJobDirs: () => [
        { path: "/home/daniel/.dgx-agent/jobs/old", mtimeMs: now - 15 * day },
        { path: "/home/daniel/.dgx-agent/jobs/fresh", mtimeMs: now - 1 * day },
      ],
      removeDir: (p: string) => removed.push(p),
    })));
    await caps["job.start"].handle({ runId: "r1", argv: ["x"] }, noopCtx);
    expect(removed).toEqual(["/home/daniel/.dgx-agent/jobs/old"]);
  });

  it("never lets a pruning failure abort the launch", async () => {
    const caps = capsByName(makeJobCaps(deps({
      listJobDirs: () => { throw new Error("EACCES"); },
    })));
    await expect(caps["job.start"].handle({ runId: "r1", argv: ["x"] }, noopCtx)).resolves.toBeTruthy();
  });
});
