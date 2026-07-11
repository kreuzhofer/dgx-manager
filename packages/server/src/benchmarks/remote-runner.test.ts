import { describe, expect, it, vi } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { nextPollAction, runTrackedRemote } from "./remote-runner.js";

describe("nextPollAction", () => {
  it("keeps polling a live job", () => {
    expect(nextPollAction({ kind: "active" }, false)).toBe("continue");
  });

  it("finishes on a clean exit", () => {
    expect(nextPollAction({ kind: "exited", code: 0 }, true)).toBe("finish");
  });

  it("finishes on a non-zero exit — the run failed, the poll did not", () => {
    expect(nextPollAction({ kind: "exited", code: 1 }, true)).toBe("finish");
  });

  it("fails only when the unit is gone and no exit file exists", () => {
    expect(nextPollAction({ kind: "missing" }, false)).toBe("fail");
  });

  /**
   * THE invariant. A cap timeout, a busy box, a dropped WS: we could not ask. The
   * job is almost certainly still running. Skip the tick. Failing here would kill
   * an 80-minute eval on one slow poll — the same absent-vs-unknown conflation
   * that tore down four healthy GLM-5.2 ranks on 2026-07-09.
   */
  test.prop([fc.boolean(), fc.string()])(
    "an unknown status never fails or finishes a run",
    (hasExit, reason) => {
      expect(nextPollAction({ kind: "unknown", reason }, hasExit)).toBe("continue");
    },
  );

  // A unit systemd reports gone, but a result file the wrapper writes just before
  // exiting DOES exist: the job finished, the unit was just already reaped. Finish,
  // don't fail — this is the belt-and-braces the server adds on top of the agent's
  // own exit-file resolution.
  it("finishes a missing unit when a result file exists", () => {
    expect(nextPollAction({ kind: "missing" }, true)).toBe("finish");
  });

  /** Invariant: active and unknown never finish — only exited, or missing+result. */
  test.prop([
    fc.oneof(
      fc.constant({ kind: "active" as const }),
      fc.string().map((reason) => ({ kind: "unknown" as const, reason })),
    ),
    fc.boolean(),
  ])("active and unknown never finish, whatever the exit-file flag", (status, hasExit) => {
    expect(nextPollAction(status, hasExit)).not.toBe("finish");
  });
});

describe("runTrackedRemote", () => {
  const baseOpts = {
    runId: "r1", nodeId: "n1", argv: ["uvx", "lm_eval"], resultGlob: "results_*.json",
    pollMs: 1, onLog: () => {}, onOffset: () => {},
  };

  it("starts the job, drains logs, and returns the result", async () => {
    const invoke = vi.fn(async (_n: string, name: string) => {
      if (name === "job.start") return { ok: true, data: { unit: "dgxbench-r1", jobDir: "/j" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "hello\n", nextOffset: 6, truncated: false } };
      if (name === "job.status") return { ok: true, data: { kind: "exited", code: 0 } };
      if (name === "job.result") return { ok: true, data: { raw: '{"results":{}}' } };
      throw new Error("unexpected " + name);
    });
    const lines: string[] = [];
    const r = await runTrackedRemote({ ...baseOpts, invoke, onLog: (l) => lines.push(l) });
    expect(r.exitCode).toBe(0);
    expect(r.rawOutput).toBe('{"results":{}}');
    expect(lines).toContain("hello");
  });

  // A cap timeout mid-run must not end the run.
  it("survives an inconclusive status and keeps polling", async () => {
    let statusCalls = 0;
    const invoke = vi.fn(async (_n: string, name: string) => {
      if (name === "job.start") return { ok: true, data: { unit: "u", jobDir: "/j" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
      if (name === "job.status") {
        statusCalls += 1;
        if (statusCalls < 3) return { ok: false, error: "cap timeout" };
        return { ok: true, data: { kind: "exited", code: 0 } };
      }
      if (name === "job.result") return { ok: true, data: { raw: "{}" } };
      throw new Error("unexpected " + name);
    });
    const r = await runTrackedRemote({ ...baseOpts, invoke });
    expect(statusCalls).toBe(3);
    expect(r.exitCode).toBe(0);
  });

  it("reports a non-zero exit without a result", async () => {
    const invoke = vi.fn(async (_n: string, name: string) => {
      if (name === "job.start") return { ok: true, data: { unit: "u", jobDir: "/j" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
      if (name === "job.status") return { ok: true, data: { kind: "exited", code: 2 } };
      if (name === "job.result") return { ok: true, data: { raw: null } };
      throw new Error("unexpected " + name);
    });
    const r = await runTrackedRemote({ ...baseOpts, invoke });
    expect(r.exitCode).toBe(2);
    expect(r.rawOutput).toBeNull();
  });

  it("throws when the job cannot be started", async () => {
    const invoke = vi.fn(async () => ({ ok: false, error: "sudo: a password is required" }));
    await expect(runTrackedRemote({ ...baseOpts, invoke })).rejects.toThrow(/password is required/);
  });

  // The unit was reaped before we polled, but the wrapper had already written
  // result.json — the run finished, do not fail it.
  it("finishes when the unit is gone but a result file exists", async () => {
    const invoke = vi.fn(async (_n: string, name: string) => {
      if (name === "job.start") return { ok: true, data: { unit: "u", jobDir: "/j" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
      if (name === "job.status") return { ok: true, data: { kind: "missing" } };
      if (name === "job.result") return { ok: true, data: { raw: '{"results":{}}' } };
      throw new Error("unexpected " + name);
    });
    const r = await runTrackedRemote({ ...baseOpts, invoke });
    expect(r.exitCode).toBe(0);
    expect(r.rawOutput).toBe('{"results":{}}');
  });

  // Genuinely orphaned: unit gone AND no result. This must throw.
  it("throws when the unit is gone with no result", async () => {
    const invoke = vi.fn(async (_n: string, name: string) => {
      if (name === "job.start") return { ok: true, data: { unit: "u", jobDir: "/j" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
      if (name === "job.status") return { ok: true, data: { kind: "missing" } };
      if (name === "job.result") return { ok: true, data: { raw: null } };
      throw new Error("unexpected " + name);
    });
    await expect(runTrackedRemote({ ...baseOpts, invoke })).rejects.toThrow(/vanished/);
  });

  it("persists the log offset as it advances, so a restart can reattach", async () => {
    const offsets: number[] = [];
    let done = false;
    const invoke = vi.fn(async (_n: string, name: string) => {
      if (name === "job.start") return { ok: true, data: { unit: "u", jobDir: "/j" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "abc", nextOffset: 3, truncated: false } };
      if (name === "job.status") { const r = done ? { kind: "exited", code: 0 } : { kind: "active" }; done = true; return { ok: true, data: r }; }
      if (name === "job.result") return { ok: true, data: { raw: "{}" } };
      throw new Error("unexpected " + name);
    });
    await runTrackedRemote({ ...baseOpts, invoke, onOffset: (o) => offsets.push(o) });
    expect(offsets.at(-1)).toBe(3);
  });
});

describe("runTrackedRemote reattach (skipStart)", () => {
  const base = { runId: "r1", nodeId: "n1", argv: ["uvx", "x"], pollMs: 1, onLog: () => {} };

  it("does not call job.start when skipStart is set", async () => {
    const seen: string[] = [];
    const invoke = vi.fn(async (_n: string, name: string) => {
      seen.push(name);
      if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
      if (name === "job.status") return { ok: true, data: { kind: "exited", code: 0 } };
      if (name === "job.result") return { ok: true, data: { raw: "{}" } };
      throw new Error("unexpected " + name);
    });
    const r = await runTrackedRemote({ ...base, invoke, skipStart: true, startOffset: 42 });
    expect(seen).not.toContain("job.start");
    expect(r.exitCode).toBe(0);
  });

  it("still calls job.start when skipStart is absent", async () => {
    const seen: string[] = [];
    const invoke = vi.fn(async (_n: string, name: string) => {
      seen.push(name);
      if (name === "job.start") return { ok: true, data: { unit: "u", jobDir: "/j" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
      if (name === "job.status") return { ok: true, data: { kind: "exited", code: 0 } };
      if (name === "job.result") return { ok: true, data: { raw: "{}" } };
      throw new Error("unexpected " + name);
    });
    await runTrackedRemote({ ...base, invoke });
    expect(seen[0]).toBe("job.start");
  });
});
