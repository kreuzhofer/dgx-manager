import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { reconcileDgxrunAction } from "./dgxrun-reconcile.js";
import type { DgxrunInspect } from "./dgxrun.js";

const found = (state: string): DgxrunInspect => ({ kind: "found", name: "dgxrun_d1", state, restartCount: 0 });

describe("reconcileDgxrunAction", () => {
  // The one that matters: a slow/busy docker daemon must never look like a dead
  // rank, because one `failed` tears down all four ranks of a healthy cluster.
  it("skips when docker inspect is inconclusive", () => {
    const a = reconcileDgxrunAction({ kind: "unknown", reason: "spawnSync timeout" }, { rank: 0 });
    expect(a.kind).toBe("skip");
  });

  it("fails a rank whose container docker positively says is absent", () => {
    expect(reconcileDgxrunAction({ kind: "absent" }, { rank: 2 })).toMatchObject({
      kind: "report", status: "failed",
    });
  });

  it("fails a rank whose container exists but is not running", () => {
    for (const state of ["exited", "created", "paused", "dead"]) {
      expect(reconcileDgxrunAction(found(state), { rank: 1 })).toMatchObject({
        kind: "report", status: "failed",
      });
    }
  });

  it("re-announces the phase for a live head instead of a terminal status", () => {
    expect(reconcileDgxrunAction(found("running"), { rank: 0, port: 8000 })).toEqual({ kind: "phase" });
  });

  it("reports a live worker as running", () => {
    expect(reconcileDgxrunAction(found("running"), { rank: 3 })).toEqual({ kind: "report", status: "running" });
  });

  /**
   * Invariant: an `unknown` inspect NEVER produces a report, for any rank. This is
   * the property whose violation destroyed a healthy cluster — a report of `failed`
   * from any single rank is unrecoverable.
   */
  test.prop([fc.integer({ min: 0, max: 64 }), fc.string()])(
    "an inconclusive inspect never reports anything, at any rank",
    (rank, reason) => {
      expect(reconcileDgxrunAction({ kind: "unknown", reason }, { rank }).kind).toBe("skip");
    },
  );

  /** Invariant: `failed` is reported only when docker actually answered. */
  test.prop([
    fc.oneof(
      fc.constant<DgxrunInspect>({ kind: "absent" }),
      fc.string({ minLength: 1 }).map((s) => found(s)),
      fc.string().map((reason): DgxrunInspect => ({ kind: "unknown", reason })),
    ),
    fc.integer({ min: 0, max: 8 }),
  ])("never fails a rank on an unanswered inspect", (inspect, rank) => {
    const a = reconcileDgxrunAction(inspect, { rank });
    if (a.kind === "report" && a.status === "failed") {
      expect(inspect.kind).not.toBe("unknown");
    }
  });
});
