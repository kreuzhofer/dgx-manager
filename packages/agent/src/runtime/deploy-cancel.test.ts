import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { createCancelRegistry, launchExitAction } from "./deploy-cancel.js";

describe("createCancelRegistry", () => {
  it("a stop requested during an in-flight launch is visible to that launch", () => {
    const r = createCancelRegistry();
    r.beginDeploy("d1");
    r.requestCancel("d1", true);
    expect(r.pendingCancel("d1")).toEqual({ deleteAfter: true });
  });

  // The whole point: `docker run -d` exits after `docker rm -f` already ran and
  // found nothing. Without a surviving cancel the container would be orphaned.
  it("keeps the cancel until the launch reaps it", () => {
    const r = createCancelRegistry();
    r.beginDeploy("d1");
    r.requestCancel("d1", false);
    expect(r.pendingCancel("d1")).not.toBeNull(); // still pending at onExit time
    r.forget("d1");
    expect(r.pendingCancel("d1")).toBeNull();
  });

  // Restart reuses the deployment id. A stop followed by a restart must launch,
  // not be cancelled by the stop that preceded it.
  it("a new launch supersedes an earlier cancel for the same id", () => {
    const r = createCancelRegistry();
    r.requestCancel("d1", false);
    r.beginDeploy("d1");
    expect(r.pendingCancel("d1")).toBeNull();
  });

  it("carries deleteAfter so the manager still deletes the row", () => {
    const r = createCancelRegistry();
    r.requestCancel("d1", true);
    expect(r.pendingCancel("d1")?.deleteAfter).toBe(true);
    r.requestCancel("d2", false);
    expect(r.pendingCancel("d2")?.deleteAfter).toBe(false);
  });

  it("isolates deployments from each other", () => {
    const r = createCancelRegistry();
    r.beginDeploy("a");
    r.beginDeploy("b");
    r.requestCancel("a", false);
    expect(r.pendingCancel("a")).not.toBeNull();
    expect(r.pendingCancel("b")).toBeNull();
  });

  it("reports no cancel for an id it has never seen", () => {
    expect(createCancelRegistry().pendingCancel("never")).toBeNull();
  });

  /**
   * Invariant: for any interleaving of operations on one id, the deployment is
   * pending-cancel iff the LAST operation touching it was `requestCancel`.
   * This is the whole contract — `beginDeploy` and `forget` both clear, and a
   * cancel survives arbitrarily long until one of them arrives.
   */
  test.prop([
    fc.array(fc.constantFrom("begin", "cancel", "forget"), { minLength: 1, maxLength: 40 }),
  ])("pending iff the last op was a cancel", (ops) => {
    const r = createCancelRegistry();
    for (const op of ops) {
      if (op === "begin") r.beginDeploy("x");
      else if (op === "cancel") r.requestCancel("x", false);
      else r.forget("x");
    }
    expect(r.pendingCancel("x") !== null).toBe(ops[ops.length - 1] === "cancel");
  });

  /** Invariant: operations on one id never affect another id. */
  test.prop([fc.string({ minLength: 1 }), fc.string({ minLength: 1 })])(
    "cancelling one id never cancels a different id",
    (a, b) => {
      fc.pre(a !== b);
      const r = createCancelRegistry();
      r.requestCancel(a, true);
      expect(r.pendingCancel(b)).toBeNull();
    },
  );
});

describe("launchExitAction", () => {
  it("a clean launch with no cancel is running", () => {
    expect(launchExitAction({ code: 0, rank: 0, cancel: null })).toEqual({ kind: "running" });
  });

  it("a non-zero exit with no cancel is a failure naming the rank", () => {
    const a = launchExitAction({ code: 125, rank: 2, cancel: null });
    expect(a.kind).toBe("failed");
    expect(a).toMatchObject({ error: expect.stringContaining("rank 2") });
  });

  it("a null exit code (killed) is a failure", () => {
    expect(launchExitAction({ code: null, rank: 0, cancel: null }).kind).toBe("failed");
  });

  // The race: container came up, stop already requested. Tear it down.
  it("a cancel on a successful launch means cancelled, not running", () => {
    expect(launchExitAction({ code: 0, rank: 0, cancel: { deleteAfter: true } }))
      .toEqual({ kind: "cancelled", deleteAfter: true });
  });

  // Precedence: reporting `failed` here would trip coordinated teardown and, for a
  // DELETE, strand the row. The user asked for it gone either way.
  it("a cancel WINS over a failed launch", () => {
    expect(launchExitAction({ code: 1, rank: 3, cancel: { deleteAfter: false } }))
      .toEqual({ kind: "cancelled", deleteAfter: false });
  });

  /** Invariant: whenever a cancel is pending the action is `cancelled`, whatever the exit code. */
  test.prop([
    fc.oneof(fc.integer({ min: -5, max: 130 }), fc.constant(null)),
    fc.integer({ min: 0, max: 15 }),
    fc.boolean(),
  ])("a pending cancel always yields cancelled, preserving deleteAfter", (code, rank, deleteAfter) => {
    expect(launchExitAction({ code, rank, cancel: { deleteAfter } }))
      .toEqual({ kind: "cancelled", deleteAfter });
  });

  /** Invariant: with no cancel, `running` iff the exit code is exactly 0. */
  test.prop([fc.oneof(fc.integer({ min: -5, max: 130 }), fc.constant(null))])(
    "without a cancel, running iff exit code 0",
    (code) => {
      const a = launchExitAction({ code, rank: 0, cancel: null });
      expect(a.kind === "running").toBe(code === 0);
    },
  );
});
