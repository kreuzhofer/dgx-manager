import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { reconcileAction } from "./reconcile.js";

const remote = { runnerNodeId: "n1" };
const legacy = { runnerNodeId: null };

describe("reconcileAction", () => {
  it("fails a legacy local run (preserves today's behavior)", () => {
    expect(reconcileAction(legacy, null)).toBe("fail-legacy");
  });
  it("resumes a remote job still running", () => {
    expect(reconcileAction(remote, { kind: "active" })).toBe("resume");
  });
  it("finalizes a remote job that finished during downtime", () => {
    expect(reconcileAction(remote, { kind: "exited", code: 0 })).toBe("finalize");
  });
  it("fails a remote job whose unit is genuinely gone", () => {
    expect(reconcileAction(remote, { kind: "missing" })).toBe("fail-orphan");
  });
  /** Invariant: an unreachable agent at boot must NOT declare the run dead. */
  test.prop([fc.string()])("an unknown status resumes rather than fails", (reason) => {
    expect(reconcileAction(remote, { kind: "unknown", reason })).toBe("resume");
  });
  it("resumes when the agent is offline at boot (null status)", () => {
    expect(reconcileAction(remote, null)).toBe("resume");
  });
});
