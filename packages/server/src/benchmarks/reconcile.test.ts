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

describe("reconcileAction — a run that depended on the manager's reasoning proxy (#22)", () => {
  const proxied = { runnerNodeId: "n1", usesManagerProxy: true };

  it("fails an ACTIVE proxied job instead of resuming it into a closed socket", () => {
    // The job survived the restart; its target did not. Resuming re-created a
    // proxy on a NEW random port that the already-running job knows nothing
    // about, so the run burned GPU until it died on a urllib3 traceback.
    expect(reconcileAction(proxied, { kind: "active" })).toBe("fail-proxy-lost");
  });

  it("still FINALIZES a proxied job that had already exited", () => {
    // It may well have succeeded before the restart — failing it would discard
    // a complete result.
    expect(reconcileAction(proxied, { kind: "exited", code: 0 })).toBe("finalize");
  });

  it("still RESUMES a proxied job whose status we could not obtain", () => {
    // We cannot distinguish "doomed" from "already finished" without asking, so
    // the existing never-declare-dead-blind rule wins over the proxy fact.
    expect(reconcileAction(proxied, null)).toBe("resume");
    expect(reconcileAction(proxied, { kind: "unknown", reason: "agent offline" })).toBe("resume");
  });

  it("leaves non-proxied runs behaving exactly as before", () => {
    const plain = { runnerNodeId: "n1", usesManagerProxy: false };
    expect(reconcileAction(plain, { kind: "active" })).toBe("resume");
    expect(reconcileAction({ runnerNodeId: "n1" }, { kind: "active" })).toBe("resume");
  });
});
