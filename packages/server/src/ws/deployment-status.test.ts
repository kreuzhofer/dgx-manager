import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { deploymentStatusUpdate, HEALTHY_STATUS } from "./deployment-status.js";

describe("deploymentStatusUpdate", () => {
  it("persists the agent's error so a failed deploy is not indistinguishable from a stopped one", () => {
    const d = deploymentStatusUpdate({ status: "failed", error: "ValueError: Free memory on device cuda:0" });
    expect(d.status).toBe("failed");
    expect(d.error).toBe("ValueError: Free memory on device cuda:0");
  });

  // The crash sequence is: agent reports `failed` WITH an error, then teardown
  // reports `stopped` with NO error. Blindly writing `error` would erase the only
  // record of why the deploy died — the exact bug this helper exists to prevent.
  it("leaves a previously stored error untouched when a later tick carries none", () => {
    const d = deploymentStatusUpdate({ status: "stopped" });
    expect("error" in d).toBe(false);
  });

  it("clears a stale error once the deployment is actually healthy", () => {
    const d = deploymentStatusUpdate({ status: "running", port: 8000 });
    expect(d.error).toBeNull();
    expect(d.port).toBe(8000);
  });

  it("treats an empty-string error as no error at all", () => {
    expect("error" in deploymentStatusUpdate({ status: "stopped", error: "" })).toBe(false);
  });

  it("zeroes vramActual on every terminal status", () => {
    for (const status of ["stopped", "failed", "evicted"]) {
      expect(deploymentStatusUpdate({ status, vramActual: 99 }).vramActual).toBe(0);
    }
  });

  it("passes vramActual through while the deployment is live", () => {
    expect(deploymentStatusUpdate({ status: "running", vramActual: 42 }).vramActual).toBe(42);
    expect(deploymentStatusUpdate({ status: "loading", vramActual: "17" }).vramActual).toBe(17);
  });

  it("omits port when the agent did not report one, so a bound port is never clobbered", () => {
    expect("port" in deploymentStatusUpdate({ status: "loading" })).toBe(false);
    expect("vramActual" in deploymentStatusUpdate({ status: "loading" })).toBe(false);
  });

  // Invariant: the ONLY way `error` is set to null is a healthy status. Any other
  // status with no error must leave the column alone.
  test.prop([
    fc.string({ minLength: 1 }).filter((s) => s !== HEALTHY_STATUS),
  ])("never clears an error except on the healthy status", (status) => {
    const d = deploymentStatusUpdate({ status });
    expect("error" in d).toBe(false);
  });

  // Invariant: a non-empty error is always persisted verbatim, whatever the status.
  test.prop([fc.string({ minLength: 1 }), fc.string({ minLength: 1 })])(
    "always persists a non-empty error verbatim",
    (status, error) => {
      expect(deploymentStatusUpdate({ status, error }).error).toBe(error);
    },
  );
});
