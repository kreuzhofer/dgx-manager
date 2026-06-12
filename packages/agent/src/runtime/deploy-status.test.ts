import { describe, it, expect } from "vitest";
import { classifyDeadContainer, reconcileDeployStatus } from "./deploy-status.js";

describe("classifyDeadContainer", () => {
  // An intentional stop (user cmd:undeploy → stopRecipe set `stopping`) must be
  // reported as `stopped`, never as a crash, and carries no error.
  it("reports a user-intended stop as 'stopped' with no error", () => {
    expect(classifyDeadContainer(true)).toEqual({ status: "stopped" });
    expect(classifyDeadContainer(true, "ignored")).toEqual({ status: "stopped" });
  });

  // An unintended container death is a crash → `failed`, with the engine's last
  // error if known, else the generic message.
  it("reports an unintended death as 'failed' with the last error or a default", () => {
    expect(classifyDeadContainer(false)).toEqual({
      status: "failed",
      error: "Container stopped unexpectedly",
    });
    expect(classifyDeadContainer(false, null)).toEqual({
      status: "failed",
      error: "Container stopped unexpectedly",
    });
    expect(classifyDeadContainer(false, "CUDA illegal memory access")).toEqual({
      status: "failed",
      error: "CUDA illegal memory access",
    });
  });
});

describe("reconcileDeployStatus", () => {
  // Invariant: listed workload is always "running" regardless of launcher state.
  // Unlisted + alive launcher means the workload is still coming up ("deploying").
  // Unlisted + dead launcher means the workload failed to start or crashed ("failed").
  it("running when listed, deploying when launcher alive but not listed, else failed", () => {
    expect(reconcileDeployStatus({ launcherAlive: false, listed: true })).toBe("running");
    expect(reconcileDeployStatus({ launcherAlive: true, listed: true })).toBe("running");
    expect(reconcileDeployStatus({ launcherAlive: true, listed: false })).toBe("deploying");
    expect(reconcileDeployStatus({ launcherAlive: false, listed: false })).toBe("failed");
  });
});
