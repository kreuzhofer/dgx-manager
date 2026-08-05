import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { reconcileOllamaAction } from "./ollama-reconcile.js";

describe("reconcileOllamaAction", () => {
  // The reboot case this exists for. Fleet policy leaves Ollama disabled at
  // boot, so after a node restarts the service is down and the model with it —
  // and nothing else in the agent restores an Ollama deployment. Without this
  // the deployment stays dead until someone redeploys it by hand.
  it("restores a deployment whose service is down", () => {
    expect(reconcileOllamaAction({ serviceRunning: false, modelLoaded: false })).toEqual({
      kind: "restore",
      reason: "Ollama is not running",
    });
  });

  // Ollama loads a model on demand, so a running service with the model
  // unloaded is not broken — it is idle. Reloading it here would pull a model
  // into memory nobody has asked for.
  it("leaves a running service with an unloaded model alone", () => {
    expect(reconcileOllamaAction({ serviceRunning: true, modelLoaded: false })).toEqual({
      kind: "idle",
      reason: "model not resident; Ollama loads it on demand",
    });
  });

  it("reports a deployment that is already serving", () => {
    expect(reconcileOllamaAction({ serviceRunning: true, modelLoaded: true })).toEqual({
      kind: "serving",
      reason: "model resident",
    });
  });

  // A deployment the operator stopped must not be resurrected by a reconnect.
  it("never restores a deployment that is being stopped", () => {
    for (const serviceRunning of [true, false]) {
      expect(reconcileOllamaAction({ serviceRunning, modelLoaded: false, stopping: true })).toEqual({
        kind: "skip",
        reason: "undeploy in progress",
      });
    }
  });

  /**
   * Invariant: a restore is proposed only when the service is actually down and
   * the operator has not asked for the deployment to stop. Restoring in any
   * other state would either fight an undeploy or start work nobody requested.
   */
  test.prop([fc.boolean(), fc.boolean(), fc.boolean()])(
    "only restores a non-stopping deployment whose service is down",
    (serviceRunning, modelLoaded, stopping) => {
      const action = reconcileOllamaAction({ serviceRunning, modelLoaded, stopping });
      if (action.kind === "restore") {
        expect(serviceRunning).toBe(false);
        expect(stopping).toBeFalsy();
      }
      expect(action.reason.length).toBeGreaterThan(0);
    },
  );
});
