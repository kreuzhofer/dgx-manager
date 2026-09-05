import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { reconcileOllamaAction, shouldRetryOllamaReconcile } from "./ollama-reconcile.js";

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

/**
 * The second half of the 2026-09-04 agenthost reboot bug.
 *
 * The reconcile above runs only when the agent registers over the WebSocket.
 * On that boot the restore failed (Ollama's systemd job was still queued
 * behind network-online.target), the deployment was reported `failed` — and
 * then nothing ever looked again. The agent stayed connected, so no second
 * reconcile ran, even though Ollama came up 76s later and the agent's own 15s
 * health tick was successfully polling /api/ps the whole time.
 *
 * `shouldRetryOllamaReconcile` lets the health tick drive a retry, so a
 * transient boot-time failure heals itself instead of needing a manual
 * `systemctl restart dgx-agent`.
 */
describe("shouldRetryOllamaReconcile", () => {
  it("retries when a restore is still outstanding", () => {
    expect(shouldRetryOllamaReconcile({ pendingRestores: 1, inFlight: false })).toBe(true);
  });

  // Nothing failed, so a retry would be pure churn against systemd + the API.
  it("stays quiet when no restore is outstanding", () => {
    expect(shouldRetryOllamaReconcile({ pendingRestores: 0, inFlight: false })).toBe(false);
  });

  // The reconcile can wait minutes for a slow boot. The health tick fires every
  // 15s, so without this guard a single failure would pile up reconciles, each
  // issuing its own `systemctl start` and status report.
  it("never starts a second reconcile while one is running", () => {
    expect(shouldRetryOllamaReconcile({ pendingRestores: 3, inFlight: true })).toBe(false);
  });

  /**
   * Invariant: a retry happens only when there is outstanding work AND no
   * reconcile is already in flight. Both conditions are necessary.
   */
  test.prop([fc.nat({ max: 5 }), fc.boolean()])(
    "retries only when work is outstanding and nothing is in flight",
    (pendingRestores, inFlight) => {
      const retry = shouldRetryOllamaReconcile({ pendingRestores, inFlight });
      expect(retry).toBe(pendingRestores > 0 && !inFlight);
      if (retry) {
        expect(pendingRestores).toBeGreaterThan(0);
        expect(inFlight).toBe(false);
      }
    },
  );
});
