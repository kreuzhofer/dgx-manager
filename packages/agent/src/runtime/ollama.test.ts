/**
 * Property + unit tests for the pure Ollama eviction-state-machine helper.
 *
 * Background: the agent has a 15s health-check loop that watches Ollama's
 * /api/ps to detect models that fell out of GPU memory. The original
 * decision logic ("evict when !loaded && prev !== 'evicted'") false-fired
 * during the first tick after a restart — `prev` was still "running" from
 * the previous deploy cycle, so a transient "not yet loaded" got reported
 * as eviction.
 *
 * `decideOllamaStateTransition` is the pure decision function pulled out
 * of the loop so we can pin its invariants here.
 */
import { describe, it, expect, vi } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { decideOllamaStateTransition, ensureOllamaRunning } from "./ollama.js";
import type { EnsureOllamaDeps } from "./ollama.js";

const prevStateArb = fc.constantFrom<string | undefined>(
  undefined,
  "running",
  "evicted",
);

describe("decideOllamaStateTransition", () => {
  /**
   * Eviction is a transition, not a state: only flag evicted when we
   * previously observed the model loaded ("running") AND we now observe
   * it not loaded. Anything else with !loaded is either still-starting
   * (prev=undefined) or already-evicted (prev="evicted") — neither
   * deserves a fresh "evicted" report.
   */
  test.prop([fc.boolean(), prevStateArb])(
    "only reports 'evicted' when prev === 'running' and !loaded",
    (loaded, prev) => {
      const result = decideOllamaStateTransition(loaded, prev);
      if (result === "evicted") {
        expect(loaded).toBe(false);
        expect(prev).toBe("running");
      }
    },
  );

  /**
   * Running is reported on the first tick we see the model loaded after
   * either a fresh deploy (prev=undefined) or a recovered eviction
   * (prev="evicted"). Once we've already reported running, subsequent
   * loaded ticks must stay silent to avoid status spam.
   */
  test.prop([fc.boolean(), prevStateArb])(
    "only reports 'running' when loaded and prev is undefined or 'evicted'",
    (loaded, prev) => {
      const result = decideOllamaStateTransition(loaded, prev);
      if (result === "running") {
        expect(loaded).toBe(true);
        expect(prev === undefined || prev === "evicted").toBe(true);
      }
    },
  );

  /**
   * No-op when nothing has changed: a loaded model with prev="running"
   * (steady state) and a not-loaded model with prev="evicted" (steady
   * state) both produce null.
   */
  it("returns null for the two steady states", () => {
    expect(decideOllamaStateTransition(true, "running")).toBeNull();
    expect(decideOllamaStateTransition(false, "evicted")).toBeNull();
  });

  /**
   * The restart bug specifically: when prev is undefined (fresh
   * deploy cycle, ollamaLastState was cleared) and the model is not
   * yet in VRAM, we must NOT report evicted.
   */
  it("does not evict on the first tick of a fresh deploy", () => {
    expect(decideOllamaStateTransition(false, undefined)).toBeNull();
  });
});

/**
 * Tests for the on-demand Ollama service start that runs at the top of every
 * Ollama deploy.
 *
 * Background: fleet policy disables Ollama's systemd autostart on all nodes
 * (the unauthenticated :11434 API once loaded a 15 GB model mid-vLLM-deploy
 * and killed a cluster startup). So when the manager sends an Ollama deploy,
 * the service is usually stopped — the agent must `sudo -n systemctl start
 * ollama` and wait (bounded) for the API to answer, or fail the deploy
 * loudly. `ensureOllamaRunning` takes injected isRunning/startService/sleep
 * so none of this touches systemd or HTTP.
 */
describe("ensureOllamaRunning", () => {
  /** Builds deps where isRunning yields the given sequence (last value
   *  repeats). sleep resolves immediately so tests never actually wait. */
  function makeDeps(runningSequence: boolean[]): EnsureOllamaDeps & {
    isRunning: ReturnType<typeof vi.fn>;
    startService: ReturnType<typeof vi.fn>;
    sleep: ReturnType<typeof vi.fn>;
  } {
    let i = 0;
    return {
      isRunning: vi.fn(async () => runningSequence[Math.min(i++, runningSequence.length - 1)]),
      startService: vi.fn(async () => {}),
      sleep: vi.fn(async () => {}),
    };
  }

  it("does not touch systemd when the API is already reachable", async () => {
    const deps = makeDeps([true]);
    await ensureOllamaRunning(deps);
    expect(deps.startService).not.toHaveBeenCalled();
    expect(deps.sleep).not.toHaveBeenCalled();
  });

  it("starts the service when stopped, then polls until the API answers", async () => {
    // Not running at the pre-check, still not running on the first poll,
    // reachable on the second poll.
    const deps = makeDeps([false, false, true]);
    await ensureOllamaRunning(deps);
    expect(deps.startService).toHaveBeenCalledTimes(1);
    expect(deps.sleep).toHaveBeenCalledTimes(2);
    // start must happen BEFORE any readiness poll
    expect(deps.startService.mock.invocationCallOrder[0]).toBeLessThan(
      deps.sleep.mock.invocationCallOrder[0],
    );
  });

  /** A failing `systemctl start` (missing sudoers rule, masked unit, …)
   *  must fail the deploy with a message that keeps the underlying cause
   *  (execFile errors already name the command) — no silent fallback into
   *  the pull step against a dead API. */
  it("throws with a clear message when the start command fails", async () => {
    const deps = makeDeps([false]);
    deps.startService.mockRejectedValue(new Error("sudo: a password is required"));
    await expect(ensureOllamaRunning(deps)).rejects.toThrow(
      /Failed to start Ollama service: sudo: a password is required/,
    );
    expect(deps.sleep).not.toHaveBeenCalled();
  });

  /** The readiness wait is bounded: exactly maxAttempts polls (one sleep
   *  each), then a clear timeout error — never an infinite loop. */
  it("throws after bounded attempts when the API never becomes reachable", async () => {
    const deps = makeDeps([false]);
    await expect(
      ensureOllamaRunning(deps, { maxAttempts: 4, intervalMs: 100 }),
    ).rejects.toThrow(/did not become reachable after 4 attempts over ~0s/);
    expect(deps.startService).toHaveBeenCalledTimes(1);
    expect(deps.sleep).toHaveBeenCalledTimes(4);
    // opts.intervalMs must flow through to the injected sleep
    expect(deps.sleep).toHaveBeenCalledWith(100);
    // pre-check + 4 polls
    expect(deps.isRunning).toHaveBeenCalledTimes(5);
  });
});
