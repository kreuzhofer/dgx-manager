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
import { describe, it, expect } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { decideOllamaStateTransition } from "./ollama.js";

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
