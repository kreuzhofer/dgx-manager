import { describe, expect, it } from "vitest";
import { BENCHMARK_PRESETS, getPreset, listPresets } from "./presets.js";

describe("BENCHMARK_PRESETS", () => {
  it("exposes the five expected presets by id", () => {
    expect(listPresets().map((p) => p.id).sort()).toEqual([
      "chat-long",
      "chat-short",
      "code-32k",
      "quick-smoke",
      "throughput",
    ]);
  });

  it("every preset has at least one pp value, one tg value, and runs>=1", () => {
    for (const p of listPresets()) {
      expect(p.config.pp.length).toBeGreaterThan(0);
      expect(p.config.tg.length).toBeGreaterThan(0);
      expect(p.config.runs).toBeGreaterThanOrEqual(1);
    }
  });

  it("getPreset returns the preset by id", () => {
    const p = getPreset("quick-smoke");
    expect(p?.id).toBe("quick-smoke");
  });

  it("getPreset returns undefined for unknown ids", () => {
    expect(getPreset("does-not-exist")).toBeUndefined();
  });

  it("quick-smoke is small enough to finish in under a minute on a single GPU", () => {
    const p = getPreset("quick-smoke")!;
    // Heuristic: a single short prompt × generation × 1 run.
    expect(p.config.pp).toEqual([128]);
    expect(p.config.tg).toEqual([32]);
    expect(p.config.runs).toBe(1);
    expect(p.config.concurrency).toEqual([1]);
  });
});

// The presets themselves are kept narrow; if you add or rename one, update
// this list-level test too — that's intentional, presets are part of the
// product surface.
