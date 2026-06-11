import { describe, expect, it } from "vitest";
import { BENCHMARK_PRESETS, BenchmarkConfig, getPreset, listPresets } from "./presets.js";

describe("BENCHMARK_PRESETS", () => {
  it("exposes the five throughput presets plus four tool-eval presets by id", () => {
    expect(listPresets().map((p) => p.id).sort()).toEqual([
      "chat-long",
      "chat-short",
      "code-32k",
      "quick-smoke",
      "throughput",
      "tool-eval-full",
      "tool-eval-hardmode",
      "tool-eval-pressure",
      "tool-eval-quick",
    ]);
  });

  it("every throughput preset has at least one pp value, one tg value, and runs>=1", () => {
    for (const p of listPresets().filter((p) => p.kind === "throughput")) {
      const cfg = p.config as BenchmarkConfig;
      expect(cfg.pp.length).toBeGreaterThan(0);
      expect(cfg.tg.length).toBeGreaterThan(0);
      expect(cfg.runs).toBeGreaterThanOrEqual(1);
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
    const cfg = p.config as BenchmarkConfig;
    // Heuristic: a single short prompt × generation × 1 run.
    expect(cfg.pp).toEqual([128]);
    expect(cfg.tg).toEqual([32]);
    expect(cfg.runs).toBe(1);
    expect(cfg.concurrency).toEqual([1]);
  });
});

// The presets themselves are kept narrow; if you add or rename one, update
// this list-level test too — that's intentional, presets are part of the
// product surface.

describe("tool-eval presets", () => {
  const ids = ["tool-eval-quick", "tool-eval-full", "tool-eval-hardmode", "tool-eval-pressure"];

  it("registers all four tool-eval presets with kind 'tool-eval'", () => {
    for (const id of ids) {
      const p = getPreset(id);
      expect(p, `preset ${id} should exist`).toBeDefined();
      expect(p!.kind).toBe("tool-eval");
    }
  });

  it("keeps the five throughput presets tagged kind 'throughput'", () => {
    const throughputIds = ["quick-smoke", "chat-short", "chat-long", "code-32k", "throughput"];
    for (const id of throughputIds) {
      expect(getPreset(id)!.kind).toBe("throughput");
    }
  });

  it("maps each tool-eval preset to the documented flag combination", () => {
    const cfg = (id: string) => getPreset(id)!.config as {
      short: boolean; hardmode: boolean; contextPressure: number | null; seed: number;
    };
    expect(cfg("tool-eval-quick")).toMatchObject({ short: true, hardmode: false, contextPressure: null });
    expect(cfg("tool-eval-full")).toMatchObject({ short: false, hardmode: false, contextPressure: null });
    expect(cfg("tool-eval-hardmode")).toMatchObject({ short: false, hardmode: true, contextPressure: null });
    expect(cfg("tool-eval-pressure")).toMatchObject({ short: false, hardmode: false, contextPressure: 0.75 });
  });

  it("every preset carries a kind field", () => {
    for (const p of BENCHMARK_PRESETS) {
      expect(["throughput", "tool-eval"]).toContain(p.kind);
    }
  });
});
