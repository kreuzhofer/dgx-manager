import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { buildToolEvalArgs } from "./tool-eval-args.js";
import type { ToolEvalConfig } from "./presets.js";

const base: ToolEvalConfig = { short: false, hardmode: false, contextPressure: null, seed: 42 };
const target = { baseUrl: "http://10.0.0.1:8000/v1", modelName: "m", outputPath: "/out/result.json" };

function valuesAfter(args: string[], flag: string): string[] {
  const idx = args.indexOf(flag);
  if (idx < 0) return [];
  const tail = args.slice(idx + 1);
  const stop = tail.findIndex((t) => t.startsWith("--"));
  return stop === -1 ? tail : tail.slice(0, stop);
}

describe("buildToolEvalArgs", () => {
  it("always emits base-url, explicit model, json-file and seed", () => {
    const args = buildToolEvalArgs(base, target);
    expect(valuesAfter(args, "--base-url")).toEqual(["http://10.0.0.1:8000/v1"]);
    expect(valuesAfter(args, "--model")).toEqual(["m"]);
    expect(valuesAfter(args, "--json-file")).toEqual(["/out/result.json"]);
    expect(valuesAfter(args, "--seed")).toEqual(["42"]);
  });

  it("omits --short, --hardmode and --context-pressure for the full default", () => {
    const args = buildToolEvalArgs(base, target);
    expect(args).not.toContain("--short");
    expect(args).not.toContain("--hardmode");
    expect(args).not.toContain("--context-pressure");
  });

  it("includes --short only when short is set", () => {
    expect(buildToolEvalArgs({ ...base, short: true }, target)).toContain("--short");
  });

  it("includes --hardmode only when hardmode is set", () => {
    expect(buildToolEvalArgs({ ...base, hardmode: true }, target)).toContain("--hardmode");
  });

  it("includes --context-pressure with one value token only when non-null", () => {
    const on = buildToolEvalArgs({ ...base, contextPressure: 0.75 }, target);
    expect(valuesAfter(on, "--context-pressure")).toEqual(["0.75"]);
    const off = buildToolEvalArgs({ ...base, contextPressure: null }, target);
    expect(off).not.toContain("--context-pressure");
  });

  // Invariant: --model is always present exactly once (so the interactive
  // model picker can never hang a headless run).
  test.prop([fc.boolean(), fc.boolean(), fc.option(fc.float({ min: 0, max: 1, noNaN: true }), { nil: null })])(
    "always passes --model exactly once regardless of variant flags",
    (short, hardmode, contextPressure) => {
      const args = buildToolEvalArgs({ short, hardmode, contextPressure, seed: 7 }, target);
      expect(args.filter((a) => a === "--model").length).toBe(1);
    },
  );
});
