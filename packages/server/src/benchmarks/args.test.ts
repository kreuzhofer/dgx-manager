import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { buildBenchyArgs } from "./args.js";
import type { BenchmarkConfig } from "./presets.js";

const baseConfig: BenchmarkConfig = {
  pp: [128, 512],
  tg: [32, 128],
  depth: [0, 4096],
  runs: 3,
  concurrency: [1, 4],
  latencyMode: "api",
  enablePrefixCaching: false,
  skipCoherence: false,
};

// Returns the contiguous run of argv values that follow a flag, up to but
// not including the next flag (--xxx) or the end of argv. Useful because
// llama-benchy uses argparse nargs='+', so each list flag is followed by
// one or more positional value tokens.
function valuesAfter(args: string[], flag: string): string[] {
  const idx = args.indexOf(flag);
  if (idx < 0) return [];
  const tail = args.slice(idx + 1);
  const stop = tail.findIndex((t) => t.startsWith("--"));
  return stop === -1 ? tail : tail.slice(0, stop);
}

describe("buildBenchyArgs", () => {
  it("emits --base-url, --model and the JSON output path", () => {
    const args = buildBenchyArgs(baseConfig, {
      baseUrl: "http://10.0.0.1:8000",
      modelName: "llama-3.1-8b",
      outputPath: "/output/result.json",
    });
    expect(args).toContain("--base-url");
    expect(args).toContain("http://10.0.0.1:8000");
    expect(args).toContain("--model");
    expect(args).toContain("llama-3.1-8b");
    expect(args).toContain("--save-result");
    expect(args).toContain("/output/result.json");
    expect(args).toContain("--format");
    expect(args).toContain("json");
  });

  it("emits pp/tg/depth/concurrency as separate space-separated argv tokens", () => {
    const args = buildBenchyArgs(baseConfig, {
      baseUrl: "http://10.0.0.1:8000",
      modelName: "m",
      outputPath: "/output/r.json",
    });
    expect(valuesAfter(args, "--pp")).toEqual(["128", "512"]);
    expect(valuesAfter(args, "--tg")).toEqual(["32", "128"]);
    expect(valuesAfter(args, "--depth")).toEqual(["0", "4096"]);
    expect(valuesAfter(args, "--concurrency")).toEqual(["1", "4"]);
    expect(valuesAfter(args, "--runs")).toEqual(["3"]);
  });

  it("includes --enable-prefix-caching only when enabled", () => {
    const off = buildBenchyArgs(
      { ...baseConfig, enablePrefixCaching: false },
      { baseUrl: "u", modelName: "m", outputPath: "/o" },
    );
    const on = buildBenchyArgs(
      { ...baseConfig, enablePrefixCaching: true },
      { baseUrl: "u", modelName: "m", outputPath: "/o" },
    );
    expect(off).not.toContain("--enable-prefix-caching");
    expect(on).toContain("--enable-prefix-caching");
  });

  it("includes --skip-coherence only when enabled", () => {
    const on = buildBenchyArgs(
      { ...baseConfig, skipCoherence: true },
      { baseUrl: "u", modelName: "m", outputPath: "/o" },
    );
    expect(on).toContain("--skip-coherence");
  });

  it("passes latency-mode through", () => {
    const args = buildBenchyArgs(
      { ...baseConfig, latencyMode: "generation" },
      { baseUrl: "u", modelName: "m", outputPath: "/o" },
    );
    expect(valuesAfter(args, "--latency-mode")).toEqual(["generation"]);
  });

  // Invariant: each list flag appears exactly once, and the run of value tokens
  // immediately after it matches the input array length and contents (as strings).
  test.prop([
    fc.array(fc.integer({ min: 1, max: 100000 }), { minLength: 1, maxLength: 5 }),
    fc.array(fc.integer({ min: 1, max: 100000 }), { minLength: 1, maxLength: 5 }),
    fc.array(fc.integer({ min: 0, max: 200000 }), { minLength: 1, maxLength: 5 }),
    fc.array(fc.integer({ min: 1, max: 256 }), { minLength: 1, maxLength: 5 }),
  ])("emits one flag occurrence followed by N value tokens", (pp, tg, depth, concurrency) => {
    const args = buildBenchyArgs(
      { ...baseConfig, pp, tg, depth, concurrency },
      { baseUrl: "u", modelName: "m", outputPath: "/o" },
    );
    const cases: Array<[string, number[]]> = [
      ["--pp", pp],
      ["--tg", tg],
      ["--depth", depth],
      ["--concurrency", concurrency],
    ];
    for (const [flag, values] of cases) {
      // Exactly one occurrence of the flag itself.
      expect(args.filter((a) => a === flag).length).toBe(1);
      // Values immediately after the flag match input as strings.
      expect(valuesAfter(args, flag)).toEqual(values.map(String));
    }
  });
});
