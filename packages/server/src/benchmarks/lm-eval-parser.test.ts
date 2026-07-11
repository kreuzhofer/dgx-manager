import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseLmEvalResults } from "./lm-eval-parser.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixture = readFileSync(
  join(__dirname, "../__tests__/integration/benchmarks.fixtures/lm-eval-result.json"),
  "utf-8",
);

describe("parseLmEvalResults", () => {
  it("returns the primary metric ×100 as primaryScore", () => {
    const { primaryScore } = parseLmEvalResults(fixture, "ifeval", "prompt_level_strict_acc");
    expect(primaryScore).toBeCloseTo(42, 5);
  });

  it("emits a breakdown row per numeric metric with paired stderr and n-samples", () => {
    const { metrics } = parseLmEvalResults(fixture, "ifeval", "prompt_level_strict_acc");
    const row = metrics.find((m) => m.task === "ifeval" && m.metric === "prompt_level_strict_acc")!;
    expect(row).toMatchObject({ value: 0.42, stderr: 0.021, isGroup: false, nSamples: 100 });
  });

  it("treats a non-numeric stderr ('N/A') as null", () => {
    const { metrics } = parseLmEvalResults(fixture, "ifeval", "prompt_level_strict_acc");
    const row = metrics.find((m) => m.metric === "inst_level_strict_acc")!;
    expect(row.stderr).toBeNull();
  });

  it("flags group-level rows via the groups object", () => {
    const { metrics } = parseLmEvalResults(fixture, "bbh", "exact_match");
    expect(metrics.find((m) => m.task === "bbh")!.isGroup).toBe(true);
    expect(metrics.find((m) => m.task === "bbh_boolean_expressions")!.isGroup).toBe(false);
  });

  it("never emits a stderr key as its own metric row", () => {
    const { metrics } = parseLmEvalResults(fixture, "ifeval", "prompt_level_strict_acc");
    expect(metrics.some((m) => m.metric.endsWith("_stderr"))).toBe(false);
  });

  it("throws when the primary metric is absent", () => {
    expect(() => parseLmEvalResults(fixture, "ifeval", "nonexistent")).toThrow(/missing/i);
  });

  it("throws on malformed JSON", () => {
    expect(() => parseLmEvalResults("not json", "ifeval", "prompt_level_strict_acc")).toThrow(/parse/i);
  });

  it("throws when the results object is missing", () => {
    expect(() => parseLmEvalResults("{}", "ifeval", "prompt_level_strict_acc")).toThrow(/results/i);
  });

  it("excludes a metric whose value is non-numeric", () => {
    const json = JSON.stringify({
      results: { t: { "acc,none": 0.5, "acc_stderr,none": 0.01, "broken,none": "N/A" } },
    });
    const { metrics } = parseLmEvalResults(json, "t", "acc");
    expect(metrics.some((m) => m.metric === "broken")).toBe(false);
    expect(metrics.some((m) => m.metric === "acc")).toBe(true);
  });

  it("throws when the primary metric value is present but non-numeric", () => {
    const json = JSON.stringify({ results: { t: { "exact_match,none": "N/A" } } });
    expect(() => parseLmEvalResults(json, "t", "exact_match")).toThrow(/missing/i);
  });

  it("emits exactly one row per numeric non-stderr metric (ifeval → 3)", () => {
    const { metrics } = parseLmEvalResults(fixture, "ifeval", "prompt_level_strict_acc");
    expect(metrics.filter((m) => m.task === "ifeval")).toHaveLength(3);
  });

  it("throws when the parsed top-level is not an object", () => {
    expect(() => parseLmEvalResults("null", "t", "acc")).toThrow(/results/i);
  });

  it("falls back to a non-'none' filter for the primary metric (e.g. gsm8k strict-match)", () => {
    const json = JSON.stringify({
      results: { gsm8k_cot: { "exact_match,strict-match": 0.7, "exact_match_stderr,strict-match": 0.01 } },
    });
    const { primaryScore } = parseLmEvalResults(json, "gsm8k_cot", "exact_match");
    expect(primaryScore).toBeCloseTo(70, 5);
  });
});

describe("multi-filter headline selection (GPQA-style)", () => {
  it("prefers flexible-extract over a strict-match 0, not 'the first'", () => {
    const raw = JSON.stringify({ results: { gpqa_diamond_cot_zeroshot: {
      "exact_match,strict-match": 0.0,
      "exact_match_stderr,strict-match": 0.0,
      "exact_match,flexible-extract": 0.6767676767676768,
      "exact_match_stderr,flexible-extract": 0.0333,
      "alias": "gpqa_diamond_cot_zeroshot",
    } } });
    const out = parseLmEvalResults(raw, "gpqa_diamond_cot_zeroshot", "exact_match");
    expect(out.primaryScore).toBeCloseTo(67.6767, 2);
  });

  it("falls back to the highest filter when neither none nor flexible-extract exists", () => {
    const raw = JSON.stringify({ results: { t: {
      "exact_match,strict-match": 0.1,
      "exact_match,custom-extract": 0.4,
    } } });
    const out = parseLmEvalResults(raw, "t", "exact_match");
    expect(out.primaryScore).toBeCloseTo(40, 5);
  });

  it("still prefers ,none when present", () => {
    const raw = JSON.stringify({ results: { t: {
      "exact_match,none": 0.5, "exact_match,flexible-extract": 0.9,
    } } });
    expect(parseLmEvalResults(raw, "t", "exact_match").primaryScore).toBeCloseTo(50, 5);
  });
});
