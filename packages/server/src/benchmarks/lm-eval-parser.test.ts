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
});
