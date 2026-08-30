import { describe, expect, it } from "vitest";
import type { AccuracyMetricInput } from "./lm-eval-parser.js";
import { detectExtractionFailures, extractionFailuresFor } from "./extraction-failure.js";

const row = (over: Partial<AccuracyMetricInput>): AccuracyMetricInput => ({
  task: "gpqa_diamond_cot_zeroshot",
  metric: "exact_match",
  value: 0,
  stderr: null,
  isGroup: false,
  nSamples: 198,
  filter: null,
  ...over,
});

describe("detectExtractionFailures", () => {
  // The case that motivated this: Muse Glimmer answered GPQA questions correctly
  // in prose ("This is choice B") but emitted neither "The answer is " nor a
  // parenthesised letter, so strict-match scored exactly 0.0 across all 198
  // items while flexible-extract returned ~chance. No model that knows a
  // fraction of the answers produces zero strict matches — the zero means the
  // answer was not found, not that it was wrong.
  it("flags a filter at exactly zero when a sibling filter scored above zero", () => {
    const found = detectExtractionFailures([
      row({ filter: "strict-match", value: 0 }),
      row({ filter: "flexible-extract", value: 0.227 }),
    ]);
    expect(found).toHaveLength(1);
    expect(found[0]).toMatchObject({
      task: "gpqa_diamond_cot_zeroshot",
      metric: "exact_match",
      zeroFilters: ["strict-match"],
    });
  });

  // The user-confirmed second shape: nothing was extracted by any filter. A
  // model scoring a clean 0.0 across a whole dataset under every filter is a
  // harness result, not a model result.
  it("flags a metric where every filter returned exactly zero", () => {
    const found = detectExtractionFailures([
      row({ filter: "strict-match", value: 0 }),
      row({ filter: "flexible-extract", value: 0 }),
    ]);
    expect(found).toHaveLength(1);
    expect(found[0].zeroFilters).toEqual(["strict-match", "flexible-extract"]);
    expect(found[0].bestValue).toBe(0);
  });

  it("does not flag a metric where every filter scored above zero", () => {
    expect(
      detectExtractionFailures([
        row({ filter: "strict-match", value: 0.61 }),
        row({ filter: "flexible-extract", value: 0.68 }),
      ]),
    ).toEqual([]);
  });

  // A warning that fires on legitimate zeros trains you to ignore it. A metric
  // with ONE filter at zero carries no evidence either way: nothing scored
  // above zero to prove the answer was extractable. Only a non-zero sibling, or
  // two-plus filters agreeing on zero, distinguishes "not found" from "wrong".
  it("does not flag a lone zero with no sibling to contradict it", () => {
    expect(detectExtractionFailures([row({ filter: "none", value: 0 })])).toEqual([]);
  });

  it("does not compare filters that scored over different sample counts", () => {
    expect(
      detectExtractionFailures([
        row({ filter: "strict-match", value: 0, nSamples: 198 }),
        row({ filter: "flexible-extract", value: 0.5, nSamples: 3 }),
      ]),
    ).toEqual([]);
  });

  it("does not flag a single filter that scored above zero", () => {
    expect(detectExtractionFailures([row({ filter: "none", value: 0.42 })])).toEqual([]);
  });

  it("keeps findings separate per task and per metric", () => {
    const found = detectExtractionFailures([
      row({ task: "gpqa", metric: "exact_match", filter: "strict-match", value: 0 }),
      row({ task: "gpqa", metric: "exact_match", filter: "flexible-extract", value: 0.2 }),
      row({ task: "ifeval", metric: "acc", filter: "none", value: 0.9 }),
    ]);
    expect(found).toHaveLength(1);
    expect(found[0].task).toBe("gpqa");
  });

  it("returns nothing for an empty metric list", () => {
    expect(detectExtractionFailures([])).toEqual([]);
  });
});

describe("extractionFailuresFor (stored runs)", () => {
  // Derived on read rather than persisted, so it applies to results recorded
  // before any of this existed — which is the point, since every stored GPQA
  // number is suspect until checked.
  it("detects a failure in a stored metrics blob", () => {
    const stored = JSON.stringify([
      { task: "gpqa", metric: "exact_match", value: 0, stderr: 0, isGroup: false, nSamples: 198, filter: "strict-match" },
      { task: "gpqa", metric: "exact_match", value: 0.227, stderr: 0.03, isGroup: false, nSamples: 198, filter: "flexible-extract" },
    ]);
    const found = extractionFailuresFor(stored);
    expect(found).toHaveLength(1);
    expect(found[0].bestValue).toBeCloseTo(0.227, 5);
  });

  // Rows written before the filter field existed still carry the two values;
  // the zero-with-non-zero-sibling shape is detectable without filter names.
  it("still detects a failure in rows that predate the filter field", () => {
    const legacy = JSON.stringify([
      { task: "gpqa", metric: "exact_match", value: 0, stderr: 0, isGroup: false, nSamples: 198 },
      { task: "gpqa", metric: "exact_match", value: 0.227, stderr: 0.03, isGroup: false, nSamples: 198 },
    ]);
    expect(extractionFailuresFor(legacy)).toHaveLength(1);
  });

  it("returns nothing for null, malformed, or non-array stored metrics", () => {
    expect(extractionFailuresFor(null)).toEqual([]);
    expect(extractionFailuresFor("not json")).toEqual([]);
    expect(extractionFailuresFor('{"not":"an array"}')).toEqual([]);
  });
});

describe("the incident this was built for", () => {
  // Verbatim accuracyMetrics from the Muse Glimmer GPQA-Diamond run of
  // 2026-08-30 (run cmtfegk26..., 198/198, reported 22.73). The model answered
  // correctly in prose but matched neither extraction filter, so strict-match
  // is a clean 0.0 across every item. If this stops being flagged, the feature
  // has regressed on the exact case that motivated it.
  const glimmer = JSON.stringify([
    { task: "gpqa_diamond_cot_zeroshot", metric: "sample_len", value: 198, stderr: null, isGroup: false, nSamples: 198 },
    { task: "gpqa_diamond_cot_zeroshot", metric: "exact_match", value: 0, stderr: 0, isGroup: false, nSamples: 198 },
    { task: "gpqa_diamond_cot_zeroshot", metric: "exact_match", value: 0.22727272727272727, stderr: 0.02985751567338641, isGroup: false, nSamples: 198 },
  ]);

  it("flags the Glimmer run whose 22.73 was a scoring artefact", () => {
    const found = extractionFailuresFor(glimmer);
    expect(found).toHaveLength(1);
    expect(found[0].metric).toBe("exact_match");
    expect(found[0].bestValue).toBeCloseTo(0.227, 3);
  });

  it("does not flag sample_len, which is a count rather than a score", () => {
    expect(extractionFailuresFor(glimmer).some((f) => f.metric === "sample_len")).toBe(false);
  });
});

