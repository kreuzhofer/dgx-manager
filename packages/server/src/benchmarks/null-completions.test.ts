import { describe, expect, it } from "vitest";
import { fc, it as itProp } from "@fast-check/vitest";
import type { AccuracyMetricInput } from "./lm-eval-parser.js";
import {
  assessNullCompletions,
  countNullCompletions,
  headlineRowFor,
  isNullCompletionLine,
  nullCompletionFindingFor,
} from "./null-completions.js";

// Verbatim from /mnt/tank/logs/benchmarks — note the two different source line
// numbers, which is exactly why the matcher ignores that part.
const REAL_546 =
  "2026-08-29:18:58:38 WARNING  [models.api_models:546] API returned null content. " +
  "Content filled with `LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = `. Check reasoning_content field or generation limits.";
const REAL_789 =
  "2026-07-10:07:47:06 WARNING  [models.api_models:789] API returned null content. " +
  "Content filled with `LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = `. Check reasoning_content field or generation limits.";

const row = (over: Partial<AccuracyMetricInput>): AccuracyMetricInput => ({
  task: "gpqa_diamond_cot_zeroshot",
  metric: "exact_match",
  value: 0.8131,
  stderr: 0.0277,
  isGroup: false,
  nSamples: 198,
  filter: "flexible-extract",
  ...over,
});

describe("isNullCompletionLine", () => {
  it("matches lm-eval's warning regardless of the source line number", () => {
    // The line number moved 546 -> 789 between versions we have logs from.
    // Anchoring on it would silently stop counting after an upgrade, which is
    // the same class of silent failure this module exists to catch.
    expect(isNullCompletionLine(REAL_546)).toBe(true);
    expect(isNullCompletionLine(REAL_789)).toBe(true);
  });

  it("ignores ordinary log lines", () => {
    expect(isNullCompletionLine("2026-08-29:18:58:38 INFO Running loglikelihood requests")).toBe(false);
    expect(isNullCompletionLine("")).toBe(false);
    // Must not fire on a line that merely mentions the placeholder, e.g. a
    // model echoing it back inside a sample dump.
    expect(isNullCompletionLine("sample: LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = ")).toBe(false);
  });
});

describe("countNullCompletions", () => {
  it("counts one per occurrence across a log", () => {
    const log = ["INFO start", REAL_546, "INFO progress", REAL_789, "INFO done"].join("\n");
    expect(countNullCompletions(log)).toBe(2);
  });

  it("returns 0 for a clean log and for empty input", () => {
    expect(countNullCompletions("INFO start\nINFO done")).toBe(0);
    expect(countNullCompletions("")).toBe(0);
  });
});

describe("assessNullCompletions", () => {
  /**
   * The incident this issue was filed for: GPQA-Diamond on GLM-5.3-Flash,
   * 2026-08-30. 23 of 198 items empty at a 32,768 cap, score rendered as a
   * clean 81.31. The empties are 11.6 points of possible uplift against a
   * 2.77-point stderr, so the score is a floor and must not be quoted bare.
   */
  it("calls the GPQA incident material — uplift dwarfs the stderr", () => {
    const f = assessNullCompletions({ count: 23, nSamples: 198, stderr: 0.0277 });
    expect(f).not.toBeNull();
    expect(f!.count).toBe(23);
    expect(f!.share).toBeCloseTo(0.1162, 4);
    expect(f!.maxUpliftPoints).toBeCloseTo(11.62, 2);
    expect(f!.severity).toBe("material");
  });

  it("calls a handful out of 1319 minor — inside the noise", () => {
    // 1 empty in a GSM8K full run cannot move a number whose stderr is 0.5 pts.
    const f = assessNullCompletions({ count: 1, nSamples: 1319, stderr: 0.005 });
    expect(f!.maxUpliftPoints).toBeCloseTo(0.0758, 3);
    expect(f!.severity).toBe("minor");
  });

  it("reports an unquantifiable caveat as material rather than assuming it away", () => {
    // No stderr means we cannot show the empties are harmless. Surfacing is the
    // safe direction; silently downgrading would recreate the original bug.
    expect(assessNullCompletions({ count: 5, nSamples: 198, stderr: null })!.severity).toBe("material");
    expect(assessNullCompletions({ count: 5, nSamples: null, stderr: 0.02 })!.severity).toBe("material");
  });

  it("distinguishes 'none were empty' from 'never measured'", () => {
    // Both yield no finding, but for different reasons — and a null count must
    // never be reported as a measured zero.
    expect(assessNullCompletions({ count: 0, nSamples: 198, stderr: 0.02 })).toBeNull();
    expect(assessNullCompletions({ count: null, nSamples: 198, stderr: 0.02 })).toBeNull();
  });

  /**
   * Invariant: severity is "minor" only when the whole caveat fits inside one
   * stderr. Anything a reader could mistake for signal is reported.
   */
  itProp.prop([
    fc.integer({ min: 1, max: 500 }),
    fc.integer({ min: 1, max: 2000 }),
    fc.double({ min: 0.0001, max: 0.5, noNaN: true }),
  ])("minor implies the uplift is within one stderr", (count, nSamples, stderr) => {
    const f = assessNullCompletions({ count, nSamples, stderr });
    if (f === null) return true;
    return f.severity === "minor" ? (f.maxUpliftPoints ?? Infinity) <= stderr * 100 : true;
  });
});

describe("headlineRowFor", () => {
  it("picks the row the headline score came from, not an arbitrary one", () => {
    // strict-match at 0.0 and flexible-extract at 0.8131 both exist; judging the
    // caveat against strict-match's stderr would compare it to the wrong number.
    const strict = row({ filter: "strict-match", value: 0, stderr: 0.0 });
    const flexible = row({ filter: "flexible-extract", value: 0.8131, stderr: 0.0277 });
    expect(headlineRowFor(81.31, [strict, flexible])).toBe(flexible);
  });

  it("ignores group rows and falls back to the widest-sampled row", () => {
    const group = row({ isGroup: true, nSamples: 5000, value: 0.5 });
    const narrow = row({ nSamples: 50, value: 0.4 });
    const wide = row({ nSamples: 198, value: 0.3 });
    expect(headlineRowFor(null, [group, narrow, wide])).toBe(wide);
  });

  it("returns null when there is nothing to judge against", () => {
    expect(headlineRowFor(81.31, [])).toBeNull();
  });
});

describe("nullCompletionFindingFor", () => {
  it("reproduces the GPQA incident from what a run row actually stores", () => {
    const metrics = JSON.stringify([
      row({ filter: "strict-match", value: 0, stderr: 0 }),
      row({ filter: "flexible-extract", value: 0.8131, stderr: 0.0277 }),
    ]);
    const f = nullCompletionFindingFor(23, 81.31, metrics);
    expect(f).toMatchObject({ count: 23, nSamples: 198, severity: "material" });
    expect(f!.maxUpliftPoints).toBeCloseTo(11.62, 2);
  });

  it("says nothing for a run that predates counting", () => {
    expect(nullCompletionFindingFor(null, 81.31, "[]")).toBeNull();
  });

  it("still reports the count when the metrics blob is unusable", () => {
    // Malformed JSON must not swallow the caveat — we know empties happened.
    const f = nullCompletionFindingFor(4, 90, "not json");
    expect(f).toMatchObject({ count: 4, nSamples: null, severity: "material" });
  });
});
