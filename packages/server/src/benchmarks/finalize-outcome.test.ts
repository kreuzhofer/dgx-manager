import { describe, it, expect } from "vitest";
import { it as fcit } from "@fast-check/vitest";
import * as fc from "fast-check";
import { decideFinalize } from "./finalize-outcome.js";

describe("decideFinalize", () => {
  it("completes a throughput run that exited 0 with rows", () => {
    expect(
      decideFinalize({ tool: "llama-benchy", exitCode: 0, hasSummary: true, rowCount: 12 }),
    ).toEqual({ kind: "complete" });
  });

  // The defect in #95: llama-benchy can exit 0 having produced nothing —
  // every request failed, or the result file was missing so parsing yielded
  // an empty array. That used to be recorded as `completed`, and two such
  // runs compare as perfect agreement.
  it("FAILS a run that exited 0 but produced no result rows", () => {
    const out = decideFinalize({
      tool: "llama-benchy", exitCode: 0, hasSummary: true, rowCount: 0,
    });
    expect(out.kind).toBe("fail");
    expect(out).toMatchObject({ reason: expect.stringContaining("no result rows") });
  });

  it("fails on a non-zero exit, naming the tool and code", () => {
    expect(
      decideFinalize({ tool: "lm-eval", exitCode: 3, hasSummary: false }),
    ).toEqual({ kind: "fail", reason: "lm-eval exited with code 3" });
  });

  // dispatch() resolves exitCode null when the child was killed rather than
  // exiting — a cancel, or an OOM. Must not be read as success.
  it("fails on a null exit code", () => {
    expect(decideFinalize({ tool: "llama-benchy", exitCode: null, hasSummary: true, rowCount: 5 }).kind)
      .toBe("fail");
  });

  it("fails with the parse error when there is no summary to fall back on", () => {
    expect(
      decideFinalize({
        tool: "lm-eval", exitCode: 0, hasSummary: false, parseError: "no primary metric in results",
      }),
    ).toEqual({ kind: "fail", reason: "no primary metric in results" });
  });

  // Precedence preserved from the pre-existing finalizeAccuracy behaviour:
  // a summary wins over a reported parse error.
  it("completes when a summary exists even if a parse error was reported", () => {
    expect(
      decideFinalize({
        tool: "lm-eval", exitCode: 0, hasSummary: true, parseError: "partial parse warning",
      }),
    ).toEqual({ kind: "complete" });
  });

  it("says 'no summary' rather than 'exited with code 0' when nothing parsed", () => {
    const out = decideFinalize({ tool: "tool-eval-bench", exitCode: 0, hasSummary: false });
    expect(out).toMatchObject({ reason: expect.stringContaining("no summary") });
    expect(out).not.toMatchObject({ reason: expect.stringContaining("exited with code 0") });
  });

  // Runners with no row set (accuracy, tool-eval) pass no rowCount at all;
  // an empty-rows check would be meaningless for them.
  it("ignores the row check when the runner has no row set", () => {
    expect(
      decideFinalize({ tool: "lm-eval", exitCode: 0, hasSummary: true }),
    ).toEqual({ kind: "complete" });
  });
});

/**
 * The invariant the whole ticket is about: a run is only ever COMPLETE when the
 * process exited 0 *and* it produced something to compare — a summary, plus at
 * least one row whenever the runner reports rows at all. No combination of
 * inputs may complete a run that produced nothing.
 */
fcit.prop([
  fc.string({ minLength: 1, maxLength: 12 }),
  fc.oneof(fc.constant(null), fc.integer({ min: -8, max: 8 })),
  fc.boolean(),
  fc.oneof(fc.constant(undefined), fc.integer({ min: 0, max: 50 })),
  fc.oneof(fc.constant(undefined), fc.constant(null), fc.string({ maxLength: 20 })),
])(
  "completes only when exited 0, a summary exists, and rows are non-empty when reported",
  (tool, exitCode, hasSummary, rowCount, parseError) => {
    const out = decideFinalize({ tool, exitCode, hasSummary, rowCount, parseError });
    if (out.kind === "complete") {
      expect(exitCode).toBe(0);
      expect(hasSummary).toBe(true);
      if (rowCount !== undefined) expect(rowCount).toBeGreaterThan(0);
    } else {
      expect(out.reason.length).toBeGreaterThan(0);
    }
  },
);
