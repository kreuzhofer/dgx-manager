import { describe, it, expect } from "vitest";
import { it as fcIt } from "@fast-check/vitest";
import { fc } from "@fast-check/vitest";
import {
  parseResponseLengths,
  evaluateVerboseness,
  type VerbSample,
  type ThinkMode,
} from "./verboseness.js";

describe("parseResponseLengths", () => {
  it("reads completion_tokens and content length", () => {
    const r = parseResponseLengths({ content: "hello" }, { completion_tokens: 12 });
    expect(r).toEqual({ completionTokens: 12, reasoningChars: 0, contentChars: 5 });
  });

  // vLLM's nemotron_v3 parser surfaces the trace as `reasoning`, while the
  // OpenAI convention is `reasoning_content`. We must read whichever is present,
  // preferring `reasoning`.
  it("prefers `reasoning` over `reasoning_content`", () => {
    const r = parseResponseLengths(
      { content: "ans", reasoning: "abcd", reasoning_content: "ignored-longer" },
      { completion_tokens: 9 },
    );
    expect(r.reasoningChars).toBe(4);
  });

  it("falls back to `reasoning_content` when `reasoning` is absent", () => {
    const r = parseResponseLengths(
      { content: "ans", reasoning_content: "abc" },
      { completion_tokens: 9 },
    );
    expect(r.reasoningChars).toBe(3);
  });

  it("treats null/missing content, reasoning, and usage as zero", () => {
    expect(parseResponseLengths({ content: null }, undefined)).toEqual({
      completionTokens: 0,
      reasoningChars: 0,
      contentChars: 0,
    });
  });
});

describe("evaluateVerboseness", () => {
  const mk = (prompt: string, mode: ThinkMode, completionTokens: number): VerbSample => ({
    prompt,
    mode,
    lengths: { completionTokens, reasoningChars: 0, contentChars: 0 },
  });

  it("flags a sample over the token budget and counts it", () => {
    const s = evaluateVerboseness(
      [mk("p", "on", 600), mk("p", "off", 100)],
      { maxCompletionTokens: 512 },
    );
    expect(s.verdicts.find((v) => v.mode === "on")!.overBudget).toBe(true);
    expect(s.verdicts.find((v) => v.mode === "off")!.overBudget).toBe(false);
    expect(s.overBudgetCount).toBe(1);
  });

  it("computes per-mode mean tokens, null for an absent mode", () => {
    const s = evaluateVerboseness(
      [mk("a", "on", 400), mk("b", "on", 200), mk("a", "off", 100)],
      { maxCompletionTokens: 9999 },
    );
    expect(s.meanTokensByMode.on).toBe(300);
    expect(s.meanTokensByMode.off).toBe(100);
    expect(s.meanTokensByMode.medium).toBeNull();
  });

  it("reports the thinking overhead ratio (mean ON / mean OFF)", () => {
    const s = evaluateVerboseness(
      [mk("a", "on", 420), mk("a", "off", 210)],
      { maxCompletionTokens: 9999 },
    );
    expect(s.thinkingOverheadRatio).toBeCloseTo(2.0);
  });

  it("ratio is null when OFF samples are missing", () => {
    const s = evaluateVerboseness([mk("a", "on", 420)], { maxCompletionTokens: 9999 });
    expect(s.thinkingOverheadRatio).toBeNull();
  });

  // Invariant: a sample is over budget iff its completion tokens exceed the
  // threshold, and overBudgetCount is exactly the number of such verdicts.
  fcIt.prop([
    fc.array(
      fc.record({
        prompt: fc.string(),
        mode: fc.constantFrom<ThinkMode>("on", "off", "medium"),
        tokens: fc.nat({ max: 5000 }),
      }),
    ),
    fc.nat({ max: 5000 }),
  ])("overBudget matches the threshold exactly", (rows, threshold) => {
    const samples: VerbSample[] = rows.map((r) => ({
      prompt: r.prompt,
      mode: r.mode,
      lengths: { completionTokens: r.tokens, reasoningChars: 0, contentChars: 0 },
    }));
    const s = evaluateVerboseness(samples, { maxCompletionTokens: threshold });
    for (const v of s.verdicts) {
      expect(v.overBudget).toBe(v.completionTokens > threshold);
    }
    expect(s.overBudgetCount).toBe(s.verdicts.filter((v) => v.completionTokens > threshold).length);
  });

  // Invariant: raising the threshold can never flag *more* responses as
  // over-budget (monotonic non-increasing).
  fcIt.prop([fc.array(fc.nat({ max: 5000 }), { minLength: 1 }), fc.nat({ max: 5000 })])(
    "over-budget count is non-increasing as the threshold rises",
    (tokens, base) => {
      const samples: VerbSample[] = tokens.map((t, i) => ({
        prompt: `p${i}`,
        mode: "on" as ThinkMode,
        lengths: { completionTokens: t, reasoningChars: 0, contentChars: 0 },
      }));
      const low = evaluateVerboseness(samples, { maxCompletionTokens: base });
      const high = evaluateVerboseness(samples, { maxCompletionTokens: base + 1000 });
      expect(high.overBudgetCount).toBeLessThanOrEqual(low.overBudgetCount);
    },
  );
});
