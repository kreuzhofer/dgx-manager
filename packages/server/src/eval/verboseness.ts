/**
 * Verboseness eval — pure scoring logic (no IO).
 *
 * There is no ground-truth "ideal length" for a free-form response, so this
 * eval measures *how much a model emits* per task and per thinking-mode, and
 * flags responses that blow past an absolute token budget. The headline signal
 * is the thinking-overhead ratio (mean ON tokens / mean OFF tokens) — useful
 * because, empirically, "thinking = more verbose" is NOT a reliable rule for
 * Nemotron 3 Ultra (on open-ended prompts, thinking-OFF can ramble *more*).
 *
 * IO (calling the endpoint, gathering samples) lives in run-verboseness.ts;
 * everything here is pure so it can be unit/property-tested without mocks.
 */

export type ThinkMode = "on" | "off" | "medium";

export interface ResponseLengths {
  completionTokens: number;
  reasoningChars: number;
  contentChars: number;
}

export interface ChatMessageLike {
  content?: string | null;
  /** vLLM nemotron_v3 surfaces the reasoning trace here. */
  reasoning?: string | null;
  /** OpenAI convention; some parsers use this instead. */
  reasoning_content?: string | null;
}

export interface UsageLike {
  completion_tokens?: number | null;
}

/**
 * Extract response lengths from an OpenAI-style chat message + usage block.
 * Reads whichever reasoning field is present, preferring `reasoning`.
 */
export function parseResponseLengths(
  message: ChatMessageLike,
  usage: UsageLike | undefined,
): ResponseLengths {
  const reasoning = message?.reasoning ?? message?.reasoning_content ?? "";
  const content = message?.content ?? "";
  const raw = usage?.completion_tokens ?? 0;
  const completionTokens = Number.isFinite(raw) ? Math.max(0, Math.trunc(raw as number)) : 0;
  return {
    completionTokens,
    reasoningChars: (reasoning ?? "").length,
    contentChars: (content ?? "").length,
  };
}

export interface VerbSample {
  prompt: string;
  mode: ThinkMode;
  lengths: ResponseLengths;
}

export interface VerbThresholds {
  /** Responses whose completion tokens exceed this are flagged over-budget. */
  maxCompletionTokens: number;
}

export interface VerbVerdict extends ResponseLengths {
  prompt: string;
  mode: ThinkMode;
  overBudget: boolean;
}

export interface VerbSummary {
  verdicts: VerbVerdict[];
  meanTokensByMode: Record<ThinkMode, number | null>;
  overBudgetCount: number;
  /** mean ON completion tokens / mean OFF completion tokens, or null if either is missing/zero. */
  thinkingOverheadRatio: number | null;
}

const MODES: ThinkMode[] = ["on", "off", "medium"];

export function evaluateVerboseness(
  samples: VerbSample[],
  thresholds: VerbThresholds,
): VerbSummary {
  const verdicts: VerbVerdict[] = samples.map((s) => ({
    prompt: s.prompt,
    mode: s.mode,
    completionTokens: s.lengths.completionTokens,
    reasoningChars: s.lengths.reasoningChars,
    contentChars: s.lengths.contentChars,
    overBudget: s.lengths.completionTokens > thresholds.maxCompletionTokens,
  }));

  const meanFor = (mode: ThinkMode): number | null => {
    const xs = samples.filter((s) => s.mode === mode).map((s) => s.lengths.completionTokens);
    if (xs.length === 0) return null;
    return xs.reduce((a, b) => a + b, 0) / xs.length;
  };

  const meanTokensByMode = Object.fromEntries(
    MODES.map((m) => [m, meanFor(m)]),
  ) as Record<ThinkMode, number | null>;

  const overBudgetCount = verdicts.filter((v) => v.overBudget).length;

  const on = meanTokensByMode.on;
  const off = meanTokensByMode.off;
  const thinkingOverheadRatio = on !== null && off !== null && off > 0 ? on / off : null;

  return { verdicts, meanTokensByMode, overBudgetCount, thinkingOverheadRatio };
}
