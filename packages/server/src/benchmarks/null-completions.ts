import type { AccuracyMetricInput } from "./lm-eval-parser.js";

/**
 * Completions that came back EMPTY and were scored as wrong.
 *
 * When a reasoning model overruns its token budget the API returns
 * `finish_reason: "length"` with both `content` and `reasoning_content` empty.
 * lm-eval substitutes a placeholder and scores the item wrong — it does not
 * error, and the run completes with a precise-looking number that is really a
 * FLOOR. Measured on a GPQA-Diamond run 2026-08-30: 23 of 198 items (11.6%)
 * came back empty at a 32,768-token cap and the score rendered as a clean 81.31.
 *
 * A bigger cap does not fix this, it only moves the cliff — a model whose
 * `reasoning_effort` defaults high can overrun any cap — so detection is the
 * only real defence. See #20 §2.
 */

/**
 * lm-eval's warning for an empty completion.
 *
 * Deliberately matched on the message, NOT on the `[models.api_models:NNN]`
 * prefix: that source line number moves between lm-eval versions (546 and 789
 * both appear in our own logs), so anchoring on it would silently stop counting
 * after an upgrade — the exact failure mode this whole module exists to prevent.
 */
const NULL_CONTENT_MESSAGE = "API returned null content";

/** The sentinel lm-eval substitutes for the missing text, which then gets scored. */
export const LM_EVAL_NULL_PLACEHOLDER = "LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER";

export function isNullCompletionLine(line: string): boolean {
  return line.includes(NULL_CONTENT_MESSAGE);
}

/** Count empty completions in a whole run log. */
export function countNullCompletions(logText: string): number {
  if (!logText) return 0;
  let n = 0;
  for (const line of logText.split("\n")) if (isNullCompletionLine(line)) n++;
  return n;
}

export type NullCompletionFinding = {
  /** Items that returned empty and were scored wrong. */
  count: number;
  /** Items scored, when known. */
  nSamples: number | null;
  /** count / nSamples — the share of the result that is not a measurement. */
  share: number | null;
  /**
   * How much the reported score could rise if every empty item were actually
   * answered correctly. The score is a floor; this is the distance to the
   * ceiling, in the same 0–100 units as `accuracyScore`.
   */
  maxUpliftPoints: number | null;
  /**
   * "material" — the uplift exceeds the metric's own stderr, so the empties
   * could move the number by more than its noise. Do not quote the score
   * without the caveat.
   *
   * "minor" — the uplift is inside the stderr, so the empties cannot change
   * the conclusion.
   *
   * Unknown stderr resolves to "material" deliberately: an unquantifiable
   * caveat is reported rather than assumed away.
   */
  severity: "minor" | "material";
};

/**
 * Judge a null-completion count against the result it affected.
 *
 * Returns null when there is nothing to say — either nothing was empty, or the
 * run predates counting. Those two are NOT the same and must not collapse: a
 * `count` of null means "never measured", and reporting that as zero would
 * assert something we did not check.
 */
export function assessNullCompletions(input: {
  count: number | null;
  nSamples: number | null;
  stderr: number | null;
}): NullCompletionFinding | null {
  const { count } = input;
  if (count === null || !Number.isFinite(count) || count <= 0) return null;

  const nSamples =
    typeof input.nSamples === "number" && Number.isFinite(input.nSamples) && input.nSamples > 0
      ? input.nSamples
      : null;
  const share = nSamples === null ? null : count / nSamples;
  const maxUpliftPoints = share === null ? null : share * 100;

  const stderrPoints =
    typeof input.stderr === "number" && Number.isFinite(input.stderr) && input.stderr > 0
      ? input.stderr * 100
      : null;

  const severity: "minor" | "material" =
    maxUpliftPoints !== null && stderrPoints !== null && maxUpliftPoints <= stderrPoints
      ? "minor"
      : "material";

  return { count, nSamples, share, maxUpliftPoints, severity };
}

/**
 * Pick the metric row the headline score came from, so the caveat is judged
 * against the number people actually quote rather than an arbitrary row.
 *
 * Matches on value because the stored rows do not record which one was chosen
 * as primary; the parser's preference order lives in `parseLmEvalResults`.
 */
export function headlineRowFor(
  accuracyScore: number | null,
  metrics: AccuracyMetricInput[],
): AccuracyMetricInput | null {
  const rows = metrics.filter((m) => !m.isGroup);
  if (rows.length === 0) return null;
  if (accuracyScore !== null && Number.isFinite(accuracyScore)) {
    const match = rows.find((m) => Math.abs(m.value * 100 - accuracyScore) < 1e-6);
    if (match) return match;
  }
  // No score to match (a failed run) — fall back to the widest-sampled row so
  // `nSamples` is still usable for the share.
  return rows.reduce((best, m) => ((m.nSamples ?? 0) > (best.nSamples ?? 0) ? m : best), rows[0]);
}

/** Derive the finding from what a BenchmarkRun row stores. */
export function nullCompletionFindingFor(
  nullCompletions: number | null,
  accuracyScore: number | null,
  accuracyMetrics: string | null,
): NullCompletionFinding | null {
  if (nullCompletions === null || nullCompletions <= 0) return null;
  let rows: AccuracyMetricInput[] = [];
  if (accuracyMetrics) {
    try {
      const parsed: unknown = JSON.parse(accuracyMetrics);
      if (Array.isArray(parsed)) rows = parsed as AccuracyMetricInput[];
    } catch {
      rows = [];
    }
  }
  const headline = headlineRowFor(accuracyScore, rows);
  return assessNullCompletions({
    count: nullCompletions,
    nSamples: headline?.nSamples ?? null,
    stderr: headline?.stderr ?? null,
  });
}
