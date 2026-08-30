import type { AccuracyMetricInput } from "./lm-eval-parser.js";

/**
 * A task/metric whose value looks like a scoring artefact rather than a result.
 *
 * lm-eval extracts a model's answer with regex filters, and those filters only
 * match particular phrasings: `strict-match` wants the literal "The answer is ",
 * `flexible-extract` wants a parenthesised capital. A model that answers
 * correctly in prose matches neither and scores zero. The run completes and
 * reports a precise-looking number that measures the model's phrasing habits
 * rather than its knowledge.
 */
export type ExtractionFailure = {
  task: string;
  metric: string;
  /** Filters that returned exactly 0. Carries the parser's own representation -
   *  null for a metric with no `,<filter>` suffix - rather than re-encoding it,
   *  so "unnamed filter" cannot be confused with "every filter". */
  zeroFilters: (string | null)[];
  /** Highest value any filter of this metric reached; 0 when all of them failed. */
  bestValue: number;
  /** Filter that reached bestValue, so a caller can name it rather than
   *  reporting a bare number under a label that promises a name. */
  bestFilter: string | null;
};

/**
 * Find metrics whose zeros are better explained by failed extraction than by the
 * model getting answers wrong.
 *
 * Two shapes qualify:
 *   - a filter at exactly 0 alongside a sibling filter above 0, which means the
 *     answer was there and one filter could not see it;
 *   - two or more filters all at exactly 0, where nothing was extracted at all.
 *
 * A LONE zero qualifies as neither. With one filter and no sibling there is no
 * evidence the answer was extractable at all, so reporting it would fire on
 * every legitimately-zero metric - and a warning that cries wolf is worse than
 * no warning, since this one exists precisely to be trusted.
 *
 * Exactly 0 is the signal rather than "low": a model that knows even a fraction
 * of the answers still produces some matches, so a clean zero over a whole
 * dataset is a harness result, not a model result.
 *
 * Filters are only compared within the same sample count - a value measured over
 * 3 items says nothing about one measured over 198.
 *
 * Pure over parsed metrics, so it applies to an already-stored result without
 * re-running anything.
 */
export function detectExtractionFailures(
  metrics: AccuracyMetricInput[],
): ExtractionFailure[] {
  const groups = new Map<string, AccuracyMetricInput[]>();
  for (const m of metrics) {
    const key = `${m.task}\u0000${m.metric}\u0000${m.nSamples ?? ""}`;
    const g = groups.get(key);
    if (g) g.push(m);
    else groups.set(key, [m]);
  }

  const out: ExtractionFailure[] = [];
  for (const rows of groups.values()) {
    const zeros = rows.filter((r) => r.value === 0);
    if (zeros.length === 0) continue;
    // Either a sibling proves the answer was extractable, or several filters
    // agree they found nothing. One filter alone proves neither.
    const hasNonZeroSibling = zeros.length < rows.length;
    if (!hasNonZeroSibling && rows.length < 2) continue;
    const best = rows.reduce((a, b) => (b.value > a.value ? b : a));
    out.push({
      task: rows[0].task,
      metric: rows[0].metric,
      zeroFilters: zeros.map((r) => r.filter),
      bestValue: best.value,
      bestFilter: best.filter,
    });
  }
  return out;
}

/**
 * Derive failures from a stored `BenchmarkRun.accuracyMetrics` blob.
 *
 * Deliberately computed on read instead of persisted: it needs no migration and
 * it applies to results recorded before any of this existed, which matters
 * because a stored score gives no hint that its extraction failed.
 *
 * Defensive like the dashboard's own parse - a malformed value degrades to "no
 * findings" rather than breaking the response.
 */
export function extractionFailuresFor(
  accuracyMetrics: string | null,
): ExtractionFailure[] {
  if (!accuracyMetrics) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(accuracyMetrics);
  } catch {
    return [];
  }
  if (!Array.isArray(parsed)) return [];
  // Rows written before `filter` existed simply lack it; normalise so grouping
  // and reporting behave the same either way.
  const rows = parsed
    .filter((r): r is Record<string, unknown> => !!r && typeof r === "object")
    .filter((r) => typeof r.task === "string" && typeof r.metric === "string" && typeof r.value === "number")
    .map((r) => ({
      task: r.task as string,
      metric: r.metric as string,
      value: r.value as number,
      stderr: typeof r.stderr === "number" ? r.stderr : null,
      isGroup: r.isGroup === true,
      nSamples: typeof r.nSamples === "number" ? r.nSamples : null,
      filter: typeof r.filter === "string" ? r.filter : null,
    }));
  return detectExtractionFailures(rows);
}

