export type AccuracyMetricInput = {
  task: string;
  metric: string;      // metric name without the ",<filter>" suffix
  value: number;       // raw lm-eval value (0–1 for accuracies)
  stderr: number | null;
  isGroup: boolean;
  nSamples: number | null;
};

export type LmEvalSummary = {
  primaryScore: number; // primary metric ×100 (0–100)
  metrics: AccuracyMetricInput[];
};

type Obj = Record<string, unknown>;

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// Parse an lm-evaluation-harness results JSON. Metric keys look like
// "<metric>,<filter>" (e.g. "exact_match,none") with a sibling
// "<metric>_stderr,<filter>". We split on the LAST comma to recover the filter,
// skip *_stderr keys as standalone metrics (they attach to their base metric),
// and skip the "alias" string. Group-level tasks are flagged via the top-level
// `groups` object. Fail-fast: the primary metric must be present and numeric.
export function parseLmEvalResults(
  jsonText: string,
  primaryTask: string,
  primaryMetric: string,
): LmEvalSummary {
  let parsed: Obj;
  try {
    parsed = JSON.parse(jsonText) as Obj;
  } catch (e) {
    throw new Error(`failed to parse lm-eval JSON: ${(e as Error).message}`);
  }

  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("lm-eval result missing required object: results");
  }

  const results = parsed.results;
  if (!results || typeof results !== "object") {
    throw new Error("lm-eval result missing required object: results");
  }
  const groups = (parsed.groups && typeof parsed.groups === "object" ? parsed.groups : {}) as Obj;
  const nSamplesAll = (parsed["n-samples"] && typeof parsed["n-samples"] === "object"
    ? parsed["n-samples"]
    : {}) as Obj;

  const metrics: AccuracyMetricInput[] = [];
  for (const [task, taskVal] of Object.entries(results as Obj)) {
    if (!taskVal || typeof taskVal !== "object") continue;
    const entry = taskVal as Obj;
    const isGroup = Object.prototype.hasOwnProperty.call(groups, task);
    const nInfo = nSamplesAll[task] as Obj | undefined;
    const nSamples = nInfo ? numOrNull(nInfo.effective) : null;

    for (const [key, value] of Object.entries(entry)) {
      if (key === "alias") continue;
      const comma = key.lastIndexOf(",");
      const metricName = comma === -1 ? key : key.slice(0, comma);
      if (metricName.endsWith("_stderr")) continue;
      const numeric = numOrNull(value);
      if (numeric === null) continue;

      const filter = comma === -1 ? "" : key.slice(comma);
      const stderrKey = `${metricName}_stderr${filter}`;
      metrics.push({
        task,
        metric: metricName,
        value: numeric,
        stderr: numOrNull(entry[stderrKey]),
        isGroup,
        nSamples,
      });
    }
  }

  // Headline: prefer the `,none` filter, else `,flexible-extract`, else the
  // highest-scoring filter. gsm8k/mmlu_pro/gpqa report exact_match under multiple
  // extraction filters — `,strict-match` is often 0.0 because a reasoning model
  // doesn't hit the exact template, while `,flexible-extract` is the meaningful
  // number (GPQA-Diamond: strict 0.0 vs flexible 0.68). Taking "the first" filter
  // let a format-mismatch 0 masquerade as the score. Every filter stays in
  // `metrics`; fail fast only if the metric is absent under every filter.
  const primaryEntry = (results as Obj)[primaryTask] as Obj | undefined;
  let primary: number | null = null;
  if (primaryEntry) {
    const keys = Object.keys(primaryEntry).filter(
      (k) => (k === primaryMetric || k.startsWith(`${primaryMetric},`)) && !k.includes("_stderr"),
    );
    const chosen =
      keys.find((k) => k === `${primaryMetric},none`) ??
      keys.find((k) => k.endsWith(",flexible-extract")) ??
      keys.slice().sort((a, b) => (numOrNull(primaryEntry[b]) ?? -1) - (numOrNull(primaryEntry[a]) ?? -1))[0];
    if (chosen !== undefined) primary = numOrNull(primaryEntry[chosen]);
  }
  if (primary === null) {
    const row = metrics.find((m) => m.task === primaryTask && m.metric === primaryMetric);
    primary = row ? row.value : null;
  }
  if (primary === null) {
    throw new Error(
      `lm-eval result missing primary metric: ${primaryTask}/${primaryMetric}`,
    );
  }

  return { primaryScore: primary * 100, metrics };
}
