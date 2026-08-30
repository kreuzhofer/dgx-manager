import type { AccuracyMetric, BenchmarkRun } from "@/lib/benchmarks";

// Defensive parse: accuracyMetrics is server-written JSON; a malformed value
// degrades to "no breakdown" rather than blanking the detail page.
function parseMetrics(raw: string | null): AccuracyMetric[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as AccuracyMetric[]) : [];
  } catch {
    return [];
  }
}

export function AccuracyResultCard({ run }: { run: BenchmarkRun }) {
  const metrics = parseMetrics(run.accuracyMetrics);

  const failures = run.extractionFailures ?? [];

  return (
    <div className="space-y-6">
      {failures.length > 0 && (
        <div className="rounded border border-amber-700/60 bg-amber-950/40 p-3 text-sm">
          <div className="font-medium text-amber-300">
            Answer extraction may have failed &mdash; this score may not reflect the model
          </div>
          <div className="mt-1 text-amber-200/80">
            lm-eval finds the answer with regex filters that expect particular phrasings
            (&ldquo;The answer is (C)&rdquo;). A model that answers correctly in different
            words scores zero. A filter at exactly 0.0 across a whole dataset means the
            answer was not found, not that it was wrong.
          </div>
          <ul className="mt-2 space-y-0.5 font-mono text-xs text-amber-200/70">
            {failures.map((f, i) => (
              <li key={`${f.task}-${f.metric}-${i}`}>
                {f.task} / {f.metric}:{" "}
                {f.zeroFilters.map((n) => n ?? "unnamed filter").join(", ")} at 0.0
                {f.bestValue > 0 &&
                  ` (best: ${f.bestFilter ?? "unnamed filter"} at ${(f.bestValue * 100).toFixed(1)})`}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div>
        <div className="text-5xl font-semibold">
          {run.accuracyScore != null ? run.accuracyScore.toFixed(1) : "—"}
          <span className="text-xl text-gray-500">/100</span>
        </div>
        <div className="text-sm text-gray-400 mt-1">Primary metric</div>
      </div>

      <div className="space-y-1">
        <div className="text-sm font-medium text-gray-300">Per-task breakdown</div>
        {metrics.length === 0 && <div className="text-sm text-gray-500">No metric data.</div>}
        {metrics.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="text-left text-xs text-gray-500 uppercase tracking-wide border-b border-gray-800">
                <tr>
                  <th className="px-3 py-2">Task</th>
                  <th className="px-3 py-2">Metric</th>
                  <th className="px-3 py-2">Filter</th>
                  <th className="px-3 py-2 text-right">Value</th>
                  <th className="px-3 py-2 text-right">± stderr</th>
                  <th className="px-3 py-2 text-right">n</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m, i) => (
                  <tr key={`${m.task}-${m.metric}-${i}`} className="border-b border-gray-800 last:border-b-0">
                    <td className={`px-3 py-2 ${m.isGroup ? "font-medium" : "text-gray-400 pl-6"}`}>{m.task}</td>
                    <td className="px-3 py-2 text-gray-400">{m.metric}</td>
                    <td className="px-3 py-2 text-gray-500 font-mono text-xs">{m.filter ?? "\u2014"}</td>
                    <td className="px-3 py-2 text-right font-mono">{(m.value * 100).toFixed(1)}</td>
                    <td className="px-3 py-2 text-right font-mono text-gray-500">
                      {m.stderr != null ? (m.stderr * 100).toFixed(1) : "—"}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-gray-500">{m.nSamples ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
