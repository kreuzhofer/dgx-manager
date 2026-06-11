import type { BenchmarkRun } from "@/lib/benchmarks";

// Defensive parse: toolEvalSafetyWarnings is server-written JSON, but a
// malformed value should degrade to "no warnings" rather than blanking the
// whole detail page.
function parseSafetyWarnings(raw: string | null): string[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch {
    return [];
  }
}

export function ToolEvalResultCard({ run }: { run: BenchmarkRun }) {
  const warnings = parseSafetyWarnings(run.toolEvalSafetyWarnings);
  const cats = run.toolEvalCategories ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-baseline">
        <div>
          <div className="text-5xl font-semibold">{run.toolEvalScore ?? "—"}<span className="text-xl text-gray-500">/100</span></div>
          <div className="text-lg text-amber-400 mt-1">{run.toolEvalRating ?? ""}</div>
        </div>
        <div className="text-sm text-gray-400 space-y-1">
          <div>Deployability: <span className="text-gray-200">{run.toolEvalDeployability ?? "—"}</span></div>
          <div>Responsiveness: <span className="text-gray-200">{run.toolEvalResponsiveness ?? "—"}</span></div>
          <div>
            Points: <span className="text-gray-200">{run.toolEvalTotalPoints ?? "—"}/{run.toolEvalMaxPoints ?? "—"}</span>
            {" "}across {run.toolEvalTotalScenarios ?? "—"} scenarios
          </div>
        </div>
      </div>

      {warnings.length > 0 && (
        <div className="rounded border border-red-700 bg-red-950/40 p-3">
          <div className="text-sm font-medium text-red-300 mb-1">Safety warnings</div>
          <ul className="text-xs text-red-200 list-disc ml-4 space-y-0.5">
            {warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}

      <div className="space-y-2">
        <div className="text-sm font-medium text-gray-300">Category breakdown</div>
        {cats.length === 0 && <div className="text-sm text-gray-500">No category data.</div>}
        {cats.map((c) => (
          <div key={c.id} className="text-sm">
            <div className="flex justify-between text-xs text-gray-400 mb-0.5">
              <span>{c.code} · {c.label}</span>
              <span>{c.percent}% ({c.earned}/{c.maxPoints}) · {c.passCount}✓ {c.partialCount}~ {c.failCount}✗</span>
            </div>
            <div className="h-2 rounded bg-gray-800 overflow-hidden">
              <div
                className="h-full bg-emerald-500"
                style={{ width: `${Math.max(0, Math.min(100, c.percent))}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
