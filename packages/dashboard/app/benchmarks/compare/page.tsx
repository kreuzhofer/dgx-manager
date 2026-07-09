"use client";

export const dynamic = "force-dynamic";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { getBenchmark, type BenchmarkRun } from "@/lib/benchmarks";
import { BenchmarkChart, type ChartSeries } from "@/components/benchmark-chart";

const PALETTE = ["#a78bfa", "#34d399", "#f59e0b", "#60a5fa", "#f472b6", "#f87171"];

function ComparePageBody() {
  const search = useSearchParams();
  const ids = (search.get("ids") ?? "").split(",").filter(Boolean);
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);

  useEffect(() => {
    Promise.all(ids.map(getBenchmark))
      .then(setRuns)
      .catch(() => {});
    // ids is derived from the URL; reload when it changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search.get("ids")]);

  if (runs.length === 0) {
    return (
      <div className="p-6 text-gray-400">
        Pick at least two runs from the{" "}
        <Link href="/benchmarks" className="text-blue-400 hover:underline">benchmarks list</Link>.
      </div>
    );
  }

  const series: ChartSeries[] = runs.map((r, i) => ({
    label: `${r.deployment?.displayName ?? r.modelName} (${r.presetId ?? "custom"})`,
    color: PALETTE[i % PALETTE.length],
    rows: r.results ?? [],
  }));

  return (
    <div className="p-6 space-y-6">
      <div>
        <Link href="/benchmarks" className="text-sm text-blue-400 hover:underline">
          ← All benchmarks
        </Link>
        <h1 className="text-2xl font-semibold mt-1">Compare</h1>
        <ul className="text-sm text-gray-400 mt-2 space-y-1">
          {runs.map((r, i) => (
            <li key={r.id}>
              <span className="inline-block w-3 h-3 rounded-sm mr-2"
                    style={{ background: PALETTE[i % PALETTE.length] }} />
              <Link href={`/benchmarks/${r.id}`} className="hover:underline">
                {r.deployment?.displayName ?? r.modelName}
              </Link>{" "}
              · {r.presetId ?? "custom"} · {new Date(r.createdAt).toLocaleString()}
            </li>
          ))}
        </ul>
      </div>

      {runs.some((r) => r.kind === "accuracy") && (
        <div className="space-y-2">
          <h2 className="text-lg font-medium">Accuracy score</h2>
          {runs
            .filter((r) => r.kind === "accuracy")
            .map((r, i) => (
              <div key={r.id} className="text-sm">
                <div className="flex justify-between text-xs text-gray-400 mb-0.5">
                  <span>{r.deployment?.displayName ?? r.modelName} · {r.presetId ?? "custom"}</span>
                  <span>{r.accuracyScore != null ? `${r.accuracyScore.toFixed(1)}/100` : "—"}</span>
                </div>
                <div className="h-2 rounded bg-gray-800 overflow-hidden">
                  <div
                    className="h-full"
                    style={{
                      width: `${Math.max(0, Math.min(100, r.accuracyScore ?? 0))}%`,
                      background: PALETTE[i % PALETTE.length],
                    }}
                  />
                </div>
              </div>
            ))}
        </div>
      )}

      {runs.some((r) => (r.results?.length ?? 0) > 0) && (
        <>
          <BenchmarkChart series={series} metric="tps" />
          <BenchmarkChart series={series} metric="ttfrMs" />
          <BenchmarkChart series={series} metric="e2eTtftMs" />
        </>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="p-6 text-gray-400">Loading…</div>}>
      <ComparePageBody />
    </Suspense>
  );
}
