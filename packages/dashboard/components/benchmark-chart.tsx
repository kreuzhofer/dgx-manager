"use client";

import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { BenchmarkResult } from "@/lib/benchmarks";

export type ChartSeries = {
  label: string;
  color: string;
  rows: BenchmarkResult[];
};

type Metric = "tps" | "ttfrMs" | "e2eTtftMs";

const METRIC_LABEL: Record<Metric, string> = {
  tps: "Tokens / second",
  ttfrMs: "Time to first response (ms)",
  e2eTtftMs: "End-to-end TTFT (ms)",
};

// Buckets rows by (op, pp, tg, depth, concurrency) so each x-axis tick
// represents one comparable workload across all series.
function bucketKey(r: BenchmarkResult): string {
  return `${r.opType}/pp${r.pp}/tg${r.tg}/d${r.depth}/c${r.concurrency}`;
}

export function BenchmarkChart({
  series, metric,
}: {
  series: ChartSeries[];
  metric: Metric;
}) {
  const allKeys = Array.from(
    new Set(series.flatMap((s) => s.rows.map(bucketKey))),
  ).sort();

  const data = allKeys.map((key) => {
    const row: Record<string, number | string> = { workload: key };
    for (const s of series) {
      const match = s.rows.find((r) => bucketKey(r) === key);
      if (match) {
        const v = match[metric];
        if (typeof v === "number") row[s.label] = v;
      }
    }
    return row;
  });

  return (
    <div className="h-72">
      <div className="text-sm text-gray-400 mb-1">{METRIC_LABEL[metric]}</div>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="workload" stroke="#9ca3af" tick={{ fontSize: 10 }} />
          <YAxis stroke="#9ca3af" />
          <Tooltip contentStyle={{ background: "#111827", border: "1px solid #374151" }} />
          <Legend />
          {series.map((s) => (
            <Bar key={s.label} dataKey={s.label} fill={s.color} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
