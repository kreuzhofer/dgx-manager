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

const METRIC_UNIT: Record<Metric, string> = {
  tps: "t/s",
  ttfrMs: "ms",
  e2eTtftMs: "ms",
};

// Compact, readable numeric label: 1.2k for thousands, 3.4M for millions,
// otherwise one decimal place. Used by both the tooltip and the y-axis.
function formatMetric(value: number, unit: string): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M ${unit}`;
  if (abs >= 10_000) return `${(value / 1_000).toFixed(1)}k ${unit}`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(2)}k ${unit}`;
  if (abs >= 100) return `${value.toFixed(0)} ${unit}`;
  if (abs >= 10) return `${value.toFixed(1)} ${unit}`;
  return `${value.toFixed(2)} ${unit}`;
}

// Buckets rows by (op, pp, tg, depth, concurrency) so each x-axis tick
// represents one comparable workload across all series.
function bucketKey(r: BenchmarkResult): string {
  return `${r.opType}/pp${r.pp}/tg${r.tg}/d${r.depth}/c${r.concurrency}`;
}

// Lexicographic sort puts pp1024 before pp128 and c64 before c8. Sort
// numerically by the actual workload dimensions instead so the x-axis
// reads in a natural order: all pp ops first, then tg, each grouped by
// (depth → prompt size → generated tokens → concurrency).
type Bucket = {
  key: string;
  opType: string;
  pp: number;
  tg: number;
  depth: number;
  concurrency: number;
};

function compareBuckets(a: Bucket, b: Bucket): number {
  return (
    a.opType.localeCompare(b.opType) ||
    a.depth - b.depth ||
    a.pp - b.pp ||
    a.tg - b.tg ||
    a.concurrency - b.concurrency
  );
}

export function BenchmarkChart({
  series, metric,
}: {
  series: ChartSeries[];
  metric: Metric;
}) {
  const bucketsByKey = new Map<string, Bucket>();
  for (const s of series) {
    for (const r of s.rows) {
      const key = bucketKey(r);
      if (!bucketsByKey.has(key)) {
        bucketsByKey.set(key, {
          key,
          opType: r.opType,
          pp: r.pp,
          tg: r.tg,
          depth: r.depth,
          concurrency: r.concurrency,
        });
      }
    }
  }
  const sortedBuckets = Array.from(bucketsByKey.values()).sort(compareBuckets);

  const data = sortedBuckets.map((bk) => {
    const row: Record<string, number | string> = { workload: bk.key };
    for (const s of series) {
      const match = s.rows.find((r) => bucketKey(r) === bk.key);
      if (match) {
        const v = match[metric];
        if (typeof v === "number") row[s.label] = v;
      }
    }
    return row;
  });

  const unit = METRIC_UNIT[metric];

  return (
    <div className="h-72">
      <div className="text-sm text-gray-400 mb-1">{METRIC_LABEL[metric]}</div>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="workload" stroke="#9ca3af" tick={{ fontSize: 10 }} />
          <YAxis
            stroke="#9ca3af"
            tickFormatter={(v: number) => formatMetric(v, unit)}
            width={80}
          />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151" }}
            formatter={(value: number, name: string) => [formatMetric(value, unit), name]}
          />
          <Legend />
          {series.map((s) => (
            <Bar key={s.label} dataKey={s.label} fill={s.color} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
