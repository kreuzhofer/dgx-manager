"use client";

import { useMemo } from "react";
import type { BenchmarkResult } from "@/lib/benchmarks";

// Group rows by (depth, pp, tg, concurrency) so a workload's pp and tg
// throughput sit next to each other; within a group, pp before tg.
function sortRows(rows: BenchmarkResult[]): BenchmarkResult[] {
  return [...rows].sort((a, b) =>
    a.depth - b.depth ||
    a.pp - b.pp ||
    a.tg - b.tg ||
    a.concurrency - b.concurrency ||
    a.opType.localeCompare(b.opType),
  );
}

export function BenchmarkResultTable({ rows }: { rows: BenchmarkResult[] }) {
  const sorted = useMemo(() => sortRows(rows), [rows]);
  if (sorted.length === 0) {
    return <div className="text-sm text-gray-500">No results.</div>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead className="text-left text-gray-400 border-b border-gray-800">
          <tr>
            <Th>Op</Th><Th>pp</Th><Th>tg</Th><Th>depth</Th><Th>conc</Th>
            <Th>t/s</Th><Th>peak t/s</Th><Th>ttfr (ms)</Th>
            <Th>est_ppt (ms)</Th><Th>e2e_ttft (ms)</Th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.id} className="border-b border-gray-900 hover:bg-gray-900/40">
              <Td>{r.opType}</Td>
              <Td>{r.pp}</Td>
              <Td>{r.tg}</Td>
              <Td>{r.depth}</Td>
              <Td>{r.concurrency}</Td>
              <Td>{r.tps.toFixed(1)}{r.tpsStdev != null && <span className="text-gray-500"> ±{r.tpsStdev.toFixed(1)}</span>}</Td>
              <Td>{r.peakTps?.toFixed(1) ?? "—"}</Td>
              <Td>{r.ttfrMs?.toFixed(1) ?? "—"}{r.ttfrStdev != null && <span className="text-gray-500"> ±{r.ttfrStdev.toFixed(1)}</span>}</Td>
              <Td>{r.estPptMs?.toFixed(1) ?? "—"}</Td>
              <Td>{r.e2eTtftMs?.toFixed(1) ?? "—"}</Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const Th = ({ children }: { children: React.ReactNode }) => (
  <th className="px-2 py-1 font-medium">{children}</th>
);
const Td = ({ children }: { children: React.ReactNode }) => (
  <td className="px-2 py-1">{children}</td>
);
