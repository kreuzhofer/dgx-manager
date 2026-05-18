"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  deleteBenchmark, listBenchmarks, type BenchmarkRun,
} from "@/lib/benchmarks";
import { useSSE, type SseEvent } from "@/lib/sse";

export default function BenchmarksPage() {
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [filter, setFilter] = useState({
    deploymentName: "",
    presetId: "",
    status: "",
  });
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const refresh = useCallback(() => {
    listBenchmarks().then(setRuns).catch(() => {});
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useSSE((event: SseEvent) => {
    if (
      event.type === "benchmark:created" ||
      event.type === "benchmark:status" ||
      event.type === "benchmark:deleted"
    ) {
      refresh();
    }
  });

  const filtered = useMemo(() => runs.filter((r) => {
    const depName = r.deployment?.displayName ?? r.modelName ?? "";
    if (filter.deploymentName && !depName.toLowerCase().includes(filter.deploymentName.toLowerCase())) return false;
    if (filter.presetId && r.presetId !== filter.presetId) return false;
    if (filter.status && r.status !== filter.status) return false;
    return true;
  }), [runs, filter]);

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const compareHref =
    selected.size >= 2
      ? `/benchmarks/compare?ids=${Array.from(selected).join(",")}`
      : null;

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Benchmarks</h1>
        <div className="flex gap-2">
          {compareHref && (
            <Link
              href={compareHref}
              className="px-3 py-1 rounded bg-purple-700 hover:bg-purple-600 text-sm"
            >
              Compare ({selected.size})
            </Link>
          )}
        </div>
      </div>

      <div className="flex gap-2">
        <input
          placeholder="Filter by deployment/model…"
          value={filter.deploymentName}
          onChange={(e) => setFilter({ ...filter, deploymentName: e.target.value })}
          className="bg-gray-900 border border-gray-800 rounded px-2 py-1 text-sm"
        />
        <select
          value={filter.presetId}
          onChange={(e) => setFilter({ ...filter, presetId: e.target.value })}
          className="bg-gray-900 border border-gray-800 rounded px-2 py-1 text-sm">
          <option value="">All presets</option>
          <option value="quick-smoke">quick-smoke</option>
          <option value="chat-short">chat-short</option>
          <option value="chat-long">chat-long</option>
          <option value="code-32k">code-32k</option>
          <option value="throughput">throughput</option>
        </select>
        <select
          value={filter.status}
          onChange={(e) => setFilter({ ...filter, status: e.target.value })}
          className="bg-gray-900 border border-gray-800 rounded px-2 py-1 text-sm">
          <option value="">All statuses</option>
          <option value="pending">pending</option>
          <option value="running">running</option>
          <option value="completed">completed</option>
          <option value="failed">failed</option>
          <option value="canceled">canceled</option>
        </select>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="text-left text-gray-400 border-b border-gray-800">
            <tr>
              <th></th>
              <th className="px-2 py-1">When</th>
              <th className="px-2 py-1">Deployment</th>
              <th className="px-2 py-1">Model</th>
              <th className="px-2 py-1">Preset</th>
              <th className="px-2 py-1">Status</th>
              <th className="px-2 py-1">Mean t/s</th>
              <th className="px-2 py-1">Mean ttfr (ms)</th>
              <th className="px-2 py-1"></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={r.id} className="border-b border-gray-900 hover:bg-gray-900/40">
                <td className="px-2 py-1">
                  <input
                    type="checkbox"
                    checked={selected.has(r.id)}
                    onChange={() => toggle(r.id)}
                  />
                </td>
                <td className="px-2 py-1 text-gray-300">
                  {new Date(r.createdAt).toLocaleString()}
                </td>
                <td className="px-2 py-1">
                  {r.deployment?.displayName ?? r.deployment?.model.name ?? <span className="text-gray-500">(deleted)</span>}
                </td>
                <td className="px-2 py-1">{r.modelName}</td>
                <td className="px-2 py-1">{r.presetId ?? <span className="text-gray-500">custom</span>}</td>
                <td className="px-2 py-1">
                  <StatusPill status={r.status} />
                </td>
                <td className="px-2 py-1">{r.meanTps?.toFixed(1) ?? "—"}</td>
                <td className="px-2 py-1">{r.meanTtfrMs?.toFixed(1) ?? "—"}</td>
                <td className="px-2 py-1 flex gap-2">
                  <Link href={`/benchmarks/${r.id}`} className="text-blue-400 hover:underline">
                    View
                  </Link>
                  <button
                    type="button"
                    className="text-red-400 hover:underline"
                    onClick={async () => {
                      if (confirm("Delete this benchmark run?")) {
                        await deleteBenchmark(r.id);
                        refresh();
                      }
                    }}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const color = {
    completed: "bg-green-900/40 text-green-300",
    running:   "bg-blue-900/40 text-blue-300",
    pending:   "bg-gray-800 text-gray-300",
    failed:    "bg-red-900/40 text-red-300",
    canceled:  "bg-yellow-900/40 text-yellow-300",
  }[status] ?? "bg-gray-800 text-gray-300";
  return <span className={`px-2 py-0.5 rounded text-xs ${color}`}>{status}</span>;
}
