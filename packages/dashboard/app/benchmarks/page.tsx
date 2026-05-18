"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  deleteBenchmark, listBenchmarks, type BenchmarkRun,
} from "@/lib/benchmarks";
import { useSSE, type SseEvent } from "@/lib/sse";

function fmtAgo(iso: string | null): string {
  if (!iso) return "—";
  const sec = Math.max(1, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

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

  const { connected } = useSSE((event: SseEvent) => {
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

  const stats = useMemo(() => {
    const completed = runs.filter((r) => r.status === "completed").length;
    const running = runs.filter((r) => r.status === "running" || r.status === "pending").length;
    const last = runs[0]?.createdAt ?? null;
    return { total: runs.length, completed, running, lastAgo: fmtAgo(last) };
  }, [runs]);

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
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Benchmarks</h1>
        <div className="flex items-center gap-2">
          {compareHref && (
            <Link
              href={compareHref}
              className="text-xs px-2 py-1 rounded bg-green-900 text-green-300 hover:bg-green-800 transition-colors"
            >
              Compare ({selected.size}) →
            </Link>
          )}
          <span
            className={`text-xs px-2 py-1 rounded ${
              connected ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"
            }`}
          >
            {connected ? "Live" : "Disconnected"}
          </span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard label="Total Runs" value={stats.total} sub={`${filtered.length} visible`} />
        <StatCard label="Completed" value={stats.completed} sub="successful" />
        <StatCard label="In Flight" value={stats.running} sub="pending or running" />
        <StatCard label="Last Run" value={stats.lastAgo} sub={runs[0]?.deployment?.displayName ?? runs[0]?.modelName ?? "—"} />
      </div>

      {/* Filters */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6 flex flex-wrap gap-3 items-center">
        <input
          type="text"
          placeholder="Filter by deployment/model…"
          value={filter.deploymentName}
          onChange={(e) => setFilter({ ...filter, deploymentName: e.target.value })}
          className="bg-gray-950 border border-gray-800 rounded px-3 py-1.5 text-sm flex-1 min-w-[200px] focus:outline-none focus:border-gray-600"
        />
        <select
          value={filter.presetId}
          onChange={(e) => setFilter({ ...filter, presetId: e.target.value })}
          className="bg-gray-950 border border-gray-800 rounded px-3 py-1.5 text-sm focus:outline-none focus:border-gray-600"
        >
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
          className="bg-gray-950 border border-gray-800 rounded px-3 py-1.5 text-sm focus:outline-none focus:border-gray-600"
        >
          <option value="">All statuses</option>
          <option value="pending">pending</option>
          <option value="running">running</option>
          <option value="completed">completed</option>
          <option value="failed">failed</option>
          <option value="canceled">canceled</option>
        </select>
        {(filter.deploymentName || filter.presetId || filter.status) && (
          <button
            type="button"
            className="text-xs text-gray-400 hover:text-gray-200 px-2 py-1"
            onClick={() => setFilter({ deploymentName: "", presetId: "", status: "" })}
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Table card */}
      {filtered.length === 0 ? (
        <div className="text-gray-400 text-center py-16">
          {runs.length === 0 ? (
            <>
              <p className="text-lg">No benchmark runs yet.</p>
              <p className="mt-2">
                Start a run from{" "}
                <Link href="/deployments" className="text-green-400 underline">
                  Deployments
                </Link>
                {" "}— click the &quot;Benchmark&quot; button on any running row.
              </p>
            </>
          ) : (
            <>
              <p className="text-lg">No runs match the current filter.</p>
              <p className="mt-2 text-sm">Adjust the filters above or clear them to see all {runs.length} runs.</p>
            </>
          )}
        </div>
      ) : (
        <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="text-left text-xs text-gray-500 uppercase tracking-wide border-b border-gray-800 bg-gray-950/40">
                <tr>
                  <th className="px-4 py-3 w-8"></th>
                  <th className="px-4 py-3">When</th>
                  <th className="px-4 py-3">Deployment</th>
                  <th className="px-4 py-3">Preset</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3 text-right">Mean t/s</th>
                  <th className="px-4 py-3 text-right">Mean TTFR</th>
                  <th className="px-4 py-3 w-32"></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr
                    key={r.id}
                    className={`border-b border-gray-800 last:border-b-0 transition-colors ${
                      selected.has(r.id) ? "bg-green-950/30" : "hover:bg-gray-950/40"
                    }`}
                  >
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={selected.has(r.id)}
                        onChange={() => toggle(r.id)}
                        className="accent-green-500 cursor-pointer"
                        aria-label={`Select ${r.deployment?.displayName ?? r.modelName} for compare`}
                      />
                    </td>
                    <td className="px-4 py-3 text-gray-300 whitespace-nowrap">
                      <div>{new Date(r.createdAt).toLocaleString()}</div>
                      <div className="text-xs text-gray-500">{fmtAgo(r.createdAt)}</div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="font-medium">
                        {r.deployment?.displayName ?? r.deployment?.model.name ?? (
                          <span className="text-gray-500 italic">(deleted)</span>
                        )}
                      </div>
                      <div className="text-xs text-gray-500">{r.modelName}</div>
                    </td>
                    <td className="px-4 py-3 text-gray-300">
                      {r.presetId ?? <span className="text-gray-500 italic">custom</span>}
                    </td>
                    <td className="px-4 py-3">
                      <StatusPill status={r.status} />
                      {r.error && (
                        <div className="text-xs text-red-400 mt-1 max-w-[240px] truncate" title={r.error}>
                          {r.error}
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      {r.meanTps != null ? r.meanTps.toFixed(1) : <span className="text-gray-500">—</span>}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">
                      {r.meanTtfrMs != null ? r.meanTtfrMs.toFixed(0) : <span className="text-gray-500">—</span>}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex gap-3 justify-end">
                        <Link
                          href={`/benchmarks/${r.id}`}
                          className="text-xs text-green-400 hover:underline"
                        >
                          View
                        </Link>
                        <button
                          type="button"
                          className="text-xs text-red-400 hover:underline"
                          onClick={async () => {
                            if (confirm(`Delete this benchmark run?\n\n${r.deployment?.displayName ?? r.modelName}`)) {
                              await deleteBenchmark(r.id);
                              refresh();
                            }
                          }}
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
      <p className="text-xs text-gray-500 mt-0.5 truncate">{sub}</p>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const color = {
    completed: "bg-green-900 text-green-300",
    running:   "bg-blue-900 text-blue-300",
    pending:   "bg-gray-800 text-gray-400",
    failed:    "bg-red-900 text-red-300",
    canceled:  "bg-yellow-900 text-yellow-300",
  }[status] ?? "bg-gray-800 text-gray-400";
  return <span className={`text-xs px-2 py-1 rounded ${color}`}>{status}</span>;
}
