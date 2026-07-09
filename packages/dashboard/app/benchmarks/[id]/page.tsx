"use client";

import { use, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  cancelBenchmark, getBenchmark, getBenchmarkLog, type BenchmarkRun,
} from "@/lib/benchmarks";
import { BenchmarkResultTable } from "@/components/benchmark-result-table";
import { BenchmarkChart } from "@/components/benchmark-chart";
import { ToolEvalResultCard } from "@/components/tool-eval-result-card";
import { AccuracyResultCard } from "@/components/accuracy-result-card";
import { useSSE, type SseEvent } from "@/lib/sse";

export default function BenchmarkDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [run, setRun] = useState<BenchmarkRun | null>(null);
  const [log, setLog] = useState<string>("");

  const refresh = useCallback(() => {
    getBenchmark(id).then(setRun).catch(() => {});
  }, [id]);

  useEffect(() => { refresh(); }, [refresh]);

  // Seed the log from the persisted file on mount so opening a completed run
  // shows what happened (SSE only delivers events that occur while mounted).
  // The "only if still empty" guard avoids overwriting log lines that SSE
  // may have already delivered while this fetch was in flight.
  useEffect(() => {
    getBenchmarkLog(id)
      .then((text) => setLog((prev) => (prev ? prev : text)))
      .catch(() => {});
  }, [id]);

  useSSE((event: SseEvent) => {
    if (event.type === "benchmark:status") {
      const p = event.payload as { id: string };
      if (p.id === id) refresh();
    }
    if (event.type === "benchmark:log") {
      const p = event.payload as { runId: string; log: string };
      if (p.runId === id) setLog((prev) => (prev + p.log + "\n").slice(-50_000));
    }
  });

  if (!run) return <div className="p-6 text-gray-400">Loading…</div>;

  const series = [
    { label: run.deployment?.displayName ?? run.modelName, color: "#a78bfa", rows: run.results ?? [] },
  ];

  return (
    <div className="p-6 space-y-6">
      <div>
        <Link href="/benchmarks" className="text-sm text-blue-400 hover:underline">
          ← All benchmarks
        </Link>
        <h1 className="text-2xl font-semibold mt-1">
          {run.deployment?.displayName ?? run.modelName}{" "}
          <span className="text-sm text-gray-500">({run.presetId ?? "custom"})</span>
        </h1>
        <div className="text-sm text-gray-400 mt-1">
          Endpoint: <code>{run.endpointUrl}</code> · Served as <code>{run.servedModelName}</code>
        </div>
        <div className="text-sm text-gray-400">
          Started: {run.startedAt ? new Date(run.startedAt).toLocaleString() : "—"} ·
          Finished: {run.completedAt ? new Date(run.completedAt).toLocaleString() : "—"}
        </div>
        <div className="mt-2 flex gap-2 items-center">
          <span className="text-sm">Status:</span> <code>{run.status}</code>
          {run.status === "running" && (
            <button
              type="button"
              className="text-xs px-2 py-1 rounded bg-yellow-700 hover:bg-yellow-600"
              onClick={() => cancelBenchmark(id).then(refresh)}
            >
              Cancel
            </button>
          )}
        </div>
        {run.error && <div className="text-red-400 mt-2">Error: {run.error}</div>}
      </div>

      {run.kind === "accuracy" ? (
        run.status === "completed" && <AccuracyResultCard run={run} />
      ) : run.kind === "tool-eval" ? (
        run.status === "completed" && <ToolEvalResultCard run={run} />
      ) : (
        run.results && run.results.length > 0 && (
          <>
            <BenchmarkChart series={series} metric="tps" />
            <BenchmarkChart series={series} metric="ttfrMs" />
            <BenchmarkResultTable rows={run.results} />
          </>
        )
      )}

      <details className="text-sm" open>
        <summary className="cursor-pointer text-gray-400">Log</summary>
        <pre className="mt-2 p-3 bg-black rounded text-xs overflow-x-auto max-h-96">{log || "(no log)"}</pre>
      </details>

      {run.rawOutput && (
        <details className="text-sm">
          <summary className="cursor-pointer text-gray-400">
            Raw {run.kind === "tool-eval" ? "tool-eval-bench" : run.kind === "accuracy" ? "lm-eval" : "llama-benchy"} JSON
          </summary>
          <pre className="mt-2 p-3 bg-black rounded text-xs overflow-x-auto max-h-96">{run.rawOutput}</pre>
        </details>
      )}
    </div>
  );
}
