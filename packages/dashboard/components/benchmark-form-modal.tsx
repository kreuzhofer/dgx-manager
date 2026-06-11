"use client";

import { useEffect, useState } from "react";
import {
  listPresets,
  startBenchmark,
  type BenchmarkPreset,
  type BenchmarkConfig,
} from "@/lib/benchmarks";

type Props = {
  deploymentId: string;
  deploymentLabel: string;
  onClose: () => void;
  onStarted: () => void;
};

export function BenchmarkFormModal({
  deploymentId, deploymentLabel, onClose, onStarted,
}: Props) {
  const [presets, setPresets] = useState<BenchmarkPreset[]>([]);
  const [presetId, setPresetId] = useState<string>("quick-smoke");
  const [showCustom, setShowCustom] = useState(false);
  const [custom, setCustom] = useState<BenchmarkConfig>({
    pp: [128], tg: [32], depth: [0], runs: 1,
    concurrency: [1], latencyMode: "api",
    enablePrefixCaching: false, skipCoherence: false,
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { listPresets().then(setPresets).catch(() => {}); }, []);

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await startBenchmark(
        showCustom
          ? { deploymentId, config: custom }
          : { deploymentId, presetId },
      );
      onStarted();
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const parseIntList = (s: string): number[] =>
    s.split(",").map((x) => parseInt(x.trim(), 10)).filter((n) => Number.isFinite(n) && n >= 0);

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-lg p-6 w-[640px] max-h-[90vh] overflow-y-auto">
        <h2 className="text-xl font-semibold mb-1">Run benchmark</h2>
        <p className="text-sm text-gray-400 mb-4">Target: {deploymentLabel}</p>

        {!showCustom && (
          <div className="space-y-4">
            {(["throughput", "tool-eval"] as const).map((kind) => {
              const group = presets.filter((p) => p.kind === kind);
              if (group.length === 0) return null;
              return (
                <div key={kind} className="space-y-2">
                  <div className="text-xs uppercase tracking-wide text-gray-500">
                    {kind === "throughput" ? "Throughput" : "Tool-calling eval"}
                  </div>
                  {group.map((p) => (
                    <label key={p.id} className="block p-3 rounded bg-gray-800 hover:bg-gray-700 cursor-pointer">
                      <div className="flex items-start gap-2">
                        <input
                          type="radio"
                          name="preset"
                          value={p.id}
                          checked={presetId === p.id}
                          onChange={() => setPresetId(p.id)}
                          className="mt-1"
                        />
                        <div>
                          <div className="font-medium">{p.label}</div>
                          <div className="text-xs text-gray-400">{p.description}</div>
                          {p.kind === "throughput" && "pp" in p.config && (
                            <div className="text-xs text-gray-500 mt-1">
                              pp=[{p.config.pp.join(",")}] tg=[{p.config.tg.join(",")}]
                              {" "}depth=[{p.config.depth.join(",")}]
                              {" "}concurrency=[{p.config.concurrency.join(",")}]
                              {" "}runs={p.config.runs}
                            </div>
                          )}
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              );
            })}
          </div>
        )}

        <button
          type="button"
          className="text-xs text-blue-400 hover:underline mt-3"
          onClick={() => setShowCustom((v) => !v)}
        >
          {showCustom ? "Use a preset instead" : "Custom configuration…"}
        </button>

        {showCustom && (
          <div className="space-y-3 mt-3 p-3 bg-gray-800 rounded">
            <Field label="Prompt tokens (comma-separated)"
              value={custom.pp.join(",")}
              onChange={(v) => setCustom({ ...custom, pp: parseIntList(v) })} />
            <Field label="Generated tokens (comma-separated)"
              value={custom.tg.join(",")}
              onChange={(v) => setCustom({ ...custom, tg: parseIntList(v) })} />
            <Field label="Context depths (comma-separated)"
              value={custom.depth.join(",")}
              onChange={(v) => setCustom({ ...custom, depth: parseIntList(v) })} />
            <Field label="Concurrency levels (comma-separated)"
              value={custom.concurrency.join(",")}
              onChange={(v) => setCustom({ ...custom, concurrency: parseIntList(v) })} />
            <Field label="Runs per cell"
              value={String(custom.runs)}
              onChange={(v) => setCustom({ ...custom, runs: parseInt(v, 10) || 1 })} />
            <label className="block text-sm">
              Latency mode
              <select
                className="block mt-1 bg-gray-900 border border-gray-700 rounded px-2 py-1"
                value={custom.latencyMode}
                onChange={(e) => setCustom({ ...custom, latencyMode: e.target.value as BenchmarkConfig["latencyMode"] })}>
                <option value="api">api</option>
                <option value="generation">generation</option>
                <option value="none">none</option>
              </select>
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox"
                checked={custom.enablePrefixCaching}
                onChange={(e) => setCustom({ ...custom, enablePrefixCaching: e.target.checked })} />
              Enable prefix caching measurement
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox"
                checked={custom.skipCoherence}
                onChange={(e) => setCustom({ ...custom, skipCoherence: e.target.checked })} />
              Skip coherence check
            </label>
          </div>
        )}

        {error && <div className="text-red-400 text-sm mt-3">{error}</div>}

        <div className="flex justify-end gap-2 mt-6">
          <button
            type="button"
            className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600"
            onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            disabled={submitting}
            className="px-3 py-1 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50"
            onClick={submit}>
            {submitting ? "Starting…" : "Start benchmark"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Field({
  label, value, onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block text-sm">
      {label}
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="block mt-1 w-full bg-gray-900 border border-gray-700 rounded px-2 py-1"
      />
    </label>
  );
}
