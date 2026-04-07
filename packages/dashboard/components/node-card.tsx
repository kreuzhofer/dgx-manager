"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { Sparkline } from "./sparkline";

interface NetInterfaceSample {
  name: string;
  rxBytesPerSec: number;
  txBytesPerSec: number;
}

interface MetricSample {
  timestamp: number;
  gpuUtil: number;
  vramUsed: number;
  temperature: number | null;
  tps: number | null;
  activeRequests: number | null;
  netInterfaces?: NetInterfaceSample[];
}

interface Node {
  id: string;
  name: string;
  ipAddress: string;
  status: string;
  gpuModel: string | null;
  vramTotal: number | null;
  agentVersion?: string | null;
  metrics: { gpuUtil: number; vramUsed: number; tps: number | null; temperature: number | null }[];
}

const MAX_SAMPLES = 720;

const statusColors: Record<string, string> = {
  online: "bg-green-500",
  offline: "bg-red-500",
  new: "bg-gray-500",
};

export function NodeCard({
  node,
  onMetrics,
}: {
  node: Node;
  onMetrics?: (handler: (sample: MetricSample) => void) => void;
}) {
  const [history, setHistory] = useState<MetricSample[]>([]);
  const historyRef = useRef<MetricSample[]>([]);
  const rafPending = useRef(false);

  // Fetch initial history
  useEffect(() => {
    apiFetch<MetricSample[]>(`/api/nodes/${node.id}/metrics/history`)
      .then((data) => {
        historyRef.current = data;
        setHistory(data);
      })
      .catch(() => {});
  }, [node.id]);

  // Register for live updates from parent
  const appendSample = useCallback((sample: MetricSample) => {
    const buf = historyRef.current;
    buf.push(sample);
    if (buf.length > MAX_SAMPLES) buf.shift();
    // Throttle re-renders
    if (!rafPending.current) {
      rafPending.current = true;
      requestAnimationFrame(() => {
        setHistory([...historyRef.current]);
        rafPending.current = false;
      });
    }
  }, []);

  useEffect(() => {
    onMetrics?.(appendSample);
  }, [onMetrics, appendSample]);

  const latest = history.length > 0 ? history[history.length - 1] : null;
  const statusColor = statusColors[node.status] || "bg-gray-500";
  const vramMax = node.vramTotal || 128000;

  const gpuData = history.map((s) => s.gpuUtil);
  const vramData = history.map((s) => s.vramUsed);
  const tempData = history.filter((s) => s.temperature !== null).length > 0
    ? history.map((s) => s.temperature ?? 0)
    : [];
  const reqData = history.filter((s) => s.activeRequests !== null).length > 0
    ? history.map((s) => s.activeRequests ?? 0)
    : [];

  // Collect unique network interface names from history
  const ifaceNames = new Set<string>();
  for (const s of history) {
    for (const ni of s.netInterfaces || []) ifaceNames.add(ni.name);
  }
  // Build per-interface throughput arrays (combined rx+tx in Mbps)
  const netData = Array.from(ifaceNames).map((name) => ({
    name,
    data: history.map((s) => {
      const ni = s.netInterfaces?.find((n) => n.name === name);
      return ni ? (ni.rxBytesPerSec + ni.txBytesPerSec) * 8 / 1_000_000 : 0; // Mbps
    }),
  }));

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-semibold text-lg">{node.name}</h3>
          <p className="text-xs text-gray-500">
            {node.ipAddress}
            {node.gpuModel && (
              <span className="ml-2 text-gray-400">{node.gpuModel}</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${statusColor}`} />
          <span className="text-xs text-gray-400 capitalize">{node.status}</span>
        </div>
      </div>

      {/* Sparkline Graphs */}
      {latest ? (
        <div className="grid grid-cols-2 gap-3">
          <Sparkline
            data={gpuData}
            max={100}
            color="#22c55e"
            label="GPU"
            unit="%"
          />
          <Sparkline
            data={vramData}
            max={vramMax}
            color="#3b82f6"
            label="VRAM"
            currentValue={`${Math.round(latest.vramUsed)}/${vramMax} MB`}
          />
          {tempData.length > 0 && (
            <Sparkline
              data={tempData}
              max={100}
              color="#f97316"
              label="Temp"
              unit="°C"
            />
          )}
          {reqData.length > 0 ? (
            <Sparkline
              data={reqData}
              max={Math.max(...reqData, 10)}
              color="#ef4444"
              label="Requests"
              unit=""
            />
          ) : (
            <Sparkline
              data={[]}
              max={100}
              color="#ef4444"
              label="Requests"
            />
          )}
          {netData.map((iface) => (
            <Sparkline
              key={iface.name}
              data={iface.data}
              max={Math.max(...iface.data, 100)}
              color="#8b5cf6"
              label={iface.name}
              currentValue={`${iface.data.length > 0 ? iface.data[iface.data.length - 1].toFixed(0) : 0} Mbps`}
            />
          ))}
        </div>
      ) : (
        <p className="text-xs text-gray-500 italic">No metrics yet</p>
      )}
    </div>
  );
}
