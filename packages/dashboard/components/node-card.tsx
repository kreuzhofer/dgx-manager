"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { Sparkline } from "./sparkline";

interface NetInterfaceSample {
  name: string;
  rxBytesPerSec: number;
  txBytesPerSec: number;
}

interface DiskDeviceSample {
  name: string;
  readBytesPerSec: number;
  writeBytesPerSec: number;
}

interface MemorySample {
  memTotalMb: number;
  memAvailableMb: number;
  memCachedMb: number;
  swapTotalMb: number;
  swapUsedMb: number;
}

interface PressureSample {
  memorySomeAvg10: number | null;
  ioSomeAvg10: number | null;
  cpuSomeAvg10: number | null;
}

interface MetricSample {
  timestamp: number;
  gpuUtil: number;
  vramUsed: number;
  temperature: number | null;
  tps: number | null;
  activeRequests: number | null;
  netInterfaces?: NetInterfaceSample[];
  rdmaInterfaces?: NetInterfaceSample[];
  diskDevices?: DiskDeviceSample[];
  memory?: MemorySample;
  pressure?: PressureSample;
}

interface NodeDeployment {
  modelName: string;
  runtime: string;
  status: string;
}

interface Node {
  id: string;
  name: string;
  ipAddress: string;
  status: string;
  powerState?: string;
  gpuModel: string | null;
  vramTotal: number | null;
  agentVersion?: string | null;
  metrics: { gpuUtil: number; vramUsed: number; tps: number | null; temperature: number | null }[];
}

const MAX_SAMPLES = 720;
const DISPLAY_WINDOW = 360; // 30 minutes at 5s intervals

const statusColors: Record<string, string> = {
  online: "bg-green-500",
  offline: "bg-red-500",
  new: "bg-gray-500",
};

export function NodeCard({
  node,
  deployments = [],
  onMetrics,
}: {
  node: Node;
  deployments?: NodeDeployment[];
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

  const isOff = node.powerState === "off" || node.powerState === "asleep";
  // "waking" = a WOL packet was sent but the agent has not reconnected yet.
  // Treat off/asleep/waking all as "inactive" so the card dims and keeps showing
  // the Wake button — a node stuck in "waking" (WOL packet missed) can be retried.
  const isWaking = node.powerState === "waking";
  const isInactive = isOff || isWaking;

  async function power(action: "reboot" | "shutdown") {
    const verb = action === "reboot" ? "Reboot" : "Shut down";
    if (!window.confirm(`${verb} node "${node.name}"? This runs sudo on the machine and will drop its agent.`)) {
      return;
    }
    try {
      await apiFetch<unknown>(`/api/nodes/${node.id}/power`, {
        method: "POST",
        body: JSON.stringify({ action }),
      });
    } catch (e) {
      window.alert(`Power action failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function wake() {
    try {
      await apiFetch<unknown>(`/api/nodes/${node.id}/wake`, { method: "POST" });
    } catch (e) {
      window.alert(`Wake failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  // Display window: last 30 minutes only
  const displayHistory = history.slice(-DISPLAY_WINDOW);

  const latest = displayHistory.length > 0 ? displayHistory[displayHistory.length - 1] : null;
  const statusColor = statusColors[node.status] || "bg-gray-500";
  const vramMax = node.vramTotal || 128000;

  const gpuData = displayHistory.map((s) => s.gpuUtil);
  const vramData = displayHistory.map((s) => s.vramUsed);
  const tempData = displayHistory.filter((s) => s.temperature !== null).length > 0
    ? displayHistory.map((s) => s.temperature ?? 0)
    : [];
  const reqData = displayHistory.filter((s) => s.activeRequests !== null).length > 0
    ? displayHistory.map((s) => s.activeRequests ?? 0)
    : [];
  const tpsData = displayHistory.filter((s) => s.tps !== null).length > 0
    ? displayHistory.map((s) => s.tps ?? 0)
    : [];

  // Collect unique network interface names from history
  const ifaceNames = new Set<string>();
  for (const s of displayHistory) {
    for (const ni of s.netInterfaces || []) ifaceNames.add(ni.name);
  }
  // Build per-interface throughput arrays (combined rx+tx in Mbps)
  const netData = Array.from(ifaceNames).map((name) => ({
    name,
    data: displayHistory.map((s) => {
      const ni = s.netInterfaces?.find((n) => n.name === name);
      return ni ? (ni.rxBytesPerSec + ni.txBytesPerSec) * 8 / 1_000_000 : 0; // Mbps
    }),
  }));

  // RDMA/InfiniBand interfaces
  const rdmaNames = new Set<string>();
  for (const s of displayHistory) {
    for (const ri of s.rdmaInterfaces || []) rdmaNames.add(ri.name);
  }
  const rdmaData = Array.from(rdmaNames).map((name) => ({
    name,
    data: displayHistory.map((s) => {
      const ri = s.rdmaInterfaces?.find((n) => n.name === name);
      return ri ? (ri.rxBytesPerSec + ri.txBytesPerSec) * 8 / 1_000_000 : 0; // Mbps
    }),
  }));

  // System memory (GB10 unified pool: GPU+CPU share). Diagnostic for the
  // class of OOM / soft-hang failures we hit at end-of-training. Tracks
  // % used (memTotal - memAvailable) so the y-axis stays comparable
  // across nodes regardless of total RAM.
  const memData = displayHistory
    .filter((s) => s.memory && s.memory.memTotalMb > 0)
    .map((s) => Math.round(((s.memory!.memTotalMb - s.memory!.memAvailableMb) / s.memory!.memTotalMb) * 100));
  const latestMem = displayHistory.length > 0 ? displayHistory[displayHistory.length - 1]?.memory : null;
  const swapData = displayHistory
    .filter((s) => s.memory && s.memory.swapTotalMb > 0)
    .map((s) => s.memory!.swapUsedMb);

  // PSI memory pressure: % of last 10 s where at least one task was stalled
  // on memory. Earliest leading indicator of the soft-hang failure mode.
  const psiMemData = displayHistory
    .filter((s) => s.pressure && s.pressure.memorySomeAvg10 !== null)
    .map((s) => s.pressure!.memorySomeAvg10!);

  // Disk devices — combined read+write throughput in MB/s (bytes-based, unlike net)
  const diskNames = new Set<string>();
  for (const s of displayHistory) {
    for (const d of s.diskDevices || []) diskNames.add(d.name);
  }
  const diskData = Array.from(diskNames).map((name) => ({
    name,
    data: displayHistory.map((s) => {
      const d = s.diskDevices?.find((dd) => dd.name === name);
      return d ? (d.readBytesPerSec + d.writeBytesPerSec) / 1_000_000 : 0; // MB/s
    }),
  }));

  return (
    <div className={`bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors ${isInactive ? "opacity-60" : ""}`}>
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
          <div className="mt-1 flex flex-wrap gap-1">
            {deployments.length > 0 ? deployments.map((dep, i) => (
              <span key={i} className="inline-flex items-center gap-1">
                <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                  dep.status === "evicted"
                    ? "bg-yellow-900/60 text-yellow-300"
                    : dep.runtime === "ollama"
                    ? "bg-cyan-900/60 text-cyan-300"
                    : "bg-green-900/60 text-green-300"
                }`}>
                  {dep.runtime === "ollama" ? "Ollama" : "vLLM"}
                </span>
                <span className="text-[10px] text-gray-400">{dep.modelName}</span>
              </span>
            )) : (
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-500">idle</span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            {isInactive ? (
              <button
                onClick={wake}
                className="text-[10px] px-2 py-0.5 rounded bg-blue-900/60 text-blue-300 hover:bg-blue-800"
                title="Wake-on-LAN"
              >
                Wake
              </button>
            ) : (
              <>
                <button
                  onClick={() => power("reboot")}
                  className="text-[10px] px-2 py-0.5 rounded bg-gray-800 text-gray-300 hover:bg-gray-700"
                >
                  Reboot
                </button>
                <button
                  onClick={() => power("shutdown")}
                  className="text-[10px] px-2 py-0.5 rounded bg-red-900/60 text-red-300 hover:bg-red-800"
                >
                  Shutdown
                </button>
              </>
            )}
          </div>
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
          {memData.length > 0 && latestMem && (
            <Sparkline
              data={memData}
              max={100}
              color="#a855f7"
              label="Memory"
              currentValue={`${Math.round((latestMem.memTotalMb - latestMem.memAvailableMb) / 1024)}/${Math.round(latestMem.memTotalMb / 1024)} GB`}
            />
          )}
          {swapData.length > 0 && latestMem && latestMem.swapTotalMb > 0 && (
            <Sparkline
              data={swapData}
              max={Math.max(latestMem.swapTotalMb, ...swapData, 1024)}
              color="#dc2626"
              label="Swap"
              currentValue={`${Math.round(latestMem.swapUsedMb)} MB`}
            />
          )}
          {psiMemData.length > 0 && (
            <Sparkline
              data={psiMemData}
              max={Math.max(...psiMemData, 5)}
              color="#f43f5e"
              label="PSI mem"
              unit="%"
            />
          )}
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
          {tpsData.length > 0 && (
            <Sparkline
              data={tpsData}
              max={Math.max(...tpsData, 10)}
              color="#eab308"
              label="TPS"
              unit=" t/s"
            />
          )}
          {netData.map((iface) => (
            <Sparkline
              key={`net-${iface.name}`}
              data={iface.data}
              max={Math.max(...iface.data, 100)}
              color="#8b5cf6"
              label={iface.name}
              currentValue={`${iface.data.length > 0 ? iface.data[iface.data.length - 1].toFixed(0) : 0} Mbps`}
            />
          ))}
          {rdmaData.map((iface) => (
            <Sparkline
              key={`rdma-${iface.name}`}
              data={iface.data}
              max={Math.max(...iface.data, 1000)}
              color="#06b6d4"
              label={`${iface.name} (RDMA)`}
              currentValue={`${iface.data.length > 0 ? (iface.data[iface.data.length - 1] > 1000 ? (iface.data[iface.data.length - 1] / 1000).toFixed(1) + " Gbps" : iface.data[iface.data.length - 1].toFixed(0) + " Mbps") : "0 Mbps"}`}
            />
          ))}
          {diskData.map((disk) => {
            const last = disk.data.length > 0 ? disk.data[disk.data.length - 1] : 0;
            const display = last >= 1000
              ? `${(last / 1000).toFixed(2)} GB/s`
              : `${last.toFixed(1)} MB/s`;
            return (
              <Sparkline
                key={`disk-${disk.name}`}
                data={disk.data}
                max={Math.max(...disk.data, 100)}
                color="#f59e0b"
                label={`${disk.name} (disk)`}
                currentValue={display}
              />
            );
          })}
        </div>
      ) : (
        <p className="text-xs text-gray-500 italic">
          {isWaking
            ? "Waking… — click Wake to retry if it doesn't come back"
            : isOff
            ? "Powered off — Wake to bring it back"
            : "No metrics yet"}
        </p>
      )}
    </div>
  );
}
