"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { apiFetch } from "@/lib/api";
import { useSSE, type SseEvent } from "@/lib/sse";
import { NodeCard } from "@/components/node-card";

interface NodeMetric {
  gpuUtil: number;
  vramUsed: number;
  tps: number | null;
  temperature: number | null;
}

interface MetricSample {
  timestamp: number;
  gpuUtil: number;
  vramUsed: number;
  temperature: number | null;
  tps: number | null;
  activeRequests: number | null;
}

interface Node {
  id: string;
  name: string;
  ipAddress: string;
  status: string;
  gpuModel: string | null;
  vramTotal: number | null;
  agentVersion?: string | null;
  metrics: NodeMetric[];
}

interface DeploymentSummary {
  total: number;
  running: number;
}

export default function OverviewPage() {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [deployments, setDeployments] = useState<DeploymentSummary>({ total: 0, running: 0 });
  const [recipeCount, setRecipeCount] = useState(0);
  const [loading, setLoading] = useState(true);

  // Map of nodeId -> metrics callback from NodeCard
  const metricsHandlers = useRef<Record<string, (sample: MetricSample) => void>>({});

  useEffect(() => {
    Promise.all([
      apiFetch<Node[]>("/api/nodes"),
      apiFetch<{ id: string; status: string }[]>("/api/deployments"),
      apiFetch<unknown[]>("/api/recipes"),
    ])
      .then(([n, d, r]) => {
        setNodes(n);
        setDeployments({
          total: d.length,
          running: d.filter((x) => x.status === "running").length,
        });
        setRecipeCount(r.length);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleSSE = useCallback((event: SseEvent) => {
    if (event.type === "node:metrics") {
      const payload = event.payload as MetricSample & { nodeId: string; temp?: number };
      // Forward to the right NodeCard
      const handler = metricsHandlers.current[payload.nodeId];
      if (handler) {
        handler({
          timestamp: payload.timestamp,
          gpuUtil: payload.gpuUtil,
          vramUsed: payload.vramUsed,
          temperature: payload.temp ?? payload.temperature ?? null,
          tps: payload.tps ?? null,
          activeRequests: payload.activeRequests ?? null,
        });
      }
      // Update node status to online
      setNodes((prev) =>
        prev.map((n) => (n.id === payload.nodeId ? { ...n, status: "online" } : n))
      );
    }
    if (event.type === "node:status") {
      const { nodeId, status } = event.payload as { nodeId: string; status: string };
      setNodes((prev) => prev.map((n) => (n.id === nodeId ? { ...n, status } : n)));
    }
  }, []);

  const { connected } = useSSE(handleSSE);

  if (loading) return <p className="text-gray-400">Loading...</p>;

  const onlineNodes = nodes.filter((n) => n.status === "online").length;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Cluster Overview</h1>
        <span
          className={`text-xs px-2 py-1 rounded ${
            connected ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"
          }`}
        >
          {connected ? "Live" : "Disconnected"}
        </span>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard label="Nodes" value={nodes.length} sub={`${onlineNodes} online`} />
        <StatCard label="Deployments" value={deployments.total} sub={`${deployments.running} running`} />
        <StatCard label="Recipes" value={recipeCount} sub="available" />
        <StatCard
          label="GPU Utilization"
          value={
            onlineNodes > 0
              ? `${Math.round(
                  nodes.reduce((s, n) => s + (n.metrics[0]?.gpuUtil || 0), 0) / onlineNodes
                )}%`
              : "—"
          }
          sub="cluster avg"
        />
      </div>

      {nodes.length === 0 ? (
        <div className="text-gray-400 text-center py-16">
          <p className="text-lg">No nodes added yet.</p>
          <p className="mt-2">
            Go to{" "}
            <a href="/nodes" className="text-green-400 underline">
              Nodes
            </a>{" "}
            to add your first DGX Spark.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {nodes.map((node) => (
            <NodeCard
              key={node.id}
              node={node}
              onMetrics={(handler) => {
                metricsHandlers.current[node.id] = handler;
              }}
            />
          ))}
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
      <p className="text-xs text-gray-500 mt-0.5">{sub}</p>
    </div>
  );
}
