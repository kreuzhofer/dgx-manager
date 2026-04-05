"use client";

import { useEffect, useState, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { useDashboardWs } from "@/lib/ws";
import { NodeCard } from "@/components/node-card";

interface NodeMetric {
  gpuUtil: number;
  vramUsed: number;
  tps: number | null;
  temperature: number | null;
}

interface Node {
  id: string;
  name: string;
  ipAddress: string;
  status: string;
  gpuModel: string | null;
  vramTotal: number | null;
  metrics: NodeMetric[];
}

export default function OverviewPage() {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiFetch<Node[]>("/api/nodes")
      .then(setNodes)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleWsMessage = useCallback((msg: { type: string; payload: unknown }) => {
    if (msg.type === "update:metrics") {
      const { nodeId, metrics } = msg.payload as { nodeId: string; metrics: NodeMetric };
      setNodes((prev) =>
        prev.map((n) =>
          n.id === nodeId ? { ...n, metrics: [metrics], status: "online" } : n
        )
      );
    }
    if (msg.type === "update:node") {
      const node = msg.payload as Node;
      setNodes((prev) => prev.map((n) => (n.id === node.id ? { ...n, ...node } : n)));
    }
  }, []);

  const { connected } = useDashboardWs(handleWsMessage);

  if (loading) {
    return <p className="text-gray-400">Loading nodes...</p>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Cluster Overview</h1>
        <span className={`text-xs px-2 py-1 rounded ${connected ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"}`}>
          {connected ? "Live" : "Disconnected"}
        </span>
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
            <NodeCard key={node.id} node={node} />
          ))}
        </div>
      )}
    </div>
  );
}
