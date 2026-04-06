"use client";

import { useEffect, useState, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { useSSE, type SseEvent } from "@/lib/sse";

interface PrereqCheck {
  name: string;
  status: "green" | "yellow" | "red";
  detail: string;
}

interface ProvisionReport {
  reachable: boolean;
  sudoAvailable: boolean;
  systemInfo: string;
  checks: PrereqCheck[];
}

interface Node {
  id: string;
  name: string;
  ipAddress: string;
  status: string;
  provisionStatus: string;
  provisionLog: string | null;
  gpuModel: string | null;
  dockerAvailable: boolean;
  ollamaInstalled: boolean;
  createdAt: string;
}

interface ProvisionStep {
  step: string;
  status: string;
  detail?: string;
}

const checkBadge: Record<string, string> = {
  green: "bg-green-900/60 text-green-300 border-green-800",
  yellow: "bg-yellow-900/60 text-yellow-300 border-yellow-800",
  red: "bg-red-900/60 text-red-300 border-red-800",
  checking: "bg-blue-900/60 text-blue-300 border-blue-800 animate-pulse",
  installing: "bg-blue-900/60 text-blue-300 border-blue-800 animate-pulse",
  installed: "bg-green-900/60 text-green-300 border-green-800",
};

const provisionBadge: Record<string, string> = {
  pending: "bg-gray-700 text-gray-300",
  auditing: "bg-blue-900 text-blue-300",
  audited: "bg-green-900 text-green-300",
  provisioning: "bg-blue-900 text-blue-300",
  provisioned: "bg-green-900 text-green-300",
  "agent-deployed": "bg-green-900 text-green-300",
  unreachable: "bg-red-900 text-red-300",
  failed: "bg-red-900 text-red-300",
};

export default function NodesPage() {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [name, setName] = useState("");
  const [ipAddress, setIpAddress] = useState("");
  const [adding, setAdding] = useState(false);

  // Real-time provision steps per node
  const [liveSteps, setLiveSteps] = useState<Record<string, ProvisionStep[]>>({});

  const loadNodes = () =>
    apiFetch<Node[]>("/api/nodes").then(setNodes).catch(console.error);

  useEffect(() => {
    loadNodes();
  }, []);

  const handleSSE = useCallback((event: SseEvent) => {
    if (event.type === "node:provision") {
      const { nodeId, step, status, detail, provisionStatus, report } =
        event.payload as {
          nodeId: string;
          step: string;
          status: string;
          detail?: string;
          provisionStatus?: string;
          report?: ProvisionReport;
        };

      // Append live step
      setLiveSteps((prev) => {
        const existing = prev[nodeId] || [];
        // Update existing step or append new
        const idx = existing.findIndex((s) => s.step === step);
        const updated =
          idx >= 0
            ? existing.map((s, i) => (i === idx ? { step, status, detail } : s))
            : [...existing, { step, status, detail }];
        return { ...prev, [nodeId]: updated };
      });

      // Update node provision status if final
      if (provisionStatus) {
        setNodes((prev) =>
          prev.map((n) =>
            n.id === nodeId
              ? {
                  ...n,
                  provisionStatus,
                  provisionLog: report ? JSON.stringify(report) : n.provisionLog,
                }
              : n
          )
        );
        // Clear live steps after a delay when done
        if (["audited", "provisioned", "unreachable", "failed"].includes(provisionStatus)) {
          setTimeout(() => {
            setLiveSteps((prev) => {
              const copy = { ...prev };
              delete copy[nodeId];
              return copy;
            });
            loadNodes();
          }, 3000);
        }
      }
    }
    if (event.type === "node:status") {
      const { nodeId, status } = event.payload as { nodeId: string; status: string };
      setNodes((prev) => prev.map((n) => (n.id === nodeId ? { ...n, status } : n)));
    }
  }, []);

  const { connected } = useSSE(handleSSE);

  const addNode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !ipAddress) return;
    setAdding(true);
    try {
      const node = await apiFetch<Node>("/api/nodes", {
        method: "POST",
        body: JSON.stringify({ name, ipAddress }),
      });
      setNodes((prev) => [node, ...prev]);
      setName("");
      setIpAddress("");
    } catch (err) {
      alert(String(err));
    } finally {
      setAdding(false);
    }
  };

  const provision = async (nodeId: string) => {
    setNodes((prev) =>
      prev.map((n) => (n.id === nodeId ? { ...n, provisionStatus: "provisioning" } : n))
    );
    await apiFetch(`/api/nodes/${nodeId}/provision`, { method: "POST" });
  };

  const deleteNode = async (nodeId: string) => {
    if (!confirm("Delete this node and stop all its deployments?")) return;
    await apiFetch(`/api/nodes/${nodeId}`, { method: "DELETE" });
    setNodes((prev) => prev.filter((n) => n.id !== nodeId));
  };

  const getReport = (node: Node): ProvisionReport | null => {
    if (!node.provisionLog) return null;
    try {
      return JSON.parse(node.provisionLog);
    } catch {
      return null;
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Node Management</h1>
        <span
          className={`text-xs px-2 py-1 rounded ${
            connected ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"
          }`}
        >
          {connected ? "Live" : "Disconnected"}
        </span>
      </div>

      {/* Add Node Form */}
      <form
        onSubmit={addNode}
        className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6 flex gap-4 items-end"
      >
        <div className="flex-1">
          <label className="block text-xs text-gray-400 mb-1">Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="dgx-spark-01"
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
          />
        </div>
        <div className="flex-1">
          <label className="block text-xs text-gray-400 mb-1">IP Address</label>
          <input
            type="text"
            value={ipAddress}
            onChange={(e) => setIpAddress(e.target.value)}
            placeholder="192.168.1.100"
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
          />
        </div>
        <button
          type="submit"
          disabled={adding}
          className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white px-4 py-2 rounded text-sm font-medium transition-colors"
        >
          {adding ? "Adding..." : "Add Node"}
        </button>
      </form>

      {/* Node List */}
      <div className="space-y-4">
        {nodes.map((node) => {
          const report = getReport(node);
          const steps = liveSteps[node.id];
          const isProvisioning = ["provisioning"].includes(node.provisionStatus);
          const isAuditing = !!steps && !report && node.provisionStatus === "pending";

          return (
            <div
              key={node.id}
              className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors"
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold text-lg">{node.name}</h3>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded font-medium ${
                        provisionBadge[node.provisionStatus] || "bg-gray-700 text-gray-300"
                      }`}
                    >
                      {node.provisionStatus}
                    </span>
                    {node.status === "online" && (
                      <span className="flex items-center gap-1 text-[10px] text-green-400">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                        online
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {node.ipAddress}
                    {node.gpuModel && (
                      <span className="ml-2 text-gray-400">{node.gpuModel}</span>
                    )}
                  </p>
                </div>
                <div className="flex gap-2">
                  {report &&
                    report.checks.some((c) => c.status === "yellow") &&
                    !isProvisioning && (
                      <button
                        onClick={() => provision(node.id)}
                        className="bg-yellow-600 hover:bg-yellow-700 text-white px-3 py-1 rounded text-xs font-medium transition-colors"
                      >
                        Provision
                      </button>
                    )}
                  <button
                    onClick={() => deleteNode(node.id)}
                    className="bg-red-800/60 hover:bg-red-700 text-red-300 px-3 py-1 rounded text-xs font-medium transition-colors"
                  >
                    Delete
                  </button>
                </div>
              </div>

              {/* Live provisioning steps */}
              {steps && steps.length > 0 && (
                <div className="mb-3 bg-black/30 border border-gray-800 rounded p-3">
                  <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-2">
                    {isProvisioning ? "Provisioning" : "Auditing"} in progress...
                  </p>
                  <div className="space-y-1">
                    {steps.map((s, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <StepIndicator status={s.status} />
                        <span className="text-gray-300 font-medium">{s.step}</span>
                        {s.detail && (
                          <span className="text-gray-500 truncate">{s.detail}</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Prerequisite Checklist */}
              {report && report.checks.length > 0 && !steps && (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {report.checks.map((check) => (
                    <div
                      key={check.name}
                      className={`text-xs px-2.5 py-1.5 rounded border ${
                        checkBadge[check.status] || "bg-gray-800 text-gray-400 border-gray-700"
                      }`}
                    >
                      <span className="font-medium">{check.name}</span>
                      <span className="block text-[10px] opacity-80 truncate mt-0.5">
                        {check.detail}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Not yet audited */}
              {!report && !steps && node.provisionStatus === "pending" && (
                <p className="text-xs text-gray-500 italic">Audit starting...</p>
              )}
            </div>
          );
        })}
      </div>

      {nodes.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <p>No nodes yet. Add a DGX Spark above to get started.</p>
        </div>
      )}
    </div>
  );
}

function StepIndicator({ status }: { status: string }) {
  if (status === "running" || status === "checking" || status === "installing") {
    return (
      <span className="w-4 h-4 flex items-center justify-center">
        <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
      </span>
    );
  }
  if (status === "done" || status === "green" || status === "installed") {
    return (
      <span className="w-4 h-4 flex items-center justify-center text-green-400 text-[10px]">
        &#10003;
      </span>
    );
  }
  if (status === "failed" || status === "red") {
    return (
      <span className="w-4 h-4 flex items-center justify-center text-red-400 text-[10px]">
        &#10007;
      </span>
    );
  }
  if (status === "yellow") {
    return (
      <span className="w-4 h-4 flex items-center justify-center text-yellow-400 text-[10px]">
        &#9888;
      </span>
    );
  }
  return <span className="w-4 h-4 flex items-center justify-center text-gray-500 text-[10px]">&#8226;</span>;
}
