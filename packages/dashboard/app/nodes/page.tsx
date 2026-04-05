"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";

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

const statusBadge: Record<string, string> = {
  green: "bg-green-900 text-green-300",
  yellow: "bg-yellow-900 text-yellow-300",
  red: "bg-red-900 text-red-300",
};

export default function NodesPage() {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [name, setName] = useState("");
  const [ipAddress, setIpAddress] = useState("");
  const [adding, setAdding] = useState(false);

  const loadNodes = () => apiFetch<Node[]>("/api/nodes").then(setNodes).catch(console.error);

  useEffect(() => { loadNodes(); }, []);

  const addNode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !ipAddress) return;
    setAdding(true);
    try {
      await apiFetch("/api/nodes", {
        method: "POST",
        body: JSON.stringify({ name, ipAddress }),
      });
      setName("");
      setIpAddress("");
      // Poll for audit result
      setTimeout(loadNodes, 2000);
      setTimeout(loadNodes, 5000);
      loadNodes();
    } catch (err) {
      alert(String(err));
    } finally {
      setAdding(false);
    }
  };

  const provision = async (nodeId: string) => {
    await apiFetch(`/api/nodes/${nodeId}/provision`, { method: "POST" });
    setTimeout(loadNodes, 3000);
    setTimeout(loadNodes, 10000);
  };

  const deleteNode = async (nodeId: string) => {
    if (!confirm("Delete this node?")) return;
    await apiFetch(`/api/nodes/${nodeId}`, { method: "DELETE" });
    loadNodes();
  };

  const getReport = (node: Node): ProvisionReport | null => {
    if (!node.provisionLog) return null;
    try { return JSON.parse(node.provisionLog); } catch { return null; }
  };

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Node Management</h1>

      {/* Add Node Form */}
      <form onSubmit={addNode} className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6 flex gap-4 items-end">
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
          return (
            <div key={node.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h3 className="font-semibold text-lg">{node.name}</h3>
                  <p className="text-xs text-gray-500">{node.ipAddress} &middot; {node.provisionStatus}</p>
                </div>
                <div className="flex gap-2">
                  {report && report.checks.some((c) => c.status === "yellow") && (
                    <button
                      onClick={() => provision(node.id)}
                      className="bg-yellow-600 hover:bg-yellow-700 text-white px-3 py-1 rounded text-xs font-medium"
                    >
                      Provision
                    </button>
                  )}
                  <button
                    onClick={() => deleteNode(node.id)}
                    className="bg-red-800 hover:bg-red-700 text-white px-3 py-1 rounded text-xs font-medium"
                  >
                    Delete
                  </button>
                </div>
              </div>

              {/* Prerequisite Checklist */}
              {report && report.checks.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mt-3">
                  {report.checks.map((check) => (
                    <div key={check.name} className={`text-xs px-2 py-1 rounded ${statusBadge[check.status]}`}>
                      <span className="font-medium">{check.name}</span>
                      <span className="block text-[10px] opacity-80 truncate">{check.detail}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
