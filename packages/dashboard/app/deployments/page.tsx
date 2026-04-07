"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { useSSE, type SseEvent } from "@/lib/sse";

interface Recipe {
  file: string;
  name: string;
  description?: string;
  model?: string;
  container: string;
  cluster_only?: boolean;
  solo_only?: boolean;
  defaults: Record<string, unknown>;
}

interface Node {
  id: string;
  name: string;
  ipAddress: string;
  status: string;
}

interface ClusterNodeInfo {
  id: string;
  role: string;
  status: string;
  node: { name: string; ipAddress: string };
}

interface Deployment {
  id: string;
  nodeId: string;
  modelId: string;
  status: string;
  port: number | null;
  config: string | null;
  clusterMode: boolean;
  createdAt: string;
  node?: { name: string; ipAddress: string };
  model?: { name: string };
  clusterNodes?: ClusterNodeInfo[];
}

const statusStyles: Record<string, string> = {
  pending: "bg-gray-700 text-gray-300",
  starting: "bg-blue-900 text-blue-300",
  building: "bg-purple-900 text-purple-300",
  downloading: "bg-cyan-900 text-cyan-300",
  launching: "bg-blue-900 text-blue-300",
  loading: "bg-yellow-900 text-yellow-300",
  running: "bg-green-900 text-green-300",
  stopping: "bg-yellow-900 text-yellow-300",
  stopped: "bg-gray-800 text-gray-400",
  failed: "bg-red-900 text-red-300",
  removing: "bg-yellow-900 text-yellow-300",
  restarting: "bg-blue-900 text-blue-300",
};

export default function DeploymentsPage() {
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [loading, setLoading] = useState(true);

  // Deploy form state
  const [selectedRecipe, setSelectedRecipe] = useState<string>("");
  const [idleNodes, setIdleNodes] = useState<Node[]>([]);
  const [selectedNode, setSelectedNode] = useState<string>("");
  const [port, setPort] = useState("8000");
  const [maxModelLen, setMaxModelLen] = useState("");
  const [tensorParallel, setTensorParallel] = useState("");
  const [pipelineParallel, setPipelineParallel] = useState("");
  const [gpuMem, setGpuMem] = useState("");
  const [deploying, setDeploying] = useState(false);

  // Log viewer
  const [viewingLogs, setViewingLogs] = useState<string | null>(null);
  const [logs, setLogs] = useState<Record<string, string>>({});
  const logRef = useRef<HTMLPreElement>(null);

  const loadData = useCallback(() => {
    Promise.all([
      apiFetch<Recipe[]>("/api/recipes"),
      apiFetch<Node[]>("/api/nodes"),
      apiFetch<Deployment[]>("/api/deployments"),
      apiFetch<Node[]>("/api/nodes/idle"),
    ])
      .then(([r, n, d, idle]) => {
        setRecipes(r);
        setNodes(n);
        setDeployments(d);
        setIdleNodes(idle);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // SSE handler for real-time updates
  const handleSSE = useCallback((event: SseEvent) => {
    if (event.type === "deployment:status") {
      const { deploymentId, status, port, error } = event.payload as {
        deploymentId: string; status: string; port?: number; error?: string;
      };
      setDeployments((prev) =>
        prev.map((d) =>
          d.id === deploymentId
            ? { ...d, status, port: port ?? d.port }
            : d
        )
      );
      if (error) {
        setLogs((prev) => ({
          ...prev,
          [deploymentId]: (prev[deploymentId] || "") + `\n[ERROR] ${error}`,
        }));
      }
      // Refresh idle nodes on any terminal or starting state change
      if (["stopped", "failed", "starting", "running"].includes(status)) {
        apiFetch<Node[]>("/api/nodes/idle").then(setIdleNodes).catch(() => {});
      }
    }
    if (event.type === "deployment:log") {
      const { deploymentId, log } = event.payload as { deploymentId: string; log: string };
      setLogs((prev) => ({
        ...prev,
        [deploymentId]: (prev[deploymentId] || "") + log,
      }));
    }
    if (event.type === "deployment:deleted") {
      const { deploymentId } = event.payload as { deploymentId: string };
      setDeployments((prev) => prev.filter((d) => d.id !== deploymentId));
      apiFetch<Node[]>("/api/nodes/idle").then(setIdleNodes).catch(() => {});
    }
    if (event.type === "node:status") {
      const { nodeId, status } = event.payload as { nodeId: string; status: string };
      setNodes((prev) => prev.map((n) => n.id === nodeId ? { ...n, status } : n));
    }
  }, []);

  const { connected } = useSSE(handleSSE);

  // Auto-scroll logs
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs, viewingLogs]);

  const deploy = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedRecipe || !canDeploy) return;
    setDeploying(true);
    try {
      const configOverrides: Record<string, unknown> = {
        port: parseInt(port) || 8000,
      };
      if (maxModelLen) configOverrides.maxModelLen = parseInt(maxModelLen);
      if (tensorParallel) configOverrides.tensorParallel = parseInt(tensorParallel);
      if (pipelineParallel) configOverrides.pipelineParallel = parseInt(pipelineParallel);
      if (gpuMem) configOverrides.gpuMem = parseFloat(gpuMem);

      const body = needsCluster
        ? { nodeIds: "auto", recipeFile: selectedRecipe, config: configOverrides }
        : { nodeId: selectedNode || "auto", recipeFile: selectedRecipe, config: configOverrides };

      const deployment = await apiFetch<Deployment>("/api/deployments", {
        method: "POST",
        body: JSON.stringify(body),
      });
      setDeployments((prev) => [deployment, ...prev]);
      setSelectedRecipe("");
      setViewingLogs(deployment.id);
      loadData(); // Refresh idle nodes
    } catch (err) {
      alert(String(err));
    } finally {
      setDeploying(false);
    }
  };

  const stopDeployment = async (id: string) => {
    await apiFetch(`/api/deployments/${id}`, { method: "DELETE" });
    setDeployments((prev) =>
      prev.map((d) => (d.id === id ? { ...d, status: "stopping" } : d))
    );
  };

  const restartDeployment = async (id: string) => {
    await apiFetch(`/api/deployments/${id}/restart`, { method: "POST" });
    setDeployments((prev) =>
      prev.map((d) => (d.id === id ? { ...d, status: "restarting" } : d))
    );
  };

  const deleteDeployment = async (id: string) => {
    if (!confirm("Stop and delete this deployment?")) return;
    await apiFetch(`/api/deployments/${id}?delete=true`, { method: "DELETE" });
    setDeployments((prev) =>
      prev.map((d) => (d.id === id ? { ...d, status: "stopping" } : d))
    );
  };

  const selectedRecipeData = recipes.find((r) => r.file === selectedRecipe);
  const isClusterRecipe = selectedRecipeData?.cluster_only;

  // Pre-fill config when recipe changes
  const onRecipeChange = (file: string) => {
    setSelectedRecipe(file);
    const recipe = recipes.find((r) => r.file === file);
    if (recipe?.defaults) {
      setPort(String(recipe.defaults.port ?? 8000));
      setMaxModelLen(String(recipe.defaults.max_model_len ?? ""));
      setTensorParallel(String(recipe.defaults.tensor_parallel ?? ""));
      setPipelineParallel(String(recipe.defaults.pipeline_parallel ?? ""));
      setGpuMem(String(recipe.defaults.gpu_memory_utilization ?? ""));
    }
    // Auto-select first idle node for solo recipes
    if (!recipe?.cluster_only && idleNodes.length > 0) {
      setSelectedNode(idleNodes[0].id);
    }
  };

  // Compute auto-selected nodes
  // Compute required nodes from config overrides or recipe defaults
  const effectiveTP = parseInt(tensorParallel) || (selectedRecipeData?.defaults?.tensor_parallel as number) || 1;
  const effectivePP = parseInt(pipelineParallel) || (selectedRecipeData?.defaults?.pipeline_parallel as number) || 1;
  const requiredNodes = effectiveTP * effectivePP;
  const needsCluster = requiredNodes > 1;
  const canDeploy = needsCluster ? idleNodes.length >= requiredNodes : !!selectedNode || idleNodes.length >= 1;

  if (loading) return <p className="text-gray-400">Loading...</p>;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Deployments</h1>
        <span
          className={`text-xs px-2 py-1 rounded ${
            connected ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"
          }`}
        >
          {connected ? "Live" : "Disconnected"}
        </span>
      </div>

      {/* Deploy Form */}
      <form
        onSubmit={deploy}
        className="bg-gray-900 border border-gray-800 rounded-lg p-5 mb-6"
      >
        <h2 className="text-sm font-semibold text-gray-300 mb-4 uppercase tracking-wide">
          New Deployment
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div className="md:col-span-2">
            <label className="block text-xs text-gray-400 mb-1">Recipe</label>
            <select
              value={selectedRecipe}
              onChange={(e) => onRecipeChange(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
            >
              <option value="">Select a recipe...</option>
              {recipes.map((r) => {
                const tp = r.defaults?.tensor_parallel as number | undefined;
                const pp = r.defaults?.pipeline_parallel as number | undefined;
                const suffix = [
                  pp ? `PP=${pp}` : tp && tp > 1 ? `TP=${tp}` : null,
                  r.solo_only ? "solo" : null,
                ].filter(Boolean).join(", ");
                return (
                  <option key={r.file} value={r.file}>
                    {r.name}{suffix ? ` [${suffix}]` : ""}
                  </option>
                );
              })}
            </select>
          </div>
          <div className="md:col-span-2">
            <label className="block text-xs text-gray-400 mb-1">
              {needsCluster ? `Target Nodes (${requiredNodes} needed)` : "Node"}
            </label>
            {needsCluster ? (
              <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                {idleNodes.length < requiredNodes ? (
                  <span className="text-red-400">
                    Need {requiredNodes} nodes but only {idleNodes.length} idle
                  </span>
                ) : (
                  <span className="text-gray-300">
                    <span className="text-green-400 font-medium">
                      {idleNodes.slice(0, requiredNodes).map((n) => n.name).join(", ")}
                    </span>
                    <span className="text-gray-500 ml-1">({requiredNodes} of {idleNodes.length} idle)</span>
                  </span>
                )}
              </div>
            ) : (
              <select
                value={selectedNode}
                onChange={(e) => setSelectedNode(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
              >
                {idleNodes.length === 0 && (
                  <option value="">No idle nodes available</option>
                )}
                {idleNodes.map((n) => (
                  <option key={n.id} value={n.id}>
                    {n.name} ({n.ipAddress})
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>

        {/* Recipe config overrides */}
        {selectedRecipeData && (
          <>
            <div className="mt-3 p-3 bg-gray-800/50 rounded border border-gray-700/50">
              <p className="text-xs text-gray-400 mb-2">
                {selectedRecipeData.description || "No description"}
                {selectedRecipeData.model && (
                  <span className="ml-2 text-gray-500">({selectedRecipeData.model})</span>
                )}
              </p>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div>
                  <label className="block text-[10px] text-gray-500 mb-0.5">Port</label>
                  <input
                    type="number"
                    value={port}
                    onChange={(e) => setPort(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-gray-500 mb-0.5">Tensor Parallel</label>
                  <input
                    type="number"
                    value={tensorParallel}
                    onChange={(e) => setTensorParallel(e.target.value)}
                    placeholder={String(selectedRecipeData.defaults.tensor_parallel ?? "")}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-gray-500 mb-0.5">Pipeline Parallel</label>
                  <input
                    type="number"
                    value={pipelineParallel}
                    onChange={(e) => setPipelineParallel(e.target.value)}
                    placeholder={String(selectedRecipeData.defaults.pipeline_parallel ?? "")}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-gray-500 mb-0.5">Max Model Length</label>
                  <input
                    type="number"
                    value={maxModelLen}
                    onChange={(e) => setMaxModelLen(e.target.value)}
                    placeholder={String(selectedRecipeData.defaults.max_model_len ?? "")}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-gray-500 mb-0.5">GPU Memory Util</label>
                  <input
                    type="number"
                    step="0.05"
                    min="0.1"
                    max="0.99"
                    value={gpuMem}
                    onChange={(e) => setGpuMem(e.target.value)}
                    placeholder={String(selectedRecipeData.defaults.gpu_memory_utilization ?? "")}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                  />
                </div>
              </div>
            </div>
          </>
        )}

        <div className="mt-4 flex justify-end">
          <button
            type="submit"
            disabled={deploying || !selectedRecipe || !canDeploy}
            className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-5 py-2 rounded text-sm font-medium transition-colors"
          >
            {deploying ? "Deploying..." : "Deploy"}
          </button>
        </div>
      </form>

      {/* No recipes warning */}
      {recipes.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          <p>No recipes available. Connect an agent to discover vLLM recipes.</p>
        </div>
      )}

      {/* Deployments list */}
      {deployments.length > 0 && (
        <div className="space-y-3">
          {deployments.map((d) => {
            const config = d.config ? JSON.parse(d.config) : {};
            const recipeName = config.recipeFile
              ?.replace(/^recipes\//, "")
              .replace(/\.yaml$/, "");
            const isActive = ["running", "starting", "pending", "restarting", "building", "downloading", "launching", "loading"].includes(d.status);
            const isStopping = ["stopping", "removing"].includes(d.status);

            return (
              <div
                key={d.id}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div>
                      <h3 className="font-semibold">
                        {d.model?.name || recipeName || d.modelId}
                      </h3>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {d.clusterMode && d.clusterNodes ? (
                          <span className="text-indigo-400">
                            {d.clusterNodes.length} nodes
                          </span>
                        ) : (
                          <>
                            {d.node?.name || d.nodeId}
                            {d.node?.ipAddress && ` (${d.node.ipAddress})`}
                          </>
                        )}
                        {d.port && (
                          <span className="ml-2 text-gray-400">
                            :{d.port}
                          </span>
                        )}
                        <span className="ml-2">
                          {new Date(d.createdAt).toLocaleString()}
                        </span>
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs px-2.5 py-1 rounded font-medium ${
                        statusStyles[d.status] || "bg-gray-700 text-gray-300"
                      }`}
                    >
                      {d.status}
                    </span>
                    <button
                      onClick={() =>
                        setViewingLogs(viewingLogs === d.id ? null : d.id)
                      }
                      className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 transition-colors"
                    >
                      {viewingLogs === d.id ? "Hide Logs" : "Logs"}
                    </button>
                    {d.status === "running" && d.port && (
                      <a
                        href={`http://${d.node?.ipAddress || "localhost"}:${d.port}/v1/models`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-green-400 transition-colors"
                      >
                        API
                      </a>
                    )}
                    {isStopping && (
                      <span className="text-xs px-2 py-1 rounded bg-yellow-900/30 text-yellow-400 animate-pulse">
                        Stopping...
                      </span>
                    )}
                    {isActive && !isStopping && (
                      <button
                        onClick={() => stopDeployment(d.id)}
                        className="text-xs px-2 py-1 rounded bg-red-900/50 hover:bg-red-800 text-red-300 transition-colors"
                      >
                        Stop
                      </button>
                    )}
                    {(d.status === "stopped" || d.status === "failed") && (
                      <>
                        <button
                          onClick={() => restartDeployment(d.id)}
                          className="text-xs px-2 py-1 rounded bg-blue-900/50 hover:bg-blue-800 text-blue-300 transition-colors"
                        >
                          Restart
                        </button>
                        <button
                          onClick={() => deleteDeployment(d.id)}
                          className="text-xs px-2 py-1 rounded bg-red-900/50 hover:bg-red-800 text-red-300 transition-colors"
                        >
                          Delete
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {/* Cluster nodes detail */}
                {d.clusterMode && d.clusterNodes && d.clusterNodes.length > 0 && (
                  <div className="mt-3 flex gap-2 flex-wrap">
                    {d.clusterNodes.map((cn) => (
                      <div
                        key={cn.id}
                        className={`text-xs px-2 py-1 rounded border ${
                          cn.role === "head"
                            ? "border-indigo-700 bg-indigo-900/30 text-indigo-300"
                            : "border-gray-700 bg-gray-800/50 text-gray-400"
                        }`}
                      >
                        <span className="font-medium">{cn.node.name}</span>
                        <span className="ml-1 text-[10px] opacity-70">
                          {cn.role === "head" ? "HEAD" : "worker"}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Log viewer */}
                {viewingLogs === d.id && (
                  <pre
                    ref={logRef}
                    className="mt-3 bg-black/50 border border-gray-800 rounded p-3 text-xs text-gray-400 font-mono max-h-64 overflow-y-auto whitespace-pre-wrap"
                  >
                    {logs[d.id] || "Waiting for logs..."}
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      )}

      {deployments.length === 0 && recipes.length > 0 && (
        <div className="text-center py-12 text-gray-500">
          <p>No deployments yet. Select a recipe and node above to deploy.</p>
        </div>
      )}
    </div>
  );
}
