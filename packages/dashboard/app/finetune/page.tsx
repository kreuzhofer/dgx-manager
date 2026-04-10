"use client";

import { useEffect, useState, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { useSSE, type SseEvent } from "@/lib/sse";

interface TrainingRecipe {
  file: string;
  name: string;
  description?: string;
  base_model: string;
  framework: string;
  method: string;
  dataset_format?: string;
  defaults: Record<string, unknown>;
  hardware: { min_nodes: number; gpus_per_node: number; vram_estimate_mb: number };
}

interface Node {
  id: string;
  name: string;
  ipAddress: string;
  status: string;
}

interface FineTuneJob {
  id: string;
  nodeId: string;
  recipeFile: string | null;
  baseModel: string;
  method: string;
  dataset: string;
  config: string | null;
  status: string;
  progress: number | null;
  outputDir: string | null;
  outputPath: string | null;
  mergeStatus: string | null;
  mergedPath: string | null;
  deploymentId: string | null;
  logs: string | null;
  startedAt: string | null;
  completedAt: string | null;
  createdAt: string;
  node?: { name: string; ipAddress: string };
}

const statusStyles: Record<string, string> = {
  pending: "bg-gray-700 text-gray-300",
  starting: "bg-blue-900 text-blue-300",
  running: "bg-green-900 text-green-300",
  stopping: "bg-yellow-900 text-yellow-300",
  stopped: "bg-gray-800 text-gray-400",
  completed: "bg-green-900 text-green-300",
  failed: "bg-red-900 text-red-300",
};

export default function FinetunePage() {
  const [recipes, setRecipes] = useState<TrainingRecipe[]>([]);
  const [idleNodes, setIdleNodes] = useState<Node[]>([]);
  const [jobs, setJobs] = useState<FineTuneJob[]>([]);
  const [loading, setLoading] = useState(true);

  // Form state
  const [selectedRecipe, setSelectedRecipe] = useState("");
  const [selectedNode, setSelectedNode] = useState("");
  const [dataset, setDataset] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Hyperparameter overrides
  const [learningRate, setLearningRate] = useState("");
  const [batchSize, setBatchSize] = useState("");
  const [maxSeqLength, setMaxSeqLength] = useState("");
  const [loraR, setLoraR] = useState("");
  const [loraAlpha, setLoraAlpha] = useState("");
  const [numEpochs, setNumEpochs] = useState("");
  const [maxSteps, setMaxSteps] = useState("");

  // Log viewer
  const [viewingLogs, setViewingLogs] = useState<string | null>(null);
  const [logs, setLogs] = useState<Record<string, string>>({});

  // Per-job phase tracking
  const [jobPhases, setJobPhases] = useState<Record<string, {
    phase: string;
    phaseProgress: number;
    step?: number;
    totalSteps?: number;
    loss?: number;
    etaSeconds?: number;
  }>>({});

  // Per-job training metrics for loss curve
  const [jobMetrics, setJobMetrics] = useState<Record<string, {
    steps: number[];
    losses: number[];
    lrs: number[];
    evalLosses: (number | null)[];
    loaded: boolean;
  }>>({});

  const loadData = useCallback(() => {
    Promise.all([
      apiFetch<TrainingRecipe[]>("/api/training-recipes"),
      apiFetch<Node[]>("/api/nodes/idle"),
      apiFetch<FineTuneJob[]>("/api/finetune"),
    ])
      .then(([r, n, j]) => {
        setRecipes(r);
        setIdleNodes(n);
        if (n.length > 0 && !selectedNode) setSelectedNode(n[0].id);
        setJobs(j);
        // Load metrics for active/completed jobs
        for (const job of j) {
          if (["starting", "running", "completed"].includes(job.status)) {
            apiFetch<{ step: number; loss: number; lr: number | null; evalLoss: number | null }[]>(`/api/finetune/${job.id}/metrics`)
              .then(metrics => {
                if (metrics.length > 0) {
                  setJobMetrics(prev => ({
                    ...prev,
                    [job.id]: {
                      steps: metrics.map(m => m.step),
                      losses: metrics.map(m => m.loss),
                      lrs: metrics.map(m => m.lr ?? 0),
                      evalLosses: metrics.map(m => m.evalLoss),
                      loaded: true,
                    },
                  }));
                }
              })
              .catch(() => {});
          }
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const handleSSE = useCallback((event: SseEvent) => {
    if (event.type === "finetune:created") {
      const job = event.payload as unknown as FineTuneJob;
      if (job?.id) {
        setJobs((prev) => prev.some((j) => j.id === job.id) ? prev : [job, ...prev]);
      }
    }
    if (event.type === "finetune:log") {
      const { jobId, phase, phaseProgress, step, totalSteps, loss, lr, evalLoss, etaSeconds, log } = event.payload as {
        jobId: string; phase?: string; phaseProgress?: number; step?: number; totalSteps?: number; loss?: number; lr?: number; evalLoss?: number; etaSeconds?: number; log?: string;
      };
      if (log) {
        setLogs((prev) => ({
          ...prev,
          [jobId]: (prev[jobId] || "") + log,
        }));
      }
      if (phase) {
        setJobPhases((prev) => ({
          ...prev,
          [jobId]: {
            phase,
            phaseProgress: typeof phaseProgress === "number" && phaseProgress >= 0 ? phaseProgress : (prev[jobId]?.phaseProgress ?? 0),
            step: step ?? prev[jobId]?.step,
            totalSteps: totalSteps ?? prev[jobId]?.totalSteps,
            loss: loss ?? prev[jobId]?.loss,
            etaSeconds: etaSeconds ?? prev[jobId]?.etaSeconds,
          },
        }));
        // Update job status to running once training starts
        if (phase === "training" || phase === "loading") {
          setJobs((prev) =>
            prev.map((j) => j.id === jobId && j.status === "starting" ? { ...j, status: "running" } : j)
          );
        }
        // Persist training progress on the job
        if (phase === "training" && typeof phaseProgress === "number" && phaseProgress > 0) {
          setJobs((prev) =>
            prev.map((j) => j.id === jobId ? { ...j, progress: phaseProgress } : j)
          );
        }
      }
      // Accumulate metrics for loss curve
      if (typeof step === "number" && typeof loss === "number") {
        setJobMetrics((prev) => {
          const existing = prev[jobId] || { steps: [], losses: [], lrs: [], evalLosses: [], loaded: false };
          // Avoid duplicates
          if (existing.steps.length > 0 && existing.steps[existing.steps.length - 1] >= step) return prev;
          return {
            ...prev,
            [jobId]: {
              steps: [...existing.steps, step],
              losses: [...existing.losses, loss],
              lrs: [...existing.lrs, lr ?? 0],
              evalLosses: [...existing.evalLosses, evalLoss ?? null],
              loaded: existing.loaded,
            },
          };
        });
      }
    }
    if (event.type === "finetune:status") {
      const { jobId, status, outputPath, error } = event.payload as {
        jobId: string; status: string; outputPath?: string; error?: string;
      };
      setJobs((prev) =>
        prev.map((j) =>
          j.id === jobId
            ? { ...j, status, outputPath: outputPath ?? j.outputPath }
            : j
        )
      );
      if (error) {
        setLogs((prev) => ({
          ...prev,
          [jobId]: (prev[jobId] || "") + `\n[ERROR] ${error}\n`,
        }));
      }
      if (["completed", "failed", "stopped"].includes(status)) {
        apiFetch<Node[]>("/api/nodes/idle").then(setIdleNodes).catch(() => {});
      }
    }
    if (event.type === "finetune:merge-progress") {
      const { jobId, log } = event.payload as { jobId: string; log?: string };
      if (log) {
        setLogs((prev) => ({
          ...prev,
          [jobId]: (prev[jobId] || "") + log,
        }));
      }
    }
    if (event.type === "finetune:merge-status") {
      const { jobId, status, mergedPath } = event.payload as {
        jobId: string; status: string; mergedPath?: string;
      };
      setJobs((prev) =>
        prev.map((j) => j.id === jobId ? { ...j, mergeStatus: status, mergedPath: mergedPath ?? j.mergedPath } : j)
      );
    }
  }, []);

  const { connected } = useSSE(handleSSE, loadData);

  // Auto-scroll log viewers
  useEffect(() => {
    document.querySelectorAll("[data-log-viewer]").forEach((el) => {
      el.scrollTop = el.scrollHeight;
    });
  }, [logs, viewingLogs]);

  const selectedRecipeData = recipes.find((r) => r.file === selectedRecipe);

  const onRecipeChange = (file: string) => {
    setSelectedRecipe(file);
    const recipe = recipes.find((r) => r.file === file);
    if (recipe?.defaults) {
      setLearningRate(String(recipe.defaults.learning_rate ?? ""));
      setBatchSize(String(recipe.defaults.batch_size ?? ""));
      setMaxSeqLength(String(recipe.defaults.max_seq_length ?? ""));
      setLoraR(String(recipe.defaults.lora_r ?? ""));
      setLoraAlpha(String(recipe.defaults.lora_alpha ?? ""));
      setNumEpochs(String(recipe.defaults.num_train_epochs ?? ""));
      setMaxSteps(String(recipe.defaults.max_steps ?? ""));
    }
  };

  const startTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedRecipe || !selectedNode || !dataset) return;
    setSubmitting(true);
    try {
      const config: Record<string, unknown> = {};
      if (learningRate) config.learning_rate = parseFloat(learningRate);
      if (batchSize) config.batch_size = parseInt(batchSize);
      if (maxSeqLength) config.max_seq_length = parseInt(maxSeqLength);
      if (loraR) config.lora_r = parseInt(loraR);
      if (loraAlpha) config.lora_alpha = parseInt(loraAlpha);
      if (numEpochs) config.num_train_epochs = parseInt(numEpochs);
      if (maxSteps) config.max_steps = parseInt(maxSteps);

      const selectedRecipeData = recipes.find((r) => r.file === selectedRecipe);
      const needsMultiNode = (selectedRecipeData?.hardware?.min_nodes ?? 1) > 1;

      const body: Record<string, unknown> = { recipeFile: selectedRecipe, dataset, config };
      if (needsMultiNode) {
        // Send all selected idle nodes (up to min_nodes)
        const requiredNodes = selectedRecipeData!.hardware.min_nodes;
        body.nodeIds = idleNodes.slice(0, requiredNodes).map(n => n.id);
      } else {
        body.nodeId = selectedNode;
      }

      const job = await apiFetch<FineTuneJob>("/api/finetune", {
        method: "POST",
        body: JSON.stringify(body),
      });
      setJobs((prev) => [job, ...prev]);
      setViewingLogs(job.id);
      setDataset("");
    } catch (err) {
      alert(String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const stopJob = async (id: string) => {
    await apiFetch(`/api/finetune/${id}/stop`, { method: "POST" });
    setJobs((prev) =>
      prev.map((j) => (j.id === id ? { ...j, status: "stopping" } : j))
    );
  };

  const deleteJob = async (id: string) => {
    if (!confirm("Delete this fine-tune job?")) return;
    await apiFetch(`/api/finetune/${id}`, { method: "DELETE" });
    setJobs((prev) => prev.filter((j) => j.id !== id));
  };

  const mergeJob = async (id: string) => {
    try {
      await apiFetch(`/api/finetune/${id}/merge`, { method: "POST" });
      setJobs((prev) =>
        prev.map((j) => j.id === id ? { ...j, mergeStatus: "running" } : j)
      );
    } catch (err) { alert(String(err)); }
  };

  const deployJob = (job: FineTuneJob) => {
    const modelPath = job.mergedPath || `${job.outputDir}/merged`;
    const params = new URLSearchParams({
      finetuneModel: modelPath,
      finetuneJobId: job.id,
      baseModel: job.baseModel,
    });
    window.location.href = `/deployments?${params.toString()}`;
  };

  const formatEta = (seconds: number | undefined) => {
    if (seconds == null || seconds <= 0) return "";
    if (seconds < 60) return `~${seconds}s left`;
    const mins = Math.floor(seconds / 60);
    if (mins < 60) return `~${mins}m left`;
    const hrs = Math.floor(mins / 60);
    return `~${hrs}h ${mins % 60}m left`;
  };

  const formatElapsed = (startedAt: string | null) => {
    if (!startedAt) return "";
    const ms = Date.now() - new Date(startedAt).getTime();
    const mins = Math.floor(ms / 60000);
    const secs = Math.floor((ms % 60000) / 1000);
    if (mins > 60) {
      const hrs = Math.floor(mins / 60);
      return `${hrs}h ${mins % 60}m`;
    }
    return `${mins}m ${secs}s`;
  };

  if (loading) return <p className="text-gray-400">Loading...</p>;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Fine-tuning Jobs</h1>
        <span
          className={`text-xs px-2 py-1 rounded ${
            connected ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"
          }`}
        >
          {connected ? "Live" : "Disconnected"}
        </span>
      </div>

      {/* Create Job Form */}
      <form
        onSubmit={startTraining}
        className="bg-gray-900 border border-gray-800 rounded-lg p-5 mb-6"
      >
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-4">
          New Fine-Tune Job
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Training Recipe</label>
            <select
              value={selectedRecipe}
              onChange={(e) => onRecipeChange(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
            >
              <option value="">Select a recipe...</option>
              {recipes.map((r) => (
                <option key={r.file} value={r.file}>
                  {r.name} ({r.method}, {r.framework})
                </option>
              ))}
            </select>
          </div>
          <div>
            {(() => {
              const recipeData = recipes.find((r) => r.file === selectedRecipe);
              const minNodes = recipeData?.hardware?.min_nodes ?? 1;
              const needsMulti = minNodes > 1;

              if (needsMulti) {
                return (
                  <>
                    <label className="block text-xs text-gray-400 mb-1">
                      Nodes <span className="text-gray-600">({minNodes} required)</span>
                    </label>
                    <div className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                      {idleNodes.length < minNodes ? (
                        <span className="text-red-400">
                          Need {minNodes} nodes but only {idleNodes.length} idle
                        </span>
                      ) : (
                        <span className="text-green-400">
                          {idleNodes.slice(0, minNodes).map(n => n.name).join(", ")}
                          <span className="text-gray-500 ml-1">({minNodes} of {idleNodes.length} idle)</span>
                        </span>
                      )}
                    </div>
                  </>
                );
              }

              return (
                <>
                  <label className="block text-xs text-gray-400 mb-1">Node</label>
                  <select
                    value={selectedNode}
                    onChange={(e) => setSelectedNode(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                  >
                    {idleNodes.length === 0 ? (
                      <option value="">No idle nodes available</option>
                    ) : (
                      <>
                        {!selectedNode && <option value="">Select a node...</option>}
                        {idleNodes.map((n) => (
                          <option key={n.id} value={n.id}>
                            {n.name} ({n.ipAddress})
                          </option>
                        ))}
                      </>
                    )}
                  </select>
                </>
              );
            })()}
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">
              Dataset
              <span className="ml-1 text-gray-600">(NFS path or HuggingFace ID)</span>
            </label>
            <input
              type="text"
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              placeholder="/mnt/tank/data/my-dataset.jsonl"
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
            />
          </div>
        </div>

        {/* Recipe details + hyperparameter overrides */}
        {selectedRecipeData && (
          <div className="mt-3 p-3 bg-gray-800/50 rounded border border-gray-700/50">
            <p className="text-xs text-gray-400 mb-2">
              {selectedRecipeData.description || "No description"}
              <span className="ml-2 text-gray-500">
                ({selectedRecipeData.base_model})
              </span>
              {selectedRecipeData.dataset_format && (
                <span className="ml-2 text-gray-600">
                  Format: {selectedRecipeData.dataset_format}
                </span>
              )}
            </p>
            <div className="grid grid-cols-2 md:grid-cols-7 gap-3">
              <div>
                <label className="block text-[10px] text-gray-500 mb-0.5">Learning Rate</label>
                <input
                  type="number"
                  step="0.00001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(e.target.value)}
                  placeholder={String(selectedRecipeData.defaults.learning_rate ?? "")}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-0.5">Batch Size</label>
                <input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(e.target.value)}
                  placeholder={String(selectedRecipeData.defaults.batch_size ?? "")}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-0.5">Max Seq Length</label>
                <input
                  type="number"
                  value={maxSeqLength}
                  onChange={(e) => setMaxSeqLength(e.target.value)}
                  placeholder={String(selectedRecipeData.defaults.max_seq_length ?? "")}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-0.5">LoRA Rank (r)</label>
                <input
                  type="number"
                  value={loraR}
                  onChange={(e) => setLoraR(e.target.value)}
                  placeholder={String(selectedRecipeData.defaults.lora_r ?? "")}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-0.5">LoRA Alpha</label>
                <input
                  type="number"
                  value={loraAlpha}
                  onChange={(e) => setLoraAlpha(e.target.value)}
                  placeholder={String(selectedRecipeData.defaults.lora_alpha ?? "")}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-0.5">Epochs</label>
                <input
                  type="number"
                  value={numEpochs}
                  onChange={(e) => setNumEpochs(e.target.value)}
                  placeholder={String(selectedRecipeData.defaults.num_train_epochs ?? "")}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-0.5">Max Steps</label>
                <input
                  type="number"
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(e.target.value)}
                  placeholder={String(selectedRecipeData.defaults.max_steps ?? "")}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                />
              </div>
            </div>
          </div>
        )}

        <div className="mt-4 flex justify-end">
          <button
            type="submit"
            disabled={submitting || !selectedRecipe || !dataset || (() => {
              const r = recipes.find(x => x.file === selectedRecipe);
              const minNodes = r?.hardware?.min_nodes ?? 1;
              return minNodes > 1 ? idleNodes.length < minNodes : !selectedNode;
            })()}
            className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-5 py-2 rounded text-sm font-medium transition-colors"
          >
            {submitting ? "Starting..." : "Start Training"}
          </button>
        </div>
      </form>

      {/* No recipes info */}
      {recipes.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          <p>No training recipes available. Connect an agent with TRAINING_REPO_PATH configured.</p>
        </div>
      )}

      {/* Jobs list */}
      {jobs.length > 0 && (
        <div className="space-y-3">
          {jobs.map((job) => {
            const isActive = ["pending", "starting", "running"].includes(job.status);
            const isStopping = job.status === "stopping";
            const phaseInfo = jobPhases[job.id];
            const phaseLabels: Record<string, string> = {
              container: "Preparing container",
              setup: "Installing dependencies",
              downloading: "Downloading model",
              loading: "Loading model",
              tokenizing: "Tokenizing dataset",
              training: "Training",
              eval: "Evaluating",
              saving: "Saving adapter",
            };
            const phaseColors: Record<string, string> = {
              container: "bg-gray-500",
              setup: "bg-gray-500",
              downloading: "bg-cyan-500",
              loading: "bg-blue-500",
              tokenizing: "bg-indigo-500",
              training: "bg-green-500",
              eval: "bg-yellow-500",
              saving: "bg-purple-500",
            };

            return (
              <div
                key={job.id}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div>
                      <h3 className="font-semibold flex items-center gap-2">
                        <span className="text-[10px] px-1.5 py-0.5 rounded font-medium bg-purple-900/60 text-purple-300">
                          {job.method}
                        </span>
                        {job.baseModel}
                      </h3>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {job.node?.name || job.nodeId}
                        {job.node?.ipAddress && ` (${job.node.ipAddress})`}
                        <span className="ml-2 text-gray-600">{job.dataset}</span>
                        {job.startedAt && (
                          <span className="ml-2">{formatElapsed(job.startedAt)}</span>
                        )}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs px-2.5 py-1 rounded font-medium ${
                        statusStyles[job.status] || "bg-gray-700 text-gray-300"
                      }`}
                    >
                      {job.status}
                    </span>
                    <button
                      onClick={() => {
                        if (viewingLogs === job.id) {
                          setViewingLogs(null);
                        } else {
                          setViewingLogs(job.id);
                          // Fetch persisted logs from train.log file
                          const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";
                          fetch(`${apiBase}/api/finetune/${job.id}/logs`, { cache: "no-store" })
                            .then(r => r.text())
                            .then(text => {
                              if (text) {
                                // File content is the base, SSE content appends after
                                setLogs(prev => {
                                  const sseContent = prev[job.id] || "";
                                  return { ...prev, [job.id]: text + (sseContent.length > text.length ? sseContent.slice(text.length) : "") };
                                });
                              }
                            })
                            .catch(() => {});
                        }
                      }}
                      className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 transition-colors"
                    >
                      {viewingLogs === job.id ? "Hide Logs" : "Logs"}
                    </button>
                    {isStopping && (
                      <span className="text-xs px-2 py-1 rounded bg-yellow-900/30 text-yellow-400 animate-pulse">
                        Stopping...
                      </span>
                    )}
                    {isActive && !isStopping && (
                      <button
                        onClick={() => stopJob(job.id)}
                        className="text-xs px-2 py-1 rounded bg-red-900/50 hover:bg-red-800 text-red-300 transition-colors"
                      >
                        Stop
                      </button>
                    )}
                    {job.status === "completed" && (!job.mergeStatus || job.mergeStatus === "failed") && (
                      <button
                        onClick={() => mergeJob(job.id)}
                        className="text-xs px-2 py-1 rounded bg-blue-900/50 hover:bg-blue-800 text-blue-300 transition-colors"
                      >
                        Merge Model
                      </button>
                    )}
                    {job.mergeStatus === "running" && (
                      <span className="text-xs px-2 py-1 rounded bg-blue-900/30 text-blue-400 animate-pulse">
                        Merging...
                      </span>
                    )}
                    {job.mergeStatus === "completed" && (
                      <button
                        onClick={() => deployJob(job)}
                        className="text-xs px-2 py-1 rounded bg-green-900/50 hover:bg-green-800 text-green-300 transition-colors"
                      >
                        Deploy
                      </button>
                    )}
                    {!isActive && !isStopping && (
                      <button
                        onClick={() => deleteJob(job.id)}
                        className="text-xs px-2 py-1 rounded bg-red-900/50 hover:bg-red-800 text-red-300 transition-colors"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>

                {/* Phase-aware progress */}
                {isActive && phaseInfo && (
                  <div className="mt-3 space-y-1.5">
                    <div className="flex items-center justify-between text-[10px]">
                      <span className="text-gray-400 flex items-center gap-1.5">
                        <span className={`inline-block w-1.5 h-1.5 rounded-full ${phaseColors[phaseInfo.phase] || "bg-gray-500"} animate-pulse`} />
                        {phaseLabels[phaseInfo.phase] || phaseInfo.phase}
                        {phaseInfo.phase === "training" && phaseInfo.step != null && phaseInfo.totalSteps != null && (
                          <span className="text-gray-600">step {phaseInfo.step}/{phaseInfo.totalSteps}</span>
                        )}
                        {phaseInfo.loss != null && (
                          <span className="text-gray-600">loss: {phaseInfo.loss.toFixed(2)}</span>
                        )}
                      </span>
                      <span className="text-gray-500">
                        {phaseInfo.etaSeconds ? `${formatEta(phaseInfo.etaSeconds)} ` : ""}
                        {phaseInfo.phaseProgress > 0 ? `${Math.round(phaseInfo.phaseProgress * 100)}%` : ""}
                      </span>
                    </div>
                    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${phaseColors[phaseInfo.phase] || "bg-gray-500"}`}
                        style={{ width: `${Math.max(phaseInfo.phaseProgress * 100, phaseInfo.phase === "training" ? 2 : 0)}%` }}
                      />
                    </div>
                  </div>
                )}
                {isActive && !phaseInfo && (
                  <div className="mt-3">
                    <div className="flex items-center text-[10px] text-gray-500">
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-500 animate-pulse mr-1.5" />
                      Waiting for progress...
                    </div>
                  </div>
                )}

                {/* Loss curve chart */}
                {(() => {
                  const m = jobMetrics[job.id];
                  if (!m || m.losses.length < 2) return null;
                  const maxLoss = Math.max(...m.losses);
                  const minLoss = Math.min(...m.losses);
                  const range = maxLoss - minLoss || 1;
                  const w = 300, h = 60, pad = 2;
                  const gw = w - pad * 2, gh = h - pad * 2;
                  const points = m.losses.map((v, i) => {
                    const x = pad + (i / (m.losses.length - 1)) * gw;
                    const y = pad + gh - ((v - minLoss) / range) * gh;
                    return `${x},${y}`;
                  });
                  const linePath = `M${points.join("L")}`;
                  const areaPath = `${linePath}L${pad + gw},${pad + gh}L${pad},${pad + gh}Z`;
                  const lastLoss = m.losses[m.losses.length - 1];
                  return (
                    <div className="mt-3">
                      <div className="flex justify-between items-baseline mb-0.5">
                        <span className="text-[10px] text-gray-500 uppercase tracking-wide">Loss Curve</span>
                        <span className="text-xs font-mono text-gray-300">
                          {lastLoss.toFixed(2)}
                          <span className="text-gray-600 ml-1">({m.losses.length} points)</span>
                        </span>
                      </div>
                      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="w-full rounded" style={{ height: `${h}px` }}>
                        <rect x={0} y={0} width={w} height={h} rx={4} fill="rgba(0,0,0,0.2)" />
                        <path d={areaPath} fill="#22c55e" opacity={0.12} />
                        <path d={linePath} fill="none" stroke="#22c55e" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
                      </svg>
                    </div>
                  );
                })()}

                {/* Output info for completed jobs */}
                {job.status === "completed" && (
                  <div className="mt-2 text-xs space-y-0.5">
                    {job.outputPath && (
                      <p className="text-green-400">Adapter: {job.outputPath}</p>
                    )}
                    {job.mergeStatus === "completed" && job.mergedPath && (
                      <p className="text-blue-400">Merged: {job.mergedPath}</p>
                    )}
                    {job.mergeStatus === "failed" && (
                      <p className="text-red-400">Merge failed</p>
                    )}
                  </div>
                )}

                {/* Log viewer */}
                {viewingLogs === job.id && (
                  <pre
                    data-log-viewer
                    className="mt-3 bg-black/50 border border-gray-800 rounded p-3 text-xs text-gray-400 font-mono max-h-64 overflow-y-auto whitespace-pre-wrap"
                  >
                    {logs[job.id] || job.logs || "Waiting for logs..."}
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      )}

      {jobs.length === 0 && recipes.length > 0 && (
        <div className="text-center py-12 text-gray-500">
          <p>No fine-tuning jobs yet. Select a recipe, node, and dataset above to start.</p>
        </div>
      )}
    </div>
  );
}
