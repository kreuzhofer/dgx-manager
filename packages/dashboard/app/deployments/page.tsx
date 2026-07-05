"use client";

import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { toast } from "sonner";
import { apiFetch } from "@/lib/api";
import { useSSE, type SseEvent } from "@/lib/sse";
import { LogViewer } from "@/components/log-viewer";
import { BenchmarkFormModal } from "@/components/benchmark-form-modal";

interface Recipe {
  file: string;
  name: string;
  description?: string;
  model?: string;
  container: string;
  cluster_only?: boolean;
  solo_only?: boolean;
  defaults: Record<string, unknown>;
  // Target CPU arch the recipe runs on; "any" = arch-agnostic (e.g. Ollama).
  arch?: "amd64" | "arm64" | "any";
  // Present ("dgxrun") for multi-node recipes from the dgxrun catalog;
  // absent for hand-curated sparkrun recipes. Drives dropdown grouping.
  source?: "sparkrun" | "dgxrun";
  // Training recipes carry a separate `deploy:` block describing inference
  // defaults (max_model_len, gpu_memory_utilization, …). Optional because
  // hand-curated vLLM recipes don't have one. Used by the fine-tune Deploy
  // form to surface effective defaults from recipe.yaml as input placeholders.
  deploy?: Record<string, unknown>;
  /** One entry per `inference*.yaml` in the recipe dir. Surfaced only on
   *  training recipes; absent / undefined for plain vLLM serve recipes. */
  inferenceVariants?: {
    id: string;
    filename: string;
    name: string;
    description?: string;
  }[];
}

interface Node {
  id: string;
  name: string;
  ipAddress: string;
  status: string;
  arch?: "amd64" | "arm64";
  vramTotal?: number;
  metrics?: { vramUsed: number }[];
}

interface ClusterNodeInfo {
  id: string;
  nodeId: string;
  role: string;
  status: string;
  node: { id?: string; name: string; ipAddress: string };
}

interface Deployment {
  id: string;
  nodeId: string;
  modelId: string;
  status: string;
  port: number | null;
  config: string | null;
  clusterMode: boolean;
  vramEstimate: number | null;
  vramActual: number | null;
  createdAt: string;
  displayName?: string | null;
  node?: { name: string; ipAddress: string };
  // finetuneJobId on the Model row identifies fine-tune deployments; the
  // nested finetuneJob.recipeFile lets the edit-restart form look up
  // training-recipe defaults for its placeholders.
  model?: {
    name: string;
    finetuneJobId?: string | null;
    finetuneJob?: { recipeFile: string | null } | null;
  };
  clusterNodes?: ClusterNodeInfo[];
}

/**
 * Append `chunk` to `existing` while honoring `\r` carriage returns: a `\r`
 * rewinds back to the start of the current (last) line and the following
 * content overwrites it. Used to collapse tqdm-style progress updates into a
 * single in-place updating line in the log viewer.
 */
function applyCarriageReturns(existing: string, chunk: string): string {
  if (!chunk.includes("\r")) return existing + chunk;
  let buf = existing;
  // Process character-by-character is wasteful — instead split on \r and
  // apply rewind+overwrite per segment.
  const parts = chunk.split("\r");
  // First part is appended (no rewind before the first \r)
  buf += parts[0];
  for (let i = 1; i < parts.length; i++) {
    // Rewind to the start of the current line
    const lastNl = buf.lastIndexOf("\n");
    buf = lastNl === -1 ? "" : buf.slice(0, lastNl + 1);
    buf += parts[i];
  }
  return buf;
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
  evicted: "bg-yellow-900 text-yellow-300",
  failed: "bg-red-900 text-red-300",
  removing: "bg-yellow-900 text-yellow-300",
  restarting: "bg-blue-900 text-blue-300",
};

// Deploy-form platform selector: maps each CPU arch to a human label so the
// user picks "DGX Spark" / "RTX 5090" rather than arch strings.
const PLATFORMS: { arch: "arm64" | "amd64"; label: string }[] = [
  { arch: "arm64", label: "DGX Spark" },
  { arch: "amd64", label: "RTX 5090" },
];

export default function DeploymentsPage() {
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  // Training recipes are a separate endpoint from /api/recipes — they carry
  // a `deploy:` block whose fields (gpu_memory_utilization, max_model_len,
  // tensor_parallel, …) are the source of truth for fine-tune deployment
  // defaults. Used by the edit-restart form to populate placeholders.
  const [trainingRecipes, setTrainingRecipes] = useState<Recipe[]>([]);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [loading, setLoading] = useState(true);

  // Fine-tune deploy params (from URL query)
  const [finetuneModel, setFinetuneModel] = useState<string | null>(null);
  const [finetuneJobId, setFinetuneJobId] = useState<string | null>(null);
  const [finetuneBaseModel, setFinetuneBaseModel] = useState<string | null>(null);
  const [finetuneDisplayName, setFinetuneDisplayName] = useState<string | null>(null);
  const [finetuneArtifactVariant, setFinetuneArtifactVariant] = useState<string | null>(null);

  // Deploy form state
  const [runtimeMode, setRuntimeMode] = useState<"vllm" | "ollama" | "finetune">("vllm");
  const [selectedRecipe, setSelectedRecipe] = useState<string>("");
  const [ollamaModels, setOllamaModels] = useState<{ name: string; size: string; type?: string; description: string }[]>([]);
  const [selectedOllamaModel, setSelectedOllamaModel] = useState<string>("");
  const [idleNodes, setIdleNodes] = useState<Node[]>([]);
  const [selectedNode, setSelectedNode] = useState<string>("");
  // Platform-first UX: pick the target arch, then the recipe list is filtered
  // to recipes that run on it (plus arch-agnostic "any" recipes).
  const [selectedPlatform, setSelectedPlatform] = useState<string>("");
  // For cluster (TP>1) deploys: explicit per-node selection. Defaults to the
  // requiredNodes nodes with the most free VRAM (or alphabetical when no
  // metrics are available); user can override by toggling checkboxes.
  // Sending the explicit list to the server bypasses the "first N alphabetical"
  // auto-pick that ignores VRAM availability and forces the user back onto
  // already-busy nodes.
  const [selectedClusterNodes, setSelectedClusterNodes] = useState<Set<string>>(new Set());
  const [port, setPort] = useState("8000");
  const [maxModelLen, setMaxModelLen] = useState("");
  const [tensorParallel, setTensorParallel] = useState("");
  const [pipelineParallel, setPipelineParallel] = useState("");
  const [gpuMem, setGpuMem] = useState("");
  const [customDisplayName, setCustomDisplayName] = useState<string>("");
  const [deploying, setDeploying] = useState(false);

  // Per-deployment edit-before-restart form state. When a deployment ID is
  // present in editingRestart, the row expands to show editable settings
  // pre-filled from the deployment's saved config; clicking "Restart with
  // these settings" posts the overrides to /restart.
  const [editingRestart, setEditingRestart] = useState<Record<string, {
    port?: string;
    maxModelLen?: string;
    tensorParallel?: string;
    pipelineParallel?: string;
    gpuMem?: string;
    artifactVariant?: string;
  }>>({});

  // Log viewer
  const [viewingLogs, setViewingLogs] = useState<string | null>(null);
  const [logs, setLogs] = useState<Record<string, string>>({});
  // Per-deployment in-flight progress (e.g. HF download). Cleared on phase
  // change away from the matching phase.
  const [progress, setProgress] = useState<Record<string, {
    phase: string;
    phaseProgress: number;
    current?: number;
    total?: number;
    eta?: string;
  }>>({});

  // Benchmark modal state
  const [benchmarkTarget, setBenchmarkTarget] = useState<
    { id: string; label: string } | null
  >(null);
  const [latestBenchmarkStatus, setLatestBenchmarkStatus] = useState<
    Record<string, { status: string; runId: string }>
  >({});

  const loadData = useCallback(() => {
    Promise.all([
      apiFetch<Recipe[]>("/api/recipes"),
      apiFetch<Node[]>("/api/nodes"),
      apiFetch<Deployment[]>("/api/deployments"),
      apiFetch<Node[]>("/api/nodes/idle"),
      apiFetch<{ tag: string; modelName: string; size: string | null; type: "chat" | "embedding"; description: string; capabilities: string[] }[]>("/api/ollama-catalog/available"),
      apiFetch<Recipe[]>("/api/training-recipes"),
    ])
      .then(([r, n, d, idle, om, tr]) => {
        setRecipes(r);
        setNodes(n);
        setDeployments(d);
        setIdleNodes(idle);
        if (idle.length > 0 && !selectedNode) setSelectedNode(idle[0].id);
        setOllamaModels(om.map((m) => ({ name: m.tag, size: m.size ?? "", type: m.type, description: m.description })));
        setTrainingRecipes(tr);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // Parse finetune deploy params from URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const fm = params.get("finetuneModel");
    const fj = params.get("finetuneJobId");
    const fb = params.get("baseModel");
    if (fm) {
      setFinetuneModel(fm);
      setFinetuneJobId(fj);
      setFinetuneBaseModel(fb);
      setRuntimeMode("finetune");
    }
    const fn = params.get("displayName");
    if (fn) setFinetuneDisplayName(fn);
    const fav = params.get("artifactVariant");
    if (fav === "bf16" || fav === "fp8") setFinetuneArtifactVariant(fav);
  }, []);

  // When in fine-tune mode, look up the job's training recipe and pre-fill the
  // override fields from its `deploy` block. Precedence at deploy time is
  // request body > recipe.yaml `deploy` > inference[-fp8].yaml `defaults` >
  // server default. The dashboard only sees the first two layers (training
  // recipes API exposes the `deploy` block); inference-template defaults live
  // on agent disk and aren't surfaced — fine to skip since recipe.yaml is the
  // dominant override path.
  const [finetuneJobRecipeFile, setFinetuneJobRecipeFile] = useState<string | null>(null);
  useEffect(() => {
    if (runtimeMode !== "finetune" || !finetuneJobId) return;
    let cancelled = false;
    apiFetch<{ recipeFile?: string | null }>(`/api/finetune/${finetuneJobId}`)
      .then((job) => {
        if (cancelled) return;
        if (job.recipeFile) setFinetuneJobRecipeFile(job.recipeFile);
      })
      .catch(() => { /* leave placeholders blank if lookup fails */ });
    return () => { cancelled = true; };
  }, [runtimeMode, finetuneJobId]);

  const finetuneRecipeData = useMemo(
    () => (finetuneJobRecipeFile
      ? recipes.find((r) => r.file === finetuneJobRecipeFile || r.file === `${finetuneJobRecipeFile}/recipe.yaml`) ?? null
      : null),
    [finetuneJobRecipeFile, recipes],
  );

  // First-load pre-fill: when the recipe match resolves, populate the override
  // inputs so the user sees what value will be sent. Only fill fields that are
  // currently empty so user edits aren't clobbered by a late-arriving recipes
  // list update.
  const prefilledFromFinetuneRecipe = useRef(false);
  useEffect(() => {
    if (!finetuneRecipeData?.deploy || prefilledFromFinetuneRecipe.current) return;
    const d = finetuneRecipeData.deploy;
    if (maxModelLen === "" && typeof d.max_model_len === "number") {
      setMaxModelLen(String(d.max_model_len));
    }
    if (gpuMem === "" && typeof d.gpu_memory_utilization === "number") {
      setGpuMem(String(d.gpu_memory_utilization));
    }
    if (tensorParallel === "" && typeof d.tensor_parallel === "number") {
      setTensorParallel(String(d.tensor_parallel));
    }
    prefilledFromFinetuneRecipe.current = true;
  }, [finetuneRecipeData, maxModelLen, gpuMem, tensorParallel]);

  // When the training recipe exposes inference variants, auto-select the
  // only one when there's exactly one. Multi-variant case: leave it to
  // the user — show "default" as the initial selection if it exists, else
  // the first sorted entry from listInferenceVariants.
  useEffect(() => {
    if (runtimeMode !== "finetune") return;
    if (finetuneArtifactVariant) return; // user already chose
    const vs = finetuneRecipeData?.inferenceVariants ?? [];
    if (vs.length === 0) return;
    const auto = vs.length === 1 ? vs[0].id : (vs.find((v) => v.id === "default")?.id ?? vs[0].id);
    setFinetuneArtifactVariant(auto);
  }, [runtimeMode, finetuneRecipeData, finetuneArtifactVariant]);

  // SSE handler for real-time updates
  const handleSSE = useCallback((event: SseEvent) => {
    if (event.type === "deployment:status") {
      const { deploymentId, status, port, error, vramActual } = event.payload as {
        deploymentId: string; status: string; port?: number; error?: string; vramActual?: number;
      };
      setDeployments((prev) =>
        prev.map((d) =>
          d.id === deploymentId
            ? { ...d, status, port: port ?? d.port, vramActual: vramActual ?? d.vramActual }
            : d
        )
      );
      // Drop stale progress when the phase changes away from the one we were
      // tracking — otherwise a 99% download bar lingers across the loading phase.
      setProgress((prev) => {
        const cur = prev[deploymentId];
        if (cur && cur.phase !== status) {
          const { [deploymentId]: _, ...rest } = prev;
          return rest;
        }
        return prev;
      });
      if (error) {
        setLogs((prev) => ({
          ...prev,
          [deploymentId]: (prev[deploymentId] || "") + `\n[ERROR] ${error}`,
        }));
      }
      // Refresh idle nodes only on terminal state changes
      // (starting/running are handled by optimistic update in deploy())
      if (["stopped", "failed"].includes(status)) {
        apiFetch<Node[]>("/api/nodes/idle").then(setIdleNodes).catch(() => {});
        apiFetch<Node[]>("/api/nodes").then(setNodes).catch(() => {});
      }
    }
    if (event.type === "deployment:log") {
      const { deploymentId, log } = event.payload as { deploymentId: string; log: string };
      setLogs((prev) => {
        // Honor `\r` carriage returns: rewind to start of current line and
        // overwrite. The agent already collapses within a single chunk, but
        // the very first \r in a chunk may follow content from a prior chunk,
        // so we apply the same rewind here.
        const existing = prev[deploymentId] || "";
        const next = applyCarriageReturns(existing, log);
        return { ...prev, [deploymentId]: next };
      });
    }
    if (event.type === "deployment:progress") {
      const p = event.payload as {
        deploymentId: string; phase: string; phaseProgress: number;
        current?: number; total?: number; eta?: string;
      };
      setProgress((prev) => ({
        ...prev,
        [p.deploymentId]: {
          phase: p.phase, phaseProgress: p.phaseProgress,
          current: p.current, total: p.total, eta: p.eta,
        },
      }));
    }
    if (event.type === "deployment:created") {
      const dep = event.payload as unknown as Deployment;
      if (dep?.id) {
        setDeployments((prev) => prev.some((d) => d.id === dep.id) ? prev : [dep, ...prev]);
        apiFetch<Node[]>("/api/nodes/idle").then(setIdleNodes).catch(() => {});
      }
    }
    if (event.type === "deployment:deleted") {
      const { deploymentId } = event.payload as { deploymentId: string };
      setDeployments((prev) => prev.filter((d) => d.id !== deploymentId));
      apiFetch<Node[]>("/api/nodes/idle").then(setIdleNodes).catch(() => {});
      apiFetch<Node[]>("/api/nodes").then(setNodes).catch(() => {});
    }
    if (event.type === "node:status") {
      const { nodeId, status } = event.payload as { nodeId: string; status: string };
      setNodes((prev) => prev.map((n) => n.id === nodeId ? { ...n, status } : n));
    }
    if (event.type === "node:metrics") {
      const { nodeId, vramUsed } = event.payload as unknown as { nodeId: string; vramUsed: number };
      if (vramUsed !== undefined) {
        setNodes((prev) => prev.map((n) =>
          n.id === nodeId ? { ...n, metrics: [{ vramUsed }] } : n
        ));
      }
    }
    if (event.type === "ollama-catalog:updated") {
      apiFetch<{ tag: string; modelName: string; size: string | null; type: "chat" | "embedding"; description: string; capabilities: string[] }[]>("/api/ollama-catalog/available")
        .then((om) => setOllamaModels(om.map((m) => ({ name: m.tag, size: m.size ?? "", type: m.type, description: m.description }))))
        .catch(console.error);
    }
    if (event.type === "benchmark:created" || event.type === "benchmark:status") {
      const payload = event.payload as { id: string; deploymentId?: string | null; status: string };
      if (payload.deploymentId) {
        setLatestBenchmarkStatus((prev) => ({
          ...prev,
          [payload.deploymentId!]: { status: payload.status, runId: payload.id },
        }));
      }
    }
  }, []);

  // Live log streaming (deployment:log over SSE) has no replay — any chunks
  // emitted while the tab is backgrounded or the SSE socket is dropped are
  // lost, leaving a permanent hole in the on-screen log. The server persists
  // the full log to disk, so on resume we re-pull it and REPLACE the buffer
  // (the file is the source of truth) to close the gap.
  const logsRef = useRef<Record<string, string>>({});
  useEffect(() => {
    logsRef.current = logs;
  }, [logs]);

  const refetchOpenLogs = useCallback(() => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";
    for (const id of Object.keys(logsRef.current)) {
      fetch(`${apiBase}/api/deployments/${id}/logs`, { cache: "no-store" })
        .then((r) => (r.ok ? r.text() : ""))
        .then((text) => {
          if (text) setLogs((prev) => ({ ...prev, [id]: text }));
        })
        .catch(() => {});
    }
  }, []);

  // Catch up when the tab returns to the foreground (covers background
  // throttling even when the SSE socket never dropped).
  useEffect(() => {
    const onVis = () => {
      if (document.visibilityState === "visible") refetchOpenLogs();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [refetchOpenLogs]);

  // On SSE reconnect, refresh both the lists (loadData) and the open log
  // buffers so a dropped socket doesn't leave stale logs.
  const onReconnect = useCallback(() => {
    loadData();
    refetchOpenLogs();
  }, [loadData, refetchOpenLogs]);

  const { connected } = useSSE(handleSSE, onReconnect);

  const deploy = async (e: React.FormEvent) => {
    e.preventDefault();
    setDeploying(true);
    try {
      let body: Record<string, unknown>;
      const trimmedDisplayName = customDisplayName.trim();

      if (runtimeMode === "finetune" && finetuneJobId) {
        const config: Record<string, unknown> = { port: parseInt(port) || 8000 };
        if (gpuMem) config.gpuMem = parseFloat(gpuMem);
        if (maxModelLen) config.maxModelLen = parseInt(maxModelLen);
        if (tensorParallel) config.tensorParallel = parseInt(tensorParallel);
        if (pipelineParallel) config.pipelineParallel = parseInt(pipelineParallel);

        // Cluster vs solo: when TP * PP > 1, send the explicit node list
        // selectedClusterNodes already holds (the cluster picker is reused).
        const tp = parseInt(tensorParallel) || 1;
        const pp = parseInt(pipelineParallel) || 1;
        const needsClusterFt = tp * pp > 1;
        if (needsClusterFt && selectedClusterNodes.size !== tp * pp) return;
        if (!needsClusterFt && !selectedNode) return;

        const ftBody: Record<string, unknown> = { config };
        if (needsClusterFt) ftBody.nodeIds = Array.from(selectedClusterNodes);
        else ftBody.nodeId = selectedNode;
        if (finetuneArtifactVariant) ftBody.artifactVariant = finetuneArtifactVariant;
        if (trimmedDisplayName) ftBody.displayName = trimmedDisplayName;

        const result = await apiFetch<Deployment>(`/api/finetune/${finetuneJobId}/deploy`, {
          method: "POST",
          body: JSON.stringify(ftBody),
        });
        setDeployments((prev) => prev.some((d) => d.id === result.id) ? prev : [result, ...prev]);
        toast.success(`Deployed ${result.displayName ?? result.model?.name ?? "fine-tuned model"}`, {
          description: needsClusterFt
            ? `${tp * pp} nodes, head=${result.node?.name ?? "?"}`
            : `on ${result.node?.name ?? "?"}`,
        });
        setRuntimeMode("vllm");
        setFinetuneModel(null);
        setFinetuneJobId(null);
        setFinetuneDisplayName(null);
        setFinetuneArtifactVariant(null);
        // Same reset the vLLM branch does after submit — without this the
        // launch dialog comes back pre-populated with the prior session's
        // node picks + TP/PP/memory overrides, which surprised users.
        setSelectedNode("");
        setSelectedClusterNodes(new Set());
        setTensorParallel("");
        setPipelineParallel("");
        setMaxModelLen("");
        setGpuMem("");
        setPort("8000");
        setCustomDisplayName("");
        // Clear URL params
        window.history.replaceState({}, "", "/deployments");
        setDeploying(false);
        return;
      } else if (runtimeMode === "ollama") {
        if (!selectedOllamaModel || !selectedNode) return;
        const selectedModel = ollamaModels.find((m) => m.name === selectedOllamaModel);
        body = {
          runtime: "ollama",
          modelName: selectedOllamaModel,
          modelType: selectedModel?.type || "chat",
          nodeId: selectedNode,
          config: {},
        };
      } else {
        if (!selectedRecipe || !canDeploy) return;
        const configOverrides: Record<string, unknown> = {
          port: parseInt(port) || 8000,
        };
        if (maxModelLen) configOverrides.maxModelLen = parseInt(maxModelLen);
        if (tensorParallel) configOverrides.tensorParallel = parseInt(tensorParallel);
        if (pipelineParallel) configOverrides.pipelineParallel = parseInt(pipelineParallel);
        if (gpuMem) configOverrides.gpuMem = parseFloat(gpuMem);

        body = needsCluster
          ? {
              // Explicit list lets the user override which N nodes to use
              // (the dashboard pre-picks by free VRAM; the user can toggle).
              nodeIds: Array.from(selectedClusterNodes),
              recipeFile: selectedRecipe,
              config: configOverrides,
              ...(trimmedDisplayName ? { displayName: trimmedDisplayName } : {}),
            }
          : {
              nodeId: selectedNode || "auto",
              recipeFile: selectedRecipe,
              config: configOverrides,
              ...(trimmedDisplayName ? { displayName: trimmedDisplayName } : {}),
            };
      }

      const deployment = await apiFetch<Deployment>("/api/deployments", {
        method: "POST",
        body: JSON.stringify(body),
      });
      setDeployments((prev) => prev.some((d) => d.id === deployment.id) ? prev : [deployment, ...prev]);
      toast.success(`Deployed ${deployment.displayName ?? deployment.model?.name ?? "model"}`, {
        description: deployment.clusterMode
          ? `${deployment.clusterNodes?.length ?? "?"} nodes, head=${deployment.node?.name ?? "?"}`
          : `on ${deployment.node?.name ?? "?"}`,
      });
      setSelectedRecipe("");
      setSelectedOllamaModel("");
      setTensorParallel("");
      setPipelineParallel("");
      setMaxModelLen("");
      setGpuMem("");
      setPort("8000");
      setSelectedNode("");
      setSelectedClusterNodes(new Set());
      setCustomDisplayName("");
      setViewingLogs(deployment.id);
      // Immediately remove consumed nodes from idle list
      const usedIds = new Set<string>([deployment.nodeId]);
      for (const cn of deployment.clusterNodes || []) {
        usedIds.add(cn.nodeId);
      }
      setIdleNodes((prev) => prev.filter((n) => !usedIds.has(n.id)));
    } catch (err) {
      toast.error("Deploy failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setDeploying(false);
    }
  };

  const stopDeployment = async (id: string) => {
    const d = deployments.find((x) => x.id === id);
    const label = d ? `${d.displayName ?? d.model?.name ?? d.modelId} on ${d.node?.name || d.nodeId}` : id.slice(0, 12);
    if (!confirm(`Stop this deployment?\n\n${label}\n\nInference will be unavailable until it's redeployed.`)) return;
    await apiFetch(`/api/deployments/${id}`, { method: "DELETE" });
    setDeployments((prev) =>
      prev.map((d) => (d.id === id ? { ...d, status: "stopping" } : d))
    );
  };

  const restartDeployment = async (id: string, overrides?: Record<string, unknown>) => {
    try {
      await apiFetch(`/api/deployments/${id}/restart`, {
        method: "POST",
        body: overrides && Object.keys(overrides).length > 0 ? JSON.stringify({ config: overrides }) : undefined,
      });
      setDeployments((prev) =>
        prev.map((d) => (d.id === id ? { ...d, status: "restarting" } : d))
      );
      // Collapse the inline editor on success
      setEditingRestart((prev) => {
        const { [id]: _, ...rest } = prev;
        return rest;
      });
    } catch (err) {
      // Surface the error so it's visible on mobile too — apiFetch throws on
      // non-2xx (e.g. 409 from the VRAM admission check) and without this the
      // click looks silently broken.
      toast.error("Restart failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    }
  };

  // Resolves the recipe defaults to seed the edit-restart form for a given
  // deployment. Fine-tune deployments draw from the linked training
  // recipe's `deploy` block (same source the initial finetune deploy used);
  // stock vLLM deployments draw from the matching serve recipe's defaults.
  // Either returns {} when the recipe isn't (no longer) registered.
  const recipeDefaultsForDeployment = (
    d: Deployment | undefined,
    cfg: Record<string, unknown>,
  ): Record<string, unknown> => {
    const ftRecipeFile = d?.model?.finetuneJob?.recipeFile;
    if (d?.model?.finetuneJobId && ftRecipeFile) {
      return (trainingRecipes.find((r) => r.file === ftRecipeFile)?.deploy ?? {}) as Record<string, unknown>;
    }
    return (recipes.find((r) => r.file === cfg.recipeFile)?.defaults ?? {}) as Record<string, unknown>;
  };

  const beginEditRestart = (id: string) => {
    const d = deployments.find((x) => x.id === id);
    let cfg: Record<string, unknown> = {};
    try { cfg = d?.config ? JSON.parse(d.config) : {}; } catch { /* */ }
    // Fall back to recipe defaults for fields the user didn't explicitly set
    // at initial deploy — without this the dialog shows a blank GPU-mem (or
    // TP/PP/maxLen) box even though the deployment was running with a real
    // value pulled from recipe.defaults.
    const recipeDefaults = recipeDefaultsForDeployment(d, cfg);
    const pick = (cfgVal: unknown, defVal: unknown): string => {
      if (cfgVal != null) return String(cfgVal);
      if (defVal != null) return String(defVal);
      return "";
    };
    setEditingRestart((prev) => ({
      ...prev,
      [id]: {
        port: pick(cfg.port, recipeDefaults.port),
        maxModelLen: pick(cfg.maxModelLen, recipeDefaults.max_model_len),
        tensorParallel: pick(cfg.tensorParallel, recipeDefaults.tensor_parallel),
        pipelineParallel: pick(cfg.pipelineParallel, recipeDefaults.pipeline_parallel),
        gpuMem: pick(cfg.gpuMem, recipeDefaults.gpu_memory_utilization),
        artifactVariant: typeof cfg.artifactVariant === "string" ? cfg.artifactVariant : "default",
      },
    }));
  };

  const cancelEditRestart = (id: string) => {
    setEditingRestart((prev) => {
      const { [id]: _, ...rest } = prev;
      return rest;
    });
  };

  const submitEditRestart = (id: string) => {
    const fields = editingRestart[id] || {};
    const overrides: Record<string, unknown> = {};
    if (fields.port) overrides.port = parseInt(fields.port);
    if (fields.maxModelLen) overrides.maxModelLen = parseInt(fields.maxModelLen);
    if (fields.tensorParallel) overrides.tensorParallel = parseInt(fields.tensorParallel);
    if (fields.pipelineParallel) overrides.pipelineParallel = parseInt(fields.pipelineParallel);
    if (fields.gpuMem) overrides.gpuMem = parseFloat(fields.gpuMem);
    if (fields.artifactVariant) overrides.artifactVariant = fields.artifactVariant;
    return restartDeployment(id, overrides);
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

  // Recipes eligible for the vLLM recipe dropdown, after the existing
  // finetune-artifact and platform filters. Split by `source` so the
  // dropdown can group dgxrun (multi-node) recipes above sparkrun ones.
  const visibleRecipes = useMemo(() => {
    return recipes
      // Auto-generated fine-tune recipes (finetune-<12hex>) are internal
      // artifacts of the /finetune → Deploy flow, not hand-curated recipes
      // for direct selection. Hide them from the dropdown so the list stays
      // clean. Hand-named ones (e.g. finetune-qwen3.6-50step) don't match
      // the pattern and stay visible.
      .filter((r) => !/^recipes\/finetune-[a-z0-9]{12}\.yaml$/.test(r.file))
      // Platform filter: missing arch is treated as arm64; "any" recipes
      // show everywhere; show all if no platform picked.
      .filter((r) => { const a = r.arch ?? "arm64"; return a === "any" || a === selectedPlatform || !selectedPlatform; });
  }, [recipes, selectedPlatform]);
  const dgxrunRecipes = useMemo(() => visibleRecipes.filter((r) => r.source === "dgxrun"), [visibleRecipes]);
  const sparkRecipes = useMemo(() => visibleRecipes.filter((r) => r.source !== "dgxrun"), [visibleRecipes]);
  const renderRecipeOption = (r: Recipe) => {
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
  };

  // Suggest a sensible default for the deploy's Display name input: stock
  // vLLM uses the recipe's `name` field; fine-tune deploys use the FT's
  // friendly name (or Model.name fallback). Sanitized to the same character
  // class the server enforces ([A-Za-z0-9._:-]) so the prefill validates as-is.
  const defaultDisplayName = useMemo(() => {
    const sanitize = (raw: string) =>
      raw.trim().replace(/\s+/g, "-").replace(/[^A-Za-z0-9._:-]/g, "").slice(0, 128);
    if (runtimeMode === "vllm") {
      return selectedRecipeData?.name ? sanitize(selectedRecipeData.name) : "";
    }
    if (runtimeMode === "finetune") {
      const src = finetuneDisplayName ?? finetuneModel ?? "";
      return src ? sanitize(src) : "";
    }
    return "";
  }, [runtimeMode, selectedRecipeData, finetuneDisplayName, finetuneModel]);

  // Mirror the suggested default into the input whenever the source changes.
  // User edits stay (defaultDisplayName doesn't recompute while the same
  // recipe/FT is selected); switching to a different recipe resets to the new
  // default — picking a different model is a strong signal the old name no
  // longer applies.
  useEffect(() => {
    setCustomDisplayName(defaultDisplayName);
  }, [defaultDisplayName]);

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

  // Platforms we actually have a node for — only offer these in the selector.
  const availablePlatforms = useMemo(
    () => PLATFORMS.filter((p) => nodes.some((n) => n.arch === p.arch)),
    [nodes],
  );

  // Default the platform once nodes load so the controlled select has a value.
  useEffect(() => {
    if (!selectedPlatform && availablePlatforms.length > 0) {
      setSelectedPlatform(availablePlatforms[0].arch);
    }
  }, [availablePlatforms, selectedPlatform]);

  // Compute auto-selected nodes
  // Compute required nodes from config overrides or recipe defaults
  const effectiveTP = parseInt(tensorParallel) || (selectedRecipeData?.defaults?.tensor_parallel as number) || 1;
  const effectivePP = parseInt(pipelineParallel) || (selectedRecipeData?.defaults?.pipeline_parallel as number) || 1;
  const requiredNodes = effectiveTP * effectivePP;
  const needsCluster = requiredNodes > 1;

  // Per-node free VRAM (MB), computed client-side from the latest metrics
  // we already keep in `nodes`. Used to sort the cluster picker so the
  // most-free nodes are pre-selected by default.
  const nodeFreeMB = (n: Node): number => {
    const total = n.vramTotal ?? 0;
    const used = n.metrics?.[0]?.vramUsed ?? 0;
    return Math.max(0, total - used);
  };
  // Online nodes available for the cluster picker. Sorted by free VRAM
  // (most free first) so the default selection picks the nodes least
  // likely to clash with running deploys.
  const clusterCandidates = nodes
    .filter((n) => n.status === "online")
    .slice()
    .sort((a, b) => nodeFreeMB(b) - nodeFreeMB(a));

  // Initialize the cluster selection when:
  //   - the recipe changes (different requiredNodes),
  //   - tp/pp overrides change requiredNodes,
  //   - the current selection becomes stale (size mismatch, or includes
  //     a node that's no longer in the candidate list).
  // Pre-pick the top requiredNodes by free VRAM. The user can override
  // by toggling checkboxes.
  useEffect(() => {
    if (!needsCluster) {
      if (selectedClusterNodes.size > 0) setSelectedClusterNodes(new Set());
      return;
    }
    const candidateIds = new Set(clusterCandidates.map((n) => n.id));
    const currentValid = Array.from(selectedClusterNodes).filter((id) => candidateIds.has(id));
    if (currentValid.length === requiredNodes) return; // user's selection is fine
    // Auto-pick top N
    const next = new Set(clusterCandidates.slice(0, requiredNodes).map((n) => n.id));
    setSelectedClusterNodes(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [needsCluster, requiredNodes, selectedRecipe, nodes.length]);

  const canDeploy = needsCluster
    ? selectedClusterNodes.size === requiredNodes
    : !!selectedNode || idleNodes.length >= 1;

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
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
            New Deployment
          </h2>
          <div className="flex bg-gray-800 rounded p-0.5">
            <button
              type="button"
              onClick={() => { setRuntimeMode("vllm"); setSelectedOllamaModel(""); setCustomDisplayName(""); }}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${runtimeMode === "vllm" ? "bg-green-600 text-white" : "text-gray-400 hover:text-white"}`}
            >
              vLLM
            </button>
            <button
              type="button"
              onClick={() => { setRuntimeMode("ollama"); setSelectedRecipe(""); setCustomDisplayName(""); }}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${runtimeMode === "ollama" ? "bg-green-600 text-white" : "text-gray-400 hover:text-white"}`}
            >
              Ollama
            </button>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div className="md:col-span-2">
            {runtimeMode === "finetune" && finetuneModel ? (
              <div>
                <label className="block text-xs text-gray-400 mb-1">Fine-tuned Model</label>
                <div className="bg-gray-800 border border-purple-700 rounded px-3 py-2 text-sm text-purple-300">
                  <div className="font-medium">
                    {finetuneDisplayName || finetuneBaseModel || "Fine-tuned model"}
                  </div>
                  {finetuneDisplayName && finetuneBaseModel && (
                    <div className="text-[10px] text-gray-500 mt-0.5">
                      base: {finetuneBaseModel}
                    </div>
                  )}
                  <div className="text-[10px] text-gray-500 mt-0.5 truncate">{finetuneModel}</div>
                </div>
              </div>
            ) : runtimeMode === "vllm" ? (
              <>
                {availablePlatforms.length > 0 && (
                  <div className="mb-3">
                    <label className="block text-xs text-gray-400 mb-1">Platform</label>
                    <select
                      value={selectedPlatform}
                      onChange={(e) => {
                        const next = e.target.value;
                        setSelectedPlatform(next);
                        // Clear the recipe if it no longer fits the new platform.
                        const current = recipes.find((r) => r.file === selectedRecipe);
                        if (current) {
                          const a = current.arch ?? "arm64";
                          if (a !== "any" && a !== next) onRecipeChange("");
                        }
                      }}
                      className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                    >
                      {availablePlatforms.map((p) => (
                        <option key={p.arch} value={p.arch}>{p.label}</option>
                      ))}
                    </select>
                  </div>
                )}
                <label className="block text-xs text-gray-400 mb-1">Recipe</label>
                <select
                  value={selectedRecipe}
                  onChange={(e) => onRecipeChange(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                >
                  <option value="">Select a recipe...</option>
                  {dgxrunRecipes.length > 0 && (
                    <optgroup label="dgxrun (multi-node)">
                      {dgxrunRecipes.map(renderRecipeOption)}
                    </optgroup>
                  )}
                  <optgroup label="sparkrun">
                    {sparkRecipes.map(renderRecipeOption)}
                  </optgroup>
                </select>
              </>
            ) : (
              <>
                <label className="block text-xs text-gray-400 mb-1">Model</label>
                <select
                  value={selectedOllamaModel}
                  onChange={(e) => setSelectedOllamaModel(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
                >
                  <option value="">Select a model...</option>
                  {ollamaModels.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}{m.size ? ` (${m.size})` : ""} [{m.type}] — {m.description}
                    </option>
                  ))}
                </select>
              </>
            )}
          </div>
          <div className="md:col-span-2">
            <label className="block text-xs text-gray-400 mb-1">
              {runtimeMode === "ollama" ? "Node" : needsCluster ? `Target Nodes (${requiredNodes} needed)` : "Node"}
            </label>
            {(runtimeMode === "vllm" || runtimeMode === "finetune") && needsCluster ? (
              <div className="bg-gray-800 border border-gray-700 rounded p-2">
                {clusterCandidates.length < requiredNodes ? (
                  <span className="text-red-400 text-sm px-1">
                    Need {requiredNodes} online nodes; only {clusterCandidates.length} available
                  </span>
                ) : (
                  <>
                    <p className="text-[10px] text-gray-500 mb-1.5 px-1">
                      Pick exactly {requiredNodes} (sorted by free VRAM, most free first).
                      Selected: <span className={selectedClusterNodes.size === requiredNodes ? "text-green-400" : "text-yellow-400"}>{selectedClusterNodes.size}/{requiredNodes}</span>
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-1">
                      {clusterCandidates.map((n) => {
                        const checked = selectedClusterNodes.has(n.id);
                        const freeMB = nodeFreeMB(n);
                        const totalMB = n.vramTotal ?? 0;
                        const freeGB = (freeMB / 1024).toFixed(0);
                        const totalGB = (totalMB / 1024).toFixed(0);
                        const usedPct = totalMB > 0 ? Math.round(((totalMB - freeMB) / totalMB) * 100) : 0;
                        const atLimit = !checked && selectedClusterNodes.size >= requiredNodes;
                        return (
                          <label
                            key={n.id}
                            className={`flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer text-xs border transition-colors ${
                              checked
                                ? "bg-green-900/30 border-green-700 text-green-100"
                                : atLimit
                                ? "bg-gray-900/40 border-gray-800 text-gray-500 cursor-not-allowed"
                                : "bg-gray-900/40 border-gray-800 hover:border-gray-600 text-gray-300"
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={checked}
                              disabled={atLimit}
                              onChange={() => {
                                setSelectedClusterNodes((prev) => {
                                  const next = new Set(prev);
                                  if (next.has(n.id)) next.delete(n.id);
                                  else if (next.size < requiredNodes) next.add(n.id);
                                  return next;
                                });
                              }}
                              className="accent-green-500"
                            />
                            <span className="font-medium">{n.name}</span>
                            <span className="text-gray-500 ml-auto tabular-nums">
                              {freeGB}/{totalGB} GB free
                              {usedPct > 0 && <span className="text-yellow-500 ml-1">({usedPct}% used)</span>}
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            ) : (
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
                    step="any"
                    min="0"
                    value={gpuMem}
                    onChange={(e) => setGpuMem(e.target.value)}
                    placeholder={String(selectedRecipeData.defaults.gpu_memory_utilization ?? "")}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                  />
                  <p className="text-[10px] text-gray-500 mt-0.5">
                    {(() => {
                      const def = selectedRecipeData.defaults.gpu_memory_utilization;
                      if (typeof def === "number" && def > 1) {
                        return `GB (recipe default ${def}); leave blank to use the default`;
                      }
                      return `fraction 0–1 (recipe default ${def ?? "0.85"}); leave blank to use the default`;
                    })()}
                  </p>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Deployment options for finetune mode (port, TP, PP, maxModelLen, gpuMem).
            Placeholders show the effective default sourced from the fine-tune
            job's training recipe `deploy:` block (or generic fallback when the
            recipe lookup hasn't resolved or doesn't carry a value). The same
            effects pre-fill the inputs themselves so the user sees what will
            be sent without having to re-type. Editing wins server-side via
            request body > recipe.yaml > inference template default. */}
        {runtimeMode === "finetune" && (
          <div className="mt-3 p-3 bg-gray-800/50 rounded border border-gray-700/50">
            {finetuneRecipeData && (
              <p className="text-[10px] text-gray-500 mb-2">
                Defaults from{" "}
                <span className="text-gray-400 font-mono">{finetuneRecipeData.file}</span>
                {" "}— edit to override.
              </p>
            )}
            {finetuneRecipeData?.inferenceVariants && finetuneRecipeData.inferenceVariants.length > 0 && (
              <div className="mt-3">
                <label className="block text-xs text-gray-400 mb-1">
                  Inference variant
                  {finetuneRecipeData.inferenceVariants.length === 1 && (
                    <span className="ml-2 text-gray-500">(only one available — auto-selected)</span>
                  )}
                </label>
                <select
                  value={finetuneArtifactVariant ?? ""}
                  onChange={(e) => setFinetuneArtifactVariant(e.target.value || null)}
                  disabled={finetuneRecipeData.inferenceVariants.length === 1}
                  className="w-full bg-gray-950 border border-gray-800 rounded px-2 py-1.5 text-sm focus:outline-none focus:border-gray-600 disabled:opacity-60"
                >
                  {finetuneRecipeData.inferenceVariants.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.id} — {v.name}
                    </option>
                  ))}
                </select>
                {(() => {
                  const sel = finetuneRecipeData!.inferenceVariants!.find((v) => v.id === finetuneArtifactVariant);
                  return sel?.description ? (
                    <p className="mt-1 text-xs text-gray-500">{sel.description}</p>
                  ) : null;
                })()}
              </div>
            )}
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
                  placeholder={String((finetuneRecipeData?.deploy?.tensor_parallel as number | undefined) ?? 1)}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-0.5">Pipeline Parallel</label>
                <input
                  type="number"
                  value={pipelineParallel}
                  onChange={(e) => setPipelineParallel(e.target.value)}
                  placeholder={String((finetuneRecipeData?.deploy?.pipeline_parallel as number | undefined) ?? 1)}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-0.5">Max Model Length</label>
                <input
                  type="number"
                  value={maxModelLen}
                  onChange={(e) => setMaxModelLen(e.target.value)}
                  placeholder={String((finetuneRecipeData?.deploy?.max_model_len as number | undefined) ?? "")}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                />
              </div>
              <div>
                <label className="block text-[10px] text-gray-500 mb-0.5">GPU Memory Util</label>
                <input
                  type="number"
                  step="any"
                  min="0"
                  value={gpuMem}
                  onChange={(e) => setGpuMem(e.target.value)}
                  placeholder={String((finetuneRecipeData?.deploy?.gpu_memory_utilization as number | undefined) ?? 0.85)}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-green-500"
                />
                <p className="text-[10px] text-gray-500 mt-0.5">fraction 0–1; leave blank for default</p>
              </div>
            </div>
          </div>
        )}

        {(runtimeMode === "vllm" || runtimeMode === "finetune") && (
          <div className="mt-3 px-1">
            <label className="block text-[10px] text-gray-500 mb-0.5">
              Display name <span className="text-gray-600">(optional)</span>
            </label>
            <input
              type="text"
              value={customDisplayName}
              onChange={(e) => setCustomDisplayName(e.target.value)}
              placeholder="e.g. chat3d-prod"
              pattern="[A-Za-z0-9._:-]*"
              maxLength={128}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-green-500"
            />
            <p className="text-[10px] text-gray-500 mt-0.5">
              Overrides the model name in this list and in the OpenAI API (<code>/v1/models</code>). Letters, digits, dot, dash, underscore, colon.
            </p>
          </div>
        )}

        <div className="mt-4 flex justify-end">
          <button
            type="submit"
            disabled={deploying || (runtimeMode === "finetune"
              ? (() => {
                  const tp = parseInt(tensorParallel) || 1;
                  const pp = parseInt(pipelineParallel) || 1;
                  const needsClusterFt = tp * pp > 1;
                  return needsClusterFt ? selectedClusterNodes.size !== tp * pp : !selectedNode;
                })()
              : runtimeMode === "vllm" ? (!selectedRecipe || !canDeploy) : (!selectedOllamaModel || !selectedNode))}
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

      {/* Deployments list — grouped by node */}
      {deployments.length > 0 && (() => {
        // Group deployments by node — cluster deployments appear under each node
        const byNode = new Map<string, { deployment: typeof deployments[0]; role: "head" | "worker" | "solo" }[]>();
        for (const d of deployments) {
          // Add under head/solo node
          const nodeKey = d.node?.name || d.nodeId;
          if (!byNode.has(nodeKey)) byNode.set(nodeKey, []);
          byNode.get(nodeKey)!.push({ deployment: d, role: d.clusterMode ? "head" : "solo" });

          // Also add under each worker node
          if (d.clusterMode && d.clusterNodes) {
            for (const cn of d.clusterNodes) {
              if (cn.role === "worker") {
                const workerKey = cn.node.name;
                if (!byNode.has(workerKey)) byNode.set(workerKey, []);
                byNode.get(workerKey)!.push({ deployment: d, role: "worker" });
              }
            }
          }
        }

        return (
        <div className="space-y-4">
          {Array.from(byNode.entries()).sort(([a], [b]) => a.localeCompare(b)).map(([nodeName, nodeDeps]) => {
            const nodeData = nodes.find((n) => n.name === nodeName);
            return (
              <div key={nodeName}>
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-sm font-semibold text-gray-300">{nodeName}</h3>
                  {nodeData && (
                    <>
                      <span className="text-[10px] text-gray-500">{nodeData.ipAddress}</span>
                      {nodeData.vramTotal && (() => {
                        const used = nodeData.metrics?.[0]?.vramUsed || 0;
                        const total = nodeData.vramTotal!;
                        const free = total - used;
                        const pct = Math.round(used / total * 100);
                        return (
                          <span className={`text-[10px] px-1.5 py-0.5 rounded ${pct > 80 ? "bg-red-900/40 text-red-300" : pct > 50 ? "bg-yellow-900/40 text-yellow-300" : "bg-gray-800 text-gray-400"}`}>
                            {Math.round(used / 1024)}GB / {Math.round(total / 1024)}GB ({Math.round(free / 1024)}GB free)
                          </span>
                        );
                      })()}
                    </>
                  )}
                </div>
                <div className="space-y-2">
          {nodeDeps.map(({ deployment: d, role: nodeRole }) => {
            const config = d.config ? JSON.parse(d.config) : {};
            const recipeName = config.recipeFile
              ?.replace(/^recipes\//, "")
              .replace(/\.yaml$/, "");
            const isActive = ["running", "starting", "pending", "restarting", "building", "downloading", "launching", "loading"].includes(d.status);
            const isStopping = ["stopping", "removing"].includes(d.status);
            const isWorker = nodeRole === "worker";
            const isHead = nodeRole === "head";
            // Ollama deployments don't take TP/PP/maxModelLen/gpuMem — those
            // are vLLM-only. Restart goes straight to the API; no edit form.
            const isOllama = config.runtime === "ollama";
            const isFinetune = Boolean(d.model?.finetuneJobId);
            const ftRecipeFile = d.model?.finetuneJob?.recipeFile;
            const ftRecipe = ftRecipeFile
              ? trainingRecipes.find((r) => r.file === ftRecipeFile)
              : undefined;
            const ftVariants = ftRecipe?.inferenceVariants ?? [];
            // Recipe defaults are used as placeholder text in the edit-restart
            // form, so a cleared box still shows what value will actually be
            // used. For fine-tune deployments the defaults come from the
            // training recipe's deploy: block (not the vLLM serve recipes).
            const recipeDefaultsForRestart = recipeDefaultsForDeployment(d, config);

            return (
              <div
                key={`${d.id}-${nodeRole}-${nodeName}`}
                className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors"
              >
                <div className="flex flex-wrap items-center justify-between gap-y-2 gap-x-3">
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="min-w-0">
                      <h3 className="font-semibold flex flex-wrap items-center gap-2">
                        <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                          config.runtime === "ollama"
                            ? "bg-cyan-900/60 text-cyan-300"
                            : "bg-green-900/60 text-green-300"
                        }`}>
                          {config.runtime === "ollama" ? "Ollama" : "vLLM"}
                        </span>
                        {d.displayName ?? d.model?.name ?? recipeName ?? d.modelId}
                        {(d.vramEstimate || d.vramActual) && (
                          <span className="text-[10px] font-normal text-gray-500 ml-1">
                            {d.vramActual
                              ? `${Math.round(d.vramActual / 1024)}GB`
                              : `~${Math.round((d.vramEstimate || 0) / 1024)}GB est`}
                          </span>
                        )}
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
                  <div className="flex flex-wrap items-center gap-2 justify-end">
                    <span
                      className={`text-xs px-2.5 py-1 rounded font-medium ${
                        statusStyles[d.status] || "bg-gray-700 text-gray-300"
                      }`}
                    >
                      {d.status}
                    </span>
                    {isWorker && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-500">worker</span>
                    )}
                    {isHead && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-900/40 text-indigo-300">head</span>
                    )}
                    {!isWorker && (
                      <>
                        <button
                          onClick={() => {
                            if (viewingLogs === d.id) {
                              setViewingLogs(null);
                            } else {
                              setViewingLogs(d.id);
                              if (!logs[d.id]) {
                                const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";
                                fetch(`${apiBase}/api/deployments/${d.id}/logs`, { cache: "no-store" })
                                  .then(r => r.text())
                                  .then(text => {
                                    if (text) setLogs(prev => ({ ...prev, [d.id]: text + (prev[d.id] || "") }));
                                  })
                                  .catch(() => {});
                              }
                            }
                          }}
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
                        {d.status === "running" && d.port && (
                          <button
                            type="button"
                            className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-purple-300 transition-colors"
                            onClick={() => setBenchmarkTarget({
                              id: d.id,
                              label: `${d.displayName ?? d.model?.name ?? d.modelId} @ ${d.node?.name ?? "?"}`,
                            })}
                          >
                            Benchmark
                          </button>
                        )}
                        {latestBenchmarkStatus[d.id] && (
                          <a
                            href={`/benchmarks/${latestBenchmarkStatus[d.id].runId}`}
                            className="text-xs px-2 py-1 rounded bg-purple-900/40 hover:bg-purple-900/60 text-purple-200"
                          >
                            {latestBenchmarkStatus[d.id].status}
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
                        {(d.status === "stopped" || d.status === "failed" || d.status === "evicted") && (
                          <>
                            <button
                              onClick={() => {
                                if (isOllama) {
                                  restartDeployment(d.id);
                                  return;
                                }
                                editingRestart[d.id] ? cancelEditRestart(d.id) : beginEditRestart(d.id);
                              }}
                              className="text-xs px-2 py-1 rounded bg-blue-900/50 hover:bg-blue-800 text-blue-300 transition-colors"
                            >
                              {!isOllama && editingRestart[d.id] ? "Cancel" : "Restart"}
                            </button>
                            <button
                              onClick={() => deleteDeployment(d.id)}
                              className="text-xs px-2 py-1 rounded bg-red-900/50 hover:bg-red-800 text-red-300 transition-colors"
                            >
                              Delete
                            </button>
                          </>
                        )}
                      </>
                    )}
                  </div>
                </div>

                {/* Edit-and-restart form — appears when "Edit & restart" is clicked
                    on a stopped/failed/evicted deployment. Pre-filled from the
                    deployment's saved config; blank fields keep saved values. */}
                {!isOllama && editingRestart[d.id] && (
                  <div className="mt-3 p-3 bg-gray-800/50 rounded border border-blue-700/40">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-[10px] text-gray-400">
                        Edit settings then restart. Blank fields keep the saved value
                        {isFinetune ? "; placeholders show the training recipe's defaults." : "; placeholders show the recipe defaults."}
                      </p>
                      {isFinetune && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-900/60 text-amber-300 font-medium">
                          Fine-tune
                        </span>
                      )}
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                      <div>
                        <label className="block text-[10px] text-gray-500 mb-0.5">Port</label>
                        <input
                          type="number"
                          value={editingRestart[d.id].port ?? ""}
                          onChange={(e) => setEditingRestart((p) => ({ ...p, [d.id]: { ...p[d.id], port: e.target.value } }))}
                          placeholder={recipeDefaultsForRestart.port != null ? String(recipeDefaultsForRestart.port) : ""}
                          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] text-gray-500 mb-0.5">Tensor Parallel</label>
                        <input
                          type="number"
                          value={editingRestart[d.id].tensorParallel ?? ""}
                          onChange={(e) => setEditingRestart((p) => ({ ...p, [d.id]: { ...p[d.id], tensorParallel: e.target.value } }))}
                          placeholder={recipeDefaultsForRestart.tensor_parallel != null ? String(recipeDefaultsForRestart.tensor_parallel) : ""}
                          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] text-gray-500 mb-0.5">Pipeline Parallel</label>
                        <input
                          type="number"
                          value={editingRestart[d.id].pipelineParallel ?? ""}
                          onChange={(e) => setEditingRestart((p) => ({ ...p, [d.id]: { ...p[d.id], pipelineParallel: e.target.value } }))}
                          placeholder={recipeDefaultsForRestart.pipeline_parallel != null ? String(recipeDefaultsForRestart.pipeline_parallel) : ""}
                          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] text-gray-500 mb-0.5">Max Model Length</label>
                        <input
                          type="number"
                          value={editingRestart[d.id].maxModelLen ?? ""}
                          onChange={(e) => setEditingRestart((p) => ({ ...p, [d.id]: { ...p[d.id], maxModelLen: e.target.value } }))}
                          placeholder={recipeDefaultsForRestart.max_model_len != null ? String(recipeDefaultsForRestart.max_model_len) : ""}
                          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] text-gray-500 mb-0.5">GPU Memory Util</label>
                        <input
                          type="number"
                          step="any"
                          min="0"
                          value={editingRestart[d.id].gpuMem ?? ""}
                          onChange={(e) => setEditingRestart((p) => ({ ...p, [d.id]: { ...p[d.id], gpuMem: e.target.value } }))}
                          placeholder={recipeDefaultsForRestart.gpu_memory_utilization != null ? String(recipeDefaultsForRestart.gpu_memory_utilization) : ""}
                          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500"
                        />
                      </div>
                      {isFinetune && ftVariants.length > 0 && (
                        <div className="col-span-2 md:col-span-5">
                          <label className="block text-[10px] text-gray-500 mb-0.5">
                            Inference variant
                            {ftVariants.length === 1 && (
                              <span className="ml-2 text-gray-500">(only one available)</span>
                            )}
                          </label>
                          <select
                            value={editingRestart[d.id].artifactVariant ?? "default"}
                            onChange={(e) => setEditingRestart((p) => ({
                              ...p,
                              [d.id]: { ...p[d.id], artifactVariant: e.target.value },
                            }))}
                            disabled={ftVariants.length === 1}
                            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-blue-500 disabled:opacity-60"
                          >
                            {ftVariants.map((v) => (
                              <option key={v.id} value={v.id}>
                                {v.id} — {v.name}
                              </option>
                            ))}
                          </select>
                          {(() => {
                            const sel = ftVariants.find((v) => v.id === (editingRestart[d.id].artifactVariant ?? "default"));
                            return sel?.description ? (
                              <p className="mt-1 text-[10px] text-gray-500">{sel.description}</p>
                            ) : null;
                          })()}
                        </div>
                      )}
                    </div>
                    <div className="mt-2 flex justify-end gap-2">
                      <button
                        onClick={() => cancelEditRestart(d.id)}
                        className="text-xs px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={() => submitEditRestart(d.id)}
                        className="text-xs px-3 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white transition-colors"
                      >
                        Restart
                      </button>
                    </div>
                  </div>
                )}

                {/* In-flight phase progress (e.g. HF download) — only on head node */}
                {!isWorker && progress[d.id] && (
                  <div className="mt-3">
                    <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                      <span className="font-medium text-gray-300">
                        {progress[d.id].phase === "downloading" ? "Downloading model" : progress[d.id].phase}
                      </span>
                      <span className="tabular-nums">
                        {progress[d.id].phaseProgress.toFixed(0)}%
                        {progress[d.id].current != null && progress[d.id].total != null && (
                          <> · {progress[d.id].current}/{progress[d.id].total} files</>
                        )}
                        {progress[d.id].eta && <> · ETA {progress[d.id].eta}</>}
                      </span>
                    </div>
                    <div className="h-1.5 bg-gray-800 rounded overflow-hidden">
                      <div
                        className="h-full bg-cyan-500 transition-all duration-300"
                        style={{ width: `${Math.min(100, Math.max(0, progress[d.id].phaseProgress))}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Cluster nodes detail — only on head node */}
                {!isWorker && d.clusterMode && d.clusterNodes && d.clusterNodes.length > 0 && (
                  <div className="mt-3 flex gap-2 flex-wrap">
                    {d.clusterNodes.map((cn) => {
                      const cnNode = nodes.find((n) => n.name === cn.node.name);
                      const cnVram = cnNode?.metrics?.[0]?.vramUsed;
                      const cnTotal = cnNode?.vramTotal;
                      return (
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
                        {cnVram != null && cnTotal && (
                          <span className="ml-1 text-[10px] opacity-60">
                            {Math.round(cnVram / 1024)}GB/{Math.round(cnTotal / 1024)}GB
                          </span>
                        )}
                      </div>
                      );
                    })}
                  </div>
                )}

                {/* Log viewer */}
                {viewingLogs === d.id && (
                  <LogViewer content={logs[d.id] || "Waiting for logs..."} />
                )}
              </div>
            );
          })}
                </div>
              </div>
            );
          })}
        </div>
        );
      })()}

      {deployments.length === 0 && recipes.length > 0 && (
        <div className="text-center py-12 text-gray-500">
          <p>No deployments yet. Select a recipe and node above to deploy.</p>
        </div>
      )}

      {benchmarkTarget && (
        <BenchmarkFormModal
          deploymentId={benchmarkTarget.id}
          deploymentLabel={benchmarkTarget.label}
          onClose={() => setBenchmarkTarget(null)}
          onStarted={() => {/* SSE will populate latestBenchmarkStatus */}}
        />
      )}
    </div>
  );
}
