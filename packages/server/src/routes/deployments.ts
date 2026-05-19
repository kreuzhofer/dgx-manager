import { Router } from "express";
import { readFileSync, existsSync } from "fs";
import { prisma } from "../prisma.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";
import type { AgentHub } from "../ws/agent-hub.js";
import { checkVllmVramAdmission, vramShortfallMessage } from "../admission/vram.js";
import { readCatalog as readOllamaCatalog } from "../ollama/catalog-store.js";
import { ollamaVramEstimateMB } from "../ollama/vram-estimate.js";
import { normalizeDisplayName, validateDisplayNameUnique, DisplayNameError } from "../deployments/display-name.js";
import { isValidVariantSlug } from "./finetune.js";

export const deploymentsRouter = Router();

deploymentsRouter.get("/", async (_req, res) => {
  const deployments = await prisma.deployment.findMany({
    orderBy: { createdAt: "desc" },
    include: {
      node: true,
      // Surface the linked FineTuneJob.recipeFile so the dashboard's
      // edit-restart form can look up training-recipe defaults for its
      // placeholders — fine-tune deployments don't store recipeFile in
      // their saved config, it lives on the FineTuneJob row.
      model: { include: { finetuneJob: { select: { recipeFile: true } } } },
      clusterNodes: { include: { node: true } },
    },
  });
  res.json(deployments);
});

deploymentsRouter.get("/:id/logs", async (req, res) => {
  const logPath = `${SHARED_STORAGE}/logs/deployments/${req.params.id}.log`;
  if (!existsSync(logPath)) {
    return res.type("text/plain").send("");
  }
  try {
    const content = readFileSync(logPath, "utf-8");
    const tail = parseInt(req.query.tail as string);
    if (tail > 0) {
      const lines = content.split("\n");
      return res.type("text/plain").send(lines.slice(-tail).join("\n"));
    }
    res.type("text/plain").send(content);
  } catch {
    res.type("text/plain").send("");
  }
});

deploymentsRouter.post("/", async (req, res) => {
  let { nodeId, nodeIds, recipeFile, config, runtime, modelName, modelType, displayName: rawDisplayName } = req.body;
  const isOllama = runtime === "ollama";

  if (!isOllama && !recipeFile) {
    return res.status(400).json({ error: "recipeFile required for vLLM deployments" });
  }
  if (isOllama && !modelName) {
    return res.status(400).json({ error: "modelName required for Ollama deployments" });
  }

  // Normalize + uniqueness-check displayName up front. 400 on bad chars,
  // 409 on duplicate among active deployments. Ollama deploys ignore the
  // field (runtime doesn't honor it); we still reject malformed values so
  // the dashboard surfaces the error immediately.
  let displayName: string | null;
  try {
    displayName = normalizeDisplayName(rawDisplayName);
  } catch (e) {
    if (e instanceof DisplayNameError) return res.status(400).json({ error: e.message });
    throw e;
  }
  // Ollama doesn't support per-deploy renames; reject explicitly so the user
  // doesn't think it took effect.
  if (displayName && isOllama) {
    return res.status(400).json({
      error: "displayName is not supported for Ollama deployments (use the model tag).",
    });
  }
  if (displayName && !isOllama) {
    const conflict = await validateDisplayNameUnique(prisma, displayName);
    if (conflict) {
      return res.status(409).json({
        error: `Display name "${displayName}" is already in use by deployment ${conflict.conflictId}.`,
        conflict,
      });
    }
  }

  const activeStatuses = ["pending", "running", "starting", "building", "downloading", "launching", "loading", "restarting"];

  // Auto-resolve idle nodes
  if (nodeId === "auto" || nodeIds === "auto") {
    const idleNodes = await prisma.node.findMany({
      where: { status: "online" },
      orderBy: { name: "asc" },
    });

    if (idleNodes.length === 0) {
      return res.status(409).json({ error: "No online nodes available" });
    }

    if (nodeIds === "auto") {
      // Determine required node count:
      // 1. User config overrides take priority
      // 2. Fall back to recipe defaults
      // 3. Fall back to all idle nodes
      const agentHub: AgentHub = req.app.get("agentHub");
      const recipe = agentHub.getRecipes().find((r) => r.file === recipeFile);
      const recipeDefaults = recipe?.defaults || {};

      const tp = (config?.tensorParallel as number) || (recipeDefaults.tensor_parallel as number) || 1;
      const pp = (config?.pipelineParallel as number) || (recipeDefaults.pipeline_parallel as number) || 1;
      const needed = tp * pp;

      if (needed > idleNodes.length) {
        return res.status(409).json({
          error: `Recipe requires ${needed} nodes (TP=${tp} × PP=${pp}) but only ${idleNodes.length} online`,
        });
      }

      nodeIds = idleNodes.slice(0, needed).map((n) => n.id);
    } else {
      // Solo: use first idle node
      nodeId = idleNodes[0].id;
    }
  }

  const isCluster = Array.isArray(nodeIds) && nodeIds.length > 1;
  const headNodeId = isCluster ? nodeIds[0] : nodeId;

  if (!headNodeId) {
    return res.status(400).json({ error: "nodeId or nodeIds required" });
  }


  let vramEstimate = 0;

  // Compute VRAM the deployment would request, for the response body and
  // for the admission check. For Ollama it's based on the model's listed
  // size; for vLLM (solo or cluster) it's based on gpu_memory_utilization.
  if (!isCluster) {
    const node = await prisma.node.findUnique({ where: { id: headNodeId } });
    const vramTotal = node?.vramTotal || 128000;
    if (isOllama) {
      const agentHub: AgentHub = req.app.get("agentHub");
      // Resolve the parameter size from the catalog: modelName is the pull
      // tag like "llama3.1:8b". Look up the bare model name, then pick the
      // requested size out of its `sizes` list (or the bare name for sizeless
      // entries like nomic-embed-text).
      const catalog = await readOllamaCatalog();
      const [bareName, requestedSize] = modelName.includes(":")
        ? modelName.split(":", 2)
        : [modelName, null];
      const catalogEntry = catalog.entries.find((m) => m.name === bareName);
      const catalogSize =
        catalogEntry && requestedSize && catalogEntry.sizes.includes(requestedSize)
          ? requestedSize
          : catalogEntry?.sizes[0] ?? null;
      const estimate = ollamaVramEstimateMB(catalogSize);
      if (estimate !== null) {
        vramEstimate = Math.round(estimate * 1.1); // +10% overhead (KV cache, runtime)
      } else {
        // Fallback: legacy agent-shipped byte size, parsed as before. Used
        // until the operator runs the first catalog refresh.
        const legacy = agentHub.getOllamaModels().find((m) => m.name === modelName);
        if (legacy?.size) {
          const sizeMatch = legacy.size.match(/([\d.]+)\s*GB/i);
          vramEstimate = sizeMatch ? Math.round(parseFloat(sizeMatch[1]) * 1024 * 1.1) : 0;
        }
      }
    } else {
      const agentHub: AgentHub = req.app.get("agentHub");
      const recipe = agentHub.getRecipes().find((r) => r.file === recipeFile);
      const gpuMemUtil = (config?.gpuMem as number) || (recipe?.defaults?.gpu_memory_utilization as number) || 0.85;
      vramEstimate = Math.round(vramTotal * gpuMemUtil);
    }

    // Auto-bump the port if it's already in use on this node.
    if (!isOllama) {
      const requestedPort = (config?.port as number) || 8000;
      const portConflict = await prisma.deployment.findFirst({
        where: { nodeId: headNodeId, port: requestedPort, status: { in: activeStatuses } },
      });
      if (portConflict) {
        const usedPorts = await prisma.deployment.findMany({
          where: { nodeId: headNodeId, status: { in: activeStatuses }, port: { not: null } },
          select: { port: true },
        });
        const usedPortSet = new Set(usedPorts.map((d) => d.port));
        let nextPort = requestedPort;
        while (usedPortSet.has(nextPort)) nextPort++;
        if (!config) config = {};
        config.port = nextPort;
      }
    }
  }

  // Pre-flight VRAM admission check — for vLLM only, applies to both solo
  // and cluster deployments. Refuses upfront with a per-node breakdown of
  // what's holding the memory; the user decides what to stop. We never
  // auto-evict.
  if (!isOllama) {
    const agentHub: AgentHub = req.app.get("agentHub");
    const recipe = agentHub.getRecipes().find((r) => r.file === recipeFile);
    const gpuMemUtil = (config?.gpuMem as number) || (recipe?.defaults?.gpu_memory_utilization as number) || 0.85;
    const checkNodeIds = isCluster ? (nodeIds as string[]) : [headNodeId];
    const shortfalls = await checkVllmVramAdmission(checkNodeIds, gpuMemUtil);
    if (shortfalls.length > 0) {
      return res.status(409).json({
        error: `Not enough VRAM on ${shortfalls.length} of ${checkNodeIds.length} node(s): ${vramShortfallMessage(shortfalls)}`,
        shortfalls,
        gpuMemoryUtilization: gpuMemUtil,
      });
    }
  }

  // Ensure a model record exists
  const modelKey = isOllama ? modelName : recipeFile.replace(/^recipes\//, "").replace(/\.yaml$/, "");
  let model = await prisma.model.findUnique({ where: { name: modelKey } });
  if (!model) {
    model = await prisma.model.create({
      data: { name: modelKey, runtime: isOllama ? "ollama" : "vllm" },
    });
  }

  // Create deployment
  const deployment = await prisma.deployment.create({
    data: {
      modelId: model.id,
      nodeId: headNodeId,
      clusterMode: isCluster,
      vramEstimate: vramEstimate || null,
      displayName,
      config: JSON.stringify(isOllama
        ? { runtime: "ollama", modelName, modelType: modelType || "chat", ...config }
        : { recipeFile, ...config }),
    },
  });

  // For cluster, create ClusterNode records and resolve IPs
  let clusterNodeIps: string[] | undefined;
  let clusterNodeFastIps: (string | null)[] | undefined;
  if (isCluster) {
    const nodes = await prisma.node.findMany({
      where: { id: { in: nodeIds } },
    });
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));

    for (let i = 0; i < nodeIds.length; i++) {
      const nid = nodeIds[i];
      await prisma.clusterNode.create({
        data: {
          deploymentId: deployment.id,
          nodeId: nid,
          role: nid === headNodeId ? "head" : "worker",
          status: "pending",
        },
      });
    }

    // Ordered: head first, then workers. clusterNodeIps stays the
    // management network (used for Ray inter-node and as default for ssh);
    // clusterNodeFastIps is the per-node fast-fabric IP when known so the
    // agent can route bulk transfers (image sync, etc.) over it.
    clusterNodeIps = nodeIds.map((id: string) => nodeMap.get(id)?.ipAddress).filter(Boolean);
    clusterNodeFastIps = nodeIds.map((id: string) => nodeMap.get(id)?.fastIpAddress ?? null);
  }

  // Send deploy command to head agent
  const agentHub: AgentHub = req.app.get("agentHub");
  agentHub.sendToAgent(headNodeId, {
    type: "cmd:deploy",
    payload: {
      deploymentId: deployment.id,
      runtime: isOllama ? "ollama" : "vllm",
      modelName: isOllama ? modelName : undefined,
      modelType: isOllama ? (modelType || "chat") : undefined,
      recipeFile: isOllama ? undefined : recipeFile,
      // Per-deploy custom name → vLLM's --served-model-name. Undefined when
      // the user didn't set displayName, so the agent falls back to the
      // recipe's authored defaults.served_model_name.
      servedModelName: displayName ?? undefined,
      config: config || {},
      clusterNodes: clusterNodeIps,
      clusterNodeFastIps,
    },
  });

  // Return with cluster info
  const result = await prisma.deployment.findUnique({
    where: { id: deployment.id },
    include: { node: true, model: true, clusterNodes: { include: { node: true } } },
  });

  sseBroadcast({ type: "deployment:created", payload: result });
  res.status(201).json(result);
});

// DELETE /api/deployments/:id — stop deployment
deploymentsRouter.delete("/:id", async (req, res) => {
  const deployment = await prisma.deployment.findUnique({
    where: { id: req.params.id },
    include: { clusterNodes: { include: { node: true } } },
  });
  if (!deployment) return res.status(404).json({ error: "Deployment not found" });

  const agentHub: AgentHub = req.app.get("agentHub");
  const wantDelete = req.query.delete === "true";

  // For cluster deployments, pass cluster IPs to the head agent
  const clusterNodeIps = deployment.clusterMode
    ? deployment.clusterNodes.map((cn) => cn.node.ipAddress)
    : undefined;

  const deployConfig = deployment.config ? JSON.parse(deployment.config) : {};

  // Fast path: deployment is already in a terminal state. There's nothing
  // live to stop, so don't go through the agent round-trip — if the agent
  // can't reach its cluster nodes (common after a failed launch) it reports
  // "failed" without deleteAfter, and the auto-delete in agent-hub never
  // fires. Just clean up the row directly and best-effort tell the agent to
  // drop any stale tracking state.
  const terminalStatuses = ["stopped", "failed", "evicted"];
  if (wantDelete && terminalStatuses.includes(deployment.status)) {
    if (agentHub.isAgentOnline(deployment.nodeId)) {
      agentHub.sendToAgent(deployment.nodeId, {
        type: "cmd:undeploy",
        payload: {
          deploymentId: deployment.id,
          deleteAfter: false,
          clusterNodes: clusterNodeIps,
          runtime: deployConfig.runtime || "vllm",
          modelName: deployConfig.modelName,
        },
      });
    }
    await prisma.clusterNode.deleteMany({ where: { deploymentId: deployment.id } });
    await prisma.loadBalancerEndpoint.deleteMany({ where: { deploymentId: deployment.id } });
    await prisma.deployment.delete({ where: { id: deployment.id } });
    sseBroadcast({ type: "deployment:deleted", payload: { deploymentId: deployment.id } });
    return res.json({ status: "deleted" });
  }

  agentHub.sendToAgent(deployment.nodeId, {
    type: "cmd:undeploy",
    payload: {
      deploymentId: deployment.id,
      deleteAfter: wantDelete,
      clusterNodes: clusterNodeIps,
      runtime: deployConfig.runtime || "vllm",
      modelName: deployConfig.modelName,
    },
  });

  await prisma.deployment.update({
    where: { id: req.params.id },
    data: { status: "removing" },
  });

  // Update cluster node statuses
  if (deployment.clusterMode) {
    await prisma.clusterNode.updateMany({
      where: { deploymentId: deployment.id },
      data: { status: "stopping" },
    });
  }

  res.json({ status: "removing" });
});

deploymentsRouter.post("/:id/restart", async (req, res) => {
  const deployment = await prisma.deployment.findUnique({
    where: { id: req.params.id },
    include: {
      // Including finetuneJob lets the restart path detect fine-tune
      // deployments and dispatch cmd:finetune:deploy instead of cmd:deploy
      // — fine-tunes have no recipeFile in their saved config (the recipe
      // lives on the FineTuneJob row), so the legacy cmd:deploy path was
      // failing them with "No recipeFile specified".
      model: { include: { finetuneJob: true } },
      clusterNodes: { include: { node: true } },
    },
  });
  if (!deployment) return res.status(404).json({ error: "Deployment not found" });

  const agentHub: AgentHub = req.app.get("agentHub");
  const savedConfig = deployment.config ? JSON.parse(deployment.config) : {};

  // Merge any caller-supplied overrides over the saved config so the restart
  // path can be used to fix bad settings (e.g. lower max_model_len after an
  // OOM) without having to delete + re-create the deployment. recipeFile,
  // runtime, modelName, modelType are intentionally NOT overridable here —
  // those identify the deployment.
  const overrides = (req.body && typeof req.body === "object" && req.body.config && typeof req.body.config === "object")
    ? req.body.config as Record<string, unknown>
    : {};
  const RESERVED = new Set(["recipeFile", "runtime", "modelName", "modelType"]);
  for (const k of Object.keys(overrides)) {
    if (RESERVED.has(k)) delete overrides[k];
  }
  const config = { ...savedConfig, ...overrides };

  if (typeof overrides.artifactVariant !== "undefined" && !isValidVariantSlug(overrides.artifactVariant)) {
    return res.status(400).json({
      error: `Invalid artifactVariant: must match /^[a-z0-9][a-z0-9-]{0,31}$/ (got ${JSON.stringify(overrides.artifactVariant)})`,
    });
  }

  // Allow the caller to update displayName on restart (re-validating
  // uniqueness, excluding self). Body shape: { displayName: "new-name" } at
  // the top level (NOT nested under config — displayName is a column, not
  // part of the recipe config blob).
  let newDisplayName = deployment.displayName;
  if (req.body && Object.prototype.hasOwnProperty.call(req.body, "displayName")) {
    try {
      newDisplayName = normalizeDisplayName(req.body.displayName as string | null | undefined);
    } catch (e) {
      if (e instanceof DisplayNameError) return res.status(400).json({ error: e.message });
      throw e;
    }
    // Always re-validate uniqueness when restarting with a non-null displayName.
    // The `excludeDeploymentId` arg handles the "same name as me" case, so even
    // no-op restarts are safe. Without this, a deployment that went terminal
    // while another claimed its name could silently slip through on restart.
    if (newDisplayName) {
      const conflict = await validateDisplayNameUnique(prisma, newDisplayName, deployment.id);
      if (conflict) {
        return res.status(409).json({
          error: `Display name "${newDisplayName}" is already in use by deployment ${conflict.conflictId}.`,
          conflict,
        });
      }
    }
  }

  const clusterNodeIps = deployment.clusterMode
    ? deployment.clusterNodes.map((cn) => cn.node.ipAddress)
    : undefined;
  const clusterNodeFastIps = deployment.clusterMode
    ? deployment.clusterNodes.map((cn) => cn.node.fastIpAddress ?? null)
    : undefined;

  const isOllamaRestart = config.runtime === "ollama";

  // Same pre-flight VRAM check as the deploy POST. Excludes the deployment
  // being restarted from the conflict list (it's about to be relaunched, so
  // its own VRAM usage shouldn't count against itself).
  if (!isOllamaRestart) {
    const agentHub: AgentHub = req.app.get("agentHub");
    const recipe = config.recipeFile
      ? agentHub.getRecipes().find((r) => r.file === config.recipeFile)
      : undefined;
    const gpuMemUtil = (config.gpuMem as number) || (recipe?.defaults?.gpu_memory_utilization as number) || 0.85;
    const checkNodeIds = deployment.clusterMode
      ? deployment.clusterNodes.map((cn) => cn.nodeId)
      : [deployment.nodeId];
    const shortfalls = await checkVllmVramAdmission(checkNodeIds, gpuMemUtil, deployment.id);
    if (shortfalls.length > 0) {
      return res.status(409).json({
        error: `Not enough VRAM on ${shortfalls.length} of ${checkNodeIds.length} node(s): ${vramShortfallMessage(shortfalls)}`,
        shortfalls,
        gpuMemoryUtilization: gpuMemUtil,
      });
    }
  }
  // Fine-tune deployments route through cmd:finetune:deploy — they have no
  // recipeFile in saved config (the recipe lives on the FineTuneJob), and
  // the agent's finetune handler reads jobId/modelPath/baseModel/recipeFile
  // from the payload directly. Detect by the model row's finetuneJobId FK.
  const ftJob = deployment.model.finetuneJob;
  if (ftJob && !isOllamaRestart) {
    const trainingRecipe = ftJob.recipeFile
      ? agentHub.getTrainingRecipes().find((r) => r.file === ftJob.recipeFile)
      : undefined;
    const deployContainer =
      (trainingRecipe?.deploy?.container as string | undefined) ?? "vllm-node";
    // localModelPath was persisted by the original finetune deploy route as
    // the bf16 merged path; for fp8 vLLM does on-load quantization off the
    // same path, so we re-use it unchanged. Variant tells the agent which
    // inference template (inference.yaml / inference-fp8.yaml) to apply.
    const artifactVariant: string = typeof config.artifactVariant === "string" && isValidVariantSlug(config.artifactVariant)
      ? config.artifactVariant
      : "default";
    agentHub.sendToAgent(deployment.nodeId, {
      type: "cmd:finetune:deploy",
      payload: {
        jobId: ftJob.id,
        deploymentId: deployment.id,
        modelPath: config.localModelPath,
        baseModel: ftJob.baseModel,
        deployContainer,
        modelName: newDisplayName ?? deployment.model.name,
        recipeFile: ftJob.recipeFile,
        artifactVariant,
        clusterNodes: clusterNodeIps,
        clusterNodeFastIps,
        config,
      },
    });
  } else {
    agentHub.sendToAgent(deployment.nodeId, {
      type: "cmd:deploy",
      payload: {
        deploymentId: deployment.id,
        runtime: isOllamaRestart ? "ollama" : "vllm",
        modelName: isOllamaRestart ? config.modelName : undefined,
        modelType: isOllamaRestart ? (config.modelType || "chat") : undefined,
        recipeFile: isOllamaRestart ? undefined : config.recipeFile,
        servedModelName: newDisplayName ?? undefined,
        config,
        clusterNodes: clusterNodeIps,
        clusterNodeFastIps,
      },
    });
  }

  await prisma.deployment.update({
    where: { id: req.params.id },
    data: {
      status: "restarting",
      // Persist the merged config so future restarts (or the agent's own
      // reconciliation) see the updated overrides.
      ...(Object.keys(overrides).length > 0 ? { config: JSON.stringify(config) } : {}),
      ...(newDisplayName !== deployment.displayName ? { displayName: newDisplayName } : {}),
    },
  });

  res.json({ status: "restarting" });
});
