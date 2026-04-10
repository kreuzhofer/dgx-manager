import { Router } from "express";
import { readFileSync, existsSync } from "fs";
import { prisma } from "../prisma.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";
import type { AgentHub } from "../ws/agent-hub.js";

export const deploymentsRouter = Router();

deploymentsRouter.get("/", async (_req, res) => {
  const deployments = await prisma.deployment.findMany({
    orderBy: { createdAt: "desc" },
    include: {
      node: true,
      model: true,
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
  let { nodeId, nodeIds, recipeFile, config, runtime, modelName, modelType } = req.body;
  const isOllama = runtime === "ollama";

  if (!isOllama && !recipeFile) {
    return res.status(400).json({ error: "recipeFile required for vLLM deployments" });
  }
  if (isOllama && !modelName) {
    return res.status(400).json({ error: "modelName required for Ollama deployments" });
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

  // VRAM-based admission check for solo deployments
  if (!isCluster) {
    // Get node's latest metrics
    const node = await prisma.node.findUnique({ where: { id: headNodeId } });
    const latestMetric = await prisma.metricSnapshot.findFirst({
      where: { nodeId: headNodeId },
      orderBy: { timestamp: "desc" },
    });
    const vramTotal = node?.vramTotal || 128000;
    const vramUsed = latestMetric?.vramUsed || 0;
    const vramAvailable = vramTotal - vramUsed;

    // Estimate VRAM needed for this deployment
    if (isOllama) {
      // Parse size from ollama model list (e.g. "20GB" → 20000 MB)
      const agentHub: AgentHub = req.app.get("agentHub");
      const ollamaModel = agentHub.getOllamaModels().find((m) => m.name === modelName);
      if (ollamaModel?.size) {
        const sizeMatch = ollamaModel.size.match(/([\d.]+)\s*GB/i);
        vramEstimate = sizeMatch ? Math.round(parseFloat(sizeMatch[1]) * 1024 * 1.1) : 0; // +10% overhead
      }
    } else {
      // vLLM: estimate from gpu_memory_utilization
      const agentHub: AgentHub = req.app.get("agentHub");
      const recipe = agentHub.getRecipes().find((r) => r.file === recipeFile);
      const gpuMemUtil = (config?.gpuMem as number) || (recipe?.defaults?.gpu_memory_utilization as number) || 0.85;
      vramEstimate = Math.round(vramTotal * gpuMemUtil);
    }

    // Check port conflict for vLLM
    if (!isOllama) {
      const requestedPort = (config?.port as number) || 8000;
      const portConflict = await prisma.deployment.findFirst({
        where: {
          nodeId: headNodeId,
          port: requestedPort,
          status: { in: activeStatuses },
        },
      });
      if (portConflict) {
        // Auto-increment port
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

    // Check if there's enough VRAM (5% safety margin)
    const safetyMargin = vramTotal * 0.05;
    if (vramEstimate > 0 && vramEstimate > vramAvailable - safetyMargin) {
      return res.status(409).json({
        error: `Not enough VRAM: need ~${Math.round(vramEstimate / 1024)}GB, only ${Math.round(vramAvailable / 1024)}GB available`,
        vramEstimate,
        vramAvailable,
        vramTotal,
        vramUsed,
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
      config: JSON.stringify(isOllama
        ? { runtime: "ollama", modelName, modelType: modelType || "chat", ...config }
        : { recipeFile, ...config }),
    },
  });

  // For cluster, create ClusterNode records and resolve IPs
  let clusterNodeIps: string[] | undefined;
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

    // Ordered: head first, then workers
    clusterNodeIps = nodeIds.map((id: string) => nodeMap.get(id)?.ipAddress).filter(Boolean);
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
      config: config || {},
      clusterNodes: clusterNodeIps,
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
    include: { model: true, clusterNodes: { include: { node: true } } },
  });
  if (!deployment) return res.status(404).json({ error: "Deployment not found" });

  const agentHub: AgentHub = req.app.get("agentHub");
  const config = deployment.config ? JSON.parse(deployment.config) : {};

  const clusterNodeIps = deployment.clusterMode
    ? deployment.clusterNodes.map((cn) => cn.node.ipAddress)
    : undefined;

  const isOllamaRestart = config.runtime === "ollama";
  agentHub.sendToAgent(deployment.nodeId, {
    type: "cmd:deploy",
    payload: {
      deploymentId: deployment.id,
      runtime: isOllamaRestart ? "ollama" : "vllm",
      modelName: isOllamaRestart ? config.modelName : undefined,
      modelType: isOllamaRestart ? (config.modelType || "chat") : undefined,
      recipeFile: isOllamaRestart ? undefined : config.recipeFile,
      config,
      clusterNodes: clusterNodeIps,
    },
  });

  await prisma.deployment.update({
    where: { id: req.params.id },
    data: { status: "restarting" },
  });

  res.json({ status: "restarting" });
});
