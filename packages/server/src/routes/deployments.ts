import { Router } from "express";
import { prisma } from "../prisma.js";
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

deploymentsRouter.post("/", async (req, res) => {
  let { nodeId, nodeIds, recipeFile, config } = req.body;
  if (!recipeFile) {
    return res.status(400).json({ error: "recipeFile required" });
  }

  const activeStatuses = ["pending", "running", "starting", "building", "downloading", "launching", "loading", "restarting"];

  // Auto-resolve idle nodes
  if (nodeId === "auto" || nodeIds === "auto") {
    const idleNodes = await prisma.node.findMany({
      where: {
        status: "online",
        AND: [
          { deployments: { none: { status: { in: activeStatuses } } } },
          { clusterMemberships: { none: { deployment: { status: { in: activeStatuses } } } } },
        ],
      },
      orderBy: { createdAt: "asc" },
    });

    if (idleNodes.length === 0) {
      return res.status(409).json({ error: "No idle nodes available" });
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
          error: `Recipe requires ${needed} nodes (TP=${tp} × PP=${pp}) but only ${idleNodes.length} idle`,
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

  // For cluster deployments, verify no selected node is busy
  if (isCluster) {
    const busy = await prisma.deployment.findMany({
      where: {
        status: { in: activeStatuses },
        OR: [
          { nodeId: { in: nodeIds } },
          { clusterNodes: { some: { nodeId: { in: nodeIds } } } },
        ],
      },
    });
    if (busy.length > 0) {
      return res.status(409).json({
        error: "One or more selected nodes already have active deployments",
        busyNodes: busy.map((d) => d.nodeId),
      });
    }
  }

  // Ensure a model record exists
  const recipeName = recipeFile.replace(/^recipes\//, "").replace(/\.yaml$/, "");
  let model = await prisma.model.findUnique({ where: { name: recipeName } });
  if (!model) {
    model = await prisma.model.create({
      data: { name: recipeName, runtime: "vllm" },
    });
  }

  // Create deployment
  const deployment = await prisma.deployment.create({
    data: {
      modelId: model.id,
      nodeId: headNodeId,
      clusterMode: isCluster,
      config: JSON.stringify({ recipeFile, ...config }),
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
      recipeFile,
      config: config || {},
      clusterNodes: clusterNodeIps,
    },
  });

  // Return with cluster info
  const result = await prisma.deployment.findUnique({
    where: { id: deployment.id },
    include: { node: true, model: true, clusterNodes: { include: { node: true } } },
  });

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

  agentHub.sendToAgent(deployment.nodeId, {
    type: "cmd:undeploy",
    payload: {
      deploymentId: deployment.id,
      deleteAfter: wantDelete,
      clusterNodes: clusterNodeIps,
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

  agentHub.sendToAgent(deployment.nodeId, {
    type: "cmd:deploy",
    payload: {
      deploymentId: deployment.id,
      recipeFile: config.recipeFile,
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
