import { Router } from "express";
import { prisma } from "../prisma.js";
import type { AgentHub } from "../ws/agent-hub.js";

export const deploymentsRouter = Router();

deploymentsRouter.get("/", async (_req, res) => {
  const deployments = await prisma.deployment.findMany({
    orderBy: { createdAt: "desc" },
    include: { node: true, model: true },
  });
  res.json(deployments);
});

deploymentsRouter.post("/", async (req, res) => {
  const { nodeId, recipeFile, config } = req.body;
  if (!nodeId || !recipeFile) {
    return res.status(400).json({ error: "nodeId and recipeFile required" });
  }

  // Ensure a model record exists for this recipe
  const recipeName = recipeFile.replace(/^recipes\//, "").replace(/\.yaml$/, "");
  let model = await prisma.model.findUnique({ where: { name: recipeName } });
  if (!model) {
    model = await prisma.model.create({
      data: { name: recipeName, runtime: "vllm" },
    });
  }

  const deployment = await prisma.deployment.create({
    data: {
      modelId: model.id,
      nodeId,
      config: JSON.stringify({ recipeFile, ...config }),
    },
  });

  const agentHub: AgentHub = req.app.get("agentHub");
  agentHub.sendToAgent(nodeId, {
    type: "cmd:deploy",
    payload: {
      deploymentId: deployment.id,
      recipeFile,
      config: config || {},
    },
  });

  res.status(201).json(deployment);
});

deploymentsRouter.delete("/:id", async (req, res) => {
  const deployment = await prisma.deployment.findUnique({
    where: { id: req.params.id },
    include: { lbEndpoints: true },
  });
  if (!deployment) return res.status(404).json({ error: "Deployment not found" });

  const isActive = ["pending", "running", "starting", "restarting"].includes(deployment.status);

  if (isActive) {
    // Active deployment: send undeploy command, mark as removing
    const agentHub: AgentHub = req.app.get("agentHub");
    agentHub.sendToAgent(deployment.nodeId, {
      type: "cmd:undeploy",
      payload: { deploymentId: deployment.id },
    });
    await prisma.deployment.update({
      where: { id: req.params.id },
      data: { status: "removing" },
    });
    res.json({ status: "removing" });
  } else {
    // Stopped/failed: delete the record
    await prisma.loadBalancerEndpoint.deleteMany({
      where: { deploymentId: deployment.id },
    });
    await prisma.deployment.delete({ where: { id: req.params.id } });
    res.json({ deleted: true });
  }
});

deploymentsRouter.post("/:id/restart", async (req, res) => {
  const deployment = await prisma.deployment.findUnique({
    where: { id: req.params.id },
    include: { model: true },
  });
  if (!deployment) return res.status(404).json({ error: "Deployment not found" });

  const agentHub: AgentHub = req.app.get("agentHub");
  const config = deployment.config ? JSON.parse(deployment.config) : {};
  agentHub.sendToAgent(deployment.nodeId, {
    type: "cmd:deploy",
    payload: {
      deploymentId: deployment.id,
      recipeFile: config.recipeFile,
      modelName: deployment.model.name,
      runtime: deployment.model.runtime,
      config,
    },
  });

  await prisma.deployment.update({
    where: { id: req.params.id },
    data: { status: "restarting" },
  });

  res.json({ status: "restarting" });
});
