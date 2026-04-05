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
  const { modelId, nodeId, config } = req.body;
  if (!modelId || !nodeId) {
    return res.status(400).json({ error: "modelId and nodeId required" });
  }

  const model = await prisma.model.findUnique({ where: { id: modelId } });
  if (!model) return res.status(404).json({ error: "Model not found" });

  const deployment = await prisma.deployment.create({
    data: { modelId, nodeId, config: config ? JSON.stringify(config) : null },
  });

  const agentHub: AgentHub = req.app.get("agentHub");
  agentHub.sendToAgent(nodeId, {
    type: "cmd:deploy",
    payload: {
      deploymentId: deployment.id,
      modelName: model.name,
      runtime: model.runtime,
      config: config || {},
    },
  });

  res.status(201).json(deployment);
});

deploymentsRouter.delete("/:id", async (req, res) => {
  const deployment = await prisma.deployment.findUnique({ where: { id: req.params.id } });
  if (!deployment) return res.status(404).json({ error: "Deployment not found" });

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
});

deploymentsRouter.post("/:id/restart", async (req, res) => {
  const deployment = await prisma.deployment.findUnique({
    where: { id: req.params.id },
    include: { model: true },
  });
  if (!deployment) return res.status(404).json({ error: "Deployment not found" });

  const agentHub: AgentHub = req.app.get("agentHub");
  agentHub.sendToAgent(deployment.nodeId, {
    type: "cmd:deploy",
    payload: {
      deploymentId: deployment.id,
      modelName: deployment.model.name,
      runtime: deployment.model.runtime,
      config: deployment.config ? JSON.parse(deployment.config) : {},
    },
  });

  await prisma.deployment.update({
    where: { id: req.params.id },
    data: { status: "restarting" },
  });

  res.json({ status: "restarting" });
});
