import { Router } from "express";
import { prisma } from "../prisma.js";
import type { AgentHub } from "../ws/agent-hub.js";

export const finetuneRouter = Router();

finetuneRouter.get("/", async (_req, res) => {
  const jobs = await prisma.fineTuneJob.findMany({
    orderBy: { createdAt: "desc" },
    include: { node: true },
  });
  res.json(jobs);
});

finetuneRouter.get("/:id", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({
    where: { id: req.params.id },
    include: { node: true },
  });
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json(job);
});

finetuneRouter.post("/", async (req, res) => {
  const { nodeId, baseModel, method, dataset, config } = req.body;
  if (!nodeId || !baseModel || !method || !dataset) {
    return res.status(400).json({ error: "nodeId, baseModel, method, and dataset required" });
  }

  const job = await prisma.fineTuneJob.create({
    data: {
      nodeId,
      baseModel,
      method,
      dataset,
      config: config ? JSON.stringify(config) : null,
      status: "pending",
    },
  });

  const agentHub: AgentHub = req.app.get("agentHub");
  agentHub.sendToAgent(nodeId, {
    type: "cmd:finetune:start",
    payload: {
      jobId: job.id,
      baseModel,
      method,
      dataset,
      config: config || {},
    },
  });

  await prisma.fineTuneJob.update({
    where: { id: job.id },
    data: { status: "running", startedAt: new Date() },
  });

  res.status(201).json(job);
});

finetuneRouter.post("/:id/stop", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });

  const agentHub: AgentHub = req.app.get("agentHub");
  agentHub.sendToAgent(job.nodeId, {
    type: "cmd:finetune:stop",
    payload: { jobId: job.id },
  });

  await prisma.fineTuneJob.update({
    where: { id: req.params.id },
    data: { status: "stopping" },
  });

  res.json({ status: "stopping" });
});
