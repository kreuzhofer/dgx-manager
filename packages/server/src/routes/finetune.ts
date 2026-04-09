import { Router } from "express";
import { readFileSync, existsSync } from "fs";
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

finetuneRouter.get("/:id/logs", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });

  const logPath = `/mnt/tank/outputs/${job.id}/train.log`;
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

finetuneRouter.post("/", async (req, res) => {
  const { nodeId, recipeFile, dataset, config } = req.body;
  if (!nodeId || !recipeFile || !dataset) {
    return res.status(400).json({ error: "nodeId, recipeFile, and dataset required" });
  }

  // Look up recipe metadata from cached training recipes
  const agentHub: AgentHub = req.app.get("agentHub");
  const recipes = agentHub.getTrainingRecipes();
  const recipe = recipes.find((r) => r.file === recipeFile);

  const baseModel = recipe?.base_model || recipeFile;
  const method = recipe?.method || "lora";

  const job = await prisma.fineTuneJob.create({
    data: {
      nodeId,
      recipeFile,
      baseModel,
      method,
      dataset,
      config: config ? JSON.stringify(config) : null,
      status: "pending",
    },
  });

  // Set outputDir with the actual job ID
  const outputDir = `/mnt/tank/outputs/${job.id}`;
  await prisma.fineTuneJob.update({
    where: { id: job.id },
    data: { outputDir },
  });

  agentHub.sendToAgent(nodeId, {
    type: "cmd:finetune:start",
    payload: {
      jobId: job.id,
      recipeFile,
      dataset,
      outputDir: `/workspace/outputs/${job.id}`, // container path (NFS mounted)
      config: config || {},
    },
  });

  await prisma.fineTuneJob.update({
    where: { id: job.id },
    data: { status: "starting", startedAt: new Date() },
  });

  const result = await prisma.fineTuneJob.findUnique({
    where: { id: job.id },
    include: { node: true },
  });
  res.status(201).json(result);
});

finetuneRouter.delete("/:id", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });

  // If running, stop it first
  if (["pending", "starting", "running"].includes(job.status)) {
    const agentHub: AgentHub = req.app.get("agentHub");
    agentHub.sendToAgent(job.nodeId, {
      type: "cmd:finetune:stop",
      payload: { jobId: job.id },
    });
  }

  await prisma.fineTuneJob.delete({ where: { id: req.params.id } });
  res.json({ deleted: true });
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

finetuneRouter.post("/:id/merge", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });
  if (job.status !== "completed") return res.status(400).json({ error: "Job must be completed before merging" });

  const agentHub: AgentHub = req.app.get("agentHub");
  const mergedOutputDir = `/mnt/tank/outputs/${job.id}/merged`;

  agentHub.sendToAgent(job.nodeId, {
    type: "cmd:finetune:merge",
    payload: {
      jobId: job.id,
      baseModel: job.baseModel,
      adapterPath: job.outputDir ? `${job.outputDir}/lora_adapter` : job.outputPath!,
      mergedOutputDir,
    },
  });

  await prisma.fineTuneJob.update({
    where: { id: job.id },
    data: { mergeStatus: "running" },
  });

  res.json({ status: "merging", mergedOutputDir });
});

finetuneRouter.post("/:id/deploy", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({
    where: { id: req.params.id },
    include: { node: true },
  });
  if (!job) return res.status(404).json({ error: "Job not found" });

  const { nodeId, config } = req.body;
  const targetNodeId = nodeId || job.nodeId;

  // Determine model path — use merged if available, otherwise adapter
  const modelPath = job.mergedPath || (job.outputDir ? `${job.outputDir}/merged` : null);
  if (!modelPath || job.mergeStatus !== "completed") {
    return res.status(400).json({ error: "Model must be merged before deployment. Call POST /merge first." });
  }

  // Create a deployment record
  const model = await prisma.model.upsert({
    where: { name: `finetune-${job.id.slice(0, 8)}` },
    create: { name: `finetune-${job.id.slice(0, 8)}`, runtime: "vllm" },
    update: {},
  });

  const deployment = await prisma.deployment.create({
    data: {
      nodeId: targetNodeId,
      modelId: model.id,
      status: "pending",
      config: JSON.stringify({ ...config, localModelPath: modelPath }),
    },
  });

  await prisma.fineTuneJob.update({
    where: { id: job.id },
    data: { deploymentId: deployment.id },
  });

  // Look up the training recipe's deploy config for container/defaults
  const agentHub: AgentHub = req.app.get("agentHub");
  const trainingRecipe = job.recipeFile
    ? agentHub.getTrainingRecipes().find((r) => r.file === job.recipeFile)
    : undefined;
  const deployConfig = trainingRecipe?.deploy;

  agentHub.sendToAgent(targetNodeId, {
    type: "cmd:finetune:deploy",
    payload: {
      jobId: job.id,
      deploymentId: deployment.id,
      modelPath,
      baseModel: job.baseModel,
      deployContainer: deployConfig?.container || "vllm-node",
      config: {
        gpuMem: deployConfig?.gpu_memory_utilization,
        maxModelLen: deployConfig?.max_model_len,
        ...config,
      },
    },
  });

  const result = await prisma.deployment.findUnique({
    where: { id: deployment.id },
    include: { node: true, model: true },
  });
  res.status(201).json(result);
});
