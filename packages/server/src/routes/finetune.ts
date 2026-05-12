import { Router } from "express";
import { readFileSync, existsSync } from "fs";
import { prisma } from "../prisma.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";
import type { AgentHub } from "../ws/agent-hub.js";

export const finetuneRouter = Router();

finetuneRouter.get("/", async (_req, res) => {
  const jobs = await prisma.fineTuneJob.findMany({
    orderBy: { createdAt: "desc" },
    include: {
      node: true,
      clusterNodes: { include: { node: true }, orderBy: { role: "asc" } },
    },
  });
  res.json(jobs);
});

finetuneRouter.get("/:id", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({
    where: { id: req.params.id },
    include: {
      node: true,
      clusterNodes: { include: { node: true }, orderBy: { role: "asc" } },
    },
  });
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json(job);
});

finetuneRouter.get("/:id/logs", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });

  // Use the job's actual outputDir — for resumed jobs this points to the
  // previous job's directory, which is where the appended train.log lives.
  const logDir = job.outputDir || `${SHARED_STORAGE}/outputs/${job.id}`;
  const logPath = `${logDir}/train.log`;
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

finetuneRouter.get("/:id/checkpoints", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });
  if (!job.outputDir) return res.json([]);

  try {
    const { readdirSync, statSync } = await import("fs");
    const entries = readdirSync(job.outputDir, { withFileTypes: true });
    const checkpoints = entries
      .filter((e) => e.isDirectory() && e.name.startsWith("checkpoint-"))
      .map((e) => {
        const step = parseInt(e.name.replace("checkpoint-", ""), 10);
        const path = `${job.outputDir}/${e.name}`;
        let createdAt: string | undefined;
        try {
          createdAt = statSync(path).mtime.toISOString();
        } catch { /* ignore */ }
        return { step, name: e.name, path, createdAt };
      })
      .filter((c) => Number.isFinite(c.step))
      .sort((a, b) => b.step - a.step);
    res.json(checkpoints);
  } catch {
    res.json([]);
  }
});

finetuneRouter.get("/:id/metrics", async (req, res) => {
  const metrics = await prisma.trainingMetric.findMany({
    where: { jobId: req.params.id },
    orderBy: { step: "asc" },
    select: { step: true, loss: true, lr: true, evalLoss: true },
  });
  res.json(metrics);
});

finetuneRouter.post("/", async (req, res) => {
  const { nodeId, nodeIds, recipeFile, dataset, config, resumeFromJobId, displayName } = req.body;

  // Resume mode: inherit recipe/dataset/config/outputDir from the previous job
  // so HF Trainer can find the checkpoint-* dirs. The caller only needs to
  // pass nodeIds + resumeFromJobId; everything else is derived.
  let resumeJob: Awaited<ReturnType<typeof prisma.fineTuneJob.findUnique>> = null;
  if (resumeFromJobId) {
    resumeJob = await prisma.fineTuneJob.findUnique({ where: { id: resumeFromJobId } });
    if (!resumeJob) return res.status(404).json({ error: "resumeFromJobId not found" });
    if (!resumeJob.outputDir) return res.status(400).json({ error: "Previous job has no outputDir" });
  }

  const effectiveRecipeFile = recipeFile || resumeJob?.recipeFile;
  const effectiveDataset = dataset || resumeJob?.dataset;

  if ((!nodeId && !nodeIds) || !effectiveRecipeFile || !effectiveDataset) {
    return res.status(400).json({ error: "nodeId (or nodeIds), recipeFile, and dataset required (recipeFile/dataset can be inherited via resumeFromJobId)" });
  }

  // Look up recipe metadata from cached training recipes
  const agentHub: AgentHub = req.app.get("agentHub");
  const recipes = agentHub.getTrainingRecipes();
  const recipe = recipes.find((r) => r.file === effectiveRecipeFile);

  const baseModel = recipe?.base_model || effectiveRecipeFile;
  const method = recipe?.method || "lora";

  // Resolve nodes: single or multi-node
  const isMultiNode = Array.isArray(nodeIds) && nodeIds.length > 1;
  const headNodeId = isMultiNode ? nodeIds[0] : nodeId;

  // Merge config: previous job's config + new overrides on top
  const prevConfig = resumeJob?.config ? JSON.parse(resumeJob.config) : {};
  const mergedConfig = { ...prevConfig, ...(config || {}) };

  const job = await prisma.fineTuneJob.create({
    data: {
      nodeId: headNodeId,
      recipeFile: effectiveRecipeFile,
      baseModel,
      method,
      displayName: typeof displayName === "string" && displayName.trim() ? displayName.trim() : null,
      dataset: effectiveDataset,
      config: Object.keys(mergedConfig).length ? JSON.stringify(mergedConfig) : null,
      status: "pending",
    },
  });

  // Resume reuses the previous job's outputDir; fresh runs create their own
  const outputDir = resumeJob?.outputDir || `${SHARED_STORAGE}/outputs/${job.id}`;
  await prisma.fineTuneJob.update({
    where: { id: job.id },
    data: { outputDir },
  });

  // Resolve node IPs for multi-node + persist cluster membership.
  // Without persistence we'd lose the worker list after the start command
  // fires (only the head ends up on the FineTuneJob row), making it
  // impossible for the dashboard / API to show what nodes actually
  // participated in a given training run. Mirrors the ClusterNode pattern
  // used by Deployment.
  let clusterNodeIps: string[] | undefined;
  if (isMultiNode) {
    const nodes = await prisma.node.findMany({
      where: { id: { in: nodeIds } },
    });
    // Maintain order: head first, then workers
    const nodeMap = new Map(nodes.map((n) => [n.id, n.ipAddress]));
    clusterNodeIps = nodeIds.map((id: string) => nodeMap.get(id)!).filter(Boolean);

    await prisma.fineTuneClusterNode.createMany({
      data: nodeIds.map((id: string, idx: number) => ({
        jobId: job.id,
        nodeId: id,
        role: idx === 0 ? "head" : "worker",
      })),
    });
  }

  // Container path matches outputDir's basename, which differs from job.id when resuming
  const outputDirBasename = outputDir.split("/").pop()!;

  agentHub.sendToAgent(headNodeId, {
    type: "cmd:finetune:start",
    payload: {
      jobId: job.id,
      recipeFile: effectiveRecipeFile,
      dataset: effectiveDataset,
      outputDir: `/workspace/outputs/${outputDirBasename}`,
      config: mergedConfig,
      clusterNodeIps,
      resumeFromCheckpoint: !!resumeJob,
    },
  });

  await prisma.fineTuneJob.update({
    where: { id: job.id },
    data: { status: "starting", startedAt: new Date() },
  });

  const result = await prisma.fineTuneJob.findUnique({
    where: { id: job.id },
    include: {
      node: true,
      clusterNodes: { include: { node: true }, orderBy: { role: "asc" } },
    },
  });
  sseBroadcast({ type: "finetune:created", payload: result });
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

  const cleanFiles = req.query.cleanFiles === "true";
  let filesRemoved = false;
  let filesKept = false;
  let filesError: string | undefined;

  if (cleanFiles && job.outputDir) {
    // Don't nuke the directory if a resumed-child or sibling job still uses it
    const otherRefs = await prisma.fineTuneJob.count({
      where: { outputDir: job.outputDir, id: { not: job.id } },
    });
    if (otherRefs > 0) {
      filesKept = true;
    } else {
      try {
        const { rm } = await import("fs/promises");
        await rm(job.outputDir, { recursive: true, force: true });
        filesRemoved = true;
      } catch (err) {
        filesError = String(err);
      }
    }
  }

  await prisma.fineTuneJob.delete({ where: { id: req.params.id } });
  res.json({ deleted: true, filesRemoved, filesKept, filesError });
});

finetuneRouter.get("/:id/disk-usage", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });
  if (!job.outputDir) return res.json({ bytes: 0, dir: null, sharedWith: 0 });

  const sharedWith = await prisma.fineTuneJob.count({
    where: { outputDir: job.outputDir, id: { not: job.id } },
  });

  try {
    const { statSync, readdirSync } = await import("fs");
    let total = 0;
    function walk(p: string) {
      try {
        const st = statSync(p);
        if (st.isDirectory()) {
          for (const e of readdirSync(p)) walk(`${p}/${e}`);
        } else {
          total += st.size;
        }
      } catch { /* ignore unreadable */ }
    }
    walk(job.outputDir);
    res.json({ bytes: total, dir: job.outputDir, sharedWith });
  } catch {
    res.json({ bytes: 0, dir: job.outputDir, sharedWith });
  }
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
  const mergedOutputDir = `${SHARED_STORAGE}/outputs/${job.id}/merged`;

  // Recipe may specify a custom merge script (e.g. Qwen 3.6 needs a hand-
  // rolled merge because the generic PEFT path strips the multimodal
  // wrapper). Path is repo-relative; agent resolves it against the recipes
  // repo root. Falls back to the generic scripts/merge.py.
  const recipe = job.recipeFile
    ? agentHub.getTrainingRecipes().find((r) => r.file === job.recipeFile)
    : undefined;
  const mergeScript = recipe?.scripts.merge || "scripts/merge.py";

  agentHub.sendToAgent(job.nodeId, {
    type: "cmd:finetune:merge",
    payload: {
      jobId: job.id,
      baseModel: job.baseModel,
      adapterPath: job.outputDir ? `${job.outputDir}/lora_adapter` : job.outputPath!,
      mergedOutputDir,
      mergeScript,
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
  sseBroadcast({ type: "deployment:created", payload: result });
  res.status(201).json(result);
});
