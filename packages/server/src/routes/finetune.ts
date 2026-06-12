import { Router } from "express";
import { readFileSync, existsSync, unlinkSync, readdirSync } from "fs";
import { join } from "path";
import { prisma } from "../prisma.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";
import type { AgentHub } from "../ws/agent-hub.js";
import { checkVllmVramAdmission, vramShortfallMessage } from "../admission/vram.js";
import { normalizeDisplayName, validateDisplayNameUnique, DisplayNameError } from "../deployments/display-name.js";

// vLLM deploy infrastructure lives in a separate repo, mounted via NFS at
// the same path as on each agent. The auto-generated finetune-<jobId-12>.yaml
// files end up here whenever cmd:finetune:deploy fires on the head agent.
const VLLM_REPO_PATH = `${SHARED_STORAGE}/src/github/spark-vllm-docker`;

/**
 * Path to the auto-generated vLLM recipe YAML for a given FineTuneJob.
 * Must match `generateLocalModelRecipe` in packages/agent/src/runtime/vllm.ts.
 */
function generatedRecipePath(jobId: string): string {
  return join(VLLM_REPO_PATH, "recipes", `finetune-${jobId.slice(0, 12)}.yaml`);
}

/**
 * Broadcasts a recipe-rescan to every connected agent. Agents cache the
 * recipe list in memory and only resend on rescan, so after we delete YAMLs
 * server-side the dashboard's `/api/recipes` keeps showing them until the
 * next rescan. Call this from any handler that mutates the recipes dir.
 */
function broadcastRecipeRescan(agentHub: AgentHub): number {
  const nodeIds = agentHub.getConnectedNodeIds();
  for (const nodeId of nodeIds) {
    agentHub.sendToAgent(nodeId, { type: "cmd:rescan-recipes", payload: {} });
  }
  return nodeIds.length;
}

/** Variant ids are filename slugs — lowercase alphanumerics + hyphens, must
 *  start with an alphanumeric, max ~32 chars. Tight enough to keep `..` and
 *  path separators out of the filename interpolation in
 *  `inferenceFilenameForId`. */
const VARIANT_SLUG_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;

export function isValidVariantSlug(v: unknown): v is string {
  return typeof v === "string" && VARIANT_SLUG_RE.test(v);
}

export const finetuneRouter = Router();

/**
 * @openapi
 * /api/finetune:
 *   get:
 *     tags: [Fine-tune]
 *     summary: List all fine-tune jobs
 *     description: >
 *       Returns every FineTuneJob record ordered by creation date descending, each
 *       including its head Node and all ClusterNode members (ordered by role so head
 *       comes first). Status lifecycle: pending → starting → running → completed →
 *       (mergeStatus: idle → running → completed) → deployable. Use SSE
 *       (`finetune:status`, `finetune:created`) for real-time updates.
 *     responses:
 *       '200':
 *         description: Array of fine-tune job objects
 */
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

/**
 * @openapi
 * /api/finetune/{id}:
 *   get:
 *     tags: [Fine-tune]
 *     summary: Get a single fine-tune job
 *     description: >
 *       Returns the FineTuneJob record with node and cluster membership included.
 *       Useful for polling job status after creation or for displaying job details.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: Fine-tune job object
 *       '404':
 *         description: Job not found
 */
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

/**
 * @openapi
 * /api/finetune/{id}/logs:
 *   get:
 *     tags: [Fine-tune]
 *     summary: Get training log for a fine-tune job
 *     description: >
 *       Reads `$outputDir/train.log` from shared storage and returns it as
 *       `text/plain`. For resumed jobs the outputDir points to the original job's
 *       directory (where the appended log lives). Returns empty body if the log
 *       file doesn't exist yet. Supports `?tail=N` to return only the last N lines,
 *       useful for keeping the dashboard log pane current without reading the full file.
 *       Prefer tailing this file directly from shared storage for high-frequency polling.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *       - in: query
 *         name: tail
 *         required: false
 *         schema: { type: integer }
 *         description: Return only the last N lines if > 0
 *     responses:
 *       '200':
 *         description: Log content as text/plain
 *         content:
 *           text/plain:
 *             schema: { type: string }
 *       '404':
 *         description: Job not found
 */
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

/**
 * @openapi
 * /api/finetune/{id}/checkpoints:
 *   get:
 *     tags: [Fine-tune]
 *     summary: List saved checkpoints for a fine-tune job
 *     description: >
 *       Scans the job's outputDir on shared storage for `checkpoint-N` subdirectories
 *       created by HuggingFace Trainer and returns them sorted by step (newest first).
 *       Each entry includes `{ step, name, path, createdAt }`. Returns an empty array
 *       if no outputDir is set or the directory doesn't exist. Used by the resume-
 *       job flow to let users pick which checkpoint to continue from.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: Array of checkpoint objects sorted by step descending
 *       '404':
 *         description: Job not found
 */
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

/**
 * @openapi
 * /api/finetune/{id}/metrics:
 *   get:
 *     tags: [Fine-tune]
 *     summary: Get training metrics (loss curve) for a fine-tune job
 *     description: >
 *       Returns all TrainingMetric rows for the job ordered by step ascending.
 *       Each row has `{ step, loss, lr, evalLoss }`. The agent appends a row
 *       whenever a training step completes. Used by the dashboard loss-curve chart.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: Array of training metric objects
 */
finetuneRouter.get("/:id/metrics", async (req, res) => {
  const metrics = await prisma.trainingMetric.findMany({
    where: { jobId: req.params.id },
    orderBy: { step: "asc" },
    select: { step: true, loss: true, lr: true, evalLoss: true },
  });
  res.json(metrics);
});

/**
 * @openapi
 * /api/finetune/{id}:
 *   patch:
 *     tags: [Fine-tune]
 *     summary: Rename a fine-tune job
 *     description: >
 *       Updates `displayName` on the FineTuneJob row and atomically renames the linked
 *       Model row to match (so the inference catalog stays in sync). Empty or whitespace-
 *       only values clear the name back to null (dashboard falls back to the id-derived
 *       label `finetune-<8chars>`). Returns 409 if the requested model name is already
 *       taken by another model record.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               displayName: { type: string, nullable: true, description: "New display name; null or empty string clears it." }
 *     responses:
 *       '200':
 *         description: Updated fine-tune job object
 *       '400':
 *         description: No allowed fields provided or bad displayName type
 *       '404':
 *         description: Job not found
 *       '409':
 *         description: Model name already taken
 */
// PATCH /:id — mutate user-editable fields on the job. Currently scoped to
// displayName; extend the allowed fields here as future rename-able
// attributes are added. Trims whitespace; empty/whitespace-only strings
// clear the name (back to null → dashboard falls back to derived label).
finetuneRouter.patch("/:id", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });

  const updates: { displayName?: string | null } = {};
  if ("displayName" in req.body) {
    const raw = req.body.displayName;
    if (raw === null) {
      updates.displayName = null;
    } else if (typeof raw === "string") {
      const trimmed = raw.trim();
      updates.displayName = trimmed.length ? trimmed : null;
    } else {
      return res.status(400).json({ error: "displayName must be a string or null" });
    }
  }

  if (Object.keys(updates).length === 0) {
    return res.status(400).json({ error: "no allowed fields provided" });
  }

  // Apply both writes (FineTuneJob.displayName and the linked Model.name)
  // atomically. Without a transaction, a P2002 collision on Model.name
  // would leave FineTuneJob.displayName updated but the Model rename
  // rolled back — a half-applied rename that's hard to reason about.
  let result;
  try {
    result = await prisma.$transaction(async (tx) => {
      const job = await tx.fineTuneJob.update({
        where: { id: req.params.id },
        data: updates,
        include: {
          node: true,
          clusterNodes: { include: { node: true }, orderBy: { role: "asc" } },
          model: true,
        },
      });

      if ("displayName" in updates && job.model) {
        const stableName = `finetune-${job.id.slice(0, 8)}`;
        const modelName = updates.displayName || stableName;
        if (job.model.name !== modelName) {
          await tx.model.update({
            where: { id: job.model.id },
            data: { name: modelName },
          });
          job.model = { ...job.model, name: modelName };
        }
      }

      return job;
    });
  } catch (e: unknown) {
    // P2002 = unique-constraint violation on Model.name. Both writes
    // rolled back; return 409 so the user can pick a different name.
    if (typeof e === "object" && e !== null && "code" in e && (e as { code: unknown }).code === "P2002") {
      return res.status(409).json({
        error: `A model with the requested name already exists. Choose a different displayName.`,
      });
    }
    throw e;
  }

  sseBroadcast({ type: "finetune:updated", payload: result });
  res.json(result);
});

/**
 * @openapi
 * /api/finetune:
 *   post:
 *     tags: [Fine-tune]
 *     summary: Start a new fine-tune training job
 *     description: >
 *       Creates a FineTuneJob record and sends `cmd:finetune:start` to the head-node agent.
 *       The agent runs HuggingFace Trainer inside a container, writing checkpoints and
 *       training metrics to shared storage under `$SHARED_STORAGE/outputs/{jobId}`.
 *       Supports multi-node training via `nodeIds[]` (ClusterNode rows track participation).
 *       Resume mode: pass `resumeFromJobId` to inherit the previous job's recipe/dataset/
 *       outputDir and continue from the latest checkpoint. Logs stream to shared storage
 *       and can be tailed via GET /api/finetune/{id}/logs.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               nodeId: { type: string, description: "Single head-node ID for solo training." }
 *               nodeIds: { type: array, items: { type: string }, description: "Multi-node: head first, then workers." }
 *               recipeFile: { type: string, description: "Training recipe ref from GET /api/training-recipes." }
 *               dataset: { type: string, description: "Dataset identifier (matches a Dataset.name or HuggingFace repo)." }
 *               config: { type: object, description: "Training config overrides (epochs, lr, batch_size, etc.)." }
 *               resumeFromJobId: { type: string, description: "ID of a previous completed job to resume from its latest checkpoint." }
 *               displayName: { type: string, description: "Optional human-readable label for the job." }
 *     responses:
 *       '201':
 *         description: Created fine-tune job with node and cluster members included
 *       '400':
 *         description: Missing required fields or invalid resumeFromJobId
 *       '404':
 *         description: resumeFromJobId not found
 */
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

/**
 * @openapi
 * /api/finetune/{id}:
 *   delete:
 *     tags: [Fine-tune]
 *     summary: Delete a fine-tune job and optionally its output files
 *     description: >
 *       Removes the FineTuneJob record, its linked Model (if no active deployments
 *       reference it), and the auto-generated vLLM recipe YAML (if present). If
 *       the job is still running, sends `cmd:finetune:stop` first. Returns 409 if
 *       active (non-terminal) deployments still reference the linked model — stop
 *       those first. Pass `?cleanFiles=true` to also recursively delete the
 *       outputDir on shared storage; files are kept if another resumed job shares
 *       the same directory.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *       - in: query
 *         name: cleanFiles
 *         required: false
 *         schema: { type: string, enum: ['true', 'false'] }
 *         description: If "true", also delete the output directory on shared storage
 *     responses:
 *       '200':
 *         description: '{ deleted, filesRemoved, filesKept, filesError, recipeRemoved }'
 *       '404':
 *         description: Job not found
 *       '409':
 *         description: Active deployments reference the linked model
 */
finetuneRouter.delete("/:id", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });

  // Guard against deleting when the fine-tune's Model has active deployments.
  //
  // "Active" = anything that isn't a terminal status. In particular, `pending`
  // is active because the manager has already dispatched `cmd:deploy` to the
  // agent — the container is being spun up and the agent expects the
  // Deployment row to remain so it can report status back. Sweeping a
  // pending row would orphan the agent-managed container.
  //
  // Terminal statuses (stopped/failed/removed) are safe to sweep because
  // the container/process is gone. We explicitly delete those rows before
  // the FineTuneJob delete so the FK Cascade chain (FineTuneJob → Model)
  // can proceed (Deployment.modelId has no cascade of its own).
  //
  // The guard returns 409 with deployment IDs so the user knows which to
  // stop first.
  const TERMINAL_STATUSES = ["stopped", "failed", "removed"];
  const linkedModel = await prisma.model.findUnique({
    where: { finetuneJobId: job.id },
    include: { deployments: true },
  });
  if (linkedModel) {
    const active = linkedModel.deployments.filter(
      (d) => !TERMINAL_STATUSES.includes(d.status),
    );
    if (active.length > 0) {
      return res.status(409).json({
        error: `Cannot delete: ${active.length} active deployment(s) reference this model. ` +
               `Stop the deployment(s) first.`,
        deploymentIds: active.map((d) => d.id),
      });
    }
    // Sweep terminal deployments so the FineTuneJob → Model cascade can
    // proceed (Deployment.modelId has no cascade of its own).
    if (linkedModel.deployments.length > 0) {
      await prisma.deployment.deleteMany({
        where: { id: { in: linkedModel.deployments.map((d) => d.id) } },
      });
    }
  }

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

  // Best-effort cleanup of the auto-generated vLLM recipe YAML so it
  // doesn't linger in the dashboard's recipe dropdown after the job is
  // gone. The file may not exist if the job was never deployed — that's
  // fine, ENOENT is silently ignored.
  let recipeRemoved = false;
  try {
    unlinkSync(generatedRecipePath(job.id));
    recipeRemoved = true;
  } catch (err: unknown) {
    if (typeof err === "object" && err !== null && "code" in err && (err as { code: unknown }).code !== "ENOENT") {
      console.error(`[finetune.delete] failed to remove recipe YAML: ${err}`);
    }
  }
  if (recipeRemoved) {
    broadcastRecipeRescan(req.app.get("agentHub") as AgentHub);
  }

  res.json({ deleted: true, filesRemoved, filesKept, filesError, recipeRemoved });
});

/**
 * @openapi
 * /api/finetune/{id}/disk-usage:
 *   get:
 *     tags: [Fine-tune]
 *     summary: Report disk usage of a fine-tune job's output directory
 *     description: >
 *       Recursively walks the job's outputDir on shared storage and returns the total
 *       byte count, the directory path, and how many other jobs share the same
 *       directory (resumed jobs share their parent's outputDir). Returns
 *       `{ bytes: 0, dir: null, sharedWith: 0 }` if no outputDir is set.
 *       Used by the dashboard to show "delete with files" confirmation dialogs.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: '{ bytes, dir, sharedWith }'
 *       '404':
 *         description: Job not found
 */
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

/**
 * @openapi
 * /api/finetune/cleanup-orphan-models:
 *   post:
 *     tags: [Fine-tune]
 *     summary: Back-link or delete orphaned finetune Model rows
 *     description: >
 *       One-shot maintenance operation. Walks every Model row whose name matches
 *       the legacy `finetune-<8alphanum>` pattern and has no `finetuneJobId` FK.
 *       For each: if a matching FineTuneJob exists (by 8-char prefix) the FK is
 *       back-linked; if no job exists and no active Deployment references the model,
 *       the Model row is deleted. Models with active deployments are kept. Returns
 *       counts: `{ backlinked, deleted, kept_due_to_deployment }`. Idempotent — safe
 *       to re-run after fixing issues.
 *     responses:
 *       '200':
 *         description: '{ backlinked, deleted, kept_due_to_deployment }'
 *       '500':
 *         description: Partial failure — returns counts accumulated before the error
 */
// POST /cleanup-orphan-models — one-shot maintenance op. Walks every Model
// row whose name matches the legacy "finetune-<8alphanum>" pattern AND has
// finetuneJobId = NULL, then either:
//   - back-links the FK if the prefix matches an existing FineTuneJob, OR
//   - deletes the row if no matching job AND no active Deployment uses it.
//
// "Active" here = the same TERMINAL_STATUSES list used elsewhere in DELETE:
// any deployment whose status is NOT in ["stopped", "failed", "removed"]
// counts as active and protects the Model from deletion. The user must
// stop those deployments first, then re-run cleanup.
//
// Returns counts: { backlinked, deleted, kept_due_to_deployment }.
finetuneRouter.post("/cleanup-orphan-models", async (_req, res) => {
  const legacyPattern = /^finetune-([0-9a-z]{8})$/;
  const TERMINAL_STATUSES = ["stopped", "failed", "removed"];

  const candidates = await prisma.model.findMany({
    where: { finetuneJobId: null, name: { startsWith: "finetune-" } },
    include: { deployments: true },
  });

  let backlinked = 0;
  let deleted = 0;
  let kept_due_to_deployment = 0;

  try {
    for (const m of candidates) {
      const match = legacyPattern.exec(m.name);
      if (!match) continue;
      const prefix = match[1];

      // Find a job whose id starts with this prefix. cuid v2 IDs are unique
      // across the first 8 chars in practice; if a collision happens, we
      // back-link to the first match found.
      const job = await prisma.fineTuneJob.findFirst({
        where: { id: { startsWith: prefix } },
      });

      if (job) {
        // P2002 is theoretically possible here if two candidate Model rows
        // share the same 8-char id prefix AND map to the same FineTuneJob
        // (the @unique on Model.finetuneJobId enforces one Model per job).
        // cuid v2 makes a real collision astronomically unlikely; if it
        // happens, the outer try/catch surfaces it with partial counts.
        await prisma.model.update({
          where: { id: m.id },
          data: { finetuneJobId: job.id },
        });
        backlinked++;
        continue;
      }

      const active = m.deployments.filter(
        (d) => !TERMINAL_STATUSES.includes(d.status),
      );
      if (active.length > 0) {
        kept_due_to_deployment++;
        continue;
      }

      await prisma.model.delete({ where: { id: m.id } });
      deleted++;
    }
  } catch (e: unknown) {
    // Surface the failure with the partial counts so the caller knows how
    // far the cleanup got. Re-running is idempotent (back-linked rows no
    // longer match the filter; deleted rows are gone) so the caller can
    // retry after fixing the underlying issue.
    const message = e instanceof Error ? e.message : String(e);
    return res.status(500).json({
      error: `cleanup-orphan-models failed mid-loop: ${message}`,
      backlinked,
      deleted,
      kept_due_to_deployment,
    });
  }

  res.json({ backlinked, deleted, kept_due_to_deployment });
});

/**
 * @openapi
 * /api/finetune/cleanup-orphan-recipes:
 *   post:
 *     tags: [Fine-tune]
 *     summary: Delete auto-generated vLLM recipe YAMLs for deleted fine-tune jobs
 *     description: >
 *       One-shot maintenance operation. Scans the vLLM recipe repository's `recipes/`
 *       directory for files matching the auto-generated `finetune-<12alphanum>.yaml`
 *       pattern and removes those whose job-id prefix doesn't correspond to a live
 *       FineTuneJob. Non-auto-generated (hand-curated) files are left alone. Triggers
 *       a recipe rescan on all connected agents if any files were deleted. Returns
 *       `{ scanned, deleted, kept_live, kept_unparseable, removed, agents_refreshed }`.
 *     responses:
 *       '200':
 *         description: Cleanup summary
 *       '500':
 *         description: recipes directory unreadable
 */
// POST /cleanup-orphan-recipes — one-shot maintenance op. Walks the vLLM
// deploy repo's recipes dir for files matching the auto-generated
// `finetune-<12alphanum>.yaml` pattern, and removes any whose id-prefix
// doesn't correspond to a live FineTuneJob. Pairs with the DELETE-route
// cleanup that handles future deletions automatically.
//
// Returns { scanned, deleted, kept_live, kept_unparseable }.
finetuneRouter.post("/cleanup-orphan-recipes", async (req, res) => {
  const generatedPattern = /^finetune-([0-9a-z]{12})\.yaml$/;
  const recipesDir = join(VLLM_REPO_PATH, "recipes");

  let entries: string[];
  try {
    entries = readdirSync(recipesDir);
  } catch (err: unknown) {
    return res.status(500).json({ error: `recipes dir unreadable: ${String(err)}` });
  }

  const liveJobs = await prisma.fineTuneJob.findMany({ select: { id: true } });
  const livePrefixes = new Set(liveJobs.map((j) => j.id.slice(0, 12)));

  let scanned = 0;
  let deleted = 0;
  let kept_live = 0;
  let kept_unparseable = 0;
  const removed: string[] = [];

  for (const name of entries) {
    const m = generatedPattern.exec(name);
    if (!m) {
      // Not an auto-generated file (hand-curated recipe, e.g.
      // finetune-qwen3.6-50step.yaml). Leave alone.
      kept_unparseable++;
      continue;
    }
    scanned++;
    const prefix = m[1];
    if (livePrefixes.has(prefix)) {
      kept_live++;
      continue;
    }
    try {
      unlinkSync(join(recipesDir, name));
      deleted++;
      removed.push(name);
    } catch (err) {
      console.error(`[cleanup-orphan-recipes] failed to remove ${name}: ${err}`);
    }
  }

  let agents_refreshed = 0;
  if (deleted > 0) {
    agents_refreshed = broadcastRecipeRescan(req.app.get("agentHub") as AgentHub);
  }

  res.json({ scanned, deleted, kept_live, kept_unparseable, removed, agents_refreshed });
});

/**
 * @openapi
 * /api/finetune/{id}/stop:
 *   post:
 *     tags: [Fine-tune]
 *     summary: Stop a running fine-tune job
 *     description: >
 *       Sends `cmd:finetune:stop` to the head-node agent and sets the job status to
 *       `stopping`. The agent gracefully terminates the training process; the final
 *       checkpoint on disk is preserved. After stopping, the job can be resumed via
 *       POST /api/finetune with `resumeFromJobId`.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: '{ status: "stopping" }'
 *       '404':
 *         description: Job not found
 */
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

/**
 * @openapi
 * /api/finetune/{id}/merge:
 *   post:
 *     tags: [Fine-tune]
 *     summary: Merge LoRA adapter weights into the base model
 *     description: >
 *       Sends `cmd:finetune:merge` to the head-node agent. The agent runs the recipe's
 *       `scripts.merge` Python script (or the generic `scripts/merge.py` fallback),
 *       which merges the LoRA adapter stored at `{outputDir}/lora_adapter` into the
 *       base model weights and writes the result to `{outputDir}/merged`. The job must
 *       be in `completed` status. After merging, call POST /api/finetune/{id}/deploy
 *       to serve the merged model via vLLM.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: '{ status: "merging", mergedOutputDir }'
 *       '400':
 *         description: Job not completed
 *       '404':
 *         description: Job not found
 */
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

/**
 * @openapi
 * /api/finetune/{id}/quantize:
 *   post:
 *     tags: [Fine-tune]
 *     summary: Quantize a merged fine-tune model to FP8
 *     description: >
 *       Sends `cmd:finetune:quantize` to the head-node agent. The agent runs the
 *       recipe's `scripts.quantize_fp8` script to produce an FP8-quantized copy of
 *       the merged model at `{outputDir}/merged-fp8`. Requires the merge to be
 *       complete (`mergeStatus: "completed"`) and the recipe to declare
 *       `scripts.quantize_fp8`. Idempotent if already quantized. Returns 409 if
 *       quantization is already in progress. Note: on-load FP8 conversion via vLLM's
 *       `--quantization fp8` is an alternative that avoids this offline step.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: '{ status: "quantizing", quantizedPath } or { status: "quantized", quantizedPath } if already done'
 *       '400':
 *         description: Merge not complete
 *       '404':
 *         description: Job not found
 *       '409':
 *         description: Quantization already in progress
 *       '501':
 *         description: Recipe does not support quantization
 */
finetuneRouter.post("/:id/quantize", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });

  // Pre-flight: merge must be done.
  if (job.mergeStatus !== "completed" || !job.mergedPath) {
    return res.status(400).json({ error: "Job must be merged before quantizing. Call POST /merge first." });
  }

  // Recipe must support quantization. Mirrors how scripts.merge is required
  // for the merge endpoint.
  const agentHub: AgentHub = req.app.get("agentHub");
  const recipe = job.recipeFile
    ? agentHub.getTrainingRecipes().find((r) => r.file === job.recipeFile)
    : undefined;
  const quantizeScript = recipe?.scripts.quantize_fp8;
  if (!quantizeScript) {
    return res.status(501).json({
      error: `Recipe ${job.recipeFile} does not declare scripts.quantize_fp8 — quantization not supported for this recipe.`,
    });
  }

  // Idempotency: already quantized → return existing artifact.
  if (job.quantizationStatus === "quantized" && job.quantizedPath) {
    return res.json({ status: "quantized", quantizedPath: job.quantizedPath });
  }

  // In-flight: refuse to re-kick.
  if (job.quantizationStatus === "quantizing") {
    return res.status(409).json({ error: "Quantization already in progress for this job." });
  }

  const quantizedOutputDir = `${job.outputDir ?? `${SHARED_STORAGE}/outputs/${job.id}`}/merged-fp8`;

  agentHub.sendToAgent(job.nodeId, {
    type: "cmd:finetune:quantize",
    payload: {
      jobId: job.id,
      mergedPath: job.mergedPath,
      quantizedOutputDir,
      quantizeScript,
    },
  });

  await prisma.fineTuneJob.update({
    where: { id: job.id },
    data: { quantizationStatus: "quantizing", quantizedPath: quantizedOutputDir },
  });

  res.json({ status: "quantizing", quantizedPath: quantizedOutputDir });
});

/**
 * @openapi
 * /api/finetune/{id}/deploy:
 *   post:
 *     tags: [Fine-tune]
 *     summary: Deploy a merged fine-tuned model for inference
 *     description: >
 *       Creates a Deployment record and sends `cmd:finetune:deploy` to the head-node
 *       agent. The agent launches a vLLM container serving the merged BF16 or FP8
 *       weights. The training recipe's `deploy` block provides container and GPU
 *       defaults; caller overrides win via `config`. Runs VRAM admission. The
 *       `artifactVariant` field selects which artifact to serve: `"default"` (BF16
 *       merged path) or `"fp8"` (vLLM on-load FP8 conversion). Multi-node deploy via
 *       `nodeIds[]`. An optional `displayName` sets vLLM's `--served-model-name`
 *       for this specific deployment without changing the FT job's catalog identity.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               nodeId: { type: string, description: "Single-node target." }
 *               nodeIds: { type: array, items: { type: string }, description: "Multi-node: head first." }
 *               config: { type: object, description: "Inference overrides: gpuMem, maxModelLen, port…" }
 *               artifactVariant: { type: string, description: "Which artifact to serve: 'default' (BF16) or 'fp8'. Default: 'default'." }
 *               displayName: { type: string, nullable: true, description: "Per-deploy vLLM served-model-name override. Must be unique." }
 *     responses:
 *       '201':
 *         description: Created deployment record
 *       '400':
 *         description: Merge not complete, missing nodeId, or invalid artifactVariant
 *       '404':
 *         description: Job not found
 *       '409':
 *         description: VRAM shortfall or displayName conflict
 */
finetuneRouter.post("/:id/deploy", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({
    where: { id: req.params.id },
    include: { node: true },
  });
  if (!job) return res.status(404).json({ error: "Job not found" });

  const { nodeId, nodeIds, config, artifactVariant, displayName: rawDisplayName } = req.body as {
    nodeId?: string;
    nodeIds?: string[];
    config?: Record<string, unknown>;
    artifactVariant?: string;        // was: "bf16" | "fp8"
    displayName?: string | null;
  };
  const isCluster = Array.isArray(nodeIds) && nodeIds.length > 1;
  const headNodeId = isCluster ? nodeIds[0] : (nodeId || job.nodeId);
  if (!headNodeId) {
    return res.status(400).json({ error: "nodeId or nodeIds required" });
  }

  if (artifactVariant !== undefined && !isValidVariantSlug(artifactVariant)) {
    return res.status(400).json({
      error: `Invalid artifactVariant: must match ${VARIANT_SLUG_RE} (got ${JSON.stringify(artifactVariant)})`,
    });
  }

  const variant: string = artifactVariant ?? "default";

  // Both variants serve the same BF16 merged weights. FP8 deploys use vLLM's
  // on-load `--quantization fp8` (set in inference-fp8.yaml) to convert weights
  // to FP8 at model-load time. We do NOT use llmcompressor's offline quantize:
  // its hard transformers<=4.57.6 pin conflicts with newer model architectures
  // (e.g. qwen3_5) that require transformers>=5.0, and on-load conversion gives
  // us the same serving footprint without the conflict.
  const modelPath = job.mergedPath || (job.outputDir ? `${job.outputDir}/merged` : null);
  if (!modelPath || job.mergeStatus !== "completed") {
    return res.status(400).json({
      error: "Model must be merged before deployment. Call POST /merge first.",
    });
  }

  // Look up the training recipe's deploy config for container/defaults
  const agentHub: AgentHub = req.app.get("agentHub");
  const trainingRecipe = job.recipeFile
    ? agentHub.getTrainingRecipes().find((r) => r.file === job.recipeFile)
    : undefined;
  const deployConfig = trainingRecipe?.deploy;

  // Pre-flight VRAM admission, same as normal vLLM deploys. The gpuMem
  // ceiling either comes from the user's launch override or the training
  // recipe's deploy.gpu_memory_utilization default (set when the recipe
  // was authored). 0.85 fallback matches the normal-deploy default.
  const gpuMemForAdmission =
    (config?.gpuMem as number) ||
    (deployConfig?.gpu_memory_utilization as number) ||
    0.85;
  const admissionNodeIds = isCluster ? (nodeIds as string[]) : [headNodeId];
  const shortfalls = await checkVllmVramAdmission(admissionNodeIds, gpuMemForAdmission);
  if (shortfalls.length > 0) {
    return res.status(409).json({
      error: `Not enough VRAM on ${shortfalls.length} of ${admissionNodeIds.length} node(s): ${vramShortfallMessage(shortfalls)}`,
      shortfalls,
      gpuMemoryUtilization: gpuMemForAdmission,
    });
  }

  // Create a deployment record
  // Deployable Model name: prefer the user-set displayName, fall back to a
  // stable id-derived label. The finetuneJobId FK is what makes the row
  // get cleaned up automatically when the job is deleted (onDelete: Cascade
  // in schema.prisma). displayName is pre-normalized to null-or-trimmed by
  // POST/PATCH, so we don't trim again here.
  const stableName = `finetune-${job.id.slice(0, 8)}`;
  const ftModelName = job.displayName || stableName;

  // Optional per-deploy displayName override. When set it does NOT touch
  // Model.name (the FT's catalog identity stays stable); it only overrides
  // what vLLM publishes via --served-model-name AND what the dashboard
  // shows in the deployments list. Lets the same FT be deployed twice
  // under different served names (e.g. "chat3d-prod" + "chat3d-canary").
  // Note: when no override is supplied, ftModelName is passed as modelName
  // but NOT stored in Deployment.displayName, so the uniqueness constraint
  // here does not apply. A vLLM name collision is theoretically possible if
  // two FT jobs share a displayName — that's the user's responsibility,
  // guarded at the Model.name level by Prisma's P2002 unique-constraint check
  // (see the upsert below), not here.
  let perDeployDisplayName: string | null;
  try {
    perDeployDisplayName = normalizeDisplayName(rawDisplayName);
  } catch (e) {
    if (e instanceof DisplayNameError) return res.status(400).json({ error: e.message });
    throw e;
  }
  if (perDeployDisplayName) {
    const conflict = await validateDisplayNameUnique(prisma, perDeployDisplayName);
    if (conflict) {
      return res.status(409).json({
        error: `Display name "${perDeployDisplayName}" is already in use by deployment ${conflict.conflictId}.`,
        conflict,
      });
    }
  }

  // What vLLM ultimately publishes. Per-deploy override wins; otherwise
  // fall back to the fine-tune's own displayName / stable name.
  const servedModelName = perDeployDisplayName || ftModelName;

  let model;
  try {
    model = await prisma.model.upsert({
      where: { finetuneJobId: job.id },
      create: { name: ftModelName, runtime: "vllm", finetuneJobId: job.id },
      update: { name: ftModelName },
    });
  } catch (e: unknown) {
    // P2002 = Prisma unique constraint violation. Only Model.name is unique
    // among the fields we set here, so this fires when another Model row
    // (different job or hand-created) already has this name. Surface as
    // 409 so the user can rename the job and try again.
    if (typeof e === "object" && e !== null && "code" in e && (e as { code: unknown }).code === "P2002") {
      return res.status(409).json({
        error: `A model named "${ftModelName}" already exists. Rename this fine-tune (PATCH /api/finetune/${job.id}) and try again.`,
      });
    }
    throw e;
  }

  const deployment = await prisma.deployment.create({
    data: {
      nodeId: headNodeId,
      modelId: model.id,
      status: "pending",
      clusterMode: isCluster,
      displayName: perDeployDisplayName,
      // Persist artifactVariant alongside the deploy config so the restart
      // route can dispatch cmd:finetune:deploy with the original variant
      // (bf16 vs fp8). Without this, restarts default to bf16 even for
      // deployments originally launched as fp8.
      config: JSON.stringify({ ...config, localModelPath: modelPath, artifactVariant: variant }),
    },
  });

  // For multi-node deploys, persist cluster membership the same way normal
  // deploys do — one ClusterNode row per participant, head first. The agent
  // gets the node IP list separately (see clusterNodeIps/Fast below).
  let clusterNodeIps: string[] | undefined;
  let clusterNodeFastIps: (string | null)[] | undefined;
  if (isCluster) {
    const nodes = await prisma.node.findMany({ where: { id: { in: nodeIds } } });
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));
    for (let i = 0; i < nodeIds.length; i++) {
      await prisma.clusterNode.create({
        data: {
          deploymentId: deployment.id,
          nodeId: nodeIds[i],
          role: i === 0 ? "head" : "worker",
          status: "pending",
        },
      });
    }
    clusterNodeIps = nodeIds.map((id: string) => nodeMap.get(id)?.ipAddress).filter((ip): ip is string => Boolean(ip));
    clusterNodeFastIps = nodeIds.map((id: string) => nodeMap.get(id)?.fastIpAddress ?? null);
  }

  await prisma.fineTuneJob.update({
    where: { id: job.id },
    data: { deploymentId: deployment.id },
  });

  agentHub.sendToAgent(headNodeId, {
    type: "cmd:finetune:deploy",
    payload: {
      jobId: job.id,
      deploymentId: deployment.id,
      modelPath,
      baseModel: job.baseModel,
      deployContainer: deployConfig?.container || "vllm-node",
      // Per-deploy override wins; otherwise Model.name (FT's stable name).
      // Agent threads this into vLLM's --served-model-name.
      modelName: servedModelName,
      // Relative path of the training recipe (e.g.
      // "recipes/qwen3.6-27b-base-lora-attn-mlp"). The agent resolves this
      // to an absolute dir and looks for a sibling inference.yaml (bf16) or
      // inference-fp8.yaml (fp8) to use as the vLLM serve template.
      // If absent, deploy falls back to the legacy minimal auto-gen.
      recipeFile: job.recipeFile,
      // Which artifact variant to serve: bf16 uses the merged path,
      // fp8 uses the quantized path. Drives template selection on the agent.
      artifactVariant: variant,
      // clusterNodes / clusterNodeFastIps are undefined for solo deploys and
      // set to arrays for multi-node — computed above from Task 3's nodeIds[].
      clusterNodes: clusterNodeIps,
      clusterNodeFastIps,
      config: {
        // Recipe defaults first, user overrides win via the spread.
        gpuMem: deployConfig?.gpu_memory_utilization,
        maxModelLen: deployConfig?.max_model_len,
        ...config,
      },
    },
  });

  const result = await prisma.deployment.findUnique({
    where: { id: deployment.id },
    include: {
      node: true,
      model: true,
      // Without clusterNodes in the include, the live SSE event arrives
      // with no cluster info and the dashboard card shows only the head
      // until the user reloads. Matches the normal-deploy route's shape.
      clusterNodes: { include: { node: true } },
    },
  });
  sseBroadcast({ type: "deployment:created", payload: result });
  res.status(201).json(result);
});
