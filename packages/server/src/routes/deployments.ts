import { Router } from "express";
import { readFileSync, existsSync } from "fs";
import { prisma } from "../prisma.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";
import type { AgentHub } from "../ws/agent-hub.js";
import { checkVllmVramAdmission, vramShortfallMessage } from "../admission/vram.js";
import { checkRecipeArchAdmission, recipeArchMismatchMessage } from "../admission/recipe-arch.js";
import { readCatalog as readOllamaCatalog } from "../ollama/catalog-store.js";
import { ollamaVramEstimateMB } from "../ollama/vram-estimate.js";
import { normalizeDisplayName, validateDisplayNameUnique, DisplayNameError } from "../deployments/display-name.js";
import { isValidVariantSlug } from "./finetune.js";
import { resolveRecipePath } from "../deployments/recipe-path.js";
import { validateInlineRecipe, parseInlineRecipeModel } from "../deployments/recipe-inline.js";

export const deploymentsRouter = Router();

/**
 * @openapi
 * /api/deployments:
 *   get:
 *     tags: [Deployments]
 *     summary: List all deployments
 *     description: >
 *       Returns every Deployment record ordered by creation date descending, each
 *       including its linked Node, Model (with finetuneJob.recipeFile for the edit-
 *       restart form), and ClusterNode membership. Status lifecycle: pending →
 *       starting/building/downloading/launching/loading → running → (removing →)
 *       stopped/failed. Use SSE (`deployment:created`, `deployment:status`) for
 *       real-time updates after the initial load.
 *     responses:
 *       '200':
 *         description: Array of deployment objects with node, model, and clusterNodes included
 */
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

/**
 * @openapi
 * /api/deployments/{id}/logs:
 *   get:
 *     tags: [Deployments]
 *     summary: Retrieve deployment logs from shared storage
 *     description: >
 *       Reads the deployment log file from `$SHARED_STORAGE/logs/deployments/{id}.log`
 *       and returns it as `text/plain`. Returns an empty 200 body if the log file has
 *       not been created yet (deployment hasn't started, or logs are not persisted for
 *       this deployment type). Accepts a `tail` query parameter to return only the
 *       last N lines — useful for keeping the dashboard log pane up-to-date without
 *       re-reading the full file.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *         description: Deployment ID
 *       - in: query
 *         name: tail
 *         required: false
 *         schema: { type: integer }
 *         description: If > 0, return only the last N lines
 *     responses:
 *       '200':
 *         description: Log content as text/plain (empty if not yet available)
 *         content:
 *           text/plain:
 *             schema: { type: string }
 */
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

/**
 * @openapi
 * /api/deployments:
 *   post:
 *     tags: [Deployments]
 *     summary: Launch an inference deployment via sparkrun
 *     description: >
 *       Launches a recipe on one or more DGX Spark nodes. The head-node agent runs `sparkrun run`,
 *       which distributes the image+model and starts containers. Provide exactly one recipe source:
 *       `recipeFile` (registry recipe from GET /api/recipes), `recipePath` (a YAML staged under shared
 *       storage), or `recipeYaml` (inline recipe body — for remote recipe-dev machines that never touch
 *       the cluster filesystem). VRAM admission runs before launch. Returns the created Deployment.
 *       For Ollama deployments, provide `runtime: "ollama"` and `modelName` instead of a recipe source.
 *       Node selection: pass `nodeId: "auto"` for the first available online node, `nodeIds: "auto"`
 *       to auto-select a cluster sized to the recipe's tensor/pipeline parallelism, or explicit IDs.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               nodeId: { type: string, description: "Single-node target (or 'auto')." }
 *               nodeIds: { type: array, items: { type: string }, description: "Cluster node ids, head first (or 'auto')." }
 *               recipeFile: { type: string, description: "Registry recipe ref from GET /api/recipes." }
 *               recipePath: { type: string, description: "Path under shared storage to a recipe YAML." }
 *               recipeYaml: { type: string, description: "Inline sparkrun recipe YAML body (remote dev)." }
 *               config: { type: object, description: "Overrides: port, gpuMem, tensorParallel, maxModelLen, pipelineParallel…" }
 *               displayName: { type: string, description: "Optional vLLM --served-model-name and dashboard label (must be unique among active deployments)." }
 *               runtime: { type: string, enum: [vllm, ollama], description: "Inference engine. Defaults to vllm." }
 *               modelName: { type: string, description: "Required for Ollama: the model pull tag (e.g. llama3.1:8b)." }
 *               modelType: { type: string, enum: [chat, embedding], description: "Ollama model capability. Defaults to chat." }
 *     responses:
 *       '201':
 *         description: Created deployment record (node + model + clusterNodes included)
 *       '400':
 *         description: Invalid request (missing recipe source, bad inline YAML, missing modelName for Ollama)
 *       '409':
 *         description: VRAM shortfall, no online nodes, or displayName conflict
 */
deploymentsRouter.post("/", async (req, res) => {
  let { nodeId, nodeIds, recipeFile, recipePath, recipeYaml, config, runtime, modelName, modelType, displayName: rawDisplayName } = req.body;
  const isOllama = runtime === "ollama";

  if (isOllama && !modelName) {
    return res.status(400).json({ error: "modelName required for Ollama deployments" });
  }

  // Resolve the recipe source for vLLM deployments. Three mutually-exclusive
  // sources: inline YAML (D7), absolute path (D5), registry ref (recipeFile).
  let recipeRef: string | undefined;          // registry ref or validated abs path
  let inlineRecipeYaml: string | undefined;   // raw YAML body (D7)
  if (!isOllama) {
    if (recipeYaml) {
      try { validateInlineRecipe(recipeYaml); } catch (e) { return res.status(400).json({ error: (e as Error).message }); }
      inlineRecipeYaml = recipeYaml;
      console.log(`[deploy] inline recipeYaml (${Buffer.byteLength(recipeYaml, "utf8")} bytes) from ${req.ip}`);
    } else if (recipePath) {
      try { recipeRef = resolveRecipePath(recipePath, SHARED_STORAGE); } catch (e) { return res.status(400).json({ error: (e as Error).message }); }
    } else if (recipeFile) {
      recipeRef = recipeFile; // registry ref from `sparkrun list`
    } else {
      return res.status(400).json({ error: "recipeFile, recipePath, or recipeYaml required for vLLM deployments" });
    }
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
    const checkNodeIds = isCluster ? (nodeIds as string[]) : [headNodeId];

    // Arch admission — fail fast on a recipe/node CPU-arch mismatch (e.g. an
    // arm64 DGX-Spark recipe deployed to the amd64 RTX-5090 host). Only the
    // registry-ref branch (recipeFile, resolved against the catalog) is
    // guarded; inline recipeYaml and Ollama are user-authored / arch-agnostic
    // and intentionally bypass this.
    if (recipe) {
      for (const nid of checkNodeIds) {
        const node = await prisma.node.findUnique({ where: { id: nid } });
        const nodeArch = node?.arch;
        if (nodeArch && !checkRecipeArchAdmission(recipe.arch, nodeArch as "amd64" | "arm64")) {
          return res.status(400).json({
            error: `${node?.name ?? nid}: ${recipeArchMismatchMessage(recipe.arch, nodeArch as "amd64" | "arm64")}`,
            recipeArch: recipe.arch,
            nodeArch,
          });
        }
      }
    }

    const gpuMemUtil = (config?.gpuMem as number) || (recipe?.defaults?.gpu_memory_utilization as number) || 0.85;
    const shortfalls = await checkVllmVramAdmission(checkNodeIds, gpuMemUtil);
    if (shortfalls.length > 0) {
      return res.status(409).json({
        error: `Not enough VRAM on ${shortfalls.length} of ${checkNodeIds.length} node(s): ${vramShortfallMessage(shortfalls)}`,
        shortfalls,
        gpuMemoryUtilization: gpuMemUtil,
      });
    }
  }

  // Ensure a model record exists. For vLLM the key is derived from whichever
  // recipe source was supplied: registry ref (recipeFile), abs path
  // (recipeRef when recipePath was used), or a synthetic key for inline YAML.
  const vllmModelKey = recipeFile
    ? recipeFile.replace(/^recipes\//, "").replace(/\.yaml$/, "")
    : recipeRef
      ? recipeRef.replace(/.*\//, "").replace(/\.yaml$/, "")  // basename, no ext
      // Inline YAML: surface the real model identity (HF model id, falling back
      // to the served alias) so the dashboard shows a meaningful name instead of
      // an opaque timestamp. `inline-<ts>` remains only if neither is parseable.
      : (() => {
          const meta = inlineRecipeYaml ? parseInlineRecipeModel(inlineRecipeYaml) : {};
          return meta.model ?? meta.servedModelName ?? `inline-${Date.now()}`;
        })();
  const modelKey = isOllama ? modelName : vllmModelKey;
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
      // Sparkrun recipe sources (D5/D7): agent branch triggers on these.
      recipeRef: isOllama ? undefined : recipeRef,
      inlineRecipeYaml: isOllama ? undefined : inlineRecipeYaml,
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

/**
 * @openapi
 * /api/deployments/{id}:
 *   delete:
 *     tags: [Deployments]
 *     summary: Stop (or stop + delete) a deployment
 *     description: >
 *       Sends `cmd:undeploy` to the head-node agent to stop the running containers.
 *       Without `?delete=true` the Deployment row is kept (status moves to `removing`
 *       then `stopped`), allowing inspection of logs and later restart. With
 *       `?delete=true` the row (and related LB endpoints) is deleted from the DB —
 *       fast-path if already in a terminal state (stopped/failed/evicted) so the
 *       agent is not involved unnecessarily. For cluster deployments, the cluster
 *       node IPs are passed so the agent can tear down all worker containers.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *       - in: query
 *         name: delete
 *         required: false
 *         schema: { type: string, enum: ['true', 'false'] }
 *         description: If "true", also delete the deployment record after stopping
 *     responses:
 *       '200':
 *         description: '{ status: "removing" } or { status: "deleted" }'
 *       '404':
 *         description: Deployment not found
 */
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

/**
 * @openapi
 * /api/deployments/{id}/restart:
 *   post:
 *     tags: [Deployments]
 *     summary: Restart a deployment with optional config overrides
 *     description: >
 *       Re-dispatches `cmd:deploy` (or `cmd:finetune:deploy` for fine-tune deployments)
 *       to the head-node agent. Merges caller-supplied `config` overrides over the saved
 *       config, allowing fixes like lowering `max_model_len` after an OOM without deleting
 *       and re-creating the deployment. Reserved fields (`recipeFile`, `runtime`,
 *       `modelName`, `modelType`) cannot be overridden — they identify the deployment.
 *       Also accepts a `displayName` override to change the vLLM served-model-name.
 *       Runs the same VRAM admission check as the initial deploy. Status moves to
 *       `restarting` while the agent is working.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     requestBody:
 *       required: false
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               config: { type: object, description: "Config overrides (port, gpuMem, maxModelLen, tensorParallel, artifactVariant…). Reserved: recipeFile, runtime, modelName, modelType." }
 *               displayName: { type: string, nullable: true, description: "New vLLM --served-model-name / dashboard label. Must be unique among active deployments." }
 *     responses:
 *       '200':
 *         description: '{ status: "restarting" }'
 *       '400':
 *         description: Invalid artifactVariant or displayName
 *       '404':
 *         description: Deployment not found
 *       '409':
 *         description: VRAM shortfall or displayName conflict
 */
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
        // recipeRef is required for the sparkrun branch in the agent
        // (the `if (inlineRecipeYaml != null || payloadRecipeRef != null)`
        // guard). Persisted config stores the registry/path ref as
        // `recipeFile`; mirror that into `recipeRef` so restarts reach the
        // sparkrun path instead of falling through to the "No recipeRef or
        // inlineRecipeYaml specified" error.
        recipeRef: isOllamaRestart ? undefined : config.recipeFile,
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
