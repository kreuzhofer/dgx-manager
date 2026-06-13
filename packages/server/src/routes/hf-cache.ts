import { Router } from "express";
import { prisma } from "../prisma.js";
import type { AgentHub } from "../ws/agent-hub.js";
import {
  deploymentModelCandidates, groupInventories, repoUsage, type DeploymentUsage,
} from "../hf-cache/grouping.js";

export const hfCacheRouter = Router();

/** Build the deployment-usage list the in-use guard consumes. Loads ALL
 *  deployments (any status — terminal ones still set lastDeployedAt, active
 *  ones drive the guard) and assembles each one's full set of model-name
 *  candidates. Completeness here is what makes the guard sound:
 *   - Model.name + config.modelName (Ollama tag / inline-YAML HF id)
 *   - the recipe catalog's HF id resolved from config.recipeFile (registry-ref
 *     vLLM, where Model.name is only the recipe slug)
 *   - the fine-tune base model (its base weights live in the HF cache)
 *  Known gap: recipePath deploys don't persist their recipe ref, and a recipe
 *  absent from the catalog can't be resolved — both yield a conservative miss. */
async function loadDeploymentUsage(agentHub: AgentHub): Promise<DeploymentUsage[]> {
  const deployments = await prisma.deployment.findMany({
    include: {
      model: { include: { finetuneJob: { select: { baseModel: true } } } },
      clusterNodes: true,
    },
  });
  const recipeHfId = new Map<string, string>();
  for (const r of agentHub.getRecipes()) {
    if (r.model) recipeHfId.set(r.file, r.model);
  }
  return deployments.map((d) => {
    const candidates = deploymentModelCandidates(d.model.name, d.config);
    let recipeFile: string | undefined;
    if (d.config) {
      try { recipeFile = (JSON.parse(d.config) as { recipeFile?: string }).recipeFile; }
      catch { /* malformed config — skip recipe resolution */ }
    }
    if (recipeFile) {
      const hf = recipeHfId.get(recipeFile);
      if (hf) candidates.push(hf);
    }
    if (d.model.finetuneJob?.baseModel) candidates.push(d.model.finetuneJob.baseModel);
    return {
      status: d.status,
      nodeId: d.nodeId,
      createdAt: d.createdAt.toISOString(),
      label: d.displayName ?? d.model.name,
      candidates,
      clusterNodeIds: d.clusterNodes.map((cn) => cn.nodeId),
    };
  });
}

/**
 * @openapi
 * /api/hf-cache:
 *   get:
 *     tags: [HF Cache]
 *     summary: List Hugging Face cache contents per cache group
 *     description: >
 *       Returns the latest HF_HOME inventories pushed by agents, grouped by
 *       cache identity (a `.dgx-cache-id` marker file at the HF_HOME root —
 *       nodes sharing the cache over NFS report the same id and collapse into
 *       one group; NFS-less nodes each form their own). Each cached repo is
 *       enriched with `inUse` (an active deployment on the group's nodes
 *       references it — deletion is blocked) and `lastDeployedAt` (newest
 *       matching deployment of any status). Empty until agents have scanned;
 *       trigger POST /api/hf-cache/scan.
 *     responses:
 *       '200':
 *         description: '{ caches: [{ cacheId, nodes, hfHome, scannedAt, totalBytes, diskFreeBytes, error?, repos }] }'
 */
hfCacheRouter.get("/", async (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  const groups = groupInventories(agentHub.getHfCacheInventories());
  const usage = await loadDeploymentUsage(agentHub);
  const nodes = await prisma.node.findMany({ select: { id: true, name: true } });
  const nameById = new Map(nodes.map((n) => [n.id, n.name]));

  const caches = groups.map((group) => {
    const groupNodeIds = new Set(group.nodeIds);
    return {
      cacheId: group.cacheId,
      nodes: group.nodeIds.map((id) => ({
        nodeId: id,
        name: nameById.get(id) ?? id,
        connected: agentHub.isAgentOnline(id),
      })),
      hfHome: group.newest.hfHome,
      scannedAt: group.newest.scannedAt,
      totalBytes: group.newest.totalBytes,
      diskFreeBytes: group.newest.diskFreeBytes,
      error: group.newest.error,
      repos: group.newest.repos.map((r) => ({
        ...r,
        ...repoUsage(r.repoId, groupNodeIds, usage),
      })),
    };
  });
  res.json({ caches });
});

/**
 * @openapi
 * /api/hf-cache/scan:
 *   post:
 *     tags: [HF Cache]
 *     summary: Ask every connected agent to rescan its HF cache
 *     description: >
 *       Fans out `cmd:hf-cache:scan` to all connected agents. Each agent walks
 *       its HF_HOME and pushes a fresh inventory, which arrives asynchronously
 *       via the `hf-cache:inventory` SSE event. Returns 503 when no agents are
 *       connected.
 *     responses:
 *       '202':
 *         description: '{ requested: N }'
 *       '503':
 *         description: No agents connected
 */
hfCacheRouter.post("/scan", (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  const nodeIds = agentHub.getConnectedNodeIds();
  if (nodeIds.length === 0) {
    return res.status(503).json({ error: "No agents connected — nothing can scan the cache" });
  }
  for (const nodeId of nodeIds) {
    agentHub.sendToAgent(nodeId, { type: "cmd:hf-cache:scan", payload: {} });
  }
  res.status(202).json({ requested: nodeIds.length });
});

/**
 * @openapi
 * /api/hf-cache/{cacheId}:
 *   delete:
 *     tags: [HF Cache]
 *     summary: Delete a cached repo from a cache group
 *     description: >
 *       Sends `cmd:hf-cache:delete` to one connected agent in the cache group.
 *       The repo id travels as a query parameter (`?repoId=org%2Fname`) because
 *       URL-encoded slashes in path segments are unreliable across HTTP stacks.
 *       Refused with 409 while any active (non-stopped/failed) deployment on the
 *       group's nodes references the repo.
 *     parameters:
 *       - in: path
 *         name: cacheId
 *         required: true
 *         schema: { type: string }
 *       - in: query
 *         name: repoId
 *         required: true
 *         schema: { type: string }
 *         description: URL-encoded repo id, e.g. org%2Fname
 *       - in: query
 *         name: kind
 *         schema: { type: string, enum: [model, dataset], default: model }
 *     responses:
 *       '202': { description: 'Delete dispatched to an agent' }
 *       '400': { description: Missing repoId }
 *       '404': { description: Unknown cacheId or repo not in this cache }
 *       '409': { description: Repo is in use by an active deployment }
 *       '503': { description: No connected agent can reach this cache }
 */
hfCacheRouter.delete("/:cacheId", async (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  const { cacheId } = req.params;
  const repoId = typeof req.query.repoId === "string" ? req.query.repoId : "";
  const kind = req.query.kind === "dataset" ? "dataset" : "model";
  if (!repoId) return res.status(400).json({ error: "repoId query parameter is required" });

  const group = groupInventories(agentHub.getHfCacheInventories())
    .find((g) => g.cacheId === cacheId);
  if (!group) return res.status(404).json({ error: `Unknown cache: ${cacheId}` });

  const repo = group.newest.repos.find((r) => r.repoId === repoId && r.kind === kind);
  if (!repo) return res.status(404).json({ error: `Not in this cache: ${repoId} (${kind})` });

  const usage = await loadDeploymentUsage(agentHub);
  const { inUse, inUseBy } = repoUsage(repoId, new Set(group.nodeIds), usage);
  if (inUse) {
    return res.status(409).json({
      error: `${repoId} is in use by: ${inUseBy.join(", ")}. Stop those deployments first.`,
      deployments: inUseBy,
    });
  }

  const execNodeId = group.nodeIds.find((id) => agentHub.isAgentOnline(id));
  if (!execNodeId) {
    return res.status(503).json({ error: "No connected agent can reach this cache" });
  }

  agentHub.sendToAgent(execNodeId, { type: "cmd:hf-cache:delete", payload: { repoId, kind } });
  res.status(202).json({ deleting: repoId, kind, nodeId: execNodeId });
});
