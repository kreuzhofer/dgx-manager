/** Pure helpers for the HF cache management API: grouping per-node agent
 *  inventories into cache groups, and computing per-repo deployment usage.
 *  No IO here — the route layer feeds these from the hub and Prisma. */

export interface HfCacheRepo {
  repoId: string;
  kind: "model" | "dataset";
  sizeBytes: number;
  nFiles: number;
  revisions: number;
  lastModified: string;
}

export interface HfCacheNodeInventory {
  nodeId: string;       // attached by the hub when the message arrives
  cacheId: string;      // "" when the agent's scan failed before reading the marker
  hfHome: string;
  scannedAt: string;
  totalBytes: number;
  diskFreeBytes: number;
  error?: string;
  repos: HfCacheRepo[];
}

export interface CacheGroup {
  cacheId: string;
  nodeIds: string[];
  newest: HfCacheNodeInventory; // repo data source: latest scannedAt in the group
}

/** Inventories with an empty cacheId (failed scans) must not lump together —
 *  fall back to a synthetic per-node key. */
function cacheGroupKey(inv: HfCacheNodeInventory): string {
  return inv.cacheId || `node:${inv.nodeId}`;
}

export function groupInventories(inventories: HfCacheNodeInventory[]): CacheGroup[] {
  const byKey = new Map<string, HfCacheNodeInventory[]>();
  for (const inv of inventories) {
    const key = cacheGroupKey(inv);
    const members = byKey.get(key);
    if (members) members.push(inv);
    else byKey.set(key, [inv]);
  }
  return [...byKey.entries()].map(([cacheId, members]) => ({
    cacheId,
    nodeIds: members.map((m) => m.nodeId),
    // ISO-8601 strings compare correctly as strings
    newest: members.reduce((a, b) => (a.scannedAt >= b.scannedAt ? a : b)),
  }));
}

export function matchRepoToModels(
  repoId: string,
  candidates: Array<string | null | undefined>,
): boolean {
  const target = repoId.toLowerCase();
  return candidates.some((c) => typeof c === "string" && c.toLowerCase() === target);
}

/** SOME of the strings a deployment might know its model by: the Model row's
 *  name and the `modelName` in its config JSON. This covers Ollama (Model.name
 *  is the pull tag) and inline-YAML vLLM deploys (Model.name is the HF id). It
 *  does NOT cover registry-ref vLLM deploys (Model.name is the recipe slug, not
 *  the HF repo id) or fine-tune deploys (the base weights' HF id lives on the
 *  FineTuneJob). The route layer supplements this list with the recipe
 *  catalog's `model:` (resolved from config.recipeFile) and the fine-tune base
 *  model — see loadDeploymentUsage in routes/hf-cache.ts. Matching is exact
 *  (case-insensitive) string equality; getting the candidate set complete is
 *  what makes the in-use guard sound, so the route MUST feed all known names. */
export function deploymentModelCandidates(
  modelName: string | null | undefined,
  configJson: string | null | undefined,
): string[] {
  const out: string[] = [];
  if (modelName) out.push(modelName);
  if (configJson) {
    try {
      const cfg = JSON.parse(configJson) as Record<string, unknown>;
      if (typeof cfg.modelName === "string") out.push(cfg.modelName);
    } catch {
      // malformed config — fall back to the Model name candidate only
    }
  }
  return out;
}

export interface DeploymentUsage {
  status: string;
  nodeId: string;
  createdAt: string;          // ISO
  label: string;              // displayName ?? model name — shown in 409 errors
  candidates: string[];       // deploymentModelCandidates + recipe/finetune HF ids
  clusterNodeIds: string[];   // MUST be populated from Prisma clusterNodes
}

// Only fully-terminal states release a cache hold. Everything else — running,
// evicted (restorable onto the GPU), and the transient lifecycle states
// (pending/launching/deploying/stopping/removing) — is treated as in-use. This
// is the safe bias for a delete guard: a false "in use" merely annoys; a false
// "deletable" can rm the weights out from under a live container.
const TERMINAL_STATUSES = new Set(["stopped", "failed"]);

export function repoUsage(
  repoId: string,
  groupNodeIds: ReadonlySet<string>,
  deployments: DeploymentUsage[],
): { inUse: boolean; inUseBy: string[]; lastDeployedAt: string | null } {
  const touching = deployments.filter(
    (d) =>
      (groupNodeIds.has(d.nodeId) || d.clusterNodeIds.some((id) => groupNodeIds.has(id))) &&
      matchRepoToModels(repoId, d.candidates),
  );
  const active = touching.filter((d) => !TERMINAL_STATUSES.has(d.status));
  // newest createdAt wins; ISO-8601 strings compare correctly (same as groupInventories)
  const lastDeployedAt = touching.length
    ? touching.reduce((a, b) => (a.createdAt >= b.createdAt ? a : b)).createdAt
    : null;
  return { inUse: active.length > 0, inUseBy: active.map((d) => d.label), lastDeployedAt };
}
