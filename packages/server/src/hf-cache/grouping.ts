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
    const members = byKey.get(key) ?? [];
    members.push(inv);
    byKey.set(key, members);
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

/** The strings a deployment might know its model by: the Model row's name and
 *  the modelName recorded in the deployment's config JSON. Fine-tune models
 *  (local output paths) never equal an HF repoId — correctly so, their weights
 *  are not in the HF cache. */
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
  candidates: string[];       // from deploymentModelCandidates
  clusterNodeIds: string[];
}

// "evicted" is deliberately NOT terminal here: an evicted deployment can be
// restored onto the GPU and would re-read its weights from this cache.
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
  const lastDeployedAt = touching.length
    ? touching.map((d) => d.createdAt).sort().at(-1)!
    : null;
  return { inUse: active.length > 0, inUseBy: active.map((d) => d.label), lastDeployedAt };
}
