import type { DgxrunResolvedRecipe } from "./dgxrun-recipe.js";

/**
 * Pure builders for the manager→agent dgxrun fan-out. The route uses these to
 * produce one `cmd:deploy` per cluster node (head = rank 0), each carrying its
 * rank + the head's mgmt IP as `masterAddr`. Kept IO-free so the fan-out
 * contract is unit/integration testable without a live agentHub.
 */

/** torch TCPStore rendezvous port when the recipe/config doesn't override it. */
export const DEFAULT_MASTER_PORT = 29500;

export interface DgxrunDeployContext {
  deploymentId: string;
  recipe: DgxrunResolvedRecipe;
  /** Cluster node ids, head first. */
  clusterNodeIds: string[];
  /** Cluster node management IPs, head first — same order as clusterNodeIds. */
  clusterNodeIps: string[];
  masterPort?: number;
  /** Placeholder/config overrides forwarded to each agent (port, tp, gpuMem…). */
  params?: Record<string, string | number>;
}

export interface DgxrunNodeDeploy {
  nodeId: string;
  payload: {
    deploymentId: string;
    kind: "dgxrun";
    recipe: DgxrunResolvedRecipe;
    rank: number;
    nnodes: number;
    masterAddr: string;
    masterPort: number;
    headless: boolean;
    params: Record<string, string | number>;
  };
}

/**
 * Build the per-rank `cmd:deploy` payloads for a dgxrun deploy. Ranks are
 * assigned head-first (index 0 = head = rank 0). `masterAddr` is ALWAYS the
 * head's management IP; `headless` is set for every worker (rank > 0).
 */
export function buildDgxrunDeploys(ctx: DgxrunDeployContext): DgxrunNodeDeploy[] {
  if (ctx.clusterNodeIds.length === 0) throw new Error("dgxrun deploy needs at least one node");
  if (ctx.clusterNodeIps.length !== ctx.clusterNodeIds.length) {
    throw new Error("dgxrun deploy: clusterNodeIps must align 1:1 with clusterNodeIds");
  }
  const masterAddr = ctx.clusterNodeIps[0];
  if (!masterAddr) throw new Error("dgxrun deploy: head node has no management IP");
  const nnodes = ctx.clusterNodeIds.length;
  const masterPort = ctx.masterPort ?? DEFAULT_MASTER_PORT;

  return ctx.clusterNodeIds.map((nodeId, rank) => ({
    nodeId,
    payload: {
      deploymentId: ctx.deploymentId,
      kind: "dgxrun",
      recipe: ctx.recipe,
      rank,
      nnodes,
      masterAddr,
      masterPort,
      headless: rank > 0,
      params: ctx.params ?? {},
    },
  }));
}
