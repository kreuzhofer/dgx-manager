import { prisma } from "../prisma.js";

/** Minimal agentHub surface needed to fan a teardown out to cluster nodes. */
export interface TeardownHub {
  sendToAgent(nodeId: string, message: Record<string, unknown>): void;
}

/**
 * Coordinated teardown for a dgxrun deployment.
 *
 * The `mp` executor has NO recovery: if one rank dies, the whole cluster hangs
 * (we saw this end-to-end). So when ANY rank reports a failure, the manager
 * tears down EVERY rank by fanning `cmd:undeploy` to each cluster node's agent
 * (each agent owns its own local rank container).
 *
 * No-op unless the deployment is a dgxrun deployment (`config.runner ===
 * "dgxrun"`), so it's safe to call unconditionally from the status handler.
 * Returns the node ids it dispatched to (empty when not applicable).
 */
export async function coordinatedDgxrunTeardown(
  hub: TeardownHub,
  deploymentId: string,
): Promise<string[]> {
  const dep = await prisma.deployment.findUnique({
    where: { id: deploymentId },
    include: { clusterNodes: true },
  }).catch(() => null);
  if (!dep) return [];

  let cfg: Record<string, unknown> = {};
  try { cfg = dep.config ? JSON.parse(dep.config) : {}; } catch { cfg = {}; }
  if (cfg.runner !== "dgxrun") return [];

  const nodeIds = dep.clusterNodes.length > 0
    ? dep.clusterNodes.map((c) => c.nodeId)
    : [dep.nodeId];

  for (const nid of nodeIds) {
    hub.sendToAgent(nid, {
      type: "cmd:undeploy",
      payload: { deploymentId, deleteAfter: false, kind: "dgxrun" },
    });
  }
  return nodeIds;
}
