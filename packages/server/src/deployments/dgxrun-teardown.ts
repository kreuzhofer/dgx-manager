import { prisma } from "../prisma.js";
import { decideTeardown } from "./teardown-decision.js";

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
  trigger: "failed" | "stopped" = "failed",
): Promise<string[]> {
  const dep = await prisma.deployment.findUnique({
    where: { id: deploymentId },
    include: { clusterNodes: true },
  }).catch(() => null);
  if (!dep) return [];

  let cfg: Record<string, unknown> = {};
  try { cfg = dep.config ? JSON.parse(dep.config) : {}; } catch { cfg = {}; }

  const nodeIds = dep.clusterNodes.length > 0
    ? dep.clusterNodes.map((c) => c.nodeId)
    : [dep.nodeId];

  const decision = decideTeardown({ runner: cfg.runner, nodeIds, trigger });
  if (decision.kind === "skip") {
    if (cfg.runner === "dgxrun") {
      console.log(`[dgxrun] teardown skipped for ${deploymentId}: ${decision.reason}`);
    }
    return [];
  }

  for (const nid of decision.nodeIds) {
    hub.sendToAgent(nid, {
      type: "cmd:undeploy",
      payload: {
        deploymentId,
        deleteAfter: false,
        kind: "dgxrun",
        // Keep the container for post-mortem. launchDgxrun does `docker rm -f`
        // on the same name before it starts, so this is reclaimed by the next
        // deploy rather than leaking (#94).
        preserveContainer: decision.preserveContainer,
      },
    });
  }
  return decision.nodeIds;
}
