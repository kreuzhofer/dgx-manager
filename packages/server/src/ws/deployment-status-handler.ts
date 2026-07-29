import { prisma } from "../prisma.js";
import { broadcast as sseBroadcast } from "../sse.js";
import { coordinatedDgxrunTeardown, type TeardownHub } from "../deployments/dgxrun-teardown.js";
import { deploymentEndpointUrl } from "../benchmarks/endpoint.js";
import { resolvePublishedName, isAllocationInducingRuntime } from "../deployments/published-name.js";
import { deploymentStatusUpdate, isTerminalDeploymentStatus, HEALTHY_STATUS } from "./deployment-status.js";

/**
 * The `agent:deployment:status` path, lifted out of the AgentHub socket loop so
 * it can be driven directly in tests.
 *
 * Why this exists as its own module: everything here is a *consequence* of one
 * agent message — persisting the patch, publishing the name, tearing down a
 * dgxrun cluster, auto-deleting after a confirmed stop — and none of it needs a
 * WebSocket. Before this split nothing in the suite drove the hub at all, so a
 * step that the socket loop simply forgot to call was invisible to every test.
 */
export interface DeploymentStatusMessage {
  deploymentId: string;
  status: string;
  port?: number | null;
  error?: string | null;
  deleteAfter?: boolean;
  vramActual?: number | string | null;
}

export interface DeploymentStatusDeps {
  /**
   * Required: a failed rank fans a coordinated teardown out to every other
   * rank. Optional would let a caller silently skip it, and the dgxrun mp
   * executor has no recovery — one dead rank hangs the whole cluster.
   */
  hub: TeardownHub;
  /** Injected in tests so publishing a name never touches the network. */
  fetchImpl?: typeof fetch;
}

/**
 * Publish the name a client will use to reach this deployment.
 *
 * Only ever runs on the transition into serving, and only when no name is
 * stored yet — the name is a property of one serving lifetime, and
 * `deploymentStatusUpdate` clears it on any terminal status, so a restart
 * re-resolves rather than inheriting a stale name.
 *
 * Never throws: a deployment that is up and serving must not be marked failed
 * because we could not name it. An unreachable endpoint already degrades to the
 * local fallback inside the resolver.
 */
async function publishName(deploymentId: string, fetchImpl?: typeof fetch): Promise<void> {
  try {
    const d = await prisma.deployment.findUnique({
      where: { id: deploymentId },
      include: { node: true, model: true },
    });
    if (!d) return;

    // An allocation-inducing runtime's tag is authoritative and involves no
    // probe, so there is nothing a later report could improve on.
    if (d.publishedName && isAllocationInducingRuntime(d.model.runtime)) return;

    // A pinned runtime's stored name that still equals the local guess may be
    // exactly that — a guess persisted because the endpoint had not accepted
    // connections yet on the first serving report. Ask again on later reports
    // until the runtime answers for itself; otherwise a narrow startup race
    // makes the guess permanent, which is the unexplained-404 this resolution
    // exists to prevent. Serving reports are rare (status change, reconnect
    // reconciliation), so the redundant probe in the already-correct case costs
    // nothing worth a schema column to avoid.
    const fallback = d.displayName ?? d.model.name;
    if (d.publishedName && d.publishedName !== fallback) return;

    const endpointUrl = d.node.ipAddress && d.port ? deploymentEndpointUrl(d) : null;
    const publishedName = await resolvePublishedName(
      {
        runtime: d.model.runtime,
        modelName: d.model.name,
        displayName: d.displayName,
        endpointUrl,
      },
      fetchImpl,
    );
    if (!publishedName) {
      console.error(
        `[gateway] deployment ${deploymentId} resolved to an empty published name ` +
          `(model "${d.model.name}", runtime "${d.model.runtime}") — it will not be reachable`,
      );
      return;
    }
    if (publishedName === d.publishedName) return;

    await prisma.deployment.update({ where: { id: deploymentId }, data: { publishedName } });
  } catch (err) {
    console.error(`[gateway] could not publish a name for deployment ${deploymentId}:`, err);
  }
}

/** Apply one agent deployment-status report. Mirrors the old inline handler. */
export async function handleDeploymentStatus(
  msg: DeploymentStatusMessage,
  deps: DeploymentStatusDeps,
): Promise<void> {
  const { deploymentId, status, port, error, deleteAfter, vramActual } = msg;

  try {
    // deploymentStatusUpdate persists `error` (so a failed deploy is not
    // indistinguishable from a stopped one) without letting the teardown
    // tick that follows a crash erase it. See ws/deployment-status.ts.
    await prisma.deployment.update({
      where: { id: deploymentId },
      data: deploymentStatusUpdate({ status, port, error, vramActual }),
    });
  } catch {
    return; // Deployment may already be deleted
  }

  if (error) console.error(`Deployment ${deploymentId} error: ${error}`);
  const isStopped = isTerminalDeploymentStatus(status);
  sseBroadcast({
    type: "deployment:status",
    payload: {
      deploymentId,
      status,
      port,
      error,
      vramActual: isStopped ? 0 : (vramActual ? Number(vramActual) : undefined),
    },
  });

  // The deployment is serving: give it the name clients will ask for.
  if (status === HEALTHY_STATUS) await publishName(deploymentId, deps.fetchImpl);

  // Update cluster node statuses when deployment changes
  if (["stopped", "failed", HEALTHY_STATUS].includes(status)) {
    await prisma.clusterNode.updateMany({ where: { deploymentId }, data: { status } }).catch(() => {});
  }

  // dgxrun coordinated teardown: the mp executor has no recovery, so ONE dead
  // rank hangs the whole cluster. When any rank reports failed, tear down every
  // rank. No-op for non-dgxrun deployments.
  if (status === "failed") {
    await coordinatedDgxrunTeardown(deps.hub, deploymentId).catch((err) =>
      console.error(`[dgxrun] teardown failed for ${deploymentId}:`, err),
    );
  }

  // Auto-delete record after confirmed stop
  if (status === "stopped" && deleteAfter) {
    try {
      await prisma.clusterNode.deleteMany({ where: { deploymentId } });
      await prisma.deployment.delete({ where: { id: deploymentId } });
      sseBroadcast({ type: "deployment:deleted", payload: { deploymentId } });
      console.log(`Deployment ${deploymentId} deleted after stop`);
    } catch { /* already deleted */ }
  }
}
