/**
 * Decide whether undeploying `targetId` should force-stop the shared `vllm_node`
 * container.
 *
 * All vLLM deployments on a node share ONE `vllm_node` container, and the stop
 * primitives (`forceStopVllm`, `launch-cluster.sh stop`, `docker stop vllm_node`)
 * are scoped to that container *name*, not to a deployment. So a stop must fire
 * only when the deployment being removed actually owns the running container —
 * otherwise deleting any (even already-dead) deployment record would tear down
 * whatever is currently serving.
 *
 * `ownerIds` comes from the persistent deployment store (`getTrackedDeployments`),
 * which survives agent restarts, so this still holds when the in-memory process
 * map has been cleared by a restart. When the container is running but no owner
 * is tracked, we conservatively decline — an unrelated undeploy must not kill an
 * unattributable container.
 */
export function shouldForceStopSharedContainer(
  targetId: string,
  ownerIds: readonly string[],
  containerRunning: boolean,
): boolean {
  if (!containerRunning) return false;
  return ownerIds.includes(targetId);
}

/**
 * Pick the deployment that owns the node's single `vllm_node` container from the
 * tracked-deployment store. Since only one container runs per node, the owner is
 * the most-recently-started entry — this prevents a stale older record that was
 * never cleaned from being mistaken for the live owner. Returns undefined when
 * nothing is tracked.
 */
export function selectContainerOwnerId(
  tracked: ReadonlyArray<{ deploymentId: string; startedAt: string }>,
): string | undefined {
  if (tracked.length === 0) return undefined;
  return tracked.reduce((a, b) => (a.startedAt >= b.startedAt ? a : b)).deploymentId;
}
