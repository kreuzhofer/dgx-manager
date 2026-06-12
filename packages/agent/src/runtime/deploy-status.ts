/**
 * Pure classification of a dead vLLM container, so an intentional user stop is
 * not misreported as a crash.
 *
 * The health monitor detects "container gone" the same way whether the user
 * issued a stop (cmd:undeploy → stopRecipe marks the instance `stopping`) or the
 * engine crashed. Without distinguishing them, an intentional stop was reported
 * as `failed` / "Container stopped unexpectedly" — confusing in the UI.
 */
export type DeadContainerVerdict = { status: "stopped" | "failed"; error?: string };

export function classifyDeadContainer(
  intentional: boolean,
  lastError?: string | null,
): DeadContainerVerdict {
  if (intentional) return { status: "stopped" };
  return { status: "failed", error: lastError || "Container stopped unexpectedly" };
}

/**
 * Derive the deployment status for a sparkrun workload from two orthogonal
 * signals available at reconnect / health-check time:
 *
 *   - `listed`        — isWorkloadRunning returned true (workload is up in the
 *                       sparkrun cluster)
 *   - `launcherAlive` — the launcher subprocess is still running in this process
 *
 * Decision:
 *   - If the workload is listed (serving), it is "running" regardless of the
 *     launcher subprocess state.
 *   - If the launcher subprocess is alive but the workload is not yet listed,
 *     the workload is still coming up → "deploying".
 *   - If neither is true, the workload has died → "failed".
 */
export function reconcileDeployStatus(
  s: { launcherAlive: boolean; listed: boolean },
): "running" | "deploying" | "failed" {
  if (s.listed) return "running";
  return s.launcherAlive ? "deploying" : "failed";
}
