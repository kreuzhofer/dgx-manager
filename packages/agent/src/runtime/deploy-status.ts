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
