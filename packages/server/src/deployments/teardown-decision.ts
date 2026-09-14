/**
 * Deciding what an automatic dgxrun teardown should do.
 *
 * Pure so the rule can be tested without Prisma or a WebSocket hub; the
 * DB-coupled half stays in dgxrun-teardown.ts. Same split as admission/vram.ts.
 */

export type TeardownDecision =
  | { kind: "skip"; reason: string }
  | { kind: "fan"; nodeIds: string[]; preserveContainer: boolean };

export interface TeardownInput {
  /** `config.runner` — only dgxrun deployments are coordinated. */
  runner: unknown;
  /** Cluster member node ids, or the single owning node. */
  nodeIds: string[];
  /** Why the teardown fired. A failure is a post-mortem; a stop is routine. */
  trigger: "failed" | "stopped";
}

/**
 * The mp executor has no recovery: one dead rank hangs the whole cluster, so a
 * failure on ANY rank must tear down EVERY rank. That is the reason this exists
 * and it only applies when there is more than one rank.
 *
 * For a SINGLE-rank deployment "tear down every rank" means "delete the only
 * container", which buys nothing — the agent's own health path already stops a
 * failed solo container — and costs the only remaining evidence of why it died
 * (#94). So a solo deployment is skipped.
 *
 * When a multi-rank teardown does fire on a failure, the containers are
 * preserved rather than removed. A post-mortem needs a body, and `launchDgxrun`
 * already does `docker rm -f` on the same name before it starts, so a preserved
 * container is reclaimed by the next deploy instead of leaking.
 */
export function decideTeardown(input: TeardownInput): TeardownDecision {
  const { runner, nodeIds, trigger } = input;

  if (runner !== "dgxrun") return { kind: "skip", reason: "not a dgxrun deployment" };
  if (nodeIds.length === 0) return { kind: "skip", reason: "no nodes to tear down" };
  if (nodeIds.length === 1) {
    return {
      kind: "skip",
      reason: "single-rank deployment: the agent's own health path stops it, and " +
        "coordinated teardown would only destroy the evidence",
    };
  }
  return { kind: "fan", nodeIds, preserveContainer: trigger === "failed" };
}
