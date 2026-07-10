import type { DgxrunInspect } from "./dgxrun.js";

/**
 * What an agent should do about one dgxrun rank when it reconnects to the manager.
 *
 * The stakes: a rank that reports `failed` makes the manager tear down EVERY rank
 * of the mp cluster (there is no recovery in the mp executor). So the reconnect
 * reconciliation must never guess. `docker inspect` failing to answer — a 10 s
 * spawnSync timeout, a daemon busy under a weight stream — is `unknown`, NOT
 * "the container is gone". Collapsing the two is what killed a healthy four-rank
 * GLM-5.2 cluster mid-`torch.compile` (2026-07-09); it was fixed in the health
 * loop but left in this path, where an agent roll is precisely what triggers it.
 *
 * On `unknown` we report nothing and let the health loop converge: it already
 * requires `absent` on two consecutive ticks before failing a rank.
 */
export type DgxrunReconcileAction =
  | { kind: "skip"; reason: string }
  | { kind: "report"; status: "running" | "failed"; port?: number; error?: string }
  /** Head is alive; re-announce the loading phase rather than a terminal status. */
  | { kind: "phase" };

export function reconcileDgxrunAction(
  inspect: DgxrunInspect,
  opts: { rank: number; port?: number },
): DgxrunReconcileAction {
  if (inspect.kind === "unknown") {
    return { kind: "skip", reason: `docker inspect inconclusive (${inspect.reason})` };
  }
  const running = inspect.kind === "found" && inspect.state === "running";
  if (!running) {
    return { kind: "report", status: "failed", error: "dgxrun rank not running after agent restart" };
  }
  // Head lets the health loop promote it to running once /metrics binds; a worker
  // is running as soon as its container is.
  if (opts.rank === 0) return { kind: "phase" };
  return { kind: "report", status: "running" };
}
