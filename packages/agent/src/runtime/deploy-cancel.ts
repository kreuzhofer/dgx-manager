/**
 * Tracks deployments whose teardown was requested while their launch was still
 * in flight.
 *
 * Why this exists: `launchDgxrun` clears any stale container with `docker rm -f`
 * and then `spawn("docker", ["run", "-d", …])`, which **returns before the
 * container exists**. `handleCommand` is not awaited, so a `cmd:undeploy` that
 * arrives right after `cmd:deploy` runs its own `docker rm -f` against a
 * container that has not been created yet, finds nothing, and reports `stopped`.
 * The manager deletes the row; a moment later `docker run -d` finishes and the
 * container comes up untracked, holding ~100 GB/node with nothing pointing at it
 * (2026-07-09).
 *
 * So teardown cannot be a one-shot: the launch has to re-check, at the instant
 * the container provably exists (`docker run -d` exit 0), whether a stop was
 * requested in the meantime.
 *
 * A cancel is remembered until either the racing launch reaps it or a **new**
 * launch for the same id supersedes it — the latter is what keeps
 * `POST /:id/restart` (which reuses the deployment id) working after a stop.
 */

export interface CancelRecord {
  /** Propagated to the `stopped` status so the manager still deletes the row. */
  deleteAfter: boolean;
}

export interface CancelRegistry {
  /** A launch for `id` is starting. Supersedes any earlier cancel for that id. */
  beginDeploy(id: string): void;
  /** Teardown requested for `id`. Observed by a launch that is still in flight. */
  requestCancel(id: string, deleteAfter: boolean): void;
  /** The pending cancel for `id`, or null. */
  pendingCancel(id: string): CancelRecord | null;
  /** Drop any cancel for `id` (the racing launch has been torn down). */
  forget(id: string): void;
}

export function createCancelRegistry(): CancelRegistry {
  // Entries are tiny and bounded by "stops whose launch never arrived", so we
  // let an unmatched cancel sit rather than time it out and reintroduce the race.
  const cancels = new Map<string, CancelRecord>();
  return {
    beginDeploy(id) {
      cancels.delete(id);
    },
    requestCancel(id, deleteAfter) {
      cancels.set(id, { deleteAfter });
    },
    pendingCancel(id) {
      return cancels.get(id) ?? null;
    },
    forget(id) {
      cancels.delete(id);
    },
  };
}

/** Process-wide registry used by the agent's command handlers. */
export const deployCancels = createCancelRegistry();

/** What the agent must do when a rank's `docker run -d` exits. */
export type LaunchExitAction =
  | { kind: "cancelled"; deleteAfter: boolean }
  | { kind: "failed"; error: string }
  | { kind: "running" };

/**
 * Pure: decide what a rank's launch exit means.
 *
 * A pending cancel WINS over a non-zero exit. The user asked for this deployment
 * to go away; whether the container came up or the launch died on the way is
 * irrelevant to that, and reporting `failed` instead would trip the manager's
 * coordinated teardown and — when the stop carried `deleteAfter` — leave the row
 * behind. Teardown is idempotent, so acting on a cancel after a failed launch is
 * safe.
 */
export function launchExitAction(opts: {
  code: number | null;
  rank: number;
  cancel: CancelRecord | null;
}): LaunchExitAction {
  if (opts.cancel) return { kind: "cancelled", deleteAfter: opts.cancel.deleteAfter };
  if (opts.code !== 0) {
    return { kind: "failed", error: `dgxrun rank ${opts.rank} launch failed (exit ${opts.code})` };
  }
  return { kind: "running" };
}
