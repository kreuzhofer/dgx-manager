/**
 * How many requests the gateway currently has in flight to each deployment.
 *
 * This is the only mutable state the gateway keeps. It lives in the manager
 * process and is deliberately not persisted: it describes what this process is
 * doing right now, and a restart genuinely has nothing in flight.
 *
 * Kept apart from the selection rule so that rule stays a pure function of
 * counts — see selection.ts.
 */

const counts = new Map<string, number>();

/** Requests in flight to this deployment. */
export function outstandingFor(deploymentId: string): number {
  return counts.get(deploymentId) ?? 0;
}

/**
 * Mark one request as started; the returned function marks it finished.
 *
 * Release is idempotent, so a caller may invoke it from both a success path and
 * a `finally` without double-counting. It MUST be called on failure as well as
 * success: a leaked increment would permanently make a healthy member look
 * busier than it is, and least-outstanding would quietly stop routing to it.
 */
export function acquire(deploymentId: string): () => void {
  counts.set(deploymentId, outstandingFor(deploymentId) + 1);

  let released = false;
  return () => {
    if (released) return;
    released = true;
    const next = outstandingFor(deploymentId) - 1;
    if (next > 0) counts.set(deploymentId, next);
    else counts.delete(deploymentId);
  };
}

/** Current counts, for the gateway view. Empty entries are absent, not zero. */
export function outstandingSnapshot(): Record<string, number> {
  return Object.fromEntries(counts);
}

/** Test seam: forget everything. Never called in production. */
export function resetOutstanding(): void {
  counts.clear();
}
