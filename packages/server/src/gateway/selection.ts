import type { EligibleMember } from "./eligibility.js";

/**
 * Choosing between the members of a **pool**.
 *
 * Least-outstanding-requests, ties broken by rotation. A pool is not
 * necessarily uniform — the cluster's real one spans a CPU-only host and a GPU
 * host, hardware that differs by roughly an order of magnitude — and
 * round-robin across those would give half the callers a far slower answer with
 * no indication why. Least-outstanding is self-tuning: the slow member
 * accumulates in-flight requests and stops attracting new ones.
 *
 * Chosen over static weights, which are hand-maintained state that goes stale
 * whenever hardware, model, or context length changes. See
 * docs/adr/0001-inference-gateway.md.
 *
 * Pure: the caller supplies the counts and the rotation, so the rule can be
 * stated as an invariant rather than inferred from behaviour under load.
 */

/** How many requests are in flight to a given deployment right now. */
export type OutstandingCount = (deploymentId: string) => number;

/**
 * Pick the member holding the fewest in-flight requests.
 *
 * When several are tied — the ordinary case, since counts return to zero
 * between requests — `rotation` decides, so an idle pool is used evenly instead
 * of hammering whichever member happens to sort first.
 */
export function selectLeastOutstanding(
  members: EligibleMember[],
  outstanding: OutstandingCount,
  rotation: number,
): EligibleMember | null {
  if (members.length === 0) return null;

  let min = Infinity;
  for (const m of members) {
    const n = outstanding(m.deploymentId);
    if (n < min) min = n;
  }

  const tied = members.filter((m) => outstanding(m.deploymentId) === min);
  // Modulo of a negative rotation is negative in JS; normalise so any counter
  // (including one that has wrapped) indexes into the tied set.
  const index = ((rotation % tied.length) + tied.length) % tied.length;
  return tied[index];
}
