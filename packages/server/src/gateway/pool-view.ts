import { assessPool, type LivenessCheck, type MemberCandidate } from "./eligibility.js";

/**
 * What the cluster publishes, grouped for a human rather than for a client.
 *
 * The gateway itself needs one pool at a time; this is the whole picture. Its
 * reason for existing is that a **pool** is invisible in a per-deployment
 * table: two rows collapsing into one published name only reads as a unit when
 * it is grouped that way. It also shows deployments that are *not* serving
 * alongside the reason, so eligibility can be seen before a request fails
 * rather than only afterwards.
 *
 * Pure: the caller supplies liveness and in-flight counts.
 */

/** A deployment carrying a published name, with the detail the view shows. */
export interface PoolViewRow extends MemberCandidate {
  publishedName: string;
  nodeName: string;
  runtime: string | null;
}

export interface PoolViewMember {
  deploymentId: string;
  node: string;
  runtime: string | null;
  port: number | null;
  /** Requests the gateway currently has in flight to this member. */
  inflight: number;
  /** False when the member cannot take a request; `reason` says why. */
  serving: boolean;
  reason?: string;
  detail?: string;
}

export interface PoolView {
  publishedName: string;
  members: PoolViewMember[];
  /** How many members can take a request right now. */
  servingCount: number;
}

/**
 * Group published deployments into pools, marking each member serving or not.
 *
 * Pools are sorted by name and members by deployment id so the page does not
 * reshuffle between polls; serving members are listed before excluded ones, so
 * the useful rows come first.
 */
export function buildPoolView(
  rows: PoolViewRow[],
  isNodeOnline: LivenessCheck,
  outstanding: (deploymentId: string) => number,
): PoolView[] {
  const byName = new Map<string, PoolViewRow[]>();
  for (const row of rows) {
    const group = byName.get(row.publishedName);
    if (group) group.push(row);
    else byName.set(row.publishedName, [row]);
  }

  const views: PoolView[] = [];

  for (const [publishedName, group] of byName) {
    const { eligible, excluded } = assessPool(group, isNodeOnline);
    const eligibleIds = new Set(eligible.map((m) => m.deploymentId));
    const excludedById = new Map(excluded.map((m) => [m.deploymentId, m]));

    const members = [...group]
      .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
      .map((row): PoolViewMember => {
        const serving = eligibleIds.has(row.id);
        const why = excludedById.get(row.id);
        return {
          deploymentId: row.id,
          node: row.nodeName,
          runtime: row.runtime,
          port: row.port,
          inflight: outstanding(row.id),
          serving,
          ...(serving ? {} : { reason: why?.reason, detail: why?.detail }),
        };
      })
      .sort((a, b) => Number(b.serving) - Number(a.serving));

    views.push({ publishedName, members, servingCount: eligible.length });
  }

  return views.sort((a, b) =>
    a.publishedName < b.publishedName ? -1 : a.publishedName > b.publishedName ? 1 : 0,
  );
}
