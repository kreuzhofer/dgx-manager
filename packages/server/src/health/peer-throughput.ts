/**
 * Detect a degraded node by comparing its serving throughput against the other
 * members of its pool (#88).
 *
 * Split into two layers, following admission/vram.ts:
 *   - {@link evaluatePool} is pure — the caller supplies each member's samples
 *     for the window. Property-tested without a DB.
 *   - {@link assessThroughput} is the Prisma-coupled orchestrator that loads
 *     the window and the per-node deployment counts, then delegates.
 *     Exercised via integration test.
 *
 * Why throughput and not a bandwidth probe: a probe must never run against a
 * GPU that is serving, but the incident this detects (a node at half memory
 * bandwidth for ~30 hours) happened *while* the node served continuously. The
 * detector has to work on a busy node, so it uses telemetry already collected
 * and does no GPU work at all.
 */
import { prisma } from "../prisma.js";

/** One pool member's window of throughput samples. */
export interface MemberInput {
  deploymentId: string;
  nodeId: string;
  nodeName: string;
  modelId: string;
  /** Running deployments on this member's node. */
  deploymentsOnNode: number;
  /** Non-null throughput samples (tokens/s) inside the window. */
  rates: number[];
}

export type MemberState = "ok" | "suspect" | "not-comparable";

/** Why a member could not be compared. Absent when a verdict was reached. */
export type NotComparableReason =
  | "single-member"
  | "insufficient-samples"
  | "idle"
  | "multi-deployment-node"
  | "model-mismatch"
  | "no-comparable-peers";

export interface MemberVerdict {
  deploymentId: string;
  node: string;
  state: MemberState;
  /** This member's median rate over the window. */
  rate: number | null;
  peerRates: { node: string; rate: number }[];
  /** rate / median(peerRates). */
  ratio: number | null;
  reason?: NotComparableReason;
}

/** Below this share of its peers' median, a member is suspect. */
export const SUSPECT_RATIO = 0.8;

/**
 * Samples a member needs in the window before its median means anything —
 * 5 minutes of continuous serving at the 5s metric tick.
 *
 * Calibrated against the real series: in 34 hours of paired load between two
 * healthy pool members, the only hour whose ratio looked degraded (0.647) came
 * from one and two samples. Everything at full sampling stayed above 0.86.
 */
export const MIN_SAMPLES = 60;

export function median(xs: number[]): number | null {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length / 2;
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[mid - 1] + s[mid]) / 2;
}

export function evaluatePool(members: MemberInput[]): MemberVerdict[] {
  // A node's reported rate is the sum over every deployment running on it, so
  // it only attributes to this member when it is the only one there.
  const attributable = (m: MemberInput) => m.deploymentsOnNode <= 1;
  const comparable = (m: MemberInput) => attributable(m) && m.rates.length >= MIN_SAMPLES;
  const rateOf = new Map(members.map((m) => [m.deploymentId, median(m.rates)]));

  // A pool is deployments sharing a published name, which does not make them
  // the same model (#90). Mixed models measure different work, so nothing in
  // the pool is comparable — refusing wholesale beats picking a reference.
  const mixedModels = new Set(members.map((m) => m.modelId)).size > 1;

  return members.map((m): MemberVerdict => {
    const rate = rateOf.get(m.deploymentId)!;
    const peerRates = members
      .filter((p) => p.deploymentId !== m.deploymentId && comparable(p))
      .flatMap((p) => {
        const r = rateOf.get(p.deploymentId);
        return r === null || r === undefined ? [] : [{ node: p.nodeName, rate: r }];
      });

    const peerMedian = median(peerRates.map((p) => p.rate));
    const ratio = rate !== null && peerMedian ? rate / peerMedian : null;

    const base = { deploymentId: m.deploymentId, node: m.nodeName, rate, peerRates, ratio };
    // Structural reasons first: a lone member stays uncomparable however much
    // traffic it takes, so reporting "idle" there would send someone to wait
    // for a verdict that can never arrive.
    if (members.length === 1) return { ...base, state: "not-comparable", reason: "single-member" };
    if (mixedModels) return { ...base, state: "not-comparable", reason: "model-mismatch" };
    if (!attributable(m)) return { ...base, state: "not-comparable", reason: "multi-deployment-node" };
    if (m.rates.length === 0) return { ...base, state: "not-comparable", reason: "idle" };
    if (!comparable(m)) return { ...base, state: "not-comparable", reason: "insufficient-samples" };
    // Peers exist but none of them are usable this window — not the same
    // thing as having no peers at all.
    if (peerRates.length === 0) return { ...base, state: "not-comparable", reason: "no-comparable-peers" };

    return { ...base, state: ratio !== null && ratio < SUSPECT_RATIO ? "suspect" : "ok" };
  });
}

/** How far back a member's samples are drawn from. */
export const WINDOW_MINUTES = 60;

/** A published deployment, as the pool view knows it. */
export interface PoolMemberRef {
  deploymentId: string;
  nodeId: string;
  nodeName: string;
  modelId: string;
  publishedName: string;
}

export interface ThroughputVerdict extends MemberVerdict {
  windowMinutes: number;
}

/**
 * Judge every published member against the peers sharing its published name.
 *
 * Returns verdicts keyed by deployment id. Members are grouped by published
 * name — the pool — because that is the set running comparable work; a node
 * hosting members of two pools is judged once per pool.
 */
export async function assessThroughput(
  members: PoolMemberRef[],
  now: Date = new Date(),
): Promise<Map<string, ThroughputVerdict>> {
  const verdicts = new Map<string, ThroughputVerdict>();
  if (members.length === 0) return verdicts;

  const nodeIds = [...new Set(members.map((m) => m.nodeId))];
  const since = new Date(now.getTime() - WINDOW_MINUTES * 60_000);

  const [samples, running] = await Promise.all([
    prisma.metricSnapshot.findMany({
      where: { nodeId: { in: nodeIds }, timestamp: { gte: since }, tps: { not: null } },
      select: { nodeId: true, tps: true },
    }),
    // Every running deployment on these nodes, not just the published ones:
    // the node's reported rate is the sum over all of them.
    prisma.deployment.groupBy({
      by: ["nodeId"],
      where: { nodeId: { in: nodeIds }, status: "running" },
      _count: { _all: true },
    }),
  ]);

  const ratesByNode = new Map<string, number[]>();
  for (const s of samples) {
    if (s.tps === null) continue;
    const rates = ratesByNode.get(s.nodeId);
    if (rates) rates.push(s.tps);
    else ratesByNode.set(s.nodeId, [s.tps]);
  }
  const countByNode = new Map(running.map((r) => [r.nodeId, r._count._all]));

  const pools = new Map<string, PoolMemberRef[]>();
  for (const m of members) {
    const pool = pools.get(m.publishedName);
    if (pool) pool.push(m);
    else pools.set(m.publishedName, [m]);
  }

  for (const pool of pools.values()) {
    const evaluated = evaluatePool(
      pool.map((m) => ({
        deploymentId: m.deploymentId,
        nodeId: m.nodeId,
        nodeName: m.nodeName,
        modelId: m.modelId,
        deploymentsOnNode: countByNode.get(m.nodeId) ?? 1,
        rates: ratesByNode.get(m.nodeId) ?? [],
      })),
    );
    for (const v of evaluated) verdicts.set(v.deploymentId, { ...v, windowMinutes: WINDOW_MINUTES });
  }

  return verdicts;
}
