/**
 * Pre-flight VRAM admission check for vLLM deploys (solo and cluster) and
 * for restarts.
 *
 * Split into two layers:
 *   - {@link computeVramShortfall} is a pure function over a node's known
 *     state. Easy to property-test: given a snapshot and a requested
 *     gpu_memory_utilization, return a shortfall or null.
 *   - {@link checkVllmVramAdmission} is the Prisma-coupled orchestrator
 *     that loads the per-node snapshots and conflict list, then delegates
 *     to the pure function. Exercised via integration test.
 *
 * We never auto-evict; the caller is expected to surface the shortfalls to
 * the user as a 409 with the conflict list so they can decide what to stop.
 */
import { prisma } from "../prisma.js";

export type VramConflict = {
  id: string;
  name: string | null;
  status: string;
  vramActualMB: number | null;
  vramEstimateMB: number | null;
};

export type VramShortfall = {
  nodeId: string;
  nodeName: string | null;
  vramTotalMB: number;
  vramUsedMB: number;
  vramAvailableMB: number;
  vramRequestedMB: number;
  /** requested + safety margin — the actual threshold available has to clear. */
  vramThresholdMB: number;
  vramSafetyMarginMB: number;
  conflicts: VramConflict[];
};

export type NodeSnapshot = {
  nodeId: string;
  nodeName: string | null;
  /** Total VRAM in MB. Falls back to 128 GB when unknown. */
  vramTotalMB: number;
  /** VRAM currently in use in MB (most recent metric). */
  vramUsedMB: number;
  /** Active deployments on this node (solo or cluster member). Used only for the conflict list when a shortfall is reported. */
  conflicts: VramConflict[];
};

/** Default safety-margin fraction of total VRAM to keep free for driver state. */
export const SAFETY_MARGIN_FRACTION = 0.05;

/** Default total-VRAM fallback when a node hasn't reported its size. */
export const DEFAULT_VRAM_TOTAL_MB = 128_000;

/** Statuses that count as "actively holding GPU memory" when listing conflicts. */
export const ACTIVE_DEPLOYMENT_STATUSES = [
  "pending",
  "running",
  "starting",
  "building",
  "downloading",
  "launching",
  "loading",
  "restarting",
];

/**
 * Pure decision function: given a node's snapshot and a requested
 * gpu_memory_utilization (0..1), returns a {@link VramShortfall} when the
 * node can't admit the deploy, or null when it can.
 *
 * Invariants (asserted by property tests):
 *   - If `vramAvailable >= requested + safetyMargin`, returns null.
 *   - If `vramAvailable < requested + safetyMargin`, returns a shortfall.
 *   - The returned shortfall always has `vramThresholdMB > vramAvailableMB`.
 *   - All MB values are non-negative when inputs are non-negative.
 */
export function computeVramShortfall(
  snapshot: NodeSnapshot,
  gpuMemUtil: number,
  safetyMarginFraction: number = SAFETY_MARGIN_FRACTION,
): VramShortfall | null {
  const vramTotal = snapshot.vramTotalMB;
  const vramUsed = snapshot.vramUsedMB;
  const vramAvailable = Math.max(0, vramTotal - vramUsed);
  const vramRequested = Math.round(vramTotal * gpuMemUtil);
  const safetyMargin = Math.round(vramTotal * safetyMarginFraction);
  const vramThreshold = vramRequested + safetyMargin;

  if (vramAvailable >= vramThreshold) return null;

  return {
    nodeId: snapshot.nodeId,
    nodeName: snapshot.nodeName,
    vramTotalMB: vramTotal,
    vramUsedMB: vramUsed,
    vramAvailableMB: vramAvailable,
    vramRequestedMB: vramRequested,
    vramThresholdMB: vramThreshold,
    vramSafetyMarginMB: safetyMargin,
    conflicts: snapshot.conflicts,
  };
}

/**
 * Format a list of shortfalls as a single-line human-readable message
 * suitable for an HTTP error body. Pure.
 */
export function vramShortfallMessage(shortfalls: VramShortfall[]): string {
  return shortfalls
    .map((s) => {
      const nodeLabel = s.nodeName || s.nodeId.slice(0, 12);
      const requested = Math.round(s.vramRequestedMB / 1024);
      const margin = Math.round(s.vramSafetyMarginMB / 1024);
      const threshold = Math.round(s.vramThresholdMB / 1024);
      const available = Math.round(s.vramAvailableMB / 1024);
      const total = Math.round(s.vramTotalMB / 1024);
      const conflictPart = s.conflicts.length > 0
        ? ` — held by: ${s.conflicts.map((c) => `${c.name || c.id.slice(0, 8)} (${c.status})`).join(", ")}`
        : "";
      return `${nodeLabel}: needs ${threshold} GB free (${requested} GB requested + ${margin} GB safety margin) but only ${available} GB free of ${total} GB${conflictPart}`;
    })
    .join("; ");
}

/**
 * Prisma-coupled orchestrator. For each node id, loads the node + its
 * latest metric + active deployments using its GPU, then calls
 * {@link computeVramShortfall}. Returns shortfalls in input order so the
 * error message is stable.
 */
export async function checkVllmVramAdmission(
  nodeIds: string[],
  gpuMemUtil: number,
  excludeDeploymentId?: string,
): Promise<VramShortfall[]> {
  const shortfalls: VramShortfall[] = [];
  for (const nid of nodeIds) {
    const node = await prisma.node.findUnique({ where: { id: nid } });
    if (!node) continue;
    const latestMetric = await prisma.metricSnapshot.findFirst({
      where: { nodeId: nid },
      orderBy: { timestamp: "desc" },
    });
    const conflicts = await loadConflicts(nid, excludeDeploymentId);

    const result = computeVramShortfall(
      {
        nodeId: nid,
        nodeName: node.name,
        vramTotalMB: node.vramTotal || DEFAULT_VRAM_TOTAL_MB,
        vramUsedMB: latestMetric?.vramUsed || 0,
        conflicts,
      },
      gpuMemUtil,
    );
    if (result) shortfalls.push(result);
  }
  return shortfalls;
}

async function loadConflicts(
  nodeId: string,
  excludeDeploymentId?: string,
): Promise<VramConflict[]> {
  const soloConflicts = await prisma.deployment.findMany({
    where: {
      nodeId,
      status: { in: ACTIVE_DEPLOYMENT_STATUSES },
      ...(excludeDeploymentId ? { id: { not: excludeDeploymentId } } : {}),
    },
    include: { model: true },
  });
  const clusterConflicts = await prisma.clusterNode.findMany({
    where: {
      nodeId,
      deployment: {
        status: { in: ACTIVE_DEPLOYMENT_STATUSES },
        ...(excludeDeploymentId ? { id: { not: excludeDeploymentId } } : {}),
      },
    },
    include: { deployment: { include: { model: true } } },
  });
  const seen = new Set<string>();
  const conflicts: VramConflict[] = [];
  for (const d of soloConflicts) {
    if (seen.has(d.id)) continue;
    seen.add(d.id);
    conflicts.push({
      id: d.id,
      name: d.model?.name ?? null,
      status: d.status,
      vramActualMB: d.vramActual,
      vramEstimateMB: d.vramEstimate,
    });
  }
  for (const cn of clusterConflicts) {
    const d = cn.deployment;
    if (seen.has(d.id)) continue;
    seen.add(d.id);
    conflicts.push({
      id: d.id,
      name: d.model?.name ?? null,
      status: d.status,
      vramActualMB: d.vramActual,
      vramEstimateMB: d.vramEstimate,
    });
  }
  return conflicts;
}
