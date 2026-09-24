/**
 * Pre-flight VRAM admission check for vLLM deploys (solo and cluster) and
 * for restarts.
 *
 * Split into two layers:
 *   - {@link computeVramShortfall} and {@link reclaimForRestart} are pure
 *     functions over a node's known state. Easy to property-test: given a
 *     snapshot and a requested gpu_memory_utilization, return a shortfall or
 *     null.
 *   - {@link checkVllmVramAdmission} is the Prisma-coupled orchestrator
 *     that loads the per-node snapshots and conflict list, then delegates
 *     to the pure functions. Exercised via integration test.
 *
 * We never auto-evict; the caller is expected to surface the shortfalls to
 * the user as a 409 with the conflict list so they can decide what to stop.
 *
 * The one rule to keep in mind while reading: the manager measures nodes, never
 * deployments. A node's memory reading is the only measured quantity here;
 * every per-deployment figure is a claim. What a restart may claim back is its
 * authorised share and nothing more — see
 * docs/adr/0004-node-memory-attribution.md and CONTEXT.md § Node memory.
 */
import { prisma } from "../prisma.js";

export type VramConflict = {
  id: string;
  name: string | null;
  /**
   * What the holder is doing. For a deployment this is its status column; for a
   * fine-tune job it is the activity actually holding the memory (see
   * {@link fineTuneHoldingStatus}), which is not always its `status` column.
   */
  status: string;
  /**
   * What kind of holder this is. A refusal has to be readable — "held by
   * sql-lora-27b (running)" is ambiguous between a deployment you can restart
   * and a training run you would be throwing away hours of work to stop.
   */
  kind: "deployment" | "finetune";
  /**
   * Memory figures off the Deployment row, carried for display only — nothing
   * in admission reads them any more (ADR 0004, Decision 3). Always null for a
   * fine-tune job, which has no memory columns at all.
   */
  vramActualMB: number | null;
  vramEstimateMB: number | null;
};

/** Identifies the deployment being restarted, and the share it may reclaim. */
export type RestartingDeployment = {
  /** Excluded from the conflict list — it is not competing with itself. */
  deploymentId: string;
  /**
   * The gpu_memory_utilization this deployment was **authorised for**, i.e. the
   * one saved on its row — NOT the one the caller is requesting now. Restarting
   * at a larger share must not widen what the deployment is credited for
   * (ADR 0004, Decision 3).
   */
  authorisedGpuMemUtil: number;
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
  /**
   * VRAM (MB) the pending action will itself free before it needs it again —
   * the resident model of the deployment being restarted. Subtracted from
   * `vramUsedMB` so a restart isn't rejected for memory it is about to release.
   * Defaults to 0 (a fresh deploy reclaims nothing).
   *
   * Bounded by the restarting deployment's **authorised share** of the node —
   * see {@link reclaimForRestart}, which is how every caller should compute it.
   * It is emphatically NOT "whatever the node reading leaves unexplained": that
   * was #118, where a node busy with a training run computed as empty.
   */
  reclaimableMB?: number;
  /**
   * Everything the manager knows to be holding memory on this node: active
   * deployments (solo or cluster member) and in-flight fine-tune jobs. Used
   * only for the conflict list when a shortfall is reported — no arithmetic
   * reads it. Unattributed memory has no entry here by definition, so an empty
   * list on a full node is a real and expected answer.
   */
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
 * FineTuneJob `status` values that count as holding GPU memory. Mirrors
 * {@link ACTIVE_DEPLOYMENT_STATUSES} in including `pending`: a job about to
 * hold memory is treated as holding it.
 */
export const FINETUNE_HOLDING_STATUSES = ["pending", "starting", "running", "stopping"];

/**
 * What a fine-tune job is doing with a node's memory, or null when it is doing
 * nothing with it. A job holds memory across THREE independent columns, and the
 * one that names the largest allocation is not `status`:
 *
 *   - `mergeStatus: "running"` — merging loads the base model, easily the
 *     biggest allocation a job makes, and happens *after* `status` has gone
 *     `completed`.
 *   - `quantizationStatus: "quantizing"` — likewise post-training.
 *   - `status` — the training run itself.
 *
 * This is the single source of truth for "is this job holding the node", used
 * both to decide whether to list a job and to label it in the refusal. Merge
 * and quantization win over `status` when several apply, because they are the
 * larger claim and the more surprising one to be told about.
 */
export function fineTuneHoldingStatus(job: {
  status: string;
  mergeStatus: string | null;
  quantizationStatus: string | null;
}): string | null {
  if (job.mergeStatus === "running") return "merging";
  if (job.quantizationStatus === "quantizing") return "quantizing";
  if (FINETUNE_HOLDING_STATUSES.includes(job.status)) return job.status;
  return null;
}

/**
 * The memory (MB) a restart may be credited with on one node — its *reclaim*,
 * in the sense CONTEXT.md § Node memory gives that word.
 *
 * Bounded by two things and nothing else:
 *   - the restarting deployment's **authorised share** of this node
 *     (`round(vramTotal × its saved gpu_memory_utilization)`), because that is
 *     the most it was ever permitted to hold; and
 *   - the **node reading**, because it cannot release memory nobody is using.
 *
 * Everything else in the reading — a training container, a hand-started
 * benchmark, a survivor of an offboarded node — is *unattributed memory* and
 * counts against the restart. See docs/adr/0004-node-memory-attribution.md.
 *
 * `authorisedGpuMemUtil` is clamped to [0,1] and a non-finite value reclaims
 * nothing: it is recovered from a config blob, and a corrupt value must not
 * become a licence to reclaim more than a whole node.
 */
export function reclaimForRestart(
  nodeReadingMB: number,
  vramTotalMB: number,
  authorisedGpuMemUtil: number,
): number {
  // A non-finite share (a corrupt blob, a non-numeric `gpuMem` in a request body)
  // reclaims nothing: the restart is charged for the whole node reading and
  // refused. Without this the NaN propagates into every figure in the shortfall
  // and the refusal, while still correct, reports nulls instead of numbers.
  if (!Number.isFinite(authorisedGpuMemUtil)) return 0;
  const util = Math.min(1, Math.max(0, authorisedGpuMemUtil));
  const authorisedShareMB = Math.round(vramTotalMB * util);
  return Math.max(0, Math.min(authorisedShareMB, nodeReadingMB));
}

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
  // Discount VRAM the action will free before it needs it (a restart releasing
  // its own resident model). Clamp the reclaim so a bogus/negative value can't
  // inflate availability, and never let used drop below zero.
  const reclaimable = Math.max(0, snapshot.reclaimableMB ?? 0);
  const vramUsed = Math.max(0, snapshot.vramUsedMB - reclaimable);
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
 * Render one holder for the refusal. A fine-tune job is labelled as such:
 * stopping a deployment costs a relaunch, stopping a training run costs hours,
 * and the user cannot weigh that if the two read identically.
 */
function describeHolder(c: VramConflict): string {
  const label = c.name || c.id.slice(0, 8);
  return c.kind === "finetune"
    ? `${label} (${c.status}, fine-tune job)`
    : `${label} (${c.status})`;
}

/**
 * Format a list of shortfalls as a single-line human-readable message
 * suitable for an HTTP error body. Pure.
 *
 * Closes with the two ways through a refusal, because admission never evicts
 * and there is deliberately no override flag (ADR 0004, Decision 5): free the
 * memory, or ask for less of it.
 */
export function vramShortfallMessage(shortfalls: VramShortfall[]): string {
  if (shortfalls.length === 0) return "";
  const perNode = shortfalls
    .map((s) => {
      const nodeLabel = s.nodeName || s.nodeId.slice(0, 12);
      const requested = Math.round(s.vramRequestedMB / 1024);
      const margin = Math.round(s.vramSafetyMarginMB / 1024);
      const threshold = Math.round(s.vramThresholdMB / 1024);
      const available = Math.round(s.vramAvailableMB / 1024);
      const total = Math.round(s.vramTotalMB / 1024);
      const conflictPart = s.conflicts.length > 0
        ? ` — held by: ${s.conflicts.map(describeHolder).join(", ")}`
        : "";
      return `${nodeLabel}: needs ${threshold} GB free (${requested} GB requested + ${margin} GB safety margin) but only ${available} GB free of ${total} GB${conflictPart}`;
    })
    .join("; ");
  return `${perNode}. Free memory on the node (stop a holder), or ask for a smaller share with config.gpuMem`;
}

/**
 * Prisma-coupled orchestrator. For each node id, loads the node, its latest
 * metric, and its known holders (active deployments and in-flight fine-tune
 * jobs), then calls {@link computeVramShortfall}. Returns shortfalls in input
 * order so the error message is stable.
 *
 * Pass `restarting` to check a restart rather than a fresh deploy: it both
 * excludes the deployment from its own conflict list and gives it a bounded
 * reclaim. Omit it and the node reading is used as-is.
 */
export async function checkVllmVramAdmission(
  nodeIds: string[],
  requestedGpuMemUtil: number,
  restarting?: RestartingDeployment,
): Promise<VramShortfall[]> {
  const shortfalls: VramShortfall[] = [];
  for (const nid of nodeIds) {
    const node = await prisma.node.findUnique({ where: { id: nid } });
    if (!node) continue;
    const latestMetric = await prisma.metricSnapshot.findFirst({
      where: { nodeId: nid },
      orderBy: { timestamp: "desc" },
    });
    const conflicts = await loadConflicts(nid, restarting?.deploymentId);

    const vramTotalMB = node.vramTotal || DEFAULT_VRAM_TOTAL_MB;
    const vramUsedMB = latestMetric?.vramUsed || 0;

    // On a restart the node reading still includes the restarting deployment's
    // own resident model, which is torn down before relaunch — so it is not
    // charged for that. The credit is bounded by the share it was authorised
    // for, and by nothing else.
    //
    // #1 got here first and bounded it by "the reading minus everything
    // attributable to OTHER deployments" instead. That reasoning was sound as
    // far as it went: it avoided depending on whether a deployment's recorded
    // memory is per-node or summed across a cluster, which the schema never
    // settled. What it did not consider is that a node contains things the
    // manager has no row for — a training run, a container surviving an
    // offboarded node, a hand-started benchmark. All of that landed in "not
    // attributable to another deployment" and was handed to the restart, so a
    // node with nothing else deployed on it computed as empty however full it
    // was, and the restart was admitted unconditionally (#118).
    //
    // The other-deployment subtraction is gone rather than kept as a second
    // bound: the column it summed holds the whole NODE's reading for vLLM and
    // dgxrun, so two deployments sharing a node over-counted each other, the
    // credit clamped to zero, and the restart was refused for memory that was
    // its own — #1's symptom from the other side. Admission no longer reads
    // that column. See docs/adr/0004-node-memory-attribution.md.
    const reclaimableMB = restarting
      ? reclaimForRestart(vramUsedMB, vramTotalMB, restarting.authorisedGpuMemUtil)
      : 0;

    const result = computeVramShortfall(
      {
        nodeId: nid,
        nodeName: node.name,
        vramTotalMB,
        vramUsedMB,
        reclaimableMB,
        conflicts,
      },
      requestedGpuMemUtil,
    );
    if (result) shortfalls.push(result);
  }
  return shortfalls;
}

/**
 * Everything the manager knows to be holding memory on one node: active
 * deployments and in-flight fine-tune jobs. Explanation only — no arithmetic
 * reads the result (ADR 0004, Decisions 3 and 5).
 *
 * Fine-tune jobs are here because a refusal that cannot name what holds the
 * memory has failed at half its job, and training was the one holder the
 * product could see but never mention.
 */
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
  // Every fine-tune job touching this node, as head (`nodeId`) or as a member
  // of a multi-node run (`clusterNodes`) — a worker node is held by the job just
  // as the head is. Filtered in JS rather than in the `where` clause so
  // `fineTuneHoldingStatus` stays the single definition of "holding", spread
  // over three columns as it is; the table holds tens of rows, not millions.
  const fineTuneJobs = await prisma.fineTuneJob.findMany({
    where: { OR: [{ nodeId }, { clusterNodes: { some: { nodeId } } }] },
    select: {
      id: true,
      displayName: true,
      status: true,
      mergeStatus: true,
      quantizationStatus: true,
    },
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
      kind: "deployment",
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
      kind: "deployment",
      vramActualMB: d.vramActual,
      vramEstimateMB: d.vramEstimate,
    });
  }
  for (const job of fineTuneJobs) {
    const holding = fineTuneHoldingStatus(job);
    if (!holding) continue;
    conflicts.push({
      id: job.id,
      // Same label the dashboard and the fine-tune deploy route use for an
      // unnamed job, so the refusal names it the way the user sees it.
      name: job.displayName || `finetune-${job.id.slice(0, 8)}`,
      status: holding,
      kind: "finetune",
      // A FineTuneJob has no memory columns. Nothing reads these.
      vramActualMB: null,
      vramEstimateMB: null,
    });
  }
  return conflicts;
}
