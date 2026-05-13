/**
 * Pure helper: render a FineTuneJob's cluster membership as a single
 * human-readable line for the dashboard "details" panel.
 *
 * Examples:
 *   single-node:     "dgx-spark-01"
 *   single-node, no node relation:  "<nodeId>"
 *   multi-node:      "4 nodes: dgx-spark-01 (head), dgx-spark-02, dgx-spark-03, dgx-spark-04"
 */

interface ClusterNodeRow {
  node: { name: string; ipAddress: string };
  role: string;
}

export interface JobClusterShape {
  nodeId: string;
  node?: { name: string; ipAddress: string } | null;
  clusterNodes: ClusterNodeRow[];
}

export function formatClusterSummary(job: JobClusterShape): string {
  if (job.clusterNodes.length === 0) {
    return job.node?.name ?? job.nodeId;
  }

  // Head first, then workers in their existing order.
  const sorted = [...job.clusterNodes].sort((a, b) => {
    if (a.role === "head" && b.role !== "head") return -1;
    if (b.role === "head" && a.role !== "head") return 1;
    return 0;
  });

  const count = sorted.length;
  const noun = count === 1 ? "node" : "nodes";
  const rendered = sorted
    .map((c) => (c.role === "head" ? `${c.node.name} (head)` : c.node.name))
    .join(", ");
  return `${count} ${noun}: ${rendered}`;
}
