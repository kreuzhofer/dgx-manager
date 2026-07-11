import { isEvalNode } from "../nodes/role.js";

export interface EvalNodeCandidate { id: string; name: string; role: string | null; status: string }

export type EvalNodeResolution =
  | { ok: true; nodeId: string }
  | { ok: false; reason: "none" | "ambiguous"; detail: string };

/**
 * Pick the node that runs benchmarks. Exactly one online `eval` node is expected.
 *
 * Ambiguity is an error, not a coin flip: silently taking the first would make a
 * run's provenance depend on row ordering, and `BenchmarkRun.runnerNodeId` exists
 * precisely so that throughput numbers can be trusted to a host.
 */
export function resolveEvalNode(
  nodes: EvalNodeCandidate[],
  explicitId?: string,
): EvalNodeResolution {
  const online = nodes.filter((n) => isEvalNode(n.role) && n.status === "online");
  if (explicitId) {
    const hit = online.find((n) => n.id === explicitId);
    return hit
      ? { ok: true, nodeId: hit.id }
      : { ok: false, reason: "none", detail: `EVAL_NODE_ID=${explicitId} is not an online eval node` };
  }
  if (online.length === 0) {
    return { ok: false, reason: "none", detail: "no online node with role \"eval\"" };
  }
  if (online.length > 1) {
    return {
      ok: false,
      reason: "ambiguous",
      detail: `multiple online eval nodes (${online.map((n) => n.name).join(", ")}); set EVAL_NODE_ID`,
    };
  }
  return { ok: true, nodeId: online[0].id };
}
