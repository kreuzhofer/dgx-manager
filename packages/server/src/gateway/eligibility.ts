import { HEALTHY_STATUS } from "../ws/deployment-status.js";

/**
 * Which members of a pool may actually take a request.
 *
 * Pure so the rule can be stated as an invariant rather than inferred from a
 * 503. Every condition below is the difference between a served request and one
 * that hangs, so all of them must hold.
 */

/** A deployment serving the requested published name, before filtering. */
export interface MemberCandidate {
  id: string;
  status: string;
  port: number | null;
  nodeId: string;
  nodeIp: string | null;
}

/** A member that can be routed to right now. */
export interface EligibleMember {
  deploymentId: string;
  nodeId: string;
  /** `http://<ip>:<port>` — the OpenAI path is appended by the caller. */
  baseUrl: string;
}

/** Does this node have a live agent connection this instant? */
export type LivenessCheck = (nodeId: string) => boolean;

/** Why a member of the pool cannot take a request. */
export type ExclusionReason = "not-running" | "no-port" | "no-node-address" | "agent-offline";

export interface ExcludedMember {
  deploymentId: string;
  nodeId: string;
  reason: ExclusionReason;
  /** Human-readable, for the refusal body. */
  detail: string;
}

/** The whole pool, split into who can serve and who cannot — and why not. */
export interface PoolAssessment {
  eligible: EligibleMember[];
  excluded: ExcludedMember[];
}

/**
 * Filter a pool down to the members that can serve.
 *
 * Liveness comes from the agent hub rather than the deployment's status column:
 * the column can lag a dead node by up to the staleness sweep interval, and a
 * request sent into that window hangs until the socket times out instead of
 * failing fast.
 *
 * Order is preserved; choosing *between* eligible members is the caller's job.
 */
export function assessPool(
  candidates: MemberCandidate[],
  isNodeOnline: LivenessCheck,
): PoolAssessment {
  const eligible: EligibleMember[] = [];
  const excluded: ExcludedMember[] = [];

  const exclude = (c: MemberCandidate, reason: ExclusionReason, detail: string) =>
    excluded.push({ deploymentId: c.id, nodeId: c.nodeId, reason, detail });

  for (const c of candidates) {
    if (c.status !== HEALTHY_STATUS) {
      exclude(c, "not-running", `deployment is '${c.status}', not '${HEALTHY_STATUS}'`);
      continue;
    }
    if (c.port == null) {
      exclude(c, "no-port", "deployment has no port bound");
      continue;
    }
    if (!c.nodeIp) {
      exclude(c, "no-node-address", "node has no known address");
      continue;
    }
    if (!isNodeOnline(c.nodeId)) {
      exclude(c, "agent-offline", "node has no live agent connection");
      continue;
    }

    eligible.push({
      deploymentId: c.id,
      nodeId: c.nodeId,
      baseUrl: `http://${c.nodeIp}:${c.port}`,
    });
  }

  return { eligible, excluded };
}

/** Just the members that can serve. */
export function selectEligibleMembers(
  candidates: MemberCandidate[],
  isNodeOnline: LivenessCheck,
): EligibleMember[] {
  return assessPool(candidates, isNodeOnline).eligible;
}
