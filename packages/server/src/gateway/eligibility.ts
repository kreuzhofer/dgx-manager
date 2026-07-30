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
export function selectEligibleMembers(
  candidates: MemberCandidate[],
  isNodeOnline: LivenessCheck,
): EligibleMember[] {
  const eligible: EligibleMember[] = [];

  for (const c of candidates) {
    if (c.status !== HEALTHY_STATUS) continue;
    if (c.port == null) continue;
    if (!c.nodeIp) continue;
    if (!isNodeOnline(c.nodeId)) continue;

    eligible.push({
      deploymentId: c.id,
      nodeId: c.nodeId,
      baseUrl: `http://${c.nodeIp}:${c.port}`,
    });
  }

  return eligible;
}
