// Pure selection logic for the bulk agent-update endpoint (POST
// /api/nodes/update-agent-all). Split from the route so the "which nodes get a
// cmd:update" decision is unit-testable without a DB or WebSocket hub.

export interface AgentNode {
  id: string;
  name: string;
  agentVersion: string | null;
  arch?: string | null;
}

export interface AgentUpdatePlan {
  /** Online nodes not already on the target version (or all online nodes when force). */
  toUpdate: AgentNode[];
  /** Online nodes already on the target version (skipped unless force). */
  skipped: AgentNode[];
  /** Nodes whose agent isn't connected — can't be pushed to. */
  offline: AgentNode[];
}

/**
 * Partition nodes for a bulk agent roll:
 * - offline (agent not connected) → can't update
 * - already on targetVersion → skipped (unless `force`)
 * - otherwise → toUpdate
 *
 * `isOnline` is injected (AgentHub.isAgentOnline) so this stays pure.
 */
export function planAgentUpdate(
  nodes: AgentNode[],
  targetVersion: string,
  isOnline: (id: string) => boolean,
  force = false,
): AgentUpdatePlan {
  const plan: AgentUpdatePlan = { toUpdate: [], skipped: [], offline: [] };
  for (const n of nodes) {
    if (!isOnline(n.id)) plan.offline.push(n);
    else if (!force && n.agentVersion === targetVersion) plan.skipped.push(n);
    else plan.toUpdate.push(n);
  }
  return plan;
}

/** The arch query suffix for GET /api/agent/bundle (matches the single-node path). */
export function bundleArchQuery(arch: string | null | undefined): string {
  return arch === "amd64" || arch === "arm64" ? `?arch=${arch}` : "";
}
