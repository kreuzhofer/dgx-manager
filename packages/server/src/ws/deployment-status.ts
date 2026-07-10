/** The only status that means "this deployment is actually serving". */
export const HEALTHY_STATUS = "running";

/** Statuses after which the deployment holds no VRAM. */
const TERMINAL_STATUSES = ["stopped", "failed", "evicted"];

export function isTerminalDeploymentStatus(status: string): boolean {
  return TERMINAL_STATUSES.includes(status);
}

export interface AgentDeploymentStatus {
  status: string;
  port?: number | null;
  error?: string | null;
  vramActual?: number | string | null;
}

/** The Prisma `data` patch for one `agent:deployment:status` message. */
export interface DeploymentStatusUpdate {
  status: string;
  port?: number;
  error?: string | null;
  vramActual?: number;
}

/**
 * Pure: turn one agent status report into a Prisma update patch.
 *
 * The subtlety is `error`. A crashing deploy reports `failed` **with** an error and
 * then, moments later, `stopped` **without** one as the teardown completes. Writing
 * `error` unconditionally would erase the only record of why it died — which is how
 * a vLLM `ValueError: Free memory on device …` reached the dashboard as an
 * indistinguishable `stopped` (2026-07-09). So:
 *
 *   - a non-empty error is always persisted, whatever the status;
 *   - an absent error clears the column ONLY when the deployment is healthy again,
 *     so a stale failure does not haunt a later successful run;
 *   - otherwise the column is left alone.
 *
 * `port` and `vramActual` are likewise omitted when the agent did not report them,
 * so a bound port is never clobbered by a partial tick.
 */
export function deploymentStatusUpdate(
  msg: AgentDeploymentStatus,
): DeploymentStatusUpdate {
  const terminal = isTerminalDeploymentStatus(msg.status);
  const data: DeploymentStatusUpdate = { status: msg.status };

  if (msg.port != null) data.port = msg.port;

  if (terminal) {
    data.vramActual = 0;
  } else if (msg.vramActual != null && msg.vramActual !== "") {
    const n = Number(msg.vramActual);
    if (Number.isFinite(n)) data.vramActual = n;
  }

  if (msg.error) {
    data.error = msg.error;
  } else if (msg.status === HEALTHY_STATUS) {
    data.error = null;
  }

  return data;
}
