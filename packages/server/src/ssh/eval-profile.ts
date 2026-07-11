import { isEvalNode } from "../nodes/role.js";
import type { PrereqCheck } from "./provisioner.js";

/** A benchmark-runner node needs only the agent runtime + uv. Nothing model-hosting. */
export const EVAL_NODE_CHECK_NAMES: readonly string[] = ["Node.js", "uv (uvx)"];

export interface PrereqCheckLike { name: string }

/**
 * Narrow the audit's prerequisite checks to what a node's role needs.
 *
 * Identity for a normal (gpu/legacy) node. For an `eval` node it drops the whole
 * model-hosting stack (Docker, the NVIDIA toolkit, sparkrun, NVIDIA drivers) and
 * Ollama — agenthost runs a hand-installed Ollama that provisioning must not
 * touch — leaving only the agent runtime (Node.js) and uv (for `uvx lm_eval`).
 */
export function checksForRole<T extends PrereqCheckLike>(items: T[], role: string | null | undefined): T[] {
  if (!isEvalNode(role)) return items;
  return items.filter((c) => EVAL_NODE_CHECK_NAMES.includes(c.name));
}

/**
 * An eval node's job capability runs `sudo -n systemd-run`, so passwordless sudo
 * is a hard requirement — surface it at onboarding, not on the first 80-minute
 * benchmark.
 */
export function evalSudoCheck(sudoAvailable: boolean): PrereqCheck {
  return {
    name: "Passwordless sudo (eval)",
    status: sudoAvailable ? "green" : "red",
    detail: sudoAvailable
      ? "Available — job.* can run `sudo -n systemd-run`"
      : "MISSING — an eval node needs passwordless sudo for `sudo -n systemd-run`; add a NOPASSWD sudoers rule",
  };
}
