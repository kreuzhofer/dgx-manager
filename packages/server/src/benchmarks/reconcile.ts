import type { JobStatus } from "./remote-runner.js";

/**
 * What to do at boot with a BenchmarkRun still marked pending/running.
 *
 * A run WITHOUT a runnerNodeId executed as a child of the old server container
 * and died with it — the pre-existing contract, preserved.
 *
 * A run WITH one is a systemd unit on the eval node and probably survived. If the
 * agent is unreachable now (`null`) or systemd could not answer (`unknown`), resume
 * and let the poll loop discover the truth. Declaring it dead because we could not
 * ask is the mistake this whole design exists to avoid.
 *
 * `usesManagerProxy` is the one case where a STILL-RUNNING job is nonetheless
 * doomed: an accuracy run talks to a reasoning proxy bound inside the manager
 * process on an ephemeral port, so the previous container took it with it (#22).
 * That is not a guess about the job — it is a fact about the proxy, which is why
 * it can override an `active` status when an unreachable agent cannot.
 */
export function reconcileAction(
  run: { runnerNodeId: string | null; usesManagerProxy?: boolean },
  status: JobStatus | null,
): "resume" | "finalize" | "fail-orphan" | "fail-legacy" | "fail-proxy-lost" {
  if (!run.runnerNodeId) return "fail-legacy";
  if (status === null) return "resume";
  switch (status.kind) {
    // Alive, but if it depended on our proxy it is now talking to a closed
    // socket and every remaining item will fail. Ending it deliberately beats
    // letting it burn the GPU to a urllib3 traceback — and it can be SAID.
    case "active": return run.usesManagerProxy ? "fail-proxy-lost" : "resume";
    // Deliberately NOT short-circuited by usesManagerProxy: we could not ask, so
    // we cannot tell a doomed job from one that already finished successfully
    // before the restart. Resume and let the poll loop find out.
    case "unknown": return "resume";
    case "exited": return "finalize";
    case "missing": return "fail-orphan";
  }
}
