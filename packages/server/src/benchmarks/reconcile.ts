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
 */
export function reconcileAction(
  run: { runnerNodeId: string | null },
  status: JobStatus | null,
): "resume" | "finalize" | "fail-orphan" | "fail-legacy" {
  if (!run.runnerNodeId) return "fail-legacy";
  if (status === null) return "resume";
  switch (status.kind) {
    case "active": return "resume";
    case "unknown": return "resume";
    case "exited": return "finalize";
    case "missing": return "fail-orphan";
  }
}
