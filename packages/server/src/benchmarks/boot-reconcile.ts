import { prisma } from "../prisma.js";
import { reconcileAction } from "./reconcile.js";
import { executeRun } from "./execute.js";
import type { CapInvoker, JobStatus } from "./remote-runner.js";

/**
 * On boot, decide what to do with every BenchmarkRun still marked pending/running.
 *
 * Fire-and-forget from index.ts so a slow/offline eval agent never blocks server
 * startup. Legacy (local) runs died with the old container → failed. Remote runs
 * are systemd units that probably survived → resume the poll loop (which self-
 * corrects: a finished job finalizes on the first tick, a vanished one fails).
 * An agent unreachable at boot yields `unknown`/null → resume, never a false fail.
 */
export async function reconcileStaleRuns(invoke: CapInvoker): Promise<void> {
  const stale = await prisma.benchmarkRun.findMany({
    where: { status: { in: ["pending", "running"] } },
    select: { id: true, kind: true, config: true, endpointUrl: true, servedModelName: true, runnerNodeId: true, logOffset: true },
  });
  for (const row of stale) {
    if (!row.runnerNodeId) {
      await prisma.benchmarkRun.update({
        where: { id: row.id },
        data: { status: "failed", error: "server restarted before run completed", completedAt: new Date() },
      });
      continue;
    }
    let status: JobStatus | null = null;
    try {
      const r = await invoke(row.runnerNodeId, "job.status", { runId: row.id });
      status = r.ok ? (r.data as JobStatus) : null;
    } catch { status = null; }
    const action = reconcileAction(row, status);
    if (action === "fail-orphan") {
      await prisma.benchmarkRun.update({
        where: { id: row.id },
        data: { status: "failed", error: "job vanished across manager restart", completedAt: new Date() },
      });
    } else {
      // resume OR finalize: re-attach the poll loop (skipStart). It returns
      // immediately for an already-exited job and finalizes it; keeps polling
      // for an active one.
      executeRun(row, invoke, true);
    }
  }
}
