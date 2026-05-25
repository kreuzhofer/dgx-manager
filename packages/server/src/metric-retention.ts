import { prisma } from "./prisma.js";

/**
 * Delete MetricSnapshot rows strictly older than `before`. Returns the
 * number of rows deleted. The /api/nodes hot path needs the table to stay
 * bounded — without retention, 4 nodes ticking every 5s grow it by ~69k
 * rows/day, which made the unindexed query a 2.4M-row scan in production.
 */
export async function pruneMetricsOlderThan(before: Date): Promise<number> {
  const { count } = await prisma.metricSnapshot.deleteMany({
    where: { timestamp: { lt: before } },
  });
  return count;
}

interface RetentionOpts {
  retentionDays: number;
  intervalMs: number;
}

/**
 * Start the periodic retention loop. Runs immediately on call, then every
 * `intervalMs`. Returns a stop function. Errors are logged but never thrown
 * — the loop must survive transient DB issues.
 */
export function startMetricRetention(opts: RetentionOpts): () => void {
  const { retentionDays, intervalMs } = opts;

  const run = async () => {
    const before = new Date(Date.now() - retentionDays * 86400 * 1000);
    try {
      const n = await pruneMetricsOlderThan(before);
      if (n > 0) {
        console.log(`[metric-retention] pruned ${n} rows older than ${before.toISOString()}`);
      }
    } catch (err) {
      console.error("[metric-retention] prune failed:", err);
    }
  };

  void run();
  const handle = setInterval(run, intervalMs);
  return () => clearInterval(handle);
}
