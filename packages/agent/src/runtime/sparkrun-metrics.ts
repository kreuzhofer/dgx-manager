import { loadDeployments } from "./deployment-store.js";
import { isWorkloadRunning, inspectSparkrunContainer, snapshotContainerLogs } from "./sparkrun.js";
import type { VllmStatus } from "./vllm.js";

const CRASH_LOOP_THRESHOLD = 2;

/**
 * Pick a representative error line from captured container logs.
 * Looks for the first line that contains a recognisable error keyword.
 */
export function firstErrorLine(text: string): string | undefined {
  const m = text.split("\n").find((l) => /error|invalid|traceback|exception|keyerror|raise |fatal|not found/i.test(l));
  return m?.trim();
}

export interface VllmMetrics {
  numRequestsRunning?: number;
  kvCacheUsagePerc?: number;
}

/**
 * Parse a single Prometheus gauge from a metrics text blob.
 * Handles the optional `{labels}` block that vLLM Phase 0 emits.
 */
function num(text: string, key: string): number | undefined {
  // vLLM metric lines carry Prometheus {labels}; the optional `\{[^}]*\}` is REQUIRED.
  const esc = key.replace(/[:]/g, "\\:");
  const m = text.match(new RegExp(`^${esc}(\\{[^}]*\\})?\\s+([0-9.eE+-]+)`, "m"));
  return m ? Number(m[2]) : undefined;
}

/**
 * Parse key vLLM Prometheus metrics from a /metrics response body.
 *
 * Supports two kv-cache metric names:
 *   - vllm:kv_cache_usage_perc  (current Phase 0 shape)
 *   - vllm:gpu_cache_usage_perc (older vLLM builds)
 *
 * kv_cache_usage_perc takes precedence when both are present.
 */
export function parseVllmMetrics(text: string): VllmMetrics {
  return {
    numRequestsRunning: num(text, "vllm:num_requests_running"),
    kvCacheUsagePerc:
      num(text, "vllm:kv_cache_usage_perc") ?? num(text, "vllm:gpu_cache_usage_perc"),
  };
}

/**
 * Health-check all sparkrun-launched deployments tracked in the deployment
 * store. Uses `isWorkloadRunning` (check-job liveness) rather than `docker ps`,
 * and scrapes `/metrics` when the workload is alive.
 *
 * Returns an array of `VllmStatus` objects with the same shape produced by
 * `checkDeployments` in vllm.ts, so index.ts can treat both interchangeably.
 */
export async function checkSparkrunDeployments(): Promise<VllmStatus[]> {
  const deployments = loadDeployments();
  const results: VllmStatus[] = [];

  for (const d of deployments) {
    const target = d.clusterId ?? d.recipeFile;
    const hosts = d.clusterNodes ?? [];
    const running = isWorkloadRunning(target, hosts);

    // An intentional stop (cmd:undeploy marked stopping===true in the store)
    // that hasn't fully vanished yet must NOT be reported as a crash — exclude
    // it from the results so the health loop never mis-classifies it as failed.
    if (d.stopping && !running) continue;

    // Inspect the underlying container to detect crash-loops or unexpected exits.
    // A healthy loading container has state="running" + restartCount=0, so it is
    // NOT flagged here.  Only act when the container has restarted too many times
    // or is in a non-running/non-created terminal state.
    const c = inspectSparkrunContainer(d.clusterId);
    const failing = c != null && (c.restartCount >= CRASH_LOOP_THRESHOLD || (c.state !== "running" && c.state !== "created"));
    if (failing && !d.stopping) {
      const snap = snapshotContainerLogs(d.clusterId);
      results.push({
        deploymentId: d.deploymentId,
        recipeName: d.recipeName,
        port: d.port,
        alive: false,
        containerRunning: false,
        requestsRunning: null,
        requestsWaiting: null,
        kvCacheUsage: null,
        tps: null,
        crashLoop: c.restartCount >= CRASH_LOOP_THRESHOLD,
        restartCount: c.restartCount,
        capturedLog: snap || undefined,
        error: firstErrorLine(snap) ?? `container ${c.state} (restart #${c.restartCount})`,
      });
      continue;
    }

    const status: VllmStatus = {
      deploymentId: d.deploymentId,
      recipeName: d.recipeName,
      port: d.port,
      alive: running,
      containerRunning: running,
      requestsRunning: null,
      requestsWaiting: null,
      kvCacheUsage: null,
      tps: null,
    };

    if (running) {
      try {
        const res = await fetch(
          `http://localhost:${d.port}/metrics`,
          { signal: AbortSignal.timeout(3000) },
        );
        const text = await res.text();
        const m = parseVllmMetrics(text);
        status.requestsRunning = m.numRequestsRunning ?? null;
        status.kvCacheUsage = m.kvCacheUsagePerc ?? null;
      } catch {
        // Metrics endpoint not ready yet — workload still alive, metrics null
      }
    }

    results.push(status);
  }

  return results;
}
