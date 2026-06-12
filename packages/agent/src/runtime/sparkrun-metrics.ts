import { loadDeployments } from "./deployment-store.js";
import { isWorkloadRunning } from "./sparkrun.js";
import type { VllmStatus } from "./vllm.js";

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
