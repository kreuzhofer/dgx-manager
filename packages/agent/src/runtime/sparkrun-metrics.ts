import { loadDeployments } from "./deployment-store.js";
import { isWorkloadRunning, inspectSparkrunContainer, captureCrashedContainerLogs } from "./sparkrun.js";
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
  generationTokensTotal?: number;
}

/** A reading of the cumulative generation-tokens counter at a point in time. */
export interface TokenSample { total: number; time: number; }

/**
 * Tokens/sec from two `vllm:generation_tokens_total` counter samples — the
 * throughput figure the dashboard shows. The metric is a monotonic counter, so
 * the rate is the delta over elapsed wall-clock between two health-loop scrapes.
 * Returns null on the first sample (no baseline), a non-positive interval, or a
 * negative delta (the counter reset to 0 when the container restarted).
 */
export function computeTps(prev: TokenSample | undefined, curr: TokenSample): number | null {
  if (!prev) return null;
  const elapsedSec = (curr.time - prev.time) / 1000;
  if (elapsedSec <= 0) return null;
  const delta = curr.total - prev.total;
  if (delta < 0) return null; // counter reset (container restart) — skip this interval
  return Math.round((delta / elapsedSec) * 10) / 10;
}

/** Previous generation-tokens counter sample per deployment, for the tps rate. */
const prevTokens = new Map<string, TokenSample>();

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
    generationTokensTotal: num(text, "vllm:generation_tokens_total"),
  };
}

/**
 * Pure helper: determine the status string for a sparkrun deployment based on
 * container liveness and vLLM API readiness.
 *
 * The vLLM HTTP server only binds after all weight shards are loaded and the
 * engine is warmed up — a container that is running but whose /metrics endpoint
 * hasn't responded 2xx yet is still in the loading phase, not serving traffic.
 *
 * containerRunning && apiReady  → "running"
 * containerRunning && !apiReady → "starting"  (loading shards / warming up)
 */
export function sparkrunRunningStatus(
  s: Pick<VllmStatus, "containerRunning" | "apiReady">
): "running" | "starting" {
  return s.containerRunning && s.apiReady ? "running" : "starting";
}

/**
 * Parse vLLM's weight-shard load progress line:
 *   "Loading safetensors checkpoint shards:  42% Completed | 35/83 [00:10<00:12, ...]"
 *
 * Handles variable spacing. Returns null when the line does not match.
 */
const LOADING_SHARDS_RE = /Loading safetensors checkpoint shards:\s+(\d+(?:\.\d+)?)%\s+Completed\s*\|\s*(\d+)\/(\d+)/;
export function parseLoadingShards(line: string): {
  percent: number;
  current: number;
  total: number;
} | null {
  const m = line.match(LOADING_SHARDS_RE);
  if (!m) return null;
  return {
    percent: parseFloat(m[1]),
    current: parseInt(m[2], 10),
    total: parseInt(m[3], 10),
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
    // Still launching: `sparkrun run` hasn't reported a Cluster id yet, so no
    // container exists (the model may still be downloading). The cmd:deploy
    // onLog/onExit path owns status during launch — do NOT treat this as a dead
    // container, or we'd kill the in-progress deploy.
    if (!d.clusterId) continue;

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
      const snap = captureCrashedContainerLogs(d.clusterId);
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
        // apiReady = true only when /metrics responds 2xx — vLLM binds the HTTP
        // server only after all weight shards are loaded and the engine is ready.
        status.apiReady = res.ok;
        if (res.ok) {
          const text = await res.text();
          const m = parseVllmMetrics(text);
          status.requestsRunning = m.numRequestsRunning ?? null;
          status.kvCacheUsage = m.kvCacheUsagePerc ?? null;
          // tps = rate of the generation-tokens counter between this scrape and
          // the previous one for this deployment.
          if (m.generationTokensTotal != null) {
            const curr: TokenSample = { total: m.generationTokensTotal, time: Date.now() };
            status.tps = computeTps(prevTokens.get(d.deploymentId), curr);
            prevTokens.set(d.deploymentId, curr);
          }
        }
      } catch {
        // Metrics endpoint not ready yet — workload still alive, metrics null, apiReady falsy
      }
    }

    results.push(status);
  }

  // Drop counter baselines for deployments that are no longer tracked, so a
  // future deployment reusing an id doesn't inherit a stale (huge-delta) sample.
  const liveIds = new Set(deployments.map((d) => d.deploymentId));
  for (const id of prevTokens.keys()) if (!liveIds.has(id)) prevTokens.delete(id);

  return results;
}
