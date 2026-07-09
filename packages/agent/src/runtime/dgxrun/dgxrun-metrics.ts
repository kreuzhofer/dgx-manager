import { loadDeployments } from "../deployment-store.js";
import {
  inspectDgxrunContainerResult, captureCrashedDgxrunLogs, type DgxrunContainerState,
} from "./dgxrun.js";
import { stopDropCacheLoop } from "./dgxrun-dropcache.js";
import {
  parseVllmMetrics, computeTps, firstErrorLine, type TokenSample,
} from "../sparkrun-metrics.js";
import type { VllmStatus } from "../vllm.js";

const CRASH_LOOP_THRESHOLD = 2;

/**
 * How many CONSECUTIVE `absent` inspects before we declare the container gone.
 * One is not enough: docker briefly reports "no such object" while a container is
 * being recreated, and a single false "container missing" tears down every rank
 * of an mp cluster (one dead rank hangs all).
 */
const ABSENT_TICKS_TO_FAIL = 2;

/** Previous generation-tokens counter sample per deployment, for the tps rate. */
const prevTokens = new Map<string, TokenSample>();

/** Consecutive `absent` inspects per deployment. Reset on any successful inspect. */
const absentTicks = new Map<string, number>();

/**
 * Health-check dgxrun-launched deployments tracked in the local store. Mirrors
 * `checkSparkrunDeployments` but liveness comes from `docker inspect` on THIS
 * node's rank container (each agent only owns its own rank), and `/metrics` is
 * scraped ONLY on the head rank (rank 0) — headless workers never bind the vLLM
 * HTTP server, so scraping them would falsely read as "not ready".
 *
 * Returns `VllmStatus[]` (same shape as the sparkrun/vllm health loop) so
 * index.ts treats all runners interchangeably.
 */
export async function checkDgxrunDeployments(): Promise<VllmStatus[]> {
  const deployments = loadDeployments().filter((d) => d.kind === "dgxrun");
  const results: VllmStatus[] = [];

  for (const d of deployments) {
    const res = inspectDgxrunContainerResult(d.deploymentId);

    // We failed to ASK docker (timeout / busy daemon), so we know nothing about
    // the container. Reporting a failure here would tear down a healthy cluster.
    // Skip the tick and leave the last known status standing.
    if (res.kind === "unknown") continue;

    if (res.kind === "found") absentTicks.delete(d.deploymentId);

    // Docker positively reports the container gone. Require consecutive sightings
    // before believing it — a lone `absent` is routinely a recreate race.
    if (res.kind === "absent") {
      if (d.stopping) { absentTicks.delete(d.deploymentId); continue; }
      const n = (absentTicks.get(d.deploymentId) ?? 0) + 1;
      absentTicks.set(d.deploymentId, n);
      if (n < ABSENT_TICKS_TO_FAIL) continue;
      absentTicks.delete(d.deploymentId);
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
        error: "container missing",
        rank: d.rank,
      });
      continue;
    }

    const c: DgxrunContainerState = res;

    // An intentional stop (cmd:undeploy marked stopping) that already vanished
    // must not be reported as a crash.
    if (d.stopping && c.state !== "running") continue;

    const failing =
      c.restartCount >= CRASH_LOOP_THRESHOLD || (c.state !== "running" && c.state !== "created");
    if (failing && !d.stopping) {
      const snap = captureCrashedDgxrunLogs(d.deploymentId);
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
        rank: d.rank,
      });
      continue;
    }

    const running = c.state === "running";
    const isHead = (d.rank ?? 0) === 0;
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
      rank: d.rank,
    };

    // Only the head binds the API — workers are "running" purely on container liveness.
    if (running && isHead) {
      try {
        const res = await fetch(`http://localhost:${d.port}/metrics`, { signal: AbortSignal.timeout(3000) });
        status.apiReady = res.ok;
        if (res.ok) {
          // Serving now — the weight-load page-cache pressure is over; stop dropping caches.
          stopDropCacheLoop(d.deploymentId);
          const text = await res.text();
          const m = parseVllmMetrics(text);
          status.requestsRunning = m.numRequestsRunning ?? null;
          status.kvCacheUsage = m.kvCacheUsagePerc ?? null;
          if (m.generationTokensTotal != null) {
            const curr: TokenSample = { total: m.generationTokensTotal, time: Date.now() };
            status.tps = computeTps(prevTokens.get(d.deploymentId), curr);
            prevTokens.set(d.deploymentId, curr);
          }
        }
      } catch { /* metrics not ready yet — still loading */ }
    } else if (running && !isHead) {
      // Workers have no HTTP server; treat a live container as "ready" so the
      // head's status is the sole gate on the deployment's overall status.
      status.apiReady = true;
    }

    results.push(status);
  }

  const liveIds = new Set(deployments.map((d) => d.deploymentId));
  for (const id of prevTokens.keys()) if (!liveIds.has(id)) prevTokens.delete(id);

  return results;
}
