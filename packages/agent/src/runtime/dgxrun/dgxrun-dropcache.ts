import { spawn, spawnSync } from "node:child_process";

/**
 * Best-effort: `sync` + drop the Linux page cache on this node.
 *
 * On GB10 there is NO separate VRAM — weights + KV + CUDA graphs + the Linux
 * page cache all share one ~124 GB unified pool. Streaming a ~400 GB checkpoint
 * off NFS fills the page cache during load, which steals the headroom the
 * first-run CUDA-graph capture / JIT needs (a transient +15-30 GB spike) and can
 * OOM the capture. Dropping the cache during load keeps that headroom free.
 *
 * Needs root. Tries a direct write first (agent-as-root), then passwordless
 * `sudo -n`. Never throws — a node that can't drop caches simply doesn't, and
 * the deploy proceeds (it just loses the headroom benefit). Returns true if a
 * drop actually ran, so the caller can warn once if it never does.
 */
export function dropCachesOnce(): boolean {
  const cmd = "sync; echo 3 > /proc/sys/vm/drop_caches";
  const opts = { stdio: "ignore" as const, timeout: 10_000 };
  // Direct write works when the agent runs as root.
  if (spawnSync("sh", ["-c", cmd], opts).status === 0) return true;
  // Fall back to passwordless sudo (the node's SSH user typically has NOPASSWD).
  return spawnSync("sudo", ["-n", "sh", "-c", cmd], opts).status === 0;
}

/**
 * Non-blocking drop. `dropCachesOnce` uses spawnSync, which parks the agent's
 * event loop for as long as `sync` takes — and under a ~400 GB NFS weight stream
 * that is SECONDS (observed 1-7 s per call at a 500 ms cadence). A blocked loop
 * stops the WS heartbeat (server marks the node offline) and starves the
 * `docker inspect` liveness probe, which used to be misread as "container
 * missing" and tore down every rank of a healthy deploy. Never call spawnSync on
 * the agent's hot path — same failure class as the old blocking `cmd:update`.
 *
 * Fire-and-forget: errors are ignored exactly like the sync variant.
 */
export function dropCachesAsyncOnce(): void {
  // One shell that tries the direct (root) write, then passwordless sudo.
  const cmd =
    "sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || " +
    "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'";
  const child = spawn("sh", ["-c", cmd], { stdio: "ignore", detached: false });
  child.on("error", () => { /* node without sh, or fork failure — just skip */ });
  child.unref?.();
}

interface LoopHandle {
  interval: ReturnType<typeof setInterval>;
  cap: ReturnType<typeof setTimeout>;
}
const loops = new Map<string, LoopHandle>();
/** Deployments whose drop is still running — prevents a `sync` pile-up. */
const inFlight = new Set<string>();

export interface DropLoopOpts {
  /** How often to drop, ms. Default 500 (the research value; caps cache growth). */
  intervalMs?: number;
  /** Hard backstop, ms. Default 20 min — covers weight load + capture even if the
   *  precise ready-stop never fires (e.g. on worker ranks with no API). */
  maxMs?: number;
  /** Injected for tests; defaults to the real (async) drop. */
  dropFn?: (done: () => void) => void;
}

/**
 * Start a drop-caches loop for a deployment: drop once immediately, then every
 * `intervalMs`, until `stopDropCacheLoop()` is called or `maxMs` elapses.
 * Idempotent per deploymentId — a second start while one is running is a no-op.
 *
 * Each tick is asynchronous and SKIPPED while the previous drop is still running,
 * so a slow `sync` can never queue up behind itself or block the event loop.
 */
export function startDropCacheLoop(deploymentId: string, opts: DropLoopOpts = {}): void {
  if (loops.has(deploymentId)) return;
  const intervalMs = opts.intervalMs ?? 500;
  const maxMs = opts.maxMs ?? 20 * 60_000;
  const dropFn = opts.dropFn ?? ((done: () => void) => { dropCachesAsyncOnce(); done(); });

  const tick = () => {
    if (inFlight.has(deploymentId)) return; // previous drop still running — skip
    inFlight.add(deploymentId);
    try {
      dropFn(() => inFlight.delete(deploymentId));
    } catch {
      inFlight.delete(deploymentId);
    }
  };

  tick();
  const interval = setInterval(tick, intervalMs);
  const cap = setTimeout(() => stopDropCacheLoop(deploymentId), maxMs);
  // Don't let these timers keep the process alive on shutdown.
  interval.unref?.();
  cap.unref?.();
  loops.set(deploymentId, { interval, cap });
}

/** Stop and clear a deployment's drop-caches loop. No-op if none is running. */
export function stopDropCacheLoop(deploymentId: string): void {
  inFlight.delete(deploymentId);
  const h = loops.get(deploymentId);
  if (!h) return;
  clearInterval(h.interval);
  clearTimeout(h.cap);
  loops.delete(deploymentId);
}

/** Test/introspection helper. */
export function isDropCacheLoopRunning(deploymentId: string): boolean {
  return loops.has(deploymentId);
}
