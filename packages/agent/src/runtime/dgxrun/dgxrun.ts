import { spawn, spawnSync } from "node:child_process";
import { dropCachesOnce, startDropCacheLoop, stopDropCacheLoop } from "./dgxrun-dropcache.js";
import type { ChildProcess } from "node:child_process";
import { buildDgxrunDockerArgs, type DgxrunRecipe } from "./dgxrun-args.js";
import { resolveHfHome } from "../sparkrun.js";
import { saveDeployment, removeDeployment } from "../deployment-store.js";

/**
 * dgxrun docker lifecycle — mirrors sparkrun.ts's surface (launch / stop /
 * isRunning / inspect / snapshot / captureCrashed) but keyed on a local,
 * per-rank container named `dgxrun_<deploymentId>`. Each agent owns exactly
 * ONE rank's container; the manager fans a `cmd:deploy` per node. No SSH, no
 * cross-node orchestration here.
 */

/** Container name for a deployment's local rank. */
export function dgxrunContainerName(deploymentId: string): string {
  return `dgxrun_${deploymentId}`;
}

/** Live `docker logs -f` followers, keyed by deploymentId, so undeploy can kill them. */
const logFollowers = new Map<string, ChildProcess>();

export interface DgxrunLaunchArgs {
  recipe: DgxrunRecipe;
  rank: number;
  nnodes: number;
  masterAddr: string;
  masterPort: number;
  /** Head-node serve port (for status/health). Falls back to defaults.port. */
  port?: number;
  /** Placeholder overrides forwarded from the manager's deploy config. */
  params?: Record<string, string | number | undefined>;
  /** Host HF cache dir → /cache/huggingface. Defaults to resolveHfHome(). */
  weightsDir?: string;
}

/** True when the container image is present locally (fast-fail on a miss). */
export function dgxrunImageExists(image: string): boolean {
  const r = spawnSync("docker", ["image", "inspect", image], {
    stdio: "ignore", timeout: 15_000,
  });
  return r.status === 0;
}

/**
 * Launch THIS node's rank via `docker run -d`.
 *
 * v1 does NOT distribute images across nodes (documented follow-up): the image
 * must already exist locally. We verify that up front and fail fast with a
 * clear error rather than letting `docker run` emit an opaque pull error.
 *
 * `docker run -d` returns immediately after the container starts, so a 0 exit
 * means "started" (not "serving"); the health loop promotes it to running once
 * the head's /metrics binds. A non-zero exit is a real launch failure.
 */
export function launchDgxrun(
  deploymentId: string,
  args: DgxrunLaunchArgs,
  onLog: (line: string) => void,
  onExit: (code: number | null) => void,
): void {
  const name = dgxrunContainerName(deploymentId);
  const image = args.recipe.container;
  const port = args.port ?? Number(args.recipe.defaults?.port) ?? 8000;
  const weightsDir = args.weightsDir ?? resolveHfHome();

  // Persist first so a reconnect mid-launch can reconcile this rank.
  const persist = () => saveDeployment({
    deploymentId,
    recipeFile: image,
    recipeName: args.recipe.model ?? deploymentId,
    port,
    startedAt: new Date().toISOString(),
    tp: args.nnodes,
    kind: "dgxrun",
    rank: args.rank,
    masterAddr: args.masterAddr,
    masterPort: args.masterPort,
  });

  // Fail fast if the custom image isn't on this node — v1 assumes it's present.
  if (!dgxrunImageExists(image)) {
    onLog(`[dgxrun] image "${image}" not found locally. v1 does not distribute images; ` +
      `build/load it on this node first (docker load / registry pull).\n`);
    onExit(1);
    return;
  }

  let dockerArgs: string[];
  try {
    dockerArgs = buildDgxrunDockerArgs(args.recipe, {
      containerName: name,
      weightsDir,
      rank: args.rank,
      nnodes: args.nnodes,
      masterAddr: args.masterAddr,
      masterPort: args.masterPort,
      params: args.params,
    });
  } catch (err) {
    onLog(`[dgxrun] failed to build launch args: ${err}\n`);
    onExit(1);
    return;
  }

  // Clear any stale container from a prior deploy (idempotent).
  spawnSync("docker", ["rm", "-f", name], { stdio: "ignore", timeout: 30_000 });

  persist();
  onLog(`[dgxrun] rank ${args.rank}/${args.nnodes} launching (master ${args.masterAddr}:${args.masterPort})\n`);
  onLog(`[dgxrun] docker ${dockerArgs.join(" ")}\n`);

  // Free the page cache before the container streams its weights off NFS (GB10's
  // unified pool shares page cache with CUDA-graph capture headroom).
  dropCachesOnce();
  const child = spawn("docker", dockerArgs);
  let stderr = "";
  child.stdout?.on("data", (b: Buffer) => onLog(b.toString()));
  child.stderr?.on("data", (b: Buffer) => { const s = b.toString(); stderr += s; onLog(s); });
  child.on("exit", (code) => {
    if (code === 0) {
      startLogFollower(deploymentId, name, onLog);
      // Keep the page cache clear through the weight load; stopped when the head
      // reports the API ready (dgxrun-metrics), on teardown, or after the backstop.
      startDropCacheLoop(deploymentId);
      onExit(0);
    } else {
      onLog(`[dgxrun] docker run exited ${code}${stderr ? `: ${stderr.trim().slice(0, 500)}` : ""}\n`);
      onExit(code);
    }
  });
}

/** Follow the container's logs so loading progress / vLLM output reaches the dashboard. */
function startLogFollower(deploymentId: string, name: string, onLog: (line: string) => void): void {
  const prev = logFollowers.get(deploymentId);
  if (prev) { try { prev.kill(); } catch { /* gone */ } }
  const follower = spawn("docker", ["logs", "-f", "--tail", "200", name]);
  follower.stdout?.on("data", (b: Buffer) => onLog(b.toString()));
  follower.stderr?.on("data", (b: Buffer) => onLog(b.toString()));
  follower.on("exit", () => { logFollowers.delete(deploymentId); });
  logFollowers.set(deploymentId, follower);
}

/** Tear down THIS node's rank container + kill its log follower. */
export function stopDgxrun(deploymentId: string): void {
  stopDropCacheLoop(deploymentId);
  const f = logFollowers.get(deploymentId);
  if (f) { try { f.kill(); } catch { /* gone */ } logFollowers.delete(deploymentId); }
  const name = dgxrunContainerName(deploymentId);
  try { spawnSync("docker", ["rm", "-f", name], { stdio: "ignore", timeout: 60_000 }); }
  finally { removeDeployment(deploymentId); }
}

export interface DgxrunContainerState { name: string; state: string; restartCount: number; }

/**
 * Outcome of a `docker inspect`. The three cases are NOT interchangeable:
 * `absent` means docker positively said the container does not exist, while
 * `unknown` means we failed to ask (timeout, busy/dead daemon). Collapsing
 * `unknown` into `absent` is what tore down every rank of a healthy GLM-5.2
 * cluster: under a heavy NFS weight stream the daemon was slow, inspect timed
 * out, and the agent reported "container missing" for a container that was
 * mid-`torch.compile`.
 */
export type DgxrunInspect =
  | { kind: "found"; name: string; state: string; restartCount: number }
  | { kind: "absent" }
  | { kind: "unknown"; reason: string };

/**
 * Pure: classify a `docker inspect` invocation. Docker exits non-zero with
 * `Error: No such object: <name>` (or "No such container") when the container
 * genuinely does not exist. Every OTHER non-zero exit — a spawnSync timeout
 * (status null + error), a daemon error — is `unknown`, never `absent`.
 */
export function classifyDockerInspect(
  status: number | null,
  stdout: string,
  stderr: string,
  name: string,
  error?: Error,
): DgxrunInspect {
  if (status === 0) {
    const out = (stdout || "").trim();
    if (!out) return { kind: "unknown", reason: "docker inspect returned empty output" };
    const [state, rc] = out.split(/\s+/);
    return { name, kind: "found", state, restartCount: Number(rc) || 0 };
  }
  const blob = `${stderr || ""} ${error?.message ?? ""}`;
  if (/no such (object|container)/i.test(blob)) return { kind: "absent" };
  if (status === null) {
    return { kind: "unknown", reason: error?.message || "docker inspect timed out" };
  }
  return { kind: "unknown", reason: (stderr || "").trim() || `docker inspect exited ${status}` };
}

/** Read-only docker state for a rank container, distinguishing absent from unknown. */
export function inspectDgxrunContainerResult(deploymentId: string): DgxrunInspect {
  const name = dgxrunContainerName(deploymentId);
  const r = spawnSync("docker", ["inspect", name, "--format", "{{.State.Status}} {{.RestartCount}}"],
    { encoding: "utf8", timeout: 10_000 });
  return classifyDockerInspect(r.status, r.stdout ?? "", r.stderr ?? "", name, r.error as Error | undefined);
}

/** Read-only docker state + restart count for a deployment's rank container. */
export function inspectDgxrunContainer(deploymentId: string): DgxrunContainerState | null {
  const res = inspectDgxrunContainerResult(deploymentId);
  return res.kind === "found"
    ? { name: res.name, state: res.state, restartCount: res.restartCount }
    : null;
}

// NOTE: there is deliberately no `isDgxrunRunning(): boolean` helper. Reducing an
// inspect to a boolean loses the absent/unknown distinction, and callers then read
// "docker didn't answer" as "the container is gone" — which tore down a healthy
// four-rank cluster. Use inspectDgxrunContainerResult() and handle `unknown`.

/** Read-only snapshot of the FULL container log (all restarts). */
export function snapshotDgxrunLogs(deploymentId: string): string {
  const name = dgxrunContainerName(deploymentId);
  const r = spawnSync("docker", ["logs", name],
    { encoding: "utf8", timeout: 20_000, maxBuffer: 32 * 1024 * 1024 });
  if (r.status !== 0 && !r.stdout && !r.stderr) return "";
  const headCap = (s: string | null | undefined, n: number) => {
    const t = (s || "").trim();
    return t.length > n ? t.slice(0, n) + "\n…[truncated]" : t;
  };
  const stderr = headCap(r.stderr, 120_000);
  const stdout = headCap(r.stdout, 40_000);
  return [stderr, stdout].filter(Boolean).join("\n").trim();
}

/** Stop the restart loop, then capture the full accumulated container log. */
export function captureCrashedDgxrunLogs(deploymentId: string): string {
  const name = dgxrunContainerName(deploymentId);
  spawnSync("docker", ["stop", "-t", "3", name], { timeout: 15_000 });
  return snapshotDgxrunLogs(deploymentId);
}
