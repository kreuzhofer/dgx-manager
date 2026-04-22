import { execSync, spawn, ChildProcess } from "child_process";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join } from "path";
import { SHARED_STORAGE, WORKSPACE } from "../env.js";
import { saveDeployment, removeDeployment, loadDeployments, clearDeployments, type TrackedDeployment } from "./deployment-store.js";

const VLLM_REPO_PATH =
  process.env.VLLM_REPO_PATH || `${SHARED_STORAGE}/src/github/spark-vllm-docker`;
const SSH_USER = process.env.SSH_USER || process.env.USER || "daniel";

export interface VllmStatus {
  deploymentId: string;
  recipeName: string;
  port: number;
  alive: boolean;
  containerRunning: boolean;
  requestsRunning: number | null;
  requestsWaiting: number | null;
  kvCacheUsage: number | null;
  tps: number | null;
  error?: string;
}

// Track previous token counter for TPS calculation
let prevTokenCount: { total: number; time: number } | null = null;

interface VllmInstance {
  process: ChildProcess;
  recipeName: string;
  port: number;
  stopping?: boolean;
}

const running = new Map<string, VllmInstance>();

/**
 * Generate a vLLM recipe YAML for a locally-stored model (e.g., merged fine-tune output).
 * Places it in the spark-vllm-docker recipes/ dir so run-recipe.sh can find it.
 * Returns the recipe file path relative to the repo root (e.g., "recipes/finetune-abc123.yaml").
 */
export function generateLocalModelRecipe(params: {
  jobId: string;
  modelPath: string;      // host path (e.g., ${SHARED_STORAGE}/outputs/job123/merged)
  container?: string;     // vLLM container name (from training recipe deploy config)
  port?: number;
  gpuMemoryUtilization?: number;
  maxModelLen?: number;
}): string {
  const recipeName = `finetune-${params.jobId.slice(0, 12)}`;
  const recipeFile = `recipes/${recipeName}.yaml`;
  const fullPath = join(VLLM_REPO_PATH, recipeFile);

  // launch-cluster.sh mounts ${SHARED_STORAGE}:/workspace, so translate host path to container path
  const containerModelPath = params.modelPath.replace(`${SHARED_STORAGE}/`, `${WORKSPACE}/`);

  const port = params.port ?? 8000;
  const gpuMem = params.gpuMemoryUtilization ?? 0.85;
  const maxLen = params.maxModelLen ?? 4096;

  const container = params.container || "vllm-node";

  const yaml = `# Auto-generated recipe for fine-tuned model
recipe_version: "1"
name: ${recipeName}
description: Fine-tuned model from job ${params.jobId}
model: ${containerModelPath}
container: ${container}
solo_only: true

defaults:
  port: ${port}
  host: 0.0.0.0
  gpu_memory_utilization: ${gpuMem}
  max_model_len: ${maxLen}

command: |
  vllm serve ${containerModelPath} \\
    --host {host} \\
    --port {port} \\
    --max-model-len {max_model_len} \\
    --gpu-memory-utilization {gpu_memory_utilization} \\
    --dtype auto
`;

  mkdirSync(join(VLLM_REPO_PATH, "recipes"), { recursive: true });
  writeFileSync(fullPath, yaml, "utf-8");
  console.log(`Generated vLLM recipe: ${fullPath}`);
  return recipeFile;
}

/**
 * Launch a vLLM inference server using a spark-vllm-docker recipe.
 */
export function launchRecipe(
  deploymentId: string,
  recipeFile: string,
  options?: { port?: number; gpuMem?: number; maxModelLen?: number; tensorParallel?: number; pipelineParallel?: number; clusterNodes?: string[]; clusterNodeFastIps?: (string | null)[]; skipSetup?: boolean },
  onLog?: (line: string) => void,
  onExit?: (code: number | null) => void
): number {
  const port = options?.port ?? 8000;
  const runRecipe = join(VLLM_REPO_PATH, "run-recipe.sh");

  if (!existsSync(runRecipe)) {
    throw new Error(`run-recipe.sh not found at ${runRecipe}`);
  }

  stopRecipe(deploymentId);

  const recipeName = recipeFile
    .replace(/^recipes\//, "")
    .replace(/\.yaml$/, "");

  const isCluster = options?.clusterNodes && options.clusterNodes.length > 1;
  const args = [recipeName];

  if (isCluster) {
    args.push("-n", options!.clusterNodes!.join(","));
    if (!options?.skipSetup) args.push("--setup");
  } else {
    args.push("--solo");
    if (!options?.skipSetup) args.push("--setup");
  }

  if (options?.port) args.push("--port", String(options.port));
  if (options?.gpuMem) args.push("--gpu-mem", String(options.gpuMem));
  if (options?.maxModelLen) args.push("--max-model-len", String(options.maxModelLen));
  if (options?.tensorParallel) args.push("--tp", String(options.tensorParallel));
  if (options?.pipelineParallel) args.push("--", "-pp", String(options.pipelineParallel));

  // For cluster mode, ensure workers have an image identical to the head's
  // (see syncContainerImage). Prefer each worker's fast-fabric IP when known
  // — image transfer is multi-GB and the management network is typically
  // 1 GbE while the fast network is 10/100 GbE. Falls back to mgmt IP per-node.
  if (isCluster) {
    const mgmtWorkers = options!.clusterNodes!.slice(1);
    const fastWorkers = (options?.clusterNodeFastIps ?? []).slice(1);
    const workerIps = mgmtWorkers.map((mgmt, i) => fastWorkers[i] || mgmt);
    let containerName = "vllm-node";
    try {
      const recipeContent = readFileSync(join(VLLM_REPO_PATH, recipeFile), "utf-8");
      const containerMatch = recipeContent.match(/^container:\s*(.+)$/m);
      if (containerMatch) containerName = containerMatch[1].trim();
    } catch { /* use default */ }
    syncContainerImage(containerName, workerIps, onLog);
  }

  console.log(`Launching recipe: ${runRecipe} ${args.join(" ")}`);

  // For cluster mode, set LOCAL_IP so launch-cluster.sh finds the head node
  const extraEnv: Record<string, string> = {
    HF_HOME: process.env.HF_HOME || `${SHARED_STORAGE}/models`,
  };
  if (isCluster && options!.clusterNodes![0]) {
    extraEnv.LOCAL_IP = options!.clusterNodes![0];
  }

  // Wrap in shell that ignores SIGPIPE so broken pipes from agent restart
  // don't kill run-recipe.sh
  const child = spawn("bash", ["-c", `trap '' PIPE; exec "${runRecipe}" ${args.map(a => `"${a}"`).join(" ")}`], {
    cwd: VLLM_REPO_PATH,
    env: { ...process.env, ...extraEnv },
    stdio: ["ignore", "pipe", "pipe"],
    detached: true, // Own process group — survives agent restart
  });

  child.stdout?.on("data", (data: Buffer) => {
    const line = data.toString();
    onLog?.(line);
    console.log(`[vllm:${recipeName}] ${line.trimEnd()}`);
  });

  child.stderr?.on("data", (data: Buffer) => {
    const line = data.toString();
    onLog?.(line);
    console.error(`[vllm:${recipeName}] ${line.trimEnd()}`);
  });

  child.on("exit", (code) => {
    console.log(`[vllm:${recipeName}] run-recipe exited with code ${code}`);
    // Don't remove from running map — the docker container may still be alive.
    // The onExit handler will check container status.
    onExit?.(code);

    // If container launched successfully, tail docker logs for loading progress
    if (isVllmContainerRunning()) {
      const tail = spawn("docker", ["logs", "-f", "vllm_node"], {
        stdio: ["ignore", "pipe", "pipe"],
      });
      const forwardLog = (data: Buffer) => {
        const line = data.toString();
        onLog?.(line);
      };
      tail.stdout?.on("data", forwardLog);
      tail.stderr?.on("data", forwardLog);
      tail.on("exit", () => {
        console.log(`[vllm:${recipeName}] docker log tail ended`);
      });
      // Store the tail process so we can kill it on undeploy
      const inst = running.get(deploymentId);
      if (inst) {
        (inst as unknown as Record<string, unknown>).tailProcess = tail;
      }
    }
  });

  running.set(deploymentId, { process: child, recipeName, port });

  // Persist to disk for recovery after agent restart
  saveDeployment({
    deploymentId,
    recipeFile,
    recipeName,
    port,
    startedAt: new Date().toISOString(),
    clusterNodes: isCluster ? options!.clusterNodes : undefined,
  });

  return port;
}

/** Stop a running vLLM instance. Supports cluster mode via clusterNodes. */
export function stopRecipe(deploymentId: string, clusterNodes?: string[]): boolean {
  const instance = running.get(deploymentId);
  if (!instance) return false;

  console.log(`Stopping vLLM recipe: ${instance.recipeName}`);
  instance.stopping = true;

  // Kill the run-recipe.sh process group FIRST to prevent it from relaunching
  try {
    if (instance.process.pid) process.kill(-instance.process.pid, "SIGTERM");
  } catch { /* already dead */ }

  // Kill log tail if running
  const tailProc = (instance as unknown as Record<string, unknown>).tailProcess as ChildProcess | undefined;
  if (tailProc) tailProc.kill();

  const isCluster = clusterNodes && clusterNodes.length > 1;
  const env = { ...process.env };
  if (isCluster && clusterNodes[0]) env.LOCAL_IP = clusterNodes[0];

  try {
    if (isCluster) {
      execSync(
        `${join(VLLM_REPO_PATH, "launch-cluster.sh")} -n ${clusterNodes.join(",")} stop`,
        { cwd: VLLM_REPO_PATH, timeout: 60_000, stdio: "inherit", env }
      );
    } else {
      execSync(
        `${join(VLLM_REPO_PATH, "launch-cluster.sh")} --solo stop`,
        { cwd: VLLM_REPO_PATH, timeout: 30_000, stdio: "inherit" }
      );
    }
  } catch {
    instance.process.kill("SIGTERM");
  }

  running.delete(deploymentId);
  removeDeployment(deploymentId);
  return true;
}

/**
 * Reattach to docker logs for a running container.
 * Used after agent restart to resume log streaming to the manager.
 */
export function reattachLogs(
  deploymentId: string,
  onLog?: (line: string) => void
): void {
  if (!isVllmContainerRunning()) return;
  console.log(`[vllm] Reattaching to docker logs for deployment ${deploymentId}`);
  const tail = spawn("docker", ["logs", "-f", "--since", "5s", "vllm_node"], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  const forwardLog = (data: Buffer) => onLog?.(data.toString());
  tail.stdout?.on("data", forwardLog);
  tail.stderr?.on("data", forwardLog);
  tail.on("exit", () => console.log(`[vllm] Docker log tail ended for ${deploymentId}`));
}

/** Check if a deployment's process is tracked. */
export function isRunning(deploymentId: string): boolean {
  return running.has(deploymentId);
}

/** Check if a deployment is being stopped (suppress onExit status reports). */
export function isStopping(deploymentId: string): boolean {
  return running.get(deploymentId)?.stopping === true;
}

/** Get all tracked deployment IDs. */
export function getDeploymentIds(): string[] {
  return Array.from(running.keys());
}

/** Check if any vLLM docker container is running (regardless of tracking). */
export function isVllmContainerRunning(): boolean {
  try {
    const out = execSync("docker ps --filter name=vllm_node --format '{{.Status}}'", {
      timeout: 5000, encoding: "utf-8",
    }).trim();
    return out.includes("Up");
  } catch {
    return false;
  }
}

/** Force stop any vLLM containers and clear tracking. */
export function forceStopVllm(clusterNodes?: string[]): void {
  const isCluster = clusterNodes && clusterNodes.length > 1;
  const env = { ...process.env };
  if (isCluster && clusterNodes![0]) env.LOCAL_IP = clusterNodes![0];

  try {
    if (isCluster) {
      execSync(
        `${join(VLLM_REPO_PATH, "launch-cluster.sh")} -n ${clusterNodes!.join(",")} stop`,
        { cwd: VLLM_REPO_PATH, timeout: 60_000, stdio: "pipe", env }
      );
    } else {
      execSync(
        `${join(VLLM_REPO_PATH, "launch-cluster.sh")} --solo stop`,
        { cwd: VLLM_REPO_PATH, timeout: 30_000, stdio: "pipe" }
      );
    }
  } catch {
    try {
      execSync("docker stop vllm_node", { timeout: 15_000, stdio: "pipe" });
    } catch { /* nothing running */ }
  }
  clearDeployments();
}

/** Get deployments from persistent store (survives agent restarts). */
export function getTrackedDeployments(): TrackedDeployment[] {
  return loadDeployments();
}

/**
 * True if a launch subprocess for this deployment is still alive in the
 * current agent process. Used during WS-only reconnects to distinguish a
 * still-in-progress launch (download/build/loading) from a truly orphaned
 * tracked deployment after a full agent restart.
 */
export function isLaunchInProgress(deploymentId: string): boolean {
  const inst = running.get(deploymentId);
  if (!inst) return false;
  // killed/exited child has process.exitCode set or process.killed true
  const proc = inst.process as ChildProcess & { exitCode?: number | null };
  if (inst.stopping) return false;
  if (proc.killed) return false;
  if (proc.exitCode != null) return false;
  return true;
}

/**
 * Ensure every cluster worker node runs the exact same Docker image as the
 * head. We compare image IDs (content-addressed SHA256), not creation
 * timestamps — a worker can have a newer-but-incompatible image (e.g. a
 * build with a different Ray version), and date-based comparison would skip
 * the sync and the workers would fail to join the head's Ray cluster due to
 * version mismatch. Forcing exact-ID match head→workers is the only safe rule.
 */
export function syncContainerImage(
  containerName: string,
  workerIps: string[],
  onLog?: (line: string) => void
): void {
  if (workerIps.length === 0) return;

  let localId: string;
  try {
    localId = execSync(
      `docker images ${containerName}:latest --format '{{.ID}}'`,
      { timeout: 5000, encoding: "utf-8" }
    ).trim();
  } catch {
    onLog?.(`Image ${containerName} not found locally, --setup will build it\n`);
    return;
  }

  if (!localId) {
    onLog?.(`Image ${containerName} not found locally\n`);
    return;
  }

  const mismatchedWorkers: string[] = [];

  for (const ip of workerIps) {
    try {
      const remoteId = execSync(
        `ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 ${SSH_USER}@${ip} "docker images ${containerName}:latest --format '{{.ID}}'"`,
        { timeout: 10_000, encoding: "utf-8" }
      ).trim();

      if (!remoteId) {
        onLog?.(`Image missing on ${ip}, will copy\n`);
        mismatchedWorkers.push(ip);
      } else if (remoteId !== localId) {
        onLog?.(`Image on ${ip} differs from head (${remoteId} != ${localId}), will copy\n`);
        mismatchedWorkers.push(ip);
      }
    } catch {
      onLog?.(`Cannot check image on ${ip}, will copy\n`);
      mismatchedWorkers.push(ip);
    }
  }

  if (mismatchedWorkers.length === 0) {
    onLog?.(`All workers match head's ${containerName} image (${localId})\n`);
    return;
  }

  onLog?.(`Syncing ${containerName} (${localId}) to ${mismatchedWorkers.length} worker(s)...\n`);
  try {
    const copyTargets = mismatchedWorkers.join(",");
    // --no-build is critical: without it, build-and-copy.sh REBUILDS the
    // image from source (rebuilding vLLM + flashinfer wheels) before
    // copying. That re-build can produce a different binary than what's
    // currently running on the head — we hit
    // "ImportError: vllm/_C.abi3.so: undefined symbol: ...
    //  getCurrentCUDABlasHandleEv" because the rebuilt vLLM wheel ABI
    // didn't match the libtorch in the resulting image. We want to
    // propagate the EXACT image that's already validated on the head.
    execSync(
      `${join(VLLM_REPO_PATH, "build-and-copy.sh")} -t ${containerName} -c ${copyTargets} --copy-parallel --no-build`,
      // 30 min: multi-GB Docker image save+scp+load over LAN can take
      // 15-20 min. Previous 10 min timed out silently and the launch
      // proceeded with mismatched images, hanging on Ray placement group.
      { cwd: VLLM_REPO_PATH, timeout: 1_800_000, stdio: "pipe" }
    );
    onLog?.(`Image synced to ${mismatchedWorkers.join(", ")}\n`);
  } catch (err) {
    // Hard fail: a workers-image-mismatch in cluster mode = guaranteed
    // Ray-version-mismatch = silent placement-group hang. Better to
    // surface the failure than to launch and wait forever.
    onLog?.(`ERROR: image sync failed, aborting cluster launch: ${err}\n`);
    throw new Error(`Cluster image sync to workers failed: ${err}`);
  }
}

/**
 * Health check all running deployments.
 * Checks docker container status and scrapes vLLM metrics endpoint.
 */
export async function checkDeployments(): Promise<VllmStatus[]> {
  const results: VllmStatus[] = [];

  // Merge in-memory tracking with disk-persisted tracking
  const tracked = loadDeployments();
  const allDeployments = new Map<string, { recipeName: string; port: number; process?: ChildProcess }>();

  for (const t of tracked) {
    allDeployments.set(t.deploymentId, { recipeName: t.recipeName, port: t.port });
  }
  for (const [id, inst] of running) {
    allDeployments.set(id, { recipeName: inst.recipeName, port: inst.port, process: inst.process });
  }

  for (const [deploymentId, info] of allDeployments) {
    const proc = info.process;
    const status: VllmStatus = {
      deploymentId,
      recipeName: info.recipeName,
      port: info.port,
      alive: proc ? !proc.killed && proc.exitCode === null : false,
      containerRunning: false,
      requestsRunning: null,
      requestsWaiting: null,
      kvCacheUsage: null,
      tps: null,
    };

    // Check docker container
    try {
      const out = execSync("docker ps --filter name=vllm_node --format '{{.Status}}'", {
        timeout: 5000, encoding: "utf-8",
      }).trim();
      status.containerRunning = out.includes("Up");
      if (!status.containerRunning && out) {
        status.error = `Container status: ${out}`;
      }
    } catch {
      status.containerRunning = false;
    }

    // Scrape vLLM metrics if container is running
    if (status.containerRunning) {
      try {
        const metrics = execSync(
          `curl -sf --max-time 3 http://localhost:${info.port}/metrics`,
          { timeout: 5000, encoding: "utf-8" }
        );
        status.requestsRunning = parsePrometheusGauge(metrics, "vllm:num_requests_running");
        status.requestsWaiting = parsePrometheusGauge(metrics, "vllm:num_requests_waiting");
        status.kvCacheUsage = parsePrometheusGauge(metrics, "vllm:kv_cache_usage_perc");

        // Compute TPS from generation_tokens counter delta
        const genTokens = parsePrometheusCounter(metrics, "vllm:generation_tokens_total");
        if (genTokens !== null) {
          const now = Date.now();
          if (prevTokenCount) {
            const elapsed = (now - prevTokenCount.time) / 1000;
            if (elapsed > 0) {
              status.tps = Math.round((genTokens - prevTokenCount.total) / elapsed * 10) / 10;
            }
          }
          prevTokenCount = { total: genTokens, time: now };
        }
      } catch {
        // Metrics endpoint not ready yet
      }
    }

    // Check for recent docker errors
    if (status.containerRunning) {
      try {
        const logs = execSync(
          "docker logs --tail 5 vllm_node 2>&1",
          { timeout: 5000, encoding: "utf-8" }
        );
        const errorLine = logs.split("\n").find(
          (l) => l.includes("Error") || l.includes("FATAL") || l.includes("OOM")
        );
        if (errorLine) {
          status.error = errorLine.trim().slice(0, 200);
        }
      } catch { /* ignore */ }
    }

    results.push(status);
  }

  return results;
}

function parsePrometheusGauge(text: string, name: string): number | null {
  const regex = new RegExp(`^${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}[{\\s].*?\\s+([\\d.eE+-]+)$`, "m");
  const match = text.match(regex);
  return match ? parseFloat(match[1]) : null;
}

function parsePrometheusCounter(text: string, name: string): number | null {
  // Counters may have labels like {engine="0",model_name="..."}
  const regex = new RegExp(`^${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}[{\\s].*?\\s+([\\d.eE+-]+)$`, "m");
  const match = text.match(regex);
  return match ? parseFloat(match[1]) : null;
}
