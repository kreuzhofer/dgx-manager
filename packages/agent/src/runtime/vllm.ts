import { execSync, spawn, ChildProcess } from "child_process";
import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { saveDeployment, removeDeployment, loadDeployments, clearDeployments, type TrackedDeployment } from "./deployment-store.js";

const VLLM_REPO_PATH =
  process.env.VLLM_REPO_PATH || "/mnt/tank/src/github/spark-vllm-docker";
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
  error?: string;
}

interface VllmInstance {
  process: ChildProcess;
  recipeName: string;
  port: number;
}

const running = new Map<string, VllmInstance>();

/**
 * Launch a vLLM inference server using a spark-vllm-docker recipe.
 */
export function launchRecipe(
  deploymentId: string,
  recipeFile: string,
  options?: { port?: number; gpuMem?: number; maxModelLen?: number; tensorParallel?: number; pipelineParallel?: number; clusterNodes?: string[] },
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
    args.push("-n", options!.clusterNodes!.join(","), "--setup");
  } else {
    args.push("--solo", "--setup");
  }

  if (options?.port) args.push("--port", String(options.port));
  if (options?.gpuMem) args.push("--gpu-mem", String(options.gpuMem));
  if (options?.maxModelLen) args.push("--max-model-len", String(options.maxModelLen));
  if (options?.tensorParallel) args.push("--tp", String(options.tensorParallel));
  if (options?.pipelineParallel) args.push("--", "-pp", String(options.pipelineParallel));

  // For cluster mode, sync Docker image to all workers before launching
  if (isCluster) {
    const workerIps = options!.clusterNodes!.slice(1);
    // Read container name from recipe file
    let containerName = "vllm-node";
    try {
      const recipeContent = readFileSync(join(VLLM_REPO_PATH, recipeFile), "utf-8");
      const containerMatch = recipeContent.match(/^container:\s*(.+)$/m);
      if (containerMatch) containerName = containerMatch[1].trim();
    } catch { /* use default */ }
    onLog?.(`Checking ${containerName} image on ${workerIps.length} worker(s)...\n`);
    syncContainerImage(containerName, workerIps, onLog);
  }

  console.log(`Launching recipe: ${runRecipe} ${args.join(" ")}`);

  // For cluster mode, set LOCAL_IP so launch-cluster.sh finds the head node
  const extraEnv: Record<string, string> = {
    HF_HOME: process.env.HF_HOME || "/mnt/tank/models",
  };
  if (isCluster && options!.clusterNodes![0]) {
    extraEnv.LOCAL_IP = options!.clusterNodes![0];
  }

  const child = spawn(runRecipe, args, {
    cwd: VLLM_REPO_PATH,
    env: { ...process.env, ...extraEnv },
    stdio: ["ignore", "pipe", "pipe"],
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

/** Check if a deployment's process is tracked. */
export function isRunning(deploymentId: string): boolean {
  return running.has(deploymentId);
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
 * Ensure all cluster worker nodes have the same Docker image as the head.
 * Copies the image via build-and-copy.sh if mismatched.
 */
export function syncContainerImage(
  containerName: string,
  workerIps: string[],
  onLog?: (line: string) => void
): void {
  if (workerIps.length === 0) return;

  let localId: string;
  try {
    localId = execSync(`docker images ${containerName}:latest --format '{{.ID}}'`, {
      timeout: 5000, encoding: "utf-8",
    }).trim();
  } catch {
    onLog?.(`Warning: cannot read local image ID for ${containerName}\n`);
    return;
  }

  if (!localId) {
    onLog?.(`Image ${containerName} not found locally, --setup will build it\n`);
    return;
  }

  const staleWorkers: string[] = [];
  for (const ip of workerIps) {
    try {
      const remoteId = execSync(
        `ssh -o BatchMode=yes -o ConnectTimeout=5 ${SSH_USER}@${ip} "docker images ${containerName}:latest --format '{{.ID}}'"`,
        { timeout: 10_000, encoding: "utf-8" }
      ).trim();

      if (remoteId !== localId) {
        onLog?.(`Image mismatch on ${ip}: local=${localId.slice(0, 12)} remote=${remoteId.slice(0, 12) || "missing"}\n`);
        staleWorkers.push(ip);
      }
    } catch {
      onLog?.(`Cannot check image on ${ip}, will copy\n`);
      staleWorkers.push(ip);
    }
  }

  if (staleWorkers.length === 0) {
    onLog?.(`All workers have matching ${containerName} image (${localId.slice(0, 12)})\n`);
    return;
  }

  onLog?.(`Syncing ${containerName} image to ${staleWorkers.length} worker(s)...\n`);
  try {
    const copyTargets = staleWorkers.join(",");
    execSync(
      `${join(VLLM_REPO_PATH, "build-and-copy.sh")} -t ${containerName} -c ${copyTargets} --copy-parallel`,
      { cwd: VLLM_REPO_PATH, timeout: 600_000, stdio: "pipe" }
    );
    onLog?.(`Image synced to ${staleWorkers.join(", ")}\n`);
  } catch (err) {
    onLog?.(`Warning: image sync failed: ${err}\n`);
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
