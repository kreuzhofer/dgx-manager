import { execSync, spawn, ChildProcess } from "child_process";
import { existsSync } from "fs";
import { join } from "path";

const VLLM_REPO_PATH =
  process.env.VLLM_REPO_PATH || "/mnt/tank/src/github/spark-vllm-docker";

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
  options?: { port?: number; gpuMem?: number; maxModelLen?: number },
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

  const args = [recipeName, "--solo", "--setup"];
  if (options?.port) args.push("--port", String(options.port));
  if (options?.gpuMem) args.push("--gpu-mem", String(options.gpuMem));
  if (options?.maxModelLen) args.push("--max-model-len", String(options.maxModelLen));

  console.log(`Launching recipe: ${runRecipe} ${args.join(" ")}`);

  const child = spawn(runRecipe, args, {
    cwd: VLLM_REPO_PATH,
    env: {
      ...process.env,
      HF_HOME: process.env.HF_HOME || "/mnt/tank/models",
    },
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
    console.log(`[vllm:${recipeName}] exited with code ${code}`);
    running.delete(deploymentId);
    onExit?.(code);
  });

  running.set(deploymentId, { process: child, recipeName, port });
  return port;
}

/** Stop a running vLLM instance. */
export function stopRecipe(deploymentId: string): boolean {
  const instance = running.get(deploymentId);
  if (!instance) return false;

  console.log(`Stopping vLLM recipe: ${instance.recipeName}`);
  try {
    execSync(
      `${join(VLLM_REPO_PATH, "launch-cluster.sh")} --solo stop`,
      { cwd: VLLM_REPO_PATH, timeout: 30_000, stdio: "inherit" }
    );
  } catch {
    instance.process.kill("SIGTERM");
  }

  running.delete(deploymentId);
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

/**
 * Health check all running deployments.
 * Checks docker container status and scrapes vLLM metrics endpoint.
 */
export async function checkDeployments(): Promise<VllmStatus[]> {
  const results: VllmStatus[] = [];

  for (const [deploymentId, instance] of running) {
    const status: VllmStatus = {
      deploymentId,
      recipeName: instance.recipeName,
      port: instance.port,
      alive: !instance.process.killed && instance.process.exitCode === null,
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
          `curl -sf --max-time 3 http://localhost:${instance.port}/metrics`,
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
