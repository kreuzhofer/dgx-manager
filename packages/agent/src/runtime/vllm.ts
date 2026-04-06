import { execSync, spawn, ChildProcess } from "child_process";
import { existsSync } from "fs";
import { join } from "path";

const VLLM_REPO_PATH =
  process.env.VLLM_REPO_PATH || "/mnt/tank/src/github/spark-vllm-docker";

interface VllmInstance {
  process: ChildProcess;
  recipeName: string;
  port: number;
}

const running = new Map<string, VllmInstance>();

/**
 * Launch a vLLM inference server using a spark-vllm-docker recipe.
 * Returns the port the server will listen on.
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

  // Stop any existing instance for this deployment
  stopRecipe(deploymentId);

  // Strip recipes/ prefix and .yaml suffix to get recipe name for run-recipe.sh
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

  let logs = "";

  child.stdout?.on("data", (data: Buffer) => {
    const line = data.toString();
    logs += line;
    onLog?.(line);
    console.log(`[vllm:${recipeName}] ${line.trimEnd()}`);
  });

  child.stderr?.on("data", (data: Buffer) => {
    const line = data.toString();
    logs += line;
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

/** Stop a running vLLM instance for a deployment. */
export function stopRecipe(deploymentId: string): boolean {
  const instance = running.get(deploymentId);
  if (!instance) return false;

  console.log(`Stopping vLLM recipe: ${instance.recipeName}`);

  // Use launch-cluster.sh stop to clean up docker containers
  try {
    execSync(
      `${join(VLLM_REPO_PATH, "launch-cluster.sh")} --solo stop`,
      { cwd: VLLM_REPO_PATH, timeout: 30_000, stdio: "inherit" }
    );
  } catch {
    // Fall back to killing the process
    instance.process.kill("SIGTERM");
  }

  running.delete(deploymentId);
  return true;
}

/** Check if a deployment's vLLM instance is running. */
export function isRunning(deploymentId: string): boolean {
  return running.has(deploymentId);
}
