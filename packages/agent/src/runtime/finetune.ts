/**
 * Fine-tune job runner for DGX Manager agent.
 * Launches training in a Docker container using recipe scripts.
 */

import { spawn, execSync, ChildProcess } from "child_process";
import { existsSync } from "fs";
import { join } from "path";
import { getTrainingRepoPath, type TrainingRecipe } from "../training-recipes.js";

interface FinetuneInstance {
  jobId: string;
  containerName: string;
  execProcess: ChildProcess | null;
  aborted: boolean;
}

const running = new Map<string, FinetuneInstance>();

type TrainingPhase = "setup" | "downloading" | "loading" | "tokenizing" | "training" | "eval" | "saving";

interface ProgressInfo {
  phase: TrainingPhase;
  phaseProgress?: number; // 0-1 within current phase
  step?: number;
  totalSteps?: number;
  loss?: number;
}

/**
 * Detect the current training phase from a log line.
 */
function detectPhase(line: string): TrainingPhase | null {
  const l = line.toLowerCase();
  if (l.includes("downloading") || l.includes("fetching") || l.includes("config.json") || l.includes(".safetensors")) return "downloading";
  if (l.includes("loading model") || l.includes("loading weights") || l.includes("from_pretrained") || l.includes("model loaded")) return "loading";
  if (l.includes("tokeniz")) return "tokenizing";
  if (l.includes("starting training") || l.includes("'loss'") || /\d+\/\d+\s*\[/.test(line)) return "training";
  if (l.includes("eval_loss") || l.includes("evaluation")) return "eval";
  if (l.includes("saving model") || l.includes("lora adapter saved") || l.includes("save_pretrained")) return "saving";
  return null;
}

/**
 * Parse training log output for progress info.
 */
function parseProgress(line: string, currentPhase: TrainingPhase): ProgressInfo | null {
  // HF download progress: "Downloading model.safetensors:  45%|████▌ | 1.2G/2.5G"
  // or tqdm download bars: " 45%|████▌     | 1.2G/2.5G [01:23<01:30, 15.2MB/s]"
  // or file-count: "Fetching 12 files: 75%|████ | 9/12"
  if (currentPhase === "downloading") {
    const pctMatch = line.match(/(\d+)%\|/);
    if (pctMatch) {
      return { phase: "downloading", phaseProgress: parseInt(pctMatch[1]) / 100 };
    }
  }

  // Model loading progress: "Loading weights:  45%|████▌     | 907/2011"
  const loadMatch = line.match(/[Ll]oading\s+weights?:\s*(\d+)%/) || (currentPhase === "loading" && line.match(/(\d+)%\|.*\|\s*\d+\/\d+/));
  if (loadMatch) {
    return { phase: "loading", phaseProgress: parseInt(loadMatch[1]) / 100 };
  }

  // Tokenizing progress: "Tokenizing (num_proc=4):  60%|██████    | 3/5"
  const tokenMatch = line.match(/[Tt]okeniz.*?(\d+)%/);
  if (tokenMatch) {
    return { phase: "tokenizing", phaseProgress: parseInt(tokenMatch[1]) / 100 };
  }

  // Training loss dict: {'loss': '51.79', ...}
  const dictMatch = line.match(/\{'loss':\s*'?([\d.]+)/);
  if (dictMatch) {
    return { phase: "training", loss: parseFloat(dictMatch[1]) };
  }

  // Training tqdm: " 33%|███▎      | 1/3 [00:02<00:04,  2.13s/it]" or "  0%|          | 0/3"
  // Must be in training phase to avoid matching loading/tokenizing tqdm
  if (currentPhase === "training") {
    const tqdmMatch = line.match(/\s+(\d+)%\|.*\|\s*(\d+)\/(\d+)/);
    if (tqdmMatch) {
      const step = parseInt(tqdmMatch[2]);
      const total = parseInt(tqdmMatch[3]);
      const lossMatch = line.match(/loss[=:]\s*([\d.]+)/i);
      return {
        phase: "training",
        phaseProgress: total > 0 ? step / total : 0,
        step,
        totalSteps: total,
        loss: lossMatch ? parseFloat(lossMatch[1]) : undefined,
      };
    }
  }

  // Eval progress
  if (currentPhase === "eval" || line.includes("eval")) {
    const evalMatch = line.match(/\{'eval_loss':\s*'?([\d.]+)/);
    if (evalMatch) {
      return { phase: "eval", loss: parseFloat(evalMatch[1]) };
    }
  }

  return null;
}

/**
 * Wait for the container's entrypoint to signal readiness.
 * Polls for /tmp/.ready inside the container.
 */
async function waitForReady(
  containerName: string,
  timeoutMs: number = 300_000,
  onLog?: (line: string) => void
): Promise<boolean> {
  const start = Date.now();
  onLog?.("Waiting for container setup to complete...\n");

  // Stream docker logs while waiting
  const logProc = spawn("docker", ["logs", "-f", containerName], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  logProc.stdout?.on("data", (data: Buffer) => onLog?.(data.toString()));
  logProc.stderr?.on("data", (data: Buffer) => onLog?.(data.toString()));

  while (Date.now() - start < timeoutMs) {
    try {
      execSync(`docker exec ${containerName} test -f /tmp/.ready`, {
        timeout: 5_000,
        stdio: "ignore",
      });
      logProc.kill();
      onLog?.("Container ready.\n");
      return true;
    } catch {
      await new Promise((r) => setTimeout(r, 5_000));
    }
  }
  logProc.kill();
  return false;
}

export interface FinetuneCallbacks {
  onLog: (line: string) => void;
  onProgress: (phase: string, phaseProgress: number, extra?: { step?: number; totalSteps?: number; loss?: number }) => void;
  onComplete: (status: "completed" | "failed" | "stopped", outputPath?: string, error?: string) => void;
}

/**
 * Start a fine-tune job inside a Docker container.
 */
export async function startFinetuneJob(
  jobId: string,
  recipeFile: string,
  dataset: string,
  outputDir: string,
  config: Record<string, unknown>,
  callbacks: FinetuneCallbacks
): Promise<void> {
  const repoPath = getTrainingRepoPath();
  const recipeDir = join(repoPath, recipeFile);
  const recipeYamlPath = join(recipeDir, "recipe.yaml");

  if (!existsSync(recipeYamlPath)) {
    callbacks.onComplete("failed", undefined, `Recipe not found: ${recipeFile}`);
    return;
  }

  // Discover the recipe to get its metadata
  const { discoverTrainingRecipes } = await import("../training-recipes.js");
  const allRecipes = discoverTrainingRecipes() as TrainingRecipe[];
  const recipe = allRecipes.find((r) => r.file === recipeFile);
  if (!recipe) {
    callbacks.onComplete("failed", undefined, `Could not parse recipe: ${recipeFile}`);
    return;
  }

  const containerName = `dgx-finetune-${jobId.slice(0, 12)}`;
  const instance: FinetuneInstance = { jobId, containerName, execProcess: null, aborted: false };
  running.set(jobId, instance);

  try {
    // Stop any existing container with the same name
    try {
      execSync(`docker rm -f ${containerName}`, { timeout: 15_000, stdio: "ignore" });
    } catch { /* ignore */ }

    // Flush page cache
    try {
      execSync("sync && echo 3 > /proc/sys/vm/drop_caches", { timeout: 10_000, stdio: "ignore" });
    } catch { /* ignore — needs root */ }

    // Build custom image if needed (e.g., Unsloth)
    if (recipe.container.build_context) {
      try {
        // Check if image already exists
        execSync(`docker image inspect ${recipe.container.image}`, { timeout: 10_000, stdio: "ignore" });
        callbacks.onLog(`Image ${recipe.container.image} already exists.\n`);
      } catch {
        callbacks.onLog(`Building image ${recipe.container.image} from ${recipe.container.build_context}...\n`);
        callbacks.onLog(`This may take 10-20 minutes on first build.\n`);
        const buildProc = spawn("docker", [
          "build", "-t", recipe.container.image, recipe.container.build_context,
        ], { stdio: ["ignore", "pipe", "pipe"] });
        buildProc.stdout?.on("data", (data: Buffer) => callbacks.onLog(data.toString()));
        buildProc.stderr?.on("data", (data: Buffer) => callbacks.onLog(data.toString()));
        await new Promise<void>((resolve, reject) => {
          buildProc.on("exit", (code) => {
            if (code === 0) resolve();
            else reject(new Error(`Docker build failed with code ${code}`));
          });
        });
        callbacks.onLog(`Image ${recipe.container.image} built successfully.\n`);
      }
    }

    callbacks.onLog(`Starting container ${containerName} with image ${recipe.container.image}...\n`);

    // Build docker run command
    const entrypointPath = `/workspace/src/github/dgx-manager-fine-tune-recipes/${recipeFile}/${recipe.scripts.entrypoint}`;
    const dockerArgs = [
      "run", "-d",
      "--name", containerName,
      "--gpus", "all",
      "--network", "host",
      "--ipc", "host",
      "--privileged",
      "--ulimit", "memlock=-1",
      "--shm-size=1g",
      "--user", "root",
      "-e", "CUDA_VISIBLE_DEVICES=0",
      "-e", "PYTHONUNBUFFERED=1",
      "-e", `HF_HOME=/workspace/models`,
      "-e", "HF_HUB_OFFLINE=0",
      "-e", "HF_DATASETS_CACHE=/tmp/hf_datasets_cache",
    ];

    // Pass HF_TOKEN if available
    if (process.env.HF_TOKEN) {
      dockerArgs.push("-e", `HF_TOKEN=${process.env.HF_TOKEN}`);
    }

    // Add InfiniBand device if available
    try {
      execSync("test -d /dev/infiniband", { timeout: 2_000, stdio: "ignore" });
      dockerArgs.push("--device=/dev/infiniband/");
      dockerArgs.push("-e", "NCCL_IB_DISABLE=0");
    } catch { /* no IB */ }

    dockerArgs.push(
      "-v", "/mnt/tank:/workspace",
      "--entrypoint", "bash",
      recipe.container.image,
      entrypointPath
    );

    // Pull image first (skip if locally built)
    if (!recipe.container.build_context) {
      callbacks.onLog(`Pulling image ${recipe.container.image}...\n`);
      execSync(`docker pull ${recipe.container.image}`, { timeout: 1800_000, stdio: "ignore" });
    }

    callbacks.onLog(`Starting container...\n`);
    execSync(`docker ${dockerArgs.join(" ")}`, { timeout: 120_000 });

    if (instance.aborted) {
      cleanup(containerName, jobId);
      return;
    }

    // Wait for container readiness
    const ready = await waitForReady(containerName, 300_000, callbacks.onLog);
    if (!ready) {
      callbacks.onComplete("failed", undefined, "Container setup timed out after 5 minutes");
      cleanup(containerName, jobId);
      return;
    }

    if (instance.aborted) {
      cleanup(containerName, jobId);
      return;
    }

    // Translate host NFS paths to container paths (/mnt/tank → /workspace)
    let resolvedDataset = dataset;
    if (resolvedDataset.startsWith("/mnt/tank/")) {
      resolvedDataset = resolvedDataset.replace("/mnt/tank/", "/workspace/");
    }

    // If dataset is a HuggingFace ID (not an absolute path), download it first
    if (!resolvedDataset.startsWith("/")) {
      callbacks.onLog(`Downloading dataset: ${dataset}...\n`);
      try {
        const dlResult = execSync(
          `docker exec ${containerName} python -c "from datasets import load_dataset; ds = load_dataset('${dataset}', split='train'); ds.to_json('/tmp/hf_dataset.jsonl')"`,
          { timeout: 600_000 }
        );
        callbacks.onLog(dlResult.toString());
        resolvedDataset = "/tmp/hf_dataset.jsonl";
        callbacks.onLog(`Dataset downloaded to ${resolvedDataset}\n`);
      } catch (err) {
        callbacks.onComplete("failed", undefined, `Failed to download dataset: ${err}`);
        cleanup(containerName, jobId);
        return;
      }
    }

    if (instance.aborted) {
      cleanup(containerName, jobId);
      return;
    }

    // Merge recipe defaults with user config overrides
    const merged = { ...recipe.defaults, ...config };

    // Build launch command args
    const launchPath = `/workspace/src/github/dgx-manager-fine-tune-recipes/${recipeFile}/${recipe.scripts.launch}`;
    const trainArgs: string[] = [
      "--model_name", recipe.base_model,
      "--dataset", resolvedDataset,
      "--output_dir", outputDir,
    ];

    // Add all merged config as CLI args
    for (const [key, value] of Object.entries(merged)) {
      if (value !== undefined && value !== null && key !== "lora_target_modules") {
        trainArgs.push(`--${key}`, String(value));
      }
    }
    // Handle lora_target_modules specially (comma-separated string)
    if (merged.lora_target_modules) {
      trainArgs.push("--lora_target_modules", String(merged.lora_target_modules));
    }

    callbacks.onLog(`\nLaunching training: ${launchPath}\n`);
    callbacks.onLog(`Args: ${trainArgs.join(" ")}\n\n`);

    // Exec the launch script inside the container
    const execProc = spawn("docker", [
      "exec", containerName,
      "bash", launchPath,
      ...trainArgs,
    ], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    instance.execProcess = execProc;

    let currentPhase: TrainingPhase = "setup";

    const handleOutput = (data: Buffer) => {
      const text = data.toString();
      callbacks.onLog(text);

      for (const line of text.split(/[\n\r]+/)) {
        if (!line.trim()) continue;

        // Detect phase transitions
        const newPhase = detectPhase(line);
        if (newPhase && newPhase !== currentPhase) {
          currentPhase = newPhase;
          // Notify phase change even without numeric progress
          callbacks.onProgress(currentPhase, 0);
        }

        // Parse progress within current phase
        const progress = parseProgress(line, currentPhase);
        if (progress) {
          currentPhase = progress.phase;
          if (progress.phaseProgress !== undefined) {
            callbacks.onProgress(progress.phase, progress.phaseProgress, {
              step: progress.step,
              totalSteps: progress.totalSteps,
              loss: progress.loss,
            });
          } else if (progress.loss !== undefined) {
            callbacks.onProgress(progress.phase, -1, { loss: progress.loss });
          }
        }
      }
    };

    execProc.stdout?.on("data", handleOutput);
    execProc.stderr?.on("data", handleOutput);

    execProc.on("exit", (code) => {
      if (instance.aborted) {
        // Stop was requested — don't report completion
        return;
      }

      if (code === 0) {
        const adapterPath = `${outputDir}/lora_adapter`;
        callbacks.onProgress("saving", 1.0);
        callbacks.onComplete("completed", adapterPath);
      } else {
        callbacks.onComplete("failed", undefined, `Training exited with code ${code}`);
      }

      cleanup(containerName, jobId);
    });

  } catch (err) {
    callbacks.onComplete("failed", undefined, `Failed to start training: ${err}`);
    cleanup(containerName, jobId);
  }
}

/**
 * Stop a running fine-tune job.
 */
export function stopFinetuneJob(jobId: string): boolean {
  const instance = running.get(jobId);
  if (!instance) return false;

  instance.aborted = true;

  // Kill the exec process
  if (instance.execProcess?.pid) {
    try {
      process.kill(-instance.execProcess.pid, "SIGTERM");
    } catch {
      try { instance.execProcess.kill("SIGTERM"); } catch { /* ignore */ }
    }
  }

  cleanup(instance.containerName, jobId);
  return true;
}

/** Clean up Docker container and remove from tracking. */
function cleanup(containerName: string, jobId: string) {
  try {
    execSync(`docker rm -f ${containerName}`, { timeout: 30_000, stdio: "ignore" });
  } catch { /* ignore */ }
  running.delete(jobId);
}
