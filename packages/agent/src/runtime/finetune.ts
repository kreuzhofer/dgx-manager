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

/**
 * Check for running finetune containers after agent restart.
 * Reattaches log streaming so progress continues to flow to the dashboard.
 */
export function reattachFinetuneJobs(
  sendMsg: (type: string, payload: Record<string, unknown>) => void
): void {
  try {
    const output = execSync(
      'docker ps --format "{{.Names}}" --filter "name=dgx-finetune-" --filter "name=dgx-merge-"',
      { timeout: 10_000 }
    ).toString().trim();

    if (!output) return;

    for (const containerName of output.split("\n")) {
      // Extract jobId prefix from container name: dgx-finetune-{jobId12chars} or dgx-merge-{jobId12chars}
      const isMerge = containerName.startsWith("dgx-merge-");
      const jobIdPrefix = containerName.replace("dgx-finetune-", "").replace("dgx-merge-", "");
      if (!jobIdPrefix) continue;

      console.log(`[finetune] Reattaching to running ${isMerge ? "merge" : "training"} container: ${containerName}`);

      // Find the train.log or merge.log file written by lib/logging.py
      const logFileName = isMerge ? "merge.log" : "train.log";
      let logFilePath: string | null = null;
      try {
        const findOut = execSync(
          `docker exec ${containerName} find /workspace/outputs -name "${logFileName}" -maxdepth 3 2>/dev/null`,
          { timeout: 5_000 }
        ).toString().trim();
        logFilePath = findOut.split("\n")[0] || null;
      } catch { /* no log file yet */ }

      if (logFilePath) {
        // tail -f the log file — reliable across agent restarts
        const tailCmd = ["exec", containerName, "tail", "-n", "5", "-f", logFilePath];
        const logProc = spawn("docker", tailCmd, {
          stdio: ["ignore", "pipe", "pipe"],
        });

        const msgType = isMerge ? "agent:finetune:merge-progress" : "agent:finetune:progress";
        const speedSmoother = new SpeedSmoother(20);
        let currentPhase: TrainingPhase = isMerge ? "loading" : "training";

        const handleReattach = (data: Buffer) => {
          const text = data.toString();
          sendMsg(msgType, { jobId: jobIdPrefix, log: text });

          for (const line of text.split(/[\n\r]+/)) {
            if (!line.trim()) continue;
            const newPhase = detectPhase(line);
            if (newPhase && newPhase !== currentPhase) {
              currentPhase = newPhase;
              sendMsg(msgType, { jobId: jobIdPrefix, phase: currentPhase, phaseProgress: 0 });
            }
            const progress = parseProgress(line, currentPhase);
            if (progress) {
              currentPhase = progress.phase;
              if (progress.iterSpeed) speedSmoother.push(progress.iterSpeed);
              if (progress.phaseProgress !== undefined) {
                const etaSeconds = progress.step != null && progress.totalSteps != null
                  ? speedSmoother.eta(progress.totalSteps - progress.step)
                  : undefined;
                sendMsg(msgType, {
                  jobId: jobIdPrefix, phase: progress.phase, phaseProgress: progress.phaseProgress,
                  step: progress.step, totalSteps: progress.totalSteps, loss: progress.loss, etaSeconds,
                });
              } else if (progress.loss !== undefined) {
                sendMsg(msgType, { jobId: jobIdPrefix, phase: progress.phase, phaseProgress: -1, loss: progress.loss });
              }
            }
          }
        };

        logProc.stdout?.on("data", handleReattach);
        logProc.stderr?.on("data", handleReattach);

        logProc.on("exit", () => {
          console.log(`[finetune] Log stream ended for ${containerName}`);
        });

        running.set(jobIdPrefix, { jobId: jobIdPrefix, containerName, execProcess: logProc, aborted: false });
      }
    }
  } catch (err) {
    console.error("[finetune] Error reattaching:", err);
  }
}

type TrainingPhase = "container" | "setup" | "downloading" | "loading" | "tokenizing" | "training" | "eval" | "saving";

interface ProgressInfo {
  phase: TrainingPhase;
  phaseProgress?: number; // 0-1 within current phase
  step?: number;
  totalSteps?: number;
  loss?: number;
  lr?: number;
  evalLoss?: number;
  iterSpeed?: number; // iterations per second from tqdm
}

/**
 * Rolling average for smoothing iteration speed estimates.
 */
class SpeedSmoother {
  private samples: number[] = [];
  private maxSamples: number;

  constructor(windowSize: number = 20) {
    this.maxSamples = windowSize;
  }

  push(speed: number): void {
    if (speed > 0) {
      this.samples.push(speed);
      if (this.samples.length > this.maxSamples) {
        this.samples.shift();
      }
    }
  }

  /** Smoothed speed (iterations per second). */
  get average(): number {
    if (this.samples.length === 0) return 0;
    return this.samples.reduce((a, b) => a + b, 0) / this.samples.length;
  }

  /** Estimated seconds remaining given items left. */
  eta(remaining: number): number | undefined {
    const avg = this.average;
    if (avg <= 0 || remaining <= 0) return undefined;
    return Math.round(remaining / avg);
  }

  reset(): void {
    this.samples = [];
  }
}

/**
 * Detect the current training phase from a log line.
 * Order matters — more specific matches first.
 */
function detectPhase(line: string): TrainingPhase | null {
  const l = line.toLowerCase();
  // Container lifecycle
  if (l.includes("starting container") || l.includes("pulling image")) return "container";
  if (l.includes("container ready") || l.includes("=== ready ===")) return "setup";
  // Download (HF model files)
  if (l.includes("downloading") || l.includes("fetching") && (l.includes("files") || l.includes("model"))) return "downloading";
  // Model loading
  if (l.includes("loading model") || l.includes("loading weights") || l.includes("from_pretrained") || l.includes("model loaded")) return "loading";
  // Tokenizing — only trigger on the explicit label, NOT on tqdm bars
  if (l.includes("tokeniz")) return "tokenizing";
  // Training — only trigger on explicit markers, not tqdm
  if (l.includes("[train]") || l.includes("starting training") || l.includes("'loss'")) return "training";
  // Eval
  if (l.includes("eval_loss") || l.includes("evaluation") || l.includes("running eval")) return "eval";
  // Saving
  if (l.includes("lora adapter saved") || l.includes("save_pretrained")) return "saving";
  return null;
}

/**
 * Parse training log output for progress info.
 * Only extracts numeric progress from lines that match the CURRENT phase.
 */
function parseProgress(line: string, currentPhase: TrainingPhase): ProgressInfo | null {
  // Explicit training log from our callback: [TRAIN] step=5/50 loss=2.345 lr=0.0002
  const trainLogMatch = line.match(/\[TRAIN\]\s+step=(\d+)\/(\d+)\s+loss=([\d.eE+\-?]+)(?:\s+lr=([\d.eE+\-]+))?/);
  if (trainLogMatch) {
    const step = parseInt(trainLogMatch[1]);
    const total = parseInt(trainLogMatch[2]);
    const loss = trainLogMatch[3] !== "?" ? parseFloat(trainLogMatch[3]) : undefined;
    const lr = trainLogMatch[4] ? parseFloat(trainLogMatch[4]) : undefined;
    return {
      phase: "training",
      step,
      totalSteps: total,
      phaseProgress: total > 0 ? step / total : 0,
      loss,
      lr,
    };
  }

  // Eval result: [EVAL] eval_loss=11.38
  const evalLogMatch = line.match(/\[EVAL\]\s+eval_loss=([\d.]+)/);
  if (evalLogMatch) {
    return { phase: "eval", evalLoss: parseFloat(evalLogMatch[1]) };
  }

  // Training loss dict: {'loss': '51.79', ...} or {'loss': 51.79, ...}
  const dictMatch = line.match(/\{'loss':\s*'?([\d.]+)/);
  if (dictMatch) {
    return { phase: "training", loss: parseFloat(dictMatch[1]) };
  }

  // Generic tqdm bar: " 33%|███▎ | 1/3 [00:02<00:04, 2.66it/s]"
  // Route to the correct phase based on currentPhase
  const tqdmMatch = line.match(/\s*(\d+)%\|.*\|\s*(\d+)\/(\d+)/);
  if (tqdmMatch) {
    const pct = parseInt(tqdmMatch[1]);
    const current = parseInt(tqdmMatch[2]);
    const total = parseInt(tqdmMatch[3]);

    // Parse iteration speed: "2.66it/s" or "1.23s/it" or "10277.50 examples/s"
    let iterSpeed: number | undefined;
    const itsMatch = line.match(/([\d.]+)\s*it\/s/);
    const sitMatch = line.match(/([\d.]+)\s*s\/it/);
    const exMatch = line.match(/([\d.]+)\s*examples\/s/);
    if (itsMatch) iterSpeed = parseFloat(itsMatch[1]);
    else if (sitMatch) iterSpeed = 1 / parseFloat(sitMatch[1]);
    else if (exMatch) iterSpeed = parseFloat(exMatch[1]);

    if (currentPhase === "downloading") {
      return { phase: "downloading", phaseProgress: pct / 100, iterSpeed };
    }
    if (currentPhase === "loading") {
      return { phase: "loading", phaseProgress: pct / 100, iterSpeed };
    }
    if (currentPhase === "tokenizing") {
      return { phase: "tokenizing", phaseProgress: pct / 100, iterSpeed };
    }
    if (currentPhase === "training") {
      const lossMatch = line.match(/loss[=:]\s*([\d.]+)/i);
      return {
        phase: "training",
        phaseProgress: total > 0 ? current / total : 0,
        step: current,
        totalSteps: total,
        loss: lossMatch ? parseFloat(lossMatch[1]) : undefined,
        iterSpeed,
      };
    }
    if (currentPhase === "eval") {
      return { phase: "eval", phaseProgress: pct / 100, iterSpeed };
    }
    return { phase: currentPhase, phaseProgress: pct / 100, iterSpeed };
  }

  // Eval results
  const evalMatch = line.match(/\{'eval_loss':\s*'?([\d.]+)/);
  if (evalMatch) {
    return { phase: "eval", loss: parseFloat(evalMatch[1]) };
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
  onProgress: (phase: string, phaseProgress: number, extra?: { step?: number; totalSteps?: number; loss?: number; lr?: number; evalLoss?: number; etaSeconds?: number }) => void;
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

    // Pre-create output dir on host
    const hostOutputDir = outputDir.replace("/workspace/", "/mnt/tank/");
    try { execSync(`mkdir -p "${hostOutputDir}" && chmod 777 "${hostOutputDir}"`, { timeout: 5_000 }); } catch { /* */ }

    callbacks.onLog(`Starting container ${containerName} with image ${recipe.container.image}...\n`);

    // Build docker run command — run as root for pip install but output dir is pre-created
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
    // HuggingFace dataset IDs (e.g., "b-mc2/sql-create-context") are passed
    // directly to the training script — it handles download via load_dataset()

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
    const speedSmoother = new SpeedSmoother(20);

    const handleOutput = (data: Buffer) => {
      const text = data.toString();
      callbacks.onLog(text);

      for (const line of text.split(/[\n\r]+/)) {
        if (!line.trim()) continue;

        // Detect phase transitions
        const newPhase = detectPhase(line);
        if (newPhase && newPhase !== currentPhase) {
          currentPhase = newPhase;
          speedSmoother.reset(); // reset ETA on phase change
          callbacks.onProgress(currentPhase, 0);
        }

        // Parse progress within current phase
        const progress = parseProgress(line, currentPhase);
        if (progress) {
          currentPhase = progress.phase;

          // Feed speed smoother
          if (progress.iterSpeed) {
            speedSmoother.push(progress.iterSpeed);
          }

          if (progress.phaseProgress !== undefined) {
            // Compute ETA from smoothed speed
            let etaSeconds: number | undefined;
            if (progress.step != null && progress.totalSteps != null) {
              etaSeconds = speedSmoother.eta(progress.totalSteps - progress.step);
            } else if (progress.phaseProgress > 0 && progress.phaseProgress < 1) {
              // Estimate remaining from progress ratio and speed
              const elapsed = speedSmoother.average;
              if (elapsed > 0) {
                // phaseProgress = done/total, so remaining fraction = 1 - phaseProgress
                // Can't compute without total count, use tqdm's own ETA as fallback
              }
            }

            callbacks.onProgress(progress.phase, progress.phaseProgress, {
              step: progress.step,
              totalSteps: progress.totalSteps,
              loss: progress.loss,
              lr: progress.lr,
              evalLoss: progress.evalLoss,
              etaSeconds,
            });
          } else if (progress.loss !== undefined || progress.evalLoss !== undefined) {
            callbacks.onProgress(progress.phase, -1, { loss: progress.loss, lr: progress.lr, evalLoss: progress.evalLoss });
          }
        }
      }
    };

    execProc.stdout?.on("data", handleOutput);
    execProc.stderr?.on("data", handleOutput);

    execProc.on("exit", (code) => {
      if (instance.aborted) {
        return;
      }

      // Make output files world-readable (container runs as root)
      try {
        execSync(`docker exec ${containerName} chmod -R a+rw ${outputDir}`, { timeout: 30_000, stdio: "ignore" });
      } catch { /* best effort */ }

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

/**
 * Merge a LoRA adapter into the base model and save the full merged model.
 */
export async function mergeLoraAdapter(
  jobId: string,
  baseModel: string,
  adapterPath: string,
  mergedOutputDir: string,
  callbacks: FinetuneCallbacks
): Promise<void> {
  const containerName = `dgx-merge-${jobId.slice(0, 12)}`;
  const containerImage = "nvcr.io/nvidia/pytorch:25.11-py3";
  const mergeScript = "/workspace/src/github/dgx-manager-fine-tune-recipes/scripts/merge.py";

  // Translate NFS paths to container paths
  const containerAdapterPath = adapterPath.replace("/mnt/tank/", "/workspace/");
  const containerOutputDir = mergedOutputDir.replace("/mnt/tank/", "/workspace/");

  try {
    try { execSync(`docker rm -f ${containerName}`, { timeout: 15_000, stdio: "ignore" }); } catch { /* */ }

    callbacks.onLog(`Starting merge container...\n`);
    callbacks.onProgress("loading", 0);

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
      "-e", "HF_HOME=/workspace/models",
      "-e", "HF_HUB_OFFLINE=0",
    ];
    if (process.env.HF_TOKEN) {
      dockerArgs.push("-e", `HF_TOKEN=${process.env.HF_TOKEN}`);
    }
    dockerArgs.push(
      "-v", "/mnt/tank:/workspace",
      "--entrypoint", "sleep",
      containerImage,
      "infinity"
    );

    execSync(`docker ${dockerArgs.join(" ")}`, { timeout: 120_000 });

    // Install merge deps
    callbacks.onLog("Installing merge dependencies...\n");
    try {
      execSync(`docker exec ${containerName} pip install -q peft transformers accelerate safetensors`, { timeout: 300_000, stdio: "ignore" });
    } catch (err) {
      callbacks.onComplete("failed", undefined, `Failed to install merge deps: ${err}`);
      try { execSync(`docker rm -f ${containerName}`, { timeout: 15_000, stdio: "ignore" }); } catch { /* */ }
      return;
    }
    callbacks.onLog("Dependencies installed.\n");

    callbacks.onLog(`\nRunning merge: ${baseModel} + ${containerAdapterPath} → ${containerOutputDir}\n\n`);

    const mergeProc = spawn("docker", [
      "exec", containerName,
      "python", mergeScript,
      "--base_model", baseModel,
      "--adapter_path", containerAdapterPath,
      "--output_dir", containerOutputDir,
    ], { stdio: ["ignore", "pipe", "pipe"] });

    mergeProc.stdout?.on("data", (data: Buffer) => {
      const text = data.toString();
      callbacks.onLog(text);
      const l = text.toLowerCase();
      if (l.includes("loading base model")) callbacks.onProgress("loading", 0.1);
      if (l.includes("base model loaded")) callbacks.onProgress("loading", 0.4);
      if (l.includes("adapter loaded")) callbacks.onProgress("loading", 0.5);
      if (l.includes("merge complete")) callbacks.onProgress("saving", 0.6);
      if (l.includes("saving merged model")) callbacks.onProgress("saving", 0.7);
      if (l.includes("model saved")) callbacks.onProgress("saving", 0.9);
      if (l.includes("saving tokenizer")) callbacks.onProgress("saving", 0.95);
    });
    mergeProc.stderr?.on("data", (data: Buffer) => callbacks.onLog(data.toString()));

    await new Promise<void>((resolve, reject) => {
      mergeProc.on("exit", (code) => {
        // Make output files world-readable
        try {
          execSync(`docker exec ${containerName} chmod -R a+rw ${containerOutputDir}`, { timeout: 30_000, stdio: "ignore" });
        } catch { /* best effort */ }

        if (code === 0) {
          callbacks.onProgress("saving", 1.0);
          callbacks.onComplete("completed", mergedOutputDir);
        } else {
          callbacks.onComplete("failed", undefined, `Merge failed with exit code ${code}`);
        }
        try { execSync(`docker rm -f ${containerName}`, { timeout: 15_000, stdio: "ignore" }); } catch { /* */ }
        resolve();
      });
    });
  } catch (err) {
    callbacks.onComplete("failed", undefined, `Merge error: ${err}`);
    try { execSync(`docker rm -f ${containerName}`, { timeout: 15_000, stdio: "ignore" }); } catch { /* */ }
  }
}
