/**
 * Fine-tune job runner for DGX Manager agent.
 * Launches training in a Docker container using recipe scripts.
 */

import { spawn, execSync, ChildProcess } from "child_process";
import { existsSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { SHARED_STORAGE, WORKSPACE, toContainerPath, toHostPath } from "../env.js";
import { getTrainingRepoPath, type TrainingRecipe } from "../training-recipes.js";

interface FinetuneInstance {
  jobId: string;
  containerName: string;
  execProcess: ChildProcess | null;
  aborted: boolean;
  workerIps?: string[];
  watchdogTimer?: NodeJS.Timeout;
}

const running = new Map<string, FinetuneInstance>();

/**
 * Persist multi-node worker IPs to NFS so cleanup can find them after agent restart.
 * Without this, a restarted agent reattaches to the head container but loses the
 * worker list — stop commands then leak worker containers on other nodes.
 */
function clusterFilePath(hostOutputDir: string): string {
  return `${hostOutputDir}/.cluster-workers.json`;
}

function persistClusterWorkers(hostOutputDir: string, workerIps: string[]): void {
  try {
    writeFileSync(clusterFilePath(hostOutputDir), JSON.stringify(workerIps));
  } catch (err) {
    console.error(`[finetune] Failed to persist cluster workers:`, err);
  }
}

function loadClusterWorkers(hostOutputDir: string): string[] | undefined {
  try {
    const raw = readFileSync(clusterFilePath(hostOutputDir), "utf-8");
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.every((x) => typeof x === "string")) {
      return parsed;
    }
  } catch { /* missing or invalid */ }
  return undefined;
}

/** Find the host output dir for a job by scanning shared storage. */
function findOutputDirForJob(jobId: string): string | undefined {
  try {
    const out = execSync(
      `ls -d ${SHARED_STORAGE}/outputs/${jobId}* 2>/dev/null | head -1`,
      { timeout: 5_000, shell: "/bin/bash" }
    ).toString().trim();
    return out || undefined;
  } catch {
    return undefined;
  }
}

/**
 * Force-clean a job's containers across head + workers when the agent has no
 * in-memory record of it. Used as fallback in stopFinetuneJob after restart.
 */
function forceCleanupJob(jobId: string): boolean {
  const containerName = `dgx-finetune-${jobId.slice(0, 12)}`;
  let didSomething = false;

  // Recover worker list from persisted state (if multi-node)
  const outputDir = findOutputDirForJob(jobId);
  const workerIps = outputDir ? loadClusterWorkers(outputDir) : undefined;

  // Head container — silent if not present
  try {
    execSync(`docker rm -f ${containerName}`, { timeout: 30_000, stdio: "ignore" });
    didSomething = true;
  } catch { /* not present */ }

  // Worker containers via SSH
  if (workerIps && workerIps.length > 0) {
    const sshUser = process.env.SSH_USER || process.env.USER || "daniel";
    for (const ip of workerIps) {
      try {
        execSync(
          `ssh -o StrictHostKeyChecking=no ${sshUser}@${ip} "docker rm -f ${containerName}"`,
          { timeout: 15_000, stdio: "ignore" }
        );
        didSomething = true;
      } catch { /* not present or unreachable */ }
    }
  }

  return didSomething;
}

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
      // Use the jobIdPrefix from container name to find the correct output directory
      const logFileName = isMerge ? "merge.log" : "train.log";
      let logFilePath: string | null = null;
      try {
        // First try exact match by job ID prefix
        const exactPath = `${WORKSPACE}/outputs/${jobIdPrefix}*/${logFileName}`;
        const findOut = execSync(
          `docker exec ${containerName} bash -c 'ls ${exactPath} 2>/dev/null || find ${WORKSPACE}/outputs -name "${logFileName}" -path "*${jobIdPrefix}*" -maxdepth 3 2>/dev/null'`,
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

        // Completion-detection state. Without this, when the agent is
        // restarted mid-training (e.g. cmd:update), the original execProc
        // that fires `agent:finetune:complete` on container exit is lost.
        // The reattached log stream then runs forever even after training
        // succeeds — DB stays at status=starting, /merge endpoint is
        // gated, deploy is gated. We fix that by watching the log for
        // the script's own success marker and emitting the completion
        // event ourselves. Idempotent — we only fire once.
        let completionEmitted = false;
        const successMarker = isMerge
          ? /Merged model saved to (\S+)/i
          : /LoRA adapter saved to (\S+)/i;
        const failureMarker = /(Traceback \(most recent call last\)|RuntimeError|FAILED|train\.py FAILED|merge\.py FAILED)/i;

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
                  step: progress.step, totalSteps: progress.totalSteps, loss: progress.loss,
                  lr: progress.lr, evalLoss: progress.evalLoss, etaSeconds,
                });
              } else if (progress.loss !== undefined || progress.evalLoss !== undefined) {
                sendMsg(msgType, {
                  jobId: jobIdPrefix, phase: progress.phase, phaseProgress: -1,
                  loss: progress.loss, lr: progress.lr, evalLoss: progress.evalLoss,
                });
              }
            }

            if (completionEmitted) continue;
            const ok = line.match(successMarker);
            if (ok) {
              completionEmitted = true;
              const outputPath = ok[1] || undefined;
              const completeMsg = isMerge ? "agent:finetune:merge-complete" : "agent:finetune:complete";
              const payload = isMerge
                ? { jobId: jobIdPrefix, status: "completed", mergedPath: outputPath ?? null }
                : { jobId: jobIdPrefix, status: "completed", outputPath };
              console.log(`[finetune][reattach] ${completeMsg} for ${jobIdPrefix} (output=${outputPath})`);
              sendMsg(completeMsg, payload);
            } else if (failureMarker.test(line)) {
              completionEmitted = true;
              const completeMsg = isMerge ? "agent:finetune:merge-complete" : "agent:finetune:complete";
              const payload = isMerge
                ? { jobId: jobIdPrefix, status: "failed", error: "training failed (detected via reattached log)" }
                : { jobId: jobIdPrefix, status: "failed", error: "training failed (detected via reattached log)" };
              console.log(`[finetune][reattach] ${completeMsg} (failed) for ${jobIdPrefix}`);
              sendMsg(completeMsg, payload);
            }
          }
        };

        logProc.stdout?.on("data", handleReattach);
        logProc.stderr?.on("data", handleReattach);

        logProc.on("exit", () => {
          console.log(`[finetune] Log stream ended for ${containerName}`);
        });

        // Restore cluster worker IPs from NFS so cleanup can reach them after restart
        const outputDir = findOutputDirForJob(jobIdPrefix);
        const workerIps = outputDir ? loadClusterWorkers(outputDir) : undefined;

        running.set(jobIdPrefix, { jobId: jobIdPrefix, containerName, execProcess: logProc, aborted: false, workerIps });

        // Cover the case where the success marker was already in the log
        // BEFORE the agent restarted: tail starts at -n 5 so we'd see only
        // the very last lines. If the saved-marker happens to be in those
        // 5 lines we'll catch it; otherwise the operator needs to re-trigger
        // (the log was already flushed). For training runs the saved
        // marker is the LAST line emitted, so -n 5 covers the typical case.
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
  clusterNodeIps: string[] | undefined,
  callbacks: FinetuneCallbacks,
  resumeFromCheckpoint: boolean = false
): Promise<void> {
  const isMultiNode = clusterNodeIps && clusterNodeIps.length > 1;
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
  const workerIps = isMultiNode ? clusterNodeIps!.slice(1) : undefined;
  const instance: FinetuneInstance = { jobId, containerName, execProcess: null, aborted: false, workerIps };
  running.set(jobId, instance);

  // Persist multi-node worker IPs to NFS so cleanup survives agent restart.
  if (workerIps && workerIps.length > 0) {
    const hostOutputDir = toHostPath(outputDir);
    try { execSync(`mkdir -p "${hostOutputDir}"`, { timeout: 5_000 }); } catch { /* */ }
    persistClusterWorkers(hostOutputDir, workerIps);
  }

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
    const hostOutputDir = toHostPath(outputDir);
    try { execSync(`mkdir -p "${hostOutputDir}" && chmod 777 "${hostOutputDir}"`, { timeout: 5_000 }); } catch { /* */ }

    callbacks.onLog(`Starting container ${containerName} with image ${recipe.container.image}...\n`);

    // Build docker run command — run as root for pip install but output dir is pre-created
    const entrypointPath = `${WORKSPACE}/src/github/dgx-manager-fine-tune-recipes/${recipeFile}/${recipe.scripts.entrypoint}`;
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
      "-e", `HF_HOME=${WORKSPACE}/models`,
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
      dockerArgs.push("-e", "NCCL_SOCKET_IFNAME=enp1s0f0np0");
      dockerArgs.push("-e", "GLOO_SOCKET_IFNAME=enp1s0f0np0");
      dockerArgs.push("-e", "NCCL_IB_HCA=rocep1s0f0");
    } catch { /* no IB */ }

    dockerArgs.push("-v", `${SHARED_STORAGE}:${WORKSPACE}`);
    // Mount SSH keys for multi-node inter-container communication
    if (isMultiNode) {
      const sshDir = `${process.env.HOME || "/root"}/.ssh`;
      dockerArgs.push("-v", `${sshDir}:/tmp/.ssh:ro`);
    }
    dockerArgs.push("--entrypoint", "bash", recipe.container.image, entrypointPath);

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

    // Translate host paths to container paths
    let resolvedDataset = dataset;
    if (resolvedDataset.startsWith(`${SHARED_STORAGE}/`)) {
      resolvedDataset = toContainerPath(resolvedDataset);
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
    const launchPath = `${WORKSPACE}/src/github/dgx-manager-fine-tune-recipes/${recipeFile}/${recipe.scripts.launch}`;
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

    if (resumeFromCheckpoint) {
      trainArgs.push("--resume_from_checkpoint", "true");
      callbacks.onLog(`\n=== RESUMING from latest checkpoint in ${outputDir} ===\n\n`);
    }

    // Multi-node: generate hostfile, start worker containers via SSH
    if (isMultiNode && clusterNodeIps) {
      callbacks.onLog(`\nMulti-node training: ${clusterNodeIps.length} nodes\n`);

      // Generate hostfile inside container
      const hostfileContent = clusterNodeIps.map(ip => `${ip} slots=1`).join("\\n");
      execSync(`docker exec ${containerName} bash -c 'echo -e "${hostfileContent}" > /tmp/hostfile.txt'`, { timeout: 5_000 });
      callbacks.onLog(`Hostfile: ${clusterNodeIps.join(", ")}\n`);

      // Start training containers on worker nodes via SSH
      const workerIps = clusterNodeIps.slice(1);
      const sshUser = process.env.SSH_USER || process.env.USER || "daniel";
      for (const workerIp of workerIps) {
        callbacks.onLog(`Starting worker container on ${workerIp}...\n`);
        try {
          // Start the same container image on the worker via host SSH
          const workerDockerCmd = [
            "docker run -d",
            `--name ${containerName}`,
            "--gpus all --network host --ipc host --privileged",
            "--ulimit memlock=-1 --shm-size=1g --user root",
            `-e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1`,
            `-e HF_HOME=${WORKSPACE}/models -e HF_HUB_OFFLINE=0`,
            // NCCL InfiniBand config — must match head node
            `-e NCCL_IB_DISABLE=0`,
            `-e NCCL_SOCKET_IFNAME=enp1s0f0np0`,
            `-e GLOO_SOCKET_IFNAME=enp1s0f0np0`,
            `-e NCCL_IB_HCA=rocep1s0f0`,
            `--device=/dev/infiniband/`,
            process.env.HF_TOKEN ? `-e HF_TOKEN=${process.env.HF_TOKEN}` : "",
            `-v ${SHARED_STORAGE}:${WORKSPACE}`,
            `-v /home/${sshUser}/.ssh:/tmp/.ssh:ro`,
            `--entrypoint bash ${recipe.container.image}`,
            entrypointPath,
          ].filter(Boolean).join(" ");

          execSync(`ssh -o StrictHostKeyChecking=no ${sshUser}@${workerIp} "docker rm -f ${containerName} 2>/dev/null; ${workerDockerCmd}"`, {
            timeout: 120_000,
          });

          // Wait for worker readiness
          let workerReady = false;
          for (let attempt = 0; attempt < 60; attempt++) {
            try {
              execSync(`ssh ${sshUser}@${workerIp} "docker exec ${containerName} test -f /tmp/.ready"`, { timeout: 5_000, stdio: "ignore" });
              workerReady = true;
              break;
            } catch { /* not ready yet */ }
            await new Promise(r => setTimeout(r, 5_000));
          }
          if (!workerReady) {
            callbacks.onComplete("failed", undefined, `Worker ${workerIp} setup timed out`);
            cleanup(containerName, jobId);
            return;
          }
          callbacks.onLog(`Worker ${workerIp} ready.\n`);

          // Generate hostfile on worker too
          execSync(`ssh ${sshUser}@${workerIp} "docker exec ${containerName} bash -c 'echo -e \\"${hostfileContent}\\" > /tmp/hostfile.txt'"`, { timeout: 5_000 });

          // Launch training on worker, redirecting stdout+stderr to a per-worker
          // log file on shared NFS so we can post-mortem worker failures (without
          // this, detached `docker exec -d` discards all output and we fly blind
          // when ranks > 0 crash).
          const workerLogPath = `${outputDir}/worker-${workerIp.replace(/\./g, "_")}.log`;
          const workerLaunchCmd = `docker exec -d ${containerName} bash -c 'bash ${launchPath} --hostfile /tmp/hostfile.txt ${trainArgs.join(" ")} > ${workerLogPath} 2>&1'`;
          execSync(`ssh ${sshUser}@${workerIp} "${workerLaunchCmd}"`, { timeout: 30_000 });
          callbacks.onLog(`Worker ${workerIp} training launched (log: ${workerLogPath}).\n`);
        } catch (err) {
          callbacks.onComplete("failed", undefined, `Failed to start worker ${workerIp}: ${err}`);
          cleanup(containerName, jobId);
          return;
        }
      }

      // Add --hostfile to head node args
      trainArgs.unshift("--hostfile", "/tmp/hostfile.txt");

      // Watchdog: poll worker container status every 30s. If any worker's
      // container exits while head is still running, force cleanup. Without
      // this, a worker crash inside an NCCL collective leaves head hanging
      // forever (because TORCH_NCCL_ASYNC_ERROR_HANDLING=1 disables timeout
      // to tolerate 26B ZeRO-3 all-gather), leaking containers and pinning GPU.
      const sshUserForWatchdog = process.env.SSH_USER || process.env.USER || "daniel";
      const watchdogWorkerIps = clusterNodeIps!.slice(1);
      instance.watchdogTimer = setInterval(() => {
        if (instance.aborted) return;
        for (const ip of watchdogWorkerIps) {
          try {
            const status = execSync(
              `ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${sshUserForWatchdog}@${ip} "docker inspect -f '{{.State.Status}}' ${containerName} 2>/dev/null || echo missing"`,
              { timeout: 10_000 }
            ).toString().trim();
            if (status !== "running") {
              callbacks.onLog(`\n[watchdog] Worker ${ip} container is ${status}. Force-stopping job.\n`);
              if (instance.watchdogTimer) clearInterval(instance.watchdogTimer);
              instance.aborted = true;
              cleanup(containerName, jobId);
              callbacks.onComplete("failed", undefined, `Worker ${ip} container exited (${status})`);
              return;
            }
          } catch {
            // Transient SSH/network failure — skip this tick
          }
        }
      }, 30_000);
    }

    callbacks.onLog(`\nLaunching training: ${launchPath}\n`);
    callbacks.onLog(`Args: ${trainArgs.join(" ")}\n\n`);

    // Exec the launch script inside the container (head node for multi-node)
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
 *
 * If the job is tracked locally, kill its exec process and clean up via the
 * normal path. Otherwise fall back to force cleanup using state persisted to
 * NFS — this handles the case where the agent restarted and lost the in-memory
 * record of the job (e.g. multi-node workers need to be cleaned even when the
 * head agent doesn't remember them).
 */
export function stopFinetuneJob(jobId: string): boolean {
  const instance = running.get(jobId);

  if (instance) {
    instance.aborted = true;
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

  // Not tracked — force cleanup using on-disk state
  return forceCleanupJob(jobId);
}

/** Clean up Docker container and remove from tracking. */
function cleanup(containerName: string, jobId: string) {
  const instance = running.get(jobId);

  // Stop the watchdog first so it doesn't re-trigger mid-cleanup
  if (instance?.watchdogTimer) {
    clearInterval(instance.watchdogTimer);
  }

  // Stop worker containers on remote nodes
  if (instance?.workerIps) {
    const sshUser = process.env.SSH_USER || process.env.USER || "daniel";
    for (const ip of instance.workerIps) {
      try {
        execSync(`ssh -o StrictHostKeyChecking=no ${sshUser}@${ip} "docker rm -f ${containerName}"`, { timeout: 15_000, stdio: "ignore" });
      } catch { /* ignore */ }
    }
  }

  // Stop head container
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
  const mergeScript = `${WORKSPACE}/src/github/dgx-manager-fine-tune-recipes/scripts/merge.py`;

  // Translate NFS paths to container paths
  const containerAdapterPath = toContainerPath(adapterPath);
  const containerOutputDir = toContainerPath(mergedOutputDir);

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
      "-e", `HF_HOME=${WORKSPACE}/models`,
      "-e", "HF_HUB_OFFLINE=0",
    ];
    if (process.env.HF_TOKEN) {
      dockerArgs.push("-e", `HF_TOKEN=${process.env.HF_TOKEN}`);
    }
    dockerArgs.push(
      "-v", `${SHARED_STORAGE}:${WORKSPACE}`,
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
