/**
 * FP8 quantization runner for merged fine-tune artifacts.
 *
 * Mirrors mergeLoraAdapter() in runtime/finetune.ts but runs the
 * recipe's quantize_fp8 script (defaults to scripts/quantize_fp8.py)
 * against the merged BF16 dir, producing a sibling merged-fp8/ dir.
 *
 * Single-GB10 single-container — no multi-node coordination needed
 * for 27B-class models (output is ~27 GB, comfortably fits one
 * GB10's 122 GB unified pool with headroom for transient activations).
 */

import { spawn, execSync, type ChildProcess } from "child_process";
import { createWriteStream, mkdirSync } from "fs";
import { dirname, join } from "path";
import { SHARED_STORAGE, WORKSPACE, toContainerPath } from "../env.js";

export interface QuantizeCallbacks {
  onLog: (line: string) => void;
  onProgress: (phase: "loading" | "quantizing" | "saving", progress: number) => void;
  onComplete: (status: "completed" | "failed", outputPath?: string, error?: string) => void;
}

interface QuantizeProgressUpdate {
  phase: "loading" | "quantizing" | "saving";
  progress: number;
}

/**
 * Pure helper: classify a stdout line from quantize_fp8.py into a
 * progress update, or null if the line isn't a recognized milestone.
 * Exported so unit tests can exercise it without a docker container.
 */
export function detectQuantizeProgress(line: string): QuantizeProgressUpdate | null {
  const l = line.toLowerCase();
  if (l.includes("[quantize_fp8] loading model")) return { phase: "loading", progress: 0.1 };
  if (l.includes("[quantize_fp8] model loaded")) return { phase: "loading", progress: 0.3 };
  if (l.includes("[quantize_fp8] applying fp8_dynamic")) return { phase: "quantizing", progress: 0.5 };
  if (l.includes("[quantize_fp8] quantization complete")) return { phase: "quantizing", progress: 0.8 };
  if (l.includes("[quantize_fp8] saving fp8 model")) return { phase: "saving", progress: 0.85 };
  if (l.includes("[quantize_fp8] fp8 model saved")) return { phase: "saving", progress: 0.95 };
  if (l.includes("[quantize_fp8] ok")) return { phase: "saving", progress: 1.0 };
  return null;
}

/**
 * Quantize a merged BF16 model to FP8_DYNAMIC W8A8 using the recipe's
 * quantize_fp8 script. Writes to {mergedDir}/../merged-fp8/.
 */
export async function quantizeMergedToFp8(
  jobId: string,
  mergedPath: string,
  quantizedOutputDir: string,
  callbacks: QuantizeCallbacks,
  quantizeScriptRelative: string,
): Promise<void> {
  const containerName = `dgx-quantize-${jobId.slice(0, 12)}`;
  const containerImage = "nvcr.io/nvidia/pytorch:25.11-py3";
  const quantizeScript = `${WORKSPACE}/src/github/dgx-manager-fine-tune-recipes/${quantizeScriptRelative}`;

  const containerMergedPath = toContainerPath(mergedPath);
  const containerOutputDir = toContainerPath(quantizedOutputDir);

  const quantizeLogPath = join(dirname(quantizedOutputDir), "quantize.log");
  try { mkdirSync(dirname(quantizeLogPath), { recursive: true }); } catch { /* */ }
  const logStream = createWriteStream(quantizeLogPath, { flags: "a" });
  const tee = (s: string) => { callbacks.onLog(s); try { logStream.write(s); } catch { /* */ } };

  try {
    try { execSync(`docker rm -f ${containerName}`, { timeout: 15_000, stdio: "ignore" }); } catch { /* */ }

    tee(`[agent] Starting quantize container (script=${quantizeScriptRelative})\n`);
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
      "-v", `${SHARED_STORAGE}:${WORKSPACE}`,
      "--entrypoint", "sleep",
      containerImage,
      "infinity",
    ];

    execSync(`docker ${dockerArgs.join(" ")}`, { timeout: 120_000 });

    tee("[agent] Installing quantization dependencies...\n");
    try {
      execSync(
        `docker exec ${containerName} pip install -q llmcompressor transformers accelerate safetensors`,
        { timeout: 600_000, stdio: "ignore" },
      );
    } catch (err) {
      tee(`[agent] Failed to install quantization deps: ${err}\n`);
      callbacks.onComplete("failed", undefined, `Failed to install quantization deps: ${err}`);
      try { logStream.end(); } catch { /* */ }
      try { execSync(`docker rm -f ${containerName}`, { timeout: 15_000, stdio: "ignore" }); } catch { /* */ }
      return;
    }
    tee("[agent] Dependencies installed.\n");

    tee(`[agent] Running quantize: ${containerMergedPath} -> ${containerOutputDir}\n`);
    tee(`[agent] Script: ${quantizeScript}\n\n`);

    const quantProc: ChildProcess = spawn("docker", [
      "exec", containerName,
      "python", quantizeScript,
      "--model-dir", containerMergedPath,
      "--output-dir", containerOutputDir,
    ], { stdio: ["ignore", "pipe", "pipe"] });

    quantProc.stdout?.on("data", (data: Buffer) => {
      const text = data.toString();
      tee(text);
      for (const line of text.split("\n")) {
        const update = detectQuantizeProgress(line);
        if (update) callbacks.onProgress(update.phase, update.progress);
      }
    });
    quantProc.stderr?.on("data", (data: Buffer) => tee(data.toString()));

    await new Promise<void>((resolve) => {
      quantProc.on("exit", (code) => {
        tee(`\n[agent] Quantize process exited with code ${code}\n`);
        try {
          execSync(`docker exec ${containerName} chmod -R a+rw ${containerOutputDir}`, { timeout: 30_000, stdio: "ignore" });
        } catch { /* best effort */ }

        if (code === 0) {
          tee(`[agent] FP8 model saved to ${quantizedOutputDir}\n`);
          callbacks.onComplete("completed", quantizedOutputDir);
        } else {
          callbacks.onComplete("failed", undefined, `Quantize failed with exit code ${code}`);
        }
        try { logStream.end(); } catch { /* */ }
        resolve();
      });
    });
  } finally {
    try { execSync(`docker rm -f ${containerName}`, { timeout: 30_000, stdio: "ignore" }); } catch { /* */ }
  }
}
