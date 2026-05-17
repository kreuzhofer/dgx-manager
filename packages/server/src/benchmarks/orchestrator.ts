import { spawn, type ChildProcess } from "node:child_process";
import { readFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import {
  parseBenchyResults,
  summarizeResults,
  type BenchmarkResultInput,
} from "./parser.js";

const LLAMA_BENCHY_SPEC =
  process.env.LLAMA_BENCHY_VERSION
    ? `llama-benchy==${process.env.LLAMA_BENCHY_VERSION}`
    : "llama-benchy";

// In-memory registry of in-flight runs. Lost on restart — see Task 9 for the
// boot-time reconciliation that marks any orphaned "running" rows as failed.
const ACTIVE: Map<string, ChildProcess> = new Map();

export type RunBenchmarkOpts = {
  runId: string;
  args: string[];        // llama-benchy argv (from buildBenchyArgs)
  outputDir: string;     // host path; must contain the --save-result path the args point at
  onLog: (line: string) => void;
};

export type RunBenchmarkResult = {
  exitCode: number | null;
  results: BenchmarkResultInput[];
  summary: { meanTps: number | null; meanTtfrMs: number | null };
  rawOutput: string | null;
};

export function runBenchmark(opts: RunBenchmarkOpts): Promise<RunBenchmarkResult> {
  mkdirSync(opts.outputDir, { recursive: true, mode: 0o777 });

  const argv = ["--from", LLAMA_BENCHY_SPEC, "llama-benchy", ...opts.args];

  return new Promise((resolve) => {
    const child = spawn("uvx", argv, {
      stdio: ["ignore", "pipe", "pipe"],
      // detached:true so we can kill the whole process group (uvx may
      // spawn a python subprocess) via process.kill(-pid).
      detached: true,
      // PYTHONUNBUFFERED is required: when llama-benchy's stdout is piped
      // (not a tty), Python switches to block buffering and our onLog
      // never fires until process exit. Forcing unbuffered IO makes the
      // live log stream as the benchmark runs.
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    ACTIVE.set(opts.runId, child);

    child.stdout?.on("data", (b: Buffer) => {
      for (const line of b.toString().split("\n")) {
        if (line) opts.onLog(line);
      }
    });
    child.stderr?.on("data", (b: Buffer) => {
      for (const line of b.toString().split("\n")) {
        if (line) opts.onLog(line);
      }
    });

    child.on("close", (code) => {
      ACTIVE.delete(opts.runId);
      const resultPath = join(opts.outputDir, "result.json");
      let results: BenchmarkResultInput[] = [];
      let rawOutput: string | null = null;
      if (code === 0 && existsSync(resultPath)) {
        try {
          rawOutput = readFileSync(resultPath, "utf-8");
          results = parseBenchyResults(rawOutput);
        } catch (e) {
          opts.onLog(`[parser] ${(e as Error).message}`);
        }
      }
      resolve({
        exitCode: code,
        results,
        summary: summarizeResults(results),
        rawOutput,
      });
    });
  });
}

export function cancelBenchmark(runId: string): boolean {
  const child = ACTIVE.get(runId);
  if (!child) return false;
  // Signal the whole process group — we spawned with detached:true, so uvx
  // and its python subprocess are in their own group. SIGTERM gives
  // llama-benchy a chance to flush partial results before exit.
  if (child.pid) {
    try {
      process.kill(-child.pid, "SIGTERM");
    } catch {
      // ESRCH if the group already exited between detect-and-signal
      child.kill("SIGTERM");
    }
  } else {
    child.kill("SIGTERM");
  }
  return true;
}

export function isRunActive(runId: string): boolean {
  return ACTIVE.has(runId);
}
