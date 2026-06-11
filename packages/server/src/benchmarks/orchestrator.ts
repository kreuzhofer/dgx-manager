import { spawn, type ChildProcess } from "node:child_process";
import { readFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import {
  parseBenchyResults,
  summarizeResults,
  type BenchmarkResultInput,
} from "./parser.js";
import {
  parseToolEvalResults,
  type ToolEvalSummary,
} from "./tool-eval-parser.js";

const LLAMA_BENCHY_SPEC =
  process.env.LLAMA_BENCHY_VERSION
    ? `llama-benchy==${process.env.LLAMA_BENCHY_VERSION}`
    : "llama-benchy";

// Pinned upstream commit (tool-eval-bench v2.0.6). Overridable for upgrades.
const TOOL_EVAL_SPEC =
  process.env.TOOL_EVAL_BENCH_REF ||
  "git+https://github.com/SeraphimSerapis/tool-eval-bench.git@c3868bff099592c9a1045de2c9a3dc24abebb7fb";

// In-memory registry of in-flight runs. Lost on restart — see Task 9 for the
// boot-time reconciliation that marks any orphaned "running" rows as failed.
const ACTIVE: Map<string, ChildProcess> = new Map();

type SpawnTrackedOpts = {
  runId: string;
  command: string;   // executable, e.g. "uvx"
  args: string[];    // full argv for the executable
  outputDir: string; // mkdir'd before spawn; must contain the result.json path
  onLog: (line: string) => void;
};

type SpawnTrackedResult = { exitCode: number | null; rawOutput: string | null };

// Shared, kind-agnostic plumbing: mkdir the output dir, spawn the process in
// its own group (so we can kill the whole group on cancel), stream stdout and
// stderr line-by-line to onLog, and on exit read result.json verbatim. Parsing
// is left to the caller because it differs per benchmark kind.
function spawnTracked(opts: SpawnTrackedOpts): Promise<SpawnTrackedResult> {
  mkdirSync(opts.outputDir, { recursive: true, mode: 0o777 });

  return new Promise((resolve) => {
    const child = spawn(opts.command, opts.args, {
      stdio: ["ignore", "pipe", "pipe"],
      // detached:true so we can kill the whole process group (uvx may spawn a
      // python subprocess) via process.kill(-pid).
      detached: true,
      // PYTHONUNBUFFERED forces line buffering on the child's piped stdout so
      // our onLog fires live instead of only at process exit.
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    ACTIVE.set(opts.runId, child);

    const pump = (b: Buffer) => {
      for (const line of b.toString().split("\n")) {
        if (line) opts.onLog(line);
      }
    };
    child.stdout?.on("data", pump);
    child.stderr?.on("data", pump);

    child.on("close", (code) => {
      ACTIVE.delete(opts.runId);
      const resultPath = join(opts.outputDir, "result.json");
      let rawOutput: string | null = null;
      if (code === 0 && existsSync(resultPath)) {
        try {
          rawOutput = readFileSync(resultPath, "utf-8");
        } catch (e) {
          opts.onLog(`[read] ${(e as Error).message}`);
        }
      }
      resolve({ exitCode: code, rawOutput });
    });
  });
}

export type RunBenchmarkOpts = {
  runId: string;
  args: string[];     // llama-benchy argv (from buildBenchyArgs)
  outputDir: string;
  onLog: (line: string) => void;
};

export type RunBenchmarkResult = {
  exitCode: number | null;
  results: BenchmarkResultInput[];
  summary: { meanTps: number | null; meanTtfrMs: number | null };
  rawOutput: string | null;
};

export async function runBenchmark(opts: RunBenchmarkOpts): Promise<RunBenchmarkResult> {
  const { exitCode, rawOutput } = await spawnTracked({
    runId: opts.runId,
    command: "uvx",
    args: ["--from", LLAMA_BENCHY_SPEC, "llama-benchy", ...opts.args],
    outputDir: opts.outputDir,
    onLog: opts.onLog,
  });

  let results: BenchmarkResultInput[] = [];
  if (exitCode === 0 && rawOutput !== null) {
    try {
      results = parseBenchyResults(rawOutput);
    } catch (e) {
      opts.onLog(`[parser] ${(e as Error).message}`);
    }
  }
  return { exitCode, results, summary: summarizeResults(results), rawOutput };
}

export type RunToolEvalOpts = {
  runId: string;
  args: string[];     // tool-eval-bench argv (from buildToolEvalArgs)
  outputDir: string;
  onLog: (line: string) => void;
};

export type RunToolEvalResult = {
  exitCode: number | null;
  summary: ToolEvalSummary | null;
  rawOutput: string | null;
};

export async function runToolEval(opts: RunToolEvalOpts): Promise<RunToolEvalResult> {
  const { exitCode, rawOutput } = await spawnTracked({
    runId: opts.runId,
    command: "uvx",
    args: ["--from", TOOL_EVAL_SPEC, "tool-eval-bench", ...opts.args],
    outputDir: opts.outputDir,
    onLog: opts.onLog,
  });

  let summary: ToolEvalSummary | null = null;
  if (exitCode === 0 && rawOutput !== null) {
    try {
      summary = parseToolEvalResults(rawOutput);
    } catch (e) {
      opts.onLog(`[parser] ${(e as Error).message}`);
    }
  }
  return { exitCode, summary, rawOutput };
}

export function cancelBenchmark(runId: string): boolean {
  const child = ACTIVE.get(runId);
  if (!child) return false;
  // Signal the whole process group — we spawned with detached:true, so the CLI
  // and its python subprocess are in their own group. SIGTERM gives the tool a
  // chance to flush partial results before exit.
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
