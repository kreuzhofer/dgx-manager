import { join } from "node:path";
import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import { prisma } from "../prisma.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";
import { runBenchmark, runToolEval, runAccuracy } from "./orchestrator.js";
import { type CapInvoker } from "./remote-runner.js";
import { buildBenchyArgs } from "./args.js";
import { buildToolEvalArgs } from "./tool-eval-args.js";
import { decideFinalize } from "./finalize-outcome.js";
import { countNullCompletions } from "./null-completions.js";
import { explainFailure } from "./proxy-loss.js";
import type { BenchmarkConfig, ToolEvalConfig, AccuracyConfig } from "./presets.js";

/** Where a run's log lives. Pure — safe to call without creating anything. */
export function benchmarkLogPath(runId: string): string {
  return join(SHARED_STORAGE, "logs", "benchmarks", `${runId}.log`);
}

/** Deterministic per-run IO — identical in the route and at boot reattach. */
export function benchmarkIo(runId: string) {
  const outputDir = join(SHARED_STORAGE, "benchmarks", runId);
  const logDir = join(SHARED_STORAGE, "logs", "benchmarks");
  mkdirSync(logDir, { recursive: true, mode: 0o777 });
  const logPath = benchmarkLogPath(runId);
  const onLog = (line: string) => {
    try { appendFileSync(logPath, line + "\n", { mode: 0o666 }); } catch { /* keep streaming */ }
    sseBroadcast({ type: "benchmark:log", payload: { runId, log: line } });
  };
  const onOffset = (offset: number) => {
    void prisma.benchmarkRun.update({ where: { id: runId }, data: { logOffset: offset } }).catch(() => {});
  };
  return { outputDir, resultPath: join(outputDir, "result.json"), onLog, onOffset };
}

/**
 * Count empty completions for a finished run by re-reading its log.
 *
 * Returns null when the log cannot be read, which keeps the "never measured"
 * case distinct from a measured zero — storing 0 here would claim we checked.
 */
function readNullCompletions(runId: string): number | null {
  try {
    return countNullCompletions(readFileSync(benchmarkLogPath(runId), "utf8"));
  } catch {
    return null;
  }
}

export async function finishFailed(runId: string, message: string): Promise<void> {
  const current = await prisma.benchmarkRun.findUnique({ where: { id: runId } });
  if (current?.status === "canceled") return;
  await prisma.benchmarkRun.update({
    where: { id: runId },
    data: { status: "failed", completedAt: new Date(), error: message },
  });
  sseBroadcast({ type: "benchmark:status", payload: { id: runId, status: "failed", error: message } });
}

export async function finalizeAccuracy(runId: string, r: Awaited<ReturnType<typeof runAccuracy>>): Promise<void> {
  const current = await prisma.benchmarkRun.findUnique({ where: { id: runId } });
  if (current?.status === "canceled") return;
  const outcome = decideFinalize({
    tool: "lm-eval", exitCode: r.exitCode, hasSummary: Boolean(r.summary), parseError: r.error,
  });
  if (outcome.kind === "complete" && r.summary) {
    await prisma.benchmarkRun.update({
      where: { id: runId },
      data: {
        status: "completed",
        completedAt: new Date(),
        rawOutput: r.rawOutput,
        accuracyScore: r.summary.primaryScore,
        accuracyMetrics: JSON.stringify(r.summary.metrics),
        // #20 §2. Counted from the LOG rather than accumulated in onLog, so the
        // number survives a manager restart mid-run — boot-reconcile reattaches
        // running benchmarks, and an in-memory tally would silently reset to a
        // partial count that looks like a complete one.
        nullCompletions: readNullCompletions(runId),
      },
    });
    const final = await prisma.benchmarkRun.findUnique({ where: { id: runId } });
    sseBroadcast({ type: "benchmark:status", payload: final });
  } else if (outcome.kind === "fail" && r.exitCode === 0) {
    // Process succeeded but there is nothing worth recording — an unparseable
    // result, or no summary at all. Surface the real reason and keep the raw
    // JSON so the detail page can show it.
    await prisma.benchmarkRun.update({
      where: { id: runId },
      data: { status: "failed", completedAt: new Date(), error: outcome.reason, rawOutput: r.rawOutput },
    });
    sseBroadcast({
      type: "benchmark:status",
      payload: { id: runId, status: "failed", error: outcome.reason },
    });
  } else {
    const bare = outcome.kind === "fail" ? outcome.reason : "lm-eval produced no summary";
    // "lm-eval exited with code 1" is true and useless. When the log shows the
    // job was refused at the manager's reasoning proxy, say THAT instead — the
    // 2026-08-29 run needed a urllib3 traceback and docker events to explain
    // (#22). Falls back to the bare reason when there is no evidence.
    await finishFailed(runId, explainAccuracyFailure(runId, bare, current?.endpointUrl ?? null));
  }
}

/** Upgrade an accuracy failure reason using the run's own log, when it explains it. */
function explainAccuracyFailure(runId: string, reason: string, endpointUrl: string | null): string {
  let log = "";
  try { log = readFileSync(benchmarkLogPath(runId), "utf8"); } catch { return reason; }
  return explainFailure(reason, log, {
    proxyHosts: [process.env.MANAGER_ADVERTISE_HOST, "127.0.0.1"],
    endpointUrl,
  });
}

export async function finalizeToolEval(runId: string, r: Awaited<ReturnType<typeof runToolEval>>): Promise<void> {
  const current = await prisma.benchmarkRun.findUnique({ where: { id: runId } });
  if (current?.status === "canceled") return;
  const outcome = decideFinalize({
    tool: "tool-eval-bench", exitCode: r.exitCode, hasSummary: Boolean(r.summary),
  });
  if (outcome.kind === "complete" && r.summary) {
    const s = r.summary;
    await prisma.benchmarkRun.update({
      where: { id: runId },
      data: {
        status: "completed",
        completedAt: new Date(),
        rawOutput: r.rawOutput,
        toolEvalScore: s.finalScore,
        toolEvalRating: s.rating,
        toolEvalDeployability: s.deployability,
        toolEvalResponsiveness: s.responsiveness,
        toolEvalTotalScenarios: s.totalScenarios,
        toolEvalTotalPoints: s.totalPoints,
        toolEvalMaxPoints: s.maxPoints,
        toolEvalSafetyWarnings: JSON.stringify(s.safetyWarnings),
        toolEvalCategories: { create: s.categories },
      },
    });
    const final = await prisma.benchmarkRun.findUnique({
      where: { id: runId },
      include: { toolEvalCategories: true },
    });
    sseBroadcast({ type: "benchmark:status", payload: final });
  } else {
    await finishFailed(runId, outcome.kind === "fail" ? outcome.reason : "tool-eval-bench produced no summary");
  }
}

export async function finalizeThroughput(runId: string, r: Awaited<ReturnType<typeof runBenchmark>>): Promise<void> {
  // SIGTERM from cancel exits the child non-zero; if the row was already
  // flipped to "canceled" by the cancel route, leave it alone.
  const current = await prisma.benchmarkRun.findUnique({ where: { id: runId } });
  if (current?.status === "canceled") return;
  // `summary` is always an object here (summarizeResults of a possibly-empty
  // array), so it cannot stand in for "produced data" — the row count is what
  // distinguishes a real run from one whose every request failed. See #95.
  const outcome = decideFinalize({
    tool: "llama-benchy",
    exitCode: r.exitCode,
    hasSummary: Boolean(r.summary),
    rowCount: r.results.length,
  });
  if (outcome.kind === "complete") {
    await prisma.benchmarkRun.update({
      where: { id: runId },
      data: {
        status: "completed",
        completedAt: new Date(),
        rawOutput: r.rawOutput,
        meanTps: r.summary.meanTps,
        meanTtfrMs: r.summary.meanTtfrMs,
        results: { create: r.results },
      },
    });
  } else {
    await finishFailed(runId, outcome.reason);
    return;
  }
  const final = await prisma.benchmarkRun.findUnique({
    where: { id: runId },
    include: { results: true },
  });
  sseBroadcast({ type: "benchmark:status", payload: final });
}

export interface RunnableRow {
  id: string; kind: string; config: string;
  endpointUrl: string; servedModelName: string;
  runnerNodeId: string | null; logOffset: number;
}

/**
 * Run a benchmark's execution to completion and persist the result. Shared by the
 * POST route (skipStart=false, fresh) and boot reconciliation (skipStart=true,
 * resuming the already-started systemd job from its persisted logOffset).
 * Fire-and-forget: it wires .then(finalize)/.catch(finishFailed) and returns.
 */
export function executeRun(
  row: RunnableRow,
  invoke: CapInvoker | undefined,
  skipStart: boolean,
): void {
  const { outputDir, resultPath, onLog, onOffset } = benchmarkIo(row.id);
  const config = JSON.parse(row.config);
  const runnerNodeId = row.runnerNodeId ?? undefined;
  const startOffset = skipStart ? row.logOffset : undefined;
  const common = { runId: row.id, outputDir, onLog, onOffset, runnerNodeId, invoke, skipStart, startOffset };

  // Remote tools run with cwd=jobDir on the eval node and write into jobDir/out,
  // which the wrapper's `find` scans. SHARED_STORAGE (the manager's /mnt/tank) is
  // not mounted there, so a remote run must use a job-dir-relative path.
  const remote = runnerNodeId != null;
  const toolOutputPath = remote ? "out/result.json" : resultPath; // benchy / tool-eval
  const toolOutputDir = remote ? "out" : outputDir;                // lm-eval --output_path

  if (row.kind === "accuracy") {
    runAccuracy({ ...common, outputDir: toolOutputDir, config: config as AccuracyConfig, endpointV1Url: row.endpointUrl, servedModel: row.servedModelName })
      .then((r) => finalizeAccuracy(row.id, r))
      .catch((e) => finishFailed(row.id, (e as Error).message));
  } else if (row.kind === "tool-eval") {
    const args = buildToolEvalArgs(config as ToolEvalConfig, { baseUrl: row.endpointUrl, modelName: row.servedModelName, outputPath: toolOutputPath });
    runToolEval({ ...common, args })
      .then((r) => finalizeToolEval(row.id, r))
      .catch((e) => finishFailed(row.id, (e as Error).message));
  } else {
    const args = buildBenchyArgs(config as BenchmarkConfig, { baseUrl: row.endpointUrl, modelName: row.servedModelName, outputPath: toolOutputPath });
    runBenchmark({ ...common, args })
      .then((r) => finalizeThroughput(row.id, r))
      .catch((e) => finishFailed(row.id, (e as Error).message));
  }
}
