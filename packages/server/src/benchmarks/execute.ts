import { join } from "node:path";
import { appendFileSync, mkdirSync } from "node:fs";
import { prisma } from "../prisma.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";
import { runBenchmark, runToolEval, runAccuracy } from "./orchestrator.js";
import { type CapInvoker } from "./remote-runner.js";
import { buildBenchyArgs } from "./args.js";
import { buildToolEvalArgs } from "./tool-eval-args.js";
import type { BenchmarkConfig, ToolEvalConfig, AccuracyConfig } from "./presets.js";

/** Deterministic per-run IO — identical in the route and at boot reattach. */
export function benchmarkIo(runId: string) {
  const outputDir = join(SHARED_STORAGE, "benchmarks", runId);
  const logDir = join(SHARED_STORAGE, "logs", "benchmarks");
  mkdirSync(logDir, { recursive: true, mode: 0o777 });
  const logPath = join(logDir, `${runId}.log`);
  const onLog = (line: string) => {
    try { appendFileSync(logPath, line + "\n", { mode: 0o666 }); } catch { /* keep streaming */ }
    sseBroadcast({ type: "benchmark:log", payload: { runId, log: line } });
  };
  const onOffset = (offset: number) => {
    void prisma.benchmarkRun.update({ where: { id: runId }, data: { logOffset: offset } }).catch(() => {});
  };
  return { outputDir, resultPath: join(outputDir, "result.json"), onLog, onOffset };
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
  if (r.exitCode === 0 && r.summary) {
    await prisma.benchmarkRun.update({
      where: { id: runId },
      data: {
        status: "completed",
        completedAt: new Date(),
        rawOutput: r.rawOutput,
        accuracyScore: r.summary.primaryScore,
        accuracyMetrics: JSON.stringify(r.summary.metrics),
      },
    });
    const final = await prisma.benchmarkRun.findUnique({ where: { id: runId } });
    sseBroadcast({ type: "benchmark:status", payload: final });
  } else if (r.exitCode === 0 && r.error) {
    // Process succeeded but results couldn't be parsed — surface the real
    // reason (e.g. a missing primary metric) and keep the raw JSON so the
    // detail page can show it.
    await prisma.benchmarkRun.update({
      where: { id: runId },
      data: { status: "failed", completedAt: new Date(), error: r.error, rawOutput: r.rawOutput },
    });
    sseBroadcast({
      type: "benchmark:status",
      payload: { id: runId, status: "failed", error: r.error },
    });
  } else {
    await finishFailed(runId, `lm-eval exited with code ${r.exitCode}`);
  }
}

export async function finalizeToolEval(runId: string, r: Awaited<ReturnType<typeof runToolEval>>): Promise<void> {
  const current = await prisma.benchmarkRun.findUnique({ where: { id: runId } });
  if (current?.status === "canceled") return;
  if (r.exitCode === 0 && r.summary) {
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
    await finishFailed(runId, `tool-eval-bench exited with code ${r.exitCode}`);
  }
}

export async function finalizeThroughput(runId: string, r: Awaited<ReturnType<typeof runBenchmark>>): Promise<void> {
  // SIGTERM from cancel exits the child non-zero; if the row was already
  // flipped to "canceled" by the cancel route, leave it alone.
  const current = await prisma.benchmarkRun.findUnique({ where: { id: runId } });
  if (current?.status === "canceled") return;
  if (r.exitCode === 0) {
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
    await finishFailed(runId, `llama-benchy exited with code ${r.exitCode}`);
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
