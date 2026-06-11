import express, { type Request, type Response } from "express";
import { join } from "node:path";
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { prisma } from "../prisma.js";
import { broadcast as sseBroadcast } from "../sse.js";
import {
  BENCHMARK_PRESETS,
  getPreset,
  type BenchmarkConfig,
  type ToolEvalConfig,
} from "../benchmarks/presets.js";
import { buildBenchyArgs } from "../benchmarks/args.js";
import { buildToolEvalArgs } from "../benchmarks/tool-eval-args.js";
import { deploymentEndpointUrl } from "../benchmarks/endpoint.js";
import {
  runBenchmark,
  runToolEval,
  cancelBenchmark,
} from "../benchmarks/orchestrator.js";

const SHARED_STORAGE =
  process.env.SHARED_STORAGE_PATH || "/mnt/tank";

export const benchmarksRouter = express.Router();

benchmarksRouter.get("/presets", (_req, res) => {
  res.json(BENCHMARK_PRESETS);
});

benchmarksRouter.get("/", async (req, res) => {
  const { deploymentId } = req.query as { deploymentId?: string };
  const runs = await prisma.benchmarkRun.findMany({
    where: deploymentId ? { deploymentId } : undefined,
    orderBy: { createdAt: "desc" },
    include: { deployment: { include: { node: true, model: true } } },
  });
  res.json(runs);
});

// Returns the full captured stdout/stderr of a run as text/plain. Empty body
// (with 200) if the log file doesn't exist yet — the dashboard treats that
// as "no log to show", same as the deployments-log endpoint convention.
benchmarksRouter.get("/:id/logs", (req, res) => {
  const logPath = join(SHARED_STORAGE, "logs", "benchmarks", `${req.params.id}.log`);
  if (!existsSync(logPath)) return res.type("text/plain").send("");
  try {
    res.type("text/plain").send(readFileSync(logPath, "utf-8"));
  } catch {
    res.type("text/plain").send("");
  }
});

benchmarksRouter.get("/:id", async (req, res) => {
  const run = await prisma.benchmarkRun.findUnique({
    where: { id: req.params.id },
    include: {
      results: true,
      toolEvalCategories: true,
      deployment: { include: { node: true, model: true } },
    },
  });
  if (!run) return res.status(404).json({ error: "not found" });
  res.json(run);
});

benchmarksRouter.delete("/:id", async (req, res) => {
  const existing = await prisma.benchmarkRun.findUnique({
    where: { id: req.params.id },
  });
  if (!existing) return res.status(404).end();
  await prisma.benchmarkRun.delete({ where: { id: req.params.id } });
  sseBroadcast({ type: "benchmark:deleted", payload: { id: req.params.id } });
  res.status(204).end();
});

benchmarksRouter.post("/:id/cancel", async (req, res) => {
  const run = await prisma.benchmarkRun.findUnique({
    where: { id: req.params.id },
  });
  if (!run) return res.status(404).end();
  cancelBenchmark(run.id);
  const updated = await prisma.benchmarkRun.update({
    where: { id: run.id },
    data: { status: "canceled", completedAt: new Date() },
  });
  sseBroadcast({ type: "benchmark:status", payload: updated });
  res.json(updated);
});

type StartBody = {
  deploymentId?: string;
  presetId?: string;
  config?: BenchmarkConfig;
};

benchmarksRouter.post("/", async (req: Request, res: Response) => {
  const { deploymentId, presetId, config: customConfig } = req.body as StartBody;

  if (!deploymentId) {
    return res.status(400).json({ error: "deploymentId is required" });
  }
  if (!presetId && !customConfig) {
    return res.status(400).json({ error: "presetId or config is required" });
  }

  const deployment = await prisma.deployment.findUnique({
    where: { id: deploymentId },
    include: { node: true, model: true },
  });
  if (!deployment) {
    return res.status(404).json({ error: "deployment not found" });
  }
  if (deployment.status !== "running") {
    return res
      .status(409)
      .json({ error: "deployment is not running" });
  }

  const inflight = await prisma.benchmarkRun.findFirst({
    where: {
      deploymentId,
      status: { in: ["pending", "running"] },
    },
  });
  if (inflight) {
    return res
      .status(409)
      .json({ error: "a benchmark is already running for this deployment" });
  }

  let kind: "throughput" | "tool-eval" = "throughput";
  let config: BenchmarkConfig | ToolEvalConfig;
  if (presetId) {
    const preset = getPreset(presetId);
    if (!preset) return res.status(400).json({ error: "unknown presetId" });
    kind = preset.kind;
    config = preset.config;
  } else {
    // Custom config is throughput-only; tool-eval runs must use a preset.
    config = customConfig!;
  }

  let endpointUrl: string;
  try {
    // llama-benchy follows the OpenAI client convention where `--base-url`
    // already includes the `/v1` segment (it appends `/chat/completions`
    // directly to it). Without the `/v1`, vLLM 404s the warmup request.
    endpointUrl = deploymentEndpointUrl(deployment) + "/v1";
  } catch (e) {
    return res.status(409).json({ error: (e as Error).message });
  }
  const servedModelName = deployment.displayName ?? deployment.model.name;

  const run = await prisma.benchmarkRun.create({
    data: {
      deploymentId,
      presetId: presetId ?? null,
      kind,
      modelName: deployment.model.name,
      endpointUrl,
      servedModelName,
      config: JSON.stringify(config),
      status: "pending",
    },
  });
  sseBroadcast({ type: "benchmark:created", payload: run });

  const outputDir = join(SHARED_STORAGE, "benchmarks", run.id);

  // Move to "running" immediately so the dashboard reflects state.
  await prisma.benchmarkRun.update({
    where: { id: run.id },
    data: { status: "running", startedAt: new Date() },
  });
  sseBroadcast({
    type: "benchmark:status",
    payload: { id: run.id, status: "running", deploymentId: run.deploymentId },
  });

  // Persist every log line to disk so the detail page can show the full log
  // for a completed run (SSE only delivers events while the page is mounted).
  const logDir = join(SHARED_STORAGE, "logs", "benchmarks");
  mkdirSync(logDir, { recursive: true, mode: 0o777 });
  const logPath = join(logDir, `${run.id}.log`);
  const onLog = (line: string) => {
    try {
      appendFileSync(logPath, line + "\n", { mode: 0o666 });
    } catch {
      // Disk-full or perms — keep streaming via SSE even if persistence fails.
    }
    sseBroadcast({ type: "benchmark:log", payload: { runId: run.id, log: line } });
  };
  const resultPath = join(outputDir, "result.json");

  const finishFailed = async (message: string) => {
    await prisma.benchmarkRun.update({
      where: { id: run.id },
      data: { status: "failed", completedAt: new Date(), error: message },
    });
    sseBroadcast({
      type: "benchmark:status",
      payload: { id: run.id, status: "failed", error: message },
    });
  };

  if (kind === "tool-eval") {
    const args = buildToolEvalArgs(config as ToolEvalConfig, {
      baseUrl: endpointUrl,
      modelName: servedModelName,
      outputPath: resultPath,
    });
    runToolEval({ runId: run.id, args, outputDir, onLog })
      .then(async (r) => {
        const current = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
        if (current?.status === "canceled") return;
        if (r.exitCode === 0 && r.summary) {
          const s = r.summary;
          await prisma.benchmarkRun.update({
            where: { id: run.id },
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
            where: { id: run.id },
            include: { toolEvalCategories: true },
          });
          sseBroadcast({ type: "benchmark:status", payload: final });
        } else {
          await finishFailed(`tool-eval-bench exited with code ${r.exitCode}`);
        }
      })
      .catch((e) => finishFailed((e as Error).message));
  } else {
    const args = buildBenchyArgs(config as BenchmarkConfig, {
      baseUrl: endpointUrl,
      modelName: servedModelName,
      outputPath: resultPath,
    });
    runBenchmark({ runId: run.id, args, outputDir, onLog })
      .then(async (r) => {
        // SIGTERM from cancel exits the child non-zero; if the row was already
        // flipped to "canceled" by the cancel route, leave it alone.
        const current = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
        if (current?.status === "canceled") return;
        if (r.exitCode === 0) {
          await prisma.benchmarkRun.update({
            where: { id: run.id },
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
          await finishFailed(`llama-benchy exited with code ${r.exitCode}`);
          return;
        }
        const final = await prisma.benchmarkRun.findUnique({
          where: { id: run.id },
          include: { results: true },
        });
        sseBroadcast({ type: "benchmark:status", payload: final });
      })
      .catch((e) => finishFailed((e as Error).message));
  }

  res.status(201).json(run);
});
