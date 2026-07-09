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
  type AccuracyConfig,
} from "../benchmarks/presets.js";
import { buildBenchyArgs } from "../benchmarks/args.js";
import { buildToolEvalArgs } from "../benchmarks/tool-eval-args.js";
import { deploymentEndpointUrl, resolveServedModelName } from "../benchmarks/endpoint.js";
import {
  runBenchmark,
  runToolEval,
  runAccuracy,
  cancelBenchmark,
} from "../benchmarks/orchestrator.js";

const SHARED_STORAGE =
  process.env.SHARED_STORAGE_PATH || "/mnt/tank";

export const benchmarksRouter = express.Router();

/**
 * @openapi
 * /api/benchmarks/presets:
 *   get:
 *     tags: [Benchmarks]
 *     summary: List available benchmark presets
 *     description: >
 *       Returns all built-in benchmark presets with their `id`, `kind`
 *       (`throughput` or `tool-eval`), and default configuration. Pass a preset's
 *       `id` as `presetId` in POST /api/benchmarks to run it. Throughput presets
 *       use llama-benchy; tool-eval presets use tool-eval-bench.
 *     responses:
 *       '200':
 *         description: Array of benchmark preset objects
 */
benchmarksRouter.get("/presets", (_req, res) => {
  res.json(BENCHMARK_PRESETS);
});

/**
 * @openapi
 * /api/benchmarks:
 *   get:
 *     tags: [Benchmarks]
 *     summary: List benchmark runs (optionally filtered by deployment)
 *     description: >
 *       Returns BenchmarkRun records ordered by creation date descending. Each record
 *       includes the linked Deployment (with Node and Model). Pass `?deploymentId=X`
 *       to filter to runs for a specific deployment. Results include both throughput
 *       metrics (`meanTps`, `meanTtfrMs`) and tool-eval scores (`toolEvalScore`,
 *       `toolEvalRating`, etc.) depending on the run's `kind`.
 *     parameters:
 *       - in: query
 *         name: deploymentId
 *         required: false
 *         schema: { type: string }
 *         description: Filter runs to a specific deployment ID
 *     responses:
 *       '200':
 *         description: Array of benchmark run objects with deployment included
 */
benchmarksRouter.get("/", async (req, res) => {
  const { deploymentId } = req.query as { deploymentId?: string };
  const runs = await prisma.benchmarkRun.findMany({
    where: deploymentId ? { deploymentId } : undefined,
    orderBy: { createdAt: "desc" },
    include: { deployment: { include: { node: true, model: true } } },
  });
  res.json(runs);
});

/**
 * @openapi
 * /api/benchmarks/{id}/logs:
 *   get:
 *     tags: [Benchmarks]
 *     summary: Return captured stdout/stderr of a benchmark run
 *     description: >
 *       Reads the benchmark log file from `$SHARED_STORAGE/logs/benchmarks/{id}.log`
 *       and returns it as `text/plain`. The server appends every stdout/stderr line
 *       from the llama-benchy or tool-eval-bench process to this file in real-time.
 *       Returns an empty 200 body if the file doesn't exist yet. During a live run,
 *       individual log lines also flow over SSE as `benchmark:log` events.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: Log content as text/plain
 *         content:
 *           text/plain:
 *             schema: { type: string }
 */
benchmarksRouter.get("/:id/logs", (req, res) => {
  const logPath = join(SHARED_STORAGE, "logs", "benchmarks", `${req.params.id}.log`);
  if (!existsSync(logPath)) return res.type("text/plain").send("");
  try {
    res.type("text/plain").send(readFileSync(logPath, "utf-8"));
  } catch {
    res.type("text/plain").send("");
  }
});

/**
 * @openapi
 * /api/benchmarks/{id}:
 *   get:
 *     tags: [Benchmarks]
 *     summary: Get a single benchmark run with full results
 *     description: >
 *       Returns the BenchmarkRun record with all nested result data: `results`
 *       (throughput per-prompt stats), `toolEvalCategories` (tool-eval per-category
 *       breakdown), and the linked Deployment (with Node and Model). Use this for
 *       the benchmark detail page.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: Benchmark run with results and categories included
 *       '404':
 *         description: Run not found
 */
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

/**
 * @openapi
 * /api/benchmarks/{id}:
 *   delete:
 *     tags: [Benchmarks]
 *     summary: Delete a benchmark run record
 *     description: >
 *       Permanently removes the BenchmarkRun row and all nested result rows
 *       (BenchmarkResult and ToolEvalCategory). Log files on shared storage are
 *       not deleted. Broadcasts `benchmark:deleted` over SSE. Returns 204 on
 *       success (no body).
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '204':
 *         description: Deleted (no body)
 *       '404':
 *         description: Run not found
 */
benchmarksRouter.delete("/:id", async (req, res) => {
  const existing = await prisma.benchmarkRun.findUnique({
    where: { id: req.params.id },
  });
  if (!existing) return res.status(404).end();
  await prisma.benchmarkRun.delete({ where: { id: req.params.id } });
  sseBroadcast({ type: "benchmark:deleted", payload: { id: req.params.id } });
  res.status(204).end();
});

/**
 * @openapi
 * /api/benchmarks/{id}/cancel:
 *   post:
 *     tags: [Benchmarks]
 *     summary: Cancel a running benchmark
 *     description: >
 *       Sends SIGTERM to the llama-benchy or tool-eval-bench child process and
 *       immediately marks the run as `canceled` in the DB. Broadcasts
 *       `benchmark:status` over SSE. After cancellation the run is a terminal
 *       record that can be deleted. Returns the updated BenchmarkRun record.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: Updated BenchmarkRun with status canceled
 *       '404':
 *         description: Run not found
 */
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

/**
 * @openapi
 * /api/benchmarks:
 *   post:
 *     tags: [Benchmarks]
 *     summary: Start a new benchmark run against a running deployment
 *     description: >
 *       Creates a BenchmarkRun record and launches the benchmark tool server-side
 *       (no agent involvement). For `kind: throughput` runs, uses llama-benchy via
 *       `uvx`; for `kind: tool-eval` runs, uses tool-eval-bench via `uvx`. The
 *       benchmark process streams output via SSE (`benchmark:log`) and writes
 *       to `$SHARED_STORAGE/logs/benchmarks/{id}.log`. Results (meanTps,
 *       meanTtfrMs for throughput; toolEvalScore/toolEvalRating for tool-eval) are
 *       stored when the process exits. The deployment must be in `running` status;
 *       only one benchmark can run per deployment at a time. Pass `presetId` from
 *       GET /api/benchmarks/presets for standard configurations, or supply a custom
 *       `config` object for throughput-only runs. Broadcasts `benchmark:created`
 *       and `benchmark:status` over SSE.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required: [deploymentId]
 *             properties:
 *               deploymentId: { type: string, description: "ID of the running Deployment to benchmark." }
 *               presetId: { type: string, description: "Preset id from GET /api/benchmarks/presets. Mutually exclusive with config." }
 *               config: { type: object, description: "Custom throughput config object (concurrency, maxTokens, numPrompts, etc.). Use presetId for tool-eval runs." }
 *     responses:
 *       '201':
 *         description: Created BenchmarkRun record (status=pending, then quickly transitions to running)
 *       '400':
 *         description: Missing deploymentId, presetId/config, or unknown presetId
 *       '404':
 *         description: Deployment not found
 *       '409':
 *         description: Deployment not running, or a benchmark is already in progress for this deployment
 */
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

  let kind: "throughput" | "tool-eval" | "accuracy" = "throughput";
  let config: BenchmarkConfig | ToolEvalConfig | AccuracyConfig;
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
  // Ask the running endpoint what it actually serves — vLLM uses the recipe's
  // --served-model-name, which may differ from displayName/model.name. Falls
  // back to those if the endpoint isn't reachable yet.
  const servedModelName = await resolveServedModelName(
    endpointUrl,
    deployment.displayName ?? deployment.model.name,
  );

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

  if (kind === "accuracy") {
    runAccuracy({
      runId: run.id,
      config: config as AccuracyConfig,
      endpointV1Url: endpointUrl,
      servedModel: servedModelName,
      outputDir,
      onLog,
    })
      .then(async (r) => {
        const current = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
        if (current?.status === "canceled") return;
        if (r.exitCode === 0 && r.summary) {
          await prisma.benchmarkRun.update({
            where: { id: run.id },
            data: {
              status: "completed",
              completedAt: new Date(),
              rawOutput: r.rawOutput,
              accuracyScore: r.summary.primaryScore,
              accuracyMetrics: JSON.stringify(r.summary.metrics),
            },
          });
          const final = await prisma.benchmarkRun.findUnique({ where: { id: run.id } });
          sseBroadcast({ type: "benchmark:status", payload: final });
        } else {
          await finishFailed(`lm-eval exited with code ${r.exitCode}`);
        }
      })
      .catch((e) => finishFailed((e as Error).message));
  } else if (kind === "tool-eval") {
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
