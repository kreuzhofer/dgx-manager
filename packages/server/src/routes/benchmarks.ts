import express, { type Request, type Response } from "express";
import { join } from "node:path";
import { prisma } from "../prisma.js";
import { broadcast as sseBroadcast } from "../sse.js";
import {
  BENCHMARK_PRESETS,
  getPreset,
  type BenchmarkConfig,
} from "../benchmarks/presets.js";
import { buildBenchyArgs } from "../benchmarks/args.js";
import { deploymentEndpointUrl } from "../benchmarks/endpoint.js";
import {
  runBenchmark,
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

benchmarksRouter.get("/:id", async (req, res) => {
  const run = await prisma.benchmarkRun.findUnique({
    where: { id: req.params.id },
    include: {
      results: true,
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

  let config: BenchmarkConfig;
  if (presetId) {
    const preset = getPreset(presetId);
    if (!preset) return res.status(400).json({ error: "unknown presetId" });
    config = preset.config;
  } else {
    config = customConfig!;
  }

  let endpointUrl: string;
  try {
    endpointUrl = deploymentEndpointUrl(deployment);
  } catch (e) {
    return res.status(409).json({ error: (e as Error).message });
  }
  const servedModelName = deployment.displayName ?? deployment.model.name;

  const run = await prisma.benchmarkRun.create({
    data: {
      deploymentId,
      presetId: presetId ?? null,
      modelName: deployment.model.name,
      endpointUrl,
      servedModelName,
      config: JSON.stringify(config),
      status: "pending",
    },
  });
  sseBroadcast({ type: "benchmark:created", payload: run });

  const outputDir = join(SHARED_STORAGE, "benchmarks", run.id);
  const args = buildBenchyArgs(config, {
    baseUrl: endpointUrl,
    modelName: servedModelName,
    outputPath: join(outputDir, "result.json"),
  });

  // Move to "running" immediately so the dashboard reflects state.
  await prisma.benchmarkRun.update({
    where: { id: run.id },
    data: { status: "running", startedAt: new Date() },
  });
  sseBroadcast({
    type: "benchmark:status",
    payload: { id: run.id, status: "running" },
  });

  runBenchmark({
    runId: run.id,
    args,
    outputDir,
    onLog: (line) => {
      sseBroadcast({
        type: "benchmark:log",
        payload: { runId: run.id, log: line },
      });
    },
  })
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
        await prisma.benchmarkRun.update({
          where: { id: run.id },
          data: {
            status: "failed",
            completedAt: new Date(),
            error: `llama-benchy exited with code ${r.exitCode}`,
          },
        });
      }
      const final = await prisma.benchmarkRun.findUnique({
        where: { id: run.id },
        include: { results: true },
      });
      sseBroadcast({ type: "benchmark:status", payload: final });
    })
    .catch(async (e) => {
      await prisma.benchmarkRun.update({
        where: { id: run.id },
        data: {
          status: "failed",
          completedAt: new Date(),
          error: (e as Error).message,
        },
      });
      sseBroadcast({
        type: "benchmark:status",
        payload: { id: run.id, status: "failed", error: (e as Error).message },
      });
    });

  res.status(201).json(run);
});
