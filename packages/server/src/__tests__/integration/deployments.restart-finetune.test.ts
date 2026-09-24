/**
 * The restart route used to always send cmd:deploy with config.recipeFile,
 * but fine-tune deployments don't have a recipeFile in their saved config
 * (it lives on the FineTuneJob row). The agent rejected those restarts
 * with "No recipeFile specified" — the deployment went terminal-failed
 * with no useful surface.
 *
 * These tests pin the fix: restart of a fine-tune deployment must send
 * cmd:finetune:deploy with the same payload shape the original finetune
 * deploy route uses, drawing the recipe + base model from the linked
 * FineTuneJob and the artifactVariant from saved config (default bf16).
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";
import type { TrainingRecipe } from "../../ws/agent-hub.js";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-deploy-restart-ft-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      DATABASE_URL: `file:${DB_PATH}`,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "User consented to db push --force-reset against per-suite SQLite test databases in /tmp on 2026-05-03 (option #1)",
    },
    stdio: "pipe",
  });
  ({ prisma } = await import("../../prisma.js"));
  ({ deploymentsRouter } = await import("../../routes/deployments.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

type SentMessage = { nodeId: string; message: { type: string; payload: Record<string, unknown> } };

/**
 * The slice of the real `TrainingRecipe` the restart route reads. Typed against
 * it rather than `unknown` so renaming `deploy.gpu_memory_utilization` fails
 * here instead of leaving these tests quietly green.
 */
type StubTrainingRecipe = Pick<TrainingRecipe, "file" | "name" | "deploy">;

const TRAINING_RECIPE: StubTrainingRecipe = {
  file: "recipes/test-training",
  name: "Test Training Recipe",
  deploy: {
    container: "vllm-node-custom",
    gpu_memory_utilization: 0.85,
    max_model_len: 8192,
  },
};

function makeStubHub(trainingRecipe: StubTrainingRecipe = TRAINING_RECIPE) {
  const sent: SentMessage[] = [];
  return {
    hub: {
      getRecipes: () => [],
      getTrainingRecipes: () => [trainingRecipe],
      getOllamaModels: () => [],
      sendToAgent: (nodeId: string, message: { type: string; payload: Record<string, unknown> }) => {
        sent.push({ nodeId, message });
      },
    },
    sent,
  };
}

function makeApp(hub: unknown) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.set("sshExec", async () => ({ code: 0, stdout: "false", stderr: "" }));
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

async function wipeAll() {
  await prisma.clusterNode.deleteMany();
  await prisma.deployment.deleteMany();
  await prisma.fineTuneJob.deleteMany();
  await prisma.model.deleteMany();
  await prisma.metricSnapshot.deleteMany();
  await prisma.node.deleteMany();
}

beforeEach(wipeAll);

async function seedFineTuneDeployment(opts: { artifactVariant?: string; status?: string } = {}) {
  const node = await prisma.node.create({
    data: { name: "n1", status: "online", vramTotal: 128000, ipAddress: "10.0.0.10" },
  });
  const job = await prisma.fineTuneJob.create({
    data: {
      nodeId: node.id,
      baseModel: "meta-llama/Llama-3.1-8B",
      method: "lora",
      dataset: "test-dataset",
      recipeFile: "recipes/test-training",
      status: "completed",
      mergeStatus: "completed",
      outputDir: "/mnt/tank/outputs/job1",
      mergedPath: "/mnt/tank/outputs/job1/merged",
    },
  });
  const model = await prisma.model.create({
    data: { name: "finetune-job1", runtime: "vllm", finetuneJobId: job.id },
  });
  const config: Record<string, unknown> = {
    port: 8000,
    gpuMem: 0.8,
    maxModelLen: 128000,
    tensorParallel: 4,
    localModelPath: "/mnt/tank/outputs/job1/merged",
  };
  if (opts.artifactVariant) config.artifactVariant = opts.artifactVariant;
  const deployment = await prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: opts.status ?? "failed",
      port: 8000,
      displayName: "chat3d-prod",
      config: JSON.stringify(config),
    },
  });
  return { node, job, model, deployment };
}

describe("POST /api/deployments/:id/restart for fine-tune deployments", () => {
  it("sends cmd:finetune:deploy (not cmd:deploy) when the model has a finetuneJobId", async () => {
    const { deployment, job } = await seedFineTuneDeployment();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(res.status).toBe(200);
    expect(sent).toHaveLength(1);
    expect(sent[0].message.type).toBe("cmd:finetune:deploy");
    // The agent's cmd:finetune:deploy handler keys on these fields:
    expect(sent[0].message.payload.jobId).toBe(job.id);
    expect(sent[0].message.payload.deploymentId).toBe(deployment.id);
    expect(sent[0].message.payload.modelPath).toBe("/mnt/tank/outputs/job1/merged");
    expect(sent[0].message.payload.baseModel).toBe("meta-llama/Llama-3.1-8B");
    expect(sent[0].message.payload.recipeFile).toBe("recipes/test-training");
    expect(sent[0].message.payload.modelName).toBe("chat3d-prod");
  });

  it("defaults artifactVariant to 'default' when the saved config doesn't have it", async () => {
    // Pre-variant deployments have no artifactVariant stored. The route now
    // sends "default" (the canonical slug for inference.yaml); the agent maps
    // both "default" and legacy "bf16" to inference.yaml, so this is safe.
    const { deployment } = await seedFineTuneDeployment(); // no artifactVariant set
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(res.status).toBe(200);
    expect(sent[0].message.payload.artifactVariant).toBe("default");
  });

  it("preserves artifactVariant=fp8 from saved config", async () => {
    const { deployment } = await seedFineTuneDeployment({ artifactVariant: "fp8" });
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(res.status).toBe(200);
    expect(sent[0].message.payload.artifactVariant).toBe("fp8");
  });

  it("uses the training recipe's deploy.container when available", async () => {
    const { deployment } = await seedFineTuneDeployment();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(sent[0].message.payload.deployContainer).toBe("vllm-node-custom");
  });

  it("falls back to vllm-node when the training recipe is no longer registered", async () => {
    const { deployment } = await seedFineTuneDeployment();
    const hub = {
      getRecipes: () => [],
      getTrainingRecipes: () => [],
      getOllamaModels: () => [],
      sendToAgent: (_n: string, _m: unknown) => { /* noop */ },
    };
    // Re-build hub with capture
    const sent: SentMessage[] = [];
    const capturingHub = {
      ...hub,
      sendToAgent: (nodeId: string, message: { type: string; payload: Record<string, unknown> }) => {
        sent.push({ nodeId, message });
      },
    };
    const app = makeApp(capturingHub);

    await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(sent[0].message.payload.deployContainer).toBe("vllm-node");
  });

  it("merges caller-supplied config overrides (e.g. lower maxModelLen) into the payload AND persists them", async () => {
    const { deployment } = await seedFineTuneDeployment();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post(`/api/deployments/${deployment.id}/restart`)
      .send({ config: { maxModelLen: 32768 } });

    expect(res.status).toBe(200);
    const payloadConfig = sent[0].message.payload.config as Record<string, unknown>;
    expect(payloadConfig.maxModelLen).toBe(32768);
    // Original fields should be retained
    expect(payloadConfig.gpuMem).toBe(0.8);
    expect(payloadConfig.tensorParallel).toBe(4);

    // Saved config should be updated to reflect the override
    const after = await prisma.deployment.findUnique({ where: { id: deployment.id } });
    const saved = JSON.parse(after!.config!);
    expect(saved.maxModelLen).toBe(32768);
  });

  it("still uses cmd:deploy for non-finetune deployments (regression guard)", async () => {
    const node = await prisma.node.create({
      data: { name: "n1", status: "online", vramTotal: 128000, ipAddress: "10.0.0.10" },
    });
    const model = await prisma.model.create({ data: { name: "plain-model", runtime: "vllm" } });
    const deployment = await prisma.deployment.create({
      data: {
        nodeId: node.id,
        modelId: model.id,
        status: "failed",
        port: 8000,
        config: JSON.stringify({ port: 8000, recipeFile: "recipes/plain.yaml" }),
      },
    });
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    await request(app).post(`/api/deployments/${deployment.id}/restart`).send({});

    expect(sent[0].message.type).toBe("cmd:deploy");
    expect(sent[0].message.payload.recipeFile).toBe("recipes/plain.yaml");
  });
});

/**
 * kreuzhofer/dgx-manager#123 — the authorised share a fine-tune deployment
 * recovers on restart.
 *
 * #118 made the share explicit as `config.authorisedGpuMem`, with a fallback
 * chain for rows created before it. Every term in that chain missed for a
 * fine-tune deployment: the sparkrun-catalog lookup is keyed on
 * `config.recipeFile`, and a fine-tune deployment has none — its recipe lives
 * on the FineTuneJob. So such a row recovered a flat 0.85 however large a share
 * its training recipe declared, and the restart was charged for memory that was
 * genuinely its own.
 *
 * Nodes here are 128,000 MB, so the safety margin is 6,400 MB and a share of
 * 0.85 is 108,800 MB against the recipe's 0.9 = 115,200 MB. The node reads
 * 124,000 MB — all of it the deployment's own resident model — which is the band
 * where those two shares disagree about admission.
 */
const NODE_TOTAL_MB = 128_000;
const NODE_READING_MB = 124_000;
/** A training recipe declaring a larger share than the 0.85 fallback. */
const RECIPE_AT_090: StubTrainingRecipe = {
  ...TRAINING_RECIPE,
  deploy: { ...TRAINING_RECIPE.deploy!, gpu_memory_utilization: 0.9 },
};
/** A training recipe that declares a container but no share at all. */
const RECIPE_NO_SHARE: StubTrainingRecipe = {
  ...TRAINING_RECIPE,
  deploy: { container: "vllm-node-custom", max_model_len: 8192 },
};

/**
 * A fine-tune deployment whose saved config is exactly `config` — no `gpuMem`
 * and no `authorisedGpuMem` unless the caller supplies them — on a node already
 * reading `NODE_READING_MB`.
 */
async function seedFineTuneDeploymentWithConfig(
  config: Record<string, unknown>,
  readingMB: number = NODE_READING_MB,
) {
  const node = await prisma.node.create({
    data: { name: "n1", status: "online", vramTotal: NODE_TOTAL_MB, ipAddress: "10.0.0.10" },
  });
  await prisma.metricSnapshot.create({
    data: { nodeId: node.id, vramUsed: readingMB, gpuUtil: 0, timestamp: new Date() },
  });
  const job = await prisma.fineTuneJob.create({
    data: {
      nodeId: node.id,
      baseModel: "meta-llama/Llama-3.1-8B",
      method: "lora",
      dataset: "test-dataset",
      recipeFile: "recipes/test-training",
      // Terminal on every holding column, so the job itself is not a conflict —
      // the memory on this node is the deployment's own resident model.
      status: "completed",
      mergeStatus: "completed",
      outputDir: "/mnt/tank/outputs/job1",
      mergedPath: "/mnt/tank/outputs/job1/merged",
    },
  });
  const model = await prisma.model.create({
    data: { name: "finetune-job1", runtime: "vllm", finetuneJobId: job.id },
  });
  const deployment = await prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: "running",
      port: 8000,
      config: JSON.stringify({ port: 8000, localModelPath: "/mnt/tank/outputs/job1/merged", ...config }),
    },
  });
  return { node, job, model, deployment };
}

describe("POST /api/deployments/:id/restart — the share a fine-tune deployment recovers (#123)", () => {
  it("recovers the training recipe's share, so a restart is not refused for memory that is its own", async () => {
    // Reads 118,000 rather than the 124,000 the refusals below use. #122 made the
    // REQUEST reach the training recipe too, so both sides of this case now
    // resolve 0.9 and the band where the authorised term is observable moved down
    // with them. Verified to still redden without the term: at 0.85 authorised the
    // node computes 118,800 free against 121,600 needed.
    const { deployment } = await seedFineTuneDeploymentWithConfig({}, 118_000);
    const { hub, sent } = makeStubHub(RECIPE_AT_090);

    const res = await request(makeApp(hub))
      .post(`/api/deployments/${deployment.id}/restart`)
      .send({});

    // Credited 115,200 of the 118,000 the node reads, leaving 2,800 counted and
    // 125,200 free against the 121,600 needed. Recovering 0.85 instead credits
    // only 108,800, and the restart 409s.
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("restarting");
    expect(sent).toHaveLength(1);
    expect(sent[0].message.type).toBe("cmd:finetune:deploy");
  });

  /**
   * #122 — the other half of the same dead link. The REQUEST chain could not see
   * a training recipe either, so a fine-tune restart was checked against 0.85
   * while the container it launched used the recipe's share. #118 and #123 fixed
   * what a restart is CREDITED; this is what it is CHECKED against.
   */
  it("asks admission to clear the training recipe's share, not the 0.85 fallback", async () => {
    const { deployment } = await seedFineTuneDeploymentWithConfig({}, 124_000);
    const { hub, sent } = makeStubHub(RECIPE_AT_090);

    const res = await request(makeApp(hub))
      .post(`/api/deployments/${deployment.id}/restart`)
      .send({});

    // Credited its authorised 0.9 (115,200), leaving 8,800 counted and 119,200
    // free. The recipe's 0.9 needs 121,600 and is refused; the 0.85 fallback
    // needed only 115,200 and was admitted.
    expect(res.status).toBe(409);
    expect(res.body.gpuMemoryUtilization).toBe(0.9);
    expect(res.body.shortfalls[0].vramThresholdMB).toBe(121_600);
    expect(res.body.shortfalls[0].vramUsedMB).toBe(124_000 - 115_200);
    expect(res.body.error).toContain("config.gpuMem");
    expect(sent).toHaveLength(0);
  });

  // The refusals, which is where the chain ORDER is observable: each row differs
  // only in which term wins, and `vramUsedMB` is the node reading minus the share
  // that won. A max-of-chain implementation would admit all three.
  it.each([
    {
      why: "falls back to 0.85 when the training recipe declares no share",
      config: {} as Record<string, unknown>,
      recipe: RECIPE_NO_SHARE,
      shareMB: 108_800,
      requested: 0.85,
    },
    {
      // `authorisedGpuMem` feeds the authorised chain ONLY. The request falls
      // through it to the training recipe's 0.9 (#122), so this row shows the two
      // chains resolving to genuinely different numbers from one row.
      why: "lets a stored authorisedGpuMem win over the recipe, even when it is smaller",
      config: { authorisedGpuMem: 0.5 },
      recipe: RECIPE_AT_090,
      shareMB: 64_000,
      requested: 0.9,
    },
    {
      // A saved `gpuMem`, by contrast, is the first term of BOTH chains — it is
      // what the deployment asked for and what it was therefore authorised for.
      why: "lets a stored gpuMem win over the recipe for a row predating authorisedGpuMem",
      config: { gpuMem: 0.8 },
      recipe: RECIPE_AT_090,
      shareMB: 102_400,
      requested: 0.8,
    },
  ])("$why", async ({ config, recipe, shareMB, requested }) => {
    const { deployment } = await seedFineTuneDeploymentWithConfig(config);
    const { hub, sent } = makeStubHub(recipe);

    const res = await request(makeApp(hub))
      .post(`/api/deployments/${deployment.id}/restart`)
      .send({});

    expect(res.status).toBe(409);
    expect(res.body.shortfalls[0].vramUsedMB).toBe(NODE_READING_MB - shareMB);
    // The requested share is resolved independently of the authorised one, from a
    // chain that skips the row's `authorisedGpuMem` entirely — so these rows show
    // the two landing on genuinely different numbers (#122 gave the request chain
    // the training recipe; before it, every row here resolved 0.85).
    expect(res.body.gpuMemoryUtilization).toBe(requested);
    expect(sent).toHaveLength(0);
  });
});
