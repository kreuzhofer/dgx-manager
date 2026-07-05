/**
 * Integration test: GET /api/recipes merges the in-repo @dgxrun catalog
 * (packages/server/src/deployments/dgxrun-catalog.ts) with whatever agents
 * report via agentHub.getRecipes().
 *
 * Mirrors the bootstrap in deployments.dgxrun.test.ts: per-suite sqlite via
 * mkdtempSync + DATABASE_URL set BEFORE importing prisma, schema applied via
 * `npx prisma db push --force-reset`.
 *
 * DGXRUN_RECIPES_DIR is pointed at the repo's real recipes/dgxrun directory
 * (resolved relative to this test file) BEFORE importing dgxrun-catalog.ts /
 * recipes.ts, so getDgxrunCatalog() picks it up on first read.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-recipes-catalog-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

// Repo root is four levels up from this file:
// packages/server/src/__tests__/integration/dgxrun-catalog.test.ts
const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "../../../../..");
process.env.DGXRUN_RECIPES_DIR = join(REPO_ROOT, "recipes/dgxrun");

let prisma: typeof import("../../prisma.js").prisma;
let recipesRouter: typeof import("../../routes/recipes.js").recipesRouter;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      DATABASE_URL: `file:${DB_PATH}`,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "User consented to db push --force-reset against per-suite SQLite test databases in /tmp on 2026-07-05 (option #1)",
    },
    stdio: "pipe",
  });
  ({ prisma } = await import("../../prisma.js"));
  ({ recipesRouter } = await import("../../routes/recipes.js"));
  ({ deploymentsRouter } = await import("../../routes/deployments.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

async function wipeAll() {
  await prisma.loadBalancerEndpoint.deleteMany({});
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.metricSnapshot.deleteMany({});
  await prisma.fineTuneJob.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
}

type SentMessage = { nodeId: string; message: { type: string; payload: Record<string, unknown> } };

function makeStubHub() {
  return {
    getRecipes: () => [],
    getOllamaModels: () => [],
    getConnectedNodeIds: () => [] as string[],
    sendToAgent: () => {},
  };
}

function makeStubDeployHub() {
  const sent: SentMessage[] = [];
  const hub = {
    getRecipes: () => [],
    getTrainingRecipes: () => [],
    getOllamaModels: () => [],
    isAgentOnline: (_id: string) => true,
    onlineNodeIds: () => [] as string[],
    sendToAgent: (nodeId: string, message: unknown) => sent.push({ nodeId, message: message as SentMessage["message"] }),
  };
  return { hub, sent };
}

function makeApp(hub: unknown) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.use("/api/recipes", recipesRouter);
  return app;
}

function makeDeployApp(hub: unknown) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.set("sshExec", async () => ({ code: 0, stdout: "false", stderr: "" }));
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

async function seedCluster(n: number) {
  const ids: string[] = [];
  for (let i = 0; i < n; i++) {
    const id = `dgx-catalog-node-${i}`;
    await prisma.node.create({
      data: {
        id, name: `dgx-spark-catalog-${i}`,
        ipAddress: `192.168.44.${60 + i}`,
        vramTotal: 122_502, status: "online",
      },
    });
    ids.push(id);
  }
  return ids;
}

describe("GET /api/recipes includes @dgxrun catalog", () => {
  it("lists the @dgxrun/glm-5.2-awq-15pct recipe with source dgxrun", async () => {
    const app = makeApp(makeStubHub());

    const res = await request(app).get("/api/recipes");

    expect(res.status).toBe(200);
    const hit = res.body.find((r: any) => r.file === "@dgxrun/glm-5.2-awq-15pct");
    expect(hit).toBeTruthy();
    expect(hit.source).toBe("dgxrun");
  });
});

describe("POST /api/deployments with @dgxrun recipeFile", () => {
  it("routes an @dgxrun/ recipeFile to the dgxrun runner (per-rank fan-out)", async () => {
    await wipeAll();
    const ids = await seedCluster(4);

    const { hub, sent } = makeStubDeployHub();
    const app = makeDeployApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeIds: ids, recipeFile: "@dgxrun/glm-5.2-awq-15pct" });

    expect(res.status).toBe(201);

    // One dgxrun cmd:deploy per node, resolved from the in-repo catalog YAML
    // (not the sparkrun path) — same shape asserted in deployments.dgxrun.test.ts.
    expect(sent).toHaveLength(4);
    expect(sent.map((s) => s.nodeId)).toEqual(ids);
    for (const s of sent) {
      expect(s.message.type).toBe("cmd:deploy");
      expect(s.message.payload.kind).toBe("dgxrun");
      const recipe = s.message.payload.recipe as Record<string, unknown>;
      expect(recipe.container).toBe("vllm-node-tf5-glm52-b12x:probe");
      expect(s.message.payload.masterAddr).toBe("192.168.44.60");
    }
    expect(sent.map((s) => s.message.payload.rank)).toEqual([0, 1, 2, 3]);
    expect(sent.map((s) => s.message.payload.headless)).toEqual([false, true, true, true]);

    const created = await prisma.deployment.findUnique({
      where: { id: res.body.id }, include: { clusterNodes: true },
    });
    expect(created?.clusterMode).toBe(true);
    expect(created?.clusterNodes).toHaveLength(4);
    expect(JSON.parse(created!.config!).runner).toBe("dgxrun");
  });

  it("falls through to the sparkrun path for a non-@dgxrun recipeFile", async () => {
    await wipeAll();
    const ids = await seedCluster(2);
    const { hub, sent } = makeStubDeployHub();
    const app = makeDeployApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeIds: ids, recipeFile: "some-registry/plain-recipe" });

    // Unresolvable in-repo, so it's treated as a registry ref for sparkrun —
    // unchanged existing behavior.
    expect(res.status).toBe(201);
    expect(sent).toHaveLength(1);
    expect(sent[0].message.payload.kind).not.toBe("dgxrun");
    const created = await prisma.deployment.findUnique({ where: { id: res.body.id } });
    expect(JSON.parse(created!.config!).runner).toBeUndefined();
  });
});
