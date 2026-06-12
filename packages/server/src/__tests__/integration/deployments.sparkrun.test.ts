/**
 * Integration tests for the sparkrun recipe-source paths on POST /api/deployments
 * (Task 10 — D5/D7).
 *
 * Covers:
 *   - recipePath path-traversal → 400, no cmd:deploy emitted
 *   - recipeYaml empty / missing required key → 400, no cmd:deploy emitted
 *   - recipeYaml valid inline → 201, cmd:deploy payload carries inlineRecipeYaml
 *   - no recipe source at all (vLLM) → 400
 *
 * Pattern mirrors deployments.vram-admission.test.ts.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

// Per-suite SQLite — must be set BEFORE any prisma import.
const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-sparkrun-test-"));
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

/** Wipe all tables in FK-dependency order. */
async function wipeAll() {
  await prisma.loadBalancerEndpoint.deleteMany({});
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.metricSnapshot.deleteMany({});
  await prisma.fineTuneJob.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
}

type SentMessage = { nodeId: string; message: unknown };

/**
 * Minimal stub AgentHub that records every sendToAgent call so tests can
 * assert the emitted cmd:deploy payload.
 */
function makeStubHub() {
  const sent: SentMessage[] = [];
  const hub = {
    getRecipes: () => [],
    getTrainingRecipes: () => [],
    getOllamaModels: () => [],
    isAgentOnline: (_id: string) => false,
    onlineNodeIds: () => [] as string[],
    sendToAgent: (nodeId: string, message: unknown) => sent.push({ nodeId, message }),
  };
  return { hub, sent };
}

function makeApp(hub: unknown) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

// ---------------------------------------------------------------------------
// Validation-only tests (return before any DB work — no node required)
// ---------------------------------------------------------------------------

describe("POST /api/deployments — recipe-source validation (400 paths)", () => {
  it("rejects a traversal recipePath with 400 and emits no cmd:deploy", async () => {
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "node-a", recipePath: "../../etc/passwd" });

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/resolves outside shared storage/i);
    expect(sent).toHaveLength(0);
  });

  it("rejects empty recipeYaml with 400 and emits no cmd:deploy", async () => {
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "node-a", recipeYaml: "   " });

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/empty/i);
    expect(sent).toHaveLength(0);
  });

  it("rejects recipeYaml missing required keys with 400 and emits no cmd:deploy", async () => {
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "node-a", recipeYaml: "description: just a comment\n" });

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/does not look like a sparkrun recipe/i);
    expect(sent).toHaveLength(0);
  });

  it("rejects a vLLM deployment with no recipe source with 400", async () => {
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "node-a" }); // no runtime = vLLM default, no recipe

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/recipeFile, recipePath, or recipeYaml required/i);
    expect(sent).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Happy-path inline YAML test (requires DB setup — node with enough VRAM)
// ---------------------------------------------------------------------------

describe("POST /api/deployments — inline recipeYaml happy path", () => {
  it("accepts valid inline recipeYaml and emits cmd:deploy carrying inlineRecipeYaml", async () => {
    await wipeAll();

    // Seed a single node with plenty of VRAM and no existing deployments
    // so the admission check passes.
    await prisma.node.create({
      data: {
        id: "node-inline-1",
        name: "dgx-spark-inline-01",
        ipAddress: "192.168.44.99",
        vramTotal: 122_502,
        status: "online",
      },
    });

    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const yaml = "model: myorg/my-model\nruntime: vllm\n";

    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "node-inline-1", recipeYaml: yaml });

    expect(res.status).toBe(201);
    expect(res.body.id).toBeDefined();

    // The route must have dispatched exactly one cmd:deploy to the node.
    expect(sent).toHaveLength(1);
    expect(sent[0].nodeId).toBe("node-inline-1");

    const msg = sent[0].message as { type: string; payload: Record<string, unknown> };
    expect(msg.type).toBe("cmd:deploy");
    expect(msg.payload.inlineRecipeYaml).toBe(yaml);
    // recipeRef should be undefined for the inline case.
    expect(msg.payload.recipeRef).toBeUndefined();
    // recipeFile should be undefined (no registry ref provided).
    expect(msg.payload.recipeFile).toBeUndefined();
    expect(msg.payload.runtime).toBe("vllm");
  });

  it("names the Model record after the recipe's model id (not inline-<ts>)", async () => {
    await wipeAll();
    await prisma.node.create({
      data: {
        id: "node-inline-2", name: "dgx-spark-inline-02", ipAddress: "192.168.44.96",
        vramTotal: 122_502, status: "online",
      },
    });
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const yaml = "model: google/gemma-4-12B-it-qat-w4a16-ct\nruntime: vllm\ndefaults:\n  served_model_name: gemma4-12b-unified\n";
    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "node-inline-2", recipeYaml: yaml });

    expect(res.status).toBe(201);
    const created = await prisma.deployment.findUnique({
      where: { id: res.body.id }, include: { model: true },
    });
    // The HF model id is the title — no opaque `inline-<timestamp>`.
    expect(created?.model.name).toBe("google/gemma-4-12B-it-qat-w4a16-ct");
    expect(created?.model.name).not.toMatch(/^inline-\d+$/);
  });

  it("falls back to served_model_name when the recipe has no top-level model:", async () => {
    await wipeAll();
    await prisma.node.create({
      data: {
        id: "node-inline-3", name: "dgx-spark-inline-03", ipAddress: "192.168.44.95",
        vramTotal: 122_502, status: "online",
      },
    });
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    // command-style recipe (no top-level model:) — served_model_name supplies the label
    const yaml = "command: vllm serve foo --served-model-name my-alias\nruntime: vllm\ndefaults:\n  served_model_name: my-alias\n";
    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "node-inline-3", recipeYaml: yaml });

    expect(res.status).toBe(201);
    const created = await prisma.deployment.findUnique({
      where: { id: res.body.id }, include: { model: true },
    });
    expect(created?.model.name).toBe("my-alias");
  });
});

// ---------------------------------------------------------------------------
// recipePath → recipeRef forwarding (validation only — we can't resolve a
// real file path in CI, but the 400 is covered above; here we verify the
// happy-path conversion using a path that stays inside SHARED_STORAGE).
// ---------------------------------------------------------------------------

describe("POST /api/deployments — recipePath → recipeRef forwarding", () => {
  it("resolves a valid recipePath inside SHARED_STORAGE and emits it as recipeRef", async () => {
    await wipeAll();

    await prisma.node.create({
      data: {
        id: "node-rp-1",
        name: "dgx-spark-rp-01",
        ipAddress: "192.168.44.98",
        vramTotal: 122_502,
        status: "online",
      },
    });

    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    // Use a relative path that resolves *inside* SHARED_STORAGE (/mnt/tank).
    // The file doesn't need to exist — the route only validates the path, not
    // the file contents.
    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "node-rp-1", recipePath: "recipes/my-model.yaml" });

    expect(res.status).toBe(201);
    expect(sent).toHaveLength(1);

    const msg = sent[0].message as { type: string; payload: Record<string, unknown> };
    expect(msg.type).toBe("cmd:deploy");
    // recipeRef must be the resolved absolute path (inside /mnt/tank).
    expect(typeof msg.payload.recipeRef).toBe("string");
    expect((msg.payload.recipeRef as string)).toContain("recipes/my-model.yaml");
    expect(msg.payload.inlineRecipeYaml).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Restart fix: POST /:id/restart on a vLLM deployment must send recipeRef
// so the agent's sparkrun branch is triggered (not the "No recipeRef" error).
// ---------------------------------------------------------------------------

describe("POST /api/deployments/:id/restart — vLLM deployment sends recipeRef", () => {
  it("emits cmd:deploy with recipeRef set to the stored recipeFile", async () => {
    await wipeAll();

    // Seed a node and a vLLM deployment that was created with a registry recipe.
    await prisma.node.create({
      data: {
        id: "node-restart-1",
        name: "dgx-spark-restart-01",
        ipAddress: "192.168.44.97",
        vramTotal: 122_502,
        status: "online",
      },
    });

    const recipeFileRef = "@sparkrun-transitional/qwen3-1.7b-vllm";
    const savedConfig = {
      recipeFile: recipeFileRef,
      port: 8000,
      gpuMem: 0.85,
    };

    const model = await prisma.model.create({
      data: {
        name: "qwen3-1.7b",
        runtime: "vllm",
      },
    });

    const deployment = await prisma.deployment.create({
      data: {
        nodeId: "node-restart-1",
        modelId: model.id,
        status: "failed", // typical state before a restart
        config: JSON.stringify(savedConfig),
      },
    });

    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post(`/api/deployments/${deployment.id}/restart`)
      .send({});

    expect(res.status).toBe(200);
    expect(res.body.status).toBe("restarting");

    // Exactly one cmd:deploy must have been emitted.
    expect(sent).toHaveLength(1);
    const msg = sent[0].message as { type: string; payload: Record<string, unknown> };
    expect(msg.type).toBe("cmd:deploy");

    // recipeRef must be set (not undefined) so the agent's sparkrun branch fires.
    expect(msg.payload.recipeRef).toBe(recipeFileRef);
    // recipeFile is still sent for backwards compatibility.
    expect(msg.payload.recipeFile).toBe(recipeFileRef);
  });
});
