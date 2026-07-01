import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let hfCacheRouter: typeof import("../../routes/hf-cache.js").hfCacheRouter;
let recordRepoDeployment: typeof import("../../hf-cache/repo-deployment.js").recordRepoDeployment;
let loadRepoLastDeployed: typeof import("../../hf-cache/repo-deployment.js").loadRepoLastDeployed;

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
  ({ hfCacheRouter } = await import("../../routes/hf-cache.js"));
  ({ recordRepoDeployment, loadRepoLastDeployed } = await import("../../hf-cache/repo-deployment.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  // FK-dependency order
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.fineTuneJob.deleteMany({});
  await prisma.node.deleteMany({});
  await prisma.repoDeployment.deleteMany({});
});

/** Minimal stand-in for AgentHub — only what the router touches. */
function makeHub() {
  const hub = {
    inventories: [] as Record<string, unknown>[],
    recipes: [] as { file: string; model?: string }[],
    sent: [] as { nodeId: string; message: { type: string; payload: Record<string, unknown> } }[],
    online: new Set<string>(),
    getHfCacheInventories() { return hub.inventories; },
    getRecipes() { return hub.recipes; },
    getConnectedNodeIds() { return [...hub.online]; },
    isAgentOnline(id: string) { return hub.online.has(id); },
    sendToAgent(nodeId: string, message: { type: string; payload: Record<string, unknown> }) {
      hub.sent.push({ nodeId, message });
    },
  };
  return hub;
}

function makeApp(hub: ReturnType<typeof makeHub>) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.use("/api/hf-cache", hfCacheRouter);
  return app;
}

function repo(repoId: string, sizeBytes = 1000) {
  return {
    repoId, kind: "model", sizeBytes, nFiles: 3, revisions: 1,
    lastModified: "2026-06-01T00:00:00.000Z",
  };
}

function inv(nodeId: string, cacheId: string, repos: ReturnType<typeof repo>[], extra: Record<string, unknown> = {}) {
  return {
    nodeId, cacheId, hfHome: "/mnt/tank/models",
    scannedAt: "2026-06-13T00:00:00.000Z",
    totalBytes: repos.reduce((s, r) => s + r.sizeBytes, 0),
    diskFreeBytes: 1_000_000, repos, ...extra,
  };
}

describe("GET /api/hf-cache", () => {
  it("groups shared-cache nodes and resolves node names + connectivity", async () => {
    const n1 = await prisma.node.create({ data: { name: "spark-1" } });
    const n2 = await prisma.node.create({ data: { name: "spark-2" } });
    const hub = makeHub();
    hub.inventories = [inv(n1.id, "shared", [repo("org/alpha")]), inv(n2.id, "shared", [repo("org/alpha")])];
    hub.online.add(n1.id);

    const res = await request(makeApp(hub)).get("/api/hf-cache");
    expect(res.status).toBe(200);
    expect(res.body.caches).toHaveLength(1);
    const cache = res.body.caches[0];
    expect(cache.cacheId).toBe("shared");
    expect(cache.nodes.map((n: { name: string }) => n.name).sort()).toEqual(["spark-1", "spark-2"]);
    expect(cache.nodes.find((n: { name: string }) => n.name === "spark-1").connected).toBe(true);
    expect(cache.nodes.find((n: { name: string }) => n.name === "spark-2").connected).toBe(false);
  });

  it("keeps per-node caches separate", async () => {
    const n1 = await prisma.node.create({ data: { name: "spark-1" } });
    const n2 = await prisma.node.create({ data: { name: "spark-2" } });
    const hub = makeHub();
    hub.inventories = [inv(n1.id, "local-1", [repo("org/alpha")]), inv(n2.id, "local-2", [repo("org/beta")])];

    const res = await request(makeApp(hub)).get("/api/hf-cache");
    expect(res.body.caches).toHaveLength(2);
  });

  it("enriches repos with inUse and lastDeployedAt from deployments (by Model.name)", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const model = await prisma.model.create({ data: { name: "org/alpha", runtime: "vllm" } });
    await prisma.deployment.create({
      data: { nodeId: node.id, modelId: model.id, status: "running", displayName: "alpha-prod" },
    });
    const staleModel = await prisma.model.create({ data: { name: "org/old", runtime: "vllm" } });
    await prisma.deployment.create({
      data: { nodeId: node.id, modelId: staleModel.id, status: "stopped" },
    });

    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("org/alpha"), repo("org/old"), repo("org/never")])];

    const res = await request(makeApp(hub)).get("/api/hf-cache");
    const repos = res.body.caches[0].repos;
    const byId = Object.fromEntries(repos.map((r: { repoId: string }) => [r.repoId, r]));
    expect(byId["org/alpha"].inUse).toBe(true);
    expect(byId["org/alpha"].inUseBy).toEqual(["alpha-prod"]);
    expect(byId["org/old"].inUse).toBe(false);
    expect(byId["org/old"].lastDeployedAt).not.toBeNull();
    expect(byId["org/never"].inUse).toBe(false);
    expect(byId["org/never"].lastDeployedAt).toBeNull();
  });

  it("flags in-use for a REGISTRY-REF vLLM deploy via the recipe catalog HF id (not Model.name)", async () => {
    // The dangerous case: Model.name is the recipe slug, the cached repo is the
    // HF id. Matching must resolve config.recipeFile -> recipe.model.
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const model = await prisma.model.create({ data: { name: "gemma4-26b-a4b", runtime: "vllm" } });
    await prisma.deployment.create({
      data: {
        nodeId: node.id, modelId: model.id, status: "running", displayName: "gemma-prod",
        config: JSON.stringify({ recipeFile: "recipes/gemma4-26b-a4b.yaml" }),
      },
    });
    const hub = makeHub();
    hub.recipes = [{ file: "recipes/gemma4-26b-a4b.yaml", model: "google/gemma-4-26B-A4B-it" }];
    hub.inventories = [inv(node.id, "shared", [repo("google/gemma-4-26B-A4B-it")])];

    const res = await request(makeApp(hub)).get("/api/hf-cache");
    const r = res.body.caches[0].repos.find((x: { repoId: string }) => x.repoId === "google/gemma-4-26B-A4B-it");
    expect(r.inUse).toBe(true);
    expect(r.inUseBy).toEqual(["gemma-prod"]);
  });

  it("flags in-use for a fine-tune deploy's base model (base weights load from cache)", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const job = await prisma.fineTuneJob.create({
      data: { nodeId: node.id, baseModel: "google/gemma-2-9b", method: "lora", dataset: "d" },
    });
    const model = await prisma.model.create({
      data: { name: "my-finetune", runtime: "vllm", finetuneJobId: job.id },
    });
    await prisma.deployment.create({
      data: { nodeId: node.id, modelId: model.id, status: "running", displayName: "ft-prod" },
    });
    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("google/gemma-2-9b")])];

    const res = await request(makeApp(hub)).get("/api/hf-cache");
    const r = res.body.caches[0].repos.find((x: { repoId: string }) => x.repoId === "google/gemma-2-9b");
    expect(r.inUse).toBe(true);
  });

  it("shows lastDeployedAt from the durable RepoDeployment table with NO live deployment (the bug)", async () => {
    // Reproduces the reported bug: the deployment was created then deleted (the
    // cleanup workflow hard-deletes rows), so there is no Deployment/Model row
    // left — yet the repo must still show when it was last deployed.
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    await prisma.repoDeployment.create({
      data: { repoId: "qwen/qwen3-32b-awq", lastDeployedAt: new Date("2026-06-20T12:00:00.000Z") },
    });
    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("Qwen/Qwen3-32B-AWQ")])];

    const res = await request(makeApp(hub)).get("/api/hf-cache");
    const r = res.body.caches[0].repos.find((x: { repoId: string }) => x.repoId === "Qwen/Qwen3-32B-AWQ");
    // Case-insensitive lookup against the lowercased key, and no live deployment.
    expect(r.lastDeployedAt).toBe("2026-06-20T12:00:00.000Z");
    expect(r.inUse).toBe(false);
  });

  it("takes the newer of the live-deployment date and the durable record", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const model = await prisma.model.create({ data: { name: "org/alpha", runtime: "vllm" } });
    await prisma.deployment.create({
      data: { nodeId: node.id, modelId: model.id, status: "running", displayName: "alpha" },
    });
    // Durable record is OLDER than the live deployment (createdAt ~ now).
    await prisma.repoDeployment.create({
      data: { repoId: "org/alpha", lastDeployedAt: new Date("2020-01-01T00:00:00.000Z") },
    });
    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("org/alpha")])];

    const res = await request(makeApp(hub)).get("/api/hf-cache");
    const r = res.body.caches[0].repos.find((x: { repoId: string }) => x.repoId === "org/alpha");
    expect(new Date(r.lastDeployedAt).getTime()).toBeGreaterThan(new Date("2020-01-01T00:00:00.000Z").getTime());
  });

  it("returns an empty caches list when no agent has reported", async () => {
    const res = await request(makeApp(makeHub())).get("/api/hf-cache");
    expect(res.status).toBe(200);
    expect(res.body.caches).toEqual([]);
  });

  it("flags in-use when a multi-node deploy's WORKER node holds the cache (clusterNodes seam)", async () => {
    // Head node is elsewhere; the cache lives on a worker node in the group.
    const head = await prisma.node.create({ data: { name: "head-1" } });
    const worker = await prisma.node.create({ data: { name: "worker-1" } });
    const model = await prisma.model.create({ data: { name: "org/alpha", runtime: "vllm" } });
    const dep = await prisma.deployment.create({
      data: { nodeId: head.id, modelId: model.id, status: "running", displayName: "tp-prod", clusterMode: true },
    });
    await prisma.clusterNode.create({ data: { deploymentId: dep.id, nodeId: worker.id, role: "worker" } });

    const hub = makeHub();
    // The cache group is the worker node only (its own local cache id).
    hub.inventories = [inv(worker.id, "worker-cache", [repo("org/alpha")])];

    const res = await request(makeApp(hub)).get("/api/hf-cache");
    const cache = res.body.caches.find((c: { cacheId: string }) => c.cacheId === "worker-cache");
    const r = cache.repos.find((x: { repoId: string }) => x.repoId === "org/alpha");
    expect(r.inUse).toBe(true);
    expect(r.inUseBy).toEqual(["tp-prod"]);
  });
});

describe("recordRepoDeployment / loadRepoLastDeployed", () => {
  it("round-trips keys and keeps the newest timestamp on re-record", async () => {
    await recordRepoDeployment(prisma, ["org/alpha", "org/beta"], new Date("2026-06-01T00:00:00.000Z"));
    let map = await loadRepoLastDeployed(prisma);
    expect(map.get("org/alpha")).toBe("2026-06-01T00:00:00.000Z");
    expect(map.get("org/beta")).toBe("2026-06-01T00:00:00.000Z");

    // A later deploy overwrites with the newer date.
    await recordRepoDeployment(prisma, ["org/alpha"], new Date("2026-06-15T00:00:00.000Z"));
    map = await loadRepoLastDeployed(prisma);
    expect(map.get("org/alpha")).toBe("2026-06-15T00:00:00.000Z");
    expect(map.get("org/beta")).toBe("2026-06-01T00:00:00.000Z");
  });

  it("no-ops on an empty key set", async () => {
    await recordRepoDeployment(prisma, [], new Date("2026-06-01T00:00:00.000Z"));
    expect((await loadRepoLastDeployed(prisma)).size).toBe(0);
  });
});

describe("POST /api/hf-cache/scan", () => {
  it("503 when no agents are connected", async () => {
    const res = await request(makeApp(makeHub())).post("/api/hf-cache/scan");
    expect(res.status).toBe(503);
  });

  it("fans out cmd:hf-cache:scan to every connected agent", async () => {
    const hub = makeHub();
    hub.online.add("n1").add("n2");
    const res = await request(makeApp(hub)).post("/api/hf-cache/scan");
    expect(res.status).toBe(202);
    expect(res.body.requested).toBe(2);
    expect(hub.sent.map((s) => s.message.type)).toEqual(["cmd:hf-cache:scan", "cmd:hf-cache:scan"]);
  });
});

describe("DELETE /api/hf-cache/:cacheId", () => {
  it("400 without a repoId", async () => {
    const hub = makeHub();
    hub.inventories = [inv("n1", "shared", [repo("org/alpha")])];
    const res = await request(makeApp(hub)).delete("/api/hf-cache/shared");
    expect(res.status).toBe(400);
  });

  it("404 for an unknown cache or repo", async () => {
    const hub = makeHub();
    hub.inventories = [inv("n1", "shared", [repo("org/alpha")])];
    const app = makeApp(hub);
    expect((await request(app).delete("/api/hf-cache/nope?repoId=org%2Falpha")).status).toBe(404);
    expect((await request(app).delete("/api/hf-cache/shared?repoId=org%2Fghost")).status).toBe(404);
  });

  it("409 when the repo is in use by an active deployment, and sends nothing", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const model = await prisma.model.create({ data: { name: "org/alpha", runtime: "vllm" } });
    await prisma.deployment.create({
      data: { nodeId: node.id, modelId: model.id, status: "running", displayName: "alpha-prod" },
    });
    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("org/alpha")])];
    hub.online.add(node.id);

    const res = await request(makeApp(hub)).delete("/api/hf-cache/shared?repoId=org%2Falpha");
    expect(res.status).toBe(409);
    expect(res.body.error).toContain("alpha-prod");
    expect(hub.sent).toHaveLength(0);
  });

  it("503 when no agent in the cache group is connected", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("org/alpha")])];
    const res = await request(makeApp(hub)).delete("/api/hf-cache/shared?repoId=org%2Falpha");
    expect(res.status).toBe(503);
  });

  it("202 + sends cmd:hf-cache:delete to a connected group member", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("org/alpha")])];
    hub.online.add(node.id);

    const res = await request(makeApp(hub)).delete("/api/hf-cache/shared?repoId=org%2Falpha&kind=model");
    expect(res.status).toBe(202);
    expect(hub.sent).toHaveLength(1);
    expect(hub.sent[0].nodeId).toBe(node.id);
    expect(hub.sent[0].message).toEqual({
      type: "cmd:hf-cache:delete",
      payload: { repoId: "org/alpha", kind: "model" },
    });
  });

  it("deletes a dataset-kind repo and passes kind through to the agent", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [
      { repoId: "squad", kind: "dataset", sizeBytes: 500, nFiles: 2, revisions: 1, lastModified: "2026-06-01T00:00:00.000Z" },
    ])];
    hub.online.add(node.id);

    const res = await request(makeApp(hub)).delete("/api/hf-cache/shared?repoId=squad&kind=dataset");
    expect(res.status).toBe(202);
    expect(hub.sent[0].message).toEqual({ type: "cmd:hf-cache:delete", payload: { repoId: "squad", kind: "dataset" } });
  });

  it("404s when repoId exists but the kind does not match", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("org/alpha")])]; // kind "model"
    hub.online.add(node.id);
    // ask to delete it as a dataset → no matching repo of that kind
    const res = await request(makeApp(hub)).delete("/api/hf-cache/shared?repoId=org%2Falpha&kind=dataset");
    expect(res.status).toBe(404);
    expect(hub.sent).toHaveLength(0);
  });
});
