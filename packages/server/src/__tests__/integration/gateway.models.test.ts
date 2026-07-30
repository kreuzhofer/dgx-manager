/**
 * Integration tests for the inference gateway's served surface.
 *
 * One seam for everything a client can observe: the gateway router over HTTP
 * (supertest), a per-test SQLite, a stubbed hub for the liveness signal, and a
 * REAL ephemeral upstream standing in for a node — so "nothing reaches a node"
 * is asserted against a server that records requests, not assumed.
 *
 * The fake-upstream helper follows benchmarks/reasoning-proxy.test.ts; the
 * per-suite-SQLite harness follows deployments.vram-admission.test.ts.
 */
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import http from "node:http";
import request from "supertest";
import express from "express";

// Per-suite SQLite. Must be set before any module that imports prisma.
const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

// Dynamic imports so the env var above is in place before prisma loads.
let prisma: typeof import("../../prisma.js").prisma;
let gatewayRouter: typeof import("../../gateway/router.js").gatewayRouter;

beforeAll(async () => {
  // PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION: see the note in
  // deployments.vram-admission.test.ts — DATABASE_URL here always points at a
  // freshly-mkdtemp'd SQLite file in /tmp.
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
  ({ gatewayRouter } = await import("../../gateway/router.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

async function wipeAll() {
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
}
afterEach(wipeAll);

/**
 * A node's model server. Records every request it receives so a test can prove
 * the gateway never forwarded one. Answers /v1/models the way a real Ollama
 * node would — advertising everything present on the box, including models
 * nobody deployed.
 */
function fakeUpstream(advertises: string[]) {
  const received: string[] = [];
  return new Promise<{ host: string; port: number; received: string[]; close: () => Promise<void> }>(
    (resolve) => {
      const server = http.createServer((req, res) => {
        received.push(`${req.method} ${req.url}`);
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ object: "list", data: advertises.map((id) => ({ id })) }));
      });
      server.listen(0, "127.0.0.1", () => {
        const addr = server.address() as { port: number };
        resolve({
          host: "127.0.0.1",
          port: addr.port,
          received,
          close: () => new Promise((r) => server.close(() => r())),
        });
      });
    },
  );
}

/** Mounts ONLY the gateway, with a stub hub, exactly as index.ts does. */
function makeApp(agentHub: unknown = { isAgentOnline: () => true }) {
  const app = express();
  app.set("agentHub", agentHub);
  app.use("/v1", gatewayRouter);
  return app;
}

// Node.name and Model.name are unique, so each seed needs its own.
let seedSeq = 0;

async function seedRunningDeployment(opts: {
  publishedName: string | null;
  status?: string;
  ip?: string;
  port?: number;
  runtime?: string;
}) {
  const n = ++seedSeq;
  const node = await prisma.node.create({
    data: { name: `node-${n}`, ipAddress: opts.ip ?? `10.0.0.${n}`, status: "online" },
  });
  const model = await prisma.model.create({
    data: { name: `catalog-${n}`, runtime: opts.runtime ?? "vllm" },
  });
  return prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: opts.status ?? "running",
      port: opts.port ?? 8000,
      publishedName: opts.publishedName,
    },
  });
}

describe("GET /v1/models", () => {
  it("lists the published names of running deployments", async () => {
    await seedRunningDeployment({ publishedName: "glm-5.2", ip: "10.0.0.1" });
    await seedRunningDeployment({ publishedName: "qwen3-embedding:8b", ip: "10.0.0.2" });

    const res = await request(makeApp()).get("/v1/models");

    expect(res.status).toBe(200);
    expect(res.body.object).toBe("list");
    expect(res.body.data.map((m: { id: string }) => m.id)).toEqual([
      "glm-5.2",
      "qwen3-embedding:8b",
    ]);
    expect(res.body.data[0]).toMatchObject({ object: "model", owned_by: "dgx-manager" });
  });

  // The containment property the whole design rests on: a node may hold models
  // nobody deployed, and those must be neither discoverable nor callable.
  it("never advertises a model the node holds but nobody deployed", async () => {
    const upstream = await fakeUpstream(["qwen3-embedding:8b", "some-secret-model", "llama3:70b"]);
    await seedRunningDeployment({
      publishedName: "qwen3-embedding:8b",
      ip: upstream.host,
      port: upstream.port,
      runtime: "ollama",
    });

    const res = await request(makeApp()).get("/v1/models");

    expect(res.body.data.map((m: { id: string }) => m.id)).toEqual(["qwen3-embedding:8b"]);
    // Synthesized from the database, not proxied — the node was never asked.
    expect(upstream.received).toEqual([]);
    await upstream.close();
  });

  it("omits a deployment that is not running", async () => {
    await seedRunningDeployment({ publishedName: "live", ip: "10.0.0.1" });
    await seedRunningDeployment({ publishedName: "dead", status: "stopped", ip: "10.0.0.2" });

    const res = await request(makeApp()).get("/v1/models");

    expect(res.body.data.map((m: { id: string }) => m.id)).toEqual(["live"]);
  });

  it("omits a running deployment that has no published name yet", async () => {
    await seedRunningDeployment({ publishedName: null, ip: "10.0.0.1" });

    const res = await request(makeApp()).get("/v1/models");

    expect(res.body.data).toEqual([]);
  });

  it("collapses a pool serving one name into a single entry", async () => {
    await seedRunningDeployment({ publishedName: "qwen3-embedding:8b", ip: "10.0.0.1" });
    await seedRunningDeployment({ publishedName: "qwen3-embedding:8b", ip: "10.0.0.2" });

    const res = await request(makeApp()).get("/v1/models");

    expect(res.body.data).toHaveLength(1);
  });
});

describe("the gateway's allowlist", () => {
  // A runtime's own API must be unreachable through the gateway — including the
  // operations that mutate what a node holds (pull, delete, push).
  it.each([
    ["POST", "/v1/chat/completions"],
    ["POST", "/v1/embeddings"],
    ["POST", "/v1/completions"],
    ["GET", "/v1/models/glm-5.2"],
    ["POST", "/api/pull"],
    ["DELETE", "/api/delete"],
    ["GET", "/api/tags"],
    ["GET", "/anything-else"],
  ])("refuses %s %s without contacting a node", async (method, path) => {
    const upstream = await fakeUpstream(["whatever"]);
    await seedRunningDeployment({
      publishedName: "glm-5.2",
      ip: upstream.host,
      port: upstream.port,
    });

    const app = makeApp();
    const res = await (request(app) as unknown as Record<string, (p: string) => request.Test>)[
      method.toLowerCase()
    ](path.startsWith("/v1") ? path : `/v1${path}`);

    expect(res.status).toBe(404);
    expect(res.body.error).toMatchObject({ type: "invalid_request_error" });
    expect(upstream.received).toEqual([]);
    await upstream.close();
  });

  // Proves the mount sits AHEAD of the global JSON parser: were it behind, a
  // body over the parser's 100kb default would be rejected with 413 before the
  // gateway ever saw the request. Ticket #7 depends on this ordering to carry
  // megabyte-scale long-context prompts.
  it("is reached before the body-size limit that guards the management API", async () => {
    const oversized = { prompt: "x".repeat(200_000) };

    const res = await request(makeApp())
      .post("/v1/chat/completions")
      .set("content-type", "application/json")
      .send(oversized);

    expect(res.status).toBe(404);
    expect(res.status).not.toBe(413);
  });
});
