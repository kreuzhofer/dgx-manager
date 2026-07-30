/**
 * End-to-end across the seam every other suite stubs one side of.
 *
 * The chain is: agent status message -> hub handler -> Deployment.publishedName
 * -> gateway routing. Elsewhere the producing side asserts the column and stops,
 * and the consuming side seeds the column directly — so nothing checked that the
 * name the hub WRITES is the name the gateway ROUTES ON. Two lifecycle defects
 * (a rename leaving a stale name published, and an idle eviction unpublishing a
 * model permanently) survived thirteen green commits in exactly that gap.
 *
 * These tests drive the real handler and then make a real request through the
 * real router, against a real upstream.
 */
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import http from "node:http";
import request from "supertest";
import express from "express";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let gatewayRouter: typeof import("../../gateway/router.js").gatewayRouter;
let handleDeploymentStatus: typeof import("../../ws/deployment-status-handler.js").handleDeploymentStatus;
let publishNamesForNode: typeof import("../../ws/deployment-status-handler.js").publishNamesForNode;
let resetOutstanding: typeof import("../../gateway/inflight.js").resetOutstanding;

beforeAll(async () => {
  // PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION: see deployments.vram-admission.test.ts.
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
  ({ handleDeploymentStatus, publishNamesForNode } = await import(
    "../../ws/deployment-status-handler.js"
  ));
  ({ resetOutstanding } = await import("../../gateway/inflight.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

afterEach(async () => {
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
  resetOutstanding();
});

const noopHub = { sendToAgent: () => {} };

/** An upstream that reports whatever served name it is told to. */
function upstreamServing(servedName: string) {
  const received: string[] = [];
  return new Promise<{ host: string; port: number; received: string[]; close: () => Promise<void> }>(
    (resolve) => {
      const server = http.createServer((req, res) => {
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
          res.setHeader("content-type", "application/json");
          if (req.url === "/v1/models") {
            res.end(JSON.stringify({ data: [{ id: servedName }] }));
            return;
          }
          received.push(Buffer.concat(chunks).toString());
          res.end(JSON.stringify({ ok: true, servedBy: servedName }));
        });
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

function makeApp() {
  const app = express();
  app.set("agentHub", { isAgentOnline: () => true });
  app.use("/v1", gatewayRouter);
  return app;
}

let seq = 0;
async function seedDeployment(opts: { runtime: string; modelName: string; displayName?: string | null; port: number }) {
  const n = ++seq;
  const node = await prisma.node.create({
    data: { name: `node-${n}`, ipAddress: "127.0.0.1", status: "online" },
  });
  const model = await prisma.model.create({
    data: { name: opts.modelName, runtime: opts.runtime },
  });
  const deployment = await prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: "loading",
      port: opts.port,
      displayName: opts.displayName ?? null,
    },
  });
  return { node, deployment };
}

describe("hub -> published name -> gateway", () => {
  it("routes on the name the hub resolved from the runtime", async () => {
    const upstream = await upstreamServing("what-vllm-really-serves");
    const { deployment } = await seedDeployment({
      runtime: "vllm",
      modelName: "@catalog/some-recipe",
      displayName: "friendly",
      port: upstream.port,
    });

    await handleDeploymentStatus(
      { deploymentId: deployment.id, status: "running", port: upstream.port },
      { hub: noopHub },
    );

    // The list advertises what the runtime answers to, not the local guess.
    const models = await request(makeApp()).get("/v1/models");
    expect(models.body.data.map((m: { id: string }) => m.id)).toEqual(["what-vllm-really-serves"]);

    // And a request for that exact name reaches the node.
    const chat = await request(makeApp())
      .post("/v1/chat/completions")
      .send({ model: "what-vllm-really-serves" });
    expect(chat.status).toBe(200);
    expect(upstream.received).toHaveLength(1);

    // The guess it might have been named is NOT routable.
    const wrong = await request(makeApp()).post("/v1/chat/completions").send({ model: "friendly" });
    expect(wrong.status).toBe(404);

    await upstream.close();
  });

  // Regression: a restart may rename the deployment or change its recipe. The
  // gateway advertised the pre-restart name forever and every request 404'd at
  // the runtime, because the name was only resolved when none was stored.
  it("republishes under the new name after a restart renames the deployment", async () => {
    const upstream = await upstreamServing("v1-name");
    const { deployment } = await seedDeployment({
      runtime: "vllm",
      modelName: "@catalog/r",
      displayName: "v1-name",
      port: upstream.port,
    });
    await handleDeploymentStatus(
      { deploymentId: deployment.id, status: "running", port: upstream.port },
      { hub: noopHub },
    );
    expect((await prisma.deployment.findUnique({ where: { id: deployment.id } }))?.publishedName)
      .toBe("v1-name");
    await upstream.close();

    // Restart clears the name (routes/deployments.ts) and the runtime comes back
    // answering to something else.
    await prisma.deployment.update({
      where: { id: deployment.id },
      data: { status: "restarting", publishedName: null, displayName: "v2-name" },
    });
    const restarted = await upstreamServing("v2-name");
    await prisma.deployment.update({
      where: { id: deployment.id },
      data: { port: restarted.port },
    });

    await handleDeploymentStatus(
      { deploymentId: deployment.id, status: "running", port: restarted.port },
      { hub: noopHub },
    );

    const models = await request(makeApp()).get("/v1/models");
    expect(models.body.data.map((m: { id: string }) => m.id)).toEqual(["v2-name"]);
    await restarted.close();
  });

  // Regression: Ollama unloads idle models and the agent reports `evicted`.
  // Clearing the published name there made the model vanish from the gateway a
  // few idle minutes after last use, and it could never come back — the only
  // thing that reloads it is a request, which the gateway had started 404ing.
  it("keeps an idle-evicted model published so it can come back", async () => {
    const upstream = await upstreamServing("unused-for-ollama");
    const { deployment } = await seedDeployment({
      runtime: "ollama",
      modelName: "qwen3-embedding:8b",
      port: upstream.port,
    });

    await handleDeploymentStatus(
      { deploymentId: deployment.id, status: "running", port: upstream.port },
      { hub: noopHub },
    );
    expect((await prisma.deployment.findUnique({ where: { id: deployment.id } }))?.publishedName)
      .toBe("qwen3-embedding:8b");

    await handleDeploymentStatus(
      { deploymentId: deployment.id, status: "evicted" },
      { hub: noopHub },
    );

    const after = await prisma.deployment.findUnique({ where: { id: deployment.id } });
    expect(after?.publishedName).toBe("qwen3-embedding:8b");
    expect(after?.vramActual).toBe(0);

    // Loading again must not need a fresh name.
    await handleDeploymentStatus(
      { deploymentId: deployment.id, status: "running", port: upstream.port },
      { hub: noopHub },
    );
    const models = await request(makeApp()).get("/v1/models");
    expect(models.body.data.map((m: { id: string }) => m.id)).toEqual(["qwen3-embedding:8b"]);

    await upstream.close();
  });

  it("unpublishes a stopped deployment", async () => {
    const upstream = await upstreamServing("bye");
    const { deployment } = await seedDeployment({
      runtime: "vllm",
      modelName: "m",
      displayName: "bye",
      port: upstream.port,
    });
    await handleDeploymentStatus(
      { deploymentId: deployment.id, status: "running", port: upstream.port },
      { hub: noopHub },
    );
    await handleDeploymentStatus({ deploymentId: deployment.id, status: "stopped" }, { hub: noopHub });

    const models = await request(makeApp()).get("/v1/models");
    expect(models.body.data).toEqual([]);
    await upstream.close();
  });

  // Rollout: agents only report a status when it CHANGES, so a deployment that
  // was already running when the manager restarted never sends another message.
  // Without reconciliation on reconnect the gateway would serve nothing at all
  // until every deployment was restarted by hand.
  it("names deployments already serving when an agent reconnects", async () => {
    const upstream = await upstreamServing("already-serving");
    const { node, deployment } = await seedDeployment({
      runtime: "vllm",
      modelName: "m",
      displayName: "guess",
      port: upstream.port,
    });
    // Already running, never named — the state a rollout finds.
    await prisma.deployment.update({
      where: { id: deployment.id },
      data: { status: "running" },
    });
    expect((await request(makeApp()).get("/v1/models")).body.data).toEqual([]);

    await publishNamesForNode(node.id);

    const models = await request(makeApp()).get("/v1/models");
    expect(models.body.data.map((m: { id: string }) => m.id)).toEqual(["already-serving"]);
    await upstream.close();
  });
});
