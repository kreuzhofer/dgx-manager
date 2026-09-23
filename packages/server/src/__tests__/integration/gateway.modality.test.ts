/**
 * Integration tests for modality routing.
 *
 * Same seam as gateway.proxy.test.ts: the router over HTTP, a per-test SQLite,
 * a stubbed hub for liveness, and a real ephemeral upstream — which matters
 * here because half of what is under test is a request that must NEVER reach a
 * node. An image model handed a chat completion accepts it and never answers,
 * so "did the upstream see it" is the assertion that distinguishes a working
 * guard from a hang.
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
  const { resetOutstanding } = await import("../../gateway/inflight.js");
  resetOutstanding();
  const { resetRotations } = await import("../../gateway/rotation.js");
  resetRotations();
}
afterEach(wipeAll);

interface Recorded { url: string; body: string }

function fakeUpstream() {
  const received: Recorded[] = [];
  return new Promise<{ host: string; port: number; received: Recorded[]; close: () => Promise<void> }>(
    (resolve) => {
      const server = http.createServer((req, res) => {
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
          received.push({ url: req.url ?? "", body: Buffer.concat(chunks).toString("utf8") });
          res.setHeader("content-type", "application/json");
          res.end(JSON.stringify({ data: [{ b64_json: "aGVsbG8=" }] }));
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
async function seedMember(opts: {
  publishedName: string;
  host: string;
  port: number;
  config?: string | null;
}) {
  const n = ++seq;
  const node = await prisma.node.create({
    data: { name: `node-${n}`, ipAddress: opts.host, status: "online" },
  });
  const model = await prisma.model.create({ data: { name: `catalog-${n}`, runtime: "vllm" } });
  return prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: "running",
      port: opts.port,
      publishedName: opts.publishedName,
      config: opts.config ?? null,
    },
  });
}

const IMAGE_CONFIG = JSON.stringify({ runner: "dgxrun", modality: "image" });

describe("POST /v1/images/generations", () => {
  it("forwards to an image deployment, preserving the image path upstream", async () => {
    const upstream = await fakeUpstream();
    await seedMember({
      publishedName: "Qwen/Qwen-Image-2.1",
      host: upstream.host,
      port: upstream.port,
      config: IMAGE_CONFIG,
    });

    const res = await request(makeApp())
      .post("/v1/images/generations")
      .send({ model: "Qwen/Qwen-Image-2.1", prompt: "a teapot", size: "1024x1024" });

    expect(res.status).toBe(200);
    expect(upstream.received).toHaveLength(1);
    expect(upstream.received[0].url).toBe("/v1/images/generations");
    expect(JSON.parse(upstream.received[0].body).prompt).toBe("a teapot");
    await upstream.close();
  });

  /** A text model asked for an image is refused at the gateway. The upstream
   *  assertion is the point: vLLM would 404 this path itself, but only after a
   *  round trip, and a pool of several text members would produce a different
   *  error per member. */
  it("refuses a text model and never contacts the node", async () => {
    const upstream = await fakeUpstream();
    await seedMember({ publishedName: "glm-5.2", host: upstream.host, port: upstream.port });

    const res = await request(makeApp())
      .post("/v1/images/generations")
      .send({ model: "glm-5.2", prompt: "a teapot" });

    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe("modality_mismatch");
    expect(res.body.error.message).toContain("serves text, not image");
    expect(upstream.received).toHaveLength(0);
    await upstream.close();
  });
});

describe("POST /v1/chat/completions with an image model", () => {
  /** The hang this whole feature exists to prevent: the image engine accepts a
   *  chat completion and never completes it, so the request must die here. */
  it("refuses an image model and never contacts the node", async () => {
    const upstream = await fakeUpstream();
    await seedMember({
      publishedName: "Qwen/Qwen-Image-2.1",
      host: upstream.host,
      port: upstream.port,
      config: IMAGE_CONFIG,
    });

    const res = await request(makeApp())
      .post("/v1/chat/completions")
      .send({ model: "Qwen/Qwen-Image-2.1", messages: [{ role: "user", content: "hi" }] });

    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe("modality_mismatch");
    expect(res.body.error.message).toContain("POST /v1/images/generations");
    expect(upstream.received).toHaveLength(0);
    await upstream.close();
  });

  /** Backward compatibility, the load-bearing case: every deployment that
   *  predates this field has a config blob with no modality (or none at all),
   *  and must keep serving chat exactly as before. */
  it("still forwards a deployment whose config predates the modality field", async () => {
    const upstream = await fakeUpstream();
    await seedMember({
      publishedName: "glm-5.2",
      host: upstream.host,
      port: upstream.port,
      config: JSON.stringify({ runner: "dgxrun", masterPort: 29500 }),
    });

    const res = await request(makeApp())
      .post("/v1/chat/completions")
      .send({ model: "glm-5.2", messages: [{ role: "user", content: "hi" }] });

    expect(res.status).toBe(200);
    expect(upstream.received).toHaveLength(1);
    expect(upstream.received[0].url).toBe("/v1/chat/completions");
    await upstream.close();
  });
});
