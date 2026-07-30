/**
 * Integration tests for the gateway's inference operations.
 *
 * Same seam as gateway.models.test.ts: the router over HTTP, a per-test SQLite,
 * a stubbed hub for the liveness signal, and a REAL ephemeral upstream — which
 * matters more here, because what is under test is the bytes that reach a node
 * and the bytes that come back.
 */
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import http from "node:http";
import net from "node:net";
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
  // In-flight counts are module state in the manager process; a request left
  // hanging by one test must not make the next test's pool look busy.
  const { resetOutstanding } = await import("../../gateway/inflight.js");
  resetOutstanding();
  const { resetRotations } = await import("../../gateway/rotation.js");
  resetRotations();
}
afterEach(wipeAll);

interface Recorded {
  method: string;
  url: string;
  headers: http.IncomingHttpHeaders;
  body: string;
}

/** A node's model server that records exactly what it was sent. */
function fakeUpstream(handler?: (req: http.IncomingMessage, res: http.ServerResponse) => void) {
  const received: Recorded[] = [];
  return new Promise<{ host: string; port: number; received: Recorded[]; close: () => Promise<void> }>(
    (resolve) => {
      const server = http.createServer((req, res) => {
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c));
        req.on("end", () => {
          received.push({
            method: req.method ?? "",
            url: req.url ?? "",
            headers: req.headers,
            body: Buffer.concat(chunks).toString("utf8"),
          });
          if (handler) return handler(req, res);
          res.setHeader("content-type", "application/json");
          res.end(JSON.stringify({ ok: true, echoedBytes: Buffer.concat(chunks).byteLength }));
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

function makeApp(agentHub: unknown = { isAgentOnline: () => true }) {
  const app = express();
  app.set("agentHub", agentHub);
  app.use("/v1", gatewayRouter);
  return app;
}

let seq = 0;
async function seedMember(opts: {
  publishedName: string | null;
  host: string;
  port: number;
  status?: string;
  runtime?: string;
}) {
  const n = ++seq;
  const node = await prisma.node.create({
    data: { name: `node-${n}`, ipAddress: opts.host, status: "online" },
  });
  const model = await prisma.model.create({
    data: { name: `catalog-${n}`, runtime: opts.runtime ?? "vllm" },
  });
  const deployment = await prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: opts.status ?? "running",
      port: opts.port,
      publishedName: opts.publishedName,
    },
  });
  return { node, deployment };
}

describe("POST /v1/chat/completions", () => {
  it("forwards to the deployment serving the requested name", async () => {
    const upstream = await fakeUpstream();
    await seedMember({ publishedName: "glm-5.2", host: upstream.host, port: upstream.port });

    const res = await request(makeApp())
      .post("/v1/chat/completions")
      .send({ model: "glm-5.2", messages: [{ role: "user", content: "hi" }] });

    expect(res.status).toBe(200);
    expect(res.body.ok).toBe(true);
    expect(upstream.received).toHaveLength(1);
    expect(upstream.received[0].url).toBe("/v1/chat/completions");
    expect(JSON.parse(upstream.received[0].body).model).toBe("glm-5.2");
    await upstream.close();
  });

  it("routes each name to its own deployment", async () => {
    const a = await fakeUpstream();
    const b = await fakeUpstream();
    await seedMember({ publishedName: "model-a", host: a.host, port: a.port });
    await seedMember({ publishedName: "model-b", host: b.host, port: b.port });

    await request(makeApp()).post("/v1/chat/completions").send({ model: "model-b" });

    expect(a.received).toEqual([]);
    expect(b.received).toHaveLength(1);
    await a.close();
    await b.close();
  });

  it("refuses an unknown name without contacting any node", async () => {
    const upstream = await fakeUpstream();
    await seedMember({ publishedName: "glm-5.2", host: upstream.host, port: upstream.port });

    const res = await request(makeApp())
      .post("/v1/chat/completions")
      .send({ model: "no-such-model" });

    expect(res.status).toBe(404);
    expect(res.body.error).toMatchObject({ code: "model_not_found" });
    expect(upstream.received).toEqual([]);
    await upstream.close();
  });

  it("requires a model field", async () => {
    const res = await request(makeApp()).post("/v1/chat/completions").send({ messages: [] });
    expect(res.status).toBe(400);
    expect(res.body.error).toMatchObject({ code: "missing_model" });
  });

  it("rejects a body that is not JSON", async () => {
    const res = await request(makeApp())
      .post("/v1/chat/completions")
      .set("content-type", "application/json")
      .send("{not json");
    expect(res.status).toBe(400);
  });

  // Eligibility, end to end: the name exists but nothing can serve it.
  it("refuses when the only member's node has no live agent connection", async () => {
    const upstream = await fakeUpstream();
    await seedMember({ publishedName: "glm-5.2", host: upstream.host, port: upstream.port });

    const res = await request(makeApp({ isAgentOnline: () => false }))
      .post("/v1/chat/completions")
      .send({ model: "glm-5.2" });

    expect(res.status).toBe(503);
    expect(res.body.error).toMatchObject({ code: "no_eligible_member" });
    expect(upstream.received).toEqual([]);
    await upstream.close();
  });

  it("skips an ineligible member and uses the eligible one", async () => {
    const dead = await fakeUpstream();
    const live = await fakeUpstream();
    const { node: deadNode } = await seedMember({
      publishedName: "pooled",
      host: dead.host,
      port: dead.port,
    });
    await seedMember({ publishedName: "pooled", host: live.host, port: live.port });

    const res = await request(makeApp({ isAgentOnline: (id: string) => id !== deadNode.id }))
      .post("/v1/chat/completions")
      .send({ model: "pooled" });

    expect(res.status).toBe(200);
    expect(dead.received).toEqual([]);
    expect(live.received).toHaveLength(1);
    await dead.close();
    await live.close();
  });

  it("answers 502 when the member cannot be reached at all", async () => {
    const upstream = await fakeUpstream();
    const port = upstream.port;
    await upstream.close(); // nothing is listening there now
    await seedMember({ publishedName: "glm-5.2", host: "127.0.0.1", port });

    const res = await request(makeApp())
      .post("/v1/chat/completions")
      .send({ model: "glm-5.2" });

    expect(res.status).toBe(502);
    expect(res.body.error).toMatchObject({ code: "upstream_unreachable" });
  });
});

describe("pools", () => {
  // Counts return to zero between sequential requests, so every member is tied
  // and the rotation spreads the work instead of hammering whichever sorts
  // first. Without it a two-member pool would leave one member permanently idle.
  it("spreads sequential requests across a pool", async () => {
    const a = await fakeUpstream();
    const b = await fakeUpstream();
    await seedMember({ publishedName: "pooled", host: a.host, port: a.port });
    await seedMember({ publishedName: "pooled", host: b.host, port: b.port });
    const app = makeApp();

    for (let i = 0; i < 4; i++) {
      await request(app).post("/v1/chat/completions").send({ model: "pooled" });
    }

    expect(a.received.length + b.received.length).toBe(4);
    expect(a.received.length).toBeGreaterThan(0);
    expect(b.received.length).toBeGreaterThan(0);
    await a.close();
    await b.close();
  });

  // The point of least-outstanding: a member already holding a request is
  // passed over for an idle one, which is how a slow member stops attracting
  // traffic without anyone configuring a weight.
  // Deliberately distinguishes least-outstanding from plain round-robin: with
  // one member holding a request open, EVERY subsequent request must go to the
  // idle one. Round-robin would keep dealing turns back to the busy member.
  it("keeps away from a busy member while an idle one exists", async () => {
    // Held in an object so TypeScript does not narrow it to null — the only
    // assignment happens inside the upstream handler.
    const held: { release: (() => void) | null } = { release: null };
    // The first request to arrive anywhere is held open; everything after is
    // answered immediately.
    const handler = (_req: http.IncomingMessage, res: http.ServerResponse) => {
      const respond = () => {
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ ok: true }));
      };
      if (!held.release) held.release = respond;
      else respond();
    };
    const a = await fakeUpstream(handler);
    const b = await fakeUpstream(handler);
    await seedMember({ publishedName: "pooled", host: a.host, port: a.port });
    await seedMember({ publishedName: "pooled", host: b.host, port: b.port });
    const app = makeApp();

    // .then() is what actually dispatches a supertest request; without it this
    // would never leave the starting line.
    const inFlight = request(app)
      .post("/v1/chat/completions")
      .send({ model: "pooled" })
      .then(() => undefined, () => undefined);
    await new Promise((r) => setTimeout(r, 250));

    const busy = a.received.length > 0 ? a : b;
    const idle = busy === a ? b : a;
    expect(busy.received).toHaveLength(1);

    for (let i = 0; i < 3; i++) {
      await request(app).post("/v1/chat/completions").send({ model: "pooled" });
    }

    expect(idle.received).toHaveLength(3);
    expect(busy.received).toHaveLength(1); // never went back to the busy one

    held.release?.();
    await inFlight;
    await a.close();
    await b.close();
  });

  // A leaked increment would permanently make a healthy member look busy and
  // quietly take it out of rotation.
  it("releases the count when a request fails, so the member stays usable", async () => {
    const dead = await fakeUpstream();
    const deadPort = dead.port;
    await dead.close();
    await seedMember({ publishedName: "flaky", host: "127.0.0.1", port: deadPort });
    const app = makeApp();

    const failed = await request(app).post("/v1/chat/completions").send({ model: "flaky" });
    expect(failed.status).toBe(502);

    // If the count had leaked the member would still be selectable, but its
    // count would be wrong forever; assert the accounting directly.
    const { outstandingSnapshot } = await import("../../gateway/inflight.js");
    expect(outstandingSnapshot()).toEqual({});
  });
});

describe("refusing with reasons", () => {
  // A bare "no active endpoints" cannot distinguish a node that is offline from
  // a deployment still starting — the difference between waiting and
  // investigating.
  it("names each member and why it was passed over", async () => {
    const upstream = await fakeUpstream();
    const { node: offlineNode } = await seedMember({
      publishedName: "unreachable",
      host: upstream.host,
      port: upstream.port,
    });
    const { node: startingNode } = await seedMember({
      publishedName: "unreachable",
      host: "10.0.0.77",
      port: 8000,
      status: "loading",
    });

    const res = await request(makeApp({ isAgentOnline: () => false }))
      .post("/v1/chat/completions")
      .send({ model: "unreachable" });

    expect(res.status).toBe(503);
    expect(res.body.error.code).toBe("no_eligible_member");

    const members = res.body.error.members as Array<{ node: string; reason: string }>;
    expect(members).toHaveLength(2);
    const byNode = Object.fromEntries(members.map((m) => [m.node, m.reason]));
    expect(byNode[startingNode.name]).toBe("not-running");
    expect(byNode[offlineNode.name]).toBe("agent-offline");

    // The message alone has to be enough when read from a terminal.
    expect(res.body.error.message).toContain(offlineNode.name);
    expect(res.body.error.message).toMatch(/no live agent connection/);
    await upstream.close();
  });
});

describe("POST /v1/embeddings", () => {
  // The operation that makes an allocation-inducing runtime's deployment
  // reachable from the network at all — the firewall admits only the manager.
  it("forwards an embedding request to its deployment", async () => {
    const upstream = await fakeUpstream((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ object: "list", data: [{ embedding: [0.1, 0.2] }] }));
    });
    await seedMember({
      publishedName: "qwen3-embedding:8b",
      host: upstream.host,
      port: upstream.port,
      runtime: "ollama",
    });

    const res = await request(makeApp())
      .post("/v1/embeddings")
      .send({ model: "qwen3-embedding:8b", input: "hello" });

    expect(res.status).toBe(200);
    expect(res.body.data[0].embedding).toEqual([0.1, 0.2]);
    expect(upstream.received[0].url).toBe("/v1/embeddings");
    await upstream.close();
  });
});

describe("what the gateway does and does not carry", () => {
  // The regression that motivated building this on node:http at all: the
  // management API's JSON parser caps bodies at 100kb, and a long-context
  // prompt is megabytes.
  it("carries a body far larger than the management API's parser limit", async () => {
    const upstream = await fakeUpstream();
    await seedMember({ publishedName: "glm-5.2", host: upstream.host, port: upstream.port });
    const huge = "x".repeat(3_000_000);

    const res = await request(makeApp())
      .post("/v1/chat/completions")
      .set("content-type", "application/json")
      .send(JSON.stringify({ model: "glm-5.2", prompt: huge }));

    expect(res.status).toBe(200);
    const delivered = JSON.parse(upstream.received[0].body);
    expect(delivered.prompt).toHaveLength(3_000_000);
    expect(delivered.prompt).toBe(huge);
    await upstream.close();
  });

  // Regression: an HTTP/1.1 client may send an absolute-form request target
  // (`POST http://host/v1/...`). Express exposes it verbatim on originalUrl
  // while still routing by pathname, so deriving the forward target from it let
  // a caller replace the chosen member's host and port — and the manager is the
  // one host the agent firewall admits to Ollama, making the gateway a confused
  // deputy against every node. The target must come from a constant.
  it("ignores an absolute-form request target and still forwards to the chosen member", async () => {
    const intended = await fakeUpstream();
    const attacker = await fakeUpstream();
    await seedMember({
      publishedName: "glm-5.2",
      host: intended.host,
      port: intended.port,
    });

    const app = makeApp();
    const server = app.listen(0);
    const addr = server.address() as { port: number };
    const payload = JSON.stringify({ model: "glm-5.2" });

    const status: number = await new Promise((resolve, reject) => {
      const sock = net.connect(addr.port, "127.0.0.1", () => {
        sock.write(
          `POST http://127.0.0.1:${attacker.port}/v1/chat/completions HTTP/1.1\r\n` +
            `Host: 127.0.0.1\r\n` +
            `Content-Type: application/json\r\n` +
            `Content-Length: ${Buffer.byteLength(payload)}\r\n\r\n${payload}`,
        );
      });
      let buf = "";
      sock.on("data", (d) => { buf += d.toString(); });
      sock.on("error", reject);
      setTimeout(() => {
        sock.end();
        resolve(Number(buf.split(" ")[1] ?? 0));
      }, 700);
    });
    server.close();

    expect(status).toBe(200);
    expect(intended.received).toHaveLength(1);
    expect(attacker.received).toEqual([]); // never redirected
    await intended.close();
    await attacker.close();
  });

  // The cap exists so an oversized request fails loudly instead of being
  // accumulated in manager memory. Exercised through the router with a small
  // limit rather than by actually moving 64MB.
  it("refuses a body beyond the configured maximum", async () => {
    const upstream = await fakeUpstream();
    await seedMember({ publishedName: "glm-5.2", host: upstream.host, port: upstream.port });

    const app = makeApp();
    app.set("gatewayMaxBodyBytes", 1024);

    const res = await request(app)
      .post("/v1/chat/completions")
      .set("content-type", "application/json")
      .send(JSON.stringify({ model: "glm-5.2", prompt: "x".repeat(5000) }));

    expect(res.status).toBe(413);
    expect(res.body.error).toMatchObject({ code: "request_too_large" });
    expect(upstream.received).toEqual([]);
    await upstream.close();
  });

  // Guards against anyone reintroducing a client-side timeout: a slow first
  // byte is normal for a long prefill and must not be cut off.
  it("waits for an upstream that is slow to answer", async () => {
    const upstream = await fakeUpstream((_req, res) => {
      setTimeout(() => {
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ ok: true, slow: true }));
      }, 1500);
    });
    await seedMember({ publishedName: "glm-5.2", host: upstream.host, port: upstream.port });

    const res = await request(makeApp())
      .post("/v1/chat/completions")
      .send({ model: "glm-5.2" });

    expect(res.status).toBe(200);
    expect(res.body.slow).toBe(true);
    await upstream.close();
  });

  // A client's key is meaningless to a node and forwarding it is worse than
  // ignoring it, so it must be dropped rather than relayed.
  it("accepts an authorization header and never forwards it", async () => {
    const upstream = await fakeUpstream();
    await seedMember({ publishedName: "glm-5.2", host: upstream.host, port: upstream.port });

    const res = await request(makeApp())
      .post("/v1/chat/completions")
      .set("authorization", "Bearer sk-super-secret")
      .set("cookie", "session=abc")
      .send({ model: "glm-5.2" });

    expect(res.status).toBe(200);
    expect(upstream.received[0].headers.authorization).toBeUndefined();
    expect(upstream.received[0].headers.cookie).toBeUndefined();
    await upstream.close();
  });

  // A streamed completion must arrive as it is produced; buffering it to
  // completion would turn an incremental UI into one long pause.
  it("streams a response through instead of buffering it", async () => {
    const upstream = await fakeUpstream((_req, res) => {
      res.setHeader("content-type", "text/event-stream");
      res.write("data: {\"delta\":\"one\"}\n\n");
      setTimeout(() => res.write("data: {\"delta\":\"two\"}\n\n"), 30);
      setTimeout(() => res.end("data: [DONE]\n\n"), 60);
    });
    await seedMember({ publishedName: "glm-5.2", host: upstream.host, port: upstream.port });

    // Driven over a real socket rather than supertest, so the response can be
    // observed arriving in pieces instead of as one collected buffer.
    const arrivals: number[] = [];
    const started = Date.now();
    const app = makeApp();
    const server = app.listen(0);
    const addr = server.address() as { port: number };
    const chunks: string[] = [];
    await new Promise<void>((resolve, reject) => {
      const req = http.request(
        {
          hostname: "127.0.0.1",
          port: addr.port,
          path: "/v1/chat/completions",
          method: "POST",
          headers: { "content-type": "application/json" },
        },
        (res) => {
          res.on("data", (c: Buffer) => {
            chunks.push(c.toString());
            arrivals.push(Date.now() - started);
          });
          res.on("end", () => resolve());
          res.on("error", reject);
        },
      );
      req.on("error", reject);
      req.end(JSON.stringify({ model: "glm-5.2", stream: true }));
    });
    server.close();

    expect(chunks.join("")).toContain("[DONE]");
    // Not merely "more than one chunk" — a proxy that buffered to completion
    // and re-emitted in two writes would pass that. The first event must land
    // before the upstream has written its last (at ~60ms).
    expect(arrivals.length).toBeGreaterThan(1);
    expect(arrivals[0]).toBeLessThan(60);
    await upstream.close();
  });
});
