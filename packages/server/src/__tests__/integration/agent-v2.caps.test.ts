/**
 * Integration tests for the Agent v2 capability REST endpoints (Task 12):
 *
 *   - POST /api/nodes/:id/diag  -> invokes the "diag.collect" capability via
 *     CapClient and returns its bundle.
 *   - POST /api/nodes/:id/exec  -> requires a non-blank `reason` (audited),
 *     else invokes the "exec" capability and buffers its streamed
 *     stdout/stderr chunks into the response as `result.output`.
 *
 * Both routes short-circuit to 503 (without touching CapClient) when the
 * target node's agent is offline, so a dead node fails fast instead of
 * blocking for the ~30s CapClient timeout.
 *
 * Mirrors the supertest + stub-dependency harness used by
 * nodes.power.test.ts / deployments.dgxrun.test.ts: only the nodes router is
 * mounted, and stub capClient/agentHub are injected via
 * app.set("capClient"/"agentHub", ...) so no real agent WebSocket is needed.
 */
import { beforeAll, afterAll, afterEach, describe, it, expect } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

// Per-suite SQLite. Must be set before any module that imports prisma.
const TMP_DIR = mkdtempSync(join(tmpdir(), "agent-v2-caps-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

// Dynamic imports so the env var above is in place before prisma loads.
let prisma: typeof import("../../prisma.js").prisma;
let nodesRouter: typeof import("../../routes/nodes.js").nodesRouter;

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
  ({ nodesRouter } = await import("../../routes/nodes.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

afterEach(async () => {
  await prisma.node.deleteMany();
});

// Default agentHub stub: agent online, so tests exercise the capClient path
// unless a test overrides it to prove the offline fast-path (503).
function makeApp(capClient: unknown, agentHub: unknown = { isAgentOnline: () => true }) {
  const app = express();
  app.use(express.json());
  app.set("capClient", capClient);
  app.set("agentHub", agentHub);
  app.use("/api/nodes", nodesRouter);
  return app;
}

describe("POST /api/nodes/:id/diag + /exec", () => {
  it("diag returns the agent bundle", async () => {
    const node = await prisma.node.create({ data: { name: "spark-diag", ipAddress: "192.168.44.60" } });
    const capClient = { invoke: async () => ({ ok: true, data: { memory: { totalMb: 124546 } } }) };
    const app = makeApp(capClient);

    const res = await request(app).post(`/api/nodes/${node.id}/diag`).send({});

    expect(res.status).toBe(200);
    expect(res.body.ok).toBe(true);
    expect(res.body.data.memory.totalMb).toBe(124546);
  });

  it("diag returns 502 when the capability invocation fails", async () => {
    const node = await prisma.node.create({ data: { name: "spark-diag-fail", ipAddress: "192.168.44.61" } });
    const capClient = { invoke: async () => ({ ok: false, error: "cap timeout" }) };
    const app = makeApp(capClient);

    const res = await request(app).post(`/api/nodes/${node.id}/diag`).send({});

    expect(res.status).toBe(502);
    expect(res.body.error).toMatch(/timeout/i);
  });

  it("diag returns 503 without invoking the capability when the agent is offline", async () => {
    const node = await prisma.node.create({ data: { name: "spark-diag-offline", ipAddress: "192.168.44.64" } });
    let invoked = false;
    const capClient = { invoke: async () => { invoked = true; return { ok: true, data: {} }; } };
    const app = makeApp(capClient, { isAgentOnline: () => false });

    const res = await request(app).post(`/api/nodes/${node.id}/diag`).send({});

    expect(res.status).toBe(503);
    expect(res.body.error).toMatch(/offline/i);
    expect(invoked).toBe(false);
  });

  it("exec requires a reason", async () => {
    const node = await prisma.node.create({ data: { name: "spark-exec", ipAddress: "192.168.44.62" } });
    const app = makeApp({ invoke: async () => ({ ok: true, data: {} }) });

    const res = await request(app).post(`/api/nodes/${node.id}/exec`).send({ cmd: "ls" });

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/reason/i);
  });

  it("exec returns 503 without invoking the capability when the agent is offline", async () => {
    const node = await prisma.node.create({ data: { name: "spark-exec-offline", ipAddress: "192.168.44.65" } });
    let invoked = false;
    const capClient = { invoke: async () => { invoked = true; return { ok: true, data: { code: 0, timedOut: false } }; } };
    const app = makeApp(capClient, { isAgentOnline: () => false });

    const res = await request(app)
      .post(`/api/nodes/${node.id}/exec`)
      .send({ cmd: "echo", args: ["hi"], reason: "debugging a stuck deploy" });

    expect(res.status).toBe(503);
    expect(res.body.error).toMatch(/offline/i);
    expect(invoked).toBe(false);
  });

  it("exec invokes the capability and returns its buffered output when a reason is given", async () => {
    const node = await prisma.node.create({ data: { name: "spark-exec-ok", ipAddress: "192.168.44.63" } });
    let seen: { nodeId: string; name: string; input: unknown } | null = null;
    // The real "exec" capability (packages/agent/src/caps/exec-cap.ts) resolves
    // { code, timedOut } and streams stdout/stderr separately as onChunk calls
    // — it never returns an "output" field directly. This stub mirrors that
    // real shape (unlike the previous version of this test, which stubbed a
    // fabricated { code, output } result that masked the route dropping the
    // streamed chunks) so the assertion below proves the route's onChunk
    // buffering wiring, not just pass-through of a convenient stub.
    const capClient = {
      invoke: async (
        nodeId: string,
        name: string,
        input: unknown,
        onChunk?: (c: { stream: string; data: string }) => void,
      ) => {
        seen = { nodeId, name, input };
        onChunk?.({ stream: "stdout", data: "hello" });
        return { ok: true, data: { code: 0, timedOut: false } };
      },
    };
    const app = makeApp(capClient);

    const res = await request(app)
      .post(`/api/nodes/${node.id}/exec`)
      .send({ cmd: "echo", args: ["hi"], reason: "debugging a stuck deploy" });

    expect(res.status).toBe(200);
    expect(res.body.ok).toBe(true);
    expect(res.body.result.code).toBe(0);
    expect(res.body.result.timedOut).toBe(false);
    expect(res.body.result.output).toContain("hello");
    expect(seen).toEqual({
      nodeId: node.id,
      name: "exec",
      input: { cmd: "echo", args: ["hi"], reason: "debugging a stuck deploy", timeoutMs: undefined },
    });
  });
});
