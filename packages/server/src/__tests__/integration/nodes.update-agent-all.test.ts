import { beforeAll, afterAll, afterEach, describe, it, expect, vi } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync, readFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "nodes-update-all-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

const REPO_ROOT = process.cwd().replace(/\/packages\/server.*$/, "");
// The endpoint's target is the bundled agent version (agent/package.json) — read
// it the same way so "skipped" (already-current) assertions track the real value.
const AGENT_VERSION = JSON.parse(
  readFileSync(join(REPO_ROOT, "packages/agent/package.json"), "utf-8"),
).version as string;

let prisma: typeof import("../../prisma.js").prisma;
let nodesRouter: typeof import("../../routes/nodes.js").nodesRouter;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: REPO_ROOT,
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

function makeApp(onlineIds: Set<string>, sendToAgent = vi.fn()) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", { isAgentOnline: (id: string) => onlineIds.has(id), sendToAgent });
  app.use("/api/nodes", nodesRouter);
  return app;
}

describe("POST /api/nodes/update-agent-all", () => {
  it("dispatches to online outdated nodes, skips current, marks offline", async () => {
    const a = await prisma.node.create({ data: { name: "a", ipAddress: "10.0.0.1", agentVersion: "0.5.100", arch: "arm64" } });
    const b = await prisma.node.create({ data: { name: "b", ipAddress: "10.0.0.2", agentVersion: AGENT_VERSION, arch: "arm64" } });
    const c = await prisma.node.create({ data: { name: "c", ipAddress: "10.0.0.3", agentVersion: "0.5.100", arch: "amd64" } });
    const sendToAgent = vi.fn();
    // a + b online; c offline
    const res = await request(makeApp(new Set([a.id, b.id]), sendToAgent)).post("/api/nodes/update-agent-all");

    expect(res.status).toBe(200);
    expect(res.body.version).toBe(AGENT_VERSION);
    expect(res.body.dispatched.map((n: any) => n.id)).toEqual([a.id]); // online + outdated
    expect(res.body.skipped.map((n: any) => n.id)).toEqual([b.id]);    // online + current
    expect(res.body.offline.map((n: any) => n.id)).toEqual([c.id]);    // agent offline

    expect(sendToAgent).toHaveBeenCalledTimes(1);
    const [nodeId, msg] = sendToAgent.mock.calls[0];
    expect(nodeId).toBe(a.id);
    expect(msg.type).toBe("cmd:update");
    expect(msg.payload.version).toBe(AGENT_VERSION);
    expect(msg.payload.bundleUrl).toContain("?arch=arm64");
  });

  it("force=true re-dispatches to a node already on the target", async () => {
    const b = await prisma.node.create({ data: { name: "b", ipAddress: "10.0.0.2", agentVersion: AGENT_VERSION, arch: "arm64" } });
    const sendToAgent = vi.fn();
    const res = await request(makeApp(new Set([b.id]), sendToAgent)).post("/api/nodes/update-agent-all?force=true");

    expect(res.status).toBe(200);
    expect(res.body.dispatched.map((n: any) => n.id)).toEqual([b.id]);
    expect(res.body.skipped).toEqual([]);
    expect(sendToAgent).toHaveBeenCalledTimes(1);
  });
});
