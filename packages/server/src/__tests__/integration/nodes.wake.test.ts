import { beforeAll, afterAll, afterEach, describe, it, expect, vi } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

// Per-suite SQLite. Must be set before any module that imports prisma.
const TMP_DIR = mkdtempSync(join(tmpdir(), "nodes-wake-"));
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

function makeApp(wolSend: any) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", { isAgentOnline: () => false, sendToAgent: () => {} });
  app.set("wolSend", wolSend);
  app.use("/api/nodes", nodesRouter);
  return app;
}

describe("POST /api/nodes/:id/wake", () => {
  it("sends a magic packet to the node MAC + /24 broadcast and sets powerState=waking", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41", macAddress: "aa:bb:cc:dd:ee:ff", powerState: "off" },
    });
    const wolSend = vi.fn().mockResolvedValue(undefined);
    const res = await request(makeApp(wolSend)).post(`/api/nodes/${node.id}/wake`).send();
    expect(res.status).toBe(200);
    expect(res.body.powerState).toBe("waking");
    expect(wolSend).toHaveBeenCalledWith("aa:bb:cc:dd:ee:ff", "192.168.44.255");
    const after = await prisma.node.findUnique({ where: { id: node.id } });
    expect(after?.powerState).toBe("waking");
  });

  it("returns 409 when no MAC has been captured", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41", powerState: "off" },
    });
    const res = await request(makeApp(vi.fn())).post(`/api/nodes/${node.id}/wake`).send();
    expect(res.status).toBe(409);
  });

  it("returns 404 for an unknown node", async () => {
    const res = await request(makeApp(vi.fn())).post(`/api/nodes/nope/wake`).send();
    expect(res.status).toBe(404);
  });
});
