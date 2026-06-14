import { beforeAll, afterAll, afterEach, describe, it, expect, vi } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

// Per-suite SQLite. Must be set before any module that imports prisma.
const TMP_DIR = mkdtempSync(join(tmpdir(), "nodes-power-"));
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
  await prisma.metricSnapshot.deleteMany();
  await prisma.node.deleteMany();
});

function makeApp(sshExec: any) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", { isAgentOnline: () => true, sendToAgent: () => {} });
  app.set("sshExec", sshExec);
  app.use("/api/nodes", nodesRouter);
  return app;
}

describe("POST /api/nodes/:id/power", () => {
  it("reboot: runs the systemd reboot command and sets powerState=rebooting", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const sshExec = vi.fn().mockResolvedValue({ code: 0, stdout: "", stderr: "" });
    const app = makeApp(sshExec);

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "reboot" });

    expect(res.status).toBe(200);
    expect(res.body.powerState).toBe("rebooting");
    const lastCall = sshExec.mock.calls.at(-1);
    expect(lastCall[0]).toBe("192.168.44.41");
    expect(lastCall[1]).toBe("sudo systemctl --no-block reboot");
    const after = await prisma.node.findUnique({ where: { id: node.id } });
    expect(after?.powerState).toBe("rebooting");
  });

  it("shutdown: sets powerState=off and captures MAC first", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const sshExec = vi
      .fn()
      .mockResolvedValueOnce({ code: 0, stdout: "AA:BB:CC:DD:EE:FF", stderr: "" })
      .mockResolvedValueOnce({ code: 0, stdout: "", stderr: "" });
    const app = makeApp(sshExec);

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "shutdown" });

    expect(res.status).toBe(200);
    expect(res.body.powerState).toBe("off");
    const after = await prisma.node.findUnique({ where: { id: node.id } });
    expect(after?.powerState).toBe("off");
    expect(after?.macAddress).toBe("aa:bb:cc:dd:ee:ff");
  });

  it("returns 404 for an unknown node", async () => {
    const app = makeApp(vi.fn());
    const res = await request(app).post(`/api/nodes/nope/power`).send({ action: "reboot" });
    expect(res.status).toBe(404);
  });

  it("returns 400 for an invalid action", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const app = makeApp(vi.fn());
    const res = await request(app).post(`/api/nodes/${node.id}/power`).send({ action: "explode" });
    expect(res.status).toBe(400);
  });

  it("returns 502 when the SSH command fails", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const sshExec = vi.fn().mockRejectedValue(new Error("permission denied (sudo password?)"));
    const app = makeApp(sshExec);
    const res = await request(app).post(`/api/nodes/${node.id}/power`).send({ action: "reboot" });
    expect(res.status).toBe(502);
  });
});
