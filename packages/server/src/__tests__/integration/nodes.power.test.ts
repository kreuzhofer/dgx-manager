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

// Default agent OFFLINE so these SSH-path assertions exercise the fallback.
// Agent-primary behavior is covered by the dedicated "agent channel" tests.
function makeApp(
  sshExec: any,
  opts: { online?: boolean; sendToAgent?: any } = {},
) {
  const online = opts.online ?? false;
  const sendToAgent = opts.sendToAgent ?? vi.fn();
  const app = express();
  app.use(express.json());
  app.set("agentHub", { isAgentOnline: () => online, sendToAgent });
  app.set("sshExec", sshExec);
  app.use("/api/nodes", nodesRouter);
  return app;
}

describe("POST /api/nodes/:id/power", () => {
  it("agent online: dispatches cmd:power over the WS and does not touch SSH", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const sshExec = vi.fn();
    const sendToAgent = vi.fn();
    const app = makeApp(sshExec, { online: true, sendToAgent });

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "reboot" });

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ powerState: "rebooting", via: "agent" });
    expect(sendToAgent).toHaveBeenCalledWith(node.id, {
      type: "cmd:power",
      payload: { action: "reboot", force: false },
    });
    expect(sshExec).not.toHaveBeenCalled();
    const after = await prisma.node.findUnique({ where: { id: node.id } });
    expect(after?.powerState).toBe("rebooting");
  });

  it("agent online: carries force + maps shutdown to powerState=off", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const sendToAgent = vi.fn();
    const app = makeApp(vi.fn(), { online: true, sendToAgent });

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "shutdown", force: true });

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ powerState: "off", via: "agent" });
    expect(sendToAgent).toHaveBeenCalledWith(node.id, {
      type: "cmd:power",
      payload: { action: "shutdown", force: true },
    });
  });

  it("agent online: works even without an ipAddress (no SSH needed)", async () => {
    const node = await prisma.node.create({ data: { name: "spark-test" } });
    const sendToAgent = vi.fn();
    const app = makeApp(vi.fn(), { online: true, sendToAgent });

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "reboot" });

    expect(res.status).toBe(200);
    expect(sendToAgent).toHaveBeenCalled();
  });

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
    const lastCall = sshExec.mock.calls.at(-1)!;
    expect(lastCall[0]).toBe("192.168.44.41");
    expect(lastCall[1]).toBe("sudo systemctl --no-block reboot");
    const after = await prisma.node.findUnique({ where: { id: node.id } });
    expect(after?.powerState).toBe("rebooting");
  });

  it("shutdown: captures MAC, arms WOL, then powers off with powerState=off", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    // Call order: 1) MAC capture, 2) arm WOL, 3) poweroff.
    const sshExec = vi
      .fn()
      .mockResolvedValueOnce({ code: 0, stdout: "AA:BB:CC:DD:EE:FF", stderr: "" })
      .mockResolvedValue({ code: 0, stdout: "", stderr: "" });
    const app = makeApp(sshExec);

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "shutdown" });

    expect(res.status).toBe(200);
    expect(res.body.powerState).toBe("off");
    const after = await prisma.node.findUnique({ where: { id: node.id } });
    expect(after?.powerState).toBe("off");
    expect(after?.macAddress).toBe("aa:bb:cc:dd:ee:ff");
    // WOL was armed against the node IP before the poweroff command ran.
    const cmds = sshExec.mock.calls.map((c) => c[1] as string);
    const armIdx = cmds.findIndex((c) => c.includes("ethtool -s") && c.includes("wol g"));
    const offIdx = cmds.findIndex((c) => c === "sudo systemctl --no-block poweroff");
    expect(armIdx).toBeGreaterThanOrEqual(0);
    expect(offIdx).toBeGreaterThan(armIdx);
    expect(sshExec.mock.calls[armIdx][0]).toBe("192.168.44.41");
  });

  it("shutdown still succeeds when arming WOL fails (best-effort)", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    // MAC capture ok, arm WOL rejects, poweroff ok — the arm failure must not
    // fail the request.
    const sshExec = vi
      .fn()
      .mockResolvedValueOnce({ code: 0, stdout: "AA:BB:CC:DD:EE:FF", stderr: "" })
      .mockRejectedValueOnce(new Error("ethtool: not found"))
      .mockResolvedValueOnce({ code: 0, stdout: "", stderr: "" });
    const app = makeApp(sshExec);

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "shutdown" });

    expect(res.status).toBe(200);
    expect(res.body.powerState).toBe("off");
  });

  it("force reboot: issues an immediate --force --force reboot", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const sshExec = vi.fn().mockResolvedValue({ code: 0, stdout: "", stderr: "" });
    const app = makeApp(sshExec);

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "reboot", force: true });

    expect(res.status).toBe(200);
    expect(res.body.powerState).toBe("rebooting");
    const lastCall = sshExec.mock.calls.at(-1)!;
    expect(lastCall[0]).toBe("192.168.44.41");
    expect(lastCall[1]).toBe("sudo systemctl --force --force reboot");
  });

  it("force shutdown: issues an immediate --force --force poweroff", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const sshExec = vi
      .fn()
      .mockResolvedValueOnce({ code: 0, stdout: "AA:BB:CC:DD:EE:FF", stderr: "" })
      .mockResolvedValue({ code: 0, stdout: "", stderr: "" });
    const app = makeApp(sshExec);

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "shutdown", force: true });

    expect(res.status).toBe(200);
    expect(res.body.powerState).toBe("off");
    const cmds = sshExec.mock.calls.map((c) => c[1] as string);
    expect(cmds).toContain("sudo systemctl --force --force poweroff");
  });

  it("force reboot: a severed SSH connection (hard reset) is treated as success", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    // A --force --force reboot kills the connection before the exec returns, so
    // the executor rejects with a timeout — that must NOT be a 502 for a force
    // action, and powerState must still advance to "rebooting".
    const sshExec = vi
      .fn()
      .mockRejectedValue(new Error("SSH command timed out after 8000ms"));
    const app = makeApp(sshExec);

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "reboot", force: true });

    expect(res.status).toBe(200);
    expect(res.body.powerState).toBe("rebooting");
    const after = await prisma.node.findUnique({ where: { id: node.id } });
    expect(after?.powerState).toBe("rebooting");
  });

  it("force reboot: a fast real error (sudo password) still returns 502", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const sshExec = vi
      .fn()
      .mockRejectedValue(new Error("permission denied (sudo password required)"));
    const app = makeApp(sshExec);

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "reboot", force: true });

    expect(res.status).toBe(502);
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
