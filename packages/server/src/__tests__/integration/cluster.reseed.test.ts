import { beforeAll, afterAll, afterEach, describe, it, expect, vi } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "cluster-reseed-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let clusterRouter: typeof import("../../routes/cluster.js").clusterRouter;
let resetKnownHostsGuard: typeof import("../../ssh/known-hosts.js").resetKnownHostsGuard;

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
  ({ clusterRouter } = await import("../../routes/cluster.js"));
  ({ resetKnownHostsGuard } = await import("../../ssh/known-hosts.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

afterEach(async () => {
  resetKnownHostsGuard();
  await prisma.node.deleteMany();
});

function makeApp(sshExec: any) {
  const app = express();
  app.use(express.json());
  app.set("sshExec", sshExec);
  app.use("/api/cluster", clusterRouter);
  return app;
}

const IP_OUT = "2: e0 inet 192.168.44.41/24 scope global\n3: e1 inet 192.168.100.41/24 scope global";

describe("POST /api/cluster/reseed-known-hosts", () => {
  it("200 with a per-node report when nodes are seeded", async () => {
    await prisma.node.create({ data: { name: "n1", ipAddress: "192.168.44.41", status: "online" } });
    const sshExec = vi.fn(async (_host: string, command: string) =>
      command.includes("ip -o -4 addr") ? { code: 0, stdout: IP_OUT, stderr: "" } : { code: 0, stdout: "", stderr: "" },
    );
    const res = await request(makeApp(sshExec)).post("/api/cluster/reseed-known-hosts").send({});
    expect(res.status).toBe(200);
    expect(res.body.perNode).toHaveLength(1);
    expect(res.body.perNode[0].ok).toBe(true);
    expect(res.body.trustedIps).toContain("192.168.44.41");
  });

  it("502 when every node is unreachable (gather fails, no IPs discovered)", async () => {
    await prisma.node.create({ data: { name: "n1", ipAddress: "192.168.44.41", status: "online" } });
    const sshExec = vi.fn(async () => {
      throw new Error("unreachable");
    });
    const res = await request(makeApp(sshExec)).post("/api/cluster/reseed-known-hosts").send({});
    expect(res.status).toBe(502);
    // When gather fails for all nodes no IPs are discovered, so the seed phase
    // is skipped entirely and perNode is empty.
    expect(res.body.perNode).toHaveLength(0);
    expect(res.body.trustedIps).toHaveLength(0);
  });
});
