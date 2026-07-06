/**
 * Integration test for GET /api/deployments/:id/claude-launch.
 * Mirrors deployments.vram-admission.test.ts: per-suite SQLite set before
 * prisma import, only deploymentsRouter mounted, supertest, no port bind.
 * A stub fetchImpl is injected via app.set("fetchImpl", …) so served-name
 * resolution is deterministic and offline.
 */
import { afterAll, beforeAll, afterEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;

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
  ({ deploymentsRouter } = await import("../../routes/deployments.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

afterEach(async () => {
  await prisma.deployment.deleteMany();
  await prisma.model.deleteMany();
  await prisma.node.deleteMany();
});

// Stub that resolves /v1/models to a fixed served id.
function makeApp(servedId = "served-xyz") {
  const app = express();
  app.use(express.json());
  app.set("fetchImpl", async () => ({
    ok: true,
    text: async () => JSON.stringify({ data: [{ id: servedId }] }),
  }));
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

async function seedDeployment(overrides: { status?: string; port?: number | null; displayName?: string | null } = {}) {
  const node = await prisma.node.create({ data: { name: `n-${Math.random().toString(36).slice(2)}`, ipAddress: "10.0.0.5" } });
  const model = await prisma.model.create({ data: { name: `m-${Math.random().toString(36).slice(2)}`, runtime: "vllm" } });
  return prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: overrides.status ?? "running",
      port: overrides.port === undefined ? 8000 : overrides.port,
      displayName: overrides.displayName ?? null,
    },
  });
}

describe("GET /api/deployments/:id/claude-launch", () => {
  it("returns both shell snippets with the live served model name", async () => {
    const d = await seedDeployment();
    const res = await request(makeApp("live-served-name")).get(`/api/deployments/${d.id}/claude-launch`);
    expect(res.status).toBe(200);
    expect(res.body.baseUrl).toBe("http://10.0.0.5:8000");
    expect(res.body.model).toBe("live-served-name");
    expect(res.body.shells.bash).toContain("export ANTHROPIC_BASE_URL='http://10.0.0.5:8000'");
    expect(res.body.shells.bash).toContain("export ANTHROPIC_DEFAULT_OPUS_MODEL='live-served-name'");
    expect(res.body.shells.bash).not.toContain("/v1");
    expect(res.body.shells.powershell).toContain("$env:ANTHROPIC_BASE_URL = 'http://10.0.0.5:8000'");
  });

  it("404s for an unknown deployment id", async () => {
    const res = await request(makeApp()).get("/api/deployments/does-not-exist/claude-launch");
    expect(res.status).toBe(404);
  });

  it("409s when the deployment is not running", async () => {
    const d = await seedDeployment({ status: "stopped" });
    const res = await request(makeApp()).get(`/api/deployments/${d.id}/claude-launch`);
    expect(res.status).toBe(409);
  });

  it("409s when the deployment has no port", async () => {
    const d = await seedDeployment({ port: null });
    const res = await request(makeApp()).get(`/api/deployments/${d.id}/claude-launch`);
    expect(res.status).toBe(409);
  });
});
