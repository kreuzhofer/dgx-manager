import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
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
let writeCatalog: typeof import("../../ollama/catalog-store.js").writeCatalog;

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
  ({ writeCatalog } = await import("../../ollama/catalog-store.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
  await prisma.setting.deleteMany({});
});

function makeApp() {
  const sentMessages: { nodeId: string; message: unknown }[] = [];
  const hub = {
    getRecipes: () => [],
    getOllamaModels: () => [],   // empty: forces use of catalog
    sendToAgent: (nodeId: string, message: unknown) => {
      sentMessages.push({ nodeId, message });
    },
  };
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.use("/api/deployments", deploymentsRouter);
  return { app, sentMessages };
}

describe("POST /api/deployments with runtime=ollama uses the catalog for VRAM estimation", () => {
  it("derives vramEstimate from a catalog parameter size", async () => {
    await prisma.node.create({
      data: { id: "n1", name: "node1", status: "online", vramTotal: 128_000 },
    });
    await writeCatalog([
      { name: "llama3.1", description: "Meta", type: "chat", sizes: ["8b", "70b"], capabilities: ["tools"], updatedAt: null },
    ]);
    const { app } = makeApp();
    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "n1", runtime: "ollama", modelName: "llama3.1:8b" });
    // Adapt status check to what the route actually returns (read the existing
    // vram-admission test for the response shape — likely 200 with deployment row).
    expect([200, 201]).toContain(res.status);
    // 8b @ Q4 ≈ 8 × 0.55 × 1024 ≈ 4506 MB. Allow generous bounds.
    const persisted = await prisma.deployment.findFirst();
    expect(persisted?.vramEstimate ?? 0).toBeGreaterThan(4000);
    expect(persisted?.vramEstimate ?? 0).toBeLessThan(5000);
  });
});
