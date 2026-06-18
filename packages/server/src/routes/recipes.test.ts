/**
 * Integration test for the arch-aware recipe filter on GET /api/recipes.
 *
 * The route reads the recipe catalog from a stubbed AgentHub (in-memory) but
 * looks up the node's arch from Prisma, so this uses the per-suite SQLite
 * harness (same pattern as deployments.vram-admission.test.ts).
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../prisma.js").prisma;
let recipesRouter: typeof import("./recipes.js").recipesRouter;

const CATALOG = [
  { file: "@rtx/a", name: "A", arch: "amd64", defaults: {} },
  { file: "@official/b", name: "B", arch: "arm64", defaults: {} },
  { file: "ollama:c", name: "C", arch: "any", defaults: {} },
];

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
  ({ prisma } = await import("../prisma.js"));
  ({ recipesRouter } = await import("./recipes.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

function makeApp() {
  const app = express();
  app.use(express.json());
  app.set("agentHub", { getRecipes: () => CATALOG });
  app.use("/api/recipes", recipesRouter);
  return app;
}

describe("GET /api/recipes — arch filter", () => {
  it("returns all recipes when no nodeId is given (back-compat)", async () => {
    const app = makeApp();
    const res = await request(app).get("/api/recipes");
    expect(res.status).toBe(200);
    expect(res.body.map((r: { file: string }) => r.file)).toEqual([
      "@rtx/a",
      "@official/b",
      "ollama:c",
    ]);
  });

  it("returns only amd64 + any recipes for an amd64 node", async () => {
    await prisma.node.deleteMany({});
    await prisma.node.create({
      data: { id: "amd-node", name: "aihost01", arch: "amd64", status: "online" },
    });
    const app = makeApp();
    const res = await request(app).get("/api/recipes?nodeId=amd-node");
    expect(res.status).toBe(200);
    expect(res.body.map((r: { file: string }) => r.file)).toEqual(["@rtx/a", "ollama:c"]);
  });

  it("returns only arm64 + any recipes for an arm64 node", async () => {
    await prisma.node.deleteMany({});
    await prisma.node.create({
      data: { id: "arm-node", name: "spark-01", arch: "arm64", status: "online" },
    });
    const app = makeApp();
    const res = await request(app).get("/api/recipes?nodeId=arm-node");
    expect(res.status).toBe(200);
    expect(res.body.map((r: { file: string }) => r.file)).toEqual(["@official/b", "ollama:c"]);
  });

  it("returns the full catalog when nodeId is unknown (no node to filter by)", async () => {
    await prisma.node.deleteMany({});
    const app = makeApp();
    const res = await request(app).get("/api/recipes?nodeId=does-not-exist");
    expect(res.status).toBe(200);
    expect(res.body).toHaveLength(3);
  });
});
