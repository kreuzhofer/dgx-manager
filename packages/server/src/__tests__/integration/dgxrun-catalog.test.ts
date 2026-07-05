/**
 * Integration test: GET /api/recipes merges the in-repo @dgxrun catalog
 * (packages/server/src/deployments/dgxrun-catalog.ts) with whatever agents
 * report via agentHub.getRecipes().
 *
 * Mirrors the bootstrap in deployments.dgxrun.test.ts: per-suite sqlite via
 * mkdtempSync + DATABASE_URL set BEFORE importing prisma, schema applied via
 * `npx prisma db push --force-reset`.
 *
 * DGXRUN_RECIPES_DIR is pointed at the repo's real recipes/dgxrun directory
 * (resolved relative to this test file) BEFORE importing dgxrun-catalog.ts /
 * recipes.ts, so getDgxrunCatalog() picks it up on first read.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-recipes-catalog-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

// Repo root is four levels up from this file:
// packages/server/src/__tests__/integration/dgxrun-catalog.test.ts
const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "../../../../..");
process.env.DGXRUN_RECIPES_DIR = join(REPO_ROOT, "recipes/dgxrun");

let prisma: typeof import("../../prisma.js").prisma;
let recipesRouter: typeof import("../../routes/recipes.js").recipesRouter;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      DATABASE_URL: `file:${DB_PATH}`,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "User consented to db push --force-reset against per-suite SQLite test databases in /tmp on 2026-07-05 (option #1)",
    },
    stdio: "pipe",
  });
  ({ prisma } = await import("../../prisma.js"));
  ({ recipesRouter } = await import("../../routes/recipes.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

function makeStubHub() {
  return {
    getRecipes: () => [],
    getOllamaModels: () => [],
    getConnectedNodeIds: () => [] as string[],
    sendToAgent: () => {},
  };
}

function makeApp(hub: unknown) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.use("/api/recipes", recipesRouter);
  return app;
}

describe("GET /api/recipes includes @dgxrun catalog", () => {
  it("lists the @dgxrun/glm-5.2-awq-15pct recipe with source dgxrun", async () => {
    const app = makeApp(makeStubHub());

    const res = await request(app).get("/api/recipes");

    expect(res.status).toBe(200);
    const hit = res.body.find((r: any) => r.file === "@dgxrun/glm-5.2-awq-15pct");
    expect(hit).toBeTruthy();
    expect(hit.source).toBe("dgxrun");
  });
});
