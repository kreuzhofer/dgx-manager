import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
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
let ollamaCatalogRouter: typeof import("../../routes/ollama-catalog.js").ollamaCatalogRouter;
let writeCatalog: typeof import("../../ollama/catalog-store.js").writeCatalog;
let writeEnabled: typeof import("../../ollama/catalog-store.js").writeEnabled;

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
  ({ ollamaCatalogRouter } = await import("../../routes/ollama-catalog.js"));
  ({ writeCatalog, writeEnabled } = await import("../../ollama/catalog-store.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  await prisma.setting.deleteMany({});
});

function makeApp() {
  const app = express();
  app.use(express.json());
  app.use("/api/ollama-catalog", ollamaCatalogRouter);
  return app;
}

describe("GET /api/ollama-catalog/catalog", () => {
  it("returns empty + null timestamp when nothing cached", async () => {
    const res = await request(makeApp()).get("/api/ollama-catalog/catalog");
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ entries: [], fetchedAt: null });
  });

  it("returns persisted entries and timestamp", async () => {
    const ts = await writeCatalog([
      { name: "llama3.1", description: "Meta Llama", type: "chat", sizes: ["8b", "70b"], capabilities: ["tools"], updatedAt: null },
    ]);
    const res = await request(makeApp()).get("/api/ollama-catalog/catalog");
    expect(res.body.fetchedAt).toBe(ts);
    expect(res.body.entries).toHaveLength(1);
    expect(res.body.entries[0].name).toBe("llama3.1");
    expect(res.body.entries[0].sizes).toEqual(["8b", "70b"]);
  });
});

describe("GET /api/ollama-catalog/enabled and PUT /api/ollama-catalog/enabled", () => {
  it("returns empty list when nothing enabled", async () => {
    const res = await request(makeApp()).get("/api/ollama-catalog/enabled");
    expect(res.body).toEqual({ enabled: [] });
  });

  it("PUT replaces the enabled list and GET reflects it", async () => {
    const app = makeApp();
    const put = await request(app)
      .put("/api/ollama-catalog/enabled")
      .send({ enabled: ["llama3.1:8b", "qwen3:8b"] });
    expect(put.status).toBe(200);
    const get = await request(app).get("/api/ollama-catalog/enabled");
    expect(get.body.enabled).toEqual(["llama3.1:8b", "qwen3:8b"]);
  });

  it("PUT rejects non-array body with 400", async () => {
    const res = await request(makeApp())
      .put("/api/ollama-catalog/enabled")
      .send({ enabled: "not-an-array" });
    expect(res.status).toBe(400);
  });
});

describe("GET /api/ollama-catalog/available", () => {
  it("flattens catalog × enabled into deployable tag rows", async () => {
    await writeCatalog([
      { name: "llama3.1", description: "A", type: "chat", sizes: ["8b", "70b"], capabilities: ["tools"], updatedAt: null },
      { name: "qwen3",   description: "B", type: "chat", sizes: ["8b", "32b"], capabilities: ["tools"], updatedAt: null },
      { name: "phi4",    description: "C", type: "chat", sizes: ["14b"], capabilities: [], updatedAt: null },
      { name: "nomic-embed-text", description: "Emb", type: "embedding", sizes: [], capabilities: ["embedding"], updatedAt: null },
    ]);
    await writeEnabled(["llama3.1:8b", "qwen3:8b", "qwen3:32b", "nomic-embed-text"]);
    const res = await request(makeApp()).get("/api/ollama-catalog/available");
    expect(res.status).toBe(200);
    const tags = (res.body as { tag: string }[]).map((r) => r.tag);
    expect(tags).toEqual(["llama3.1:8b", "qwen3:8b", "qwen3:32b", "nomic-embed-text"]);
    const llama = (res.body as { tag: string; modelName: string; size: string | null }[])
      .find((r) => r.tag === "llama3.1:8b")!;
    expect(llama.modelName).toBe("llama3.1");
    expect(llama.size).toBe("8b");
    const emb = (res.body as { tag: string; size: string | null }[])
      .find((r) => r.tag === "nomic-embed-text")!;
    expect(emb.size).toBeNull();
  });

  it("drops enabled tags that no longer exist in the catalog", async () => {
    await writeCatalog([
      { name: "llama3.1", description: "A", type: "chat", sizes: ["8b"], capabilities: [], updatedAt: null },
    ]);
    await writeEnabled(["llama3.1:8b", "llama3.1:405b", "ghost-model:1b"]);
    const res = await request(makeApp()).get("/api/ollama-catalog/available");
    expect((res.body as { tag: string }[]).map((r) => r.tag)).toEqual(["llama3.1:8b"]);
  });

  it("returns [] when nothing is enabled even if catalog is non-empty", async () => {
    await writeCatalog([
      { name: "llama3.1", description: "A", type: "chat", sizes: ["8b"], capabilities: [], updatedAt: null },
    ]);
    const res = await request(makeApp()).get("/api/ollama-catalog/available");
    expect(res.body).toEqual([]);
  });
});

describe("POST /api/ollama-catalog/catalog/refresh", () => {
  it("calls the fetcher and persists the result", async () => {
    vi.resetModules();
    vi.doMock("../../ollama/catalog-fetcher.js", () => ({
      refreshCatalog: async () => ({
        entries: [{ name: "fake", description: "stub", type: "chat", sizes: [], capabilities: [] }],
        fetchedAt: "2026-05-12T00:00:00.000Z",
      }),
    }));
    const { ollamaCatalogRouter: r } = await import("../../routes/ollama-catalog.js");
    const app = express();
    app.use(express.json());
    app.use("/api/ollama-catalog", r);
    const res = await request(app).post("/api/ollama-catalog/catalog/refresh");
    expect(res.status).toBe(200);
    expect(res.body.entries).toHaveLength(1);
    expect(res.body.entries[0].name).toBe("fake");
    vi.doUnmock("../../ollama/catalog-fetcher.js");
  });
});
