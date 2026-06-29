import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

// Per-suite SQLite. Must be set before any module that imports prisma.
const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-reg-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let registriesRouter: typeof import("../../routes/registries.js").registriesRouter;

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
  ({ registriesRouter } = await import("../../routes/registries.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  await prisma.sparkrunRegistry.deleteMany();
});

import { seedDefaultRegistries } from "../../registries/seed.js";

function appWithStubHub() {
  const sent: { nodeId: string; msg: Record<string, unknown> }[] = [];
  const agentHub = {
    sendToAgent: (nodeId: string, msg: Record<string, unknown>) => sent.push({ nodeId, msg }),
    getConnectedNodeIds: () => ["node-a"],
  };
  const app = express();
  app.use(express.json());
  app.set("agentHub", agentHub);
  app.use("/api/registries", registriesRouter);
  return { app, sent };
}

describe("/api/registries", () => {
  it("POST creates, persists, and broadcasts cmd:set-registries", async () => {
    const { app, sent } = appWithStubHub();
    const res = await request(app).post("/api/registries").send({
      name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes",
    });
    expect(res.status).toBe(201);
    expect(await prisma.sparkrunRegistry.count()).toBe(1);
    expect(sent).toHaveLength(1);
    expect(sent[0].msg.type).toBe("cmd:set-registries");
  });

  it("POST rejects invalid input with 400 and does NOT broadcast", async () => {
    const { app, sent } = appWithStubHub();
    const res = await request(app).post("/api/registries").send({ name: "BAD NAME", url: "x", subpath: "" });
    expect(res.status).toBe(400);
    expect(sent).toHaveLength(0);
    expect(await prisma.sparkrunRegistry.count()).toBe(0);
  });

  it("POST rejects a duplicate name with 409", async () => {
    const { app } = appWithStubHub();
    const body = { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes" };
    await request(app).post("/api/registries").send(body);
    const res = await request(app).post("/api/registries").send(body);
    expect(res.status).toBe(409);
  });

  it("GET lists seeded registries; DELETE removes and broadcasts", async () => {
    const { app, sent } = appWithStubHub();
    await seedDefaultRegistries(prisma);
    const list = await request(app).get("/api/registries");
    expect(list.body.length).toBeGreaterThanOrEqual(7);
    const atlas = list.body.find((r: { name: string }) => r.name === "atlas");
    const del = await request(app).delete(`/api/registries/${atlas.id}`);
    expect(del.status).toBe(200);
    expect(sent.at(-1)!.msg.type).toBe("cmd:set-registries");
    expect(await prisma.sparkrunRegistry.findUnique({ where: { name: "atlas" } })).toBeNull();
  });

  it("PATCH updates fields and broadcasts", async () => {
    const { app, sent } = appWithStubHub();
    const created = await request(app).post("/api/registries").send({
      name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes",
    });
    const res = await request(app).patch(`/api/registries/${created.body.id}`).send({ visible: false });
    expect(res.status).toBe(200);
    expect(res.body.visible).toBe(false);
    expect(sent.at(-1)!.msg.type).toBe("cmd:set-registries");
  });
});
