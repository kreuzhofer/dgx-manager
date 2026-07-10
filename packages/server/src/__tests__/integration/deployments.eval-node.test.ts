import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-evalnode-test-"));
process.env.DATABASE_URL = `file:${join(TMP_DIR, "test.db")}`;

let prisma: typeof import("../../prisma.js").prisma;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "I understand this is a test database and consent to it being reset",
    },
    stdio: "ignore",
  });
  ({ prisma } = await import("../../prisma.js"));
  ({ deploymentsRouter } = await import("../../routes/deployments.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

function makeApp() {
  const app = express();
  app.use(express.json());
  app.set("agentHub", {
    getRecipes: () => [],
    getTrainingRecipes: () => [],
    getOllamaModels: () => [{ name: "nomic-embed-text", size: "274MB", description: "" }],
    isAgentOnline: () => true,
    onlineNodeIds: () => [] as string[],
    sendToAgent: () => {},
  });
  app.set("sshExec", async () => ({ code: 0, stdout: "false", stderr: "" }));
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

async function wipeAll() {
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
}

describe("eval-node deploy admission", () => {
  it("refuses a vllm deployment onto an eval node with 400", async () => {
    await wipeAll();
    await prisma.node.create({
      data: {
        id: "evalnode", name: "agenthost", ipAddress: "192.168.44.15",
        status: "online", role: "eval", arch: "amd64",
      },
    });

    const res = await request(makeApp())
      .post("/api/deployments")
      .send({ nodeId: "evalnode", recipeYaml: "runner: dgxrun\ncommand: vllm serve x\ncontainer: img\n" });

    expect(res.status).toBe(400);
    expect(res.body.error).toContain("agenthost");
    expect(res.body.error).toContain("ollama");
  });

  it("permits an ollama deployment onto an eval node", async () => {
    await wipeAll();
    await prisma.node.create({
      data: {
        id: "evalnode", name: "agenthost", ipAddress: "192.168.44.15",
        status: "online", role: "eval", arch: "amd64", vramTotal: 0,
      },
    });

    const res = await request(makeApp())
      .post("/api/deployments")
      .send({ nodeId: "evalnode", runtime: "ollama", modelName: "nomic-embed-text" });

    expect(res.status).toBe(201);
  });

  // The role only restricts. A normal node is unaffected.
  it("leaves a gpu node unrestricted", async () => {
    await wipeAll();
    await prisma.node.create({
      data: {
        id: "gpunode", name: "dgx-1", ipAddress: "192.168.44.36",
        status: "online", vramTotal: 122_502, arch: "arm64",
      },
    });
    const res = await request(makeApp())
      .post("/api/deployments")
      .send({ nodeId: "gpunode", recipeYaml: "runner: dgxrun\ncommand: vllm serve x\ncontainer: img\n" });

    expect(res.status).not.toBe(400);
  });
});
