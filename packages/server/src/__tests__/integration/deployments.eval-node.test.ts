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

describe("auto node selection skips eval nodes", () => {
  // I2 regression: idleNodes was unfiltered and orderBy name:"asc", so an eval
  // node named "agenthost" sorted before a gpu node like "dgx-1" and got
  // auto-picked for a vLLM/dgxrun deploy — which then always 400s at the role
  // guard instead of falling through to a real GPU node.
  const SOLO_DGXRUN_YAML = [
    "runner: dgxrun",
    "model: org/glm-5.2",
    "container: my-vllm-image:probe",
    "defaults:",
    "  port: 8000",
    "  host: 0.0.0.0",
    "  tensor_parallel: 1",
    "  gpu_memory_utilization: 0.5",
    "  served_model_name: glm-5.2",
    "command: |",
    "  vllm serve {model} --served-model-name {served_model_name} --host {host} --port {port} -tp {tensor_parallel} --distributed-executor-backend mp",
  ].join("\n");

  // tensor_parallel: 2 so nodeIds:"auto" sizes a 2-node cluster — keeps
  // idleNodes.slice(0, needed) at length > 1 (isCluster stays true), avoiding
  // an unrelated pre-existing quirk where a needed=1 nodeIds:"auto" resolution
  // never populates the legacy `nodeId` variable that headNodeId falls back to.
  const CLUSTER_DGXRUN_YAML = SOLO_DGXRUN_YAML.replace("tensor_parallel: 1", "tensor_parallel: 2");

  async function seedEvalAndGpuNodes(gpuCount = 1) {
    // Name ordering is load-bearing: "agenthost" < "dgx-*" alphabetically, so
    // an unfiltered auto-resolve query picks the eval node first.
    await prisma.node.create({
      data: {
        id: "evalnode", name: "agenthost", ipAddress: "192.168.44.15",
        status: "online", role: "eval", arch: "amd64", vramTotal: 0,
      },
    });
    for (let i = 1; i <= gpuCount; i++) {
      await prisma.node.create({
        data: {
          id: `gpunode${i}`, name: `dgx-${i}`, ipAddress: `192.168.44.${35 + i}`,
          status: "online", vramTotal: 122_502, arch: "arm64",
        },
      });
    }
  }

  it("nodeId:'auto' targets the gpu node for a dgxrun/vllm deploy, not the eval node", async () => {
    await wipeAll();
    await seedEvalAndGpuNodes(1);

    const res = await request(makeApp())
      .post("/api/deployments")
      .send({ nodeId: "auto", recipeYaml: SOLO_DGXRUN_YAML });

    expect(res.status).toBe(201);
    expect(res.body.nodeId).toBe("gpunode1");
  });

  it("nodeIds:'auto' sizes a cluster from the gpu nodes only, excluding the eval node", async () => {
    await wipeAll();
    await seedEvalAndGpuNodes(2);

    const res = await request(makeApp())
      .post("/api/deployments")
      .send({ nodeIds: "auto", recipeYaml: CLUSTER_DGXRUN_YAML });

    expect(res.status).toBe(201);
    expect(res.body.nodeId).toBe("gpunode1"); // head = first gpu node, never the eval node

    const clusterNodes = await prisma.clusterNode.findMany({ where: { deploymentId: res.body.id } });
    expect(clusterNodes.map((c) => c.nodeId).sort()).toEqual(["gpunode1", "gpunode2"]);
  });

  it("still lets an Ollama auto-deploy land on the eval node (role only restricts vLLM/dgxrun)", async () => {
    await wipeAll();
    await seedEvalAndGpuNodes(1);

    const res = await request(makeApp())
      .post("/api/deployments")
      .send({ nodeId: "auto", runtime: "ollama", modelName: "nomic-embed-text" });

    expect(res.status).toBe(201);
    // Either online node is a legal Ollama target — the point is it never 400s
    // because the eval node was wrongly filtered out for Ollama too.
    expect(["evalnode", "gpunode1"]).toContain(res.body.nodeId);
  });
});
