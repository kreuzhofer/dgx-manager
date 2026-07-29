/**
 * Integration test for the hub's `agent:deployment:status` path — the first
 * test in the suite to drive it at all.
 *
 * Every other suite stubs the AgentHub, so the step where a deployment that has
 * started serving gets its **published name** resolved and persisted had no
 * coverage: the resolver could be perfect and the handler could simply never
 * call it. That is the integration seam this closes.
 *
 * Follows the per-suite-SQLite harness established in
 * deployments.vram-admission.test.ts: DATABASE_URL set before importing prisma,
 * schema applied via `prisma db push --force-reset` with explicit AI-action
 * consent, wipeAll() between tests.
 */
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

// Per-suite SQLite. Must be set before any module that imports prisma.
const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

// Dynamic imports so the env var above is in place before prisma loads.
let prisma: typeof import("../../prisma.js").prisma;
let handleDeploymentStatus: typeof import("../../ws/deployment-status-handler.js").handleDeploymentStatus;

beforeAll(async () => {
  // PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION: see the note in
  // deployments.vram-admission.test.ts — DATABASE_URL here always points at a
  // freshly-mkdtemp'd SQLite file in /tmp.
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
  ({ handleDeploymentStatus } = await import("../../ws/deployment-status-handler.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

/** Wipe all tables in FK-dependency order so the next test starts clean. */
async function wipeAll() {
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.metricSnapshot.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
}
afterEach(wipeAll);

/** Answers an OpenAI `/v1/models` request with one served id. */
function servingFetch(id: string) {
  return vi.fn(async () =>
    new Response(JSON.stringify({ data: [{ id }] }), { status: 200 }),
  ) as unknown as typeof fetch;
}

async function seedDeployment(opts: {
  runtime: string;
  modelName: string;
  displayName?: string | null;
  status?: string;
  port?: number | null;
}) {
  const node = await prisma.node.create({
    data: { name: "n1", ipAddress: "10.0.0.1", status: "online" },
  });
  const model = await prisma.model.create({
    data: { name: opts.modelName, runtime: opts.runtime },
  });
  return prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: opts.status ?? "loading",
      port: opts.port === undefined ? 8000 : opts.port,
      displayName: opts.displayName ?? null,
    },
  });
}

describe("agent:deployment:status — publishing a name", () => {
  it("publishes the name a pinned runtime reports once it is serving", async () => {
    const d = await seedDeployment({
      runtime: "vllm",
      modelName: "@dgxrun/glm-5.2-long",
      displayName: "glm-5.2",
    });

    await handleDeploymentStatus(
      { deploymentId: d.id, status: "running", port: 8000 },
      { fetchImpl: servingFetch("glm-5.2-served") },
    );

    const after = await prisma.deployment.findUnique({ where: { id: d.id } });
    expect(after?.status).toBe("running");
    expect(after?.publishedName).toBe("glm-5.2-served");
  });

  it("publishes an allocation-inducing runtime's model tag without probing", async () => {
    const d = await seedDeployment({
      runtime: "ollama",
      modelName: "qwen3-embedding:8b",
      port: 11434,
    });
    const probe = servingFetch("some-other-model-on-the-node");

    await handleDeploymentStatus(
      { deploymentId: d.id, status: "running", port: 11434 },
      { fetchImpl: probe },
    );

    const after = await prisma.deployment.findUnique({ where: { id: d.id } });
    expect(after?.publishedName).toBe("qwen3-embedding:8b");
    expect(probe).not.toHaveBeenCalled();
  });

  it("does not publish a name for a deployment that is not serving", async () => {
    const d = await seedDeployment({ runtime: "vllm", modelName: "m" });

    await handleDeploymentStatus(
      { deploymentId: d.id, status: "loading" },
      { fetchImpl: servingFetch("nope") },
    );

    const after = await prisma.deployment.findUnique({ where: { id: d.id } });
    expect(after?.publishedName).toBeNull();
  });

  // A published name describes one serving lifetime. Left in place, a stopped
  // deployment would keep appearing in what the cluster publishes.
  it("clears the published name when the deployment stops", async () => {
    const d = await seedDeployment({ runtime: "vllm", modelName: "m", displayName: "alias" });
    await handleDeploymentStatus(
      { deploymentId: d.id, status: "running", port: 8000 },
      { fetchImpl: servingFetch("alias") },
    );
    expect((await prisma.deployment.findUnique({ where: { id: d.id } }))?.publishedName).toBe("alias");

    await handleDeploymentStatus({ deploymentId: d.id, status: "stopped" }, {});

    const after = await prisma.deployment.findUnique({ where: { id: d.id } });
    expect(after?.publishedName).toBeNull();
  });

  // The name is resolved once per serving lifetime: repeated running ticks (the
  // agent re-reports on reconnect) must not re-probe the endpoint.
  it("resolves once — a repeated running report does not probe again", async () => {
    const d = await seedDeployment({ runtime: "vllm", modelName: "m" });
    const probe = servingFetch("first-answer");

    await handleDeploymentStatus({ deploymentId: d.id, status: "running", port: 8000 }, { fetchImpl: probe });
    await handleDeploymentStatus({ deploymentId: d.id, status: "running", port: 8000 }, { fetchImpl: probe });

    expect(probe).toHaveBeenCalledTimes(1);
    expect((await prisma.deployment.findUnique({ where: { id: d.id } }))?.publishedName).toBe("first-answer");
  });

  // A deployment that is up must never be failed by our inability to name it.
  it("still records a serving deployment when the endpoint is unreachable", async () => {
    const d = await seedDeployment({
      runtime: "vllm",
      modelName: "catalog-name",
      displayName: null,
    });
    const dead = vi.fn(async () => { throw new Error("ECONNREFUSED"); }) as unknown as typeof fetch;

    await handleDeploymentStatus(
      { deploymentId: d.id, status: "running", port: 8000 },
      { fetchImpl: dead },
    );

    const after = await prisma.deployment.findUnique({ where: { id: d.id } });
    expect(after?.status).toBe("running");
    expect(after?.publishedName).toBe("catalog-name");
  });

  it("ignores a status report for a deployment that no longer exists", async () => {
    await expect(
      handleDeploymentStatus({ deploymentId: "gone", status: "running", port: 8000 }, {}),
    ).resolves.toBeUndefined();
  });
});
