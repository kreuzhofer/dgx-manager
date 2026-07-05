/**
 * Integration test for AgentHub.sweepStale() — the periodic sweep that marks
 * "online" nodes whose lastSeen has gone stale (past STALE_THRESHOLD_MS) as
 * offline, since a half-open WS socket's 'close' event can lag arbitrarily
 * long (observed 76+ min on a real DGX). See packages/server/src/ws/staleness.ts
 * (Task 1) for the pure `selectStaleNodes` helper this consumes.
 *
 * Follows the per-suite-SQLite harness established in
 * deployments.vram-admission.test.ts: DATABASE_URL set before importing
 * prisma, schema applied via `prisma db push --force-reset` with explicit
 * AI-action consent, wipeAll() between tests.
 */
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
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
let AgentHub: typeof import("../../ws/agent-hub.js").AgentHub;

beforeAll(async () => {
  // PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION: Prisma 7's CLI refuses
  // destructive operations when invoked by an AI agent unless this env
  // var carries an explicit user consent record. The user opted in
  // ("#1" -> option 1: grant consent for tests, on 2026-05-03) on the
  // explicit understanding that DATABASE_URL here always points at a
  // freshly-mkdtemp'd SQLite file in /tmp - there is no way for the
  // command to touch any real database.
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
  ({ AgentHub } = await import("../../ws/agent-hub.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

/** Wipe all tables in FK-dependency order so the next test starts clean. */
async function wipeAll() {
  await prisma.loadBalancerEndpoint.deleteMany({});
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.metricSnapshot.deleteMany({});
  await prisma.fineTuneJob.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
}

afterEach(async () => {
  await wipeAll();
});

describe("AgentHub.sweepStale", () => {
  it("marks a stale online node offline and leaves a fresh one online", async () => {
    const hub = new AgentHub();
    hub.stop(); // don't let the auto-interval fire during the test

    const fresh = await prisma.node.create({ data: { name: "fresh", status: "online", lastSeen: new Date() } });
    const stale = await prisma.node.create({ data: { name: "stale", status: "online", lastSeen: new Date(Date.now() - 60_000) } });
    (hub as unknown as { agents: Map<string, unknown> }).agents.set(stale.id, { ws: {} }); // seed the map to assert deletion

    await hub.sweepStale();

    expect((await prisma.node.findUnique({ where: { id: stale.id } }))!.status).toBe("offline");
    expect((await prisma.node.findUnique({ where: { id: fresh.id } }))!.status).toBe("online");
    expect((hub as unknown as { agents: Map<string, unknown> }).agents.has(stale.id)).toBe(false);

    hub.stop();
  });

  it("a swept-offline node returns to online when a heartbeat updates it", async () => {
    const n = await prisma.node.create({ data: { name: "recover", status: "offline", lastSeen: new Date(Date.now() - 60_000) } });
    // simulate the metric handler's self-heal update
    await prisma.node.update({ where: { id: n.id }, data: { lastSeen: new Date(), status: "online" } });
    expect((await prisma.node.findUnique({ where: { id: n.id } }))!.status).toBe("online");
  });
});
