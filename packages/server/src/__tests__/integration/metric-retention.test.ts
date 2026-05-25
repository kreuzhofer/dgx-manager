/**
 * Asserts pruneMetricsOlderThan deletes rows strictly older than the cutoff
 * and leaves newer rows untouched. The server starts a periodic prune loop
 * in index.ts on boot — this test covers only the pure helper.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let pruneMetricsOlderThan: typeof import("../../metric-retention.js").pruneMetricsOlderThan;

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
  ({ pruneMetricsOlderThan } = await import("../../metric-retention.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  await prisma.metricSnapshot.deleteMany({});
  await prisma.node.deleteMany({});
  await prisma.node.create({
    data: { id: "node-a", name: "node-a", ipAddress: "10.0.0.1" },
  });
});

describe("pruneMetricsOlderThan", () => {
  it("deletes rows with timestamp < cutoff and keeps rows >= cutoff", async () => {
    const cutoff = new Date("2026-05-20T00:00:00Z");
    await prisma.metricSnapshot.createMany({
      data: [
        { nodeId: "node-a", gpuUtil: 1, vramUsed: 1, timestamp: new Date("2026-05-10T00:00:00Z") },
        { nodeId: "node-a", gpuUtil: 2, vramUsed: 2, timestamp: new Date("2026-05-19T23:59:59Z") },
        { nodeId: "node-a", gpuUtil: 3, vramUsed: 3, timestamp: new Date("2026-05-20T00:00:00Z") },
        { nodeId: "node-a", gpuUtil: 4, vramUsed: 4, timestamp: new Date("2026-05-25T00:00:00Z") },
      ],
    });

    const deleted = await pruneMetricsOlderThan(cutoff);

    expect(deleted).toBe(2);
    const remaining = await prisma.metricSnapshot.findMany({ orderBy: { timestamp: "asc" } });
    expect(remaining.map((r) => r.gpuUtil)).toEqual([3, 4]);
  });

  it("returns 0 when nothing matches", async () => {
    await prisma.metricSnapshot.create({
      data: { nodeId: "node-a", gpuUtil: 1, vramUsed: 1, timestamp: new Date() },
    });
    const deleted = await pruneMetricsOlderThan(new Date("2020-01-01"));
    expect(deleted).toBe(0);
  });
});
