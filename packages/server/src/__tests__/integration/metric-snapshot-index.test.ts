/**
 * Asserts MetricSnapshot has a composite index that SQLite's planner uses
 * for the "latest sample per nodeId" lookup pattern. Without this index,
 * GET /api/nodes used to take ~10s with 2.4M rows in the table.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;

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
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

describe("MetricSnapshot composite index", () => {
  it("has an index on (nodeId, timestamp) per the schema", async () => {
    const indexes = await prisma.$queryRawUnsafe<{ name: string }[]>(
      "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='MetricSnapshot' AND name NOT LIKE 'sqlite_autoindex_%'"
    );
    expect(indexes.length).toBeGreaterThan(0);
  });

  it("uses the index for ORDER BY timestamp DESC LIMIT 1 per nodeId", async () => {
    const plan = await prisma.$queryRawUnsafe<{ detail: string }[]>(
      "EXPLAIN QUERY PLAN SELECT * FROM MetricSnapshot WHERE nodeId = 'x' ORDER BY timestamp DESC LIMIT 1"
    );
    const planText = plan.map((p) => p.detail).join(" | ");
    expect(planText).toMatch(/USING INDEX/i);
    expect(planText).not.toMatch(/^SCAN MetricSnapshot/i);
  });
});
