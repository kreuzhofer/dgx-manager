import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execSync } from "node:child_process";
import { beforeAll, afterAll, expect, it, describe } from "vitest";

const dir = mkdtempSync(join(tmpdir(), "reg-seed-"));
const DB_PATH = join(dir, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

const { prisma } = await import("../../prisma.js");

beforeAll(() => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      DATABASE_URL: `file:${DB_PATH}`,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "I have been authorized by Daniel to run destructive Prisma operations in this per-suite test database.",
    },
    stdio: "pipe",
  });
});

afterAll(async () => {
  await prisma.$disconnect();
});

import { seedDefaultRegistries, DEFAULT_REGISTRIES } from "../../registries/seed.js";

describe("seedDefaultRegistries", () => {
  it("inserts the defaults into an empty table and is idempotent", async () => {
    await prisma.sparkrunRegistry.deleteMany();
    const first = await seedDefaultRegistries(prisma);
    expect(first).toBe(DEFAULT_REGISTRIES.length);
    expect(await prisma.sparkrunRegistry.count()).toBe(DEFAULT_REGISTRIES.length);

    const second = await seedDefaultRegistries(prisma); // no-op when populated
    expect(second).toBe(0);
    expect(await prisma.sparkrunRegistry.count()).toBe(DEFAULT_REGISTRIES.length);
  });

  it("seeds eugr as visible and atlas as hidden", async () => {
    const eugr = await prisma.sparkrunRegistry.findUnique({ where: { name: "eugr" } });
    const atlas = await prisma.sparkrunRegistry.findUnique({ where: { name: "atlas" } });
    expect(eugr?.visible).toBe(true);
    expect(atlas?.visible).toBe(false);
  });
});
