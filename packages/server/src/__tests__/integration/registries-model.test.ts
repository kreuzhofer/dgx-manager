import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execSync } from "node:child_process";
import { beforeAll, afterAll, expect, it, describe } from "vitest";

const dir = mkdtempSync(join(tmpdir(), "reg-model-"));
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

describe("SparkrunRegistry model", () => {
  it("persists and reads back a registry row", async () => {
    await prisma.sparkrunRegistry.create({
      data: { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes" },
    });
    const found = await prisma.sparkrunRegistry.findUnique({ where: { name: "rtx" } });
    expect(found?.url).toBe("https://github.com/kreuzhofer/rtx-recipe-registry.git");
    expect(found?.visible).toBe(true); // default
  });
});
