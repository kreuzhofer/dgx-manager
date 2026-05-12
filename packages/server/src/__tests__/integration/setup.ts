import { beforeAll, vi } from "vitest";

// vitest.config.ts uses pool=forks with singleFork=true so every test file
// runs in one process. prisma.ts caches its PrismaClient on globalThis, so
// without intervention, file 2's dynamic `import("../../prisma.js")` would
// return file 1's cached module whose adapter points at file 1's tmp DB —
// which file 1's afterAll already rm -rf'd. Result: "Cannot open database
// because the directory does not exist" on every query.
//
// Clearing both caches before each suite gives that suite a fresh
// PrismaClient bound to whatever DATABASE_URL is in effect at the time the
// suite calls `await import("../../prisma.js")`.
beforeAll(() => {
  delete (globalThis as { prisma?: unknown }).prisma;
  vi.resetModules();
});
