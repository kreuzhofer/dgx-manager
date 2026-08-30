import { beforeAll, vi } from "vitest";

// Same singleFork leak, different global. Four benchmark integration suites set
// `process.env.SHARED_STORAGE_PATH` at module scope, and modules that read it
// freeze the value at import — packages/agent/src/env.ts does exactly that:
//   export const SHARED_STORAGE = process.env.SHARED_STORAGE_PATH || "/mnt/tank";
// With one process for the whole run, whichever file imports that module first
// decides its value for every later file, so a suite asserting the default
// passed or failed on file ordering alone. Resetting here — at module scope,
// which runs BEFORE each test file's own module scope — restores the default
// before anything can observe a previous suite's value. A suite that needs its
// own path still sets it afterwards and is unaffected.
//
// Deliberately not in the beforeAll below: that runs AFTER the test file's
// module scope, so it would delete the very value the suite had just set.
delete process.env.SHARED_STORAGE_PATH;

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
