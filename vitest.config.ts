import { defineConfig } from "vitest/config";

// Monorepo Vitest config. Tests live next to source as `*.test.ts`.
// Integration tests that need a real Prisma DB live under
// `packages/<pkg>/src/__tests__/integration/` and are matched separately
// so they can be run with `npm test -- integration` or excluded with
// `npm test -- --exclude '**/integration/**'` if they ever get slow.
export default defineConfig({
  test: {
    include: [
      "packages/*/src/**/*.test.ts",
      "packages/*/src/**/__tests__/**/*.test.ts",
    ],
    exclude: [
      "**/node_modules/**",
      "**/dist/**",
      "**/.next/**",
      "**/generated/**",
    ],
    environment: "node",
    // Property tests can run several iterations; keep an eye on this if it
    // creeps up. 10s/test is plenty for unit + property work.
    testTimeout: 10_000,
    hookTimeout: 30_000,
    // Each integration test creates a per-suite SQLite file. Sequential
    // execution avoids any DB filename collisions and keeps stack traces
    // readable. Unit tests are fast enough that the parallelism loss is
    // negligible at this size.
    pool: "forks",
    poolOptions: { forks: { singleFork: true } },
  },
});
