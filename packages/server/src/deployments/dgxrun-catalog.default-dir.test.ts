/**
 * Guards the DGXRUN_RECIPES_DIR default: it must resolve relative to this
 * module (not process.cwd()), so the catalog isn't silently empty when
 * `npm run dev:server` runs with cwd = packages/server (not the repo root).
 *
 * The suite runs `pool: "forks"` with `singleFork: true` (see
 * vitest.config.ts), so all test files share one Node process and mutations
 * to `process.env` leak across files even though each file's module graph is
 * reset. Several sibling suites (e.g. routes/recipes.test.ts) set
 * DGXRUN_RECIPES_DIR at module scope and never restore it, so this test must
 * not assume the env var is unset just because it doesn't set it itself —
 * it explicitly deletes it right before importing the module under test.
 */
import { describe, it, expect, beforeAll } from "vitest";

delete process.env.DGXRUN_RECIPES_DIR;

let getDgxrunCatalog: typeof import("./dgxrun-catalog.js").getDgxrunCatalog;

beforeAll(async () => {
  delete process.env.DGXRUN_RECIPES_DIR;
  ({ getDgxrunCatalog } = await import("./dgxrun-catalog.js"));
});

describe("DGXRUN_RECIPES_DIR default (no env override)", () => {
  it("resolves to the real in-repo recipes/dgxrun dir and finds the glm-5.2 recipe", () => {
    const catalog = getDgxrunCatalog();
    expect(catalog.some((r) => r.file === "@dgxrun/glm-5.2-awq-15pct")).toBe(true);
  });
});
