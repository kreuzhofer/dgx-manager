import { describe, it, expect } from "vitest";
import { it as fcIt } from "@fast-check/vitest";
import { fc } from "@fast-check/vitest";
import { normalizeDisplayName, DisplayNameError } from "./display-name.js";

describe("normalizeDisplayName", () => {
  it("returns null for null/undefined input", () => {
    expect(normalizeDisplayName(null)).toBeNull();
    expect(normalizeDisplayName(undefined)).toBeNull();
  });

  it("returns null for empty / whitespace-only strings (treats them as unset)", () => {
    expect(normalizeDisplayName("")).toBeNull();
    expect(normalizeDisplayName("   ")).toBeNull();
    expect(normalizeDisplayName("\t\n")).toBeNull();
  });

  it("trims surrounding whitespace", () => {
    expect(normalizeDisplayName("  chat3d-prod  ")).toBe("chat3d-prod");
  });

  it("accepts URL-safe characters: letters, digits, dot, dash, underscore, colon", () => {
    expect(normalizeDisplayName("Chat3D_v1.0:fp8-prod")).toBe("Chat3D_v1.0:fp8-prod");
  });

  it("rejects strings containing whitespace (would break OpenAI model field)", () => {
    expect(() => normalizeDisplayName("chat 3d")).toThrow(DisplayNameError);
  });

  it("rejects strings containing slashes (would break loadbalancer routing)", () => {
    expect(() => normalizeDisplayName("vendor/chat3d")).toThrow(DisplayNameError);
  });

  it("rejects strings containing other special characters", () => {
    expect(() => normalizeDisplayName("chat@3d")).toThrow(DisplayNameError);
    expect(() => normalizeDisplayName("chat#3d")).toThrow(DisplayNameError);
  });

  it("rejects strings longer than 128 chars", () => {
    const tooLong = "a".repeat(129);
    expect(() => normalizeDisplayName(tooLong)).toThrow(DisplayNameError);
  });

  it("accepts exactly 128 chars", () => {
    const max = "a".repeat(128);
    expect(normalizeDisplayName(max)).toBe(max);
  });

  /**
   * Invariant: for any string drawn from the URL-safe character set, the
   * normalizer is idempotent — calling it twice produces the same result.
   */
  fcIt.prop([
    fc.stringMatching(/^[A-Za-z0-9._:-]{1,128}$/),
  ])("is idempotent on already-normalized inputs", (s) => {
    const once = normalizeDisplayName(s);
    const twice = normalizeDisplayName(once);
    expect(twice).toBe(once);
  });
});

import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { execSync } from "child_process";
import { beforeAll, afterAll } from "vitest";
import { validateDisplayNameUnique, ACTIVE_DEPLOYMENT_STATUSES } from "./display-name.js";

// Per-suite SQLite — same pattern as deployments.vram-admission.test.ts.
const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-displayname-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../prisma.js").prisma;

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
  ({ prisma } = await import("../prisma.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

async function seedDeployment(opts: {
  displayName?: string | null;
  status?: string;
}) {
  // Walk the FK chain: Node → Model → Deployment.
  const node = await prisma.node.create({
    data: { name: `node-${Math.random().toString(36).slice(2, 8)}` },
  });
  const model = await prisma.model.create({
    data: { name: `model-${Math.random().toString(36).slice(2, 8)}`, runtime: "vllm" },
  });
  return prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: opts.status ?? "running",
      displayName: opts.displayName ?? null,
    },
  });
}

describe("validateDisplayNameUnique", () => {
  it("returns null when name is null (skip check for unnamed deployments)", async () => {
    const result = await validateDisplayNameUnique(prisma, null);
    expect(result).toBeNull();
  });

  it("returns null when no other deployment has the name", async () => {
    const result = await validateDisplayNameUnique(prisma, "unused-name");
    expect(result).toBeNull();
  });

  it("returns conflict info when an active deployment has the name", async () => {
    const existing = await seedDeployment({ displayName: "chat3d-prod", status: "running" });
    const result = await validateDisplayNameUnique(prisma, "chat3d-prod");
    expect(result).toEqual({ conflictId: existing.id, conflictName: "chat3d-prod" });
  });

  it("ignores deployments in terminal statuses (failed, stopped, removed)", async () => {
    await seedDeployment({ displayName: "freed-name", status: "failed" });
    const result = await validateDisplayNameUnique(prisma, "freed-name");
    expect(result).toBeNull();
  });

  it("ignores the excluded deployment (used on restart)", async () => {
    const own = await seedDeployment({ displayName: "self-restart", status: "running" });
    const result = await validateDisplayNameUnique(prisma, "self-restart", own.id);
    expect(result).toBeNull();
  });

  it("exports the active-status list so route handlers stay aligned", () => {
    // Documented contract: at minimum, "running" and "starting" are active.
    // This guards against accidental drift between the helper and the routes.
    expect(ACTIVE_DEPLOYMENT_STATUSES).toContain("running");
    expect(ACTIVE_DEPLOYMENT_STATUSES).toContain("starting");
    expect(ACTIVE_DEPLOYMENT_STATUSES).toContain("pending");
  });
});
