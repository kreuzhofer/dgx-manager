# Deployment Display Name Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user set a per-deployment custom display name that overrides the model name in the dashboard deployments list AND becomes the served-model-name vLLM publishes via `/v1/models`.

**Architecture:** Add a `Deployment.displayName` column. The deploy form (stock vLLM + fine-tune vLLM) gains an optional input. POST endpoints normalize the value, enforce uniqueness across *running* deployments, persist it, and pass it to the agent as `servedModelName` (stock vLLM) / `modelName` (fine-tune; existing field). The agent's `cmd:deploy` (stock vLLM path) is extended to forward the name as a `--served-model-name <name>` passthrough arg to `vllm serve`. The dashboard displays `deployment.displayName ?? deployment.model.name` everywhere a deployment is shown.

**Tech Stack:** Prisma 7 + SQLite, Express 5, Next.js 15 dashboard, Vitest + supertest + fast-check for tests. vLLM via `spark-vllm-docker` run-recipe.sh (passthrough flag after `--`).

**Scope confirmed from spec Q&A:**
- Settable only at deploy creation. Editing = stop + redeploy with new name. No "rename running pod" endpoint.
- vLLM only (fine-tune + stock vLLM both supported). Ollama is **out of scope** — its served name is the registry tag.
- Uniqueness enforced across deployments in active statuses (`pending`, `running`, `starting`, `building`, `downloading`, `launching`, `loading`, `restarting`). Two deployments with the same display name = 409.
- Stored as a new column `Deployment.displayName String?` (mirrors `FineTuneJob.displayName` precedent).
- The restart endpoint preserves the existing displayName by default and accepts an override (re-validates uniqueness excluding self).

---

## File Structure

**Server (`packages/server/`):**

| File | Responsibility |
|------|----------------|
| `prisma/schema.prisma` (root) | Add `displayName String?` to `Deployment`. |
| `src/deployments/display-name.ts` (new) | Pure helpers: `normalizeDisplayName`, `validateDisplayNameUnique`. |
| `src/deployments/display-name.test.ts` (new) | Unit + property tests for `normalizeDisplayName`. |
| `src/routes/deployments.ts` (modify) | POST + restart read/persist/validate `displayName`; forward to agent payload as `servedModelName`. |
| `src/routes/finetune.ts` (modify) | POST `/:id/deploy` reads/persists/validates `displayName`; forwards to agent as `modelName` (overrides today's auto-derived value). |
| `src/__tests__/integration/deployments.display-name.test.ts` (new) | Integration: happy path, conflict path, restart preservation/override, finetune path. |

**Agent (`packages/agent/`):**

| File | Responsibility |
|------|----------------|
| `src/index.ts` (modify) | `cmd:deploy` reads optional `servedModelName` from payload and threads it into `launchRecipe`. (`cmd:finetune:deploy` already reads `modelName`; no change.) |
| `src/runtime/vllm.ts` (modify) | `launchRecipe` accepts `servedModelName?: string` and appends `--served-model-name <name>` to the post-`--` passthrough args. |
| `src/runtime/vllm.test.ts` (modify) | Add unit case for the passthrough flag composition. |
| `package.json` (modify) | Bump patch version via `./scripts/bump-agent-version.sh`. |

**Dashboard (`packages/dashboard/`):**

| File | Responsibility |
|------|----------------|
| `app/deployments/page.tsx` (modify) | Add "Display name" input to the vLLM + finetune deploy forms; send `displayName` in POST body; render `d.displayName ?? d.model?.name` in the deployments table. |
| `app/page.tsx` (modify) | Render `displayName ?? model.name` in the dashboard overview cards. |
| `lib/api.ts` (modify if `Deployment` type is centrally typed) | Add `displayName?: string \| null` to the `Deployment` interface. |

---

## Pre-flight: read these before starting

- `prisma/schema.prisma:68-84` — current `Deployment` model (your migration target).
- `packages/server/src/routes/deployments.ts` — POST handler (lines 43-257) and restart handler (303-382). Note the `activeStatuses` array (line 54), reused for uniqueness check.
- `packages/server/src/routes/finetune.ts:736-820` — deploy handler. Note how `model.name` and `modelName` are currently derived (`displayName || stableName`).
- `packages/agent/src/index.ts:499-620` — `cmd:deploy` handler (stock vLLM path).
- `packages/agent/src/index.ts:820-919` — `cmd:finetune:deploy` handler (note `servedModelName: modelName` at line 873).
- `packages/agent/src/runtime/vllm.ts:172-260` — `launchRecipe` (note the `--` passthrough at lines 209 and how `clusterNodes` etc. are sliced).
- `packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts:1-75` — canonical integration-test scaffold (per-suite SQLite, `force-reset`, stub agentHub).
- `packages/server/src/admission/vram.test.ts:1-40` — canonical property-test scaffold (`@fast-check/vitest`).

---

## Task 1: Schema migration — add `Deployment.displayName`

**Files:**
- Modify: `prisma/schema.prisma:68-84`

- [ ] **Step 1: Add the column**

In `prisma/schema.prisma`, find the `Deployment` model (line 68) and add `displayName` directly under `id`:

```prisma
model Deployment {
  id           String                 @id @default(cuid())
  // User-supplied friendly name. When set, it overrides `model.name` in the
  // dashboard list AND is passed to vLLM as `--served-model-name`, so the
  // deployment surfaces under this name in OpenAI-API responses
  // (`/v1/models` and the `model` field of completions). Null falls back to
  // `model.name`. Normalized and uniqueness-validated at the route layer
  // (src/deployments/display-name.ts) — null entries skip the uniqueness
  // check so unnamed deployments coexist freely.
  displayName  String?
  nodeId       String
  modelId      String
  status       String                 @default("pending")
  port         Int?
  config       String?
  clusterMode  Boolean                @default(false)
  vramEstimate Int?
  vramActual   Int?
  createdAt    DateTime               @default(now())
  updatedAt    DateTime               @updatedAt
  node         Node                   @relation(fields: [nodeId], references: [id])
  model        Model                  @relation(fields: [modelId], references: [id])
  lbEndpoints  LoadBalancerEndpoint[]
  clusterNodes ClusterNode[]
}
```

- [ ] **Step 2: Apply schema to local dev DB and regenerate the client**

Run from the repo root:

```bash
npm run db:push
npm run db:generate
```

Expected: `db push` completes ("✔ Generated Prisma Client"). `db:generate` regenerates `packages/server/src/generated/prisma/`.

- [ ] **Step 3: Verify the regenerated client has the new field**

```bash
grep -n "displayName" packages/server/src/generated/prisma/index.d.ts | head -5
```

Expected: at least one match showing `displayName: string | null` in a `Deployment` interface.

- [ ] **Step 4: Run existing tests to confirm nothing broke**

```bash
npm test
```

Expected: all existing tests still pass. The new column is nullable, so existing inserts that don't set it remain valid.

- [ ] **Step 5: Commit**

```bash
git add prisma/schema.prisma packages/server/src/generated/prisma
git commit -m "db: add Deployment.displayName column for custom served-model-name"
```

---

## Task 2: Pure helper — `normalizeDisplayName`

**Files:**
- Create: `packages/server/src/deployments/display-name.ts`
- Create: `packages/server/src/deployments/display-name.test.ts`

**What it does:** Takes raw user input (possibly null/undefined/whitespace) and returns either a clean string or `null`. Rejects empty/illegal values by *throwing*; routes catch and 400.

**Allowed character set:** `[A-Za-z0-9._:-]` (matches what vLLM's `--served-model-name` accepts safely and what URL-routes well). Spaces and slashes are rejected — they break the OpenAI-API `model` field and the loadbalancer's routing key.

**Max length:** 128 chars (well below any vLLM/Prisma cap; consistent with `FineTuneJob.displayName` usage).

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/deployments/display-name.test.ts`:

```typescript
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
npx vitest run packages/server/src/deployments/display-name.test.ts
```

Expected: FAIL — `Cannot find module './display-name.js'` (the implementation file doesn't exist yet).

- [ ] **Step 3: Write the implementation**

Create `packages/server/src/deployments/display-name.ts`:

```typescript
/**
 * Per-deployment custom display name validation + normalization.
 *
 * The display name has two consumers:
 *   1. The dashboard deployments list (rendered as `displayName ?? model.name`).
 *   2. vLLM's `--served-model-name` flag, which sets the value that surfaces
 *      via the OpenAI API's `/v1/models` and the `model` field of completions.
 *
 * Both consumers need a value that's safe in URLs and HTTP `model` fields.
 * We restrict to `[A-Za-z0-9._:-]` (letters, digits, dot, dash, underscore,
 * colon) — the same alphabet HuggingFace model ids and Docker image tags use.
 * Spaces and slashes are rejected: spaces break clients that don't quote the
 * model field; slashes collide with REST path segments in the loadbalancer.
 */
export class DisplayNameError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DisplayNameError";
  }
}

const ALLOWED = /^[A-Za-z0-9._:-]+$/;
const MAX_LENGTH = 128;

/**
 * Normalize raw user input.
 *
 * @returns `null` when the input is null/undefined/empty/whitespace-only.
 *          Otherwise returns the trimmed string after validation passes.
 * @throws  {DisplayNameError} when the input is non-empty but contains
 *          disallowed characters or exceeds {@link MAX_LENGTH}.
 */
export function normalizeDisplayName(raw: string | null | undefined): string | null {
  if (raw === null || raw === undefined) return null;
  const trimmed = raw.trim();
  if (trimmed.length === 0) return null;
  if (trimmed.length > MAX_LENGTH) {
    throw new DisplayNameError(
      `Display name must be ${MAX_LENGTH} characters or fewer (got ${trimmed.length}).`,
    );
  }
  if (!ALLOWED.test(trimmed)) {
    throw new DisplayNameError(
      "Display name may only contain letters, digits, dot, dash, underscore, and colon " +
        "(rejected: " + JSON.stringify(trimmed) + ").",
    );
  }
  return trimmed;
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
npx vitest run packages/server/src/deployments/display-name.test.ts
```

Expected: PASS (9 named cases + 1 property test = 10 tests total).

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/deployments/display-name.ts packages/server/src/deployments/display-name.test.ts
git commit -m "server: add normalizeDisplayName helper for Deployment.displayName"
```

---

## Task 3: Pure-ish helper — `validateDisplayNameUnique`

**Files:**
- Modify: `packages/server/src/deployments/display-name.ts`
- Modify: `packages/server/src/deployments/display-name.test.ts`

**What it does:** Given a prisma client, a candidate display name, and an optional `excludeDeploymentId`, returns `null` if the name is free or returns `{ conflictId, conflictName }` if some active deployment already owns it. We test this with the prisma client directly — no integration test scaffolding needed for the helper itself (route-level integration tests in Task 4-6 cover end-to-end behavior).

We keep the active-statuses list inside this file (small constant) so the helper is self-contained and unit-testable.

- [ ] **Step 1: Add the failing test (extends the same test file)**

Append to `packages/server/src/deployments/display-name.test.ts`:

```typescript
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
npx vitest run packages/server/src/deployments/display-name.test.ts
```

Expected: FAIL — `validateDisplayNameUnique` and `ACTIVE_DEPLOYMENT_STATUSES` are not exported.

- [ ] **Step 3: Add the implementation**

Append to `packages/server/src/deployments/display-name.ts`:

```typescript
import type { PrismaClient } from "../generated/prisma/index.js";

/**
 * Statuses that count as "this deployment is using its display name right now."
 * Mirrors the activeStatuses list in routes/deployments.ts:54; centralized
 * here so the uniqueness check and the routes can't drift apart.
 */
export const ACTIVE_DEPLOYMENT_STATUSES = [
  "pending",
  "running",
  "starting",
  "building",
  "downloading",
  "launching",
  "loading",
  "restarting",
] as const;

export interface DisplayNameConflict {
  conflictId: string;
  conflictName: string;
}

/**
 * Check whether a candidate display name is free.
 *
 * @param prisma                The prisma client to query.
 * @param name                  The (already normalized) candidate. `null` is a
 *                              no-op — unnamed deployments coexist freely.
 * @param excludeDeploymentId   Used on the restart path to exclude the
 *                              deployment that's about to be relaunched from
 *                              its own conflict check.
 * @returns `null` if free, otherwise a `DisplayNameConflict` identifying the
 *          deployment currently holding the name.
 */
export async function validateDisplayNameUnique(
  prisma: PrismaClient,
  name: string | null,
  excludeDeploymentId?: string,
): Promise<DisplayNameConflict | null> {
  if (name === null) return null;
  const conflict = await prisma.deployment.findFirst({
    where: {
      displayName: name,
      status: { in: [...ACTIVE_DEPLOYMENT_STATUSES] },
      ...(excludeDeploymentId ? { id: { not: excludeDeploymentId } } : {}),
    },
    select: { id: true, displayName: true },
  });
  if (!conflict || !conflict.displayName) return null;
  return { conflictId: conflict.id, conflictName: conflict.displayName };
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
npx vitest run packages/server/src/deployments/display-name.test.ts
```

Expected: PASS (all previous tests + 6 new cases).

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/deployments/display-name.ts packages/server/src/deployments/display-name.test.ts
git commit -m "server: add validateDisplayNameUnique for Deployment uniqueness"
```

---

## Task 4: Wire `displayName` into POST /api/deployments

**Files:**
- Modify: `packages/server/src/routes/deployments.ts:43-257`
- Create: `packages/server/src/__tests__/integration/deployments.display-name.test.ts`

The POST handler needs to: destructure `displayName` from body, normalize it, validate uniqueness, persist it on the Deployment row, and pass it to the agent's `cmd:deploy` payload as `servedModelName`. Errors from the normalizer become HTTP 400; conflicts become 409.

- [ ] **Step 1: Write the failing integration test**

Create `packages/server/src/__tests__/integration/deployments.display-name.test.ts`:

```typescript
/**
 * Integration coverage for Deployment.displayName on the stock vLLM and
 * finetune deploy routes, plus the restart preservation/override path.
 *
 * Pattern matches finetune.naming-and-cleanup.test.ts: per-suite SQLite,
 * supertest, stub agentHub that captures outgoing messages.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-deploy-displayname-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;

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
  ({ deploymentsRouter } = await import("../../routes/deployments.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

const RECIPE = {
  file: "recipes/test-recipe.yaml",
  name: "Test Recipe",
  defaults: { gpu_memory_utilization: 0.5 },
};

function makeStubHub() {
  const sent: { nodeId: string; message: { type: string; payload: Record<string, unknown> } }[] = [];
  return {
    hub: {
      getRecipes: () => [RECIPE],
      getOllamaModels: () => [],
      sendToAgent: (nodeId: string, message: { type: string; payload: Record<string, unknown> }) => {
        sent.push({ nodeId, message });
      },
    },
    sent,
  };
}

function makeApp(hub: unknown) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

async function wipeAll() {
  // FK-safe deletion order.
  await prisma.loadBalancerEndpoint.deleteMany();
  await prisma.clusterNode.deleteMany();
  await prisma.deployment.deleteMany();
  await prisma.model.deleteMany();
  await prisma.metricSnapshot.deleteMany();
  await prisma.node.deleteMany();
}

async function seedNode(name = "n1") {
  return prisma.node.create({
    data: {
      name,
      status: "online",
      vramTotal: 128000,
      ipAddress: "10.0.0.10",
    },
  });
}

beforeEach(wipeAll);

describe("POST /api/deployments with displayName", () => {
  it("persists displayName when supplied and forwards it as servedModelName", async () => {
    const node = await seedNode();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "chat3d-prod",
        config: { port: 8000 },
      });

    expect(res.status).toBe(201);
    expect(res.body.displayName).toBe("chat3d-prod");

    const row = await prisma.deployment.findUnique({ where: { id: res.body.id } });
    expect(row?.displayName).toBe("chat3d-prod");

    expect(sent).toHaveLength(1);
    expect(sent[0].message.type).toBe("cmd:deploy");
    expect(sent[0].message.payload.servedModelName).toBe("chat3d-prod");
  });

  it("leaves displayName null when omitted, and does not send servedModelName", async () => {
    const node = await seedNode();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        config: { port: 8000 },
      });

    expect(res.status).toBe(201);
    expect(res.body.displayName).toBeNull();
    expect(sent[0].message.payload.servedModelName).toBeUndefined();
  });

  it("rejects illegal characters with 400", async () => {
    const node = await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    const res = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "chat 3d", // space rejected
        config: { port: 8000 },
      });

    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/letters, digits/);
  });

  it("rejects duplicate displayName among running deployments with 409", async () => {
    const node = await seedNode();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    // First deploy claims the name.
    await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "chat3d-prod",
        config: { port: 8000 },
      });

    // Mark it running so it counts toward uniqueness (handler creates as pending).
    await prisma.deployment.updateMany({
      where: { displayName: "chat3d-prod" },
      data: { status: "running" },
    });

    // Second deploy with same name → conflict.
    const node2 = await seedNode("n2");
    const res = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node2.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "chat3d-prod",
        config: { port: 8000 },
      });

    expect(res.status).toBe(409);
    expect(res.body.error).toMatch(/already in use/);
  });
});

describe("POST /api/deployments/:id/restart with displayName", () => {
  it("preserves the existing displayName when body has no override", async () => {
    const node = await seedNode();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const created = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "preserved-name",
        config: { port: 8000 },
      });
    sent.length = 0; // clear send log

    const restart = await request(app)
      .post(`/api/deployments/${created.body.id}/restart`)
      .send({});

    expect(restart.status).toBe(200);
    const row = await prisma.deployment.findUnique({ where: { id: created.body.id } });
    expect(row?.displayName).toBe("preserved-name");
    expect(sent[0].message.payload.servedModelName).toBe("preserved-name");
  });

  it("accepts a displayName override and re-validates uniqueness (excluding self)", async () => {
    const node = await seedNode();
    const { hub, sent } = makeStubHub();
    const app = makeApp(hub);

    const created = await request(app)
      .post("/api/deployments")
      .send({
        nodeId: node.id,
        recipeFile: "recipes/test-recipe.yaml",
        displayName: "original",
        config: { port: 8000 },
      });
    await prisma.deployment.update({
      where: { id: created.body.id },
      data: { status: "running" },
    });
    sent.length = 0;

    const restart = await request(app)
      .post(`/api/deployments/${created.body.id}/restart`)
      .send({ displayName: "renamed" });

    expect(restart.status).toBe(200);
    const row = await prisma.deployment.findUnique({ where: { id: created.body.id } });
    expect(row?.displayName).toBe("renamed");
    expect(sent[0].message.payload.servedModelName).toBe("renamed");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
npx vitest run packages/server/src/__tests__/integration/deployments.display-name.test.ts
```

Expected: FAIL — the handler doesn't accept `displayName` yet; all assertions on `res.body.displayName` are null/undefined and the `servedModelName` payload field is missing.

- [ ] **Step 3: Modify `routes/deployments.ts` — POST handler**

In `packages/server/src/routes/deployments.ts`, add the import near the top (around line 9):

```typescript
import { normalizeDisplayName, validateDisplayNameUnique, DisplayNameError } from "../deployments/display-name.js";
```

Update the destructure at the top of the POST handler (line 44):

```typescript
deploymentsRouter.post("/", async (req, res) => {
  let { nodeId, nodeIds, recipeFile, config, runtime, modelName, modelType, displayName: rawDisplayName } = req.body;
  const isOllama = runtime === "ollama";

  // Normalize + uniqueness-check displayName up front. 400 on bad chars,
  // 409 on duplicate among active deployments. Ollama deploys ignore the
  // field (runtime doesn't honor it); we still reject malformed values so
  // the dashboard surfaces the error immediately.
  let displayName: string | null;
  try {
    displayName = normalizeDisplayName(rawDisplayName);
  } catch (e) {
    if (e instanceof DisplayNameError) return res.status(400).json({ error: e.message });
    throw e;
  }
  if (displayName && !isOllama) {
    const conflict = await validateDisplayNameUnique(prisma, displayName);
    if (conflict) {
      return res.status(409).json({
        error: `Display name "${displayName}" is already in use by deployment ${conflict.conflictId}.`,
        conflict,
      });
    }
  }
  // Ollama doesn't support per-deploy renames; reject explicitly so the user
  // doesn't think it took effect.
  if (displayName && isOllama) {
    return res.status(400).json({
      error: "displayName is not supported for Ollama deployments (use the model tag).",
    });
  }

  // ...rest of the existing handler unchanged until the create.
```

Then in the `prisma.deployment.create` call (line 192), add `displayName` to the data:

```typescript
const deployment = await prisma.deployment.create({
  data: {
    modelId: model.id,
    nodeId: headNodeId,
    clusterMode: isCluster,
    vramEstimate: vramEstimate || null,
    displayName,
    config: JSON.stringify(isOllama
      ? { runtime: "ollama", modelName, modelType: modelType || "chat", ...config }
      : { recipeFile, ...config }),
  },
});
```

Then in the `agentHub.sendToAgent` call (line 235), add `servedModelName: displayName ?? undefined` to the payload (only for the vLLM branch; Ollama already short-circuits above):

```typescript
agentHub.sendToAgent(headNodeId, {
  type: "cmd:deploy",
  payload: {
    deploymentId: deployment.id,
    runtime: isOllama ? "ollama" : "vllm",
    modelName: isOllama ? modelName : undefined,
    modelType: isOllama ? (modelType || "chat") : undefined,
    recipeFile: isOllama ? undefined : recipeFile,
    // Per-deploy custom name → vLLM's --served-model-name. Undefined when
    // the user didn't set displayName, so the agent falls back to the
    // recipe's authored defaults.served_model_name.
    servedModelName: displayName ?? undefined,
    config: config || {},
    clusterNodes: clusterNodeIps,
    clusterNodeFastIps,
  },
});
```

- [ ] **Step 4: Modify `routes/deployments.ts` — restart handler**

In the restart handler (line 303), after the existing destructure of `overrides`, add:

```typescript
// Allow the caller to update displayName on restart (re-validating
// uniqueness, excluding self). Body shape: { displayName: "new-name" } at
// the top level (NOT nested under config — displayName is a column, not
// part of the recipe config blob).
let newDisplayName = deployment.displayName;
if (req.body && Object.prototype.hasOwnProperty.call(req.body, "displayName")) {
  try {
    newDisplayName = normalizeDisplayName(req.body.displayName as string | null | undefined);
  } catch (e) {
    if (e instanceof DisplayNameError) return res.status(400).json({ error: e.message });
    throw e;
  }
  if (newDisplayName !== deployment.displayName) {
    const conflict = await validateDisplayNameUnique(prisma, newDisplayName, deployment.id);
    if (conflict) {
      return res.status(409).json({
        error: `Display name "${newDisplayName}" is already in use by deployment ${conflict.conflictId}.`,
        conflict,
      });
    }
  }
}
```

Then in the `agentHub.sendToAgent` call inside restart, add the `servedModelName`:

```typescript
agentHub.sendToAgent(deployment.nodeId, {
  type: "cmd:deploy",
  payload: {
    deploymentId: deployment.id,
    runtime: isOllamaRestart ? "ollama" : "vllm",
    modelName: isOllamaRestart ? config.modelName : undefined,
    modelType: isOllamaRestart ? (config.modelType || "chat") : undefined,
    recipeFile: isOllamaRestart ? undefined : config.recipeFile,
    servedModelName: newDisplayName ?? undefined,
    config,
    clusterNodes: clusterNodeIps,
    clusterNodeFastIps,
  },
});
```

And in the prisma update at the end of restart, persist `newDisplayName` whenever it changed:

```typescript
await prisma.deployment.update({
  where: { id: req.params.id },
  data: {
    status: "restarting",
    ...(Object.keys(overrides).length > 0 ? { config: JSON.stringify(config) } : {}),
    ...(newDisplayName !== deployment.displayName ? { displayName: newDisplayName } : {}),
  },
});
```

- [ ] **Step 5: Run the integration test**

```bash
npx vitest run packages/server/src/__tests__/integration/deployments.display-name.test.ts
```

Expected: PASS (all 6 cases under the two `describe` blocks).

- [ ] **Step 6: Run the full server test suite to ensure no regressions**

```bash
npm test
```

Expected: all tests still pass.

- [ ] **Step 7: Commit**

```bash
git add packages/server/src/routes/deployments.ts packages/server/src/__tests__/integration/deployments.display-name.test.ts
git commit -m "server: accept Deployment.displayName on POST + restart; forward as servedModelName"
```

---

## Task 5: Wire `displayName` into POST /api/finetune/:id/deploy

**Files:**
- Modify: `packages/server/src/routes/finetune.ts:736-820`
- Modify: `packages/server/src/__tests__/integration/deployments.display-name.test.ts` (append new `describe` block)

The finetune deploy route already passes `modelName` to the agent (which the agent uses as `servedModelName`). Today that value is derived from `FineTuneJob.displayName` or a stable `finetune-<id>` fallback. We add an OPTIONAL `displayName` body field on the deploy request that overrides this just for the deployment — so the same fine-tune can be deployed twice under different names.

- [ ] **Step 1: Append the failing integration test**

Append to `packages/server/src/__tests__/integration/deployments.display-name.test.ts` (you'll need to add the finetune router import at the top of the file too):

```typescript
// Add to the imports at the top of the file:
// let finetuneRouter: typeof import("../../routes/finetune.js").finetuneRouter;
//
// And inside beforeAll, after importing deploymentsRouter:
// ({ finetuneRouter } = await import("../../routes/finetune.js"));

describe("POST /api/finetune/:id/deploy with displayName override", () => {
  // Local app helper that mounts BOTH routers.
  function makeFtApp(hub: unknown) {
    const app = express();
    app.use(express.json());
    app.set("agentHub", hub);
    app.use("/api/finetune", finetuneRouter);
    app.use("/api/deployments", deploymentsRouter);
    return app;
  }

  function makeFtStubHub() {
    const sent: { nodeId: string; message: { type: string; payload: Record<string, unknown> } }[] = [];
    return {
      hub: {
        getRecipes: () => [],
        getOllamaModels: () => [],
        getTrainingRecipes: () => [
          {
            file: "recipes/test-attn-mlp",
            name: "Test FT Recipe",
            base_model: "Qwen/Qwen3.6-27B",
            method: "lora",
            defaults: {},
            scripts: { merge: "scripts/merge.py" },
            deploy: { gpu_memory_utilization: 0.5 },
          },
        ],
        sendToAgent: (nodeId: string, message: { type: string; payload: Record<string, unknown> }) => {
          sent.push({ nodeId, message });
        },
      },
      sent,
    };
  }

  async function seedCompletedJob(node: Awaited<ReturnType<typeof seedNode>>) {
    return prisma.fineTuneJob.create({
      data: {
        nodeId: node.id,
        baseModel: "Qwen/Qwen3.6-27B",
        method: "lora",
        dataset: "/tmp/ds.jsonl",
        recipeFile: "recipes/test-attn-mlp",
        status: "completed",
        mergeStatus: "completed",
        mergedPath: "/tmp/merged",
        displayName: "chat3d-build123d-01",
      },
    });
  }

  it("uses the FineTuneJob.displayName when no per-deploy displayName is supplied", async () => {
    const node = await seedNode();
    const job = await seedCompletedJob(node);
    const { hub, sent } = makeFtStubHub();
    const app = makeFtApp(hub);

    const res = await request(app)
      .post(`/api/finetune/${job.id}/deploy`)
      .send({ nodeId: node.id, config: { port: 8000 } });

    expect(res.status).toBe(201);
    expect(res.body.displayName).toBeNull();
    expect(sent[0].message.type).toBe("cmd:finetune:deploy");
    // Falls back to Model.name, which is the FineTuneJob.displayName.
    expect(sent[0].message.payload.modelName).toBe("chat3d-build123d-01");
  });

  it("uses the per-deploy displayName when supplied (overrides FT name for this deploy)", async () => {
    const node = await seedNode();
    const job = await seedCompletedJob(node);
    const { hub, sent } = makeFtStubHub();
    const app = makeFtApp(hub);

    const res = await request(app)
      .post(`/api/finetune/${job.id}/deploy`)
      .send({
        nodeId: node.id,
        displayName: "chat3d-prod-variant-a",
        config: { port: 8000 },
      });

    expect(res.status).toBe(201);
    expect(res.body.displayName).toBe("chat3d-prod-variant-a");
    expect(sent[0].message.payload.modelName).toBe("chat3d-prod-variant-a");
  });

  it("rejects 409 when the per-deploy displayName conflicts with an active deployment", async () => {
    const node = await seedNode();
    const job = await seedCompletedJob(node);
    const { hub } = makeFtStubHub();
    const app = makeFtApp(hub);

    // Seed an existing running deployment with the contested name.
    await prisma.deployment.create({
      data: {
        nodeId: node.id,
        modelId: (await prisma.model.create({ data: { name: "other", runtime: "vllm" } })).id,
        status: "running",
        displayName: "taken-name",
      },
    });

    const res = await request(app)
      .post(`/api/finetune/${job.id}/deploy`)
      .send({
        nodeId: node.id,
        displayName: "taken-name",
        config: { port: 8000 },
      });

    expect(res.status).toBe(409);
    expect(res.body.error).toMatch(/already in use/);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
npx vitest run packages/server/src/__tests__/integration/deployments.display-name.test.ts
```

Expected: FAIL on the new `describe` block — the route doesn't read `displayName` from the body yet.

- [ ] **Step 3: Modify `routes/finetune.ts`**

Add the import near the top:

```typescript
import { normalizeDisplayName, validateDisplayNameUnique, DisplayNameError } from "../deployments/display-name.js";
```

In the deploy handler, near where `config` is destructured, add `displayName` reading + validation. Then at the `prisma.deployment.create` call (around line 765), persist it. Finally, at the `agentHub.sendToAgent` call (around line 802), prefer the per-deploy name over the model name.

Concretely, find this block in the existing handler (around line 740-745):

```typescript
const stableName = `finetune-${job.id.slice(0, 8)}`;
const modelName = job.displayName || stableName;
```

Replace with:

```typescript
const stableName = `finetune-${job.id.slice(0, 8)}`;
const ftModelName = job.displayName || stableName;

// Optional per-deploy displayName override. When set it does NOT touch
// Model.name (the FT's catalog identity stays stable); it only overrides
// what vLLM publishes via --served-model-name AND what the dashboard
// shows in the deployments list. Lets the same FT be deployed twice
// under different served names (e.g. "chat3d-prod" + "chat3d-canary").
let perDeployDisplayName: string | null;
try {
  perDeployDisplayName = normalizeDisplayName(
    (req.body as { displayName?: string | null | undefined } | undefined)?.displayName,
  );
} catch (e) {
  if (e instanceof DisplayNameError) return res.status(400).json({ error: e.message });
  throw e;
}
if (perDeployDisplayName) {
  const conflict = await validateDisplayNameUnique(prisma, perDeployDisplayName);
  if (conflict) {
    return res.status(409).json({
      error: `Display name "${perDeployDisplayName}" is already in use by deployment ${conflict.conflictId}.`,
      conflict,
    });
  }
}

// What vLLM ultimately publishes. Per-deploy override wins; otherwise
// fall back to the fine-tune's own displayName / stable name.
const servedModelName = perDeployDisplayName || ftModelName;
```

Then in the `prisma.deployment.create` call (around line 765), add `displayName`:

```typescript
const deployment = await prisma.deployment.create({
  data: {
    nodeId: headNodeId,
    modelId: model.id,
    status: "pending",
    clusterMode: isCluster,
    displayName: perDeployDisplayName,
    config: JSON.stringify({ ...config, localModelPath: modelPath }),
  },
});
```

And in the agent payload (around line 802), pass `servedModelName` instead of `model.name`:

```typescript
agentHub.sendToAgent(headNodeId, {
  type: "cmd:finetune:deploy",
  payload: {
    jobId: job.id,
    deploymentId: deployment.id,
    modelPath,
    baseModel: job.baseModel,
    deployContainer: deployConfig?.container || "vllm-node",
    // Per-deploy override wins; otherwise Model.name (FT's stable name).
    // Agent threads this into vLLM's --served-model-name.
    modelName: servedModelName,
    recipeFile: job.recipeFile,
    artifactVariant: variant,
    clusterNodes: clusterNodeIps,
    clusterNodeFastIps,
    config: {
      // ...existing config build unchanged
    },
  },
});
```

(Keep the rest of the function unchanged.)

- [ ] **Step 4: Run test to verify it passes**

```bash
npx vitest run packages/server/src/__tests__/integration/deployments.display-name.test.ts
```

Expected: PASS (all 3 new finetune cases + previously passing cases).

- [ ] **Step 5: Run full suite**

```bash
npm test
```

Expected: all tests pass. Note: the existing `finetune.naming-and-cleanup.test.ts` uses `model.name` semantics, which we did NOT change — only added a new layer above it.

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/routes/finetune.ts packages/server/src/__tests__/integration/deployments.display-name.test.ts
git commit -m "server: accept per-deploy displayName override on /api/finetune/:id/deploy"
```

---

## Task 6: Agent — thread `servedModelName` through `launchRecipe` for stock vLLM

**Files:**
- Modify: `packages/agent/src/runtime/vllm.ts` (the `launchRecipe` function around line 172-260)
- Modify: `packages/agent/src/runtime/vllm.test.ts` (add a unit case)
- Modify: `packages/agent/src/index.ts:499-620` (the `cmd:deploy` case)

The fine-tune path is already complete — `cmd:finetune:deploy` reads `modelName` from the payload and passes it as `servedModelName` to `generateLocalModelRecipe`, which substitutes it into the materialized YAML template (line 873).

The stock vLLM path is the gap. `cmd:deploy` doesn't read `servedModelName` today, and `launchRecipe` doesn't have a hook to pass `--served-model-name`. `run-recipe.py` already accepts arbitrary passthrough args after `--` (the help text on line 680 advertises `-- --served-model-name my-api`), so the fix is mechanical.

- [ ] **Step 1: Write the failing test**

Open `packages/agent/src/runtime/vllm.test.ts`. If it doesn't exist, create it. Find the existing tests for `launchRecipe` argument composition (likely a test that snapshots `args` or asserts on what's appended). If there's no existing argv test, add a small one. Add the new case:

```typescript
import { describe, it, expect, vi } from "vitest";

describe("launchRecipe servedModelName passthrough", () => {
  it("appends --served-model-name <name> to the post-`--` passthrough args when servedModelName is set", () => {
    // The test asserts on the argv that would be sent to bash -c. We capture
    // it via a spy on child_process.spawn. If a different testing pattern
    // already exists in this file, follow it; otherwise import buildLaunchArgs
    // (Step 2 extracts the arg-building into a pure helper so we can test it
    // directly without process forking).
    const { buildLaunchArgs } = require("./vllm.js");

    const args = buildLaunchArgs({
      recipeName: "test-recipe",
      options: {
        port: 8000,
        servedModelName: "my-custom-name",
      },
    });

    // The passthrough args come after "--" (or are appended as a final
    // grouping). Either way, both flag and value must be present and adjacent.
    const flagIdx = args.indexOf("--served-model-name");
    expect(flagIdx).toBeGreaterThan(-1);
    expect(args[flagIdx + 1]).toBe("my-custom-name");
  });

  it("does NOT append --served-model-name when servedModelName is undefined", () => {
    const { buildLaunchArgs } = require("./vllm.js");
    const args = buildLaunchArgs({
      recipeName: "test-recipe",
      options: { port: 8000 },
    });
    expect(args.indexOf("--served-model-name")).toBe(-1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
npx vitest run packages/agent/src/runtime/vllm.test.ts
```

Expected: FAIL — `buildLaunchArgs` is not exported.

- [ ] **Step 3: Extract `buildLaunchArgs` into a pure function and add `servedModelName` support**

In `packages/agent/src/runtime/vllm.ts`, refactor the argv-building inside `launchRecipe` into an exported pure helper, then call it from `launchRecipe`. Add `servedModelName` to the options type and append the passthrough flag.

```typescript
export interface LaunchRecipeOptions {
  port?: number;
  gpuMem?: number;
  maxModelLen?: number;
  tensorParallel?: number;
  pipelineParallel?: number;
  clusterNodes?: string[];
  clusterNodeFastIps?: (string | null)[];
  skipSetup?: boolean;
  /**
   * Optional per-deploy served-model-name. When set, appended as
   * `--served-model-name <value>` to the post-`--` passthrough args so vLLM's
   * OpenAI API surface (`/v1/models`, completion responses) reports this name
   * instead of the recipe's authored default.
   */
  servedModelName?: string;
}

/**
 * Pure helper: build the argv array for run-recipe.sh given a recipe name and
 * options. Extracted so it can be unit-tested without spawning a process.
 */
export function buildLaunchArgs(params: {
  recipeName: string;
  options?: LaunchRecipeOptions;
}): string[] {
  const { recipeName, options } = params;
  const args: string[] = [recipeName];

  const isCluster = options?.clusterNodes && options.clusterNodes.length > 1;

  if (isCluster) {
    args.push("-n", options!.clusterNodes!.join(","));
    if (!options?.skipSetup) args.push("--setup");
  } else {
    args.push("--solo");
    if (!options?.skipSetup) args.push("--setup");
  }

  if (options?.port) args.push("--port", String(options.port));
  if (options?.gpuMem) args.push("--gpu-mem", String(options.gpuMem));
  if (options?.maxModelLen) args.push("--max-model-len", String(options.maxModelLen));
  if (options?.tensorParallel) args.push("--tp", String(options.tensorParallel));
  if (ETH_IF) args.push("--eth-if", ETH_IF);
  if (IB_IF) args.push("--ib-if", IB_IF);

  // Post-`--` passthrough args (forwarded verbatim to `vllm serve`).
  // Existing convention in this file: pipelineParallel uses this slot.
  const passthrough: string[] = [];
  if (options?.pipelineParallel) passthrough.push("-pp", String(options.pipelineParallel));
  if (options?.servedModelName) {
    passthrough.push("--served-model-name", options.servedModelName);
  }
  if (passthrough.length > 0) args.push("--", ...passthrough);

  return args;
}
```

Then replace the existing inline argv-building in `launchRecipe` (lines 193-209) with a call to `buildLaunchArgs`:

```typescript
const args = buildLaunchArgs({ recipeName, options });
```

(Keep everything below `args` — the `syncContainerImage` block, the `spawn` invocation, the log/exit handlers — exactly as it is.)

- [ ] **Step 4: Update the `cmd:deploy` case to read and forward `servedModelName`**

In `packages/agent/src/index.ts` around line 500, add `servedModelName` to the destructure:

```typescript
case "cmd:deploy": {
  const {
    deploymentId,
    recipeFile,
    config,
    clusterNodes,
    clusterNodeFastIps,
    runtime,
    modelName,
    modelType,
    servedModelName,
  } = msg.payload as {
    deploymentId: string;
    recipeFile?: string;
    config?: Record<string, unknown>;
    clusterNodes?: string[];
    clusterNodeFastIps?: (string | null)[];
    runtime?: string;
    modelName?: string;
    modelType?: "chat" | "embedding";
    /** Per-deploy override for vLLM's --served-model-name. */
    servedModelName?: string;
  };
```

Then in the call to `launchRecipe` (line 580), add `servedModelName` to the options:

```typescript
const port = launchRecipe(
  deploymentId,
  recipeFile,
  {
    port: (config?.port as number) ?? 8000,
    gpuMem: config?.gpuMem as number,
    maxModelLen: config?.maxModelLen as number,
    tensorParallel: config?.tensorParallel as number,
    pipelineParallel: config?.pipelineParallel as number,
    clusterNodes,
    clusterNodeFastIps,
    skipSetup: recipeModelIsLocal,
    servedModelName,
  },
  // ...existing log/exit handlers
);
```

- [ ] **Step 5: Run tests**

```bash
npx vitest run packages/agent/src/runtime/vllm.test.ts
```

Expected: PASS (both new cases).

- [ ] **Step 6: Bump agent version (required after ANY agent edit)**

```bash
./scripts/bump-agent-version.sh
```

Expected: prints "Bumped agent version to 0.5.X" where X is one higher than before.

- [ ] **Step 7: Run full suite**

```bash
npm test
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add packages/agent/src/runtime/vllm.ts packages/agent/src/runtime/vllm.test.ts packages/agent/src/index.ts packages/agent/package.json
git commit -m "agent: thread servedModelName through cmd:deploy → launchRecipe → --served-model-name"
```

---

## Task 7: Dashboard — add `displayName` input to deploy forms

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx:340-440`

There are three deploy code paths in this file: stock vLLM (line 403), finetune (line 344), and Ollama (line 393). We add the field to stock vLLM and finetune only — Ollama doesn't support per-deploy renames.

- [ ] **Step 1: Add state for the input near other form state**

Find the other `useState` declarations near the top of the component (search for `setTensorParallel` or `setMaxModelLen`). Add a new piece of state:

```tsx
const [customDisplayName, setCustomDisplayName] = useState<string>("");
```

- [ ] **Step 2: Add the input element to both vLLM-deploy form blocks in the modal**

Find the modal's form layout (it should be near the `selectedRecipe` / `selectedNode` controls in JSX). Above the "Deploy" button, in the vLLM and finetune sections, add the input:

```tsx
{(runtimeMode === "vllm" || runtimeMode === "finetune") && (
  <label className="flex flex-col gap-1 text-sm">
    <span className="text-zinc-400">
      Display name <span className="text-zinc-600">(optional)</span>
    </span>
    <input
      type="text"
      value={customDisplayName}
      onChange={(e) => setCustomDisplayName(e.target.value)}
      placeholder="e.g. chat3d-prod"
      pattern="[A-Za-z0-9._:\\-]*"
      maxLength={128}
      className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-xs"
    />
    <span className="text-xs text-zinc-500">
      Overrides the model name in this list and in the OpenAI API
      (<code>/v1/models</code>). Letters, digits, dot, dash, underscore, colon.
    </span>
  </label>
)}
```

(If the existing form uses a different className convention, adapt to match — copy from the nearest existing label/input pair.)

- [ ] **Step 3: Send `displayName` in both POST bodies**

In the finetune branch (around line 359-362), modify `ftBody`:

```typescript
const ftBody: Record<string, unknown> = { config };
if (needsClusterFt) ftBody.nodeIds = Array.from(selectedClusterNodes);
else ftBody.nodeId = selectedNode;
if (finetuneArtifactVariant) ftBody.artifactVariant = finetuneArtifactVariant;
const trimmedName = customDisplayName.trim();
if (trimmedName) ftBody.displayName = trimmedName;
```

In the stock vLLM branch (around line 413-421), modify `body`:

```typescript
const trimmedName = customDisplayName.trim();
body = needsCluster
  ? {
      nodeIds: Array.from(selectedClusterNodes),
      recipeFile: selectedRecipe,
      config: configOverrides,
      ...(trimmedName ? { displayName: trimmedName } : {}),
    }
  : {
      nodeId: selectedNode || "auto",
      recipeFile: selectedRecipe,
      config: configOverrides,
      ...(trimmedName ? { displayName: trimmedName } : {}),
    };
```

- [ ] **Step 4: Reset the input on successful deploy**

In both success-handling blocks (around lines 384-389 and 434-441), add `setCustomDisplayName("")` alongside the existing form resets.

- [ ] **Step 5: Surface server-side errors in the toast**

The existing `apiFetch` wrapper throws on non-2xx (verify by reading `lib/api.ts`). The catch block of the form-submit handler should already render the error message; double-check that a 400/409 with `error: "..."` from our handlers becomes a visible toast. If not, add:

```typescript
} catch (err) {
  toast.error("Deploy failed", { description: err instanceof Error ? err.message : String(err) });
}
```

- [ ] **Step 6: Manual smoke test (no automated test for the UI — call out in PR notes)**

Start dev mode (`npm run dev:dashboard`) and verify:
  - The input appears for vLLM + finetune flows, NOT for Ollama.
  - Submitting with a valid name produces a 201 and a deployment with that displayName.
  - Submitting with an invalid name (e.g. "foo bar") produces a visible 400 error.
  - Submitting a duplicate name produces a visible 409 error.

- [ ] **Step 7: Commit**

```bash
git add packages/dashboard/app/deployments/page.tsx
git commit -m "dashboard: add optional displayName input to vLLM + finetune deploy forms"
```

---

## Task 8: Dashboard — render `displayName ?? model.name` in deployment lists

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx` (lines 369, 429, 461, 1047)
- Modify: `packages/dashboard/app/page.tsx` (line 184)
- Modify (if present): the centralized `Deployment` TS type — likely in `packages/dashboard/lib/api.ts` or inline.

- [ ] **Step 1: Add `displayName` to the Deployment TS interface**

Find where `Deployment` is typed in the dashboard. Run:

```bash
grep -rn "interface Deployment\|type Deployment" packages/dashboard/ | head -5
```

In whichever file defines it, add:

```typescript
displayName?: string | null;
```

- [ ] **Step 2: Update the toast messages in deployments/page.tsx**

Line 369 (finetune branch) — change:

```typescript
toast.success(`Deployed ${result.model?.name ?? "fine-tuned model"}`, {
```

to:

```typescript
toast.success(`Deployed ${result.displayName ?? result.model?.name ?? "fine-tuned model"}`, {
```

Line 429 (stock vLLM/Ollama branch) — same change:

```typescript
toast.success(`Deployed ${deployment.displayName ?? deployment.model?.name ?? "model"}`, {
```

- [ ] **Step 3: Update the deployment-name labels**

Line 461 — change:

```typescript
const label = d ? `${d.model?.name || d.modelId} on ${d.node?.name || d.nodeId}` : id.slice(0, 12);
```

to:

```typescript
const label = d ? `${d.displayName ?? d.model?.name ?? d.modelId} on ${d.node?.name || d.nodeId}` : id.slice(0, 12);
```

Line 1047 (the actual deployments table cell):

```typescript
{d.displayName ?? d.model?.name ?? recipeName ?? d.modelId}
```

(If `d.displayName` is set AND it differs from `d.model?.name`, optionally render the underlying name as a small subtitle: that's a UX nicety, not required. Leave a TODO comment if you skip it.)

- [ ] **Step 4: Update the dashboard overview**

In `packages/dashboard/app/page.tsx` line 184:

```typescript
modelName: d.displayName ?? d.model?.name ?? "unknown",
```

- [ ] **Step 5: Manual smoke**

Start `npm run dev` and look at:
  - `/deployments` table — display name shown when set, falls back when not.
  - `/` dashboard overview — same.
  - Toast on deploy — same.

- [ ] **Step 6: Commit**

```bash
git add packages/dashboard/
git commit -m "dashboard: render Deployment.displayName ?? model.name everywhere"
```

---

## Task 9: End-to-end check + docs

**Files:**
- Modify (optional): `CLAUDE.md` if you want to document the new field for future agents.

- [ ] **Step 1: Full repo test run**

```bash
npm test
```

Expected: all green.

- [ ] **Step 2: Manual deploy through Docker Compose**

```bash
./scripts/build-agent-bundles.sh && \
  MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build
```

Then in the dashboard:
1. Launch a stock vLLM deploy with a recipe + `displayName=manual-test-1`. Confirm:
   - Dashboard shows `manual-test-1`.
   - `curl http://<host>:8000/v1/models | jq .data[].id` returns `manual-test-1`.
2. Restart that deployment without changing displayName. Confirm vLLM still publishes `manual-test-1`.
3. Restart with `{ "displayName": "manual-test-2" }`. Confirm dashboard + `/v1/models` both reflect `manual-test-2`.
4. Try to deploy a second model with `displayName=manual-test-2` → expect a 409.
5. For a fine-tune: deploy with `displayName=ft-renamed-once`, then a second time with `displayName=ft-renamed-twice`. Confirm two distinct deployments visible, both backed by the same merged model, but with different names in `/v1/models`.

- [ ] **Step 3: Final commit if any docs changed**

```bash
git add CLAUDE.md
git commit -m "docs: note Deployment.displayName behavior"
```

(Skip the commit if no doc change.)

---

## Self-Review Notes

**Spec coverage:**
- ✅ Custom display name persisted per deployment → Task 1 (schema) + Task 4 (POST persist) + Task 5 (finetune POST persist).
- ✅ Surfaces in dashboard deployment list → Task 8.
- ✅ Surfaces in OpenAI API (`/v1/models`) → Task 4 (passes `servedModelName`) + Task 6 (agent threads it to `--served-model-name`).
- ✅ Same fine-tune deployable twice under different names → Task 5 (per-deploy override doesn't touch Model.name).
- ✅ Settable only at deploy creation → Task 7 (form only at launch).
- ✅ Restart preserves + can override → Task 4 step 4.
- ✅ Uniqueness across active deployments → Task 3 + integration tests.
- ✅ vLLM only (stock + finetune) → Task 4 explicitly 400s for Ollama displayName.

**Type consistency:** `displayName` (camelCase) used everywhere. `servedModelName` reserved for the agent-side payload field (already in use). `modelName` retained for the existing finetune→agent payload (we override its value but not its name).

**Placeholder scan:** No `TODO`, no `// implement later`. Every step shows the code or the exact command.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-17-deployment-display-name.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
