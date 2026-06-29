# UI-Managed Sparkrun Registries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator manage the cluster's sparkrun recipe registries from the dgx-manager UI, with the manager DB as the single source of truth that pushes config to every node's `~/.config/sparkrun/registries.yaml`.

**Architecture:** A new `SparkrunRegistry` Prisma table holds the cluster-wide registry list. REST CRUD at `/api/registries` mutates it and broadcasts `cmd:set-registries` over the existing agent WebSocket. Each agent renders its `registries.yaml` (via a purpose-built, property-tested serializer — no YAML dependency), atomically writes it, re-runs `sparkrun list`, and reports the refreshed catalog back through the existing `agent:recipes` path. Agents also reconcile to the DB set on every (re)connect.

**Tech Stack:** TypeScript (ESM, strict), Prisma 7 + better-sqlite3 (`db push`, no migrations), Express 5, Next.js 15 / React 19, Vitest + fast-check + supertest.

## Global Constraints

- TypeScript strict mode, ES modules; intra-package imports use the `.js` extension.
- Prisma access is the shared singleton: `import { prisma } from "../prisma.js"`. Other deps come via `app.set(...)` / `req.app.get(...)`.
- Schema changes use `npm run db:generate` + `npm run db:push` — **never** create a migration. The new table is additive (non-destructive).
- **MANDATORY:** after editing any file under `packages/agent/src/`, run `./scripts/bump-agent-version.sh` once (before committing).
- The agent runtime must gain **no new production dependency** (lean cross-built bundle). `yaml` may be added only as a **root devDependency** for tests.
- Integration tests use a per-suite SQLite via `mkdtempSync` + `DATABASE_URL=file:/tmp/...` set BEFORE importing prisma, apply schema with `npx prisma db push --force-reset`, and set `PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION` per-suite only. Canonical reference: `packages/server/src/__tests__/integration/deployments.vram-admission.test.ts`.
- Run `npm test` green before claiming any task done.
- The wire contract `RegistryWire` is duplicated in server and agent (no shared package — mirrors the existing `VllmRecipe`/`Recipe` duplication). The two definitions and the `registries.yaml` key names (`tuning_subpath`, `benchmark_subpath`, `mods_subpath`) MUST stay in sync.

---

### Task 1: Add the `SparkrunRegistry` Prisma model

**Files:**
- Modify: `prisma/schema.prisma` (append a model)
- Test: `packages/server/src/__tests__/integration/registries-model.test.ts`

**Interfaces:**
- Produces: Prisma model `SparkrunRegistry` with fields `id, name, url, subpath, description, visible, tuningSubpath, benchmarkSubpath, modsSubpath, sortOrder, createdAt, updatedAt`. Client accessor: `prisma.sparkrunRegistry`.

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/__tests__/integration/registries-model.test.ts` (mirror the header of `deployments.vram-admission.test.ts` — temp DB + `db push --force-reset`):

```ts
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execSync } from "node:child_process";
import { beforeAll, afterAll, expect, it, describe } from "vitest";

const dir = mkdtempSync(join(tmpdir(), "reg-model-"));
process.env.DATABASE_URL = `file:${join(dir, "test.db")}`;
process.env.PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION =
  "I have been authorized by Daniel to run destructive Prisma operations in this per-suite test database.";

const { prisma } = await import("../../prisma.js");

beforeAll(() => {
  execSync("npx prisma db push --force-reset --skip-generate", {
    env: { ...process.env },
    stdio: "inherit",
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/registries-model.test.ts`
Expected: FAIL — `prisma.sparkrunRegistry` is undefined (model does not exist yet).

- [ ] **Step 3: Add the model to the schema**

Append to `prisma/schema.prisma`:

```prisma
model SparkrunRegistry {
  id               String   @id @default(cuid())
  name             String   @unique
  url              String
  subpath          String
  description      String?
  visible          Boolean  @default(true)
  tuningSubpath    String?
  benchmarkSubpath String?
  modsSubpath      String?
  sortOrder        Int      @default(0)
  createdAt        DateTime @default(now())
  updatedAt        DateTime @updatedAt
}
```

- [ ] **Step 4: Regenerate the client and push to the dev DB**

Run: `npm run db:generate && npm run db:push`
Expected: client regenerated under `packages/server/src/generated/prisma/`; `db push` reports the new table created, no data loss prompt.

- [ ] **Step 5: Run test to verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/registries-model.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add prisma/schema.prisma packages/server/src/generated/prisma packages/server/src/__tests__/integration/registries-model.test.ts
git commit -m "feat(server): add SparkrunRegistry model"
```

---

### Task 2: Pure wire mapper `registryRowsToWire`

**Files:**
- Create: `packages/server/src/registries/wire.ts`
- Test: `packages/server/src/registries/wire.test.ts`

**Interfaces:**
- Produces:
  - `interface RegistryWire { name: string; url: string; subpath: string; description?: string; visible?: boolean; tuning_subpath?: string; benchmark_subpath?: string; mods_subpath?: string }`
  - `registryRowsToWire(rows): RegistryWire[]` — maps DB rows (camelCase) to the snake_case wire/`registries.yaml` shape, ordered by `sortOrder` then `name`, omitting null optionals, and including `visible` only when `false`.

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/registries/wire.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { registryRowsToWire } from "./wire.js";

const row = (over: Record<string, unknown> = {}) => ({
  id: "x", name: "eugr", url: "https://github.com/eugr/spark-vllm-docker", subpath: "recipes",
  description: null, visible: true, tuningSubpath: null, benchmarkSubpath: null, modsSubpath: "mods",
  sortOrder: 0, createdAt: new Date(0), updatedAt: new Date(0), ...over,
});

describe("registryRowsToWire", () => {
  it("maps camelCase rows to snake_case wire shape, omitting null optionals", () => {
    expect(registryRowsToWire([row()])).toEqual([
      { name: "eugr", url: "https://github.com/eugr/spark-vllm-docker", subpath: "recipes", mods_subpath: "mods" },
    ]);
  });

  it("includes visible only when false", () => {
    expect(registryRowsToWire([row({ visible: false })])[0].visible).toBe(false);
    expect(registryRowsToWire([row({ visible: true })])[0]).not.toHaveProperty("visible");
  });

  it("orders by sortOrder then name", () => {
    const out = registryRowsToWire([row({ name: "b", sortOrder: 1 }), row({ name: "a", sortOrder: 1 }), row({ name: "z", sortOrder: 0 })]);
    expect(out.map((r) => r.name)).toEqual(["z", "a", "b"]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/registries/wire.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `wire.ts`**

```ts
export interface RegistryWire {
  name: string;
  url: string;
  subpath: string;
  description?: string;
  visible?: boolean;
  tuning_subpath?: string;
  benchmark_subpath?: string;
  mods_subpath?: string;
}

interface RegistryRow {
  name: string;
  url: string;
  subpath: string;
  description: string | null;
  visible: boolean;
  tuningSubpath: string | null;
  benchmarkSubpath: string | null;
  modsSubpath: string | null;
  sortOrder: number;
}

export function registryRowsToWire(rows: RegistryRow[]): RegistryWire[] {
  return [...rows]
    .sort((a, b) => a.sortOrder - b.sortOrder || a.name.localeCompare(b.name))
    .map((r) => {
      const w: RegistryWire = { name: r.name, url: r.url, subpath: r.subpath };
      if (r.description != null) w.description = r.description;
      if (r.visible === false) w.visible = false;
      if (r.tuningSubpath != null) w.tuning_subpath = r.tuningSubpath;
      if (r.benchmarkSubpath != null) w.benchmark_subpath = r.benchmarkSubpath;
      if (r.modsSubpath != null) w.mods_subpath = r.modsSubpath;
      return w;
    });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/registries/wire.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/registries/wire.ts packages/server/src/registries/wire.test.ts
git commit -m "feat(server): pure registryRowsToWire mapper"
```

---

### Task 3: Pure validator `validateRegistry`

**Files:**
- Create: `packages/server/src/registries/validate.ts`
- Test: `packages/server/src/registries/validate.test.ts`

**Interfaces:**
- Produces:
  - `interface RegistryInput { name; url; subpath; description?; visible?; tuningSubpath?; benchmarkSubpath?; modsSubpath?; sortOrder? }`
  - `type ValidationResult = { ok: true; value: Required-ish RegistryInput } | { ok: false; error: string }`
  - `validateRegistry(input: unknown): ValidationResult` — boundary validation, fail-fast.

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/registries/validate.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { validateRegistry } from "./validate.js";

const ok = { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes" };

describe("validateRegistry", () => {
  it("accepts a minimal valid registry", () => {
    const r = validateRegistry(ok);
    expect(r.ok).toBe(true);
  });

  it.each([
    ["bad name (uppercase)", { ...ok, name: "RTX" }],
    ["bad name (space)", { ...ok, name: "my reg" }],
    ["empty name", { ...ok, name: "" }],
    ["bad url scheme", { ...ok, url: "file:///etc/passwd" }],
    ["empty subpath", { ...ok, subpath: "" }],
    ["path traversal subpath", { ...ok, subpath: "../secrets" }],
    ["non-object", null],
  ])("rejects %s", (_label, input) => {
    const r = validateRegistry(input);
    expect(r.ok).toBe(false);
  });

  it("accepts scp-style git url", () => {
    expect(validateRegistry({ ...ok, url: "git@github.com:org/repo.git" }).ok).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/registries/validate.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `validate.ts`**

```ts
export interface RegistryInput {
  name: string;
  url: string;
  subpath: string;
  description?: string | null;
  visible?: boolean;
  tuningSubpath?: string | null;
  benchmarkSubpath?: string | null;
  modsSubpath?: string | null;
  sortOrder?: number;
}

export type ValidationResult =
  | { ok: true; value: RegistryInput }
  | { ok: false; error: string };

const NAME_RE = /^[a-z0-9-]+$/;
const URL_RE = /^(https?:\/\/|git:\/\/|ssh:\/\/|git@)/;

function badPath(s: string): boolean {
  return s.length === 0 || s.startsWith("/") || s.split("/").includes("..");
}

export function validateRegistry(input: unknown): ValidationResult {
  if (typeof input !== "object" || input === null) return { ok: false, error: "Body must be an object" };
  const i = input as Record<string, unknown>;

  if (typeof i.name !== "string" || !NAME_RE.test(i.name))
    return { ok: false, error: "name must match ^[a-z0-9-]+$" };
  if (typeof i.url !== "string" || !URL_RE.test(i.url))
    return { ok: false, error: "url must be an http(s)/git/ssh URL" };
  if (typeof i.subpath !== "string" || badPath(i.subpath))
    return { ok: false, error: "subpath must be a non-empty relative path (no '..' or leading '/')" };

  for (const k of ["description", "tuningSubpath", "benchmarkSubpath", "modsSubpath"] as const) {
    if (i[k] != null && typeof i[k] !== "string") return { ok: false, error: `${k} must be a string` };
  }
  for (const k of ["tuningSubpath", "benchmarkSubpath", "modsSubpath"] as const) {
    if (typeof i[k] === "string" && badPath(i[k] as string)) return { ok: false, error: `${k} must be a relative path` };
  }
  if (i.visible != null && typeof i.visible !== "boolean") return { ok: false, error: "visible must be a boolean" };
  if (i.sortOrder != null && typeof i.sortOrder !== "number") return { ok: false, error: "sortOrder must be a number" };

  return {
    ok: true,
    value: {
      name: i.name,
      url: i.url,
      subpath: i.subpath,
      description: (i.description as string | null) ?? null,
      visible: i.visible == null ? true : (i.visible as boolean),
      tuningSubpath: (i.tuningSubpath as string | null) ?? null,
      benchmarkSubpath: (i.benchmarkSubpath as string | null) ?? null,
      modsSubpath: (i.modsSubpath as string | null) ?? null,
      sortOrder: (i.sortOrder as number | undefined) ?? 0,
    },
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/registries/validate.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/registries/validate.ts packages/server/src/registries/validate.test.ts
git commit -m "feat(server): pure validateRegistry boundary validator"
```

---

### Task 4: Default-registry seed (`seedDefaultRegistries`)

**Files:**
- Create: `packages/server/src/registries/seed.ts`
- Test: `packages/server/src/__tests__/integration/registries-seed.test.ts`

**Interfaces:**
- Consumes: `prisma.sparkrunRegistry` (Task 1).
- Produces: `DEFAULT_REGISTRIES` (array of 7 seed objects) and `seedDefaultRegistries(prisma): Promise<number>` (returns number inserted; 0 when table already populated).

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/__tests__/integration/registries-seed.test.ts` (same temp-DB header as Task 1):

```ts
// ... identical temp-DB + db push --force-reset header as registries-model.test.ts ...
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/registries-seed.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `seed.ts`**

Values copied verbatim from the live `~/.config/sparkrun/registries.yaml`:

```ts
import type { PrismaClient } from "../generated/prisma/client.js";

export const DEFAULT_REGISTRIES = [
  { name: "sparkrun-testing", url: "https://github.com/dbotwinick/sparkrun-recipe-registry.git", subpath: "testing/recipes",
    description: "Sparkrun testing registry for recipes, tuning configs, and benchmark profiles", visible: false,
    tuningSubpath: "testing/tuning", benchmarkSubpath: "testing/benchmarking", modsSubpath: null, sortOrder: 0 },
  { name: "sparkrun-transitional", url: "https://github.com/dbotwinick/sparkrun-recipe-registry.git", subpath: "transitional/recipes",
    description: "Transitional registry for recipes", visible: true,
    tuningSubpath: "testing/tuning", benchmarkSubpath: null, modsSubpath: null, sortOrder: 1 },
  { name: "official", url: "https://github.com/spark-arena/recipe-registry.git", subpath: "official-recipes",
    description: "Official Spark Arena registry for recipes, tuning configs, and benchmark profiles", visible: true,
    tuningSubpath: "tuning", benchmarkSubpath: "benchmarking", modsSubpath: "official-mods", sortOrder: 2 },
  { name: "experimental", url: "https://github.com/spark-arena/recipe-registry.git", subpath: "experimental-recipes",
    description: "Spark Arena registry for experimental recipes", visible: false,
    tuningSubpath: null, benchmarkSubpath: null, modsSubpath: "experimental-mods", sortOrder: 3 },
  { name: "community", url: "https://github.com/spark-arena/community-recipe-registry.git", subpath: "recipes",
    description: "Community registry for sparkrun", visible: false,
    tuningSubpath: "tuning", benchmarkSubpath: "benchmarking", modsSubpath: null, sortOrder: 4 },
  { name: "eugr", url: "https://github.com/eugr/spark-vllm-docker", subpath: "recipes",
    description: "Official eugr/spark-vllm-docker repo recipes", visible: true,
    tuningSubpath: null, benchmarkSubpath: null, modsSubpath: "mods", sortOrder: 5 },
  { name: "atlas", url: "https://github.com/Avarok-Cybersecurity/atlas-recipes.git", subpath: "recipes",
    description: "Atlas recipes", visible: false,
    tuningSubpath: null, benchmarkSubpath: null, modsSubpath: null, sortOrder: 6 },
] as const;

/** Insert the standard registries only when the table is empty. Returns rows inserted. */
export async function seedDefaultRegistries(prisma: PrismaClient): Promise<number> {
  const count = await prisma.sparkrunRegistry.count();
  if (count > 0) return 0;
  await prisma.sparkrunRegistry.createMany({ data: DEFAULT_REGISTRIES.map((r) => ({ ...r })) });
  return DEFAULT_REGISTRIES.length;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/registries-seed.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/registries/seed.ts packages/server/src/__tests__/integration/registries-seed.test.ts
git commit -m "feat(server): idempotent default-registry seed"
```

---

### Task 5: Push helpers (`pushRegistriesToConnectedAgents`)

**Files:**
- Create: `packages/server/src/registries/push.ts`
- Test: `packages/server/src/registries/push.test.ts`

**Interfaces:**
- Consumes: `prisma.sparkrunRegistry`, `registryRowsToWire` (Task 2).
- Produces:
  - `interface AgentSink { sendToAgent(nodeId: string, msg: Record<string, unknown>): void; getConnectedNodeIds(): string[] }`
  - `loadRegistryWire(): Promise<RegistryWire[]>` — read DB, map to wire.
  - `pushRegistriesToAgent(sink, nodeId): Promise<void>` — send `cmd:set-registries` to one node.
  - `pushRegistriesToConnectedAgents(sink): Promise<void>` — send to all connected nodes.
- The wire message shape: `{ type: "cmd:set-registries", payload: { registries: RegistryWire[] } }`.

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/registries/push.test.ts` (uses a fake sink + spies on the loader via a real per-suite DB is overkill; instead inject a fake `prisma`-less loader by testing the sink-fanout with a stubbed `loadRegistryWire`). Test the pure fanout against a fake sink:

```ts
import { describe, it, expect, vi } from "vitest";
import * as push from "./push.js";

function fakeSink() {
  const sent: { nodeId: string; msg: Record<string, unknown> }[] = [];
  return {
    sent,
    sendToAgent: (nodeId: string, msg: Record<string, unknown>) => sent.push({ nodeId, msg }),
    getConnectedNodeIds: () => ["node-a", "node-b"],
  };
}

describe("pushRegistriesToConnectedAgents", () => {
  it("sends cmd:set-registries to every connected node", async () => {
    vi.spyOn(push, "loadRegistryWire").mockResolvedValue([
      { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes" },
    ]);
    const sink = fakeSink();
    await push.pushRegistriesToConnectedAgents(sink);
    expect(sink.sent.map((s) => s.nodeId)).toEqual(["node-a", "node-b"]);
    expect(sink.sent[0].msg).toEqual({
      type: "cmd:set-registries",
      payload: { registries: [{ name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes" }] },
    });
  });
});
```

> Note: `vi.spyOn(push, "loadRegistryWire")` requires the functions to call each other through the module object. Implement `pushRegistriesToAgent`/`pushRegistriesToConnectedAgents` to reference `loadRegistryWire` via the module namespace import (see Step 3), so the spy is observed.

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/registries/push.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `push.ts`**

```ts
import { prisma } from "../prisma.js";
import { registryRowsToWire, type RegistryWire } from "./wire.js";

export interface AgentSink {
  sendToAgent(nodeId: string, msg: Record<string, unknown>): void;
  getConnectedNodeIds(): string[];
}

// Exported (not destructured) so tests can spy on it via the module namespace.
export async function loadRegistryWire(): Promise<RegistryWire[]> {
  const rows = await prisma.sparkrunRegistry.findMany();
  return registryRowsToWire(rows);
}

export async function pushRegistriesToAgent(sink: AgentSink, nodeId: string): Promise<void> {
  const registries = await mod.loadRegistryWire();
  sink.sendToAgent(nodeId, { type: "cmd:set-registries", payload: { registries } });
}

export async function pushRegistriesToConnectedAgents(sink: AgentSink): Promise<void> {
  const registries = await mod.loadRegistryWire();
  for (const nodeId of sink.getConnectedNodeIds()) {
    sink.sendToAgent(nodeId, { type: "cmd:set-registries", payload: { registries } });
  }
}

import * as mod from "./push.js";
```

> The `import * as mod from "./push.js"` self-import lets the spy in Step 1 intercept `loadRegistryWire`. ESM circular self-import is resolved lazily at call time, so this is safe.

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/registries/push.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/registries/push.ts packages/server/src/registries/push.test.ts
git commit -m "feat(server): registry push-to-agent helpers"
```

---

### Task 6: REST route `/api/registries` + boot seed + mount

**Files:**
- Create: `packages/server/src/routes/registries.ts`
- Modify: `packages/server/src/index.ts` (import + mount router; call `seedDefaultRegistries` at boot)
- Test: `packages/server/src/__tests__/integration/registries.test.ts`

**Interfaces:**
- Consumes: `validateRegistry` (Task 3), `pushRegistriesToConnectedAgents` (Task 5), `prisma.sparkrunRegistry`.
- Produces: `registriesRouter` with `GET /`, `POST /`, `PATCH /:id`, `DELETE /:id`. Every mutation calls `pushRegistriesToConnectedAgents(req.app.get("agentHub"))`.

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/__tests__/integration/registries.test.ts` (temp-DB header as Task 1, plus express + supertest + a stub agentHub):

```ts
// ... temp-DB header (DATABASE_URL + consent + db push --force-reset in beforeAll) ...
import express from "express";
import request from "supertest";
import { registriesRouter } from "../../routes/registries.js";
import { seedDefaultRegistries } from "../../registries/seed.js";

function appWithStubHub() {
  const sent: { nodeId: string; msg: Record<string, unknown> }[] = [];
  const agentHub = {
    sendToAgent: (nodeId: string, msg: Record<string, unknown>) => sent.push({ nodeId, msg }),
    getConnectedNodeIds: () => ["node-a"],
  };
  const app = express();
  app.use(express.json());
  app.set("agentHub", agentHub);
  app.use("/api/registries", registriesRouter);
  return { app, sent };
}

beforeEach(async () => { await prisma.sparkrunRegistry.deleteMany(); });

describe("/api/registries", () => {
  it("POST creates, persists, and broadcasts cmd:set-registries", async () => {
    const { app, sent } = appWithStubHub();
    const res = await request(app).post("/api/registries").send({
      name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes",
    });
    expect(res.status).toBe(201);
    expect(await prisma.sparkrunRegistry.count()).toBe(1);
    expect(sent).toHaveLength(1);
    expect(sent[0].msg.type).toBe("cmd:set-registries");
  });

  it("POST rejects invalid input with 400 and does NOT broadcast", async () => {
    const { app, sent } = appWithStubHub();
    const res = await request(app).post("/api/registries").send({ name: "BAD NAME", url: "x", subpath: "" });
    expect(res.status).toBe(400);
    expect(sent).toHaveLength(0);
    expect(await prisma.sparkrunRegistry.count()).toBe(0);
  });

  it("POST rejects a duplicate name with 409", async () => {
    const { app } = appWithStubHub();
    const body = { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes" };
    await request(app).post("/api/registries").send(body);
    const res = await request(app).post("/api/registries").send(body);
    expect(res.status).toBe(409);
  });

  it("GET lists seeded registries; DELETE removes and broadcasts", async () => {
    const { app, sent } = appWithStubHub();
    await seedDefaultRegistries(prisma);
    const list = await request(app).get("/api/registries");
    expect(list.body.length).toBeGreaterThanOrEqual(7);
    const atlas = list.body.find((r: { name: string }) => r.name === "atlas");
    const del = await request(app).delete(`/api/registries/${atlas.id}`);
    expect(del.status).toBe(200);
    expect(sent.at(-1)!.msg.type).toBe("cmd:set-registries");
    expect(await prisma.sparkrunRegistry.findUnique({ where: { name: "atlas" } })).toBeNull();
  });

  it("PATCH updates fields and broadcasts", async () => {
    const { app, sent } = appWithStubHub();
    const created = await request(app).post("/api/registries").send({
      name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes",
    });
    const res = await request(app).patch(`/api/registries/${created.body.id}`).send({ visible: false });
    expect(res.status).toBe(200);
    expect(res.body.visible).toBe(false);
    expect(sent.at(-1)!.msg.type).toBe("cmd:set-registries");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/registries.test.ts`
Expected: FAIL — `routes/registries.js` not found.

- [ ] **Step 3: Implement `routes/registries.ts`**

```ts
import { Router } from "express";
import { prisma } from "../prisma.js";
import { validateRegistry } from "../registries/validate.js";
import { pushRegistriesToConnectedAgents, type AgentSink } from "../registries/push.js";

export const registriesRouter = Router();

/**
 * @openapi
 * /api/registries:
 *   get: { tags: [Registries], summary: List sparkrun registries }
 *   post: { tags: [Registries], summary: Create a registry (pushes to all online nodes) }
 */
registriesRouter.get("/", async (_req, res) => {
  const rows = await prisma.sparkrunRegistry.findMany({ orderBy: [{ sortOrder: "asc" }, { name: "asc" }] });
  res.json(rows);
});

registriesRouter.post("/", async (req, res) => {
  const v = validateRegistry(req.body);
  if (!v.ok) return res.status(400).json({ error: v.error });
  const existing = await prisma.sparkrunRegistry.findUnique({ where: { name: v.value.name } });
  if (existing) return res.status(409).json({ error: `Registry '${v.value.name}' already exists` });
  const row = await prisma.sparkrunRegistry.create({ data: v.value });
  await pushRegistriesToConnectedAgents(req.app.get("agentHub") as AgentSink);
  res.status(201).json(row);
});

registriesRouter.patch("/:id", async (req, res) => {
  const current = await prisma.sparkrunRegistry.findUnique({ where: { id: req.params.id } });
  if (!current) return res.status(404).json({ error: "Not found" });
  // Validate the merged record so partial updates can't produce an invalid row.
  const merged = { ...current, ...req.body };
  const v = validateRegistry(merged);
  if (!v.ok) return res.status(400).json({ error: v.error });
  if (v.value.name !== current.name) {
    const clash = await prisma.sparkrunRegistry.findUnique({ where: { name: v.value.name } });
    if (clash) return res.status(409).json({ error: `Registry '${v.value.name}' already exists` });
  }
  const row = await prisma.sparkrunRegistry.update({ where: { id: req.params.id }, data: v.value });
  await pushRegistriesToConnectedAgents(req.app.get("agentHub") as AgentSink);
  res.json(row);
});

registriesRouter.delete("/:id", async (req, res) => {
  await prisma.sparkrunRegistry.delete({ where: { id: req.params.id } }).catch(() => {});
  await pushRegistriesToConnectedAgents(req.app.get("agentHub") as AgentSink);
  res.json({ status: "deleted" });
});
```

- [ ] **Step 4: Mount the router and seed at boot in `index.ts`**

Add the import alongside the other route imports (near line 21):

```ts
import { registriesRouter } from "./routes/registries.js";
import { seedDefaultRegistries } from "./registries/seed.js";
```

Add the mount alongside the other `app.use` lines (near line 71):

```ts
app.use("/api/registries", registriesRouter);
```

Add the boot seed next to the existing `prisma.benchmarkRun.updateMany(...)` startup block (defensive — never crash boot if the table is missing because `db push` hasn't run yet):

```ts
try {
  const seeded = await seedDefaultRegistries(prisma);
  if (seeded > 0) console.log(`Seeded ${seeded} default sparkrun registries`);
} catch (err) {
  console.error("Skipping registry seed (table not ready — run `npm run db:push`):", err);
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run packages/server/src/__tests__/integration/registries.test.ts`
Expected: PASS (all 5 cases).

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/routes/registries.ts packages/server/src/index.ts packages/server/src/__tests__/integration/registries.test.ts
git commit -m "feat(server): /api/registries CRUD + boot seed + push on mutate"
```

---

### Task 7: Reconcile registries to each agent on (re)connect

**Files:**
- Modify: `packages/server/src/ws/agent-hub.ts` (call `pushRegistriesToAgent` after `agent:register` and `agent:register-token`)
- Test: `packages/server/src/registries/push.test.ts` (extend with a single-node test)

**Interfaces:**
- Consumes: `pushRegistriesToAgent` (Task 5). `AgentHub` already satisfies `AgentSink` (`sendToAgent`, `getConnectedNodeIds`).

- [ ] **Step 1: Write the failing test**

Append to `packages/server/src/registries/push.test.ts`:

```ts
describe("pushRegistriesToAgent", () => {
  it("sends cmd:set-registries to exactly one node", async () => {
    vi.spyOn(push, "loadRegistryWire").mockResolvedValue([
      { name: "eugr", url: "https://github.com/eugr/spark-vllm-docker", subpath: "recipes" },
    ]);
    const sink = fakeSink();
    await push.pushRegistriesToAgent(sink, "node-a");
    expect(sink.sent).toEqual([
      { nodeId: "node-a", msg: { type: "cmd:set-registries", payload: { registries: [{ name: "eugr", url: "https://github.com/eugr/spark-vllm-docker", subpath: "recipes" }] } } },
    ]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/registries/push.test.ts`
Expected: PASS for the new test if Task 5 implemented `pushRegistriesToAgent` correctly — if it already passes, this step just confirms the contract; proceed to wire the hub.

- [ ] **Step 3: Wire the reconcile call into `agent-hub.ts`**

Add the import near the other imports at the top of `ws/agent-hub.ts`:

```ts
import { pushRegistriesToAgent } from "../registries/push.js";
```

In the `case "agent:register":` block, after the `prisma.node.update(...)` and `console.log(...)` (around line 195), add:

```ts
// Reconcile this node's sparkrun registries to the manager's source-of-truth set.
await pushRegistriesToAgent(this, nodeId!).catch((err) =>
  console.error(`Failed to push registries to ${nodeId}:`, err),
);
```

In the `case "agent:register-token":` block, after `this.agents.set(nodeId, { ws, nodeId })` and the acceptance send (around line 281), add the same call (note `nodeId` is non-null here):

```ts
await pushRegistriesToAgent(this, nodeId).catch((err) =>
  console.error(`Failed to push registries to ${nodeId}:`, err),
);
```

- [ ] **Step 4: Run the server test suite**

Run: `npx vitest run packages/server`
Expected: PASS (no regressions; agent-hub compiles with the new call).

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/ws/agent-hub.ts packages/server/src/registries/push.test.ts
git commit -m "feat(server): reconcile registries to agents on (re)connect"
```

---

### Task 8: Agent YAML renderer `renderRegistriesYaml` (pure, property-tested)

**Files:**
- Create: `packages/agent/src/registries.ts` (renderer + types only in this task)
- Test: `packages/agent/src/registries.test.ts`
- Modify: root `package.json` (add `yaml` as a **devDependency** for the test only)

**Interfaces:**
- Produces:
  - `interface RegistryWire { name; url; subpath; description?; visible?; tuning_subpath?; benchmark_subpath?; mods_subpath? }` (mirror of the server's; keep in sync).
  - `renderRegistriesYaml(registries: RegistryWire[]): string` — emits a sparkrun-compatible `registries.yaml`. All string values double-quoted with `\`, `"`, and newline escaped; `visible` emitted only when `false`; null/undefined optionals omitted.

- [ ] **Step 1: Add `yaml` as a root devDependency (test-only parser)**

Run: `npm install -D yaml`
Expected: `yaml` appears under root `devDependencies`. (Agent runtime gains no production dep.)

- [ ] **Step 2: Write the failing test**

Create `packages/agent/src/registries.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { test as propTest, fc } from "@fast-check/vitest";
import { parse } from "yaml";
import { renderRegistriesYaml, type RegistryWire } from "./registries.js";

describe("renderRegistriesYaml", () => {
  it("renders a known registry to parseable YAML", () => {
    const out = renderRegistriesYaml([
      { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes", description: "amd64 RTX", visible: false, tuning_subpath: "tuning" },
    ]);
    expect(out.startsWith("registries:\n")).toBe(true);
    expect(parse(out)).toEqual({
      registries: [
        { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes", description: "amd64 RTX", visible: false, tuning_subpath: "tuning" },
      ],
    });
  });

  it("escapes quotes and backslashes in descriptions", () => {
    const out = renderRegistriesYaml([
      { name: "x", url: "https://h/r.git", subpath: "recipes", description: 'has "quotes" and \\ slash' },
    ]);
    expect(parse(out).registries[0].description).toBe('has "quotes" and \\ slash');
  });

  /**
   * Invariant: rendering an arbitrary registry list then YAML-parsing it yields
   * back exactly the same logical data — name/url/subpath always present, optional
   * string fields preserved verbatim (escaping is correct), and `visible` is `false`
   * in the parsed output iff it was `false` in the input.
   */
  propTest.prop([
    fc.array(
      fc.record({
        name: fc.string({ minLength: 1 }),
        url: fc.string({ minLength: 1 }),
        subpath: fc.string({ minLength: 1 }),
        description: fc.option(fc.string(), { nil: undefined }),
        visible: fc.option(fc.boolean(), { nil: undefined }),
        tuning_subpath: fc.option(fc.string({ minLength: 1 }), { nil: undefined }),
      }),
      { maxLength: 8 },
    ),
  ])("round-trips through a real YAML parser", (regs: RegistryWire[]) => {
    const parsed = parse(renderRegistriesYaml(regs)) as { registries: RegistryWire[] };
    expect(parsed.registries).toHaveLength(regs.length);
    regs.forEach((r, i) => {
      const p = parsed.registries[i];
      expect(p.name).toBe(r.name);
      expect(p.url).toBe(r.url);
      expect(p.subpath).toBe(r.subpath);
      if (r.description != null) expect(p.description).toBe(r.description);
      if (r.tuning_subpath != null) expect(p.tuning_subpath).toBe(r.tuning_subpath);
      expect(p.visible === false).toBe(r.visible === false);
    });
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/registries.test.ts`
Expected: FAIL — `./registries.js` not found.

- [ ] **Step 4: Implement the renderer in `registries.ts`**

```ts
export interface RegistryWire {
  name: string;
  url: string;
  subpath: string;
  description?: string;
  visible?: boolean;
  tuning_subpath?: string;
  benchmark_subpath?: string;
  mods_subpath?: string;
}

/** Double-quote a scalar with YAML-safe escaping (backslash, quote, newline). */
function q(s: string): string {
  return '"' + s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n") + '"';
}

/** Render a sparkrun-compatible registries.yaml. Optional null/undefined fields
 *  are omitted; `visible` is emitted only when false (sparkrun defaults to true). */
export function renderRegistriesYaml(registries: RegistryWire[]): string {
  const lines: string[] = ["registries:"];
  for (const r of registries) {
    lines.push(`- name: ${q(r.name)}`);
    lines.push(`  url: ${q(r.url)}`);
    lines.push(`  subpath: ${q(r.subpath)}`);
    if (r.description != null) lines.push(`  description: ${q(r.description)}`);
    if (r.visible === false) lines.push(`  visible: false`);
    if (r.tuning_subpath != null) lines.push(`  tuning_subpath: ${q(r.tuning_subpath)}`);
    if (r.benchmark_subpath != null) lines.push(`  benchmark_subpath: ${q(r.benchmark_subpath)}`);
    if (r.mods_subpath != null) lines.push(`  mods_subpath: ${q(r.mods_subpath)}`);
  }
  return lines.join("\n") + "\n";
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/registries.test.ts`
Expected: PASS (unit + property cases).

- [ ] **Step 6: Commit** (do NOT bump agent version yet — Task 9 adds the IO + handler and bumps once)

```bash
git add packages/agent/src/registries.ts packages/agent/src/registries.test.ts package.json package-lock.json
git commit -m "feat(agent): pure registries.yaml renderer + property test"
```

---

### Task 9: Agent file writer + `cmd:set-registries` handler

**Files:**
- Modify: `packages/agent/src/registries.ts` (add `registriesConfigPath` + `writeRegistriesFile`)
- Modify: `packages/agent/src/index.ts` (add `cmd:set-registries` case to the command switch)
- Test: `packages/agent/src/registries-write.test.ts`
- Run: `./scripts/bump-agent-version.sh`

**Interfaces:**
- Consumes: `renderRegistriesYaml` (Task 8), `discoverRecipes` + `sendMsg` (existing in `index.ts`).
- Produces:
  - `registriesConfigPath(): string` → `<HOME>/.config/sparkrun/registries.yaml`.
  - `writeRegistriesFile(registries: RegistryWire[]): void` — atomic write (mkdir -p, temp file + rename).
  - New command handler `cmd:set-registries` that writes the file, re-discovers, and emits `agent:recipes` (mirrors `cmd:rescan-recipes` at `index.ts:976`).

- [ ] **Step 1: Write the failing test**

Create `packages/agent/src/registries-write.test.ts`:

```ts
import { describe, it, expect, beforeEach } from "vitest";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parse } from "yaml";

describe("writeRegistriesFile", () => {
  beforeEach(() => {
    process.env.HOME = mkdtempSync(join(tmpdir(), "agent-home-"));
  });

  it("writes <HOME>/.config/sparkrun/registries.yaml atomically and parseably", async () => {
    const { writeRegistriesFile, registriesConfigPath } = await import("./registries.js?write");
    writeRegistriesFile([
      { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes" },
    ]);
    const path = registriesConfigPath();
    expect(path).toBe(join(process.env.HOME!, ".config", "sparkrun", "registries.yaml"));
    const parsed = parse(readFileSync(path, "utf8"));
    expect(parsed.registries[0].name).toBe("rtx");
  });
});
```

> The `?write` query suffix dodges ESM module caching so each test re-reads `process.env.HOME`. If `registriesConfigPath` computes `HOME` lazily at call time (see Step 2), you can drop the suffix and import normally.

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/agent/src/registries-write.test.ts`
Expected: FAIL — `writeRegistriesFile` not exported.

- [ ] **Step 3: Add the IO functions to `registries.ts`**

Append to `packages/agent/src/registries.ts`:

```ts
import { mkdirSync, writeFileSync, renameSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

/** Path sparkrun reads its registry list from (computed lazily so HOME is honored). */
export function registriesConfigPath(): string {
  return join(homedir(), ".config", "sparkrun", "registries.yaml");
}

/** Atomically write registries.yaml: render → temp file → rename. */
export function writeRegistriesFile(registries: RegistryWire[]): void {
  const path = registriesConfigPath();
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, renderRegistriesYaml(registries), "utf8");
  renameSync(tmp, path);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/agent/src/registries-write.test.ts`
Expected: PASS.

- [ ] **Step 5: Add the `cmd:set-registries` handler to `index.ts`**

Add the import near the existing `import { discoverRecipes } from "./recipes.js";` (line 8):

```ts
import { writeRegistriesFile, type RegistryWire } from "./registries.js";
```

Add a new case to the command switch, immediately before `case "cmd:rescan-recipes":` (around line 976):

```ts
case "cmd:set-registries": {
  const registries = (msg.payload?.registries ?? []) as RegistryWire[];
  try {
    writeRegistriesFile(registries);
    const recipes = discoverRecipes();
    sendMsg("agent:recipes", { recipes });
    console.log(`Applied ${registries.length} registries; re-discovered ${recipes.length} recipes`);
  } catch (err) {
    console.error("cmd:set-registries failed:", err);
  }
  break;
}
```

- [ ] **Step 6: Bump the agent version (MANDATORY)**

Run: `./scripts/bump-agent-version.sh`
Expected: `packages/agent/package.json` patch version incremented.

- [ ] **Step 7: Run the agent test suite**

Run: `npx vitest run packages/agent`
Expected: PASS (renderer, writer, plus existing agent tests).

- [ ] **Step 8: Commit**

```bash
git add packages/agent/src/registries.ts packages/agent/src/registries-write.test.ts packages/agent/src/index.ts packages/agent/package.json
git commit -m "feat(agent): cmd:set-registries writes registries.yaml + re-discovers"
```

---

### Task 10: Dashboard — Registries management section

**Files:**
- Modify: `packages/dashboard/lib/api.ts` (registry types + CRUD fetchers)
- Create: `packages/dashboard/components/registries-section.tsx`
- Modify: `packages/dashboard/app/settings/page.tsx` (render `<RegistriesSection />`)

**Interfaces:**
- Consumes: `GET/POST/PATCH/DELETE /api/registries` (Task 6).
- Produces: `SparkrunRegistry` type + `listRegistries/createRegistry/updateRegistry/deleteRegistry` in `lib/api.ts`; `<RegistriesSection />` client component.

- [ ] **Step 1: Add API helpers to `lib/api.ts`**

Append to `packages/dashboard/lib/api.ts`:

```ts
export interface SparkrunRegistry {
  id: string;
  name: string;
  url: string;
  subpath: string;
  description: string | null;
  visible: boolean;
  tuningSubpath: string | null;
  benchmarkSubpath: string | null;
  modsSubpath: string | null;
  sortOrder: number;
}

export type NewRegistry = Pick<SparkrunRegistry, "name" | "url" | "subpath"> &
  Partial<Pick<SparkrunRegistry, "description" | "visible">>;

export const listRegistries = () => apiFetch<SparkrunRegistry[]>("/api/registries");

export const createRegistry = (body: NewRegistry) =>
  apiFetch<SparkrunRegistry>("/api/registries", { method: "POST", body: JSON.stringify(body) });

export const updateRegistry = (id: string, body: Partial<NewRegistry>) =>
  apiFetch<SparkrunRegistry>(`/api/registries/${id}`, { method: "PATCH", body: JSON.stringify(body) });

export const deleteRegistry = (id: string) =>
  apiFetch<{ status: string }>(`/api/registries/${id}`, { method: "DELETE" });
```

- [ ] **Step 2: Create `components/registries-section.tsx`**

Mirror the styling of the Join Tokens section in `app/settings/page.tsx`:

```tsx
"use client";

import { useEffect, useState, useCallback } from "react";
import { toast } from "sonner";
import { listRegistries, createRegistry, deleteRegistry, updateRegistry, type SparkrunRegistry } from "@/lib/api";

const EMPTY = { name: "", url: "", subpath: "recipes" };

export function RegistriesSection() {
  const [rows, setRows] = useState<SparkrunRegistry[]>([]);
  const [form, setForm] = useState(EMPTY);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try { setRows(await listRegistries()); } catch (e) { toast.error((e as Error).message); }
  }, []);
  useEffect(() => { load(); }, [load]);

  async function add() {
    setBusy(true);
    try {
      await createRegistry(form);
      toast.success(`Added registry '${form.name}' — pushed to online nodes`);
      setForm(EMPTY);
      load();
    } catch (e) { toast.error((e as Error).message); } finally { setBusy(false); }
  }

  async function remove(r: SparkrunRegistry) {
    if (!confirm(`Delete registry '${r.name}'? It will be removed from all nodes.`)) return;
    try { await deleteRegistry(r.id); toast.success(`Removed '${r.name}'`); load(); }
    catch (e) { toast.error((e as Error).message); }
  }

  async function toggleVisible(r: SparkrunRegistry) {
    try { await updateRegistry(r.id, { visible: !r.visible }); load(); }
    catch (e) { toast.error((e as Error).message); }
  }

  return (
    <section className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <h2 className="text-lg font-semibold mb-4">Sparkrun Registries</h2>
      <p className="text-sm text-gray-400 mb-4">
        Recipe registries cloned by sparkrun on every node. Changes are pushed to all online nodes immediately.
      </p>

      <div className="flex flex-wrap gap-2 mb-6">
        <input className="bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm" placeholder="name (a-z0-9-)"
          value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
        <input className="bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm flex-1 min-w-[16rem]" placeholder="git url"
          value={form.url} onChange={(e) => setForm({ ...form, url: e.target.value })} />
        <input className="bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm" placeholder="subpath"
          value={form.subpath} onChange={(e) => setForm({ ...form, subpath: e.target.value })} />
        <button disabled={busy || !form.name || !form.url} onClick={add}
          className="bg-green-600 hover:bg-green-500 disabled:opacity-40 rounded px-3 py-1 text-sm transition-colors">
          Add
        </button>
      </div>

      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-500 text-left border-b border-gray-800">
            <th className="pb-2 font-medium">Name</th>
            <th className="pb-2 font-medium">URL</th>
            <th className="pb-2 font-medium">Subpath</th>
            <th className="pb-2 font-medium">Visible</th>
            <th className="pb-2 font-medium"></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-b border-gray-800/50">
              <td className="py-2 font-mono">{r.name}</td>
              <td className="py-2 text-gray-400 truncate max-w-[20rem]">{r.url}</td>
              <td className="py-2 text-gray-400 font-mono">{r.subpath}</td>
              <td className="py-2">
                <button onClick={() => toggleVisible(r)} className={r.visible ? "text-green-400" : "text-gray-600"}>
                  {r.visible ? "visible" : "hidden"}
                </button>
              </td>
              <td className="py-2">
                <button onClick={() => remove(r)} className="text-red-500 hover:text-red-400 text-xs transition-colors">Delete</button>
              </td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr><td colSpan={5} className="py-4 text-center text-gray-600">No registries</td></tr>
          )}
        </tbody>
      </table>
    </section>
  );
}
```

- [ ] **Step 3: Render the section in the settings page**

In `packages/dashboard/app/settings/page.tsx`, add the import near the top:

```tsx
import { RegistriesSection } from "@/components/registries-section";
```

Add `<RegistriesSection />` just above `<OllamaModelsSection />` (near line 131):

```tsx
      <RegistriesSection />
      <OllamaModelsSection />
```

- [ ] **Step 4: Verify the dashboard builds**

Run: `npm run build -w @dgx-manager/dashboard` (or `npm run build`)
Expected: Next.js build succeeds with no type errors.

- [ ] **Step 5: Manual smoke test (documented, not automated)**

Bring the stack up per CLAUDE.md (`./scripts/build-agent-bundles.sh && MANAGER_ADVERTISE_HOST=… SSH_USER=… docker compose up -d --build`), open the Settings page, add the `rtx` registry (`https://github.com/kreuzhofer/rtx-recipe-registry.git`, subpath `recipes`), and confirm: the row appears, online nodes re-report recipes, and `@rtx/*` amd64 recipes surface on the RTX node's deploy form. Note this in the PR (UI-only behavior not covered by automated tests).

- [ ] **Step 6: Commit**

```bash
git add packages/dashboard/lib/api.ts packages/dashboard/components/registries-section.tsx packages/dashboard/app/settings/page.tsx
git commit -m "feat(dashboard): manage sparkrun registries from Settings"
```

---

### Task 11: Full suite green + final commit

- [ ] **Step 1: Run the entire test suite**

Run: `npm test`
Expected: all suites PASS.

- [ ] **Step 2: If anything is red, fix it before proceeding.** Do not claim done with a red suite (CLAUDE.md).

- [ ] **Step 3: Confirm the agent version was bumped exactly once** (Task 9 Step 6). Run `git log --oneline -1 -- packages/agent/package.json` to verify.

---

## Self-Review

**Spec coverage:**
- §A data model → Task 1. §B seed → Task 4 + boot call in Task 6. §C server API + validation + push → Tasks 2/3/5/6. §D agent handler + renderer → Tasks 8/9. §E dashboard → Task 10. §F data flow (mutate→push; connect→reconcile) → Tasks 6 + 7. Risk mitigations (hard validation, atomic write, report-back, seed-if-empty) → Tasks 3, 9, 6/9, 4. Testing matrix → property (Task 8), unit (Tasks 2/3), integration happy+error (Task 6), seed (Task 4). All spec sections map to a task.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; the one manual step (Task 10 Step 5) is explicitly labeled non-automated per the spec's "environmental behavior" carve-out.

**Type consistency:** `RegistryWire` snake_case fields are identical in server `wire.ts` and agent `registries.ts`. The `cmd:set-registries` payload shape (`{ registries: RegistryWire[] }`) is identical in `push.ts` (producer) and the agent handler (consumer). `seedDefaultRegistries`/`registryRowsToWire`/`validateRegistry`/`pushRegistriesToAgent`/`pushRegistriesToConnectedAgents`/`renderRegistriesYaml`/`writeRegistriesFile`/`registriesConfigPath` names are used consistently across tasks. DB field names (`tuningSubpath` etc.) match between schema (Task 1), seed (Task 4), wire mapper (Task 2), and dashboard type (Task 10).
