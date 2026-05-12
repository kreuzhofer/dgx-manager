# Ollama Catalog Selection & Auto-Propagating Deployments — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the agent-shipped hardcoded `ollama-models.json` with a Settings-page UI that lets the cluster admin (a) refresh the Ollama model catalog on demand, (b) check which models should be available for deployment, and (c) get live download progress in the deployments page when an unpulled model is launched.

**Architecture:**
- **Catalog** (the universe of pickable models) is fetched on demand from `https://ollama.com/library`, parsed server-side with `cheerio`, and cached in the `Setting` table under two keys: `ollama.catalog.json` (the model array) and `ollama.catalog.fetchedAt` (ISO timestamp). The parser filters out cloud-only models (cards with the cloud marker and zero local parameter sizes) so the cluster only ever sees locally-pullable entries. Each entry exposes `name`, `description`, `type` (chat/embedding), `sizes[]` (parameter sizes like `"8b"`, `"70b"`), and `capabilities[]` (`tools`, `thinking`, `vision`, `embedding`, `audio`).
- **Selection** (the user-curated subset enabled for the cluster) lives in `Setting` under `ollama.enabled.json` as a JSON string-array of **tag identifiers** — `"model:size"` (e.g. `"llama3.1:8b"`, `"qwen3:32b"`) for sized models, or the bare model name for sizeless ones (`"nomic-embed-text"`). Auto-persistence: checkbox toggle on the Settings page debounces 400ms then PUTs to `/api/ollama-catalog/enabled`.
- **Propagation**: the deployments page calls a new `GET /api/ollama-catalog/available` that returns a **flat list of deployable tag rows** (one row per enabled tag, carrying its parent model's description, type, capabilities). Any PUT to the enabled list emits SSE `ollama-catalog:updated`, and the deployments page reloads its dropdown without a page refresh.
- **Download progress**: the agent's existing `pullModel()` already streams Ollama's pull progress through an `onLog` callback. We add a structured `agent:ollama:pull-progress` WS message alongside the existing log lines; the server forwards it as a `deployment:progress` SSE event (phase `"downloading"`), which the deployments page already renders. We additionally surface a `sonner` toast that lives across pages while a pull is in flight.

**Tech Stack:** TypeScript strict + ESM, Express 5, Prisma 7 (SQLite), Next.js 15 + React 19 + Tailwind 4, `cheerio` (new server dep), `sonner` (new dashboard dep), Vitest + supertest.

**Out of scope:** Per-user preferences (system is single-user / no auth); Ollama models that aren't in the upstream catalog (the agent's `ollama-models.json` is removed once the catalog flow lands); embedding-vs-chat editing in the UI (we display the `type` shown by Ollama's library but don't expose an override); cloud-only models (filtered out at the parser — we deploy locally).

---

## File Structure

**New files (server):**
- `packages/server/src/ollama/catalog-parser.ts` — pure HTML → `OllamaCatalogEntry[]` parser. No IO; fully unit/property-testable.
- `packages/server/src/ollama/catalog-store.ts` — read/write the cached catalog and enabled list against the `Setting` table. Pure DB IO.
- `packages/server/src/ollama/catalog-fetcher.ts` — HTTP fetch + parse + store glue. The one place that touches `fetch()`.
- `packages/server/src/routes/ollama-catalog.ts` — `GET /catalog`, `POST /catalog/refresh`, `GET /available`, `GET /enabled`, `PUT /enabled`.
- `packages/server/src/ollama/catalog-parser.test.ts` — unit + property tests.
- `packages/server/src/__tests__/integration/ollama-catalog.routes.test.ts` — supertest integration tests.

**Modified files (server):**
- `packages/server/src/index.ts` — mount the new router.
- `packages/server/src/routes/deployments.ts` — VRAM estimation now reads catalog instead of agent's `getOllamaModels()`.
- `packages/server/src/ws/agent-hub.ts` — handle new `agent:ollama:pull-progress` message, broadcast as `deployment:progress`.

**Modified files (agent):**
- `packages/agent/src/runtime/ollama.ts` — `pullModel()` accepts `onProgress({ phase, percent, current, total })` alongside `onLog`; emits structured progress.
- `packages/agent/src/index.ts` — wire `onProgress` to a new `sendMsg("agent:ollama:pull-progress", …)` call inside the Ollama deploy path. Bump agent version via the existing script.
- **Deleted:** `packages/agent/src/ollama-models.json` (no longer the source of truth).

**New files (dashboard):**
- `packages/dashboard/components/ollama-models-section.tsx` — Settings-page section: refresh button, last-refreshed timestamp, scrollable checkbox list, debounced auto-persist.
- `packages/dashboard/components/deployment-pull-toast.tsx` — small helper that hooks into SSE and renders a sticky toast per in-flight Ollama deploy.
- `packages/dashboard/lib/use-debounced-callback.ts` — 8-line `useDebouncedCallback` hook.

**Modified files (dashboard):**
- `packages/dashboard/package.json` — add `sonner`.
- `packages/dashboard/app/layout.tsx` — mount `<Toaster />` and `<DeploymentPullToast />` (the latter needs to live inside the SSE-listening tree, so it goes into a small client wrapper).
- `packages/dashboard/app/settings/page.tsx` — add `<OllamaModelsSection />` after the existing sections.
- `packages/dashboard/app/deployments/page.tsx` — change the `/api/recipes/ollama-models` fetch to `/api/ollama-catalog/available`, react to `ollama-catalog:updated` SSE.

**Settings keys used:**
- `ollama.catalog.json` — JSON-stringified `OllamaCatalogEntry[]`.
- `ollama.catalog.fetchedAt` — ISO timestamp.
- `ollama.enabled.json` — JSON-stringified `string[]` of tag identifiers (e.g. `["llama3.1:8b","qwen3:8b","nomic-embed-text"]`). A sized model contributes one key per size the user enabled; a sizeless model contributes just its bare name.

---

## Task 1: Server — Ollama catalog HTML parser

**Files:**
- Create: `packages/server/src/ollama/catalog-parser.ts`
- Create: `packages/server/src/ollama/catalog-parser.test.ts`
- Modify: `packages/server/package.json` — add `cheerio` to `dependencies`.

**Why pure-function-first:** the parser is the one piece of this feature whose correctness is fully decidable from text input. It belongs in its own module so it can be unit-tested without HTTP, without Prisma, and without the agent. Pattern mirrors `packages/server/src/admission/vram.ts` per CLAUDE.md.

- [ ] **Step 1: Install cheerio**

Run from repo root:
```bash
npm install --workspace=packages/server cheerio@^1.0.0
```

Verify it lands in `packages/server/package.json` under `dependencies`.

- [ ] **Step 2: Write the failing parser test**

Create `packages/server/src/ollama/catalog-parser.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { it as fcIt } from "@fast-check/vitest";
import fc from "fast-check";
import { parseCatalogHtml } from "./catalog-parser.js";

const SAMPLE_LIBRARY_HTML = `
<html><body>
  <ul role="list">
    <li>
      <a href="/library/llama3.1">
        <h2>llama3.1</h2>
        <p>Meta's Llama 3.1 family.</p>
        <span x-test="size">4.7GB</span>
        <span x-test="size">40GB</span>
        <span x-test="pulls">1.2M</span>
      </a>
    </li>
    <li>
      <a href="/library/nomic-embed-text">
        <h2>nomic-embed-text</h2>
        <p>A high-performing open embedding model.</p>
        <span x-test="size">274MB</span>
        <span x-test="capability">embedding</span>
      </a>
    </li>
  </ul>
</body></html>`;

describe("parseCatalogHtml", () => {
  it("extracts entries from a library page", () => {
    const entries = parseCatalogHtml(SAMPLE_LIBRARY_HTML);
    expect(entries).toHaveLength(2);
    expect(entries[0]).toMatchObject({
      name: "llama3.1",
      description: "Meta's Llama 3.1 family.",
      type: "chat",
    });
    expect(entries[0].sizes).toEqual(["4.7GB", "40GB"]);
    expect(entries[1]).toMatchObject({ name: "nomic-embed-text", type: "embedding" });
  });

  it("returns [] on malformed HTML", () => {
    expect(parseCatalogHtml("")).toEqual([]);
    expect(parseCatalogHtml("<p>no models here</p>")).toEqual([]);
  });

  it("skips entries with no name", () => {
    expect(parseCatalogHtml(`<a href="/library/"><h2></h2></a>`)).toEqual([]);
  });

  /**
   * Property: feeding the parser any string never throws and always returns
   * an array — the catalog refresh path treats parse failure as "no models"
   * not as an error, so the parser must not surface exceptions to callers.
   */
  fcIt.prop([fc.string()])("never throws on arbitrary input", (s) => {
    expect(() => parseCatalogHtml(s)).not.toThrow();
    expect(Array.isArray(parseCatalogHtml(s))).toBe(true);
  });
});
```

- [ ] **Step 3: Run the test and watch it fail**

```bash
npx vitest run packages/server/src/ollama/catalog-parser.test.ts
```
Expected: FAIL — `Cannot find module './catalog-parser.js'`.

- [ ] **Step 4: Implement the parser**

Create `packages/server/src/ollama/catalog-parser.ts`:
```ts
import * as cheerio from "cheerio";

export interface OllamaCatalogEntry {
  /** Bare model name as shown by Ollama, e.g. "llama3.1" or "qwen3-vl". */
  name: string;
  /** One-line description from the library page; "" if missing. */
  description: string;
  /** "embedding" if Ollama tags it as such, otherwise "chat". */
  type: "chat" | "embedding";
  /** Tag-level size strings as displayed ("4.7GB", "274MB"). May be empty. */
  sizes: string[];
}

/**
 * Parse Ollama's https://ollama.com/library page into a flat list.
 *
 * Layout assumption: each model is rendered as an <a href="/library/{name}">
 * inside a list, containing an <h2> with the name, a description <p>, and
 * size/capability spans. We intentionally stay loose — Ollama tweaks markup
 * occasionally, and any heuristic that finds an /library/{slug} anchor with
 * an h2 child is good enough.
 *
 * Never throws: malformed input yields [].
 */
export function parseCatalogHtml(html: string): OllamaCatalogEntry[] {
  try {
    const $ = cheerio.load(html);
    const entries: OllamaCatalogEntry[] = [];
    $('a[href^="/library/"]').each((_, el) => {
      const href = $(el).attr("href") || "";
      const name = href.replace(/^\/library\//, "").split(/[/?#]/)[0].trim();
      if (!name) return;
      const heading = $(el).find("h2").first().text().trim();
      if (!heading) return;
      const description = $(el).find("p").first().text().trim();
      const sizes: string[] = [];
      $(el).find("*").each((_i, child) => {
        const t = $(child).text().trim();
        if (/^\d+(\.\d+)?\s*(GB|MB|KB|TB)$/i.test(t)) sizes.push(t);
      });
      const text = $(el).text().toLowerCase();
      const type: "chat" | "embedding" = /\bembedding\b/.test(text) ? "embedding" : "chat";
      entries.push({ name: heading || name, description, type, sizes });
    });
    // Dedupe by name (the library page sometimes lists the same slug twice).
    const seen = new Set<string>();
    return entries.filter((e) => (seen.has(e.name) ? false : seen.add(e.name)));
  } catch {
    return [];
  }
}
```

- [ ] **Step 5: Run the tests and verify they pass**

```bash
npx vitest run packages/server/src/ollama/catalog-parser.test.ts
```
Expected: PASS for all 4 test cases (3 unit + 1 property).

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/ollama/catalog-parser.ts \
        packages/server/src/ollama/catalog-parser.test.ts \
        packages/server/package.json packages/server/package-lock.json package-lock.json
git commit -m "server: add ollama catalog HTML parser"
```

---

## Task 2: Server — catalog store (Setting-backed persistence)

**Files:**
- Create: `packages/server/src/ollama/catalog-store.ts`
- Modify: integration test fixtures used in Task 4 will exercise this.

The store wraps the three Setting keys behind typed accessors so route handlers and the deployments.ts VRAM estimation path don't duplicate JSON-parse logic.

- [ ] **Step 1: Implement the store**

Create `packages/server/src/ollama/catalog-store.ts`:
```ts
import { prisma } from "../prisma.js";
import type { OllamaCatalogEntry } from "./catalog-parser.js";

const CATALOG_KEY = "ollama.catalog.json";
const FETCHED_AT_KEY = "ollama.catalog.fetchedAt";
const ENABLED_KEY = "ollama.enabled.json";

export interface CatalogSnapshot {
  entries: OllamaCatalogEntry[];
  fetchedAt: string | null;
}

/**
 * A flat, deployable tag — what the deployments page renders in its dropdown
 * and what the user actually pulls with `ollama pull`. Sized models flatten
 * into one row per size ("llama3.1:8b", "llama3.1:70b"); sizeless models
 * (nomic-embed-text, wizardlm) flatten into a single row with `size === null`
 * and `tag === modelName`.
 */
export interface AvailableTag {
  tag: string;            // "llama3.1:8b" or "nomic-embed-text"
  modelName: string;      // "llama3.1"
  size: string | null;    // "8b" or null for sizeless models
  type: "chat" | "embedding";
  description: string;
  capabilities: string[];
}

async function readSetting(key: string): Promise<string | null> {
  const row = await prisma.setting.findUnique({ where: { key } });
  return row?.value ?? null;
}

async function writeSetting(key: string, value: string): Promise<void> {
  await prisma.setting.upsert({
    where: { key },
    create: { key, value },
    update: { value },
  });
}

export async function readCatalog(): Promise<CatalogSnapshot> {
  const [raw, fetchedAt] = await Promise.all([
    readSetting(CATALOG_KEY),
    readSetting(FETCHED_AT_KEY),
  ]);
  if (!raw) return { entries: [], fetchedAt };
  try {
    const parsed = JSON.parse(raw) as OllamaCatalogEntry[];
    return { entries: Array.isArray(parsed) ? parsed : [], fetchedAt };
  } catch {
    return { entries: [], fetchedAt };
  }
}

export async function writeCatalog(entries: OllamaCatalogEntry[]): Promise<string> {
  const fetchedAt = new Date().toISOString();
  await writeSetting(CATALOG_KEY, JSON.stringify(entries));
  await writeSetting(FETCHED_AT_KEY, fetchedAt);
  return fetchedAt;
}

export async function readEnabled(): Promise<string[]> {
  const raw = await readSetting(ENABLED_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

export async function writeEnabled(tags: string[]): Promise<void> {
  // Dedupe + drop non-strings defensively — the PUT route validates, but the
  // DB shouldn't trust callers.
  const cleaned = Array.from(new Set(tags.filter((t) => typeof t === "string" && t.length > 0)));
  await writeSetting(ENABLED_KEY, JSON.stringify(cleaned));
}

/**
 * Flatten the catalog × enabled-set into the deployable rows the deployments
 * page actually wants. Enabled tags reference either:
 *   - "model:size" — must match a `OllamaCatalogEntry` whose `sizes` list
 *     contains `size`, OR
 *   - "model" (no colon) — must match an `OllamaCatalogEntry` with the same
 *     bare name and an empty `sizes` list (sizeless / embedding models).
 *
 * Entries that no longer exist in the catalog (model removed upstream) are
 * silently dropped — the UI surfaces this as the model disappearing from
 * the dropdown, which is the right outcome.
 */
export async function readAvailable(): Promise<AvailableTag[]> {
  const [{ entries }, enabled] = await Promise.all([readCatalog(), readEnabled()]);
  const byName = new Map(entries.map((e) => [e.name, e]));
  const rows: AvailableTag[] = [];
  for (const tag of enabled) {
    const [modelName, size] = tag.includes(":") ? tag.split(":", 2) : [tag, null];
    const entry = byName.get(modelName);
    if (!entry) continue;
    if (size === null) {
      // Sizeless: only valid if the catalog entry has no sizes either.
      if (entry.sizes.length > 0) continue;
    } else {
      if (!entry.sizes.includes(size)) continue;
    }
    rows.push({
      tag,
      modelName,
      size,
      type: entry.type,
      description: entry.description,
      capabilities: entry.capabilities,
    });
  }
  return rows;
}
```

- [ ] **Step 2: Commit (no test yet — the store is exercised through Task 4's route tests)**

```bash
git add packages/server/src/ollama/catalog-store.ts
git commit -m "server: add ollama catalog store"
```

---

## Task 3: Server — catalog fetcher (HTTP + parse + store)

**Files:**
- Create: `packages/server/src/ollama/catalog-fetcher.ts`

This is the one place that hits the network. Kept tiny so the route layer stays trivial.

- [ ] **Step 1: Implement the fetcher**

Create `packages/server/src/ollama/catalog-fetcher.ts`:
```ts
import { parseCatalogHtml, type OllamaCatalogEntry } from "./catalog-parser.js";
import { writeCatalog } from "./catalog-store.js";

const LIBRARY_URL = process.env.OLLAMA_LIBRARY_URL || "https://ollama.com/library";
const FETCH_TIMEOUT_MS = 15_000;

export interface RefreshResult {
  entries: OllamaCatalogEntry[];
  fetchedAt: string;
}

/**
 * Fetch the public Ollama library page, parse it, and write the result into
 * the Setting table. Throws on network failure or non-2xx — the route handler
 * surfaces that to the user (refresh is explicit, so failure should be loud).
 *
 * If parsing succeeds but yields zero entries, we still persist the empty
 * list and the timestamp — that's a real catalog state (Ollama outage,
 * markup overhaul) and we want the UI to reflect it, not show stale data.
 */
export async function refreshCatalog(): Promise<RefreshResult> {
  const res = await fetch(LIBRARY_URL, {
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    headers: { "User-Agent": "dgx-manager/ollama-catalog" },
  });
  if (!res.ok) {
    throw new Error(`Ollama library fetch failed: HTTP ${res.status}`);
  }
  const html = await res.text();
  const entries = parseCatalogHtml(html);
  const fetchedAt = await writeCatalog(entries);
  return { entries, fetchedAt };
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/server/src/ollama/catalog-fetcher.ts
git commit -m "server: add ollama catalog fetcher"
```

---

## Task 4: Server — catalog routes

**Files:**
- Create: `packages/server/src/routes/ollama-catalog.ts`
- Create: `packages/server/src/__tests__/integration/ollama-catalog.routes.test.ts`
- Modify: `packages/server/src/index.ts`

Five endpoints under `/api/ollama-catalog`:
- `GET /catalog` — cached catalog + fetched-at timestamp.
- `POST /catalog/refresh` — fetch from upstream, persist, return result.
- `GET /enabled` — the user-selected name list.
- `PUT /enabled` — replace the selected name list; broadcasts SSE.
- `GET /available` — intersection of catalog × enabled, used by the deployments page.

- [ ] **Step 1: Write the failing integration test**

Create `packages/server/src/__tests__/integration/ollama-catalog.routes.test.ts`:
```ts
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let ollamaCatalogRouter: typeof import("../../routes/ollama-catalog.js").ollamaCatalogRouter;
let writeCatalog: typeof import("../../ollama/catalog-store.js").writeCatalog;
let writeEnabled: typeof import("../../ollama/catalog-store.js").writeEnabled;

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
  ({ ollamaCatalogRouter } = await import("../../routes/ollama-catalog.js"));
  ({ writeCatalog, writeEnabled } = await import("../../ollama/catalog-store.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  await prisma.setting.deleteMany({});
});

function makeApp() {
  const app = express();
  app.use(express.json());
  app.use("/api/ollama-catalog", ollamaCatalogRouter);
  return app;
}

describe("GET /api/ollama-catalog/catalog", () => {
  it("returns empty + null timestamp when nothing cached", async () => {
    const res = await request(makeApp()).get("/api/ollama-catalog/catalog");
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ entries: [], fetchedAt: null });
  });

  it("returns persisted entries and timestamp", async () => {
    const ts = await writeCatalog([
      { name: "llama3.1", description: "Meta Llama", type: "chat", sizes: ["8b", "70b"], capabilities: ["tools"] },
    ]);
    const res = await request(makeApp()).get("/api/ollama-catalog/catalog");
    expect(res.body.fetchedAt).toBe(ts);
    expect(res.body.entries).toHaveLength(1);
    expect(res.body.entries[0].name).toBe("llama3.1");
    expect(res.body.entries[0].sizes).toEqual(["8b", "70b"]);
  });
});

describe("GET /api/ollama-catalog/enabled and PUT /api/ollama-catalog/enabled", () => {
  it("returns empty list when nothing enabled", async () => {
    const res = await request(makeApp()).get("/api/ollama-catalog/enabled");
    expect(res.body).toEqual({ enabled: [] });
  });

  it("PUT replaces the enabled list and GET reflects it", async () => {
    const app = makeApp();
    const put = await request(app)
      .put("/api/ollama-catalog/enabled")
      .send({ enabled: ["llama3.1:8b", "qwen3:8b"] });
    expect(put.status).toBe(200);
    const get = await request(app).get("/api/ollama-catalog/enabled");
    expect(get.body.enabled).toEqual(["llama3.1:8b", "qwen3:8b"]);
  });

  it("PUT rejects non-array body with 400", async () => {
    const res = await request(makeApp())
      .put("/api/ollama-catalog/enabled")
      .send({ enabled: "not-an-array" });
    expect(res.status).toBe(400);
  });
});

describe("GET /api/ollama-catalog/available", () => {
  it("flattens catalog × enabled into deployable tag rows", async () => {
    await writeCatalog([
      { name: "llama3.1", description: "A", type: "chat", sizes: ["8b", "70b"], capabilities: ["tools"] },
      { name: "qwen3",   description: "B", type: "chat", sizes: ["8b", "32b"], capabilities: ["tools"] },
      { name: "phi4",    description: "C", type: "chat", sizes: ["14b"], capabilities: [] },
      { name: "nomic-embed-text", description: "Emb", type: "embedding", sizes: [], capabilities: ["embedding"] },
    ]);
    await writeEnabled(["llama3.1:8b", "qwen3:8b", "qwen3:32b", "nomic-embed-text"]);
    const res = await request(makeApp()).get("/api/ollama-catalog/available");
    expect(res.status).toBe(200);
    const tags = (res.body as { tag: string }[]).map((r) => r.tag);
    expect(tags).toEqual(["llama3.1:8b", "qwen3:8b", "qwen3:32b", "nomic-embed-text"]);
    const llama = (res.body as { tag: string; modelName: string; size: string | null }[])
      .find((r) => r.tag === "llama3.1:8b")!;
    expect(llama.modelName).toBe("llama3.1");
    expect(llama.size).toBe("8b");
    const emb = (res.body as { tag: string; size: string | null }[])
      .find((r) => r.tag === "nomic-embed-text")!;
    expect(emb.size).toBeNull();
  });

  it("drops enabled tags that no longer exist in the catalog", async () => {
    await writeCatalog([
      { name: "llama3.1", description: "A", type: "chat", sizes: ["8b"], capabilities: [] },
    ]);
    await writeEnabled(["llama3.1:8b", "llama3.1:405b", "ghost-model:1b"]);
    const res = await request(makeApp()).get("/api/ollama-catalog/available");
    expect((res.body as { tag: string }[]).map((r) => r.tag)).toEqual(["llama3.1:8b"]);
  });

  it("returns [] when nothing is enabled even if catalog is non-empty", async () => {
    await writeCatalog([
      { name: "llama3.1", description: "A", type: "chat", sizes: ["8b"], capabilities: [] },
    ]);
    const res = await request(makeApp()).get("/api/ollama-catalog/available");
    expect(res.body).toEqual([]);
  });
});

describe("POST /api/ollama-catalog/catalog/refresh", () => {
  it("calls the fetcher and persists the result", async () => {
    vi.resetModules();
    vi.doMock("../../ollama/catalog-fetcher.js", () => ({
      refreshCatalog: async () => ({
        entries: [{ name: "fake", description: "stub", type: "chat", sizes: [], capabilities: [] }],
        fetchedAt: "2026-05-12T00:00:00.000Z",
      }),
    }));
    const { ollamaCatalogRouter: r } = await import("../../routes/ollama-catalog.js");
    const app = express();
    app.use(express.json());
    app.use("/api/ollama-catalog", r);
    const res = await request(app).post("/api/ollama-catalog/catalog/refresh");
    expect(res.status).toBe(200);
    expect(res.body.entries).toHaveLength(1);
    expect(res.body.entries[0].name).toBe("fake");
    vi.doUnmock("../../ollama/catalog-fetcher.js");
  });
});
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
npx vitest run packages/server/src/__tests__/integration/ollama-catalog.routes.test.ts
```
Expected: FAIL — `Cannot find module '../../routes/ollama-catalog.js'`.

- [ ] **Step 3: Implement the router**

Create `packages/server/src/routes/ollama-catalog.ts`:
```ts
import { Router } from "express";
import {
  readCatalog,
  readEnabled,
  writeEnabled,
  readAvailable,
} from "../ollama/catalog-store.js";
import { refreshCatalog } from "../ollama/catalog-fetcher.js";
import { broadcast as sseBroadcast } from "../sse.js";

export const ollamaCatalogRouter = Router();

ollamaCatalogRouter.get("/catalog", async (_req, res) => {
  res.json(await readCatalog());
});

ollamaCatalogRouter.post("/catalog/refresh", async (_req, res) => {
  try {
    const result = await refreshCatalog();
    sseBroadcast({ type: "ollama-catalog:updated", payload: { reason: "refresh" } });
    res.json(result);
  } catch (err) {
    res.status(502).json({ error: String((err as Error).message || err) });
  }
});

ollamaCatalogRouter.get("/enabled", async (_req, res) => {
  res.json({ enabled: await readEnabled() });
});

ollamaCatalogRouter.put("/enabled", async (req, res) => {
  const body = req.body as { enabled?: unknown };
  if (!Array.isArray(body.enabled)) {
    return res.status(400).json({ error: "enabled must be an array of strings" });
  }
  const names = body.enabled.filter((x): x is string => typeof x === "string");
  await writeEnabled(names);
  sseBroadcast({ type: "ollama-catalog:updated", payload: { reason: "enabled-changed" } });
  res.json({ enabled: names });
});

ollamaCatalogRouter.get("/available", async (_req, res) => {
  res.json(await readAvailable());
});
```

- [ ] **Step 4: Mount the router**

Edit `packages/server/src/index.ts`. After the existing `import { settingsRouter } from "./routes/settings.js";` line, add:
```ts
import { ollamaCatalogRouter } from "./routes/ollama-catalog.js";
```
After the `app.use("/api/settings", settingsRouter);` line, add:
```ts
app.use("/api/ollama-catalog", ollamaCatalogRouter);
```

- [ ] **Step 5: Run the tests and verify they pass**

```bash
npx vitest run packages/server/src/__tests__/integration/ollama-catalog.routes.test.ts
```
Expected: PASS for all 7 cases.

- [ ] **Step 6: Run the full suite to make sure nothing else broke**

```bash
npm test
```
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add packages/server/src/routes/ollama-catalog.ts \
        packages/server/src/__tests__/integration/ollama-catalog.routes.test.ts \
        packages/server/src/index.ts
git commit -m "server: ollama-catalog routes (catalog, enabled, available, refresh)"
```

---

## Task 5: Server — switch deployments VRAM estimation to use the catalog

**Files:**
- Create: `packages/server/src/ollama/vram-estimate.ts` — pure parameter-size → MB helper, unit-testable.
- Create: `packages/server/src/ollama/vram-estimate.test.ts`
- Modify: `packages/server/src/routes/deployments.ts` — use the new helper, fall back to the agent's curated list (legacy byte sizes) when neither catalog nor parameter size is available.

Right now `deployments.ts:107-113` derives Ollama VRAM from `agentHub.getOllamaModels()` by parsing `"4.7GB"`-style strings. The catalog gives us *parameter* sizes (`"8b"`, `"70b"`), not byte sizes, so we need a small conversion: Ollama defaults to Q4_K_M quantization which lands around `~0.55 GB per billion params` plus modest KV/cache overhead. We use that as the upper-bound VRAM estimate for admission; `vramActual` corrects it after `nvidia-smi` reports post-load.

- [ ] **Step 1: Write the unit test for the parameter-size → MB helper**

Create `packages/server/src/ollama/vram-estimate.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { ollamaVramEstimateMB } from "./vram-estimate.js";

describe("ollamaVramEstimateMB", () => {
  it.each([
    ["8b", 4506],     // 8 × 0.55 GB × 1024 ≈ 4506 MB
    ["70b", 39424],   // 70 × 0.55 GB × 1024
    ["405b", 228096], // 405 × 0.55 GB × 1024
    ["1.5b", 845],    // 1.5 × 0.55 GB × 1024
    ["270m", 149],    // 0.27 × 0.55 GB × 1024
    ["e2b", 1126],    // Gemma "effective 2b" — parsed as 2b
    ["e4b", 2253],
  ])("parses %s as ~%d MB", (size, expected) => {
    const got = ollamaVramEstimateMB(size);
    // Allow ±2 MB for rounding drift.
    expect(got).toBeGreaterThanOrEqual(expected - 2);
    expect(got).toBeLessThanOrEqual(expected + 2);
  });

  it.each(["", "huge", "12.34xyz", "GB", null as unknown as string, undefined as unknown as string])(
    "returns null for unparseable input %p",
    (s) => {
      expect(ollamaVramEstimateMB(s)).toBeNull();
    },
  );
});
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
npx vitest run packages/server/src/ollama/vram-estimate.test.ts
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the helper**

Create `packages/server/src/ollama/vram-estimate.ts`:
```ts
/**
 * Parse an Ollama parameter-size tag ("8b", "70b", "e2b", "270m") into a VRAM
 * estimate in MB. Ollama defaults to Q4_K_M quantization which lands around
 * 0.55 GB per billion params; we use that as the upper-bound estimate for
 * admission. The agent reports `vramActual` once nvidia-smi has post-load
 * truth, so the estimate is only load-bearing for the refuse-before-launch
 * check.
 *
 * Recognized suffixes:
 *   - "b"  — billions of params
 *   - "m"  — millions of params (e.g. "270m")
 *   - "e2b"/"e4b" — Gemma "effective N billion" notation; treated as Nb.
 *
 * Returns null if the input doesn't match the expected shape.
 */
const Q4_GB_PER_BILLION = 0.55;
const MB_PER_GB = 1024;

export function ollamaVramEstimateMB(rawSize: string | null | undefined): number | null {
  if (!rawSize || typeof rawSize !== "string") return null;
  const m = rawSize.trim().toLowerCase().match(/^e?(\d+(?:\.\d+)?)\s*([bm])$/);
  if (!m) return null;
  const value = parseFloat(m[1]);
  if (!Number.isFinite(value)) return null;
  const billionsParams = m[2] === "b" ? value : value / 1000;
  return Math.round(billionsParams * Q4_GB_PER_BILLION * MB_PER_GB);
}
```

- [ ] **Step 4: Run the unit test and verify it passes**

```bash
npx vitest run packages/server/src/ollama/vram-estimate.test.ts
```
Expected: PASS.

- [ ] **Step 5: Write the failing integration test**

Add a new test file `packages/server/src/__tests__/integration/deployments.ollama-vram.test.ts`:
```ts
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;
let writeCatalog: typeof import("../../ollama/catalog-store.js").writeCatalog;

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
  ({ writeCatalog } = await import("../../ollama/catalog-store.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
  await prisma.setting.deleteMany({});
});

function makeApp() {
  const sentMessages: { nodeId: string; message: unknown }[] = [];
  const hub = {
    getRecipes: () => [],
    getOllamaModels: () => [],   // empty: forces use of catalog
    sendToAgent: (nodeId: string, message: unknown) => {
      sentMessages.push({ nodeId, message });
    },
  };
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.use("/api/deployments", deploymentsRouter);
  return { app, sentMessages };
}

describe("POST /api/deployments with runtime=ollama uses the catalog for VRAM estimation", () => {
  it("derives vramEstimate from a catalog parameter size", async () => {
    await prisma.node.create({
      data: { id: "n1", name: "node1", status: "online", vramTotal: 128_000 },
    });
    await writeCatalog([
      { name: "llama3.1", description: "Meta", type: "chat", sizes: ["8b", "70b"], capabilities: ["tools"] },
    ]);
    const { app } = makeApp();
    const res = await request(app)
      .post("/api/deployments")
      .send({ nodeId: "n1", runtime: "ollama", modelName: "llama3.1:8b" });
    // Adapt status check to what the route actually returns (read the existing
    // vram-admission test for the response shape — likely 200 with deployment row).
    expect([200, 201]).toContain(res.status);
    // 8b @ Q4 ≈ 8 × 0.55 × 1024 ≈ 4506 MB. Allow generous bounds.
    const persisted = await prisma.deployment.findFirst();
    expect(persisted?.vramEstimate ?? 0).toBeGreaterThan(4000);
    expect(persisted?.vramEstimate ?? 0).toBeLessThan(5000);
  });
});
```

Inspect the actual POST response shape first (look at the existing test or the route — adapt `expect(res.status).toBe(...)` to what the route returns). Reading the persisted row via prisma is the safest invariant regardless of response shape.

- [ ] **Step 6: Run the integration test and verify it fails**

```bash
npx vitest run packages/server/src/__tests__/integration/deployments.ollama-vram.test.ts
```
Expected: FAIL — `vramEstimate` will be null/0 because the route still parses GB strings.

- [ ] **Step 7: Update deployments.ts to consult the catalog first**

In `packages/server/src/routes/deployments.ts`, add these imports near the top with the other imports:
```ts
import { readCatalog as readOllamaCatalog } from "../ollama/catalog-store.js";
import { ollamaVramEstimateMB } from "../ollama/vram-estimate.js";
```
Then replace the Ollama VRAM-estimation block (currently around lines 107-113):
```ts
    if (isOllama) {
      const agentHub: AgentHub = req.app.get("agentHub");
      const ollamaModel = agentHub.getOllamaModels().find((m) => m.name === modelName);
      if (ollamaModel?.size) {
        const sizeMatch = ollamaModel.size.match(/([\d.]+)\s*GB/i);
        vramEstimate = sizeMatch ? Math.round(parseFloat(sizeMatch[1]) * 1024 * 1.1) : 0; // +10% overhead
      }
    } else {
```
with:
```ts
    if (isOllama) {
      const agentHub: AgentHub = req.app.get("agentHub");
      // Resolve the parameter size from the catalog: modelName is the pull
      // tag like "llama3.1:8b". Look up the bare model name, then pick the
      // requested size out of its `sizes` list (or the bare name for sizeless
      // entries like nomic-embed-text).
      const catalog = await readOllamaCatalog();
      const [bareName, requestedSize] = modelName.includes(":")
        ? modelName.split(":", 2)
        : [modelName, null];
      const catalogEntry = catalog.entries.find((m) => m.name === bareName);
      const catalogSize =
        catalogEntry && requestedSize && catalogEntry.sizes.includes(requestedSize)
          ? requestedSize
          : catalogEntry?.sizes[0] ?? null;
      const estimate = ollamaVramEstimateMB(catalogSize);
      if (estimate !== null) {
        vramEstimate = Math.round(estimate * 1.1); // +10% overhead (KV cache, runtime)
      } else {
        // Fallback: legacy agent-shipped byte size, parsed as before. Used
        // until the operator runs the first catalog refresh.
        const legacy = agentHub.getOllamaModels().find((m) => m.name === modelName);
        if (legacy?.size) {
          const sizeMatch = legacy.size.match(/([\d.]+)\s*GB/i);
          vramEstimate = sizeMatch ? Math.round(parseFloat(sizeMatch[1]) * 1024 * 1.1) : 0;
        }
      }
    } else {
```

- [ ] **Step 8: Run the test and verify it passes**

```bash
npx vitest run packages/server/src/__tests__/integration/deployments.ollama-vram.test.ts
```
Expected: PASS.

- [ ] **Step 9: Run the full suite**

```bash
npm test
```
Expected: green (especially the pre-existing `deployments.vram-admission.test.ts` — that suite seeds an empty catalog and stubs `getOllamaModels()`, so nothing changes).

- [ ] **Step 10: Commit**

```bash
git add packages/server/src/ollama/vram-estimate.ts \
        packages/server/src/ollama/vram-estimate.test.ts \
        packages/server/src/routes/deployments.ts \
        packages/server/src/__tests__/integration/deployments.ollama-vram.test.ts
git commit -m "server: deployments use ollama catalog for vram estimation"
```

---

## Task 6: Dashboard — install sonner and mount Toaster

**Files:**
- Modify: `packages/dashboard/package.json`
- Modify: `packages/dashboard/app/layout.tsx`

We need a toast surface for download-progress notifications that lives across pages (so a user can start a deploy on /deployments, navigate to /nodes, and still see the pull complete). `sonner` is a small, accessible, Tailwind-compatible toast lib.

- [ ] **Step 1: Install sonner**

```bash
npm install --workspace=packages/dashboard sonner@^1.5.0
```

- [ ] **Step 2: Mount the Toaster in the root layout**

Edit `packages/dashboard/app/layout.tsx`. Add the import:
```ts
import { Toaster } from "sonner";
```
Change the `<body>` block so it includes the Toaster (before the closing tag):
```tsx
      <body className="bg-gray-950 text-gray-100 min-h-screen">
        <TopNav />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <Toaster theme="dark" position="bottom-right" closeButton richColors />
      </body>
```

- [ ] **Step 3: Smoke test the build**

```bash
npm run build --workspace=packages/dashboard
```
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/package.json packages/dashboard/package-lock.json package-lock.json \
        packages/dashboard/app/layout.tsx
git commit -m "dashboard: add sonner toast surface to root layout"
```

---

## Task 7: Dashboard — `useDebouncedCallback` hook

**Files:**
- Create: `packages/dashboard/lib/use-debounced-callback.ts`

Tiny hook used by the checkbox list for auto-persist. Pulled out so it's reusable and trivially correct.

- [ ] **Step 1: Implement**

Create `packages/dashboard/lib/use-debounced-callback.ts`:
```ts
"use client";
import { useEffect, useRef } from "react";

export function useDebouncedCallback<Args extends unknown[]>(
  fn: (...args: Args) => void,
  delayMs: number,
): (...args: Args) => void {
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  return (...args: Args) => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => fnRef.current(...args), delayMs);
  };
}
```

- [ ] **Step 2: Commit**

```bash
git add packages/dashboard/lib/use-debounced-callback.ts
git commit -m "dashboard: add useDebouncedCallback hook"
```

---

## Task 8: Dashboard — `<OllamaModelsSection />` on Settings page

**Files:**
- Create: `packages/dashboard/components/ollama-models-section.tsx`
- Modify: `packages/dashboard/app/settings/page.tsx`

Section layout:
- Header row: "Ollama Models" + last-refreshed timestamp + Refresh button.
- Body: scrollable list of catalog entries with a checkbox each. Disabled (greyed) when catalog is empty until a refresh is performed.
- Empty state below the header when `entries.length === 0`: "Catalog is empty. Click Refresh to pull from ollama.com/library."

Selection persistence: every checkbox toggle updates local state immediately AND schedules a debounced PUT to `/api/ollama-catalog/enabled`. A small "Saved" / "Saving…" indicator next to the header reflects state.

- [ ] **Step 1: Implement the component**

Create `packages/dashboard/components/ollama-models-section.tsx`:
```tsx
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch } from "@/lib/api";
import { useDebouncedCallback } from "@/lib/use-debounced-callback";

interface CatalogEntry {
  name: string;
  description: string;
  type: "chat" | "embedding";
  sizes: string[];
  capabilities: string[];
}

interface CatalogResponse {
  entries: CatalogEntry[];
  fetchedAt: string | null;
}

type SaveState = "idle" | "saving" | "saved" | "error";

interface Row {
  tag: string;          // "llama3.1:8b" or "nomic-embed-text"
  modelName: string;
  size: string | null;
  description: string;
  type: "chat" | "embedding";
  capabilities: string[];
}

/** Flatten catalog entries into one row per deployable tag. */
function flatten(entries: CatalogEntry[]): Row[] {
  const rows: Row[] = [];
  for (const e of entries) {
    if (e.sizes.length === 0) {
      rows.push({ tag: e.name, modelName: e.name, size: null, description: e.description, type: e.type, capabilities: e.capabilities });
    } else {
      for (const size of e.sizes) {
        rows.push({ tag: `${e.name}:${size}`, modelName: e.name, size, description: e.description, type: e.type, capabilities: e.capabilities });
      }
    }
  }
  return rows;
}

export function OllamaModelsSection() {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);
  const [enabled, setEnabled] = useState<Set<string>>(new Set());
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [filter, setFilter] = useState("");

  const rows = useMemo(() => flatten(catalog), [catalog]);
  const visibleRows = useMemo(() => {
    if (!filter.trim()) return rows;
    const q = filter.trim().toLowerCase();
    return rows.filter(
      (r) => r.tag.toLowerCase().includes(q) || r.description.toLowerCase().includes(q),
    );
  }, [rows, filter]);

  const load = useCallback(async () => {
    try {
      const [cat, en] = await Promise.all([
        apiFetch<CatalogResponse>("/api/ollama-catalog/catalog"),
        apiFetch<{ enabled: string[] }>("/api/ollama-catalog/enabled"),
      ]);
      setCatalog(cat.entries);
      setFetchedAt(cat.fetchedAt);
      setEnabled(new Set(en.enabled));
    } catch (err) {
      setRefreshError(String(err));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const persistEnabled = useCallback(async (tags: string[]) => {
    setSaveState("saving");
    try {
      await apiFetch("/api/ollama-catalog/enabled", {
        method: "PUT",
        body: JSON.stringify({ enabled: tags }),
      });
      setSaveState("saved");
      setTimeout(() => setSaveState((s) => (s === "saved" ? "idle" : s)), 1500);
    } catch {
      setSaveState("error");
    }
  }, []);

  const debouncedPersist = useDebouncedCallback(persistEnabled, 400);

  const toggle = (tag: string) => {
    setEnabled((prev) => {
      const next = new Set(prev);
      if (next.has(tag)) next.delete(tag);
      else next.add(tag);
      debouncedPersist(Array.from(next));
      return next;
    });
  };

  const refresh = async () => {
    setRefreshing(true);
    setRefreshError(null);
    try {
      const res = await apiFetch<CatalogResponse>("/api/ollama-catalog/catalog/refresh", {
        method: "POST",
      });
      setCatalog(res.entries);
      setFetchedAt(res.fetchedAt);
    } catch (err) {
      setRefreshError(String(err));
    } finally {
      setRefreshing(false);
    }
  };

  const saveLabel: Record<SaveState, string> = {
    idle: "",
    saving: "Saving…",
    saved: "Saved",
    error: "Save failed",
  };

  return (
    <section className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-lg font-semibold">Ollama Models</h2>
        <div className="flex items-center gap-3">
          <span className={`text-xs ${saveState === "error" ? "text-red-400" : "text-gray-500"}`}>
            {saveLabel[saveState]}
          </span>
          <button
            type="button"
            onClick={refresh}
            disabled={refreshing}
            className="px-3 py-1 text-xs rounded bg-green-700/30 border border-green-700/60 text-green-300 hover:bg-green-700/50 disabled:opacity-50"
          >
            {refreshing ? "Refreshing…" : "Refresh catalog"}
          </button>
        </div>
      </div>

      <p className="text-xs text-gray-500 mb-3">
        Pulled from <code>ollama.com/library</code> (cloud-only models excluded).{" "}
        {fetchedAt
          ? `Last refreshed ${new Date(fetchedAt).toLocaleString()}.`
          : "Never refreshed."}{" "}
        Check the model:tag combinations you want available on the Deployments page.
      </p>

      {refreshError && (
        <p className="text-xs text-red-400 mb-3">Refresh failed: {refreshError}</p>
      )}

      {catalog.length === 0 ? (
        <p className="text-sm text-gray-600 py-4">
          Catalog is empty. Click <strong>Refresh catalog</strong> to pull the latest from ollama.com.
        </p>
      ) : (
        <>
          <input
            type="text"
            placeholder="Filter by name or description"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-full mb-2 bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-xs focus:outline-none focus:border-green-500"
          />
          <ul className="max-h-96 overflow-y-auto divide-y divide-gray-800/60 border border-gray-800 rounded">
            {visibleRows.map((r) => {
              const checked = enabled.has(r.tag);
              return (
                <li key={r.tag} className="px-3 py-2 flex items-center gap-3 hover:bg-gray-800/30">
                  <input
                    type="checkbox"
                    className="accent-green-500"
                    checked={checked}
                    onChange={() => toggle(r.tag)}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-gray-200 truncate">
                      {r.tag}{" "}
                      <span className="text-[10px] uppercase tracking-wide text-gray-500 ml-1">
                        {r.type}
                      </span>
                      {r.capabilities.length > 0 && (
                        <span className="text-[10px] text-indigo-400 ml-2">
                          {r.capabilities.join(" · ")}
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 truncate">{r.description}</div>
                  </div>
                </li>
              );
            })}
            {visibleRows.length === 0 && (
              <li className="px-3 py-4 text-center text-gray-600 text-sm">No matches.</li>
            )}
          </ul>
        </>
      )}
    </section>
  );
}
```

- [ ] **Step 2: Mount it on the Settings page**

Edit `packages/dashboard/app/settings/page.tsx`. Add the import near the top:
```ts
import { OllamaModelsSection } from "@/components/ollama-models-section";
```
Then add the section at the bottom of the JSX, just before the closing `</div>` of the root container, after the existing "Quick Install" section:
```tsx
      <OllamaModelsSection />
    </div>
  );
}
```

- [ ] **Step 3: Smoke test by running the dev dashboard**

```bash
npm run dev:dashboard
```
In another terminal, ensure the server is running too:
```bash
npm run dev:server
```
Open http://localhost:3000/settings. Verify:
- The "Ollama Models" section appears at the bottom of the page.
- "Refresh catalog" fires `POST /api/ollama-catalog/catalog/refresh` and the list populates.
- Toggling checkboxes shows "Saving…" → "Saved" within ~500ms.
- Reloading the page restores the checked state.

Stop both dev processes when done.

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/components/ollama-models-section.tsx \
        packages/dashboard/app/settings/page.tsx
git commit -m "dashboard: ollama models selection section on settings page"
```

---

## Task 9: Dashboard — Deployments page uses `/api/ollama-catalog/available` and reacts to live updates

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx`

The deployments page currently fetches `/api/recipes/ollama-models` (line 155). Switch to `/api/ollama-catalog/available` and add an SSE handler so a change on the Settings page propagates without a manual reload.

The `/available` endpoint now returns rows shaped as:
```ts
type AvailableRow = {
  tag: string;             // "llama3.1:8b" or "nomic-embed-text"
  modelName: string;
  size: string | null;
  type: "chat" | "embedding";
  description: string;
  capabilities: string[];
};
```
The existing `ollamaModels` state holds `{ name, size, type?, description }` where `name` is the actual pull tag passed to `POST /api/deployments` as `modelName`. We map `tag → name` and `size` (param size like `"8b"`) into the existing field, which is rendered for display only.

- [ ] **Step 1: Switch the fetch URL**

In `packages/dashboard/app/deployments/page.tsx`, locate the `loadData` callback (around line 149). Replace this line:
```ts
      apiFetch<{ name: string; size: string; type?: string; description: string }[]>("/api/recipes/ollama-models"),
```
with:
```ts
      apiFetch<{ tag: string; modelName: string; size: string | null; type: "chat" | "embedding"; description: string; capabilities: string[] }[]>("/api/ollama-catalog/available"),
```
Just below, the destructuring `.then(([r, n, d, idle, om]) => {` already assigns to `om`. Update the `setOllamaModels` call to adapt the new shape:
```ts
        setOllamaModels(
          om.map((m) => ({
            name: m.tag,
            size: m.size ?? "",
            type: m.type,
            description: m.description,
          })),
        );
```
(`setOllamaModels` keeps its existing field shape so the existing `<option>` JSX at lines 600-611 doesn't change.)

- [ ] **Step 2: React to SSE `ollama-catalog:updated`**

Inside the `handleSSE` callback (around line 188), add a new branch alongside the existing event handlers:
```ts
    if (event.type === "ollama-catalog:updated") {
      apiFetch<{ tag: string; modelName: string; size: string | null; type: "chat" | "embedding"; description: string; capabilities: string[] }[]>(
        "/api/ollama-catalog/available",
      )
        .then((om) =>
          setOllamaModels(
            om.map((m) => ({
              name: m.tag,
              size: m.size ?? "",
              type: m.type,
              description: m.description,
            })),
          ),
        )
        .catch(() => {});
    }
```

- [ ] **Step 3: Smoke test in the browser**

Start `npm run dev` from repo root. Open /settings and /deployments side-by-side (two tabs). On the Settings tab, refresh the catalog and check/uncheck a model. The Deployments tab's "Model" dropdown should update within ~500ms without a page reload.

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/app/deployments/page.tsx
git commit -m "dashboard: deployments uses ollama-catalog/available + live updates"
```

---

## Task 10: Agent — emit structured pull-progress

**Files:**
- Modify: `packages/agent/src/runtime/ollama.ts`
- Modify: `packages/agent/src/index.ts`
- Run: `./scripts/bump-agent-version.sh`

The agent already streams Ollama pull progress through `onLog` (which becomes free-text deployment logs). We add a parallel `onProgress` callback that emits structured `{ status, percent, current, total }` so the server can broadcast a clean `deployment:progress` event instead of relying on log-line regex.

- [ ] **Step 1: Extend `pullModel()` and `deployModel()` signatures**

In `packages/agent/src/runtime/ollama.ts`:

Add this interface near the top, just below `OllamaStatus`:
```ts
export interface OllamaPullProgress {
  /** Raw status string from Ollama (e.g. "pulling manifest", "downloading"). */
  status: string;
  /** 0-100 if Ollama reported byte counts, else null. */
  percent: number | null;
  /** Bytes pulled so far if reported. */
  current: number | null;
  /** Total bytes for the current layer if reported. */
  total: number | null;
}
```
Update `pullModel`'s signature and body to accept `onProgress`. Replace the existing `pullModel` function with:
```ts
async function pullModel(
  modelName: string,
  onLog?: (line: string) => void,
  signal?: AbortSignal,
  onProgress?: (p: OllamaPullProgress) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new Error("Aborted")); return; }

    const url = new URL(`${OLLAMA_API}/api/pull`);

    const req = http.request(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    }, (res) => {
      let buffer = "";
      res.on("data", (chunk: Buffer) => {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const msg = JSON.parse(line);
            if (msg.status) {
              const percent = msg.completed && msg.total
                ? Math.round(msg.completed / msg.total * 100)
                : null;
              const pct = percent !== null ? ` ${percent}%` : "";
              onLog?.(`${msg.status}${pct}\n`);
              onProgress?.({
                status: String(msg.status),
                percent,
                current: typeof msg.completed === "number" ? msg.completed : null,
                total: typeof msg.total === "number" ? msg.total : null,
              });
            }
            if (msg.error) {
              reject(new Error(msg.error));
              return;
            }
          } catch { /* partial JSON */ }
        }
      });
      res.on("end", () => resolve());
      res.on("error", reject);
    });

    req.on("error", (err) => {
      if (signal?.aborted) resolve();
      else reject(err);
    });
    signal?.addEventListener("abort", () => req.destroy());
    req.write(JSON.stringify({ name: modelName, stream: true }));
    req.end();
  });
}
```
Now update `deployModel`'s signature to thread `onProgress` through. Change the signature line to:
```ts
export async function deployModel(
  deploymentId: string,
  modelName: string,
  onLog?: (line: string) => void,
  onStatus?: (status: string, error?: string) => void,
  modelType?: "chat" | "embedding",
  onProgress?: (p: OllamaPullProgress) => void,
): Promise<{ port: number; vramActual: number }> {
```
And update the `await pullModel(...)` call inside it (currently line 139) to:
```ts
    await pullModel(modelName, onLog, abortController.signal, onProgress);
```

- [ ] **Step 2: Wire the callback in the agent's deploy handler**

In `packages/agent/src/index.ts`, find the call site of `ollamaDeployModel(...)` (search for `ollamaDeployModel`). It currently passes `onLog` and `onStatus`. Add a sixth argument that emits the new WS message:
```ts
await ollamaDeployModel(
  deploymentId,
  modelName,
  /* onLog   */ (line) => sendMsg("agent:deployment:log", { deploymentId, log: line }),
  /* onStatus*/ (status, error) => sendMsg("agent:deployment:status", { deploymentId, status, port: 11434, error }),
  modelType,
  /* onProgress */ (p) =>
    sendMsg("agent:ollama:pull-progress", {
      deploymentId,
      status: p.status,
      percent: p.percent,
      current: p.current,
      total: p.total,
    }),
);
```
**Adapt the existing surrounding code** — keep all currently-passed arguments intact; the only change is appending the sixth argument. Read the current invocation first and edit minimally.

- [ ] **Step 3: Bump the agent version**

Per CLAUDE.md, any change under `packages/agent/src/` requires bumping the agent version:
```bash
./scripts/bump-agent-version.sh
```
Verify `packages/agent/package.json` shows the bumped patch number.

- [ ] **Step 4: Type-check the agent**

```bash
npm run build --workspace=packages/agent
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/ollama.ts \
        packages/agent/src/index.ts \
        packages/agent/package.json
git commit -m "agent: emit structured pull-progress for ollama deploys"
```

---

## Task 11: Server — forward `agent:ollama:pull-progress` as `deployment:progress` SSE

**Files:**
- Modify: `packages/server/src/ws/agent-hub.ts`

The dashboard's deployments page already renders `deployment:progress` events with `phase` and `phaseProgress`. We translate the new agent message into that shape so the existing renderer just works — no new SSE event type, no new dashboard wiring needed for the inline-row progress bar.

- [ ] **Step 1: Write the failing unit test**

We don't have a unit test for `AgentHub` today, so the simpler check is a small focused test that exercises the translation. Create `packages/server/src/ws/agent-hub.pull-progress.test.ts`:
```ts
import { describe, expect, it, vi } from "vitest";

// Lightweight test: import the translation function only if we extract it.
// For now we assert via behavior: feed a parsed message and observe sse.broadcast.

vi.mock("../sse.js", () => ({
  broadcast: vi.fn(),
}));
vi.mock("../prisma.js", () => ({
  prisma: { /* not touched in this test */ },
}));
vi.mock("../metrics-buffer.js", () => ({
  metricsBuffer: { push: vi.fn() },
}));

import { broadcast as sseBroadcast } from "../sse.js";
import { handleOllamaPullProgress } from "./agent-hub.js";

describe("handleOllamaPullProgress", () => {
  it("translates a pull-progress payload to a deployment:progress SSE event", () => {
    handleOllamaPullProgress({
      deploymentId: "dep-1",
      status: "downloading",
      percent: 42,
      current: 4200000,
      total: 10000000,
    });
    expect(sseBroadcast).toHaveBeenCalledWith({
      type: "deployment:progress",
      payload: {
        deploymentId: "dep-1",
        phase: "downloading",
        phaseProgress: 42,
        current: 4200000,
        total: 10000000,
      },
    });
  });

  it("passes through non-downloading statuses verbatim as the phase", () => {
    handleOllamaPullProgress({
      deploymentId: "dep-2",
      status: "pulling manifest",
      percent: null,
      current: null,
      total: null,
    });
    expect(sseBroadcast).toHaveBeenLastCalledWith({
      type: "deployment:progress",
      payload: {
        deploymentId: "dep-2",
        phase: "pulling manifest",
        phaseProgress: 0,
        current: null,
        total: null,
      },
    });
  });
});
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
npx vitest run packages/server/src/ws/agent-hub.pull-progress.test.ts
```
Expected: FAIL — `handleOllamaPullProgress is not exported`.

- [ ] **Step 3: Implement the handler**

In `packages/server/src/ws/agent-hub.ts`, add this exported function just below the existing interfaces (before `class AgentHub`):
```ts
export interface OllamaPullProgressMsg {
  deploymentId: string;
  status: string;
  percent: number | null;
  current: number | null;
  total: number | null;
}

/**
 * Translate an `agent:ollama:pull-progress` payload into the canonical
 * `deployment:progress` SSE shape the dashboard already renders. Kept as a
 * named export so it can be unit-tested without spinning up a WebSocket.
 */
export function handleOllamaPullProgress(payload: OllamaPullProgressMsg): void {
  sseBroadcast({
    type: "deployment:progress",
    payload: {
      deploymentId: payload.deploymentId,
      phase: payload.status === "downloading" ? "downloading" : payload.status,
      phaseProgress: payload.percent ?? 0,
      current: payload.current,
      total: payload.total,
    },
  });
}
```
Then add a case to the switch in `handleConnection`'s message handler (alongside the existing `agent:deployment:progress` case at line 415):
```ts
          case "agent:ollama:pull-progress": {
            handleOllamaPullProgress(msg.payload as OllamaPullProgressMsg);
            break;
          }
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
npx vitest run packages/server/src/ws/agent-hub.pull-progress.test.ts
```
Expected: both tests PASS.

- [ ] **Step 5: Run the full suite**

```bash
npm test
```
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/ws/agent-hub.ts \
        packages/server/src/ws/agent-hub.pull-progress.test.ts
git commit -m "server: forward ollama pull-progress as deployment:progress SSE"
```

---

## Task 12: Dashboard — Ollama pull toast notification

**Files:**
- Create: `packages/dashboard/components/deployment-pull-toast.tsx`
- Modify: `packages/dashboard/app/layout.tsx`

The deployments-page inline row already shows progress for an in-flight pull (because we reused the existing `deployment:progress` SSE branch). The user spec also asks for a *notification* visible regardless of page — a sticky toast that lives in the global Toaster and updates as progress comes in.

Design: a small client component mounted in the root layout that subscribes to SSE. When it sees a `deployment:progress` event with `phase === "downloading"`, it opens (or updates) a sticky `sonner.toast.custom` per `deploymentId`. When a `deployment:status` event arrives with status `running`/`failed`/`stopped` for that deployment, the toast resolves and auto-dismisses.

- [ ] **Step 1: Implement the toast helper**

Create `packages/dashboard/components/deployment-pull-toast.tsx`:
```tsx
"use client";

import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { useSSE, type SseEvent } from "@/lib/sse";

interface ToastState {
  toastId: string | number;
  modelName?: string;
}

export function DeploymentPullToast() {
  const active = useRef<Map<string, ToastState>>(new Map());

  const handle = (event: SseEvent) => {
    if (event.type === "deployment:progress") {
      const p = event.payload as {
        deploymentId: string;
        phase: string;
        phaseProgress: number;
        current?: number | null;
        total?: number | null;
      };
      if (p.phase !== "downloading") return;
      const prev = active.current.get(p.deploymentId);
      const message = `Pulling model… ${p.phaseProgress}%`;
      const description =
        p.current != null && p.total != null
          ? `${formatBytes(p.current)} / ${formatBytes(p.total)}`
          : "Streaming layers from Ollama registry";
      if (prev) {
        toast.message(message, { id: prev.toastId, description, duration: Infinity });
      } else {
        const id = toast.loading(message, { description, duration: Infinity });
        active.current.set(p.deploymentId, { toastId: id });
      }
    }
    if (event.type === "deployment:status") {
      const p = event.payload as { deploymentId: string; status: string; error?: string };
      const prev = active.current.get(p.deploymentId);
      if (!prev) return;
      if (p.status === "running") {
        toast.success("Model loaded", { id: prev.toastId, duration: 4000 });
        active.current.delete(p.deploymentId);
      } else if (["failed", "stopped", "evicted"].includes(p.status)) {
        toast.error(p.status === "failed" ? "Pull failed" : "Deployment stopped", {
          id: prev.toastId,
          description: p.error,
          duration: 6000,
        });
        active.current.delete(p.deploymentId);
      }
    }
    if (event.type === "deployment:deleted") {
      const p = event.payload as { deploymentId: string };
      const prev = active.current.get(p.deploymentId);
      if (prev) {
        toast.dismiss(prev.toastId);
        active.current.delete(p.deploymentId);
      }
    }
  };

  // Use a ref-stable handler since useSSE captures it once.
  const handleRef = useRef(handle);
  handleRef.current = handle;
  useSSE((e) => handleRef.current(e));

  useEffect(() => {
    // Dismiss any orphaned toasts on unmount (e.g. dev HMR reload).
    return () => {
      for (const { toastId } of active.current.values()) toast.dismiss(toastId);
      active.current.clear();
    };
  }, []);

  return null;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
```

- [ ] **Step 2: Mount it in the root layout**

Edit `packages/dashboard/app/layout.tsx`. Add the import:
```ts
import { DeploymentPullToast } from "@/components/deployment-pull-toast";
```
And include it inside `<body>`, right after `<Toaster …/>`:
```tsx
        <Toaster theme="dark" position="bottom-right" closeButton richColors />
        <DeploymentPullToast />
```

- [ ] **Step 3: Manual end-to-end smoke test**

Bring up the stack:
```bash
./scripts/build-agent-bundles.sh
MANAGER_ADVERTISE_HOST=$(hostname -I | awk '{print $1}') SSH_USER=$USER docker compose up -d --build
```
Wait for an agent to be online. Then:
1. Open http://localhost:3000/settings → "Refresh catalog" → tick a small model that isn't yet pulled locally (e.g. `mistral:7b` if not already cached).
2. Navigate to http://localhost:3000/deployments. Pick the Ollama runtime tab. The newly-enabled model should appear in the dropdown.
3. Select it, pick an online node, click Deploy.
4. Immediately navigate away to /nodes. The bottom-right toast should be visible and updating with the pull percentage.
5. Wait for it to resolve. Confirm:
   - Toast turns green and says "Model loaded" → auto-dismisses in 4s.
   - Returning to /deployments shows the deployment as `running`.

If the model was already pulled previously, Ollama skips the download — to force a real pull, `docker exec` into the Ollama daemon and remove the model (or pick a model you haven't pulled).

Bring the stack down when done:
```bash
docker compose down
```

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/components/deployment-pull-toast.tsx \
        packages/dashboard/app/layout.tsx
git commit -m "dashboard: global toast for ollama pull progress"
```

---

## Task 13: Final sweep — remove the dead legacy `ollama-models.json` path

**Files:**
- Delete: `packages/agent/src/ollama-models.json`
- Modify: `packages/agent/src/runtime/ollama.ts` — `getOllamaModels()` becomes a no-op returning `[]`.
- Modify: `packages/agent/src/index.ts` — drop the `agent:ollama-models` send.
- Modify: `packages/server/src/routes/recipes.ts` — `/ollama-models` endpoint now returns `[]` (kept temporarily for older agents that may still poll it, or remove if there are no external consumers).
- Modify: `packages/server/src/ws/agent-hub.ts` — keep handling `agent:ollama-models` but treat it as legacy / no-op (older agents may still emit it; no errors).
- Run: `./scripts/bump-agent-version.sh`

The catalog flow fully owns model selection; the agent should not be the source of truth anymore. We *don't* delete `getOllamaModels()` from the server's `AgentHub` interface because the deployments route's fallback at Task 5 references it for legacy compatibility — but the agent stops emitting, so the in-memory list stays empty.

- [ ] **Step 1: Delete the JSON file**

```bash
git rm packages/agent/src/ollama-models.json
```

- [ ] **Step 2: Reduce `getOllamaModels()` to a no-op**

In `packages/agent/src/runtime/ollama.ts`, replace:
```ts
/** Load curated model list. */
export function getOllamaModels(): OllamaModel[] {
  try {
    return JSON.parse(readFileSync(join(__dirname, "../../ollama-models.json"), "utf-8"));
  } catch {
    return [];
  }
}
```
with:
```ts
/**
 * Legacy: the manager's Settings page is now the source of truth for the
 * Ollama catalog. Kept exported so callers don't break; always returns [].
 */
export function getOllamaModels(): OllamaModel[] {
  return [];
}
```
Also remove the now-unused `readFileSync` import if nothing else uses it. Check the top of the file — if `readFileSync` and `join` are only used by the removed body, drop them.

- [ ] **Step 3: Stop sending `agent:ollama-models`**

In `packages/agent/src/index.ts`, locate the block at lines 225-232 (the `agent:ollama-models` send). Delete it entirely:
```ts
    // Report available Ollama models
    const ollamaModels = getOllamaModels();
    if (ollamaModels.length > 0) {
      ws!.send(JSON.stringify({
        type: "agent:ollama-models",
        payload: { models: ollamaModels },
      }));
    }
```
Also drop the now-unused `getOllamaModels` from the import at line 10 if nothing else references it (search for other usages in that file first).

- [ ] **Step 4: Bump agent version**

```bash
./scripts/bump-agent-version.sh
```

- [ ] **Step 5: Confirm the server tolerates legacy agents**

No code change required: the existing `agent:ollama-models` handler in `agent-hub.ts:271-275` stores whatever it receives in `this.ollamaModels`. Legacy agents send a non-empty list; new agents send nothing. The deployments route prefers the catalog (Task 5), so either way works. Just verify by reading the existing handler — no edit needed.

- [ ] **Step 6: Run the full test suite**

```bash
npm test
```
Expected: green.

- [ ] **Step 7: End-to-end smoke (optional but recommended)**

Bring the stack back up per Task 12. Verify:
- /settings still works — the new section is the only place models appear.
- /deployments with runtime=ollama only shows checked models.
- A fresh deploy of an unpulled model still shows the toast and rolls into `running`.

- [ ] **Step 8: Commit**

```bash
git add packages/agent/src/runtime/ollama.ts \
        packages/agent/src/index.ts \
        packages/agent/package.json
git commit -m "agent: remove legacy ollama-models.json (catalog is now source of truth)"
```

---

## Done criteria

- Settings page has an "Ollama Models" section with refresh button, last-refreshed timestamp, and a checkbox list driven by `ollama.com/library`.
- Toggling checkboxes auto-persists (debounced ~400ms) to `Setting` table key `ollama.enabled.json`.
- Deployments page's Ollama dropdown reflects the current selection live (without a manual reload) when the Settings page changes.
- Deploying an Ollama model that isn't cached locally produces a sticky bottom-right toast with a live percentage, visible across page navigation.
- `npm test` is green.
- Agent version is bumped per the policy.
