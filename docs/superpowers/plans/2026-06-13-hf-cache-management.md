# HF Cache Management Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dashboard page that lists what's in the Hugging Face cache (`HF_HOME`) per cache group (shared NFS or per-node local disk), with size / last-deployed info and safe deletion of stale repos.

**Architecture:** Agents scan their own `HF_HOME/hub` with a native Node fs walk and push an `agent:hf-cache` inventory message (same pattern as recipes); inventories are grouped by a `.dgx-cache-id` UUID marker file so a shared NFS cache shows once; the server enriches repos with in-use/last-deployed info from the Deployment table and exposes `GET/POST/DELETE` under `/api/hf-cache`; the dashboard's stub Models page becomes the cache manager.

**Tech Stack:** TypeScript (ES modules, strict), Express 5, Prisma/SQLite, ws, Next.js 15 + Tailwind, Vitest + fast-check + supertest.

**Spec:** `docs/superpowers/specs/2026-06-12-hf-cache-management-design.md`. Two deliberate deviations (amended into the spec in Task 8):
1. `DELETE` carries the repo id as a **query parameter** (`DELETE /api/hf-cache/:cacheId?repoId=org%2Fname&kind=model`) instead of a path segment — URL-encoded slashes in path segments are unreliable across HTTP stacks.
2. Org-less legacy repo ids (`gpt2`, `squad`) are valid — the cache really contains `models--gpt2` style dirs — so a repoId is 1 **or** 2 safe segments, not strictly `org/name`.

**PRECONDITION:** The working tree has pre-existing uncommitted changes in `packages/agent/` (`package.json`, `src/runtime/sparkrun.ts`, `src/runtime/sparkrun.test.ts`). Ask the user to commit or stash them BEFORE starting — Task 4 runs the agent version-bump script, which edits `packages/agent/package.json` and must be committable on its own. Never `git add -A`; stage only the files named in each task.

**File map:**

| File | Responsibility |
|---|---|
| `packages/agent/src/runtime/hf-cache.ts` (create) | Pure-ish cache logic: dir-name codec, scan, cache-id marker, traversal-guarded delete |
| `packages/agent/src/runtime/hf-cache.test.ts` (create) | Unit + property tests for the above |
| `packages/agent/src/index.ts` (modify) | Thin `cmd:hf-cache:scan` / `cmd:hf-cache:delete` handlers + inventory push |
| `packages/server/src/hf-cache/grouping.ts` (create) | Pure: group inventories by cacheId, match repos to deployments, in-use/last-deployed |
| `packages/server/src/hf-cache/grouping.test.ts` (create) | Unit + property tests for the above |
| `packages/server/src/ws/agent-hub.ts` (modify) | Store `agent:hf-cache` inventories per nodeId, SSE-broadcast, getter |
| `packages/server/src/routes/hf-cache.ts` (create) | REST: GET inventory (grouped+enriched), POST scan, DELETE repo; OpenAPI JSDoc |
| `packages/server/src/index.ts` (modify) | Mount `/api/hf-cache` |
| `packages/server/src/__tests__/integration/hf-cache.routes.test.ts` (create) | supertest integration: grouping, enrichment, 409/404/503 paths |
| `packages/dashboard/lib/hf-cache.ts` (create) | Pure UI helpers: `formatBytes`, `sortRepos`, shared types |
| `packages/dashboard/lib/hf-cache.test.ts` (create) | Unit tests for UI helpers |
| `packages/dashboard/app/models/page.tsx` (rewrite) | The cache management page |
| `packages/dashboard/components/top-nav.tsx` (modify) | Add the `/models` nav link (missing today) |

---

### Task 1: Agent repo-dir codec (`parseRepoDirName`, `repoDirName`, `isSafeRepoId`)

The HF cache stores a repo `org/name` as a directory `models--org--name` (datasets as `datasets--…`). huggingface_hub's own convention: split on `--`, first part is the kind, the rest joined with `/`. Org-less legacy repos (`gpt2`) produce `models--gpt2`.

**Files:**
- Create: `packages/agent/src/runtime/hf-cache.ts`
- Test: `packages/agent/src/runtime/hf-cache.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `packages/agent/src/runtime/hf-cache.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import { fc, it as fcIt } from "@fast-check/vitest";
import { parseRepoDirName, repoDirName, isSafeRepoId } from "./hf-cache.js";

describe("parseRepoDirName", () => {
  it("decodes a model repo dir", () => {
    expect(parseRepoDirName("models--meta-llama--Llama-3.1-8B-Instruct")).toEqual({
      kind: "model",
      repoId: "meta-llama/Llama-3.1-8B-Instruct",
    });
  });

  it("decodes a dataset repo dir", () => {
    expect(parseRepoDirName("datasets--HuggingFaceH4--ultrachat_200k")).toEqual({
      kind: "dataset",
      repoId: "HuggingFaceH4/ultrachat_200k",
    });
  });

  it("decodes an org-less legacy repo (models--gpt2)", () => {
    expect(parseRepoDirName("models--gpt2")).toEqual({ kind: "model", repoId: "gpt2" });
  });

  it("does not split single dashes inside names", () => {
    expect(parseRepoDirName("models--meta-llama--Meta-Llama-3-8B")).toEqual({
      kind: "model",
      repoId: "meta-llama/Meta-Llama-3-8B",
    });
  });

  it("returns null for non-repo hub entries", () => {
    expect(parseRepoDirName("version.txt")).toBeNull();
    expect(parseRepoDirName(".locks")).toBeNull();
    expect(parseRepoDirName("spaces--foo--bar")).toBeNull(); // unsupported kind
    expect(parseRepoDirName("models--")).toBeNull();          // empty segment
  });
});

describe("repoDirName", () => {
  it("encodes model and dataset repos", () => {
    expect(repoDirName("model", "meta-llama/Llama-3.1-8B-Instruct"))
      .toBe("models--meta-llama--Llama-3.1-8B-Instruct");
    expect(repoDirName("dataset", "squad")).toBe("datasets--squad");
  });
});

/** A single repoId segment as HF allows it: letters, digits, dot, dash,
 *  underscore — excluding `.`/`..` and any `--` (which would be ambiguous in
 *  the directory encoding, a limitation huggingface_hub shares). */
const segmentArb = fc
  .stringMatching(/^[A-Za-z0-9._-]{1,32}$/)
  .filter((s) => s !== "." && s !== ".." && !s.includes("--"));

const repoIdArb = fc
  .oneof(segmentArb, fc.tuple(segmentArb, segmentArb).map(([a, b]) => `${a}/${b}`));

describe("codec round-trip", () => {
  /** Invariant: for any valid repoId whose segments contain no `--`,
   *  encoding to a cache dir name and parsing it back is the identity. */
  fcIt.prop([repoIdArb, fc.constantFrom("model" as const, "dataset" as const)])(
    "parseRepoDirName(repoDirName(kind, id)) === {kind, id}",
    (repoId, kind) => {
      expect(parseRepoDirName(repoDirName(kind, repoId))).toEqual({ kind, repoId });
    },
  );
});

describe("isSafeRepoId", () => {
  it("accepts normal one- and two-segment ids", () => {
    expect(isSafeRepoId("gpt2")).toBe(true);
    expect(isSafeRepoId("meta-llama/Llama-3.1-8B-Instruct")).toBe(true);
  });

  it("rejects traversal and malformed ids", () => {
    expect(isSafeRepoId("")).toBe(false);
    expect(isSafeRepoId("..")).toBe(false);
    expect(isSafeRepoId("../etc")).toBe(false);
    expect(isSafeRepoId("a/..")).toBe(false);
    expect(isSafeRepoId("./a")).toBe(false);
    expect(isSafeRepoId("a/b/c")).toBe(false);
    expect(isSafeRepoId("/etc")).toBe(false);
    expect(isSafeRepoId("a b")).toBe(false);
    expect(isSafeRepoId("a\\b")).toBe(false);
  });

  /** Invariant: every id our generator considers valid is accepted. */
  fcIt.prop([repoIdArb])("accepts all generator-valid ids", (repoId) => {
    expect(isSafeRepoId(repoId)).toBe(true);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run packages/agent/src/runtime/hf-cache.test.ts`
Expected: FAIL — `Cannot find module './hf-cache.js'` (or equivalent resolve error).

- [ ] **Step 3: Write the implementation**

Create `packages/agent/src/runtime/hf-cache.ts`:

```typescript
import {
  existsSync, lstatSync, readdirSync, readFileSync, rmSync, statfsSync, writeFileSync,
} from "node:fs";
import { join, resolve, sep } from "node:path";
import { randomUUID } from "node:crypto";

export type RepoKind = "model" | "dataset";

export interface CachedRepo {
  repoId: string;
  kind: RepoKind;
  sizeBytes: number;
  nFiles: number;
  revisions: number;     // snapshot dir count
  lastModified: string;  // ISO — max file mtime = download/update time
}

export interface HfCacheInventoryPayload {
  cacheId: string;
  hfHome: string;
  scannedAt: string;
  totalBytes: number;
  diskFreeBytes: number;
  error?: string;
  repos: CachedRepo[];
}

const KIND_BY_PREFIX: Record<string, RepoKind> = { models: "model", datasets: "dataset" };

/** Decode an HF cache dir name (`models--org--name`) into kind + repoId.
 *  Mirrors huggingface_hub: split on `--`, first part is the kind, the rest
 *  joined with `/`. Returns null for hub sidecars (`.locks`, `version.txt`)
 *  and unsupported kinds (`spaces--…`). */
export function parseRepoDirName(dirName: string): { kind: RepoKind; repoId: string } | null {
  const parts = dirName.split("--");
  if (parts.length < 2) return null;
  const kind = KIND_BY_PREFIX[parts[0]];
  if (!kind) return null;
  const segments = parts.slice(1);
  if (segments.some((s) => s.length === 0)) return null;
  return { kind, repoId: segments.join("/") };
}

export function repoDirName(kind: RepoKind, repoId: string): string {
  return `${kind}s--${repoId.split("/").join("--")}`;
}

const REPO_SEGMENT_RE = /^[A-Za-z0-9._-]+$/;

/** A repoId is safe to turn into a deletable path iff it is 1–2 segments of
 *  HF-legal characters, with `.`/`..` explicitly excluded (the char class
 *  alone would admit them). Org-less legacy ids ("gpt2") are 1 segment. */
export function isSafeRepoId(repoId: string): boolean {
  const segments = repoId.split("/");
  if (segments.length < 1 || segments.length > 2) return false;
  return segments.every((s) => REPO_SEGMENT_RE.test(s) && s !== "." && s !== "..");
}
```

(The fs imports are unused until Tasks 2–3 — that's fine, TypeScript doesn't error on unused imports outside `noUnusedLocals` for imports used later in the same task sequence; if `npm run build` complains at the end of this task, keep only `node:path`/`node:crypto` lines you need and add the rest in Task 2/3.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run packages/agent/src/runtime/hf-cache.test.ts`
Expected: PASS (all describe blocks green).

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/hf-cache.ts packages/agent/src/runtime/hf-cache.test.ts
git commit -m "feat(agent): HF cache repo dir-name codec + repoId safety check

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Agent `deleteCachedRepo` (path-safety boundary)

**Files:**
- Modify: `packages/agent/src/runtime/hf-cache.ts`
- Test: `packages/agent/src/runtime/hf-cache.test.ts`

- [ ] **Step 1: Write the failing tests**

Append to `packages/agent/src/runtime/hf-cache.test.ts` (add the new imports to the existing import lines):

```typescript
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { deleteCachedRepo } from "./hf-cache.js";

/** Build a minimal fake cache: hfHome/hub/<repoDir>/blobs/weights */
function makeFakeCache(repoDirs: string[]): string {
  const hfHome = mkdtempSync(join(tmpdir(), "hf-cache-test-"));
  for (const dir of repoDirs) {
    const blobs = join(hfHome, "hub", dir, "blobs");
    mkdirSync(blobs, { recursive: true });
    writeFileSync(join(blobs, "weights"), "x".repeat(1000));
  }
  return hfHome;
}

describe("deleteCachedRepo", () => {
  it("deletes the targeted repo dir and leaves siblings alone", () => {
    const hfHome = makeFakeCache(["models--org--alpha", "models--org--beta"]);
    deleteCachedRepo(hfHome, "model", "org/alpha");
    expect(existsSync(join(hfHome, "hub", "models--org--alpha"))).toBe(false);
    expect(existsSync(join(hfHome, "hub", "models--org--beta"))).toBe(true);
    rmSync(hfHome, { recursive: true, force: true });
  });

  it("deletes dataset repos via kind", () => {
    const hfHome = makeFakeCache(["datasets--squad"]);
    deleteCachedRepo(hfHome, "dataset", "squad");
    expect(existsSync(join(hfHome, "hub", "datasets--squad"))).toBe(false);
    rmSync(hfHome, { recursive: true, force: true });
  });

  it("throws for a repo that is not in the cache", () => {
    const hfHome = makeFakeCache([]);
    expect(() => deleteCachedRepo(hfHome, "model", "org/ghost")).toThrow(/not in cache/i);
    rmSync(hfHome, { recursive: true, force: true });
  });

  /** Invariant: any unsafe repoId is rejected BEFORE any filesystem access,
   *  and nothing outside hub/ is ever touched. We plant a sentinel file
   *  outside hub/ and assert it survives every attempt. */
  fcIt.prop([fc.string({ minLength: 1, maxLength: 64 }).filter((s) => !isSafeRepoId(s))])(
    "rejects every unsafe repoId without touching the filesystem",
    (badId) => {
      const hfHome = makeFakeCache(["models--org--alpha"]);
      writeFileSync(join(hfHome, "sentinel.txt"), "intact");
      expect(() => deleteCachedRepo(hfHome, "model", badId)).toThrow(/invalid repoId/i);
      expect(existsSync(join(hfHome, "sentinel.txt"))).toBe(true);
      expect(existsSync(join(hfHome, "hub", "models--org--alpha"))).toBe(true);
      rmSync(hfHome, { recursive: true, force: true });
    },
  );

  it("rejects classic traversal attempts", () => {
    const hfHome = makeFakeCache(["models--org--alpha"]);
    for (const evil of ["../..", "a/..", "../hub", "/etc", "..\\..", "org/../alpha"]) {
      expect(() => deleteCachedRepo(hfHome, "model", evil)).toThrow();
    }
    rmSync(hfHome, { recursive: true, force: true });
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run packages/agent/src/runtime/hf-cache.test.ts`
Expected: FAIL — `deleteCachedRepo` is not exported.

- [ ] **Step 3: Write the implementation**

Append to `packages/agent/src/runtime/hf-cache.ts`:

```typescript
/** SECURITY BOUNDARY: the only code that turns a wire-supplied repoId into a
 *  filesystem path. Validates the id, then verifies the resolved target is a
 *  strict descendant of hfHome/hub before rm -rf. */
export function deleteCachedRepo(hfHome: string, kind: RepoKind, repoId: string): void {
  if (!isSafeRepoId(repoId)) throw new Error(`invalid repoId: ${JSON.stringify(repoId)}`);
  const hub = resolve(hfHome, "hub");
  const target = resolve(hub, repoDirName(kind, repoId));
  if (!target.startsWith(hub + sep)) {
    throw new Error(`refusing to delete outside hub/: ${JSON.stringify(repoId)}`);
  }
  if (!existsSync(target)) throw new Error(`not in cache: ${repoId} (${kind})`);
  rmSync(target, { recursive: true });
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run packages/agent/src/runtime/hf-cache.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/hf-cache.ts packages/agent/src/runtime/hf-cache.test.ts
git commit -m "feat(agent): traversal-guarded HF cache repo deletion

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Agent scan — `scanHfCache`, `readOrCreateCacheId`, `buildInventory`

**Files:**
- Modify: `packages/agent/src/runtime/hf-cache.ts`
- Test: `packages/agent/src/runtime/hf-cache.test.ts`

- [ ] **Step 1: Write the failing tests**

Append to `packages/agent/src/runtime/hf-cache.test.ts` (extend the node:fs import with `symlinkSync`, `readFileSync`; import the new functions):

```typescript
import { symlinkSync, readFileSync } from "node:fs";
import { scanHfCache, readOrCreateCacheId, buildInventory } from "./hf-cache.js";

describe("scanHfCache", () => {
  it("reports size from blobs without double-counting snapshot symlinks", () => {
    const hfHome = mkdtempSync(join(tmpdir(), "hf-scan-test-"));
    const repoDir = join(hfHome, "hub", "models--org--alpha");
    mkdirSync(join(repoDir, "blobs"), { recursive: true });
    mkdirSync(join(repoDir, "snapshots", "abc123"), { recursive: true });
    mkdirSync(join(repoDir, "refs"), { recursive: true });
    writeFileSync(join(repoDir, "blobs", "blob1"), "x".repeat(5000));
    writeFileSync(join(repoDir, "refs", "main"), "abc123");
    // HF layout: snapshots contain symlinks back into blobs/
    symlinkSync(join("..", "..", "blobs", "blob1"), join(repoDir, "snapshots", "abc123", "model.safetensors"));

    const repos = scanHfCache(hfHome);
    expect(repos).toHaveLength(1);
    expect(repos[0].repoId).toBe("org/alpha");
    expect(repos[0].kind).toBe("model");
    expect(repos[0].revisions).toBe(1);
    // ≥ blob size, < blob size + 1KB slack (symlink + refs are tiny, not 5000 again)
    expect(repos[0].sizeBytes).toBeGreaterThanOrEqual(5000);
    expect(repos[0].sizeBytes).toBeLessThan(6024);
    expect(new Date(repos[0].lastModified).getTime()).toBeGreaterThan(0);
    rmSync(hfHome, { recursive: true, force: true });
  });

  it("skips non-repo hub entries", () => {
    const hfHome = mkdtempSync(join(tmpdir(), "hf-scan-test-"));
    mkdirSync(join(hfHome, "hub", ".locks"), { recursive: true });
    writeFileSync(join(hfHome, "hub", "version.txt"), "1");
    expect(scanHfCache(hfHome)).toEqual([]);
    rmSync(hfHome, { recursive: true, force: true });
  });

  it("returns [] for a cache with no hub/ dir yet (fresh install, not an error)", () => {
    const hfHome = mkdtempSync(join(tmpdir(), "hf-scan-test-"));
    expect(scanHfCache(hfHome)).toEqual([]);
    rmSync(hfHome, { recursive: true, force: true });
  });

  it("throws when hfHome itself is missing (unmounted NFS must be loud)", () => {
    expect(() => scanHfCache("/nonexistent/hf-home-xyz")).toThrow(/does not exist/i);
  });
});

describe("readOrCreateCacheId", () => {
  it("creates a marker once and returns the same id on subsequent calls", () => {
    const hfHome = mkdtempSync(join(tmpdir(), "hf-id-test-"));
    const first = readOrCreateCacheId(hfHome);
    expect(first).toMatch(/^[0-9a-f-]{36}$/);
    expect(readOrCreateCacheId(hfHome)).toBe(first);
    expect(readFileSync(join(hfHome, ".dgx-cache-id"), "utf8").trim()).toBe(first);
    rmSync(hfHome, { recursive: true, force: true });
  });

  it("respects a pre-existing marker (the shared-NFS case)", () => {
    const hfHome = mkdtempSync(join(tmpdir(), "hf-id-test-"));
    writeFileSync(join(hfHome, ".dgx-cache-id"), "shared-cache-uuid\n");
    expect(readOrCreateCacheId(hfHome)).toBe("shared-cache-uuid");
    rmSync(hfHome, { recursive: true, force: true });
  });

  it("throws when hfHome is missing", () => {
    expect(() => readOrCreateCacheId("/nonexistent/hf-home-xyz")).toThrow(/does not exist/i);
  });
});

describe("buildInventory", () => {
  it("assembles cacheId, totals, free space and repos", () => {
    const hfHome = mkdtempSync(join(tmpdir(), "hf-inv-test-"));
    const blobs = join(hfHome, "hub", "models--org--alpha", "blobs");
    mkdirSync(blobs, { recursive: true });
    writeFileSync(join(blobs, "blob1"), "x".repeat(2000));

    const inv = buildInventory(hfHome);
    expect(inv.hfHome).toBe(hfHome);
    expect(inv.cacheId).toMatch(/^[0-9a-f-]{36}$/);
    expect(inv.repos).toHaveLength(1);
    expect(inv.totalBytes).toBe(inv.repos[0].sizeBytes);
    expect(inv.diskFreeBytes).toBeGreaterThan(0);
    expect(new Date(inv.scannedAt).getTime()).toBeGreaterThan(0);
    rmSync(hfHome, { recursive: true, force: true });
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run packages/agent/src/runtime/hf-cache.test.ts`
Expected: FAIL — `scanHfCache` is not exported.

- [ ] **Step 3: Write the implementation**

Append to `packages/agent/src/runtime/hf-cache.ts`:

```typescript
interface WalkStats { sizeBytes: number; nFiles: number; lastModifiedMs: number }

/** Recursive size walk using lstat: snapshot symlinks count as their own
 *  ~0-byte entries, so blob bytes are never counted twice. */
function walk(dir: string): WalkStats {
  const acc: WalkStats = { sizeBytes: 0, nFiles: 0, lastModifiedMs: 0 };
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) {
      const sub = walk(p);
      acc.sizeBytes += sub.sizeBytes;
      acc.nFiles += sub.nFiles;
      acc.lastModifiedMs = Math.max(acc.lastModifiedMs, sub.lastModifiedMs);
    } else {
      const st = lstatSync(p);
      acc.sizeBytes += st.size;
      acc.nFiles += 1;
      acc.lastModifiedMs = Math.max(acc.lastModifiedMs, st.mtimeMs);
    }
  }
  return acc;
}

export function scanHfCache(hfHome: string): CachedRepo[] {
  if (!existsSync(hfHome)) throw new Error(`HF_HOME does not exist: ${hfHome}`);
  const hub = join(hfHome, "hub");
  if (!existsSync(hub)) return []; // fresh cache — nothing downloaded yet
  const repos: CachedRepo[] = [];
  for (const entry of readdirSync(hub, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const parsed = parseRepoDirName(entry.name);
    if (!parsed) continue; // .locks and other sidecars
    const repoDir = join(hub, entry.name);
    const stats = walk(repoDir);
    const snapshotsDir = join(repoDir, "snapshots");
    const revisions = existsSync(snapshotsDir)
      ? readdirSync(snapshotsDir, { withFileTypes: true }).filter((e) => e.isDirectory()).length
      : 0;
    const lastModifiedMs = stats.lastModifiedMs || lstatSync(repoDir).mtimeMs;
    repos.push({
      ...parsed,
      sizeBytes: stats.sizeBytes,
      nFiles: stats.nFiles,
      revisions,
      lastModified: new Date(lastModifiedMs).toISOString(),
    });
  }
  return repos;
}

/** Cache identity marker. Nodes sharing the cache over NFS read the same
 *  UUID, so the server can group their inventories into one cache group.
 *  `wx` write loses the create race gracefully on shared storage. */
export function readOrCreateCacheId(hfHome: string): string {
  if (!existsSync(hfHome)) throw new Error(`HF_HOME does not exist: ${hfHome}`);
  const marker = join(hfHome, ".dgx-cache-id");
  if (existsSync(marker)) {
    const existing = readFileSync(marker, "utf8").trim();
    if (existing) return existing;
  }
  const id = randomUUID();
  try {
    writeFileSync(marker, `${id}\n`, { flag: "wx" });
    return id;
  } catch {
    const existing = readFileSync(marker, "utf8").trim();
    if (existing) return existing;
    throw new Error(`cache-id marker exists but is unreadable/empty: ${marker}`);
  }
}

export function buildInventory(hfHome: string): HfCacheInventoryPayload {
  const cacheId = readOrCreateCacheId(hfHome);
  const repos = scanHfCache(hfHome);
  const fsStat = statfsSync(hfHome);
  return {
    cacheId,
    hfHome,
    scannedAt: new Date().toISOString(),
    totalBytes: repos.reduce((sum, r) => sum + r.sizeBytes, 0),
    diskFreeBytes: fsStat.bavail * fsStat.bsize,
    repos,
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run packages/agent/src/runtime/hf-cache.test.ts`
Expected: PASS (all suites in the file).

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/runtime/hf-cache.ts packages/agent/src/runtime/hf-cache.test.ts
git commit -m "feat(agent): HF cache scan, cache-id marker, inventory builder

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Wire `cmd:hf-cache:*` into the agent + version bump

Thin glue only — all logic was tested in Tasks 1–3. There is no automated test for the WS dispatch itself (consistent with every other `cmd:*` case in `index.ts`); the integration story is covered server-side in Task 7 and manually at the end.

**Files:**
- Modify: `packages/agent/src/index.ts` (imports near top; new function next to `sendMsg` ~line 431; new cases after `case "cmd:rescan-recipes"` which ends near line 966)
- Modify (via script): `packages/agent/package.json`

- [ ] **Step 1: Add the import**

In `packages/agent/src/index.ts`, alongside the other `./runtime/` imports at the top of the file, add:

```typescript
import { buildInventory, deleteCachedRepo, type RepoKind } from "./runtime/hf-cache.js";
import { resolveHfHome } from "./runtime/sparkrun.js";
```

(If `resolveHfHome` is already imported from `./runtime/sparkrun.js`, just extend that existing import.)

- [ ] **Step 2: Add the inventory sender**

Directly below the existing `sendMsg` function (`packages/agent/src/index.ts:431-434`), add:

```typescript
/** Scan HF_HOME and push the inventory. `error` carries a preceding command
 *  failure (e.g. a failed delete) so it surfaces in the dashboard instead of
 *  vanishing. A scan failure itself (unmounted HF_HOME) is also reported as
 *  an inventory with `error` — never silently dropped. */
function sendHfCacheInventory(error?: string) {
  const hfHome = resolveHfHome();
  try {
    const inventory = buildInventory(hfHome);
    sendMsg("agent:hf-cache", { ...inventory, ...(error ? { error } : {}) });
  } catch (err) {
    // cacheId "" → the server falls back to a per-node group for error rows
    sendMsg("agent:hf-cache", {
      cacheId: "",
      hfHome,
      scannedAt: new Date().toISOString(),
      totalBytes: 0,
      diskFreeBytes: 0,
      repos: [],
      error: error ? `${error}; scan also failed: ${err}` : `scan failed: ${err}`,
    });
  }
}
```

- [ ] **Step 3: Add the command cases**

In `handleCommand`, immediately after the closing brace of `case "cmd:rescan-recipes": { … }` (around `packages/agent/src/index.ts:966`), add:

```typescript
    case "cmd:hf-cache:scan": {
      console.log("[hf-cache] scan requested");
      sendHfCacheInventory();
      break;
    }

    case "cmd:hf-cache:delete": {
      const { repoId, kind } = msg.payload as { repoId: string; kind?: RepoKind };
      try {
        deleteCachedRepo(resolveHfHome(), kind ?? "model", repoId);
        console.log(`[hf-cache] deleted ${kind ?? "model"} ${repoId}`);
        sendHfCacheInventory();
      } catch (err) {
        console.error(`[hf-cache] delete failed: ${err}`);
        sendHfCacheInventory(`delete ${repoId} failed: ${err}`);
      }
      break;
    }
```

- [ ] **Step 4: Verify it compiles and existing tests still pass**

Run: `npx vitest run packages/agent && npm run build --workspace=packages/agent`
Expected: tests PASS, build succeeds with no TypeScript errors.

- [ ] **Step 5: Bump the agent version (MANDATORY — covers all agent edits from Tasks 1–4)**

Run: `./scripts/bump-agent-version.sh`
Expected: prints the new patch version; `packages/agent/package.json` version incremented.

- [ ] **Step 6: Commit**

```bash
git add packages/agent/src/index.ts packages/agent/package.json
git commit -m "feat(agent): handle cmd:hf-cache:scan/delete, push agent:hf-cache inventory

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Server pure module — grouping, matching, usage

**Files:**
- Create: `packages/server/src/hf-cache/grouping.ts`
- Test: `packages/server/src/hf-cache/grouping.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `packages/server/src/hf-cache/grouping.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import { fc, it as fcIt } from "@fast-check/vitest";
import {
  groupInventories, matchRepoToModels, deploymentModelCandidates, repoUsage,
  type HfCacheNodeInventory, type DeploymentUsage,
} from "./grouping.js";

function inv(nodeId: string, cacheId: string, scannedAt = "2026-06-13T10:00:00.000Z"): HfCacheNodeInventory {
  return {
    nodeId, cacheId, scannedAt,
    hfHome: "/mnt/tank/models", totalBytes: 0, diskFreeBytes: 0, repos: [],
  };
}

describe("groupInventories", () => {
  it("merges nodes sharing a cacheId into one group (shared NFS)", () => {
    const groups = groupInventories([inv("n1", "cache-A"), inv("n2", "cache-A")]);
    expect(groups).toHaveLength(1);
    expect(groups[0].cacheId).toBe("cache-A");
    expect(groups[0].nodeIds.sort()).toEqual(["n1", "n2"]);
  });

  it("keeps distinct cacheIds separate (per-node local disks)", () => {
    const groups = groupInventories([inv("n1", "cache-A"), inv("n2", "cache-B")]);
    expect(groups).toHaveLength(2);
  });

  it("handles mixed topologies", () => {
    const groups = groupInventories([inv("n1", "shared"), inv("n2", "shared"), inv("n3", "local-3")]);
    expect(groups).toHaveLength(2);
    expect(groups.find((g) => g.cacheId === "shared")!.nodeIds).toHaveLength(2);
  });

  it("falls back to a per-node group when cacheId is empty (scan-error inventories)", () => {
    const groups = groupInventories([inv("n1", ""), inv("n2", "")]);
    expect(groups).toHaveLength(2);
    expect(groups.map((g) => g.cacheId).sort()).toEqual(["node:n1", "node:n2"]);
  });

  it("the newest scannedAt inventory wins within a group", () => {
    const stale = inv("n1", "shared", "2026-06-13T09:00:00.000Z");
    const fresh = inv("n2", "shared", "2026-06-13T11:00:00.000Z");
    fresh.totalBytes = 42;
    const groups = groupInventories([stale, fresh]);
    expect(groups[0].newest.nodeId).toBe("n2");
    expect(groups[0].newest.totalBytes).toBe(42);
  });
});

describe("matchRepoToModels", () => {
  /** Invariant: matching is case-insensitive in both directions and ignores
   *  null/undefined candidates. */
  fcIt.prop([fc.stringMatching(/^[A-Za-z0-9._-]{1,20}\/[A-Za-z0-9._-]{1,20}$/)])(
    "matches its own upper/lower-cased variants",
    (repoId) => {
      expect(matchRepoToModels(repoId, [repoId.toUpperCase()])).toBe(true);
      expect(matchRepoToModels(repoId, [repoId.toLowerCase()])).toBe(true);
      expect(matchRepoToModels(repoId, [null, undefined, "unrelated/model"])).toBe(false);
    },
  );
});

describe("deploymentModelCandidates", () => {
  it("collects model name and config modelName", () => {
    expect(deploymentModelCandidates("org/m1", JSON.stringify({ modelName: "org/m2" })))
      .toEqual(["org/m1", "org/m2"]);
  });

  it("survives malformed config JSON", () => {
    expect(deploymentModelCandidates("org/m1", "{not json")).toEqual(["org/m1"]);
  });

  it("handles nulls", () => {
    expect(deploymentModelCandidates(null, null)).toEqual([]);
  });
});

function dep(over: Partial<DeploymentUsage>): DeploymentUsage {
  return {
    status: "running", nodeId: "n1", createdAt: "2026-06-01T00:00:00.000Z",
    label: "my-deploy", candidates: ["org/alpha"], clusterNodeIds: [],
    ...over,
  };
}

describe("repoUsage", () => {
  const group = new Set(["n1", "n2"]);

  it("flags in-use for an active deployment on a group node", () => {
    const usage = repoUsage("org/alpha", group, [dep({})]);
    expect(usage.inUse).toBe(true);
    expect(usage.inUseBy).toEqual(["my-deploy"]);
  });

  it("stopped/failed deployments do not block but still set lastDeployedAt", () => {
    const usage = repoUsage("org/alpha", group, [
      dep({ status: "stopped", createdAt: "2026-05-01T00:00:00.000Z" }),
      dep({ status: "failed", createdAt: "2026-05-20T00:00:00.000Z" }),
    ]);
    expect(usage.inUse).toBe(false);
    expect(usage.lastDeployedAt).toBe("2026-05-20T00:00:00.000Z");
  });

  it("evicted deployments count as in use (they can be restored onto the GPU)", () => {
    expect(repoUsage("org/alpha", group, [dep({ status: "evicted" })]).inUse).toBe(true);
  });

  it("ignores deployments on nodes outside the cache group", () => {
    const usage = repoUsage("org/alpha", group, [dep({ nodeId: "other" })]);
    expect(usage.inUse).toBe(false);
    expect(usage.lastDeployedAt).toBeNull();
  });

  it("counts multi-node deployments whose cluster nodes intersect the group", () => {
    const usage = repoUsage("org/alpha", group, [dep({ nodeId: "other", clusterNodeIds: ["n2"] })]);
    expect(usage.inUse).toBe(true);
  });

  it("never matches a repo no deployment references", () => {
    expect(repoUsage("org/unrelated", group, [dep({})]).lastDeployedAt).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run packages/server/src/hf-cache/grouping.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Create `packages/server/src/hf-cache/grouping.ts`:

```typescript
/** Pure helpers for the HF cache management API: grouping per-node agent
 *  inventories into cache groups, and computing per-repo deployment usage.
 *  No IO here — the route layer feeds these from the hub and Prisma. */

export interface HfCacheRepo {
  repoId: string;
  kind: "model" | "dataset";
  sizeBytes: number;
  nFiles: number;
  revisions: number;
  lastModified: string;
}

export interface HfCacheNodeInventory {
  nodeId: string;       // attached by the hub when the message arrives
  cacheId: string;      // "" when the agent's scan failed before reading the marker
  hfHome: string;
  scannedAt: string;
  totalBytes: number;
  diskFreeBytes: number;
  error?: string;
  repos: HfCacheRepo[];
}

export interface CacheGroup {
  cacheId: string;
  nodeIds: string[];
  newest: HfCacheNodeInventory; // repo data source: latest scannedAt in the group
}

/** Inventories with an empty cacheId (failed scans) must not lump together —
 *  fall back to a synthetic per-node key. */
function cacheGroupKey(inv: HfCacheNodeInventory): string {
  return inv.cacheId || `node:${inv.nodeId}`;
}

export function groupInventories(inventories: HfCacheNodeInventory[]): CacheGroup[] {
  const byKey = new Map<string, HfCacheNodeInventory[]>();
  for (const inv of inventories) {
    const key = cacheGroupKey(inv);
    const members = byKey.get(key) ?? [];
    members.push(inv);
    byKey.set(key, members);
  }
  return [...byKey.entries()].map(([cacheId, members]) => ({
    cacheId,
    nodeIds: members.map((m) => m.nodeId),
    // ISO-8601 strings compare correctly as strings
    newest: members.reduce((a, b) => (a.scannedAt >= b.scannedAt ? a : b)),
  }));
}

export function matchRepoToModels(
  repoId: string,
  candidates: Array<string | null | undefined>,
): boolean {
  const target = repoId.toLowerCase();
  return candidates.some((c) => typeof c === "string" && c.toLowerCase() === target);
}

/** The strings a deployment might know its model by: the Model row's name and
 *  the modelName recorded in the deployment's config JSON. Fine-tune models
 *  (local output paths) never equal an HF repoId — correctly so, their weights
 *  are not in the HF cache. */
export function deploymentModelCandidates(
  modelName: string | null | undefined,
  configJson: string | null | undefined,
): string[] {
  const out: string[] = [];
  if (modelName) out.push(modelName);
  if (configJson) {
    try {
      const cfg = JSON.parse(configJson) as Record<string, unknown>;
      if (typeof cfg.modelName === "string") out.push(cfg.modelName);
    } catch {
      // malformed config — fall back to the Model name candidate only
    }
  }
  return out;
}

export interface DeploymentUsage {
  status: string;
  nodeId: string;
  createdAt: string;          // ISO
  label: string;              // displayName ?? model name — shown in 409 errors
  candidates: string[];       // from deploymentModelCandidates
  clusterNodeIds: string[];
}

// "evicted" is deliberately NOT terminal here: an evicted deployment can be
// restored onto the GPU and would re-read its weights from this cache.
const TERMINAL_STATUSES = new Set(["stopped", "failed"]);

export function repoUsage(
  repoId: string,
  groupNodeIds: ReadonlySet<string>,
  deployments: DeploymentUsage[],
): { inUse: boolean; inUseBy: string[]; lastDeployedAt: string | null } {
  const touching = deployments.filter(
    (d) =>
      (groupNodeIds.has(d.nodeId) || d.clusterNodeIds.some((id) => groupNodeIds.has(id))) &&
      matchRepoToModels(repoId, d.candidates),
  );
  const active = touching.filter((d) => !TERMINAL_STATUSES.has(d.status));
  const lastDeployedAt = touching.length
    ? touching.map((d) => d.createdAt).sort().at(-1)!
    : null;
  return { inUse: active.length > 0, inUseBy: active.map((d) => d.label), lastDeployedAt };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run packages/server/src/hf-cache/grouping.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/hf-cache/grouping.ts packages/server/src/hf-cache/grouping.test.ts
git commit -m "feat(server): pure HF cache grouping + deployment-usage helpers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Hub — store `agent:hf-cache` inventories and broadcast

Thin glue (store + log + SSE), consistent with `agent:recipes`; the logic it feeds is tested in Tasks 5 and 7.

**Files:**
- Modify: `packages/server/src/ws/agent-hub.ts`

- [ ] **Step 1: Add the import and field**

At the top of `packages/server/src/ws/agent-hub.ts`, with the other local imports:

```typescript
import type { HfCacheNodeInventory } from "../hf-cache/grouping.js";
```

In the `AgentHub` class, next to `private ollamaModels: OllamaModelInfo[] = [];` (~line 95):

```typescript
  /** Latest HF-cache inventory per node, pushed by agents on cmd:hf-cache:scan
   *  or after a delete. In-memory only — the filesystem is the source of truth. */
  private hfCacheInventories = new Map<string, HfCacheNodeInventory>();
```

- [ ] **Step 2: Add the message case**

In the agent-message `switch`, directly after the `case "agent:ollama-models": { … }` block (~line 360):

```typescript
          case "agent:hf-cache": {
            if (!nodeId) break;
            const inventory: HfCacheNodeInventory = {
              ...(msg.payload as Omit<HfCacheNodeInventory, "nodeId">),
              nodeId,
            };
            this.hfCacheInventories.set(nodeId, inventory);
            console.log(
              `[hf-cache] inventory from ${nodeId}: ${inventory.repos?.length ?? 0} repos` +
                (inventory.error ? ` (error: ${inventory.error})` : ""),
            );
            sseBroadcast({ type: "hf-cache:inventory", payload: inventory });
            break;
          }
```

- [ ] **Step 3: Add the getter**

Next to `getOllamaModels()` (~line 131):

```typescript
  getHfCacheInventories(): HfCacheNodeInventory[] {
    return [...this.hfCacheInventories.values()];
  }
```

- [ ] **Step 4: Verify compile + existing tests**

Run: `npx vitest run packages/server && npm run build --workspace=packages/server`
Expected: existing tests PASS, build clean.

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/ws/agent-hub.ts
git commit -m "feat(server): store agent:hf-cache inventories per node, broadcast via SSE

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: REST routes `/api/hf-cache` + integration tests

**Files:**
- Create: `packages/server/src/routes/hf-cache.ts`
- Modify: `packages/server/src/index.ts` (imports ~line 19, mounts ~line 63)
- Test: `packages/server/src/__tests__/integration/hf-cache.routes.test.ts`

- [ ] **Step 1: Write the failing integration tests**

Create `packages/server/src/__tests__/integration/hf-cache.routes.test.ts`:

```typescript
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
let hfCacheRouter: typeof import("../../routes/hf-cache.js").hfCacheRouter;

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
  ({ hfCacheRouter } = await import("../../routes/hf-cache.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  // FK-dependency order
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
});

/** Minimal stand-in for AgentHub — only what the router touches. */
function makeHub() {
  const hub = {
    inventories: [] as Record<string, unknown>[],
    sent: [] as { nodeId: string; message: { type: string; payload: Record<string, unknown> } }[],
    online: new Set<string>(),
    getHfCacheInventories() { return hub.inventories; },
    getConnectedNodeIds() { return [...hub.online]; },
    isAgentOnline(id: string) { return hub.online.has(id); },
    sendToAgent(nodeId: string, message: { type: string; payload: Record<string, unknown> }) {
      hub.sent.push({ nodeId, message });
    },
  };
  return hub;
}

function makeApp(hub: ReturnType<typeof makeHub>) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", hub);
  app.use("/api/hf-cache", hfCacheRouter);
  return app;
}

function repo(repoId: string, sizeBytes = 1000) {
  return {
    repoId, kind: "model", sizeBytes, nFiles: 3, revisions: 1,
    lastModified: "2026-06-01T00:00:00.000Z",
  };
}

function inv(nodeId: string, cacheId: string, repos: ReturnType<typeof repo>[], extra: Record<string, unknown> = {}) {
  return {
    nodeId, cacheId, hfHome: "/mnt/tank/models",
    scannedAt: "2026-06-13T00:00:00.000Z",
    totalBytes: repos.reduce((s, r) => s + r.sizeBytes, 0),
    diskFreeBytes: 1_000_000, repos, ...extra,
  };
}

describe("GET /api/hf-cache", () => {
  it("groups shared-cache nodes and resolves node names + connectivity", async () => {
    const n1 = await prisma.node.create({ data: { name: "spark-1" } });
    const n2 = await prisma.node.create({ data: { name: "spark-2" } });
    const hub = makeHub();
    hub.inventories = [inv(n1.id, "shared", [repo("org/alpha")]), inv(n2.id, "shared", [repo("org/alpha")])];
    hub.online.add(n1.id);

    const res = await request(makeApp(hub)).get("/api/hf-cache");
    expect(res.status).toBe(200);
    expect(res.body.caches).toHaveLength(1);
    const cache = res.body.caches[0];
    expect(cache.cacheId).toBe("shared");
    expect(cache.nodes.map((n: { name: string }) => n.name).sort()).toEqual(["spark-1", "spark-2"]);
    expect(cache.nodes.find((n: { name: string }) => n.name === "spark-1").connected).toBe(true);
    expect(cache.nodes.find((n: { name: string }) => n.name === "spark-2").connected).toBe(false);
  });

  it("keeps per-node caches separate", async () => {
    const n1 = await prisma.node.create({ data: { name: "spark-1" } });
    const n2 = await prisma.node.create({ data: { name: "spark-2" } });
    const hub = makeHub();
    hub.inventories = [inv(n1.id, "local-1", [repo("org/alpha")]), inv(n2.id, "local-2", [repo("org/beta")])];

    const res = await request(makeApp(hub)).get("/api/hf-cache");
    expect(res.body.caches).toHaveLength(2);
  });

  it("enriches repos with inUse and lastDeployedAt from deployments", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const model = await prisma.model.create({ data: { name: "org/alpha", runtime: "vllm" } });
    await prisma.deployment.create({
      data: { nodeId: node.id, modelId: model.id, status: "running", displayName: "alpha-prod" },
    });
    const staleModel = await prisma.model.create({ data: { name: "org/old", runtime: "vllm" } });
    await prisma.deployment.create({
      data: { nodeId: node.id, modelId: staleModel.id, status: "stopped" },
    });

    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("org/alpha"), repo("org/old"), repo("org/never")])];

    const res = await request(makeApp(hub)).get("/api/hf-cache");
    const repos = res.body.caches[0].repos;
    const byId = Object.fromEntries(repos.map((r: { repoId: string }) => [r.repoId, r]));
    expect(byId["org/alpha"].inUse).toBe(true);
    expect(byId["org/alpha"].inUseBy).toEqual(["alpha-prod"]);
    expect(byId["org/old"].inUse).toBe(false);
    expect(byId["org/old"].lastDeployedAt).not.toBeNull();
    expect(byId["org/never"].inUse).toBe(false);
    expect(byId["org/never"].lastDeployedAt).toBeNull();
  });

  it("returns an empty caches list when no agent has reported", async () => {
    const res = await request(makeApp(makeHub())).get("/api/hf-cache");
    expect(res.status).toBe(200);
    expect(res.body.caches).toEqual([]);
  });
});

describe("POST /api/hf-cache/scan", () => {
  it("503 when no agents are connected", async () => {
    const res = await request(makeApp(makeHub())).post("/api/hf-cache/scan");
    expect(res.status).toBe(503);
  });

  it("fans out cmd:hf-cache:scan to every connected agent", async () => {
    const hub = makeHub();
    hub.online.add("n1").add("n2");
    const res = await request(makeApp(hub)).post("/api/hf-cache/scan");
    expect(res.status).toBe(202);
    expect(res.body.requested).toBe(2);
    expect(hub.sent.map((s) => s.message.type)).toEqual(["cmd:hf-cache:scan", "cmd:hf-cache:scan"]);
  });
});

describe("DELETE /api/hf-cache/:cacheId", () => {
  it("400 without a repoId", async () => {
    const hub = makeHub();
    hub.inventories = [inv("n1", "shared", [repo("org/alpha")])];
    const res = await request(makeApp(hub)).delete("/api/hf-cache/shared");
    expect(res.status).toBe(400);
  });

  it("404 for an unknown cache or repo", async () => {
    const hub = makeHub();
    hub.inventories = [inv("n1", "shared", [repo("org/alpha")])];
    const app = makeApp(hub);
    expect((await request(app).delete("/api/hf-cache/nope?repoId=org%2Falpha")).status).toBe(404);
    expect((await request(app).delete("/api/hf-cache/shared?repoId=org%2Fghost")).status).toBe(404);
  });

  it("409 when the repo is in use by an active deployment, and sends nothing", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const model = await prisma.model.create({ data: { name: "org/alpha", runtime: "vllm" } });
    await prisma.deployment.create({
      data: { nodeId: node.id, modelId: model.id, status: "running", displayName: "alpha-prod" },
    });
    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("org/alpha")])];
    hub.online.add(node.id);

    const res = await request(makeApp(hub)).delete("/api/hf-cache/shared?repoId=org%2Falpha");
    expect(res.status).toBe(409);
    expect(res.body.error).toContain("alpha-prod");
    expect(hub.sent).toHaveLength(0);
  });

  it("503 when no agent in the cache group is connected", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("org/alpha")])];
    const res = await request(makeApp(hub)).delete("/api/hf-cache/shared?repoId=org%2Falpha");
    expect(res.status).toBe(503);
  });

  it("202 + sends cmd:hf-cache:delete to a connected group member", async () => {
    const node = await prisma.node.create({ data: { name: "spark-1" } });
    const hub = makeHub();
    hub.inventories = [inv(node.id, "shared", [repo("org/alpha")])];
    hub.online.add(node.id);

    const res = await request(makeApp(hub)).delete("/api/hf-cache/shared?repoId=org%2Falpha&kind=model");
    expect(res.status).toBe(202);
    expect(hub.sent).toHaveLength(1);
    expect(hub.sent[0].nodeId).toBe(node.id);
    expect(hub.sent[0].message).toEqual({
      type: "cmd:hf-cache:delete",
      payload: { repoId: "org/alpha", kind: "model" },
    });
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run packages/server/src/__tests__/integration/hf-cache.routes.test.ts`
Expected: FAIL — `routes/hf-cache.js` not found.

- [ ] **Step 3: Write the router**

Create `packages/server/src/routes/hf-cache.ts`:

```typescript
import { Router } from "express";
import { prisma } from "../prisma.js";
import type { AgentHub } from "../ws/agent-hub.js";
import {
  deploymentModelCandidates, groupInventories, repoUsage, type DeploymentUsage,
} from "../hf-cache/grouping.js";

export const hfCacheRouter = Router();

// NOTE (as-built): this sketch was insufficient. `Model.name` is the recipe
// SLUG for registry-ref vLLM deploys, not the HF repo id, so name-only matching
// would let a running deployment's weights show as deletable (a false-negative
// on the safety guard). The shipped loadDeploymentUsage takes `agentHub` and
// enriches candidates with the recipe catalog's HF id (resolved from
// config.recipeFile via getRecipes()) and FineTuneJob.baseModel. See the
// authoritative design in docs/superpowers/specs/2026-06-12-hf-cache-management-design.md
// ("Matching (the soundness-critical part)") and routes/hf-cache.ts.
/** Map every deployment row (any status) to the shape repoUsage consumes.
 *  All statuses are loaded on purpose: terminal ones contribute
 *  lastDeployedAt, active ones drive the in-use guard. */
async function loadDeploymentUsage(agentHub: AgentHub): Promise<DeploymentUsage[]> {
  const deployments = await prisma.deployment.findMany({
    include: {
      model: { include: { finetuneJob: { select: { baseModel: true } } } },
      clusterNodes: true,
    },
  });
  const recipeHfId = new Map<string, string>();
  for (const r of agentHub.getRecipes()) if (r.model) recipeHfId.set(r.file, r.model);
  return deployments.map((d) => {
    const candidates = deploymentModelCandidates(d.model.name, d.config);
    let recipeFile: string | undefined;
    if (d.config) {
      try { recipeFile = (JSON.parse(d.config) as { recipeFile?: string }).recipeFile; }
      catch { /* malformed config — skip recipe resolution */ }
    }
    if (recipeFile) { const hf = recipeHfId.get(recipeFile); if (hf) candidates.push(hf); }
    if (d.model.finetuneJob?.baseModel) candidates.push(d.model.finetuneJob.baseModel);
    return {
      status: d.status,
      nodeId: d.nodeId,
      createdAt: d.createdAt.toISOString(),
      label: d.displayName ?? d.model.name,
      candidates,
      clusterNodeIds: d.clusterNodes.map((cn) => cn.nodeId),
    };
  });
}

/**
 * @openapi
 * /api/hf-cache:
 *   get:
 *     tags: [HF Cache]
 *     summary: List Hugging Face cache contents per cache group
 *     description: >
 *       Returns the latest HF_HOME inventories pushed by agents, grouped by
 *       cache identity (a `.dgx-cache-id` marker file at the HF_HOME root —
 *       nodes sharing the cache over NFS report the same id and collapse into
 *       one group; NFS-less nodes each form their own). Each cached repo is
 *       enriched with `inUse` (an active deployment on the group's nodes
 *       references it — deletion is blocked) and `lastDeployedAt` (newest
 *       matching deployment of any status — the staleness signal). Empty
 *       until agents have scanned; trigger POST /api/hf-cache/scan.
 *     responses:
 *       '200':
 *         description: '{ caches: [{ cacheId, nodes, hfHome, scannedAt, totalBytes, diskFreeBytes, error?, repos }] }'
 */
hfCacheRouter.get("/", async (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  const groups = groupInventories(agentHub.getHfCacheInventories());
  const usage = await loadDeploymentUsage();
  const nodes = await prisma.node.findMany({ select: { id: true, name: true } });
  const nameById = new Map(nodes.map((n) => [n.id, n.name]));

  const caches = groups.map((group) => {
    const groupNodeIds = new Set(group.nodeIds);
    return {
      cacheId: group.cacheId,
      nodes: group.nodeIds.map((id) => ({
        nodeId: id,
        name: nameById.get(id) ?? id,
        connected: agentHub.isAgentOnline(id),
      })),
      hfHome: group.newest.hfHome,
      scannedAt: group.newest.scannedAt,
      totalBytes: group.newest.totalBytes,
      diskFreeBytes: group.newest.diskFreeBytes,
      error: group.newest.error,
      repos: group.newest.repos.map((r) => ({
        ...r,
        ...repoUsage(r.repoId, groupNodeIds, usage),
      })),
    };
  });
  res.json({ caches });
});

/**
 * @openapi
 * /api/hf-cache/scan:
 *   post:
 *     tags: [HF Cache]
 *     summary: Ask every connected agent to rescan its HF cache
 *     description: >
 *       Fans out `cmd:hf-cache:scan` to all connected agents. Each agent walks
 *       its HF_HOME and pushes a fresh inventory, which arrives asynchronously
 *       via the `hf-cache:inventory` SSE event. Returns 503 when no agents are
 *       connected (fail fast — there is nothing to scan with).
 *     responses:
 *       '202':
 *         description: '{ requested: N } — number of agents the command was sent to'
 *       '503':
 *         description: No agents connected
 */
hfCacheRouter.post("/scan", (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  const nodeIds = agentHub.getConnectedNodeIds();
  if (nodeIds.length === 0) {
    return res.status(503).json({ error: "No agents connected — nothing can scan the cache" });
  }
  for (const nodeId of nodeIds) {
    agentHub.sendToAgent(nodeId, { type: "cmd:hf-cache:scan", payload: {} });
  }
  res.status(202).json({ requested: nodeIds.length });
});

/**
 * @openapi
 * /api/hf-cache/{cacheId}:
 *   delete:
 *     tags: [HF Cache]
 *     summary: Delete a cached repo from a cache group
 *     description: >
 *       Sends `cmd:hf-cache:delete` to one connected agent in the cache group.
 *       The repo id travels as a query parameter (`?repoId=org%2Fname`)
 *       because URL-encoded slashes in path segments are unreliable across
 *       HTTP stacks. Refused with 409 while any active (non-stopped/failed —
 *       including evicted) deployment on the group's nodes references the
 *       repo. The agent deletes, rescans, and pushes a fresh inventory (SSE
 *       `hf-cache:inventory`).
 *     parameters:
 *       - in: path
 *         name: cacheId
 *         required: true
 *         schema: { type: string }
 *       - in: query
 *         name: repoId
 *         required: true
 *         schema: { type: string }
 *         description: URL-encoded repo id, e.g. org%2Fname
 *       - in: query
 *         name: kind
 *         schema: { type: string, enum: [model, dataset], default: model }
 *     responses:
 *       '202':
 *         description: '{ deleting, kind, nodeId } — delete dispatched to an agent'
 *       '400':
 *         description: Missing repoId
 *       '404':
 *         description: Unknown cacheId or repo not in this cache
 *       '409':
 *         description: Repo is in use by an active deployment
 *       '503':
 *         description: No connected agent can reach this cache
 */
hfCacheRouter.delete("/:cacheId", async (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  const { cacheId } = req.params;
  const repoId = typeof req.query.repoId === "string" ? req.query.repoId : "";
  const kind = req.query.kind === "dataset" ? "dataset" : "model";
  if (!repoId) return res.status(400).json({ error: "repoId query parameter is required" });

  const group = groupInventories(agentHub.getHfCacheInventories())
    .find((g) => g.cacheId === cacheId);
  if (!group) return res.status(404).json({ error: `Unknown cache: ${cacheId}` });

  const repo = group.newest.repos.find((r) => r.repoId === repoId && r.kind === kind);
  if (!repo) return res.status(404).json({ error: `Not in this cache: ${repoId} (${kind})` });

  const usage = await loadDeploymentUsage();
  const { inUse, inUseBy } = repoUsage(repoId, new Set(group.nodeIds), usage);
  if (inUse) {
    return res.status(409).json({
      error: `${repoId} is in use by: ${inUseBy.join(", ")}. Stop those deployments first.`,
      deployments: inUseBy,
    });
  }

  const execNodeId = group.nodeIds.find((id) => agentHub.isAgentOnline(id));
  if (!execNodeId) {
    return res.status(503).json({ error: "No connected agent can reach this cache" });
  }

  agentHub.sendToAgent(execNodeId, { type: "cmd:hf-cache:delete", payload: { repoId, kind } });
  res.status(202).json({ deleting: repoId, kind, nodeId: execNodeId });
});
```

- [ ] **Step 4: Mount the router**

In `packages/server/src/index.ts`, after the `benchmarksRouter` import (line 19):

```typescript
import { hfCacheRouter } from "./routes/hf-cache.js";
```

After `app.use("/api/benchmarks", benchmarksRouter);` (line 63):

```typescript
app.use("/api/hf-cache", hfCacheRouter);
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run packages/server/src/__tests__/integration/hf-cache.routes.test.ts`
Expected: PASS (all 11 tests).

Then the full server suite (the OpenAPI spec test must still pass with the new JSDoc):

Run: `npx vitest run packages/server`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/routes/hf-cache.ts packages/server/src/index.ts packages/server/src/__tests__/integration/hf-cache.routes.test.ts
git commit -m "feat(server): /api/hf-cache routes — grouped inventory, scan fan-out, guarded delete

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Amend the spec doc for the two approved deviations

**Files:**
- Modify: `docs/superpowers/specs/2026-06-12-hf-cache-management-design.md`

- [ ] **Step 1: Update the DELETE shape**

In the spec, replace the `DELETE /api/hf-cache/:cacheId/:repoId?kind=model|dataset` bullet so it reads:

```markdown
- `DELETE /api/hf-cache/:cacheId?repoId=<url-encoded>&kind=model|dataset` (`kind`
  defaults to `model`; repoId travels as a query parameter because URL-encoded
  slashes in path segments are unreliable across HTTP stacks) → `409` with
  blocking deployment names if `inUse`; `404` if cacheId or repoId unknown;
  otherwise send `cmd:hf-cache:delete` to any *connected* agent in the group,
  `202`. `503` if no agent in the group is connected.
```

- [ ] **Step 2: Update the repoId validation rule**

In the `deleteCachedRepo` bullet, replace the regex sentence so it reads: repoId must be **1 or 2** segments of `[A-Za-z0-9._-]+` (org-less legacy repos like `gpt2` are valid), with `.` and `..` segments rejected, and the resolved path a strict descendant of `hfHome/hub`.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-06-12-hf-cache-management-design.md
git commit -m "docs: amend HF cache spec — DELETE query param, org-less repo ids

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Dashboard pure helpers (`formatBytes`, `sortRepos`)

**Files:**
- Create: `packages/dashboard/lib/hf-cache.ts`
- Test: `packages/dashboard/lib/hf-cache.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `packages/dashboard/lib/hf-cache.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import { formatBytes, sortRepos, type CacheRepo } from "./hf-cache";

describe("formatBytes", () => {
  it("formats across magnitudes", () => {
    expect(formatBytes(0)).toBe("0.0 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(8_000_000_000)).toBe("7.5 GB");
    expect(formatBytes(140 * 1024 ** 3)).toBe("140 GB");
  });

  it("is defensive about garbage", () => {
    expect(formatBytes(-1)).toBe("—");
    expect(formatBytes(NaN)).toBe("—");
  });
});

function r(over: Partial<CacheRepo>): CacheRepo {
  return {
    repoId: "org/x", kind: "model", sizeBytes: 100, nFiles: 1, revisions: 1,
    lastModified: "2026-06-01T00:00:00.000Z", lastDeployedAt: null,
    inUse: false, inUseBy: [],
    ...over,
  };
}

describe("sortRepos", () => {
  it("size: biggest first", () => {
    const sorted = sortRepos([r({ repoId: "a", sizeBytes: 1 }), r({ repoId: "b", sizeBytes: 9 })], "size");
    expect(sorted.map((x) => x.repoId)).toEqual(["b", "a"]);
  });

  it("lastDeployed: never-deployed first, then oldest deployment first", () => {
    const sorted = sortRepos(
      [
        r({ repoId: "recent", lastDeployedAt: "2026-06-10T00:00:00.000Z" }),
        r({ repoId: "never", lastDeployedAt: null }),
        r({ repoId: "old", lastDeployedAt: "2026-01-01T00:00:00.000Z" }),
      ],
      "lastDeployed",
    );
    expect(sorted.map((x) => x.repoId)).toEqual(["never", "old", "recent"]);
  });

  it("does not mutate its input", () => {
    const input = [r({ sizeBytes: 1 }), r({ sizeBytes: 2 })];
    sortRepos(input, "size");
    expect(input[0].sizeBytes).toBe(1);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run packages/dashboard/lib/hf-cache.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Create `packages/dashboard/lib/hf-cache.ts`:

```typescript
export type SortKey = "size" | "lastDeployed";

export interface CacheRepo {
  repoId: string;
  kind: "model" | "dataset";
  sizeBytes: number;
  nFiles: number;
  revisions: number;
  lastModified: string;
  lastDeployedAt: string | null;
  inUse: boolean;
  inUseBy: string[];
}

export interface CacheNode { nodeId: string; name: string; connected: boolean }

export interface CacheGroup {
  cacheId: string;
  nodes: CacheNode[];
  hfHome: string;
  scannedAt: string;
  totalBytes: number;
  diskFreeBytes: number;
  error?: string;
  repos: CacheRepo[];
}

export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v >= 100 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

/** size: biggest first (what's eating the disk). lastDeployed: most stale
 *  first — never-deployed repos lead, then oldest lastDeployedAt; ties break
 *  by size so the biggest reclaim is always on top. */
export function sortRepos(repos: CacheRepo[], key: SortKey): CacheRepo[] {
  const copy = [...repos];
  if (key === "size") return copy.sort((a, b) => b.sizeBytes - a.sizeBytes);
  return copy.sort((a, b) => {
    if (a.lastDeployedAt === b.lastDeployedAt) return b.sizeBytes - a.sizeBytes;
    if (a.lastDeployedAt === null) return -1;
    if (b.lastDeployedAt === null) return 1;
    return a.lastDeployedAt < b.lastDeployedAt ? -1 : 1;
  });
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run packages/dashboard/lib/hf-cache.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/dashboard/lib/hf-cache.ts packages/dashboard/lib/hf-cache.test.ts
git commit -m "feat(dashboard): HF cache UI helpers — formatBytes, staleness-first sort

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: The Models page + nav entry

There is no component-test infrastructure for pages in this repo (only lib tests) — the page is verified by `next build` type-checking plus the manual check in Task 11.

**Files:**
- Rewrite: `packages/dashboard/app/models/page.tsx`
- Modify: `packages/dashboard/components/top-nav.tsx:8` (the `/models` link is missing today)

- [ ] **Step 1: Add the nav link**

In `packages/dashboard/components/top-nav.tsx`, in the `LINKS` array after the Deployments entry (line 8):

```typescript
  { href: "/models", label: "Models" },
```

- [ ] **Step 2: Rewrite the page**

Replace the entire contents of `packages/dashboard/app/models/page.tsx`:

```tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { apiFetch } from "@/lib/api";
import { useSSE, type SseEvent } from "@/lib/sse";
import {
  formatBytes, sortRepos,
  type CacheGroup, type CacheRepo, type SortKey,
} from "@/lib/hf-cache";

function fmtDate(iso: string | null): string {
  return iso ? new Date(iso).toLocaleString() : "—";
}

export default function ModelsPage() {
  const [caches, setCaches] = useState<CacheGroup[] | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("size");
  const [scanning, setScanning] = useState(false);
  const autoScanned = useRef(false);

  const load = useCallback(async (): Promise<CacheGroup[] | null> => {
    try {
      const res = await apiFetch<{ caches: CacheGroup[] }>("/api/hf-cache");
      setCaches(res.caches);
      return res.caches;
    } catch (err) {
      toast.error("Failed to load cache inventory", { description: String(err) });
      return null;
    }
  }, []);

  const rescan = useCallback(async (silent = false) => {
    setScanning(true);
    try {
      const res = await apiFetch<{ requested: number }>("/api/hf-cache/scan", { method: "POST" });
      if (!silent) toast.info(`Scan requested on ${res.requested} agent(s)`);
    } catch (err) {
      // 503 (no agents) lands here — the empty state below explains the situation
      if (!silent) toast.error("Scan failed", { description: String(err) });
    } finally {
      setScanning(false);
    }
  }, []);

  useEffect(() => {
    (async () => {
      const initial = await load();
      // First visit: the server holds no inventory until agents have scanned.
      if (initial !== null && initial.length === 0 && !autoScanned.current) {
        autoScanned.current = true;
        rescan(true);
      }
    })();
  }, [load, rescan]);

  useSSE(
    useCallback((event: SseEvent) => {
      // Enrichment (inUse/lastDeployedAt) lives server-side — refetch rather
      // than patching the SSE payload in.
      if (event.type === "hf-cache:inventory") load();
    }, [load]),
    load,
  );

  async function deleteRepo(cache: CacheGroup, repo: CacheRepo) {
    const msg =
      `Delete ${repo.repoId} (${formatBytes(repo.sizeBytes)}) from ${cache.hfHome}?\n\n` +
      "The next deployment of this model will re-download it.";
    if (!confirm(msg)) return;
    try {
      await apiFetch(
        `/api/hf-cache/${encodeURIComponent(cache.cacheId)}` +
          `?repoId=${encodeURIComponent(repo.repoId)}&kind=${repo.kind}`,
        { method: "DELETE" },
      );
      toast.success(`Deleting ${repo.repoId}…`, {
        description: "The inventory will refresh when the agent has rescanned.",
      });
    } catch (err) {
      toast.error("Delete failed", { description: String(err) });
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Models</h1>
          <p className="text-gray-400 text-sm mt-1">
            Hugging Face download cache (HF_HOME) — inspect and clean up cached model weights.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm text-gray-400">
            Sort{" "}
            <select
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value as SortKey)}
              className="ml-1 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-gray-200"
            >
              <option value="size">Largest first</option>
              <option value="lastDeployed">Stalest first</option>
            </select>
          </label>
          <button
            onClick={() => rescan()}
            disabled={scanning}
            className="px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-sm font-medium"
          >
            {scanning ? "Scanning…" : "Rescan"}
          </button>
        </div>
      </div>

      {caches === null && <p className="text-gray-400">Loading…</p>}

      {caches !== null && caches.length === 0 && (
        <div className="border border-gray-800 rounded-lg p-8 text-center text-gray-400">
          No cache inventory yet. Agents report their HF cache when scanned — if nodes are
          connected, hit <span className="text-gray-200">Rescan</span>; otherwise connect an
          agent first.
        </div>
      )}

      {caches?.map((cache) => {
        const title =
          cache.nodes.length > 1
            ? `Shared cache — ${cache.nodes.map((n) => n.name).join(", ")}`
            : `Cache on ${cache.nodes[0]?.name ?? cache.cacheId}`;
        return (
          <section key={cache.cacheId} className="mb-8">
            <div className="flex items-baseline justify-between mb-2">
              <h2 className="text-lg font-semibold">{title}</h2>
              <span className="text-sm text-gray-400">
                {formatBytes(cache.totalBytes)} used · {formatBytes(cache.diskFreeBytes)} free ·{" "}
                <span className="font-mono">{cache.hfHome}</span> · scanned {fmtDate(cache.scannedAt)}
              </span>
            </div>

            {cache.error && (
              <div className="mb-3 px-3 py-2 rounded border border-red-800 bg-red-950 text-red-300 text-sm">
                {cache.error}
              </div>
            )}

            {cache.repos.length === 0 && !cache.error && (
              <p className="text-gray-500 text-sm">Cache is empty.</p>
            )}

            {cache.repos.length > 0 && (
              <table className="min-w-full text-sm">
                <thead className="text-left text-gray-400 border-b border-gray-800">
                  <tr>
                    <th className="px-2 py-1 font-medium">Repo</th>
                    <th className="px-2 py-1 font-medium">Kind</th>
                    <th className="px-2 py-1 font-medium text-right">Size</th>
                    <th className="px-2 py-1 font-medium text-right">Revisions</th>
                    <th className="px-2 py-1 font-medium">Downloaded</th>
                    <th className="px-2 py-1 font-medium">Last deployed</th>
                    <th className="px-2 py-1 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {sortRepos(cache.repos, sortKey).map((repo) => (
                    <tr key={`${repo.kind}:${repo.repoId}`} className="border-b border-gray-900">
                      <td className="px-2 py-1.5 font-mono">{repo.repoId}</td>
                      <td className="px-2 py-1.5 text-gray-400">{repo.kind}</td>
                      <td className="px-2 py-1.5 text-right">{formatBytes(repo.sizeBytes)}</td>
                      <td className="px-2 py-1.5 text-right text-gray-400">{repo.revisions}</td>
                      <td className="px-2 py-1.5 text-gray-400">{fmtDate(repo.lastModified)}</td>
                      <td className="px-2 py-1.5 text-gray-400">{fmtDate(repo.lastDeployedAt)}</td>
                      <td className="px-2 py-1.5 text-right">
                        {repo.inUse ? (
                          <span
                            className="inline-block px-2 py-0.5 rounded bg-amber-900 text-amber-300 text-xs"
                            title={`In use by: ${repo.inUseBy.join(", ")} — stop those deployments to delete`}
                          >
                            in use
                          </span>
                        ) : (
                          <button
                            onClick={() => deleteRepo(cache, repo)}
                            className="px-2 py-0.5 rounded bg-red-900 hover:bg-red-800 text-red-200 text-xs"
                          >
                            Delete
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 3: Type-check + full test suite**

Run: `npm run build --workspace=packages/dashboard && npx vitest run packages/dashboard`
Expected: `next build` succeeds (type errors here mean the page's types drifted from `lib/hf-cache.ts` — fix before continuing); lib tests PASS.

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/app/models/page.tsx packages/dashboard/components/top-nav.tsx
git commit -m "feat(dashboard): Models page — HF cache inventory, staleness sort, guarded delete

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Final verification

- [ ] **Step 1: Full test suite**

Run: `npm test`
Expected: every suite green, including the new agent, server, and dashboard tests. Do not claim completion otherwise.

- [ ] **Step 2: Full build**

Run: `npm run build`
Expected: all three packages compile.

- [ ] **Step 3: Manual end-to-end check (requires the real cluster — skip if unavailable, and say so)**

```bash
./scripts/build-agent-bundles.sh && \
  MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build
```

Then: dashboard → Models (nav) → expect an auto-scan, the shared `/mnt/tank/models` cache as ONE group listing all nodes → verify a repo backing a running deployment shows "in use" with no delete button → delete a small stale repo → confirm the row disappears after the SSE refresh and `du -sh /mnt/tank/models/hub` shrank. The dashboard will also flag outdated agents (version bump) — upgrade them so the new commands exist on the nodes.

- [ ] **Step 4: Report**

Summarize what was verified (tests, build, manual steps run or skipped) honestly — per CLAUDE.md, no completion claims without `npm test` output.

---

## Self-review notes (already applied)

- **Spec coverage:** wire protocol (T4), scan/marker/delete (T1–T3), grouping + enrichment + 409/404/503 (T5–T7), SSE (T6), page incl. auto-scan/empty/error states (T10), spec deviations documented (T8), testing tiers (property: T1/T2/T5; integration: T7; unit: T3/T9).
- **Type consistency:** `agent:hf-cache` payload (T4) = `HfCacheNodeInventory` minus `nodeId` (T6 adds it); route response shape (T7) = `CacheGroup` in `lib/hf-cache.ts` (T9); `cmd:hf-cache:delete` payload `{ repoId, kind }` matches the agent case (T4) and the integration assertion (T7).
- **Known accepted gaps:** agent WS dispatch and hub case are untested glue (consistent with the codebase); page has no component tests (no infrastructure for them); `lastModified` is download time, not last-read time (NFS atime is unreliable) — `lastDeployedAt` is the staleness signal.
