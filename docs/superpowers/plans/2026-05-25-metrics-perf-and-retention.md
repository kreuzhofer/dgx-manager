# Metrics Performance and Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the multi-second load time on the Overview / Nodes / Deployments pages, caused by `/api/nodes` doing a 2.4M-row full table scan against an unindexed, unbounded `MetricSnapshot` table.

**Architecture:**
1. Add a composite `(nodeId, timestamp DESC)` index on `MetricSnapshot` so per-node "latest" lookups stop scanning the table.
2. Add a server-side retention job that prunes `MetricSnapshot` rows older than a configurable window (default 7 days), running on boot and every hour.
3. Make `GET /api/nodes` read the latest sample from the in-memory `metricsBuffer` instead of the DB, so the most common endpoint touches zero metric rows.

**Tech Stack:** Prisma 7 + SQLite, Express 5, Vitest + supertest. Existing in-memory `metricsBuffer` (`packages/server/src/metrics-buffer.ts`). Existing integration-test pattern in `packages/server/src/__tests__/integration/`.

---

## Background — what's already true in the repo

- `MetricSnapshot` is defined in `prisma/schema.prisma` with no `@@index`. The only index is `sqlite_autoindex_MetricSnapshot_1` on the PK.
- `prisma.metricSnapshot.create` runs in `packages/server/src/ws/agent-hub.ts:357` on every agent metrics tick (every 5 s per node).
- The only existing `deleteMany` against `MetricSnapshot` outside tests is in `packages/server/src/routes/nodes.ts:271` — fires only when a node is deleted.
- `metricsBuffer` (`packages/server/src/metrics-buffer.ts`) keeps a rolling 720-sample (1 h) window per node, in-memory only. It is `push()`'d on every agent tick in `agent-hub.ts:388`.
- `GET /api/nodes` (`packages/server/src/routes/nodes.ts:26`) currently runs `prisma.node.findMany({ include: { metrics: { orderBy: { timestamp: "desc" }, take: 1 } } })`. With 2.4M rows and no `(nodeId, timestamp)` index, this is ~11 s total.
- Production DB lives in the `dgx-manager_dgx-data` Docker volume at `/app/data/dev.db`. The server container is started via `docker compose up -d server` with `MANAGER_ADVERTISE_HOST` / `SSH_USER` env vars (see CLAUDE.md).
- The agent code is not touched by any of these tasks, so the agent-version bump rule (CLAUDE.md "Agent Version Bumping") does NOT apply.

---

## File Structure

- **Modify** `prisma/schema.prisma` — add `@@index([nodeId, timestamp(sort: Desc)])` to `MetricSnapshot`.
- **Create** `packages/server/src/metric-retention.ts` — exports `pruneMetricsOlderThan(before: Date): Promise<number>` and `startMetricRetention(opts): () => void`.
- **Modify** `packages/server/src/index.ts` — start the retention loop in `main()`.
- **Modify** `packages/server/src/routes/nodes.ts` — `GET /` reads latest from `metricsBuffer` instead of `include: { metrics: … }`.
- **Modify** `.env.example` — document `METRIC_RETENTION_DAYS` (default 7).
- **Create** `packages/server/src/__tests__/integration/metric-snapshot-index.test.ts` — asserts the composite index exists and is used by SQLite's planner.
- **Create** `packages/server/src/__tests__/integration/metric-retention.test.ts` — asserts old rows are deleted and recent rows kept.
- **Create** `packages/server/src/__tests__/integration/nodes.latest-metric-from-buffer.test.ts` — asserts `GET /api/nodes` returns the buffered sample and works when the DB metric table is empty.

---

## Task 1: Add composite index on MetricSnapshot(nodeId, timestamp DESC)

**Files:**
- Modify: `prisma/schema.prisma` (the `MetricSnapshot` model block)
- Test: `packages/server/src/__tests__/integration/metric-snapshot-index.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/__tests__/integration/metric-snapshot-index.test.ts`:

```ts
/**
 * Asserts MetricSnapshot has a composite index that SQLite's planner uses
 * for the "latest sample per nodeId" lookup pattern. Without this index,
 * GET /api/nodes used to take ~10s with 2.4M rows in the table.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;

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
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

describe("MetricSnapshot composite index", () => {
  it("has an index on (nodeId, timestamp) per the schema", async () => {
    const indexes = await prisma.$queryRawUnsafe<{ name: string }[]>(
      "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='MetricSnapshot' AND name NOT LIKE 'sqlite_autoindex_%'"
    );
    expect(indexes.length).toBeGreaterThan(0);
  });

  it("uses the index for ORDER BY timestamp DESC LIMIT 1 per nodeId", async () => {
    const plan = await prisma.$queryRawUnsafe<{ detail: string }[]>(
      "EXPLAIN QUERY PLAN SELECT * FROM MetricSnapshot WHERE nodeId = 'x' ORDER BY timestamp DESC LIMIT 1"
    );
    const planText = plan.map((p) => p.detail).join(" | ");
    // The planner should USE an index (SEARCH ... USING INDEX), not SCAN.
    expect(planText).toMatch(/USING INDEX/i);
    expect(planText).not.toMatch(/^SCAN MetricSnapshot/i);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/metric-snapshot-index.test.ts`
Expected: both `it` cases FAIL — first because there are no non-autoindex indexes, second because the plan reads `SCAN MetricSnapshot`.

- [ ] **Step 3: Add the index in schema.prisma**

In `prisma/schema.prisma`, find the `MetricSnapshot` model. After the last field (`node Node @relation(...)`), add an `@@index` directive. The full block should end like this:

```prisma
model MetricSnapshot {
  // ... existing fields unchanged ...
  timestamp      DateTime @default(now())
  node           Node     @relation(fields: [nodeId], references: [id])

  @@index([nodeId, timestamp(sort: Desc)])
}
```

Then regenerate the Prisma client so the generated TS types pick up the schema (no API-surface change here, but `db push` infers from the schema and we want types in sync):

Run: `npm run db:generate`
Expected: prints `✔ Generated Prisma Client ...` with no errors.

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/metric-snapshot-index.test.ts`
Expected: both cases PASS. The plan output should include something like `SEARCH MetricSnapshot USING INDEX MetricSnapshot_nodeId_timestamp_idx`.

- [ ] **Step 5: Apply the index to the production SQLite DB**

The production DB lives in the `dgx-manager_dgx-data` Docker volume and has 2.4M existing rows. Apply the schema (non-destructive — `db push` without `--force-reset` only adds the index):

Run:
```bash
docker compose exec -T server sh -c 'cd /app && DATABASE_URL=file:/app/data/dev.db npx prisma db push --skip-generate'
```
Expected: `🚀  Your database is now in sync with your Prisma schema.` Should take a few seconds while SQLite builds the index over 2.4M rows.

Verify the endpoint is fast:
Run: `curl -s -o /dev/null -w "%{time_total}s\n" http://localhost:4000/api/nodes`
Expected: well under 100 ms (was ~11 s).

- [ ] **Step 6: Commit**

```bash
git add prisma/schema.prisma packages/server/src/__tests__/integration/metric-snapshot-index.test.ts
git commit -m "server: index MetricSnapshot(nodeId, timestamp) — kills 11s scan on /api/nodes"
```

---

## Task 2: Add metric retention pruning

**Files:**
- Create: `packages/server/src/metric-retention.ts`
- Modify: `packages/server/src/index.ts` (start the loop in `main()`)
- Modify: `.env.example` (document `METRIC_RETENTION_DAYS`)
- Test: `packages/server/src/__tests__/integration/metric-retention.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/__tests__/integration/metric-retention.test.ts`:

```ts
/**
 * Asserts pruneMetricsOlderThan deletes rows strictly older than the cutoff
 * and leaves newer rows untouched. The server starts a periodic prune loop
 * in index.ts on boot — this test covers only the pure helper.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let pruneMetricsOlderThan: typeof import("../../metric-retention.js").pruneMetricsOlderThan;

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
  ({ pruneMetricsOlderThan } = await import("../../metric-retention.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  await prisma.metricSnapshot.deleteMany({});
  await prisma.node.deleteMany({});
  await prisma.node.create({
    data: { id: "node-a", name: "node-a", ipAddress: "10.0.0.1" },
  });
});

describe("pruneMetricsOlderThan", () => {
  it("deletes rows with timestamp < cutoff and keeps rows >= cutoff", async () => {
    const cutoff = new Date("2026-05-20T00:00:00Z");
    await prisma.metricSnapshot.createMany({
      data: [
        { nodeId: "node-a", gpuUtil: 1, vramUsed: 1, timestamp: new Date("2026-05-10T00:00:00Z") },
        { nodeId: "node-a", gpuUtil: 2, vramUsed: 2, timestamp: new Date("2026-05-19T23:59:59Z") },
        { nodeId: "node-a", gpuUtil: 3, vramUsed: 3, timestamp: new Date("2026-05-20T00:00:00Z") },
        { nodeId: "node-a", gpuUtil: 4, vramUsed: 4, timestamp: new Date("2026-05-25T00:00:00Z") },
      ],
    });

    const deleted = await pruneMetricsOlderThan(cutoff);

    expect(deleted).toBe(2);
    const remaining = await prisma.metricSnapshot.findMany({ orderBy: { timestamp: "asc" } });
    expect(remaining.map((r) => r.gpuUtil)).toEqual([3, 4]);
  });

  it("returns 0 when nothing matches", async () => {
    await prisma.metricSnapshot.create({
      data: { nodeId: "node-a", gpuUtil: 1, vramUsed: 1, timestamp: new Date() },
    });
    const deleted = await pruneMetricsOlderThan(new Date("2020-01-01"));
    expect(deleted).toBe(0);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/metric-retention.test.ts`
Expected: FAIL with `Cannot find module '../../metric-retention.js'` (the file doesn't exist yet).

- [ ] **Step 3: Create the retention module**

Create `packages/server/src/metric-retention.ts`:

```ts
import { prisma } from "./prisma.js";

/**
 * Delete MetricSnapshot rows strictly older than `before`. Returns the
 * number of rows deleted. The /api/nodes hot path needs the table to stay
 * bounded — without retention, 4 nodes ticking every 5s grow it by ~69k
 * rows/day, which made the unindexed query a 2.4M-row scan in production.
 */
export async function pruneMetricsOlderThan(before: Date): Promise<number> {
  const { count } = await prisma.metricSnapshot.deleteMany({
    where: { timestamp: { lt: before } },
  });
  return count;
}

interface RetentionOpts {
  retentionDays: number;
  intervalMs: number;
}

/**
 * Start the periodic retention loop. Runs immediately on call, then every
 * `intervalMs`. Returns a stop function. Errors are logged but never thrown
 * — the loop must survive transient DB issues.
 */
export function startMetricRetention(opts: RetentionOpts): () => void {
  const { retentionDays, intervalMs } = opts;

  const run = async () => {
    const before = new Date(Date.now() - retentionDays * 86400 * 1000);
    try {
      const n = await pruneMetricsOlderThan(before);
      if (n > 0) {
        console.log(`[metric-retention] pruned ${n} rows older than ${before.toISOString()}`);
      }
    } catch (err) {
      console.error("[metric-retention] prune failed:", err);
    }
  };

  void run();
  const handle = setInterval(run, intervalMs);
  return () => clearInterval(handle);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/metric-retention.test.ts`
Expected: both `it` cases PASS.

- [ ] **Step 5: Wire the loop into the server boot**

Modify `packages/server/src/index.ts`. Add the import near the top (next to the other local imports):

```ts
import { startMetricRetention } from "./metric-retention.js";
```

Then inside `main()`, after the `benchmarkRun.updateMany` call and before `server.listen(...)`, add:

```ts
  const retentionDays = Number(process.env.METRIC_RETENTION_DAYS ?? 7);
  startMetricRetention({
    retentionDays,
    intervalMs: 60 * 60 * 1000, // 1 hour
  });
```

- [ ] **Step 6: Document the env var**

Modify `.env.example`. Add at the bottom:

```
# How many days of MetricSnapshot rows to retain (default 7). The server
# prunes older rows on boot and every hour.
METRIC_RETENTION_DAYS=7
```

- [ ] **Step 7: Verify nothing else broke**

Run: `npm test`
Expected: full test suite green, including the new retention test.

- [ ] **Step 8: Commit**

```bash
git add packages/server/src/metric-retention.ts \
        packages/server/src/index.ts \
        packages/server/src/__tests__/integration/metric-retention.test.ts \
        .env.example
git commit -m "server: prune MetricSnapshot older than METRIC_RETENTION_DAYS (default 7d)"
```

- [ ] **Step 9: Apply to the running server**

Rebuild and restart so the new retention loop kicks in and starts pruning the 2.4M backlog:

```bash
./scripts/build-agent-bundles.sh && \
  MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build server
```

Then watch the prune happen:

```bash
docker compose logs server -f | grep metric-retention
```
Expected: within ~30 s of boot, `[metric-retention] pruned <N> rows older than <date>`. Confirm DB shrank:
```bash
docker run --rm -v dgx-manager_dgx-data:/data alpine sh -c "apk add --quiet sqlite >/dev/null 2>&1 && sqlite3 /data/dev.db 'SELECT COUNT(*) FROM MetricSnapshot;'"
```

---

## Task 3: GET /api/nodes reads latest sample from metricsBuffer

**Files:**
- Modify: `packages/server/src/routes/nodes.ts` (the `GET /` handler at line 26)
- Test: `packages/server/src/__tests__/integration/nodes.latest-metric-from-buffer.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/__tests__/integration/nodes.latest-metric-from-buffer.test.ts`:

```ts
/**
 * GET /api/nodes should return the latest sample for each node from the
 * in-memory metricsBuffer, not from the MetricSnapshot table. This keeps
 * the endpoint at sub-ms latency regardless of how many DB rows exist,
 * which used to be the source of a ~10s page load.
 */
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
let nodesRouter: typeof import("../../routes/nodes.js").nodesRouter;
let metricsBuffer: typeof import("../../metrics-buffer.js").metricsBuffer;

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
  ({ nodesRouter } = await import("../../routes/nodes.js"));
  ({ metricsBuffer } = await import("../../metrics-buffer.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

function makeApp() {
  const app = express();
  app.use(express.json());
  app.use("/api/nodes", nodesRouter);
  return app;
}

beforeEach(async () => {
  await prisma.metricSnapshot.deleteMany({});
  await prisma.node.deleteMany({});
  metricsBuffer.remove("node-a");
  metricsBuffer.remove("node-b");
});

describe("GET /api/nodes", () => {
  it("returns the latest sample from metricsBuffer for each node", async () => {
    await prisma.node.createMany({
      data: [
        { id: "node-a", name: "node-a", ipAddress: "10.0.0.1" },
        { id: "node-b", name: "node-b", ipAddress: "10.0.0.2" },
      ],
    });
    metricsBuffer.push("node-a", {
      timestamp: 1_700_000_000_000,
      gpuUtil: 42,
      vramUsed: 1234,
      temperature: 55,
      tps: null,
      activeRequests: null,
    });
    // node-b deliberately has no buffered sample.

    const res = await request(makeApp()).get("/api/nodes").expect(200);

    const a = res.body.find((n: { id: string }) => n.id === "node-a");
    const b = res.body.find((n: { id: string }) => n.id === "node-b");
    expect(a.metrics).toHaveLength(1);
    expect(a.metrics[0].gpuUtil).toBe(42);
    expect(a.metrics[0].vramUsed).toBe(1234);
    expect(b.metrics).toEqual([]);
  });

  it("ignores the MetricSnapshot table entirely (returns [] when only DB rows exist)", async () => {
    await prisma.node.create({
      data: { id: "node-a", name: "node-a", ipAddress: "10.0.0.1" },
    });
    // Persisted, but NOT in the buffer.
    await prisma.metricSnapshot.create({
      data: { nodeId: "node-a", gpuUtil: 99, vramUsed: 9999, timestamp: new Date() },
    });

    const res = await request(makeApp()).get("/api/nodes").expect(200);

    const a = res.body.find((n: { id: string }) => n.id === "node-a");
    expect(a.metrics).toEqual([]);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/nodes.latest-metric-from-buffer.test.ts`
Expected: the second case FAILS — current code returns the DB row, so `a.metrics` is `[{ gpuUtil: 99, … }]` instead of `[]`. The first case may also fail because Prisma's `include` returns DB rows, not buffered ones.

- [ ] **Step 3: Update the GET / handler**

Modify `packages/server/src/routes/nodes.ts`. Replace the existing `GET /` handler block (currently around lines 26–34):

```ts
// GET /api/nodes
nodesRouter.get("/", async (_req, res) => {
  const nodes = await prisma.node.findMany({
    orderBy: { name: "asc" },
    include: {
      metrics: { orderBy: { timestamp: "desc" }, take: 1 },
    },
  });
  res.json(nodes);
});
```

with this version that reads from the in-memory buffer:

```ts
// GET /api/nodes
//
// Returns each node with `metrics: [latestSample]` (or `metrics: []` if no
// sample has been seen since boot). We read from the in-memory metricsBuffer
// instead of MetricSnapshot — the DB table is unbounded-by-design (retention
// trims it, but it can still hold weeks of rows) and a per-node "latest"
// lookup used to cost ~10s on a cold cache. Live updates flow through SSE
// (`node:metrics`) after the initial paint, so the buffer is the source of
// truth for "now" anyway.
nodesRouter.get("/", async (_req, res) => {
  const nodes = await prisma.node.findMany({ orderBy: { name: "asc" } });
  const enriched = nodes.map((n) => {
    const history = metricsBuffer.getHistory(n.id);
    const latest = history[history.length - 1];
    return {
      ...n,
      metrics: latest
        ? [
            {
              gpuUtil: latest.gpuUtil,
              vramUsed: latest.vramUsed,
              temperature: latest.temperature,
              tps: latest.tps,
              activeRequests: latest.activeRequests,
              timestamp: new Date(latest.timestamp).toISOString(),
            },
          ]
        : [],
    };
  });
  res.json(enriched);
});
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/nodes.latest-metric-from-buffer.test.ts`
Expected: both cases PASS.

- [ ] **Step 5: Run the full suite**

Run: `npm test`
Expected: full suite green. If any other test was relying on `/api/nodes` returning DB-shaped `MetricSnapshot` objects (with `id`, `nodeId`, memory/pressure fields), update those tests to assert on the trimmed shape — the dashboard reads only `gpuUtil`, `vramUsed`, `temperature`, `tps`, `activeRequests`, `timestamp`.

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/routes/nodes.ts \
        packages/server/src/__tests__/integration/nodes.latest-metric-from-buffer.test.ts
git commit -m "server: GET /api/nodes reads latest sample from metricsBuffer (no DB metric scan)"
```

- [ ] **Step 7: Verify in production**

```bash
MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build server
```

Wait ~10 s for at least one agent metric tick, then:
```bash
curl -s -o /dev/null -w "%{time_total}s\n" http://localhost:4000/api/nodes
curl -s http://localhost:4000/api/nodes | jq '.[0] | {id, name, metrics}'
```
Expected: timing < 50 ms; `metrics` is a single-element array with `gpuUtil`, `vramUsed`, etc.

Also open the dashboard at `http://localhost:3000` and confirm Overview / Nodes / Deployments paint immediately.

---

## Final verification

- [ ] All three integration tests pass: `npx vitest run packages/server/src/__tests__/integration/metric-snapshot-index.test.ts packages/server/src/__tests__/integration/metric-retention.test.ts packages/server/src/__tests__/integration/nodes.latest-metric-from-buffer.test.ts`
- [ ] Full suite green: `npm test`
- [ ] `/api/nodes` returns in < 50 ms in production.
- [ ] `docker compose logs server` shows at least one `[metric-retention] pruned …` line after boot.
- [ ] Overview / Nodes / Deployments dashboard pages paint with no perceptible delay.
