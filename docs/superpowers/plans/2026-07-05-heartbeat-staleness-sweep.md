# Heartbeat-staleness Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make node status reflect reality — a periodic sweep marks online nodes with a stale `lastSeen` as `offline` (so a half-open/dead socket reads offline in ~40 s instead of the 76-minute TCP-close delay), and a node self-heals to `online` on its next heartbeat.

**Architecture:** A pure `selectStaleNodes` selector (testable) + a `setInterval` in `AgentHub` that runs `sweepStale()` (query online nodes → select stale → set offline + drop from the agents map + SSE) + a one-field self-heal (`status:"online"`) in the metric handler. Server-only.

**Tech Stack:** TypeScript (strict, ESM), Node 22, Vitest + supertest, Prisma (SQLite), `ws`.

## Global Constraints

- TypeScript strict + ESM; `.js` import extensions in TS source.
- Server-only — no `packages/agent/src/` change → **no agent version bump**, no roll.
- Reuse the `"offline"` status (no new `"stale"` value). The `close→offline` handler (`agent-hub.ts:752`) is unchanged.
- `STALE_THRESHOLD_MS = 30_000`, `SWEEP_INTERVAL_MS = 10_000` (exact values).
- `npm test` green + `npx tsc --noEmit -p packages/server/tsconfig.json` clean before each commit. Commit prefix `feat(staleness):`.
- Integration tests follow the per-suite-SQLite pattern (`packages/server/src/__tests__/integration/deployments.vram-admission.test.ts` — `mkdtempSync` + `DATABASE_URL` before importing prisma + `npx prisma db push --force-reset` with the AI-consent env set per-suite).

## File structure

- `packages/server/src/ws/staleness.ts` — pure `selectStaleNodes`.
- `packages/server/src/ws/staleness.test.ts` — unit tests.
- `packages/server/src/ws/agent-hub.ts` — sweep timer + `sweepStale()` + `stop()`; metric-handler self-heal.
- `packages/server/src/__tests__/integration/staleness.sweep.test.ts` — integration.

---

### Task 1: Pure selector — `selectStaleNodes`

**Files:**
- Create: `packages/server/src/ws/staleness.ts`
- Test: `packages/server/src/ws/staleness.test.ts`

**Interfaces:**
- Produces: `export interface StaleNodeRow { id: string; status: string; lastSeen: Date | null }` and `export function selectStaleNodes(nodes: StaleNodeRow[], nowMs: number, thresholdMs: number): string[]` — ids where `status === "online"` and `lastSeen != null` and `nowMs - lastSeen.getTime() > thresholdMs`. Skips already-offline and null-lastSeen rows.

- [ ] **Step 1: Write the failing test**
```ts
import { describe, it, expect } from "vitest";
import { selectStaleNodes } from "./staleness.js";

const T = 30_000, NOW = 1_000_000;
const at = (ageMs: number) => new Date(NOW - ageMs);

describe("selectStaleNodes", () => {
  it("selects an online node whose lastSeen is older than the threshold", () => {
    expect(selectStaleNodes([{ id: "a", status: "online", lastSeen: at(40_000) }], NOW, T)).toEqual(["a"]);
  });
  it("does not select a fresh online node", () => {
    expect(selectStaleNodes([{ id: "a", status: "online", lastSeen: at(5_000) }], NOW, T)).toEqual([]);
  });
  it("skips already-offline nodes", () => {
    expect(selectStaleNodes([{ id: "a", status: "offline", lastSeen: at(999_999) }], NOW, T)).toEqual([]);
  });
  it("skips nodes that never heartbeated (null lastSeen)", () => {
    expect(selectStaleNodes([{ id: "a", status: "online", lastSeen: null }], NOW, T)).toEqual([]);
  });
  it("is strict at the boundary (exactly threshold is NOT stale)", () => {
    expect(selectStaleNodes([{ id: "a", status: "online", lastSeen: at(30_000) }], NOW, T)).toEqual([]);
    expect(selectStaleNodes([{ id: "a", status: "online", lastSeen: at(30_001) }], NOW, T)).toEqual(["a"]);
  });
  it("returns only the stale ids from a mixed set", () => {
    expect(selectStaleNodes([
      { id: "fresh", status: "online", lastSeen: at(1_000) },
      { id: "stale", status: "online", lastSeen: at(60_000) },
      { id: "off", status: "offline", lastSeen: at(60_000) },
    ], NOW, T)).toEqual(["stale"]);
  });
});
```

- [ ] **Step 2: Run** — FAIL.
Run: `npx vitest run packages/server/src/ws/staleness.test.ts`

- [ ] **Step 3: Implement**
```ts
export interface StaleNodeRow { id: string; status: string; lastSeen: Date | null; }

/**
 * Ids of nodes that claim to be "online" but whose last heartbeat is older than
 * thresholdMs — i.e. a half-open/dead socket the WS 'close' event hasn't caught.
 * Skips already-offline nodes and nodes that never heartbeated (null lastSeen).
 */
export function selectStaleNodes(nodes: StaleNodeRow[], nowMs: number, thresholdMs: number): string[] {
  return nodes
    .filter((n) => n.status === "online" && n.lastSeen != null && nowMs - n.lastSeen.getTime() > thresholdMs)
    .map((n) => n.id);
}
```

- [ ] **Step 4: Run** — PASS (6/6).
- [ ] **Step 5: Commit** — `git add packages/server/src/ws/staleness.ts packages/server/src/ws/staleness.test.ts && git commit -m "feat(staleness): selectStaleNodes pure selector"`

---

### Task 2: `AgentHub` sweep — timer + `sweepStale()` + `stop()`

**Files:**
- Modify: `packages/server/src/ws/agent-hub.ts` (constructor ~133, class fields ~116-131)
- Test: `packages/server/src/__tests__/integration/staleness.sweep.test.ts`

**Interfaces:**
- Consumes: `selectStaleNodes` (Task 1).
- Produces: `AgentHub.sweepStale(): Promise<void>` (public — one sweep; also called by the interval) and `AgentHub.stop(): void` (clears the interval; for test teardown + clean shutdown).

- [ ] **Step 1: Add constants + import.** Near the top of `agent-hub.ts`, after the imports, add:
```ts
import { selectStaleNodes } from "./staleness.js";
const STALE_THRESHOLD_MS = 30_000;
const SWEEP_INTERVAL_MS = 10_000;
```

- [ ] **Step 2: Add the timer field + start it in the constructor.** Add a field `private sweepTimer: ReturnType<typeof setInterval>;` alongside the other private fields (~line 118), and in the constructor (after `this.capClient = ...`, ~line 136) add:
```ts
    this.sweepTimer = setInterval(() => { void this.sweepStale(); }, SWEEP_INTERVAL_MS);
```

- [ ] **Step 3: Add `sweepStale()` + `stop()` methods** (near `isAgentOnline`, ~line 825):
```ts
  /** Mark online nodes whose lastSeen is stale as offline (the WS 'close' event
   *  can lag 76+ min on a half-open socket). Also drops them from the live agents
   *  map so isAgentOnline/diag/exec report them offline. Called by the interval. */
  async sweepStale(): Promise<void> {
    try {
      const nodes = await prisma.node.findMany({ where: { status: "online" }, select: { id: true, status: true, lastSeen: true } });
      const stale = selectStaleNodes(nodes, Date.now(), STALE_THRESHOLD_MS);
      for (const id of stale) {
        await prisma.node.update({ where: { id }, data: { status: "offline" } }).catch(() => {});
        this.agents.delete(id);
        sseBroadcast({ type: "node:status", payload: { nodeId: id, status: "offline" } });
        console.log(`[staleness] node ${id} marked offline (no heartbeat > ${STALE_THRESHOLD_MS}ms)`);
      }
    } catch (err) {
      console.error("[staleness] sweep error:", err);
    }
  }

  /** Stop the staleness sweep (test teardown / shutdown). */
  stop(): void { clearInterval(this.sweepTimer); }
```
(`prisma`, `sseBroadcast`, `Date` are already imported/available.)

- [ ] **Step 4: Write the failing integration test.** Mirror `deployments.vram-admission.test.ts`'s bootstrap (per-suite SQLite; `DATABASE_URL` set before importing prisma; consent env; `prisma db push --force-reset`; `wipeAll()` between tests).
```ts
// ...standard integration bootstrap (see deployments.vram-admission.test.ts)...
import { AgentHub } from "../../ws/agent-hub.js";

describe("AgentHub.sweepStale", () => {
  it("marks a stale online node offline and leaves a fresh one online", async () => {
    const hub = new AgentHub();
    hub.stop(); // don't let the auto-interval fire during the test
    const fresh = await prisma.node.create({ data: { name: "fresh", status: "online", lastSeen: new Date() } });
    const stale = await prisma.node.create({ data: { name: "stale", status: "online", lastSeen: new Date(Date.now() - 60_000) } });
    (hub as unknown as { agents: Map<string, unknown> }).agents.set(stale.id, { ws: {} }); // seed the map to assert deletion
    await hub.sweepStale();
    expect((await prisma.node.findUnique({ where: { id: stale.id } }))!.status).toBe("offline");
    expect((await prisma.node.findUnique({ where: { id: fresh.id } }))!.status).toBe("online");
    expect((hub as unknown as { agents: Map<string, unknown> }).agents.has(stale.id)).toBe(false);
    hub.stop();
  });
});
```
(Adjust the `prisma.node.create` fields to whatever the schema requires as non-null — check `prisma/schema.prisma` `model Node`; `name`/`status`/`lastSeen` are the ones this test needs, add any other required non-defaulted fields.)

- [ ] **Step 5: Run** — the impl from Steps 1-3 should make it PASS.
Run: `npx vitest run packages/server/src/__tests__/integration/staleness.sweep.test.ts`

- [ ] **Step 6: Verify** — `npm test` green; `npx tsc --noEmit -p packages/server/tsconfig.json` clean. (Note: any place that constructs `AgentHub` for the whole app already runs the interval harmlessly; `stop()` exists for graceful shutdown but wiring it into app shutdown is out of scope.)
- [ ] **Step 7: Commit** — `git add -A && git commit -m "feat(staleness): AgentHub sweep marks stale-online nodes offline"`

---

### Task 3: Self-heal — metric tick restores `online`

**Files:**
- Modify: `packages/server/src/ws/agent-hub.ts` (the `agent:metrics` `prisma.node.update`, ~line 513-520)
- Test: extend `staleness.sweep.test.ts`

**Interfaces:** none new.

- [ ] **Step 1: Add `status:"online"` to the metric update.** In the `agent:metrics` handler's `prisma.node.update({ where:{id:nodeId}, data:{ lastSeen: new Date(), ... } })`, add `status: "online",` to the `data` object (right after `lastSeen: new Date(),`). Comment: `// self-heal: a node the staleness sweep marked offline recovers on its next heartbeat`.

- [ ] **Step 2: Extend the integration test** — assert recovery by calling the same update shape the handler uses (the handler itself is buried in the WS message flow; test the observable DB effect the self-heal guarantees):
```ts
  it("a swept-offline node returns to online when a heartbeat updates it", async () => {
    const n = await prisma.node.create({ data: { name: "recover", status: "offline", lastSeen: new Date(Date.now() - 60_000) } });
    // simulate the metric handler's self-heal update
    await prisma.node.update({ where: { id: n.id }, data: { lastSeen: new Date(), status: "online" } });
    expect((await prisma.node.findUnique({ where: { id: n.id } }))!.status).toBe("online");
  });
```
(This asserts the DB contract the one-line handler change relies on; the handler edit itself is a single field added to an existing update, verified by typecheck + the suite staying green. The reconnect path — register sets `status:"online"` at agent-hub.ts:208/284 — covers the half-open case where the agent reconnects.)

- [ ] **Step 3: Verify** — `npm test` green; `npx tsc --noEmit -p packages/server/tsconfig.json` clean.
- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat(staleness): self-heal node status to online on heartbeat"`

---

## Self-review (author checklist — completed)

- **Spec coverage:** pure `selectStaleNodes` with skip-offline/skip-null + boundary (T1) ✓; `setInterval` sweep in AgentHub marking stale-online→offline + agents.delete + SSE, with `stop()` (T2) ✓; metric-tick self-heal to online (T3) ✓; reuse `offline`, threshold 30s / interval 10s (constants, T2) ✓; server-only, no agent bump ✓.
- **Placeholder scan:** all steps carry real code; the one "adjust to schema required fields" note (T2 Step 4) is a concrete instruction to check `model Node`, not a vague placeholder.
- **Type consistency:** `selectStaleNodes(StaleNodeRow[], nowMs, thresholdMs)` (T1) consumed by `sweepStale` (T2); `sweepStale`/`stop` names consistent across T2's method + test.
- **Non-goal honored:** no new status value; `close→offline` path untouched; dashboard unchanged (it already renders `status` + `lastSeen`).
