# Heartbeat-staleness sweep — make node status stop lying (design)

**Status:** design / awaiting user review
**Date:** 2026-07-05
**Scope:** Server-only (`packages/server/src/ws/agent-hub.ts`). No agent change, no roll.

## Motivation

A node's `status` flips to `"offline"` **only** on the WebSocket `close` event (`agent-hub.ts:752-761`). A half-open / dead socket does not fire `close` until the TCP timeout — 76+ minutes today for node `.36` — so `status` reads `"online"` while `lastSeen` silently goes stale for over an hour. This "online while dead" lie repeatedly misled diagnosis this session (memory `node-status-online-unreliable`; the operator's "what were you looking at?"). There is no `lastSeen`-based reconciliation today. `isAgentOnline` (`agent-hub.ts:825`) checks `ws.readyState === OPEN`, which stays `OPEN` on a half-open socket — so it's fooled too.

**Goal:** a dead node reads `"offline"` within ~30–45 s, driven by heartbeat age, not by the TCP `close` event.

## Goals

- Periodically mark any node that is `status:"online"` but whose `lastSeen` age exceeds a threshold (30 s) as `"offline"`, remove it from the live agents map, and SSE-broadcast `node:status offline`.
- A node recovers to `"online"` automatically on its next heartbeat (self-heal), so the sweep never leaves a healthy node stuck offline.
- No flapping for healthy nodes (5 s heartbeat vs 30 s threshold).

## Non-goals

No new `"stale"` status value (reuse `"offline"`; the dashboard already shows the `lastSeen` age, so it reads "offline, last seen 40 s ago" with zero consumer changes). No agent/schema change. No change to the existing `close→offline` path (it stays; the sweep is a second, independent trigger).

## Architecture

### 1. Pure selector — `selectStaleNodes`
`export function selectStaleNodes(nodes: { id: string; status: string; lastSeen: Date | null }[], nowMs: number, thresholdMs: number): string[]` — returns the ids of nodes where `status === "online"` and `lastSeen != null` and `nowMs - lastSeen.getTime() > thresholdMs`. Pure, no IO. Skips already-offline nodes (no redundant writes) and nodes with null `lastSeen` (never heartbeated → left to the connect/close path). Unit-tested.

### 2. The sweep — `setInterval` in `AgentHub`
- Constants: `STALE_THRESHOLD_MS = 30_000`, `SWEEP_INTERVAL_MS = 10_000` (worst-case detection ≈ 40 s).
- In the `AgentHub` constructor, start `setInterval(() => this.sweepStale(), SWEEP_INTERVAL_MS)` (store the handle; expose a `stop()` that clears it for test teardown).
- `private async sweepStale()`: `prisma.node.findMany({ where: { status: "online" }, select: { id, status, lastSeen } })` → `selectStaleNodes(...)` → for each stale id: `prisma.node.update({ where:{id}, data:{ status:"offline" } })`, `this.agents.delete(id)` (so `isAgentOnline`/diag/exec correctly report offline despite a half-open `readyState`), and `sseBroadcast({ type:"node:status", payload:{ nodeId:id, status:"offline" } })`. Wrap in try/catch so a DB hiccup never crashes the interval.

### 3. Self-heal on heartbeat
In the `agent:metrics` handler where `lastSeen: new Date()` is written (~line 516), also set `status: "online"` in the same update. So a node the sweep marked offline flips back to online on its next metric tick (same "refresh from the live tick" philosophy as the `vramTotal` self-heal, commits 68b364d/2069d25). Register already sets `status:"online"` (lines 208/284), covering the reconnect path.

### Interaction / correctness
- The `close→offline` handler is unchanged; if it fires after the sweep already marked a node offline, it idempotently re-sets offline — harmless (the sweep only acts on `status:"online"` rows, so no duplicate broadcast storm).
- 30 s threshold vs 5 s heartbeat ⇒ a healthy node never trips; a transient hiccup < 30 s never trips; and the self-heal recovers any node that briefly did — so no flapping.

## Testing (low risk; one pure selector + interval wiring)
- **Unit** (`agent-hub.staleness.test.ts` or a co-located `staleness.test.ts`): `selectStaleNodes` — fresh online → not selected; online + lastSeen older than threshold → selected; already `offline` → skipped; `lastSeen: null` → skipped; exactly-at-threshold boundary.
- **Integration** (server suite, stub-free where possible): drive `sweepStale()` against a per-suite SQLite with one fresh + one stale online node; assert the stale one is set `offline` and the fresh one untouched; assert a subsequent `agent:metrics` for the swept node flips it back to `online`.
- `npm test` green. No agent version bump (server-only).

## Isolation / boundaries
- `selectStaleNodes` — pure, testable, no DB/WS.
- `AgentHub.sweepStale` + the interval — the only new stateful surface; `stop()` for clean test teardown.
- Metric-handler self-heal — one field added to an existing update.
- No dashboard, schema, or agent change.

## References
- The lie: `agent-hub.ts:752` (close-only offline), `:825` (`isAgentOnline` fooled by half-open readyState). Memory `node-status-online-unreliable`, `agent-cmd-update-spawnsync-wedges-agent` (the `.36` case).
- Self-heal precedent: `vramTotal` (commits 68b364d, 2069d25).
