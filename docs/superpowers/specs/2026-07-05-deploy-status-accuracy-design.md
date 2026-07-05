# Deploy-status accuracy — report status transitions independent of the VRAM throttle (design)

**Status:** design / approved to plan
**Date:** 2026-07-05
**Scope:** Agent-side (`packages/agent/src/`). No server/schema/dashboard change.

## Motivation

A multi-node deploy's status stays `"starting"` long after the model actually serves — observed live today: a dgxrun GLM deploy read `"starting"` for 15+ minutes while `/v1/models` was already serving on the head, so the operator can't tell when it's ready.

The readiness logic already exists and is correct on both runners: `checkSparkrunDeployments`/`checkDgxrunDeployments` compute `apiReady` (head's `/metrics` responds 2xx — vLLM binds its HTTP server at serve-time), and `index.ts` computes `deployStatus = sparkrunRunningStatus({containerRunning, apiReady})` (`"running"` iff apiReady, else `"starting"`) for both.

**The bug is the gate around the send.** In both the sparkrun (`index.ts` ~line 351) and dgxrun (~line 405) health blocks:
```ts
if (m.vramUsed > 0) {
  const changed = !prevVram || Math.abs(m.vramUsed - prevVram) > prevVram * 0.01;
  if (changed) sendMsg("agent:deployment:status", { status: deployStatus, ... vramActual: m.vramUsed, ... });
}
```
The status send is gated on a **>1% VRAM change** — a throttle intended for the `vramActual` piggyback. Real sequence: weights load (VRAM climbing → sends fire, all `"starting"` since `apiReady` is still false) → VRAM **stabilizes flat** (~103 GB) → cudagraph capture finishes and `apiReady` flips true — but VRAM is flat, so `changed === false`, so the `starting → running` transition **is never sent**. The status is stuck at `"starting"`. This affects **both runners**.

## Goals

- Report a deploy **status transition** (`starting → running`, and any change) whenever it happens, **independent of the VRAM-change throttle**.
- Keep throttling the `vramActual` piggyback (don't spam VRAM updates).
- Fix **both** runners consistently (shared decision logic, no drift).
- Result: the running GLM dgxrun deploy flips `starting → running` within a health tick of `/metrics` binding.

## Non-goals

No `"loading"` substate / shard-progress (YAGNI — binary `starting/running` is the accuracy fix). No separate `/v1/models` probe (`/metrics` 2xx already is the readiness signal). No server/schema/dashboard change — the server already records `agent:deployment:status`; we only fix *when the agent sends it*.

## Architecture

### 1. Pure decision helper — `packages/agent/src/runtime/deploy-report.ts`
`export function shouldReportStatus(args: { lastStatus: string | undefined; status: string; lastVram: number | undefined; vramUsed: number }): boolean`
- `true` if `status !== lastStatus` (a status transition — **including the first report**, and the `starting→running` flip that the old gate dropped).
- else `true` if `vramUsed > 0` and (`lastVram == null` or `|vramUsed - lastVram| > lastVram * 0.01`) — the VRAM-refresh case.
- else `false`.

Pure, unit-tested. The load-bearing invariant: **a status change reports even when VRAM is flat** (the exact bug), and a flat-status + flat-VRAM tick does **not** report (throttle preserved).

### 2. Wire into both health blocks (`index.ts`)
- Add a module-level `deployLastStatus = new Map<string, string>()` (mirrors the existing `vllmLastVram`).
- In each block, replace the `if (m.vramUsed > 0) { if (changed) … }` gate with:
  ```ts
  const status = deployStatus;
  if (shouldReportStatus({ lastStatus: deployLastStatus.get(id), status, lastVram: vllmLastVram.get(id), vramUsed: m.vramUsed })) {
    deployLastStatus.set(id, status);
    if (m.vramUsed > 0) vllmLastVram.set(id, m.vramUsed);
    sendMsg("agent:deployment:status", {
      deploymentId: id, status,
      port: status === "running" ? statusObj.port : undefined,
      ...(m.vramUsed > 0 ? { vramActual: m.vramUsed } : {}),
    });
  }
  ```
  So a status transition sends even when `m.vramUsed === 0` (VRAM momentarily unreadable), and `vramActual` is only piggybacked when known.
- On terminal/undeploy paths that already `untrackDeployment`, also delete the `deployLastStatus` entry so a re-deploy of the same id starts clean (mirror wherever `vllmLastVram` is cleaned; if it isn't cleaned today, a stale entry is harmless since a fresh deploy's first status differs — but delete it alongside untrack for tidiness).

### 3. Rollout
Agent change → `./scripts/bump-agent-version.sh` + WS roll to the 4 nodes (the advertise-host fix makes WS updates reliable). **Live verification:** the running GLM dgxrun deploy (head `.36`) flips `starting → running` within a tick after the roll (it's already serving, so `apiReady` is true — the fix simply lets that report through).

## Testing (risk: a status-reporting change on the deploy lifecycle)
- **Unit** (`deploy-report.test.ts`): `shouldReportStatus` — status change with flat VRAM → `true` (the bug); status change with VRAM=0 → `true`; same status + VRAM within 1% → `false`; same status + VRAM moved >1% → `true`; first report (`lastStatus` undefined) → `true`; VRAM=0 + same status → `false`.
- **Full suite** `npm test` green; agent typecheck clean; agent version bumped.
- **Manual/live:** after the roll, `GET /api/deployments` shows the GLM deploy flip to `running` and its `port` populated.

## Isolation / boundaries
- `deploy-report.ts` — pure boolean decision, no IO/WS. Testable in isolation; consumed identically by both runner blocks (no drift).
- `index.ts` — the two health blocks add the `deployLastStatus` map + swap the gate. No change to `checkSparkrunDeployments`/`checkDgxrunDeployments` (apiReady already correct) or the server.

## References
- Readiness already computed: `sparkrun-metrics.ts` `sparkrunRunningStatus` + `checkSparkrunDeployments` (`apiReady = res.ok`); `dgxrun/dgxrun-metrics.ts` `checkDgxrunDeployments` (head `apiReady = res.ok`, workers `true`).
- The buggy gate: `index.ts` ~351 (sparkrun) and ~405 (dgxrun).
