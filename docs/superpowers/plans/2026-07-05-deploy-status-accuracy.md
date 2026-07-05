# Deploy-status Accuracy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report a deploy status transition (`starting → running`) whenever it happens, instead of dropping it when VRAM has stabilized — so a serving multi-node deploy no longer reads `"starting"` forever.

**Architecture:** A pure `shouldReportStatus` helper decides when to send `agent:deployment:status`: on any status change (independent of VRAM), or on a >1% VRAM move (the existing throttle, now scoped to the `vramActual` piggyback). Wire it into both health blocks in `index.ts` via a `deployLastStatus` map. Agent-only; the server already records the status.

**Tech Stack:** TypeScript (strict, ESM), Node 22, Vitest.

## Global Constraints

- TypeScript strict + ESM; `.js` import extensions in TS source.
- Agent code change ⇒ `./scripts/bump-agent-version.sh` (once, in the final task).
- `npm test` green + `npx tsc --noEmit -p packages/agent/tsconfig.json` clean before each commit.
- No server / schema / dashboard change. `checkSparkrunDeployments`/`checkDgxrunDeployments` are unchanged (they already compute `apiReady`).
- The existing throttle map is `vllmLastVram = new Map<string, number>()` (`index.ts:119`); the two gates are at `index.ts:~351` (sparkrun) and `~405` (dgxrun), each currently:
  ```ts
  if (m.vramUsed > 0) {
    const prevVram = vllmLastVram.get(status.deploymentId);
    const changed = !prevVram || Math.abs(m.vramUsed - prevVram) > prevVram * 0.01;
    if (changed) {
      vllmLastVram.set(status.deploymentId, m.vramUsed);
      sendMsg("agent:deployment:status", { deploymentId: status.deploymentId, status: deployStatus, port: deployStatus === "running" ? status.port : undefined, vramActual: m.vramUsed });
    }
  }
  ```
- Commit prefix `fix(deploy-status):`.

## File structure

- `packages/agent/src/runtime/deploy-report.ts` — pure `shouldReportStatus`.
- `packages/agent/src/runtime/deploy-report.test.ts` — unit tests.
- `packages/agent/src/index.ts` — `deployLastStatus` map + the two gate swaps + cleanup.

---

### Task 1: Pure helper — `shouldReportStatus`

**Files:**
- Create: `packages/agent/src/runtime/deploy-report.ts`
- Test: `packages/agent/src/runtime/deploy-report.test.ts`

**Interfaces:**
- Produces: `export function shouldReportStatus(args: { lastStatus: string | undefined; status: string; lastVram: number | undefined; vramUsed: number }): boolean`.
  - `true` if `status !== lastStatus` (status transition, including the first report where `lastStatus` is `undefined`).
  - else `true` if `vramUsed > 0` and (`lastVram == null` or `Math.abs(vramUsed - lastVram) > lastVram * 0.01`).
  - else `false`.

- [ ] **Step 1: Write the failing test**
```ts
import { describe, it, expect } from "vitest";
import { shouldReportStatus } from "./deploy-report.js";

describe("shouldReportStatus", () => {
  it("reports a status change even when VRAM is flat (the bug this fixes)", () => {
    expect(shouldReportStatus({ lastStatus: "starting", status: "running", lastVram: 103000, vramUsed: 103000 })).toBe(true);
  });
  it("reports a status change even when VRAM is 0/unreadable", () => {
    expect(shouldReportStatus({ lastStatus: "starting", status: "running", lastVram: 103000, vramUsed: 0 })).toBe(true);
  });
  it("reports the first time (lastStatus undefined)", () => {
    expect(shouldReportStatus({ lastStatus: undefined, status: "starting", lastVram: undefined, vramUsed: 50000 })).toBe(true);
  });
  it("reports when VRAM moved >1% at the same status", () => {
    expect(shouldReportStatus({ lastStatus: "starting", status: "starting", lastVram: 100000, vramUsed: 102000 })).toBe(true);
  });
  it("does NOT report when status and VRAM are both stable (throttle preserved)", () => {
    expect(shouldReportStatus({ lastStatus: "running", status: "running", lastVram: 103000, vramUsed: 103200 })).toBe(false);
  });
  it("does NOT report when VRAM is 0 and status is unchanged", () => {
    expect(shouldReportStatus({ lastStatus: "running", status: "running", lastVram: 103000, vramUsed: 0 })).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**
Run: `npx vitest run packages/agent/src/runtime/deploy-report.test.ts`
Expected: FAIL — `shouldReportStatus` not exported.

- [ ] **Step 3: Write minimal implementation**
```ts
/**
 * Decide whether to (re)send an agent:deployment:status message.
 *
 * A status TRANSITION (e.g. starting -> running) is always reported, even when
 * VRAM has stabilized — the old code gated the send on a >1% VRAM change, which
 * dropped the starting->running flip once weights finished loading and VRAM went
 * flat, leaving deploys stuck at "starting". VRAM changes still trigger a resend
 * (to refresh vramActual), but only when VRAM is actually readable (>0).
 */
export function shouldReportStatus(args: {
  lastStatus: string | undefined;
  status: string;
  lastVram: number | undefined;
  vramUsed: number;
}): boolean {
  if (args.status !== args.lastStatus) return true;
  if (args.vramUsed > 0) {
    if (args.lastVram == null) return true;
    if (Math.abs(args.vramUsed - args.lastVram) > args.lastVram * 0.01) return true;
  }
  return false;
}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `npx vitest run packages/agent/src/runtime/deploy-report.test.ts` — Expected: PASS (6/6).

- [ ] **Step 5: Commit**
```bash
git add packages/agent/src/runtime/deploy-report.ts packages/agent/src/runtime/deploy-report.test.ts
git commit -m "fix(deploy-status): shouldReportStatus — report status transitions past the VRAM throttle"
```

---

### Task 2: Wire into both health blocks + bump

**Files:**
- Modify: `packages/agent/src/index.ts`
- Modify: `packages/agent/package.json` (version bump)

**Interfaces:**
- Consumes: `shouldReportStatus` (Task 1).

- [ ] **Step 1: Import + add the map.** Add near the other runtime imports: `import { shouldReportStatus } from "./runtime/deploy-report.js";`. After `const vllmLastVram = ...` (line 119) add:
```ts
const deployLastStatus = new Map<string, string>(); // deploymentId → last reported deploy status
```

- [ ] **Step 2: Swap the sparkrun gate (~line 351).** Replace the `if (m.vramUsed > 0) { const prevVram … if (changed) { … } }` block in the sparkrun health branch with:
```ts
            const status = deployStatus;
            const id = status0.deploymentId; // NOTE: rename to the block's status var — see Step 4
            if (shouldReportStatus({ lastStatus: deployLastStatus.get(id), status, lastVram: vllmLastVram.get(id), vramUsed: m.vramUsed })) {
              deployLastStatus.set(id, status);
              if (m.vramUsed > 0) vllmLastVram.set(id, m.vramUsed);
              sendMsg("agent:deployment:status", {
                deploymentId: id,
                status,
                port: status === "running" ? status0.port : undefined,
                ...(m.vramUsed > 0 ? { vramActual: m.vramUsed } : {}),
              });
            }
```

- [ ] **Step 3: Swap the dgxrun gate (~line 405).** Apply the SAME replacement in the dgxrun head branch (the `else if (status.containerRunning && isHead)` block), using that block's own status object.

- [ ] **Step 4: Fix the variable names.** The two blocks name the per-deployment status object `status` (sparkrun) / `status` (dgxrun) already — so `status0`/`id` above are placeholders. In each block: the loop variable is `status` (the `VllmStatus`), and `deployStatus = sparkrunRunningStatus(status)` is the string. So write it as: `const s = deployStatus;` for the STRING, use `status.deploymentId` for the id and `status.port` for the port. Concretely each block becomes:
```ts
            const s = deployStatus;
            if (shouldReportStatus({ lastStatus: deployLastStatus.get(status.deploymentId), status: s, lastVram: vllmLastVram.get(status.deploymentId), vramUsed: m.vramUsed })) {
              deployLastStatus.set(status.deploymentId, s);
              if (m.vramUsed > 0) vllmLastVram.set(status.deploymentId, m.vramUsed);
              sendMsg("agent:deployment:status", {
                deploymentId: status.deploymentId,
                status: s,
                port: s === "running" ? status.port : undefined,
                ...(m.vramUsed > 0 ? { vramActual: m.vramUsed } : {}),
              });
            }
```
Keep the surrounding `const m = await collectMetrics();` line (it stays — we still need `m.vramUsed`).

- [ ] **Step 5: Clean up on teardown.** In the failure branches that already call `stopSparkrun(...)`/`stopDgxrun(...)`/`untrackDeployment(status.deploymentId)` (sparkrun ~line 340-347, dgxrun failure branch), add `deployLastStatus.delete(status.deploymentId);` alongside so a re-deploy of the same id starts clean. (Mirror it next to each teardown call in those two branches.)

- [ ] **Step 6: Bump + verify.** Run `./scripts/bump-agent-version.sh`. Then `npx tsc --noEmit -p packages/agent/tsconfig.json` (clean) and `npm test` (green — no existing test should break; the health-loop behavior change is covered by the new unit test + typecheck).

- [ ] **Step 7: Commit**
```bash
git add packages/agent/src/index.ts packages/agent/package.json
git commit -m "fix(deploy-status): report status transitions in both health blocks; bump agent"
```

---

## Self-review (author checklist — completed)

- **Spec coverage:** pure `shouldReportStatus` with the exact truth table + the bug case (T1) ✓; `deployLastStatus` map + both-block gate swap sending on a transition even at VRAM=0, piggyback `vramActual` only when >0, port only when running (T2 Steps 2-4) ✓; cleanup on teardown (T2 Step 5) ✓; agent bump (T2 Step 6) ✓; no server change ✓.
- **Placeholder scan:** Step 2 uses `status0`/`id` placeholders but Step 4 gives the final, correct code (`s` for the string, `status.deploymentId`/`status.port`) — the implementer applies Step 4's version to BOTH blocks. No TBD.
- **Type consistency:** `shouldReportStatus({lastStatus, status, lastVram, vramUsed})` (T1) matches the call in T2; `deployLastStatus: Map<string,string>`, `vllmLastVram: Map<string,number>` consistent.
- **Note for implementer:** read `index.ts:345-420` first; the two blocks are near-identical — apply Step 4's final snippet to each, matching each block's own `status`/`deployStatus` vars (line numbers are anchors).
