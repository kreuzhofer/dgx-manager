# Agent v2 Phase 2 — Robust Self-Update (design)

**Status:** design / awaiting user review
**Date:** 2026-07-05
**Scope:** Agent-side (`packages/agent/src/`). Server unchanged. **Out of scope:** peer-pull from sibling nodes, container-image transfer over the fabric (separate Phase 2 pieces).

## Motivation

The agent's WS `cmd:update` handler (`index.ts:1237-1277`) runs a chain of **blocking `execSync`** calls — `curl` (120s), **`tar -xzf` (30s timeout)**, swap `/opt/dgx-agent`, `systemctl restart`. Two failure modes, both confirmed on the wire today (memory `agent-cmd-update-spawnsync-wedges-agent`):
1. The chain **blocks the agent's single-threaded event loop** for the whole update, so it stops heartbeating (looks dead) even before anything fails.
2. A step `ETIMEDOUT`s (the 30s tar extract on a slow/loaded disk is the prime suspect — 270 MB) and leaves the agent **wedged** (dead WS, no reconnect) instead of aborting cleanly.

This repeatedly killed node `.36` (the GLM head) and **reproduced even when idle** — kernel clean (not OOM/fork-starvation). And structurally, once `systemctl restart` runs, the process that did the update is **gone**, so nothing can health-check the new agent or roll back a bad update.

**Goal:** agent self-update that never wedges the agent — a slow or failed update always leaves a *working* agent running (old or new), with automatic rollback if the new agent can't reconnect.

## Architecture (Approach A — detached updater)

### 1. Detached updater — `packages/agent/src/updater.ts` (ships in the bundle)
A self-contained Node script (**Node builtins only** — no `/opt/dgx-agent` deps, so it survives the swap) invoked as `node <path> <bundleUrl> <version>`. The agent copies it to `/tmp/dgx-updater-<ts>.js` and spawns it **detached** (see §2), so it **outlives the `dgx-agent` restart** and is the thing that health-checks + rolls back.

Flow (each step logs to `/var/log/dgx-agent-update.log`):
1. **Download** bundle → `/tmp/agent-bundle-<ver>.tar.gz`, generous timeout (10 min) + 2 retries. *Before the swap, any failure = abort, `/opt/dgx-agent` untouched → old agent keeps running.*
2. **Extract** → `/opt/dgx-agent-new` (generous timeout, no 30s cap). **Verify:** `package.json` exists and its `version === <ver>`, and the entrypoint passes `node --check`. Fail verify → abort (old agent untouched).
3. **Preserve** `node-id` into the new dir.
4. **Atomic swap:** `rm -rf /opt/dgx-agent-old`; `mv /opt/dgx-agent /opt/dgx-agent-old`; `mv /opt/dgx-agent-new /opt/dgx-agent` (via `sudo`, as today).
5. **Restart:** `sudo systemctl restart dgx-agent` — the new agent starts.
6. **Health-check:** poll `/run/dgx-agent/connected` (§3) for a fresh mtime AND `systemctl is-active dgx-agent` for up to a window (90s).
7. **Pass** → write `update-result.json {version, outcome:"success"}`; done (keep `/opt/dgx-agent-old` as the next rollback point).
8. **Fail** (new agent didn't reconnect / crash-loops) → **ROLLBACK:** `mv /opt/dgx-agent /opt/dgx-agent-failed-<ver>`; `mv /opt/dgx-agent-old /opt/dgx-agent`; `sudo systemctl restart dgx-agent`; write `update-result.json {outcome:"rolled-back", error}`. The restored old agent reconnects.
9. **A single in-flight lock** `/run/dgx-agent/updating` prevents concurrent updates.

### 2. `cmd:update` handler (rewritten, non-blocking)
Replaces the `execSync` chain with:
```ts
// guard: skip if an update is already in flight (/run/dgx-agent/updating)
copyFileSync(updaterPath, `/tmp/dgx-updater-${ts}.js`);
const child = spawn("node", [`/tmp/dgx-updater-${ts}.js`, bundleUrl, version],
  { detached: true, stdio: "ignore" });
child.unref();
console.log(`[update] launched detached updater for v${version}`);
// handler returns immediately — the agent keeps its event loop + heartbeat
```
No blocking, no self-modification-in-process. If the updater fails at any point, the agent process is untouched (before swap) or rolled back (after restart) — **never wedged**.

### 3. Connected marker + result reporting (small agent change)
- On each successful manager connection (the existing "Connected to manager" path), the agent `mkdir -p /run/dgx-agent` and touches `/run/dgx-agent/connected`. The updater polls this to confirm the new agent came up. (`/run` is tmpfs — recreated on connect; fine.)
- On connect, the agent reads `/run/dgx-agent/update-result.json` if present, sends `agent:update:result { version, outcome, error }` to the manager, then deletes it — so a rolled-back/failed update is **visible** in the dashboard instead of silent (a pain we hit repeatedly). Server records it on the node/deployment log; **server route unchanged** otherwise.

### 4. What ships / deploys
The updater is `updater.js` **in the agent bundle** at a fixed path (e.g. `AGENT_DIR/updater.js`), so **no systemd-unit or install-script change** is needed. Caveat (chicken-and-egg): the *first* rollout of this fix still goes through the **old** `cmd:update` on each node; the robust path is active only for updates *after* this version is installed. `.36` (currently wedged) is recovered by a power cycle + **SSH-direct** roll for this one version; every update after is safe.

## Testing (risk: the self-update path; a bad one bricks node management)
- **Unit** (`updater.test.ts`): pure helpers with injected fs/values — `verifyExtractedBundle(dir, version)` (version match + entry present → ok/reason); `healthCheckPasses(markerMtimeMs, restartMs, windowMs)`; `parseUpdateResult`.
- **Integration** (`updater.orchestration.test.ts`): `runUpdate(args, deps)` with injected `{download, extract, verify, swap, restart, checkConnected, log}` — assert: happy path swaps+restarts+marks success; verify-fail aborts WITHOUT swapping (old agent untouched); health-check-fail triggers rollback (swap back + restart) + `rolled-back` result; download-fail aborts pre-swap.
- **Unit**: `cmd:update` handler is non-blocking — with an injected `spawn`, it returns immediately, `unref`s, and does not `execSync`.
- `npm test` green; agent typecheck clean; agent version bumped.
- **Live (post-merge):** roll the new agent (SSH-direct for `.36` after power cycle); then a subsequent WS update must (a) keep the agent heartbeating throughout, and (b) on an induced failure, roll back and report — verified via the deploy/update logs, SSH-free.

## Isolation / boundaries
- `updater.ts` — self-contained (Node builtins), pure sub-functions unit-tested, orchestration via injected exec. Runs detached from `/tmp`; independent of the `/opt/dgx-agent` it replaces.
- `cmd:update` handler — reduced to "launch detached updater"; non-blocking.
- connect handler — touch marker + report pending result (a few lines).
- No server, systemd-unit, or install-script change.

## References
- Root-cause evidence: memory `agent-cmd-update-spawnsync-wedges-agent`; current handler `index.ts:1237-1277`.
- Related fixed WS-update bug: memory `manager-advertise-host-broke-ws-updates`.
- Phase 1 (diag/exec) is what let us pull the diagnostic journal SSH-free.
