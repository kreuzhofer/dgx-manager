# Agent v2 — Phase 1: Incident-Response Core (design)

**Status:** design / approved to plan
**Date:** 2026-07-05
**Scope:** Phase 1 of the larger "agent v2" vision (management plane over the agent, SSH optional). This spec covers ONLY Phase 1. Phases 2-4 (robust self-update, image transfer over fabric, declarative node provision/restore, dgxrun consolidation) get their own specs later.

## Motivation

The current agent is a telemetry + deploy client, not a management plane. Repeatedly this session, a node's SSH wedged while its agent WebSocket stayed alive and heartbeating — yet the WS was useless for diagnosis, because the agent can neither observe the system deeply nor run anything. Worse, the failure that wedged the head node (`.36`) was **unified-memory pressure starving `fork()`** — which also breaks any management path that shells out. We were blind exactly when we needed sight.

North star for v2: **the manager can fully observe and manage a node — especially a degraded one — without SSH, and without depending on `fork()` for the basics.**

Phase 1 delivers the piece that would have resolved this entire thread: deep, **fork-free** observation + an audited break-glass exec, driven over the agent.

## Goals (Phase 1)

- **`diag.collect`** — a one-shot incident bundle read **in-process from `/proc` and `/sys`** (no `fork`, no shelling to `free`/`ss`/`top`/`dmesg`), so it works when `fork()` is failing.
- **Rich streaming metrics** — extend the existing metric tick with memory (incl. `MemAvailable`), PSI pressure (cpu/mem/io), PID + fd counts, sshd `:22` connection counts by state, and thermals. Fix the `null` memory/PSI we hit. Every field **self-heals** (sent each tick, server updates from the tick — no register-only fields).
- **`exec`** — audited, reason-required break-glass command execution over the WS (streamed output), for when SSH is inconvenient/wedged but the node can still fork.
- **A capability registry** — the extensible foundation (typed request/response over WS with correlation IDs) that Phases 2-4 hang new capabilities on.
- **Server surface** — REST to invoke `diag.collect` / `exec`, an audit store for `exec`, and metric-schema additions.

## Non-goals (Phase 1)

Self-update robustness, image transfer, declarative node provisioning, dgxrun consolidation (later phases). No dashboard UI beyond what's trivial. No mTLS yet (token auth continues; exec is audited — hardening is Phase 2). No arbitrary-shell-as-default posture beyond the audited `exec`.

## Architecture

Evolve the existing agent (`packages/agent/src/`). Two new subsystems + metric enrichment; keep WS/metrics/dgxrun intact.

### 1. Capability registry (`packages/agent/src/caps/`)
A tiny framework: a capability is `{ name, inputSchema, handler(input, ctx) }`. The WS router dispatches an inbound `agent:cap:request { id, name, input }` to the handler and replies `agent:cap:result { id, ok, data | error }`; long/streamed outputs emit `agent:cap:chunk { id, stream, data }` before the terminal result. `id` correlates request↔response; the server times out unanswered requests. This mirrors the existing message-handler switch but is table-driven and unit-testable. Phase-1 registers exactly two capabilities: `diag.collect` and `exec`.

### 2. `/proc` + `/sys` readers (`packages/agent/src/sysinfo/`) — the fork-free core
Pure parsers over fixture strings (the highest-value tests; same pattern as `sparkrun-parse`), fed by thin `readFileSync` callers:
- **memory** — `/proc/meminfo` → MemTotal, MemAvailable, MemFree, Cached, Buffers, Swap{Total,Free}. (Also the correct `vramTotal` source on unified memory — supersedes the `free -m` parse we had to fix.)
- **pressure** — `/proc/pressure/{cpu,memory,io}` → `{ some: {avg10,avg60,avg300,total}, full: {…} }`.
- **load** — `/proc/loadavg` → 1/5/15 + running/total procs.
- **pids/threads** — `/proc/sys/kernel/pid_max`, count of `/proc/[0-9]+` (readdir, no fork).
- **fds** — `/proc/sys/fs/file-nr` → allocated / max.
- **sshd sockets** — parse `/proc/net/tcp` + `tcp6` for local port `:22`, tallied by TCP state (LISTEN / SYN_RECV / ESTABLISHED / …). **This is the money read**: it distinguishes MaxStartups pre-auth pileup (many SYN_RECV/half-open) from fork starvation (few connections but resets), with zero forking.
- **kmsg** — best-effort tail of `/dev/kmsg` for recent OOM / fork / allocation errors (non-blocking open, bounded read).
- **thermals** — `/sys/class/thermal/thermal_zone*/temp`.
- **disk** — `statvfs` on key mounts (`/`, `/mnt/tank`) + `/proc/diskstats` deltas.

### 3. `diag.collect` capability
Assembles the readers above into one structured JSON bundle. GPU details (`nvidia-smi`) are appended **best-effort** — if fork fails, the bundle still returns the full `/proc` picture (which is the point). Deterministic assembly; the readers are unit-tested; the capability itself is thin.

### 4. Streaming metrics enrichment
Extend the existing `agent:metrics` tick payload with the new fields from the `/proc` readers (memory, pressure, pids/fds, sshd-conn tally, thermals). All read in-process. Server stores them and — following the `vramTotal` self-heal (68b364d/2069d25) — refreshes node fields from the tick, never register-only.

### 5. `exec` capability (break-glass)
`exec { cmd, args?, timeoutMs, reason }`. Spawns, streams stdout/stderr as `agent:cap:chunk`, returns `{ code, timedOut }`. Enforced by the agent: **`reason` required** (reject if absent/empty), a hard `timeoutMs` cap, and an audit event (`agent:audit { cap:"exec", cmd, reason, code, ts }`) emitted to the manager. `exec` forks, so it is explicitly NOT the fork-starvation path — `diag.collect` is. Documented as such.

### Server side (`packages/server/`)
- WS: handle `agent:cap:result/chunk` (correlate to pending requests, timeout), and `agent:audit` (persist).
- REST: `POST /api/nodes/:id/diag` → `diag.collect`; `POST /api/nodes/:id/exec` (body `{ cmd, args?, reason, timeoutMs }`, streams/returns output) — audited. Both fail cleanly if the agent is offline.
- Metrics: extend the snapshot/node schema with the new fields (Prisma migration); metric handler stores + self-heals them.
- Audit: an `AuditEvent` table (node, cap, cmd, reason, code, ts, actor).

### Security (Phase 1, pragmatic)
WS stays token-authenticated (existing register-token). `exec` is enabled but **fully audited + reason-required**; the manager records actor/time/cmd/reason. This is a trusted-LAN GPU cluster, so exec is on-by-default-with-audit rather than armed-per-node; mTLS + per-node arming are noted for Phase 2. The agent already runs with `NOPASSWD` sudo, so v2 changes the *interface*, not the privilege.

## Testing (risk-scaled; this touches diagnostics + a remote-exec path)
- **Unit (bulk):** each `/proc`/`/sys` parser against fixture strings — meminfo, all three PSI files, `/proc/net/tcp` `:22`-by-state (the critical one; include a MaxStartups-pileup fixture and a fork-starved fixture), file-nr, loadavg, thermals. Property-test where a clean invariant exists.
- **Unit:** capability-registry dispatch (correlation, unknown-cap, timeout) and `exec` input validation (reason required, timeout cap).
- **Integration:** server round-trips `diag`/`exec` via a stub agentHub (request→result correlation, audit persisted, offline-node handling).
- `npm test` green; agent version bumped (mandatory on agent edits).

## Isolation / boundaries
- `sysinfo/` = pure parsers + thin readers (no WS, no manager) — usable anywhere, fully testable.
- `caps/` = registry + the two handlers — depend on `sysinfo/` and an injected spawn; no direct WS coupling (handlers return data; the router serializes).
- Server capability client + audit store — symmetric, independent of the agent internals.

## Phasing (for context; only Phase 1 here)
1. **This spec** — incident-response core (fork-free diag + rich metrics + audited exec + registry).
2. Robust self-update (atomic + health-check + rollback + peer-pull) + container-image transfer over the fast fabric.
3. Declarative node provision/restore (netplan/fabric/NFS/docker-`default-shm-size`/sudoers) + manager heartbeat-staleness sweep.
4. Fold dgxrun deploy management fully onto the capability registry.

## References
- Session findings: memory `ssh-hammering-wedges-nodes`, `node-status-online-unreliable`, `glm52-shm-and-jit-findings`.
- The `vramTotal` self-heal (commits 68b364d, 2069d25) is the template for "no register-only fields."
- Testability pattern to follow: `sparkrun-args.ts`/`dgxrun-args.ts` pure builders + injected exec.
