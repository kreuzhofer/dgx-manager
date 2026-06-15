# Cross-Node `known_hosts` Auto-Seeding — Design

**Date:** 2026-06-15
**Status:** Approved (design), pending implementation
**Branch:** `feat/cross-node-known-hosts-seed`

## Problem

Multi-node deploys (e.g. tensor-parallel vLLM via sparkrun/Ray) require the **head
node to SSH into worker nodes**. That node→node SSH uses the head's OpenSSH client,
which enforces `StrictHostKeyChecking` against its `known_hosts`. On a fresh or
re-imaged node the worker's host key is unknown, so the connection fails:

```
[3/6] Distributing resources …
Host key verification failed
Pipeline FAILED rc=255
```

This is **not** the controller→node path: the manager's `sshExec`
(`packages/server/src/ssh/executor.ts`) uses the `ssh2` library with **no host-key
verification**, so the controller can always reach a node. Only the node↔node mesh
breaks.

On 2026-06-14 this blocked the gpt-oss-120b TP=2 deploy. It was unblocked by a
**manual** `ssh-keyscan` of every cluster IP into the system-wide
`/etc/ssh/ssh_known_hosts` on all four nodes. That manual seed is ephemeral — it is
lost on node re-image, and a newly onboarded node neither trusts the existing nodes
nor is trusted by them. There is currently nothing in dgx-manager that reproduces it.

A complicating fact: each GB10 node has **two** in-band (IB) fabric IPs (e.g.
`192.168.100.12` **and** `192.168.100.32`) plus its management IP
(`192.168.44.x`). The Prisma schema models only one fast IP (`Node.fastIpAddress`),
so any seeder driven purely off stored columns would miss the secondary IB IP — which
is exactly the address that still failed `rc=255` in the incident.

## Goal

Make the cross-node `known_hosts` mesh **durable and automatic**, so neither a
re-image nor onboarding a new node ever reintroduces the `rc=255` host-key blocker —
without a Prisma schema change and without modifying the on-node agent.

## Non-Goals

- Controller→node host-key verification (the `ssh2` path intentionally skips it; out of scope).
- Distributing SSH **authorized_keys** / private keys (key *trust*, i.e. `known_hosts`, only — auth is already configured cluster-wide).
- Replacing or wrapping `sparkrun setup ssh`; we own the seeding directly.
- Any change under `packages/agent/src/` (so **no agent version bump** is triggered).

## Architecture

One new server-side module, `packages/server/src/ssh/known-hosts.ts`, split
pure-from-IO exactly like `packages/server/src/admission/vram.ts`:

- **Pure helpers** (no IO; property/unit-tested): IP parsing, subnet filtering, remote-script builder.
- **Orchestrator** (`reseedClusterKnownHosts`): Prisma + `sshExec`-coupled; integration-tested with injected deps.

All trigger wiring lives in `packages/server` (provisioner, agent-hub WS handler, a new
route). Nothing touches the agent package or the database schema.

### File structure

| File | Responsibility | Change |
| --- | --- | --- |
| `packages/server/src/ssh/known-hosts.ts` | Pure helpers + `reseedClusterKnownHosts` orchestrator + single-flight/throttle guard | **Create** |
| `packages/server/src/routes/cluster.ts` | `POST /api/cluster/reseed-known-hosts` → runs reseed, returns report | **Create** |
| `packages/server/src/index.ts` | Mount `clusterRouter` at `/api/cluster` | Modify |
| `packages/server/src/ssh/provisioner.ts` | Trigger reseed after a node provisions | Modify (`provisionNode` success path) |
| `packages/server/src/ws/agent-hub.ts` | Trigger debounced reseed when an agent transitions to online | Modify (the two online-setting sites) |
| `packages/dashboard/...` (nodes page) | "Reseed SSH trust" button → POST endpoint, surface report | Modify |
| `packages/server/src/ssh/known-hosts.test.ts` | Property/unit tests for pure helpers | **Create** |
| `packages/server/src/__tests__/integration/cluster.reseed.test.ts` | Orchestrator + endpoint integration tests | **Create** |

## The seeding algorithm

`reseedClusterKnownHosts(nodes, deps)` where `deps = { sshExec, logger, broadcast }`.
Operates on the set of **online** nodes (those with a live agent / reachable). Two phases.

### Phase 1 — Gather (discover the authoritative IP set, live)

For each online node, over `sshExec`:

```sh
ip -o -4 addr show scope global
```

- `parseGlobalIpv4s(output)` → list of that node's global IPv4 addresses (strips `/prefix`).
- `filterToClusterSubnets(ips, nodes)` → keep only IPs inside the /24 of any node's
  stored `ipAddress` or `fastIpAddress`. This drops `docker0`/veth/`172.17.x` noise and
  keeps mgmt `192.168.44.x` + **both** IB `192.168.100.x` addresses.

Union across all online nodes = the **trusted IP set** (the complete set that must be
mutually trusted). A node that is unreachable in this phase is skipped and logged; its
IPs simply won't be in the set this round (it'll be picked up on its own
agent-reconnect trigger).

### Phase 2 — Seed (write trust on every node)

The scan runs **on each node** (not from the manager container, which is not on the IB
fabric and cannot `ssh-keyscan` the `100.x` addresses). For each online node, over
`sshExec`, run `buildKeyscanSeedScript(trustedIps)`:

```sh
# for each ip in trustedIps (all isValidIpv4-checked before interpolation):
sudo ssh-keygen -f /etc/ssh/ssh_known_hosts -R <ip>     # remove stale/rotated key
# then:
ssh-keyscan -T 5 <ip1> <ip2> … | sudo tee -a /etc/ssh/ssh_known_hosts >/dev/null
sudo sort -u /etc/ssh/ssh_known_hosts -o /etc/ssh/ssh_known_hosts
```

The `ssh-keygen -R` pass before the scan means a **re-imaged node's new host key
replaces the old one** rather than both accumulating (handles key rotation cleanly).
Writing to the **system-wide** `/etc/ssh/ssh_known_hosts` covers whatever user
sparkrun's SSH runs as (OpenSSH `GlobalKnownHostsFile`). This is exactly the manual
procedure that proved out on 2026-06-14.

`ssh-keyscan` is trust-on-first-use; acceptable on this fully-trusted private cluster
with passwordless sudo. Noted explicitly as a security assumption (see below).

### Concurrency, throttle, and idempotency

A module-level guard wraps the orchestrator:

- **Single-flight:** if a reseed is in progress, concurrent triggers return/await the
  same in-flight promise rather than launching a second run.
- **Throttle (rate-limiter):** if *any* reseed run completed within the last 5
  minutes, a new **automatic** trigger is skipped (logged as skipped). It is a pure
  rate-limiter on the automatic (agent-reconnect) path — it arms after any completed
  run regardless of per-node outcome, so a persistently-failed/unreachable node can
  never keep it from arming (which would let a flapping node trigger a reseed storm).
  **Deliberate** triggers — the manual endpoint **and provision-complete** — pass
  `force: true` and bypass the throttle entirely.
- Each run reseeds the whole mesh; runs are idempotent (`-R` + scan + `sort -u`), so a
  redundant run is harmless beyond the SSH cost the throttle caps.

The guard holds two pieces of in-memory state: `inFlight: Promise | null` and
`lastSuccessAt: number | null`. Time is read via an injectable `now()` dep so it is
testable without wall-clock.

## Triggers

1. **Provision-complete** — in `provisionNode`'s success path
   (`provisioner.ts`, the `nodes.ts:350` callback site), fire-and-forget
   `reseedClusterKnownHosts(...)`, errors logged not thrown. Onboarding a node
   re-meshes the whole cluster so the new node trusts everyone and vice versa.
2. **Agent (re)connect** — in `agent-hub.ts`, at the two sites where a node is set
   online, schedule a **debounced (~10 s)** reseed. A reconnect burst collapses to one
   run; subject to single-flight + the 5-min throttle. This is the trigger that catches
   a re-imaged node rejoining without re-provisioning.
3. **Manual** — `POST /api/cluster/reseed-known-hosts` returns the per-node report;
   a "Reseed SSH trust" button on the dashboard nodes page calls it. Bypasses the throttle.

## Error handling (fail-fast, observable; CLAUDE.md principle 3)

Best-effort but never silent:

- **Unreachable node** (gather or seed): skipped, logged, run continues. Cannot scan a
  down box; it self-heals on its next agent-reconnect trigger.
- **`ssh-keyscan` empty for an IP:** logged warning, continue (host may be down).
- **`sudo`/write failure on a node:** captured as that node's per-node error; does not
  abort other nodes.
- **Injection guard:** every IP is validated with `isValidIpv4`
  (`packages/server/src/ws/node-ip.ts`) **before** it is interpolated into a remote
  shell command. `buildKeyscanSeedScript` throws on any non-IPv4 input — this is the
  only place node-derived addresses reach an executed command.
- **Report:** the orchestrator returns
  `{ trustedIps: string[], perNode: Array<{ nodeId, host, ipsSeeded, ok, error? }> }`.
- **Observability:** a `cluster:reseed` SSE event broadcasts the report (reuses the
  existing `sseBroadcast`). The manual endpoint returns the report as JSON: **200** when
  ≥1 node was seeded (per-node errors included in the body), **502** only when **zero**
  nodes were seeded.

## Security

- Trust-on-first-use via `ssh-keyscan` is acceptable only because this is a private,
  fully-trusted cluster with passwordless sudo already in place. Documented assumption.
- No secrets touched. No `authorized_keys` / private-key distribution.
- All remote-command IP interpolation is `isValidIpv4`-gated (shell-injection boundary),
  matching the existing hardening in `nodes/power.ts:macCaptureCmd`.
- Writes only `/etc/ssh/ssh_known_hosts` (host-key trust), nothing else.

## Testing (medium-high risk → full rigor)

**Pure helpers** (`known-hosts.test.ts`):
- Property: `parseGlobalIpv4s` over varied `ip -o -4 addr` output → returns exactly the
  global IPv4s, no prefixes.
- Property: `filterToClusterSubnets` invariant — output ⊆ input ∧ every output IP is in
  some node's /24; non-cluster IPs (e.g. `172.17.0.1`) always excluded.
- Unit: `buildKeyscanSeedScript` contains an `-R` and a keyscan token for every IP, in
  order; **throws** on a non-IPv4 element (injection guard).

**Orchestrator** (`cluster.reseed.test.ts`, injected `sshExec` stub):
- Happy path: gather→seed sequencing; report shape; `ipsSeeded` correct.
- Unreachable node skipped (stub rejects for one host) → that node `ok:false`, others seeded.
- Single-flight: two concurrent calls → exactly one underlying gather/seed sequence.
- Throttle: second automatic call within 5 min (injected `now`) is skipped; manual bypasses.

**Endpoint** (supertest, deps via `app.set`):
- Happy path → 200 + report.
- All nodes unreachable → 502 + per-node errors.

`npm test` must be green before claiming done.

## Rollout / reversibility

- Pure additive: a new module, a new route, three trigger call-sites, one dashboard
  button. No schema migration, no agent change, no change to existing deploy logic.
- Reversible by removing the trigger call-sites; the manual endpoint can stay as an
  operator tool.
- The existing manual `/etc/ssh/ssh_known_hosts` seed remains valid in the meantime;
  this feature simply reproduces and maintains it automatically.
