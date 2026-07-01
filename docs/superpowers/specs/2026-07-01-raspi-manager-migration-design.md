# Migrating the dgx-manager to a Raspberry Pi 5 — Design

**Date:** 2026-07-01
**Status:** Approved (design), pending implementation plan
**Owner:** Daniel Kreuzhofer

## Problem

The dgx-manager control plane (Express server + Next.js dashboard, `docker compose`)
currently runs on **`gx10-01` (192.168.44.36)** — which is also a DGX Spark **inference
node**. On 2026-07-01 this co-location caused a cascading outage:

1. A ~116 GiB model deployment left the head with ~3 GiB free and swapping.
2. Restarting the manager container under that pressure got it **OOM-killed (137)**; it
   could not reclaim memory while the model held it.
3. The memory pressure (and a coincident OOM on the NFS server `nfs01`, whose
   `rpc.nfsd` failed with `errno 12 Cannot allocate memory`) left the **`/mnt/tank` NFS
   mount hung in D-state** on all 4 nodes — every `docker` op touching the manager
   container (which bind-mounts `/mnt/tank`) blocked indefinitely.
4. Recovery required rebooting all 4 nodes and restarting the NFS server.

Root cause: **the manager must not share a host with large models.** A control plane that
orchestrates 100 GiB+ deployments cannot live inside the memory blast radius of those
deployments.

## Goal

Move the manager to a dedicated, inference-free host — the **Raspberry Pi 5 at
192.168.44.14** — so a model deployment (or a node reboot) can never take the control
plane down. Preserve the full feature set and existing state.

Non-goals (this project): hardening `nfs01` itself; changing the deployment/agent
protocol; re-architecting fine-tuning; HA/replicated manager.

## Decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Feature scope on the Pi | **Full feature set**, including fine-tuning + datasets → the Pi **NFS-mounts `/mnt/tank`**. |
| 2 | NFS mount semantics | **Soft mount** (`soft,timeo=100,retrans=3,nofail,_netdev,x-systemd.automount`) so an NFS outage returns errors, not D-state hangs — the failure mode that wedged us. |
| 3 | State (SQLite DB) | **Migrate** the `dgx-data` volume to the Pi, **pruning `MetricSnapshot`** during the copy (bloat that contributed to synchronous better-sqlite3 event-loop stalls). |
| 4 | Cutover | **Canary → rollout**: Pi comes up on the copied DB while the DGX manager stays authoritative; repoint one worker, verify, then the rest; retire the DGX manager. Rollback = repoint agents back. |
| 5 | Manager image | Reuse the existing arm64 images (Pi 5 is arm64) — no code changes required to the server/dashboard for the move itself. |

## Architecture

### A. Target topology

```
Raspberry Pi 5 (192.168.44.14, arm64, inference-free)
  docker compose:  dgx-manager-server (:4000)  +  dgx-manager-dashboard (:3000)
  env:   MANAGER_ADVERTISE_HOST=192.168.44.14   SSH_USER=daniel   SHARED_STORAGE_PATH=/mnt/tank
  mounts: dgx-data:/app/data   ~/.ssh:/root/.ssh:ro   /mnt/tank:/mnt/tank  (soft NFS)

Agents (5): dgx-spark-01..04 + aihost01
  systemd env  MANAGER_URL = ws://192.168.44.14:4000/ws/agent   (was .36)
```

The Pi runs **no** GPU workload, so neither the OOM nor the memory-pressure→hung-NFS
trigger can occur. Because the Pi still mounts NFS (decision 1), decision 2's soft mount
is the safeguard: if `nfs01` dies again, the manager's fine-tune/datasets/log-read paths
return errors instead of freezing the Node event loop.

### B. What the manager actually needs on the Pi (verified against the code)

- **No NFS for core orchestration** — deployments, node management, metrics, load
  balancer, agent-bundle serving all work without `/mnt/tank`. Agent bundles are baked
  into the server image (`packages/server/agent-bundles/`, served by
  `routes/agent-bundle.ts`); node NFS checks are done over SSH by the provisioner.
- **NFS is needed for** datasets (`$SHARED_STORAGE/datasets`, `routes/datasets.ts`),
  fine-tuning (`$SHARED_STORAGE/src/github/spark-vllm-docker` + training outputs,
  `routes/finetune.ts`), and reading deployment logs (`/mnt/tank/logs/deployments/…`).
- **SSH** — the provisioner (`ssh/provisioner.ts`) and benchmarks SSH into nodes; the Pi's
  `~/.ssh` must carry the key that reaches all nodes (`SSH_USER=daniel`).
- **Bind-mount ordering caveat (learned today):** a container that bind-mounts `/mnt/tank`
  must be (re)created *after* the host NFS is mounted, or it captures the empty mountpoint.
  On the Pi, order boot so the `x-systemd.automount` unit is up before compose starts
  (compose `depends_on` can't express this — rely on the mount being present, or a
  `systemd` drop-in that starts compose after `mnt-tank.mount`).

### C. State migration (decision 3)

1. On the DGX manager: `sqlite3` dump of the `dgx-data` DB (`/app/data/…`), OR stop the
   server and copy the volume file directly (cleaner — no WAL races).
2. **Prune** before/after copy: `DELETE FROM MetricSnapshot WHERE timestamp < now()-Nd;`
   (keep a short window, e.g. 1–3 days) + `VACUUM`. Keeps nodes, deployments,
   LoadBalancerRule/Endpoint, FineTuneJob intact.
3. Restore into the Pi's `dgx-data` volume before first Pi-server boot.
4. On boot the server runs its idempotent `db push` + seed — additive, safe against the
   migrated DB.

### D. Cutover — canary → rollout (decision 4)

1. **Prep Pi:** install Docker + compose, NFS client, mount `/mnt/tank` (soft), copy
   `~/.ssh`, load/pull the arm64 images, place the migrated `dgx-data`.
2. **Start Pi manager** (`MANAGER_ADVERTISE_HOST=192.168.44.14`). DGX manager still runs
   and remains authoritative — agents are still pointed at `.36`.
3. **Canary:** on one worker (e.g. dgx-spark-04) set the agent's `MANAGER_URL` →
   `ws://192.168.44.14:4000/ws/agent` and `systemctl restart dgx-agent`. Verify on the Pi:
   node shows online + **fresh metrics** (`vramUsed`), recipes discovered, and a **throwaway
   test deploy** on that node succeeds and tears down.
4. **Rollout:** repoint the remaining 4 agents the same way. Confirm all 5 online + fresh
   on the Pi.
5. **Retire DGX manager:** `docker compose down` on `gx10-01` (frees its RAM for
   inference — a bonus). The dashboard now lives at `http://192.168.44.14:3000`.
6. **Rollback (any step):** set the affected agents' `MANAGER_URL` back to `.36` and
   restart; the DGX manager (still present until step 5) reclaims them.

### E. Agent repoint mechanism

The agent is a systemd service (`dgx-agent`) with `MANAGER_URL` in its unit environment
(set by the provisioner/install script). Repoint = edit the env (drop-in
`/etc/systemd/system/dgx-agent.service.d/*.conf` or the `EnvironmentFile`) → `systemctl
daemon-reload && systemctl restart dgx-agent`. Node identity/token is preserved (the node
re-registers under its existing id), so no re-provisioning is required — only the URL
changes. A helper script parameterized by node IP keeps this consistent across the 5
nodes and supports rollback.

## Risks & mitigations

**Medium risk** — control-plane move; brief per-node blips during repoint; a bad Pi
config could orphan agents.

- **Canary first** (decision 4) — one node validates the Pi before the fleet moves.
- **DGX manager kept as live fallback** until the last step; rollback is a URL flip.
- **Soft NFS** (decision 2) — an `nfs01` outage degrades, doesn't wedge, the Pi.
- **DB copied, not moved** — the DGX DB is untouched until retirement; canary runs on a
  copy, so a migration bug can't corrupt the original.
- **Bind-mount ordering** (§B) — ensure NFS is mounted before the Pi compose starts.
- **Pi resource check** — confirm disk for images + DB + bundles and that arm64 images
  run; RAM is ample for orchestration (no inference).

## Testing / validation

- **Pi prereqs:** `docker compose ps` both up; `/mnt/tank` mounted + writable inside the
  server container (`docker exec … ls /mnt/tank` shows live content, not the empty mount).
- **Canary node:** appears online with fresh `vramUsed`; `GET /api/recipes` non-empty;
  a test `POST /api/deployments` on that node reaches `running` then `DELETE`s cleanly.
- **Full fleet:** all 5 agents online + fresh on the Pi; existing deployments/LB rules
  present (state migrated); a benchmark run works (SSH + NFS paths).
- **NFS-failure drill (optional):** stop `nfs-kernel-server` on `nfs01`; confirm the Pi
  API stays responsive (soft mount) and fine-tune/dataset calls error rather than hang.
- **Rollback drill:** repoint the canary back to `.36`, confirm it re-registers there.

## Out of scope (future)

- Hardening `nfs01` (memory limits on `rpc.nfsd`, monitoring) — separate work.
- HA / replicated manager; moving off SQLite.
- Automating agent repoint via a manager-pushed command (today it's an SSH/systemd edit).
- Excluding the head node from deployment target selection (a related but separate fix).
