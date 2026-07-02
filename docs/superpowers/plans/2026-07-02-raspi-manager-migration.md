# RasPi 5 Manager Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the dgx-manager control plane (server + dashboard) off the co-located DGX head `gx10-01` (192.168.44.36) onto a dedicated Raspberry Pi 5 (192.168.44.14), migrating state and repointing the 5 agents, so a model deploy or node reboot can never take the control plane down.

**Architecture:** This is an **operational migration**, not a code change — the arm64 `docker compose` stack runs unchanged on the Pi (host env + host NFS mount differ). Two small repo artifacts are added: an agent-repoint helper and a DB-migrate helper. Cutover is canary→rollout with the DGX manager kept as a live fallback until the end.

**Tech Stack:** Docker + docker compose (arm64), systemd, NFS (v4, soft mount), SQLite (better-sqlite3), bash.

**Spec:** `docs/superpowers/specs/2026-07-01-raspi-manager-migration-design.md`

## Global Constraints

- Target Pi: **192.168.44.14**, arm64, inference-free. Manager advertises as `MANAGER_ADVERTISE_HOST=192.168.44.14`, `SSH_USER=daniel`, `SHARED_STORAGE_PATH=/mnt/tank`.
- NFS mount on the Pi MUST be **soft**: `soft,timeo=100,retrans=3,_netdev,nofail,x-systemd.automount` against **`192.168.44.22:/tank`** — nfs01 is dual-homed; the Pi has no route to the 100.x storage net, so it mounts nfs01's 44.x address (NOT `192.168.100.101`). Never `hard`.
- The manager container bind-mounts `/mnt/tank`; it MUST be (re)created only **after** the host NFS is mounted, or it captures the empty mountpoint.
- Migrate the `dgx-data` SQLite DB but **prune `MetricSnapshot`** to a ≤3-day window + `VACUUM`.
- Cutover order: **canary one worker (dgx-spark-04) → verify → roll out the other 4 → retire DGX manager.** Rollback at any point = flip the agent's `MANAGER_URL` drop-in back to `.36`.
- Do NOT `disable`/`stop` the DGX manager until the final task; it is the fallback.
- Agent repoint uses a systemd **drop-in** `/etc/systemd/system/dgx-agent.service.d/manager-url.conf` (never edit the generated `dgx-agent.service`).

**Reference facts (verified):**
- Nodes: dgx-spark-01 `192.168.44.36` id `cmno92dip006j36o3h3yo91p7`; dgx-spark-02 `.37` `cmno92lcz006s36o3k3yijvbp`; dgx-spark-03 `.38` `cmno92u96007236o3axqbpskv`; dgx-spark-04 `.39` `cmoo6lc8e00nq36r6zihfjfz7`; aihost01 `.30` `cmqb6l6ws1q8i36o0gqjeut5e`.
- Agent unit: `/etc/systemd/system/dgx-agent.service`, `Environment=MANAGER_URL=ws://192.168.44.36:4000/ws/agent`.
- Compose services: `dgx-manager-server-1` (:4000), `dgx-manager-dashboard-1` (:3000); volumes `dgx-data:/app/data`, `${HOME}/.ssh:/root/.ssh:ro`, `/mnt/tank:/mnt/tank`.
- All commands below run **on the Pi** unless prefixed `[DGX]` (run on `gx10-01`) or `[node]`.

---

### Task 1: Pi base prerequisites (Docker, NFS client, SSH key, repo)

**Files:** none (host setup). Prereq: SSH access `ssh daniel@192.168.44.14` with passwordless sudo.

- [ ] **Step 1: Verify Pi identity + arch**

Run: `ssh daniel@192.168.44.14 'hostname; uname -m; id -un; sudo -n true && echo SUDO_OK'`
Expected: an arm64 (`aarch64`) host, user `daniel`, `SUDO_OK`.

- [ ] **Step 2: Install Docker + compose plugin + NFS client**

```bash
ssh daniel@192.168.44.14 'sudo apt-get update && \
  sudo apt-get install -y ca-certificates curl nfs-common && \
  (command -v docker || curl -fsSL https://get.docker.com | sudo sh) && \
  sudo usermod -aG docker daniel'
```

- [ ] **Step 3: Verify docker + compose work (re-login for group)**

Run: `ssh daniel@192.168.44.14 'docker version --format "{{.Server.Version}}" && docker compose version'`
Expected: a server version prints and `Docker Compose version v2.x`. (If group not active yet, prefix `sudo`.)

- [ ] **Step 4: Copy the SSH key the manager uses to reach nodes**

```bash
# The server bind-mounts ~/.ssh; the Pi's daniel@ needs the same key that reaches all nodes.
ssh daniel@192.168.44.14 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'
scp ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub daniel@192.168.44.14:~/.ssh/ 2>/dev/null || \
  scp ~/.ssh/id_rsa ~/.ssh/id_rsa.pub daniel@192.168.44.14:~/.ssh/
```
(Use whichever key `gx10-01` uses for `daniel@192.168.44.3x`.)

- [ ] **Step 5: Verify the Pi can SSH into a node with that key**

Run: `ssh daniel@192.168.44.14 'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=6 daniel@192.168.44.37 hostname'`
Expected: prints `dgx-spark-02` (or the node's hostname) — confirms provisioning/benchmark SSH will work.

- [ ] **Step 6: Clone the repo on the Pi (for compose + build)**

```bash
ssh daniel@192.168.44.14 'git clone https://github.com/kreuzhofer/dgx-manager.git ~/dgx-manager || (cd ~/dgx-manager && git pull)'
```

- [ ] **Step 7: Commit — none (host setup, no repo change).**

---

### Task 2: NFS soft-mount on the Pi

**Files:** none in-repo (host `/etc/fstab`).

**Interfaces:** Produces a mounted, writable `/mnt/tank` on the Pi that the server container will bind-mount in Task 5.

- [ ] **Step 1: Add the soft-mount fstab entry**

```bash
ssh daniel@192.168.44.14 'echo "192.168.44.22:/tank  /mnt/tank  nfs  soft,timeo=100,retrans=3,_netdev,nofail,x-systemd.automount  0  0" | sudo tee -a /etc/fstab'
ssh daniel@192.168.44.14 'sudo mkdir -p /mnt/tank && sudo systemctl daemon-reload'
```

- [ ] **Step 2: Trigger the automount + verify content**

Run: `ssh daniel@192.168.44.14 'ls /mnt/tank | head; mountpoint -q /mnt/tank && echo MOUNTED'`
Expected: lists `datasets models benchmarks …` and prints `MOUNTED`.

- [ ] **Step 3: Verify writability + soft options**

Run: `ssh daniel@192.168.44.14 'touch /mnt/tank/logs/.pi_wt && echo WRITABLE && rm -f /mnt/tank/logs/.pi_wt; mount | grep /mnt/tank'`
Expected: `WRITABLE`, and the mount line shows `soft` and `timeo=100`.

- [ ] **Step 4: Commit — none (host config).**

---

### Task 3: Provision the manager images on the Pi

**Files:** none in-repo. Uses `scripts/build-agent-bundles.sh` + `docker compose build`.

- [ ] **Step 1: Build per-arch agent bundles (required before server build)**

```bash
ssh daniel@192.168.44.14 'cd ~/dgx-manager && docker run --privileged --rm tonistiigi/binfmt --install all && ./scripts/build-agent-bundles.sh'
```

- [ ] **Step 2: Build the server + dashboard images**

```bash
ssh daniel@192.168.44.14 'cd ~/dgx-manager && MANAGER_ADVERTISE_HOST=192.168.44.14 SSH_USER=daniel docker compose build'
```

- [ ] **Step 3: Verify images present**

Run: `ssh daniel@192.168.44.14 'docker images | grep -E "dgx-manager-(server|dashboard)"'`
Expected: both `dgx-manager-server` and `dgx-manager-dashboard` images listed.

- [ ] **Step 4: Commit — none (build artifacts).**

---

### Task 4: DB migrate + prune helper, run the migration

**Files:**
- Create: `scripts/migrate-manager-db.sh`

**Interfaces:** Produces `/tmp/dgx-data-migrated.db` on the Pi, restored into the Pi's `dgx-data` volume, with `MetricSnapshot` pruned. Consumed by Task 5's server boot.

- [ ] **Step 1: Write the migrate helper**

Create `scripts/migrate-manager-db.sh`:
```bash
#!/usr/bin/env bash
# Copy the DGX manager's SQLite DB to the Pi, pruning MetricSnapshot bloat.
# Usage: run ON the Pi.  DGX_HOST defaults to gx10-01.
set -euo pipefail
DGX_HOST="${DGX_HOST:-192.168.44.36}"
KEEP_DAYS="${KEEP_DAYS:-3}"
SRC_CONTAINER=dgx-manager-server-1
DB_IN_CONTAINER=/app/data/dev.db   # DATABASE_URL default file
WORK=/tmp/dgx-data-migrated.db

echo "[1/4] Copy DB from ${DGX_HOST} (server briefly stopped for a clean copy)"
ssh -o StrictHostKeyChecking=no daniel@"${DGX_HOST}" "cd ~/dgx-manager && docker compose stop server && \
  docker cp ${SRC_CONTAINER}:${DB_IN_CONTAINER} /tmp/dgx-src.db && docker compose start server"
scp -o StrictHostKeyChecking=no daniel@"${DGX_HOST}":/tmp/dgx-src.db "${WORK}"

echo "[2/4] Prune MetricSnapshot older than ${KEEP_DAYS} days + VACUUM"
CUTOFF=$(date -u -d "-${KEEP_DAYS} days" +%s000)   # ms epoch; adjust if timestamp is ISO
sqlite3 "${WORK}" "DELETE FROM MetricSnapshot WHERE timestamp < datetime('now','-${KEEP_DAYS} days'); VACUUM;"

echo "[3/4] Row sanity"
sqlite3 "${WORK}" "SELECT 'Node',count(*) FROM Node UNION ALL SELECT 'Deployment',count(*) FROM Deployment UNION ALL SELECT 'LoadBalancerRule',count(*) FROM LoadBalancerRule UNION ALL SELECT 'MetricSnapshot',count(*) FROM MetricSnapshot;"

echo "[4/4] Done -> ${WORK}  (restored into the volume in Task 5 Step 2)"
```
```bash
chmod +x scripts/migrate-manager-db.sh
```

- [ ] **Step 2: Install sqlite3 on the Pi + copy the script over**

```bash
ssh daniel@192.168.44.14 'sudo apt-get install -y sqlite3'
scp scripts/migrate-manager-db.sh daniel@192.168.44.14:~/dgx-manager/scripts/
```

- [ ] **Step 3: Run the migration**

Run: `ssh daniel@192.168.44.14 'cd ~/dgx-manager && ./scripts/migrate-manager-db.sh'`
Expected: the row-sanity output shows `Node 5`, non-zero `Deployment`/`LoadBalancerRule` as applicable, and a **much smaller** `MetricSnapshot` count than the source.

- [ ] **Step 4: Verify the DGX manager came back after the brief stop**

Run: `curl -s -m10 -o /dev/null -w "%{http_code}\n" http://192.168.44.36:4000/api/deployments`
Expected: `200` (the DGX manager, our fallback, is healthy again).

- [ ] **Step 5: Commit the helper**

```bash
git add scripts/migrate-manager-db.sh
git commit -m "feat(migrate): DB copy+prune helper for Pi manager migration"
```

---

### Task 5: Boot the Pi manager on the migrated DB

**Files:** none in-repo (uses the existing compose).

**Interfaces:** Consumes `/tmp/dgx-data-migrated.db` (Task 4). Produces a running Pi manager at `http://192.168.44.14:4000` / `:3000` with the DGX manager still authoritative (agents unchanged).

- [ ] **Step 1: Confirm NFS is mounted (bind-mount ordering guard)**

Run: `ssh daniel@192.168.44.14 'mountpoint -q /mnt/tank && echo MOUNTED || echo NOT_MOUNTED'`
Expected: `MOUNTED`. If not, `sudo mount /mnt/tank` first — the server container must not start before this.

- [ ] **Step 2: Seed the migrated DB into the dgx-data volume, then start**

```bash
ssh daniel@192.168.44.14 'cd ~/dgx-manager && \
  MANAGER_ADVERTISE_HOST=192.168.44.14 SSH_USER=daniel docker compose up -d --no-start server && \
  docker cp /tmp/dgx-data-migrated.db dgx-manager-server-1:/app/data/dev.db && \
  MANAGER_ADVERTISE_HOST=192.168.44.14 SSH_USER=daniel docker compose up -d'
```

- [ ] **Step 3: Verify API + dashboard + migrated state**

Run: `ssh daniel@192.168.44.14 'curl -s -m10 http://localhost:4000/api/deployments >/dev/null && echo API_OK; curl -s -m10 -o /dev/null -w "dash %{http_code}\n" http://localhost:3000'`
Expected: `API_OK` and `dash 200`.
Run: `ssh daniel@192.168.44.14 "curl -s http://localhost:4000/api/nodes | python3 -c 'import sys,json;print(len(json.load(sys.stdin)),\"nodes\")'"`
Expected: `5 nodes` (migrated). They will show **stale** `lastSeen` — expected, since no agent points here yet.

- [ ] **Step 4: Verify the server container sees the LIVE NFS (not the empty mount)**

Run: `ssh daniel@192.168.44.14 'docker exec dgx-manager-server-1 ls /mnt/tank | head'`
Expected: real content (`datasets models …`). If only `datasets logs` shows, NFS was mounted after the container — `docker compose up -d --force-recreate server` and re-check.

- [ ] **Step 5: Commit — none.**

---

### Task 6: Agent repoint helper + canary (dgx-spark-04)

**Files:**
- Create: `scripts/repoint-agent.sh`

**Interfaces:** `repoint-agent.sh <node-ip> <manager-ip>` writes the drop-in + restarts the agent. Consumed by Task 7 for the fleet.

- [ ] **Step 1: Write the repoint helper**

Create `scripts/repoint-agent.sh`:
```bash
#!/usr/bin/env bash
# Repoint a node's dgx-agent at a new manager via a systemd drop-in (reversible).
# Usage: ./repoint-agent.sh <node-ip> <manager-ip>
#        ./repoint-agent.sh <node-ip> --rollback   (remove drop-in -> unit's default .36)
set -euo pipefail
NODE="$1"; TARGET="${2:?manager-ip or --rollback}"
DROPIN=/etc/systemd/system/dgx-agent.service.d/manager-url.conf
if [ "${TARGET}" = "--rollback" ]; then
  ssh -o StrictHostKeyChecking=no daniel@"${NODE}" "sudo rm -f ${DROPIN} && sudo systemctl daemon-reload && sudo systemctl restart dgx-agent"
  echo "rolled back ${NODE} to unit default"
else
  ssh -o StrictHostKeyChecking=no daniel@"${NODE}" "sudo mkdir -p $(dirname ${DROPIN}) && \
    printf '[Service]\nEnvironment=MANAGER_URL=ws://${TARGET}:4000/ws/agent\n' | sudo tee ${DROPIN} >/dev/null && \
    sudo systemctl daemon-reload && sudo systemctl restart dgx-agent && \
    sleep 2 && systemctl show dgx-agent -p Environment | grep -o 'MANAGER_URL=[^ ]*'"
  echo "repointed ${NODE} -> ${TARGET}"
fi
```
```bash
chmod +x scripts/repoint-agent.sh
```

- [ ] **Step 2: Canary — repoint dgx-spark-04 (.39) to the Pi**

Run: `./scripts/repoint-agent.sh 192.168.44.39 192.168.44.14`
Expected: prints `MANAGER_URL=ws://192.168.44.14:4000/ws/agent` and `repointed … -> 192.168.44.14`.

- [ ] **Step 3: Verify the canary registers fresh on the Pi**

Run:
```bash
ssh daniel@192.168.44.14 "curl -s http://localhost:4000/api/nodes | python3 -c '
import sys,json,datetime
d=json.load(sys.stdin); now=datetime.datetime.now(datetime.timezone.utc)
n=[x for x in d if x[\"name\"]==\"dgx-spark-04\"][0]
age=(now-datetime.datetime.fromisoformat(n[\"lastSeen\"].replace(\"Z\",\"+00:00\"))).total_seconds()
print(\"spark-04 lastSeen_age=%ds\"%age)'"
```
Expected: `lastSeen_age` < 30s (the canary is now reporting to the Pi).

- [ ] **Step 4: Verify recipes + a throwaway deploy via the Pi on the canary**

Run (recipes present):
`ssh daniel@192.168.44.14 "curl -s http://localhost:4000/api/recipes | python3 -c 'import sys,json;print(len(json.load(sys.stdin)),\"recipes\")'"`
Expected: non-zero recipe count.
Then deploy a tiny model on spark-04 through the Pi (e.g. an embedding or 1.7B vllm recipe), poll to `running`, then `DELETE ?delete=true`. Expected: reaches `running`, endpoint answers, deletes clean. (Use a small recipe so it's quick; the point is proving deploy+teardown works end-to-end via the Pi.)

- [ ] **Step 5: Commit the helper**

```bash
git add scripts/repoint-agent.sh
git commit -m "feat(migrate): agent repoint helper (systemd drop-in, reversible)"
```

---

### Task 7: Roll out to the remaining agents + retire the DGX manager

**Files:** none new.

- [ ] **Step 1: Repoint the other 4 agents to the Pi**

```bash
for ip in 192.168.44.37 192.168.44.38 192.168.44.30 192.168.44.36; do ./scripts/repoint-agent.sh "$ip" 192.168.44.14; done
```
(`.36` is the head's own agent; it repoints to the Pi like the rest.)

- [ ] **Step 2: Verify all 5 agents fresh on the Pi**

Run:
```bash
ssh daniel@192.168.44.14 "curl -s http://localhost:4000/api/nodes | python3 -c '
import sys,json,datetime
d=json.load(sys.stdin); now=datetime.datetime.now(datetime.timezone.utc); f=0
for x in d:
    try: a=(now-datetime.datetime.fromisoformat(x[\"lastSeen\"].replace(\"Z\",\"+00:00\"))).total_seconds()
    except: a=9999
    if a<30: f+=1
print(f,\"/\",len(d),\"fresh\")'"
```
Expected: `5 / 5 fresh`.

- [ ] **Step 3: Retire the DGX manager**

Run: `ssh daniel@192.168.44.36 'cd ~/dgx-manager && docker compose down'`
Then confirm it's the Pi serving: `curl -s -m10 -o /dev/null -w "%{http_code}\n" http://192.168.44.14:3000` → `200`; `curl -s -m6 -o /dev/null -w "%{http_code}\n" http://192.168.44.36:4000/api/deployments` → `000` (down).

- [ ] **Step 4: Verify full function from the Pi**

- Existing deployments/LB rules present: `curl http://192.168.44.14:4000/api/lb/rules` and `/api/deployments` non-empty as migrated.
- Run a benchmark against a running deployment (exercises SSH + NFS from the Pi) → completes with a score.

- [ ] **Step 5: Commit — none (operational cutover). Optionally update `CLAUDE.md`/README to note the manager now lives at 192.168.44.14.**

---

### Task 8: Failure + rollback drills (validation)

**Files:** none.

- [ ] **Step 1: NFS-outage soft-mount drill**

```bash
ssh daniel@192.168.100.101 'sudo systemctl stop nfs-kernel-server'
```
Run: `ssh daniel@192.168.44.14 'curl -s -m10 -o /dev/null -w "api %{http_code}\n" http://localhost:4000/api/deployments'`
Expected: `api 200` within seconds — the manager stays responsive (soft mount); only NFS-touching calls (datasets/finetune/log-read) error. Then restore: `ssh daniel@192.168.100.101 'sudo systemctl start nfs-kernel-server'` and re-trigger the Pi mount (`ssh daniel@192.168.44.14 'ls /mnt/tank >/dev/null'`).

- [ ] **Step 2: Rollback drill on one node**

Run: `./scripts/repoint-agent.sh 192.168.44.37 --rollback` (points spark-02 back at the unit default `.36`).
Expected: with the DGX manager **down** (retired in Task 7), spark-02 will show offline everywhere — so bring the DGX manager back first if you actually want to validate reclaim, OR run this drill *before* Task 7 Step 3. Document which order you used. Re-point forward afterwards: `./scripts/repoint-agent.sh 192.168.44.37 192.168.44.14`.

- [ ] **Step 3: Commit — none.**

---

## Notes for the implementer

- **DATABASE_URL / DB filename:** the plan assumes SQLite at `/app/data/dev.db` (the repo default `file:./dev.db` under the `dgx-data` volume). If `DATABASE_URL` in `.env`/compose points elsewhere, adjust `DB_IN_CONTAINER` in `migrate-manager-db.sh` and the `docker cp` path in Task 5 Step 2 accordingly — verify with `ssh daniel@192.168.44.36 'docker exec dgx-manager-server-1 sh -lc "echo \$DATABASE_URL; ls -la /app/data"'` first.
- **MetricSnapshot.timestamp type:** the prune uses SQLite `datetime('now','-3 days')`. If `timestamp` is stored as epoch-ms integers rather than ISO text, switch the `DELETE` predicate to `timestamp < (strftime('%s','now','-3 days')*1000)`. Check with `sqlite3 … "SELECT typeof(timestamp), timestamp FROM MetricSnapshot LIMIT 1;"` before running.
- **Do the drills (Task 8) before Task 7 Step 3** if you want a live-fallback rollback test; otherwise the DGX manager is already down.


---

## Execution status (2026-07-02) — COMPLETE

Executed inline. The migration is **done**: the Pi (192.168.44.14) is the sole manager,
all 5 agents report to it, and the DGX manager is retired (head freed, ~112 GiB).

**Per-task outcome:**
- T1 ✅ Pi prereqs (Docker 29.1.3 + compose v5.2.0 installed via the plugin binary since
  `docker-compose-plugin` isn't in the Pi's apt repos; nfs-common; docker group; key
  `id_ed25519_shared` + `~/.ssh/config` copied; repo cloned).
- T2 ✅ NFS soft-mounted — **via `192.168.44.22`, not `100.101`** (Pi has no 100.x route).
- T3 ✅ Images built on the Pi (agent bundles copied from the DGX to skip the slow qemu
  cross-build; server + dashboard built with `MANAGER_ADVERTISE_HOST=192.168.44.14`).
- T4 ✅ DB migrated via online backup (no downtime) + VACUUM: 517 MB → 47 MB.
- T5 ✅ Pi manager booted on the migrated DB; container sees live NFS.
- T6 ✅ Canary = **spark-03** (spark-04 was SSH-congested); registered fresh, 105 recipes.
- T7 ✅ Rolled out to all nodes; **spark-04 needed a hard power-cycle** (its orphaned model
  shard wedged sshd — SSH `kex_exchange_identification` reset); DGX manager retired.
- T8 ⏭️ Drills skipped (extensive real-world validation already done; NFS-outage drill
  deferred to avoid re-disrupting nfs01).

**Open follow-ups:**
1. **GLM-5.2 not currently deployed** (stopped to free nodes for spark-04's reset; the
   deployment record is gone, so this is a *fresh deploy*, not an un-pause). Redeploy via
   the Pi using model `cmr0octnr1xg536k4tbroqlgl` / recipe
   `@community-kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer` → comes up at
   `max_model_len 57344` (56K), gpu-mem 0.88. Consumes a DGX node's GPU — do on demand.
2. ✅ **RESOLVED (2026-07-02)** — the Pi server image was rebuilt at 10:40, *after* the last
   `packages/server/src` commit (`37e7511`, which includes the `f439761` benchmark fix). The
   running server now carries this session's server fixes; image is current with `HEAD`.
3. Optionally harden `nfs01` (it OOM'd at 06:10, killing `rpc.nfsd` — errno 12).

**Verified state (2026-07-02, on the Pi `raspi-dev-01`):** both manager containers up; all
5 agents `online` with fresh `lastSeen` (age 0–4s); one `stopped` deployment record remains
(`qwen3-embedding:8b` on aihost01, Ollama :11434).
