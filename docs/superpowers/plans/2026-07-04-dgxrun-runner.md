# dgxrun — our own multi-node vLLM runner (replace sparkrun for mp deploys)

**Status:** spec / follow-up (not yet implemented)
**Date:** 2026-07-04
**Owner:** (unassigned)

## Motivation

The GLM-5.2 investigation (2026-07-04) proved two things end-to-end:

1. **The `ray` executor is broken on our vLLM build** — it dies at init with
   `AttributeError: 'ShmRingBuffer' object has no attribute 'buf'`
   (`ray_executor_v2.py:199`), even with 61 GB `/dev/shm`. So it is *not* a
   shm-size problem; the Ray executor's ShmRingBuffer path is broken here.
2. **The `mp` executor works** — 4-node TP, 85K context @ 0.88, **~23 tok/s**
   with MTP, via a direct `docker run` launcher (`--distributed-executor-backend
   mp` + `--ipc host` + per-node `--nnodes/--node-rank/--master-addr`).

But **sparkrun orchestrates multi-node only via Ray** and never passes mp's
`--nnodes/--node-rank`. So the *only* working config for this model is
inexpressible through sparkrun, and it cannot be deployed via the manager. The
recipe (`recipes/glm-5.2/kreuzhofer/...`) now documents this dead end.

`dgxrun` is our own runner that (a) launches the working **mp multi-node** config
the manager currently can't, and (b) gives us a recipe→launch path **we own**, so
new models are a YAML drop-in rather than a fight with an external tool.

## Goals

- A `dgxrun` runtime in the **agent** that reads the **existing recipe YAML**
  (model, container, env, `command`, defaults, `cluster_only`, `maxoutmem`) and
  orchestrates an **mp multi-node** `docker run` per node — generalizing
  `scratchpad/glm52-mp-launch.sh`.
- **Drop-in with the sparkrun runtime interface** (`packages/agent/src/runtime/
  sparkrun.ts`): `launch* / stop* / isWorkloadRunning / inspect*Container /
  snapshotContainerLogs`, keyed by `deploymentId`, persisted + reconciled on
  reconnect exactly like `kind: "sparkrun"` deployments.
- **Extensible**: adding a model = a new recipe YAML with `runner: dgxrun`; no
  code change. The docker orchestration is model-agnostic.
- **Recipe portability**: reuse the sparkrun recipe schema so recipes stay valid
  for both runners; only the multi-node *mechanism* differs.

## Non-goals

- Not replacing sparkrun for single-node or Ray-working recipes — both runners
  coexist; `runner:` (or `executor: mp` + `cluster_only`) selects dgxrun.
- Not reimplementing Ray, a scheduler, or a new recipe format.
- Not solving the vLLM Ray-executor bug upstream (orthogonal).

## Design

### Where it lives
`packages/agent/src/runtime/dgxrun/` — sibling to `sparkrun.ts`, same exported
lifecycle surface so `index.ts`'s `cmd:deploy`/`cmd:undeploy`/reconcile paths
switch on a `kind` (`"sparkrun" | "dgxrun"`) with minimal branching. Pure argv
building split into `dgxrun-args.ts` (unit-testable, mirrors `sparkrun-args.ts`).

### Orchestration model — **agent-per-node** (preferred)
The agent already runs on *every* node, so orchestrate from the manager, not via
SSH (avoids the SSH-fragility we hit — see memory `ssh-hammering-wedges-nodes`):

- The manager holds the cluster (`clusterNodes`, roles head/worker). For a
  dgxrun deploy it sends `cmd:deploy` to **each** cluster node's agent with:
  `{ deploymentId, recipe, rank, nnodes, masterAddr (head mgmt IP), masterPort,
  tp, headless: rank>0 }`.
- **Each agent launches its own rank's container locally** (`docker run -d`).
  No node SSHes another. Each agent owns its container's lifecycle, logs, and
  teardown — a clean fit for the existing manager→agent model.
- Rendezvous: vLLM's torch TCPStore on `masterAddr:masterPort` forms as ranks
  come up; tolerates the launch skew (glm52-mp-launch.sh already relies on this).

*Alternative (fallback):* head-orchestrated via SSH (a direct port of
glm52-mp-launch.sh). Simpler to lift but reintroduces SSH orchestration; use only
if per-node `cmd:deploy` fan-out proves awkward.

### The docker argv (from `dgxrun-args.ts`)
Built from the recipe, generalizing the launcher's `BASE`/`ENVV`/`SERVE`:
- Container flags: `--network host --ipc host --gpus all --device
  /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1`,
  `-v <weights>:/cache/huggingface`, `--shm-size` (belt-and-suspenders with
  `--ipc host`). **`--ipc host` is the key fix** and is a per-container flag —
  exactly what sparkrun couldn't express.
- Env: the recipe `env:` block verbatim (NCCL/RDMA, GLM52_* Triton, LD_PRELOAD
  nccl, TRITON_CACHE_DIR on NFS, timeouts).
- Command: the recipe `command` template with `{placeholders}` filled, executor
  forced to `mp`, **plus** appended `--nnodes {nnodes} --node-rank {rank}
  --master-addr {masterAddr} --master-port {masterPort}` and `--headless` for
  rank>0.

### Lifecycle surface (mirror `sparkrun.ts`)
| sparkrun.ts | dgxrun equivalent |
|---|---|
| `launchSparkrun(depId, recipeRef, opts)` | `launchDgxrun(depId, recipe, {rank, nnodes, masterAddr, masterPort, tp})` — `docker run -d` this rank |
| `stopSparkrun(depId, target, hosts)` | `stopDgxrun(depId)` — `docker rm -f` the local rank container |
| `isWorkloadRunning(target, hosts)` | `docker inspect` running + (head only) `GET /v1/models` 200 |
| `inspectSparkrunContainer` / `snapshotContainerLogs` / `captureCrashedContainerLogs` | same, over the local `dgxrun_<depId>` container |
| `checkSparkrunDeployments` / `parseLoadingShards` (sparkrun-metrics) | reuse — the `Loading safetensors` + vLLM `/metrics` scrape are runner-agnostic |

Deployments persist with `kind: "dgxrun"`; reconcile-on-reconnect reuses the
existing loop (each agent re-checks *its* rank container).

### Manager integration
- `routes/deployments.ts`: when the recipe resolves to `runner: dgxrun` (or
  `executor: mp` + `cluster_only`), build the cluster, assign ranks (head=0),
  and fan `cmd:deploy` to each node's agent with its rank + the head's mgmt IP as
  `masterAddr`.
- Status aggregation: deployment status = head rank's serve/health; a worker
  rank dying ⇒ manager tears down **all** ranks (mp deadlocks if one dies — we
  saw this). Coordinated teardown is a manager responsibility.
- Logs/metrics: head rank feeds `/metrics` + loading progress as today.

### Recipe format (minimal, backward-compatible)
Add one optional field: `runner: sparkrun | dgxrun` (default `sparkrun`). The
existing `command`, `env`, `defaults`, `container`, `cluster_only`, `maxoutmem`
are unchanged and shared. dgxrun ignores `--distributed-executor-backend` in the
template and forces `mp`. Example: flip the GLM-5.2 recipe to `runner: dgxrun`
and it deploys via the manager at the validated 85K/23 tok/s.

## Extensibility (the "add recipes later" goal)
- New mp-multi-node model → new recipe YAML with `runner: dgxrun` + its
  `container`/`command`/`env`. Zero runner code changes; orchestration is generic.
- Single-node models can use dgxrun too (`nnodes=1`, no `--node-rank`) — one
  runner for everything if we later choose to retire sparkrun.

## Phasing
1. **v1 — agent runtime + argv builder + unit tests.** `dgxrun-args.ts`
   (pure) + `dgxrun.ts` (docker lifecycle). Reuse sparkrun-metrics.
2. **v1 — manager dispatch.** Rank assignment + per-node `cmd:deploy` fan-out +
   coordinated teardown + status aggregation.
3. **Validate** with the GLM-5.2 recipe (`runner: dgxrun`) via the manager →
   reproduce 4-node mp, 85K, ~23 tok/s end-to-end.
4. **Later** — migrate/author other recipes; consider single-runner consolidation.

## Testing
- **Unit (`dgxrun-args.test.ts`)**: recipe + rank/nnodes/masterAddr → exact docker
  argv (assert `--ipc host`, `mp`, per-node `--node-rank`/`--headless`, env,
  placeholder fill). Mirrors `sparkrun-args.test.ts`.
- **Integration**: stub `docker` exec; assert deploy→running→stop and the
  reconcile path; assert coordinated teardown when a rank reports dead.
- **E2E**: GLM-5.2 via the manager on the 4-node cluster → serves + a real
  generation at ~23 tok/s.

## Risks / open questions
- **Rendezvous timing**: agents launch concurrently across nodes; `masterAddr:
  masterPort` must be reachable on the mgmt net and the TCPStore skew-tolerant
  (glm52-mp-launch.sh's concurrent-launch pattern works; port it).
- **Image consistency/distribution**: the custom container image must exist on
  all nodes (we hit ID drift; a factory-reset node had none). dgxrun should
  verify the image per node and optionally distribute it (docker save→load over
  the fabric, or a shared registry) before launch — or fail fast with a clear
  message.
- **One-rank-dies-hangs-all**: mp has no built-in recovery; the manager must
  detect a dead rank (health + container status) and tear down the whole
  cluster. Define the failure/timeout policy.
- **First-deploy Triton JIT**: sm12x kernels JIT on first inference (slow +
  OOM-prone on a cold cache; persists to `TRITON_CACHE_DIR` on NFS). dgxrun could
  pre-warm or at least surface "compiling kernels" vs "hung." (See memory
  `glm52-shm-and-jit-findings`.)
- **Unified-memory budget**: node prep (drop_caches is NOT a lever during
  serving — real GPU usage, not page cache) and the 0.88 floor / 85K ceiling are
  model-specific; keep them in the recipe, not the runner.
- **Keep recipes portable**: don't fork the schema — `runner:` is the only add.

## References
- Working launcher: `scratchpad/glm52-mp-launch.sh` (the reference orchestration).
- sparkrun runtime to mirror: `packages/agent/src/runtime/sparkrun*.ts`.
- Recipe: `community-recipe-registry/recipes/glm-5.2/kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer.yaml`.
- Findings: memory `glm52-shm-and-jit-findings`, `ssh-hammering-wedges-nodes`.
