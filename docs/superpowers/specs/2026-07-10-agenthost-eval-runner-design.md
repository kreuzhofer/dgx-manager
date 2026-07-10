# Design: move the benchmark runner to agenthost

*2026-07-10*

## Problem

`benchmarks/orchestrator.ts` spawns `uvx` as a child of the **server container on the Raspberry Pi manager**, tracked only in an in-memory `ACTIVE` map. Three consequences, all observed on 2026-07-10 during a 100-prompt IFEval run:

1. **Dev work on the Pi slows the run.** Running `npm test` plus two `tsc` invocations dropped throughput from **29.5 to 48 s/item (~60%)**.
2. **Any server restart kills the run.** `docker compose up -d --build server` recreates the container and the detached `uvx` child dies with it. Boot reconciliation then marks the row `failed`. A long run therefore blocks every rebuild for its whole duration — that run took 1h20m.
3. **The `uv` cache lives inside the server container**, so a rebuild wipes it and the next run re-downloads torch before evaluating anything.

The GPU nodes are reserved for model hosting and must not become eval targets.

## Goal

Run every benchmark kind (`throughput`, `tool-eval`, `accuracy`) on `agenthost` (192.168.44.15) as a **job that outlives both the manager and the agent**, without weakening the dashboard's existing log/status contract.

Explicitly **out of scope**: Docker on `.15`; the SWE-bench and Terminal-Bench kinds; the `eval` recipe variant and `num_concurrent` preset work. This design should make Docker-based evals straightforward later — such an eval is just another `job.start` — but they are not specified here.

## Facts established before designing

Verified rather than assumed:

- **`exec` cannot host a benchmark.** `caps/exec-cap.ts` clamps `timeoutMs` to `Math.min(…, 300_000)` — five minutes — then SIGKILLs the child. An lm-eval run is 80 minutes.
- **`CapClient` times out capability invocations server-side**, so no long-lived streaming RPC is viable either.
- **The existing `sshExec` already reaches `.15`** from inside the server container (`ssh2` library, key `id_ed25519_shared`, `SSH_USER=daniel`) → `code: 0`. SSH is therefore the *existing* remote-exec path, not a second one; the roadmap's framing of this choice was backwards. **SSH is used for provisioning/onboarding only** (as the provisioner already does). Once the agent is installed, jobs run through the capability registry — reusing its auth, audit and node addressing rather than opening a second control channel.
- **The agent runs as `User=daniel`, not root** (`systemctl cat dgx-agent`). Transient system units therefore need `sudo -n`, which `.15` allows (verified). **Prerequisite:** `daniel` must retain `NOPASSWD` sudo on the eval node; the eval provisioning profile asserts this at onboarding and fails loudly if absent, rather than discovering it on the first benchmark.
- **`agenthost`**: x86_64, 16 cores, 30 GiB RAM, 870 GB free, `systemd 255`, passwordless SSH + `sudo -n`. Docker and `uv` absent. `/mnt/tank` **not** mounted — there is no shared filesystem with the manager.
- **No CUDA GPU.** An integrated `AMD Barcelo` iGPU exists; ollama can drive it via Vulkan (`100% GPU`) but it is *not faster* than the 16 CPU cores (5.66 s vs 5.59 s per 20 embeds) because both are memory-bandwidth-bound. Left on CPU. Consequently `nvidia-smi` is absent and the agent will report `vramTotal: null`, which `metrics.ts` already degrades to `"Unknown (nvidia-smi unavailable)"` rather than crashing.
- **`GET /api/nodes/idle` returns every online node with no filtering**, so onboarding `.15` as a Node would place it in the deploy picker.
- **`.15` runs a hand-installed ollama** serving `qwen3-embedding:8b`. The stock provisioner installs sparkrun and other prerequisites and must not be pointed at it unmodified.

## Decisions

| Question | Decision |
|---|---|
| Must a run survive a manager restart? | **Yes** — runs outlive the manager. |
| Managed node or invisible infra? | **Managed agent-only node.** |
| What may `.15` host? | **Evals + Ollama. Never vLLM/dgxrun.** |
| Which kinds move? | **All three.** |
| How does the agent run an 80-minute job? | **`job.*` capability over `systemd-run` transient units.** |
| Concurrent runs per deployment? | **Refuse with 409.** |

### Why `systemd-run`

"A run outlives the manager" means "the run is not anyone's child process." `systemd` is the component on that box whose entire purpose is owning processes that outlive their launcher, and it supplies supervision, cancellation, and exit-status capture for free. Re-implementing that inside the agent (a `setsid` child plus a pidfile plus an exit-code wrapper, with pid-reuse ambiguity) is the speculative machinery YAGNI warns against. `systemd-run` is already installed.

The consequence that makes it work: **every capability call is short.** `job.start` returns as soon as systemd accepts the unit; the manager then *polls*. So neither `exec`'s 5-minute clamp nor `CapClient`'s timeout needs to change.

## Architecture

```
POST /api/benchmarks
   │  503 if the eval node's agent is offline — no silent local fallback
   │  409 if that deploymentId already has a running benchmark
   ▼
orchestrator.runTrackedRemote(runId, argv)
   │  cap: job.start ─────────────► agent on .15
   │                                  sudo -n systemd-run --unit=dgxbench-<runId> \
   │                                    -p User=daniel -p WorkingDirectory=<jobDir> \
   │                                    sh -c '<argv> > log 2>&1; echo $? > exit'
   │  ◄── { unit, jobDir }
   │
   │  poll ~3s:
   │    cap: job.logs(sinceOffset) → bytes
   │         → appendFileSync($SHARED_STORAGE/logs/benchmarks/<runId>.log)
   │         → sseBroadcast                     (dashboard contract unchanged)
   │    cap: job.status → active | exited(code) | unknown
   ▼ on exit
   cap: job.result → result.json → existing parsers (parseLmEvalResults, …)
```

### Components

Each has one purpose and a testable boundary.

- **`packages/agent/src/caps/job-cap.ts`** — `start` / `status` / `logs` / `cancel`. Thin IO: shells to `systemd-run` and `systemctl`, reads two files. Takes an injected `spawnFn` exactly as `exec-cap.ts` does, so it is unit-testable without systemd.
- **Pure helpers** (agent) — `jobUnitName(runId)`, `buildSystemdRunArgv(...)`, `parseSystemctlShow(stdout) → {active, exitCode}`, and the log-offset slice. The logic lives here; the capability carries only the IO.
- **`orchestrator.runTrackedRemote`** — sibling of `spawnTracked`, same `{exitCode, rawOutput}` contract, so `runBenchmark` / `runToolEval` / `runAccuracy` are unchanged below the transport.
- **`Node.role`** — `"gpu" | "eval"`, enforced **server-side** in `POST /api/deployments` (an `eval` node rejects any runtime but `ollama`) and surfaced to the dashboard for picker filtering. UI filtering is convenience; the server check is the boundary.
- **Eval provisioning profile** — installs the agent and `uv`, nothing else. No sparkrun, no nvidia-container-toolkit, and it does not touch the hand-installed ollama.

Job directory on `.15`: `$HOME/.dgx-agent/jobs/<runId>/{cmd,log,exit,result.json}`. `job.start` prunes job dirs older than 14 days. The `uv` cache lands in `/home/daniel/.cache/uv` on persistent disk, so a manager rebuild no longer wipes it.

### Resolving the result file remotely

The three kinds do not agree on where the result lands. `llama-benchy` and `tool-eval-bench` write `<outputDir>/result.json`; lm-eval writes a **nested** `results_*.json` that `findLmEvalResultFile` resolves by walking the directory. The manager cannot stat a remote filesystem, so that resolver cannot stay server-side.

`job.start` therefore takes a **`resultGlob`** (default `result.json`; lm-eval passes `results_*.json`). The wrapper normalises after the command exits:

```sh
<argv> > log 2>&1; echo $? > exit
f=$(find "<outputDir>" -name '<resultGlob>' -print -quit); [ -n "$f" ] && cp "$f" result.json
```

`job.result` then always reads `<jobDir>/result.json`. The existing parsers are untouched. `findLmEvalResultFile` stays only for the `BENCH_RUNNER=local` path.

### Choosing the eval node

Exactly one node with `role: "eval"` is expected. Zero online → **503**. More than one → **fail loudly** naming them, unless `EVAL_NODE_ID` disambiguates. Silently picking the first would make a run's provenance depend on row ordering, which `runnerNodeId` exists to prevent.

## Data model

Three additive columns; nullable or defaulted, so `db push --accept-data-loss` is safe exactly as `Deployment.error` was.

```prisma
model Node {
  role String @default("gpu")        // "gpu" | "eval"
}

model BenchmarkRun {
  runnerNodeId String?               // null = legacy run executed locally on the Pi
  jobUnit      String?               // systemd transient unit, dgxbench-<runId>
  logOffset    Int    @default(0)    // bytes of the remote log already persisted
}
```

`runnerNodeId` is a **provenance marker, not bookkeeping**. Throughput measures decode tok/s and TTFR from the client; moving the client to a different host changes the network path, so agenthost numbers are not comparable with the Pi-measured history. Null means "measured on the Pi". The compare view warns when throughput runs with different `runnerNodeId` are placed side by side.

## Flow

**Start.** Resolve the `eval`-role node. Agent offline → **503, no row created**. A `running` run already exists for that `deploymentId` → **409** naming the offending `runId`. Otherwise create the row, `job.start`, persist `jobUnit`, set `running`.

**Poll** (~3 s): `job.logs(logOffset)` appends to the shared-storage log and broadcasts SSE; `logOffset` advances monotonically. `job.status` decides when to stop.

**Reattach** — this is where "outlives the manager" is actually paid for. On boot, for every `pending`/`running` row **that has a `runnerNodeId`**, ask `job.status`:

- `active` → resume the poll loop from the persisted `logOffset`
- `exited` → drain the remaining log, fetch `result.json`, finalize
- genuinely gone → `failed: "job vanished across manager restart"`

Rows **without** a `runnerNodeId` keep today's behaviour — `failed: "server restarted before run completed"`. The legacy contract is preserved, not retconned.

**Cancel.** `job.cancel` → `sudo -n systemctl stop <unit>`, idempotent (unit already gone = success), then mark `canceled`.

## Failure handling

Fail fast at the boundary; never degrade silently.

- **Agent offline mid-run** → pause polling, resume on reconnect. The job is a systemd unit; it does not care that we stopped watching.
- **`job.status` inconclusive** (cap timeout, busy box) → **skip the tick.** Only "unit gone *and* no exit file" is death. This is the `absent ≠ unknown` distinction that tore down four healthy GLM-5.2 ranks on 2026-07-09; without it, one slow poll would kill an 80-minute eval. It gets a dedicated regression test.
- **Non-zero exit** → `failed`, with the log's last lines in `error`, mirroring the `deployment.error` discipline.
- **Exit 0 but the parser fails** → `failed` with the parser message (already implemented in `runAccuracy`).
- **Local execution** remains available behind an explicit `BENCH_RUNNER=local` for laptop dev. Explicit, tested, observable — never an implicit fallback.

## Testing

Risk tier: medium-high (data persistence + a management plane).

**Property tests** on the pure helpers, each stating an invariant:

- *Concatenating successive offset reads reproduces the log exactly* — no loss, no duplication, under arbitrary chunk boundaries.
- *An unparseable or absent `systemctl show` never yields `exited(0)`* — "we could not tell" must never render as "succeeded".

**Unit tests** for `job-cap` with an injected `spawnFn`, mirroring `exec-cap.test.ts`.

**Integration tests** (server) against a stub `capClient`, the same idiom as the stub `agentHub`:

- no eval node online → 503, no row created
- second run against a busy `deploymentId` → 409
- happy path persists `runnerNodeId` + `jobUnit`; `logOffset` advances
- non-zero exit records `error`
- **an `unknown` status leaves the run `running`** (regression test for the lesson above)
- boot reconciliation: `active` resumes, `exited` finalizes, legacy row fails as today
- cancel is idempotent when the unit is already gone
- `POST /api/deployments` to an `eval` node: `vllm`/`dgxrun` → 400, `ollama` → allowed

## Rollout

Shippable slices, each independently valuable:

1. `Node.role` + server-side admission. Small, standalone, immediately prevents an accidental deploy onto `.15`.
2. Agent `job` capability + pure helpers. Requires an agent roll — **roll agents before rebuilding the server** (a server restart makes every agent reconnect and reconcile).
3. Onboard `.15` with the eval provisioning profile; install `uv`; warm the `uvx` cache.
4. Orchestrator remote path, boot reconciliation, cancel, the 409 guard.
5. Flip the three kinds over; re-baseline throughput and note the `runnerNodeId` break in the compare view.

## Open questions

None blocking. Two noted for later:

- Whether the compare view should *refuse* to compare throughput across runners rather than warn.
- Whether `job.*` should eventually replace the `exec` capability's long-tail uses, or coexist with it.
