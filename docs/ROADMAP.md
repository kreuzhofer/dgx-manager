# DGX Manager Roadmap

*Last updated: 2026-07-15*

> A full-stack system for managing DGX Spark GPU clusters — node provisioning, model deployment, inference, fine-tuning, and beyond.

DGX Manager aims to be the simplest way to operate a personal or team GPU cluster. Add nodes over SSH, deploy models with one click, load-balance inference across machines, and fine-tune models on your own hardware. Built for self-hosted use with a clean web UI and zero cloud dependencies.

---

## Phase 1: Infrastructure & Node Management ✅

**Goal:** SSH-based lifecycle management for GPU nodes with real-time monitoring.

- Node lifecycle: add → audit → provision → deploy agent → monitor
- Auto-provisioning of prerequisites (Docker, nvidia-container-toolkit, Node.js, Ollama)
- Agent runs as a systemd service on each GPU node with auto-reconnect
- Real-time GPU metrics (utilization, VRAM, temperature, network, RDMA) at 5-second intervals with 1-hour rolling buffer
- Dashboard: Overview page with live sparklines, full Nodes management UI
- Agent version tracking with upgrade detection

## Phase 2: Model Deployment & Inference ✅

**Goal:** One-click model deployment with multi-node cluster support and load-balanced inference.

- Deployment via [sparkrun](https://github.com/spark-arena/sparkrun) (the head-node agent runs `sparkrun run`) — vLLM / SGLang / llama.cpp — plus Ollama (native)
- Three recipe sources on `POST /api/deployments`: a registry recipe (`recipeFile`), an NFS path (`recipePath`), or an inline `recipeYaml` body (remote-dev, no cluster-fs access)
- Solo and multi-node cluster deployments (tensor parallelism × pipeline parallelism via Ray)
- VRAM admission control with safety margins and port conflict detection
- Deployment persistence + reconnect reconciliation (kind-scoped) across agent restarts
- Real-time deployment log streaming — incl. live vLLM model-loading output via a `sparkrun logs` follower — and full status lifecycle
- Recipe catalog auto-discovered from sparkrun registries via `sparkrun list` (`POST /api/recipes/refresh` re-scans)
- Server-side load balancer: rules + endpoints API (round-robin / first-available strategy implemented); the inference proxy router (`proxy/inference-proxy.ts`) is written but **not yet mounted in the server**
- Dashboard: Deployment creation (runtime toggle, node/recipe selection), log viewer, stop/restart controls, cluster node visualization

### Remaining UI work

The server APIs for these features are complete, but the dashboard pages are still placeholders:

- **Load Balancer UI** — Rule management, endpoint assignment, strategy configuration
- **Models UI** — Model registry browser and management

### Missing API endpoints

- [ ] `GET /api/deployments/:id` — single-deployment lookup. Today only the list endpoint exists, so any script/UI watching a specific deployment has to fetch the whole list and filter client-side. Should return the same shape as a list entry (deployment + node + model + clusterNodes), 404 if not found. Trivial to add (mirror finetune.ts pattern), high value for monitoring scripts and the deployment-detail page.

### Known manager bugs (found 2026-07-09/10 — all fixed 07-10, plus a fifth found while fixing them)

Each of these cost real debugging time during the GLM-5.2 incident. Ordered by blast radius. Two of the diagnoses recorded here on 07-09 turned out to be **wrong**, and both were only caught by trying to write the fix — see the `DELETE` and `maxoutmem` entries.

- [x] **`DELETE /api/deployments/:id` right after `POST` races the launch → orphaned containers.** ✅ *Fixed 2026-07-10 (`b096b6f`, agent 0.5.771).* The mechanism was NOT "undeploy is processed before deploy". `launchDgxrun` spawns `docker run -d`, which **returns before the container exists**; `handleCommand` is not awaited, so `cmd:undeploy`'s own `docker rm -f` sweeps a container that isn't there yet, reports `stopped`, the manager drops the row — and the container then comes up untracked. Fix: a cancel registry (`runtime/deploy-cancel.ts`) that the launch re-checks at `docker run -d` exit, the one moment the container provably exists. A pending cancel beats a failed launch (else a DELETE strands the row); a new launch supersedes an older cancel (else restart-after-stop never launches). `cmd:undeploy` for an *untracked* id now also runs `docker rm -f dgxrun_<id>` — that is why a second DELETE never cleaned up an orphan.
- [x] **`POST /api/deployments/:id/restart` ignored the dgxrun runner.** ✅ *Fixed 2026-07-10 (`b096b6f`).* It fell through to the single-node sparkrun `cmd:deploy`, which launched nothing. Restart now rebuilds the per-rank fan-out with `buildDgxrunDeploys` from the `dgxrunRecipe`/`masterPort` the original POST already persisted, head first so rank 0 is the head. Returns 409 rather than dispatching `masterAddr: null` when a node has no management IP. No explicit undeploy needed — `launchDgxrun` already does `docker rm -f dgxrun_<id>` first.
- [x] **Reconnect reconciliation could fail a healthy rank on an unanswered `docker inspect`.** ✅ *Fixed 2026-07-10 (`32032bb`, agent 0.5.771).* Found while preparing the agent roll, not from a report. `isDgxrunRunning()` collapsed a 10 s spawnSync timeout / busy daemon into "not running", and one `failed` rank makes the manager tear down ALL ranks. An agent roll is exactly when the daemon is busiest, so **the roll itself could have destroyed the live 320K deployment**. Same absent-vs-unknown conflation fixed in `dgxrun-metrics.ts` for 0.5.770; this path was missed. Pure `reconcileDgxrunAction()` now returns skip|phase|report and lets the health loop converge on `unknown`. `isDgxrunRunning()` deleted outright — a footgun beside the safe API.
- [x] **A vLLM startup failure never reaches `deployment.error`.** ✅ *Fixed 2026-07-10 (`1ada1ce`).* Worse than described: `Deployment` had **no `error` column at all**, so the agent's error line was logged and SSE-broadcast but never survived a reload. Persisting it needed care — a crash reports `failed` *with* an error and then `stopped` *without* one as teardown finishes, so a blind write erases the cause. Pure helper `ws/deployment-status.ts`: keep any non-empty error; clear only on `running`; otherwise leave the column alone. Restart clears it (the row is reused). Dashboard shows a "Failure reason" banner on terminal deployments.
- [x] **`maxoutmem: true` is a silent no-op for dgxrun recipes.** ✅ *Fixed 2026-07-10 (`24b2440`).* The earlier diagnosis here was wrong: nothing was swallowed by the `.catch()`. `readMaxOutMemCmd` greps `$HOME/.cache/sparkrun/registries/<reg>/recipes`, `find` matched nothing, and the probe printed `false` — so `maxOutMemoryForDeploy` returned `applied:false` **without an error and without a log line**. Not harmless: this is the unified memory that was freed by hand before every 256K/320K launch. Fix: `parseMaxOutMemYaml()` reads the flag from YAML the manager already holds (`@dgxrun/` catalog, inline `recipeYaml`, `recipePath`) and passes it as `enabled`, skipping the SSH probe; the probe remains for plain sparkrun refs. The skip is now logged.

### Pending hardware verification (2026-07-10 fixes)

All five bug fixes are unit/integration tested and deployed, but four behaviours have **never executed against real hardware**. Ordered by what a failure would cost.

- [ ] **dgxrun restart, end to end.** `POST /api/deployments/:id/restart` → expect `{"status":"restarting","ranks":4}`, then ~15–20 min of weight load + `torch.compile`. Only ever exercised against a stub agentHub. Needs a maintenance window — a failure leaves the 320K deployment down until redeployed.
- [ ] **The orphan-race fix.** Deploy a dgxrun recipe, `DELETE` it within ~1 s, then confirm `docker ps | grep dgxrun_` is empty on every node. Blocked while GLM-5.2 holds the pool: VRAM admission 409s the deploy before anything dispatches.
- [ ] **`maxoutmem` reclaiming on real nodes.** `reclaimMemoryCmd()` has never run — the bug meant it was never reached. Fires on the next *fresh* deploy. Watch `docker compose logs server -f | grep maxoutmem` for `freed NNNN MiB, gdm=inactive`. `freed 0 MiB` / `gdm=unknown` means `sudo -n` was denied for the SSH user (every step is `|| true`); the line now says so either way.
- [x] **The "Failure reason" banner renders.** Verified 2026-07-10 by setting `error` on a terminal row; confirmed present in the deployed dashboard chunks. Does *not* exercise the agent→hub→DB write path — only a genuinely failed deploy does.

### Open decision: should `restart` reclaim memory?

- [ ] `maxOutMemoryForDeploy` is called only from `POST /api/deployments` (`routes/deployments.ts:573`). The **restart handler returns before reaching it**, so a restart does *not* free unified memory even for a recipe with `maxoutmem: true`. Arguably it should — a restart faces the same cold-start memory wall a fresh deploy does (that wall is what `min_free_kbytes` + the `gmu` guard made lethal). Deliberately left as-is rather than changed silently.

### Deploy config override UI

- [ ] Deploy form (and fine-tune `Deploy` action) should expose recipe-level config overrides — at minimum `max_model_len`, `gpu_memory_utilization`, `tensor_parallel`. Today recipe defaults (or worse, `recipe.yaml`'s `deploy.max_model_len`) win silently and the only way to override is hand-crafting a `POST /api/finetune/:id/deploy` body with `config: {maxModelLen: …}`. Concrete miss case: the qwen3.6-27b training recipe has `deploy.max_model_len: 4096` as a smoke-test leftover, which shadows `inference.yaml`/`inference-fp8.yaml`'s 128000 default, so every fine-tune deploy lands at 4k context without warning. UI should: (a) show the effective value with provenance (recipe.yaml > inference[ -fp8].yaml > server default), (b) let the user override per-deploy, (c) persist the override on the deployment record so restart preserves it.

### Deploy status accuracy

- [ ] Deployment `status: "running"` should mean vLLM is actually serving, not just that the container started. Today the manager flips to `running` when the agent reports container start (often within 60s), but for a 27B model on 2 nodes, vLLM doesn't bind port 8000 until safetensors load + Ray init + cudagraph capture finish (~5–6 min). A scripted "wait for ready" loop hits a connection-refused window even though the dashboard says "running". Options: add a `loading` sub-state until the `/v1/models` probe succeeds; or have the agent stream readiness events back over the WS and update the deployment row when the API is bound. Either way the UI should show "loading" with progress, not "running". *(June 2026 update: live vLLM model-loading logs now stream via a `sparkrun logs` follower, but `running` still fires at serve-command launch, not at `/v1/models` readiness. Also found: a crash-looping failed deploy keeps re-downloading on each docker restart while the follower narrates progress — so a failed deploy can look alive, and status can stay stale after it stops. Tracked as a status-accuracy follow-up.)* *(July 2026 update: the agent now gates `running` on an `apiReady` probe — dgxrun head checks `/metrics`, sparkrun checks `/v1/models` — AND a `shouldReportStatus` fix reports the `starting→running` transition past the VRAM throttle, which previously dropped it and left a serving multi-node deploy reading `starting` forever. So `running` now means the API answered. Remaining: surface an explicit `loading` sub-state + progress in the UI, and detect the crash-loop-redownload case.)* *(July 7–8 2026 update: the agent's log→phase classifier is now a tested `runtime/deploy-phase` module with a **`compiling`** phase and **forward-only** reporting — the post-load "Prefetching checkpoint files" line no longer regresses loading→downloading, and a reconnect no longer snaps back to `starting`. Honest `loading`/`compiling` sub-states now flow via `agent:deployment:status`; deployed in agent 0.5.756 on all 4 nodes. Crash-loop-redownload detection still open.)*

### Deploy performance (open)

- [ ] **Local per-node weights — the real deploy-speed lever.** A cold DCP deploy is ~16 min, but that's ~13 min *NFS weight load* (`Loading weights took 765s`) + only ~50 s compile. Copying the ~388 GB checkpoint to per-node local disk (off `/mnt/tank`), or a faster fabric, is what cuts the 13 min. Larger effort: 388 GB/node staging + invalidation.
- [ ] **Persist per-node JIT caches** (parked, low value) — bind-mount a node-LOCAL dir → `/root/glm-jit` in `dgxrun-args.ts`, keyed by model + container-image tag; **never NFS** (the original cross-rank cubin race). Saves only the ~50 s compile — parked because the load dominates. See `#26` in the task list for the full note.

## Phase 3: Fine-Tuning Pipeline ✅

**Goal:** Run training jobs on cluster nodes with full visibility from the dashboard, then deploy fine-tuned models for inference.

### Training

- Recipe-based training system mirroring vLLM deployment recipes
- External training recipes repo ([dgx-manager-fine-tune-recipes](https://github.com/kreuzhofer/dgx-manager-fine-tune-recipes)) with shared `lib/` for DGX Spark patches, dataset handling, logging
- Supported frameworks: DeepSpeed ZeRO-2 + LoRA, plain TRL + PEFT, Unsloth QLoRA
- Tested recipes: Gemma 4 E2B, E4B, 26B-A4B (DeepSpeed), Gemma 4 E2B (TRL), Llama 3.1 8B (Unsloth)
- Multi-format dataset support: ShareGPT, OpenAI, QA (question/context/answer), Instruct (instruction/input/output)
- HuggingFace dataset IDs loaded directly (e.g., `b-mc2/sql-create-context`)
- Dashboard: Job creation form with recipe/node/dataset selection, hyperparameter overrides
- Multi-node training: DeepSpeed ZeRO-3 across 2+ DGX Spark nodes via torchrun + hostfile
- InfiniBand/RDMA passthrough for NCCL on both head and worker containers
- NCCL timeout workaround: `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` (PyTorch bug #124950)

### Monitoring & Observability

- Phase-aware progress tracking: container → downloading → loading → tokenizing → training → eval → saving
- Real-time loss curve visualization (SVG chart, live-updating via SSE) — eval points overlaid on training loss in the same chart
- Training metrics persisted to DB with `@@unique(jobId, step)` — upsert on write prevents the 3× duplicates the agent's three parser patterns used to produce
- Eval phase detected at the START of evaluation (via `on_prediction_step`) instead of end — UI flips to "evaluating" immediately
- `[TRAIN]` and `[EVAL]` callbacks for explicit progress reporting through Docker pipes
- Smoothed ETA estimation (20-sample rolling average of iteration speed)
- **Per-rank shell-level `tee` in `lib/setup_logging.sh`** — captures EVERYTHING (Python + C/C++ extensions from PyTorch and DeepSpeed loaders), one log file per rank to avoid garbled NFS interleaving
- Log file persistence survives page refresh, agent restart, and container restart
- Agent reattaches to running training containers after restart (tails train.log) — reattach handler forwards the same fields as the live path (`evalLoss`, `lr`)
- Deployment log persistence to NFS files

### Production Resilience

- **Resume from checkpoint** — `POST /api/finetune { resumeFromJobId }` inherits the previous job's outputDir + config and passes `--resume_from_checkpoint true` to `trainer.train()`. Works with DeepSpeed when `save_only_model=False` (full ZeRO optimizer state saved alongside the LoRA adapter).
- **Configurable save/eval cadence** — `save_steps`, `eval_steps`, `save_total_limit`, `save_only_model`, `eval_fraction` all passable via the API config (not hardcoded per recipe).
- **Multi-node watchdog** — 30s poll of each worker container's state. If a worker exits while head is still running (typical symptom of a rank crashing inside NCCL while `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` disables the timeout), force-clean the head and mark job failed. Turns silent multi-day zombies into clear errors.
- **Force-cleanup path** — cluster worker IPs persisted to `$outputDir/.cluster-workers.json`, reloaded on reattach. `stopFinetuneJob` always confirms `agent:finetune:complete` to the server so jobs never get stuck in "stopping".
- **Worker log capture** — detached workers now redirect stdout/stderr to `$outputDir/worker-{ip}.log` so rank>0 crashes are debuggable.
- **Delete with cleanup** — `DELETE /api/finetune/:id?cleanFiles=true` removes the outputDir unless another job shares it (resume chain). Dashboard two-step confirm shows size + warning.
- **Confirm prompts** — destructive Stop/Restart actions on fine-tune jobs and deployments require explicit confirm with context (model, node, consequences).

### Merge & Deploy

- LoRA adapter merge: loads base model + adapter → `merge_and_unload()` → saves full model
- Gemma 4 weight key fix: remaps flattened ClippableLinear keys back to nested format for vLLM
- PEFT dispatch patch: teaches LoRA to handle Gemma4ClippableLinear without model modification
- Dynamic vLLM recipe generation for merged models (container type from training recipe config)
- Deploy button on completed+merged jobs → navigates to deployments page with model pre-filled
- Full loop tested: train → merge → deploy → serve → inference

### DGX Spark Workarounds

- pynvml monkey-patch (nvmlDeviceGetMemoryInfo not supported on GB10 unified memory)
- NFS page cache flush after safetensors shard loads
- `skip_memory_metrics=True` for HF Trainer
- Auto-increment torchrun master port for concurrent training jobs
- `chmod a+rw` on output files (containers run as root)

### Storage abstraction

All storage paths configurable via `SHARED_STORAGE_PATH` env var — no hardcoded `/mnt/tank`. Future iterations should:

- Support alternative shared storage backends (S3/MinIO, Ceph, host-local with rsync)
- Allow dataset and model transfer via the agent when shared storage is unavailable
- Provide a setup wizard or configuration file for storage topology

## Phase 3.5: Agent Bootstrap & HTTP Updates ✅

**Goal:** Remove NFS and SSH as hard dependencies for adding nodes and updating agents.

### Join Token Bootstrap

- Server generates single-use join tokens via `POST /api/tokens`
- Self-contained install script at `GET /api/agent/install.sh` — provisions Docker, nvidia-container-toolkit, Node.js, downloads agent, creates systemd service
- Agent registers with token on first connect, server auto-creates Node record
- Node ID persisted to `/opt/dgx-agent/node-id` for subsequent reconnects
- Three tiers: Full SSH+NFS, SSH only, Agent-only (join token)

### HTTP Agent Updates

- Server builds and serves agent bundle tarball at `GET /api/agent/bundle`
- `POST /api/nodes/:id/update-agent` sends update command via WebSocket
- Agent downloads bundle, swaps files atomically, restarts systemd service
- Works for all nodes regardless of bootstrap method
- ✅ `POST /api/nodes/update-agent-all` — bulk agent roll (shipped 2026-07-08). Fans `cmd:update` to every online node not already on the bundled version (`?force` includes current ones); returns `{ version, dispatched[], skipped[], offline[] }`; per-node outcome via the `node:status`/`agentVersion` SSE. Parallel dispatch (safe with 0.5.756's robust self-update). Live-verified (4 skipped, aihost01 offline). *Follow-ups:* serial-with-progress-streaming + a dashboard "Update all agents" button.
- [ ] Remove dead offline FP8 quantize plumbing now that FP8 deploys use vLLM's on-load `--quantization fp8`. Specifically: `POST /api/finetune/:id/quantize`, the agent `cmd:finetune:quantize` handler, `quantizationStatus`/`quantizedPath`/`quantizationLog`/`quantizedAt` columns, the dashboard Quantize button, `scripts/quantize_fp8.py` in the recipes repo. Driven by an unresolvable llmcompressor↔transformers pin conflict (llmcompressor 0.10 pins `transformers<=4.57.6`; qwen3_5 needs `>=5.0`). Left in place for now so the change footprint stays small; can be ripped out once we're sure on-load FP8 is the durable answer.

### Settings Dashboard

- Settings page at `/settings` with token management (create, revoke, list)
- Install command display with one-click copy
- Agent bundle version and download link

### Multi-Node SSH Key Exchange

SSH remains the coordination mechanism for multi-node training (torchrun) and vLLM clusters (Ray). To support token-bootstrapped nodes in multi-node jobs:

- [ ] Agent generates/reports SSH public key on registration
- [ ] Server stores public keys and distributes authorized_keys before multi-node jobs
- [ ] Pre-flight SSH connectivity check between participating nodes
- [ ] Clear setup documentation in install script output and Settings page

## Phase 3.6: Heterogeneous Hardware Support (in progress)

**Goal:** Let the cluster host nodes beyond DGX Spark (arm64 + Grace Blackwell) — specifically commodity amd64 machines with consumer Blackwell cards (RTX 5090-class, 32 GB).

- Per-architecture agent bundles built via `docker buildx` + QEMU (`scripts/build-agent-bundles.sh`)
- `Node.arch` tracked in DB; agent reports `process.arch` on registration; dashboard shows an arch badge
- Install script detects `uname -m` and downloads `agent-bundle-{amd64,arm64}.tar.gz`
- Agent self-audit runs on reconnect and populates `provisionLog` with prereq checks (matches the SSH-audit format), so token-onboarded nodes show the same badges as SSH-onboarded ones
- Install script now provisions Ollama + systemd drop-in (`OLLAMA_HOST=0.0.0.0`, `OLLAMA_MODELS=/mnt/tank/models/ollama` when NFS is mounted) to match what `provisionNode` did for SSH nodes
- Node rename UI (`PATCH /api/nodes/:id`) for post-onboarding cleanup
- `node:created` SSE event so new nodes appear in the UI without a reload
- Onboarding dialog shows a live health-check panel (agent connected, hostname, arch, GPU, Docker, metrics flowing, agent version) once the token is consumed

### Remaining

- [ ] **Consumer-GPU recipes** — a sparkrun registry of curated single-GPU recipes targeting 32 GB amd64 hosts (Nemotron-3-Nano-NVFP4, Qwen3-Coder-Next int4, GLM-4.7-Flash-AWQ, merged finetunes), each pinning a slim single-node container. Registered via `sparkrun registry add` so it appears in the catalog alongside `@official`; no separate recipe-repo plumbing needed now that deployment goes through sparkrun.
- [ ] **VRAM admission guard** — compare `recipe.vram_required` against `node.vramTotal` at deploy time and reject with a clear error, to prevent wasting a build cycle on models that can't fit.
- [ ] **Heterogeneous cluster guard** — scheduler refuses to form multi-node training/vLLM clusters that mix arches or GPU classes.

## Phase 3.7: Agent v2 — Management Plane over the Agent (in progress)

**Goal:** Make SSH optional for node management. The agent becomes a management plane — deep observation + safe remote execution — that keeps working when a node is degraded (wedged sshd, fork-starved), because the agent WS stays alive when SSH dies.

**Motivation (July 2026):** During the GLM-5.2 work, nodes' sshd repeatedly wedged under SSH connection load while the agent WS kept heartbeating — but the WS couldn't diagnose or manage anything, so we were blind exactly when we needed sight. The head node's wedge is most likely unified-memory pressure starving `fork()` (unconfirmed — no non-SSH channel existed to prove it), which also breaks any shell-based management path. Hence: **fork-free** observation as the foundation.

### Phase 1 — Incident-Response Core ✅

- Fork-free `/proc`+`/sys` diagnostics (`diag.collect`): memory, PSI pressure (cpu/mem/io), load, pid/fd counts, **sshd `:22` connections by TCP state** (distinguishes MaxStartups pre-auth pileup from fork-starvation), thermals, kmsg tail — read in-process so it works when `fork()` is failing
- Rich streaming metrics: the same fields on every tick, self-healing (fixes the `null` memory/PSI we hit)
- Audited, reason-required `exec` break-glass capability (streamed output + `AuditEvent`)
- A capability registry (typed WS request/result/chunk with correlation IDs) as the extensible foundation Phases 2-4 build on
- Server surface: `POST /api/nodes/:id/diag`, `POST /api/nodes/:id/exec`, audit table
- **Merged + deployed** (agent 0.5.738 on all 4 nodes). This is what let us pull the `.36` diagnostic journal SSH-free and root-cause the loaded-head death. Spec: `docs/superpowers/specs/2026-07-05-agent-v2-phase1-incident-response-design.md`

### Phase 2 — Robust self-update ✅ (transfer pieces pending)

- **Non-blocking detached updater** — `cmd:update` copies `dist/updater.js` to `/tmp` and spawns it detached+unref, so the agent keeps heartbeating through the update (fixes the blocking `execSync` chain that wedged `.36` twice — memory `agent-cmd-update-spawnsync-wedges-agent`)
- **Atomic swap with self-restore + health-check + auto-rollback + truthful outcome** (`success`/`rolled-back`/`failed`/`rollback-failed`), reported on reconnect; entrypoint guard + unconditional stale-lock clear. Merged + deployed (0.5.738, SSH-direct for the transition since the first roll still used the old path). Spec: `docs/superpowers/specs/2026-07-05-agent-v2-phase2-self-update-design.md`
- [ ] **Remaining:** peer-pull from sibling nodes + container-image transfer over the fast fabric (fixes manual `docker save|load`) — own spec

### Phase 3 — Declarative provision/restore + staleness sweep (staleness done)

- **Heartbeat-staleness sweep ✅** — `AgentHub` runs a 10 s sweep marking any `online` node whose `lastSeen` age > 30 s as `offline` + dropping it from the live agents map + SSE; self-heals to `online` (and re-asserts the agents-map entry, so it stays *reachable*) on the next heartbeat, with a one-time recovery broadcast. A dead/half-open node now reads offline in ~40 s instead of 76 min (the old close-only path). Merged + deployed + **live-verified on `.39`** via a SIGSTOP stall (0 WS close events, `[staleness] marked offline` fired). Fixes memory `node-status-online-unreliable`. Spec: `docs/superpowers/specs/2026-07-05-heartbeat-staleness-sweep-design.md`
- [ ] **Remaining:** declarative node provision/restore (netplan/fabric/NFS/docker-`default-shm-size`/sudoers) — fixes factory-reset-restore-by-hand
- [ ] Fast-follow: the recovery self-heal broadcasts `node:status:online`, but the *offline→online* transition still isn't reflected on the Nodes page for a node that recovered via a fresh `register` before its next metric (harmless redundant path); low priority
- **✅ Agent 0.5.770 (2026-07-10) — the agent was killing healthy deploys.** Blocking `spawnSync("sync; … drop_caches")` on a 500 ms loop parked the event loop for 1–7 s under NFS weight load (missed heartbeats + starved docker probe), and `inspectDgxrunContainer()` collapsed a **timed-out** `docker inspect` into "container gone", tearing down all four ranks. Fixed: async drop + in-flight guard; pure `classifyDockerInspect() → found | absent | unknown` (`unknown` skips the tick, `absent` must repeat). 16 tests.
- [ ] **Residual: the heartbeat wedge is reduced, not eliminated.** `inspectDgxrunContainer`, `snapshotDgxrunLogs` and `captureCrashedDgxrunLogs` are still `spawnSync`, so `[staleness] … marked offline` still fires during weight load — the deploy now *survives* it because `unknown ≠ absent`. Make the docker inspect async (`checkDgxrunDeployments` is already `async`). **Rule: never call `spawnSync` on the agent's hot path** — third incident of this class.

### Later phases (specs TBD)

- [ ] **Phase 4** — fold dgxrun deploy management onto the capability registry
- [ ] mTLS + per-node `exec` arming (Phase 1 ships token auth + audit)

## Phase 4: Dataset Management (in progress)

**Goal:** First-class support for training data throughout the fine-tuning workflow.

- Upload datasets via dashboard (multipart file upload to shared storage)
- Register existing NFS paths or HuggingFace dataset IDs
- Auto-detect format from first JSON line (ShareGPT, OpenAI, QA, Instruct)
- Dataset preview (first N rows displayed in dashboard)
- Dataset picker in fine-tune job creation (replaces raw text input)
- [ ] Dataset versioning and lineage tracking
- [ ] CSV/Parquet format support

## Phase 5: Evaluation & Benchmarks (in progress)

**Goal:** Measure and track model quality across deployments and fine-tuning runs.

### Benchmarking (shipped)

- Two benchmark `kind`s via `POST /api/benchmarks { deploymentId, presetId }`:
  - **`throughput`** (llama-benchy): presets `quick-smoke`, `chat-short`, `chat-long`, `code-32k`, `throughput`; per-concurrency decode tok/s (decode-only `meanTps`) + TTFR to `result.json`/DB
  - **`tool-eval`** (tool-eval-bench via `uvx`): presets `tool-eval-quick`/`-full`/`-hardmode`/`-pressure`; `toolEvalScore`/`toolEvalRating`/`toolEvalCategories`
- Dashboard: results list, per-run detail (`/benchmarks/[id]`), and a compare view (`/benchmarks/compare`)

- SQL evaluation script — implemented and validated on `b-mc2/sql-create-context`
  - **Gemma 4 E2B**: base 4% → fine-tuned 22% exact-match accuracy (+18pp, 5.5× ratio) on 50 examples
  - **Gemma 4 E4B**: base 23% → fine-tuned **90%** exact-match accuracy (+67pp, 3.9× ratio) on 100 examples
  - Key lesson: eval prompt format must match training format (no system prompt, same context layout)
  - Better SQL normalization: handles quote style, chat template artifacts, whitespace
- **Gemma 4 26B-A4B-it LoRA run — SQL eval worse than base** (investigated 2026-04-22, no re-run yet)
  - 1 full epoch, 9331 steps, 2 DGX Spark nodes (ZeRO-3), ~46h. Eval loss monotonically decreased (500→2.14 … 3500→1.93) so training *looked* fine.
  - Root cause (static analysis of adapter safetensors + base model index — no retrain needed): LoRA `target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` **never matched any MoE expert or router weight**. Adapter wrapped 394 modules: 30 language layers × 7 dense/attention projections + 27 vision-tower layers × 7 projections. Zero MoE modules.
  - Why: Gemma 4 26B-A4B uses a shared-expert pattern — every language layer has *both* a dense MLP (`mlp.{gate,up,down}_proj`, `nn.Linear`, matched) *and* an MoE block with fused 3D tensors (`experts.gate_up_proj`, `experts.down_proj`) plus `router.proj` / `router.scale` / `router.per_expert_scale`. None of those are `nn.Linear` and none end with `.gate_proj` / `.up_proj`, so PEFT's endswith matcher skips them. The MoE path (bulk of the 26B params, most of the active compute on the GB10) got zero gradient. The shared-dense path learned to emit SQL, but routing into the un-adapted experts produces a distribution mismatch → worse than base at inference.
  - Also: adapter capacity was partially wasted on the 27-layer vision tower (text-only SQL task).
  - Prompt format parity *was* verified OK — same `{context}\n\n{question}` single-user-turn in training (`lib/dataset.py:67-88`) and eval (`scripts/evaluate.py:66-72,180`). Not the cause.
  - Full fine-tuning memory estimate (bf16 + AdamW fp32): ~416 GB for params+grads+optimizer+fp32-master; needs **4–5 Sparks** with ZeRO-3 (+ ~20-30 GB activations/overhead per rank), or 3–4 Sparks with 8-bit AdamW. "A4B" is a compute optimization, not a memory one — full FT of 26B-A4B sits in the full 26B memory class. Current 2-Spark setup cannot hold it.
  - [ ] Next: re-run 26B LoRA with `target_modules` restricted to attention only (`q_proj,k_proj,v_proj,o_proj`) and scoped to `model.language_model.layers` (exclude vision tower). Cheap sanity check — if ≥ base, confirms the dense-MLP adaptation drove the regression via MoE-routing drift.
  - [ ] After that: write a proper MoE-aware LoRA dispatch — wrap `experts.gate_up_proj` / `experts.down_proj` as per-expert low-rank deltas on the fused 3D tensors, leave `router.*` frozen. Standard PEFT can't do this out of the box.
  - [ ] Harden `fix_clippable_linear_keys()` in `lib/patches.py:119-172` — the "copy any missing key from base" fallback is fragile for MoE; make it log *every* overwrite so a silent expert-key mismatch can't hide.
- ✅ **Accuracy-eval benchmark kind (lm-eval-harness)** — shipped 2026-07-09 (merged `2bb7730`). Third `kind` next to `throughput` + `tool-eval`: same `POST /api/benchmarks`, Benchmark button, detail card, list pill/score, and compare bars. **12 presets** (quick + full) across **IFEval, MMLU-Pro, GPQA-Diamond, GSM8K, BBH, MATH-hard** — the HF Open LLM Leaderboard v2 set minus MuSR. 903 tests green; server + dashboard `tsc` clean. Spec/plan: `docs/superpowers/{specs,plans}/2026-07-09-accuracy-eval-lm-eval-benchmark*`.
  - **Deviated from the design, deliberately:** runs **server-side via `uvx`** (like the other two kinds), not node-dispatched — the GPU nodes are reserved for model hosting; a dedicated benchmark host is the future path. Lineup grew from 4 to 6 (added BBH + MATH-hard).
  - **Reasoning handling:** an ephemeral localhost **strip proxy** removes `<think>…</think>` before lm-eval scores, so *stock* lm-eval tasks run unmodified (chosen over forking each task's filter pipeline).
  - **Data model:** two nullable columns (`accuracyScore`, `accuracyMetrics` JSON) on `BenchmarkRun` — no new table.
  - Final whole-branch review caught two real bugs the per-task reviews could not: the parser hardcoded lm-eval's `,none` metric filter (would have killed `gsm8k_cot`/`mmlu_pro`/`gpqa` on arrival), and a parse failure on a zero-exit run reported the nonsensical "lm-eval exited with code 0".
  - [x] **✅ VALIDATED END-TO-END (2026-07-10).** `acc-ifeval-quick` against the live GLM-5.2 @320K → **`prompt_level_strict_acc` 0.83** (±0.038, n=100), 1h20m wall. Breakdown persisted; the filter-agnostic parser resolved lm-eval's `prompt_level_strict_acc,none` key correctly. The first real run found a bug no unit test could: `local-chat-completions` needs the **`api` extra** (tenacity/aiohttp) — the old assertion `/^lm-eval\[.+\]/` was too loose to catch its absence (`224c050`).
  - **Caveat when reading IFEval numbers:** `loose` scored *below* `strict` (0.82 vs 0.83), which looks impossible since loose tries 8 variants including the raw response. Cause: `lm_eval/tasks/ifeval/utils.py` has strict and loose each call `kwargs = {k:v for … if v}` then `build_description(**kwargs)` **independently**, and `instructions.py` refills 23 dropped params with an **unseeded `random.choice`**. They can grade against different constraints. **IFEval is not bit-reproducible** — treat ±1 prompt as noise, report `prompt_level_strict_acc`.
  - [x] ✅ Pin `LM_EVAL_VERSION` — plumbed into `docker-compose.yml`, defaulted to **0.4.12** (`224c050`).
  - [ ] Warm the `uvx` cache for lm-eval in `Dockerfile.server` (only llama-benchy is warmed today, so the first run downloads torch before evaluating anything).
  - [x] ✅ GPQA `HF_TOKEN` resolved (2026-07-12) — gate accepted + token dropped at `~/.cache/huggingface/token` on the eval node; GPQA-Diamond full ran → 69.2%.
  - [x] **✅ Eval-speed unlock — DONE (2026-07-12): the `c16-64k` concurrent recipe + `num_concurrent` plumbing shipped.** Batched serving (no MTP, `--max-num-seqs 16`, 64K, PIECEWISE) + a `numConcurrent` override on `POST /api/benchmarks`. Reality check: on GB10 this buys only **~2×** aggregate (throughput-bound, not 5–8×), and a 128K variant death-spiraled on prefill-heavy agentic work (dropped). Full lessons in [[glm52-c16-64k-concurrent-recipe]]. Original note kept below for context: Accuracy evals are **model-bound, not runner-bound.** First real run (2026-07-10): IFEval `--limit 100` progresses at **~1 prompt/minute** → ~100 min. Cause: the serving recipe pins **`--max-num-seqs 1`** (chosen for decode speed), so there is no batching, and `buildLmEvalArgs` hardcodes `num_concurrent=1` to match. **Moving the runner to `agenthost` will not speed this up.** The lever is a *separate eval deployment* — higher `--max-num-seqs`, no MTP drafter, smaller context — plus making `num_concurrent` a preset field. Consider an `eval` recipe variant alongside SPEED/CONTEXT. **Confirmed by the completed run:** 100 IFEval prompts took **1h20m** at ~48 s/item. A batching eval deployment should cut that by an order of magnitude and is the prerequisite for SWE-bench being practical at all.
  - [x] **✅ Moved the lm-eval runner to `agenthost` (192.168.44.15).** Shipped 2026-07-11 via the `job.*` systemd-run design (NOT `exec` — that caps at 5 min; the manager polls short cap calls instead). Onboarded as an `eval`-role agent-only node; runs outlive the manager. See "Recent work (July 10–11)". The original motivation stands: on the Pi the `uvx` child competed for CPU (29.5 → 48 s/item under load) and a server restart killed the run.
- [x] **✅ SWE-bench on `agenthost` — DONE (2026-07-14).** Docker installed on `.15`; ran the full **Verified 500 via mini-SWE-agent** (v2.4.5, litellm over the vLLM endpoint) → **73.0% (365/500)**, scored with the swebench harness 4.1.0. Run standalone on agenthost (not yet wired as a 4th benchmark `kind` — the manager integration would reuse the `BenchmarkRun` row + a Docker-agentic runner; left as a future nicety). See [`docs/glm-5.2-benchmark-results.md`](glm-5.2-benchmark-results.md).
- [ ] Track quality metrics over time with per-model and per-run history
- [ ] Compare base models against fine-tuned variants in the dashboard (currently only per-job chart)
- [ ] Dashboard views for benchmark results and regression detection
- [ ] **30-min moving-average curve layered on the live TPS graph.** Overlay a 30-minute rolling-average tokens/sec curve on the raw per-tick throughput graph. Raw aggregate TPS is spiky under batched serving — requests complete in waves (the concurrent `c16-64k` eval recipe showed ~16 finishing together, 2026-07-12), so the instantaneous number is hard to read. A 30-min moving average shows the *sustained* throughput and makes recipe / batch-size comparisons legible. *(requested 2026-07-12)*
- [ ] Upload fine-tuned models to HuggingFace (requires HF_TOKEN)
- [ ] Multimodal fine-tuning: Gemma 4 as a visual judge (image+text training)
- [ ] **External-facing write-up (blog / LinkedIn post)** — distinct from `docs/gemma4-fine-tuning-on-dgx-spark.md` (the deep technical reference). Audience-focused "so what" framing: what DGX Manager says about the author's skills to a hiring manager / CTO / practitioner network. Pending audience + platform decisions.
  - **Raw material is now captured** in `docs/glm-5.2-256k-to-320k.md` — the 2026-07-09/10 incident-to-320K journey, with every number measured: the reboot-armed `min_free_kbytes` landmine, the reserve that prevented the capture it was meant to protect, `"unknown" ≠ "absent"` tearing down four healthy ranks, DCP4 as a red herring, and the one-line `max-num-batched-tokens` fix found by reading a traceback instead of rebuilding an image. Section 6 ("What generalises") is the spine of the post.

## Phase 6: User Auth & Multi-Tenancy

**Goal:** Support shared clusters with access control and usage tracking.

- User authentication and role-based access control (RBAC)
- API key management for programmatic access
- Usage quotas and accounting per user/team
- Audit logging for cluster operations

## Phase 7: Multi-Cluster Support

**Goal:** Manage multiple separate GPU clusters from a single DGX Manager instance.

- Cluster-as-a-resource abstraction
- Per-cluster node pools, deployment targets, and metrics
- Cross-cluster model promotion and deployment workflows
- Unified dashboard with cluster switching

---

## Recent work (July 10–11, 2026)

- **✅ Benchmark runner moved to agenthost (.15) as systemd jobs — shipped + validated on hardware.** Spec/plan `docs/superpowers/{specs,plans}/2026-07-10-agenthost-eval-runner*`; 22 commits (167e9fc..HEAD), full suite green. Built via subagent-driven-development: fresh implementer + independent review per task, then a whole-branch review that caught a **feature-breaking Critical** (remote argv baked the manager's `/mnt/tank` output path, unmounted on the eval node) the 14 green per-task tests couldn't see. A second integration bug (`uvx: not found` — systemd-run transient units don't inherit `~/.local/bin` on PATH) surfaced only on the first real eval-node run. Both fixed.
  - **Design:** agent `job.*` capability (start/status/logs/cancel/result) hands each benchmark to `systemd-run`; the manager polls short cap calls, so runs outlive both the agent and the manager. `Node.role` (`gpu`|`eval`): an eval node may host Ollama but never vLLM/dgxrun (enforced in the deploy route + provisioner audit + dashboard pickers). `BenchmarkRun.runnerNodeId` is the provenance marker (throughput/TTFR aren't comparable across hosts).
  - **E2E verified 07-11:** throughput completed on agenthost with a real score (17.46 tok/s); **restart-survival passed** — a run outlived `docker compose restart server`, reattached, and its log offset advanced 654→1937 while the systemd unit kept running; remote cancel stops the unit.
  - **agenthost:** amd64, 16 cores, 30 GiB RAM, no CUDA GPU (Ollama on CPU for embeddings, per [[agenthost-igpu-not-worth-using]]). Onboarded with agent 0.5.773, role `eval`; its hand-installed Ollama left untouched.
- **✅ Agent 0.5.771 rolled to all 4 nodes (07-10), zero downtime.** Workers (.37/.38/.39) then head (.36), ~30 s each; the GLM-5.2 320K endpoint answered `200` on every poll and real inference was verified either side of the head roll. The head's reconcile re-announces `starting`; the health tick promotes it back to `running` within 10 s. **Sequencing matters:** the server rebuild must come *after* the agent roll — restarting the server drops every agent's WS, and the reconnect runs reconciliation. With the old agents that path could have failed a healthy rank and torn the cluster down.
- **Residual (unchanged):** `snapshotDgxrunLogs` / `captureCrashedDgxrunLogs` still use blocking `spawnSync`, so the event loop can still stall during a heavy load; the deploy now survives it rather than being torn down.

### Eval-runner follow-ups (found during 07-11 validation)

- [ ] **Eval job must forward `HF_TOKEN` (and benchmark env) to the systemd unit.** `buildSystemdRunArgv` sets only PATH+HOME, so gated datasets fail on the eval node — GPQA-Diamond (`Idavidrein/gpqa`, gated) died with `DatasetNotFoundError: gated` on 07-11. Forward `HF_TOKEN`/`LM_EVAL_VERSION`/etc via `-p Environment=` when set on the manager, or drop a token at `~/.cache/huggingface/token` on the eval node. Also needs an actual HF_TOKEN with GPQA gate access — none is configured (the compose passthrough is empty).
- [x] **✅ SWE-bench inference scaffold — SOLVED via mini-SWE-agent (2026-07-12).** The single-shot oracle patches that failed `git apply` were the problem; **mini-SWE-agent** (an agentic loop that edits in a real container) produces clean patches — 0 apply failures across the Verified 500. The one gotcha: `MSWEA_COST_TRACKING=ignore_errors` (litellm doesn't know glm-5.2 pricing) and `model.model_kwargs.timeout=1800` (long-context turns exceed the 600s default → retry-reprocess stall).
- [x] **✅ Comparable-benchmark blog runs — DONE (2026-07-12…14).** Results vs official GLM-5.2: **GPQA-Diamond 69.2%** (official 91.2, the one directly-comparable number), **AIME 2026 90.0%** (27/30; official 99.2), **SWE-bench Verified 73.0%** (365/500, ±~3.9%; official reports 62.1 on the harder *Pro*). Full writeup with methodology + caveats: [`docs/glm-5.2-benchmark-results.md`](glm-5.2-benchmark-results.md). Ran on the `c16-64k` concurrent recipe; SWE-bench via mini-SWE-agent. Key finding: the gaps are largely (not entirely) harness/scaffold/budget, not weights. *(LiveCodeBench not run — deprioritized.)*
- [ ] **Agent CPU% metric.** Report per-node CPU utilization alongside the existing GPU/VRAM/thermal metrics (`agent metrics.ts` / `sysinfo`), and surface it in the dashboard. Especially useful now that agenthost is a CPU-bound eval/embedding node with no GPU metrics to show. *(requested 2026-07-11)*
- [ ] **Remote cancel can leave lm-eval worker subprocesses briefly alive.** `job.cancel` → `systemctl stop` stops the unit and empties its cgroup, but lm-eval's own multiprocessing workers can escape the cgroup and linger for a few seconds (still hitting the endpoint) before clearing. Consider a firmer kill (unit `KillMode`, or a follow-up `pkill -f` scoped to the job) so cancel is immediate.
- [ ] **`nodeIds:"auto"` sizing to exactly 1 node → spurious 400.** Pre-existing latent bug (unrelated to this branch): when auto-resolve needs exactly one node, `nodeIds.length===1` makes `isCluster` false, so `headNodeId` falls back to the never-populated legacy `nodeId` var and the request 400s with "nodeId or nodeIds required". Fix the solo/cluster branch in `routes/deployments.ts`.
- [ ] Deferred minors from the whole-branch review: wrapper `cp` failure still writes `exit 0` (M3); boot `reconcileAction("missing")` lacks the `job.result` belt-and-braces the live poll path has (M4); `BenchmarkRun.jobUnit` is a dead column (M5); `POST /api/nodes` `role` lacks OpenAPI doc + a validation test (M6).
- **✅ Accuracy-eval benchmark kind (lm-eval) shipped + merged** — 12 presets, ephemeral `<think>`-strip proxy, two nullable columns, dashboard card/pill/compare. Built spec→plan→subagent-executed with a review gate per task; the final whole-branch review caught two bugs the per-task reviews structurally could not see.
- **✅ Validated E2E against a live endpoint (2026-07-10).** First real run exposed a bug no unit test could: lm-eval's `local-chat-completions` needs the **`api` extra** (tenacity/aiohttp) — the old assertion `/^lm-eval\[.+\]/` was too loose to catch its absence. Pinned `LM_EVAL_VERSION=0.4.12`. `acc-ifeval-quick` on GLM-5.2 @320K → **prompt_level_strict_acc 0.83** (±0.038, n=100), breakdown persisted, filter-agnostic parsing resolved `prompt_level_strict_acc,none` correctly.
  - **Caveat for anyone reading IFEval numbers:** `loose` came in *below* `strict` (0.82 vs 0.83), which looks impossible since loose tries 8 variants including the raw response. Cause: in `lm_eval/tasks/ifeval/utils.py` both strict and loose do `kwargs = {k:v for … if v}` then `build_description(**kwargs)` *independently*, and `instructions.py` refills 23 dropped params with an **unseeded `random.choice`**. So they can score against different constraints. IFEval is therefore **not bit-reproducible**; treat ±1 prompt as noise.
  - Runs execute as a `uvx` child of the server container **on the Pi**: concurrent `npm test`/builds slowed this run from 29.5 → 48 s/item, and any server restart kills it. Strongest argument yet for moving the runner to `agenthost`.
- **🔥 GLM-5.2 broke overnight, then came back better — full story in `docs/glm-5.2-256k-to-320k.md`.**
  - **Root cause of the outage:** `vm.min_free_kbytes` was set to 5 GiB in `scripts/dgx-node-prep.sh` + `/etc/sysctl.d/90-dgx-dcp.conf` on 07-07 to "guarantee capture headroom", but the *running* value had been 1 GiB. **The 07-09 17:50 reboot applied the drop-in for the first time**, arming two walls at once: vLLM's startup guard (`free ≥ gmu×total`, where ~7 GiB of the unified pool sits in CUDA contexts `/proc` never shows), and `torch.compile`'s 5.00 GiB capture allocation — which failed with 9.71 GiB "free" because the kernel withholds `min_free` from vLLM too. **It missed by 0.29 GiB.** The reserve added to protect the capture is what prevented it. Fixed → **1 GiB**, verified to survive a reboot.
  - **`gmu` 0.90 → 0.88.** A later probe caught startup free at **109.41 GiB vs the 109.46 required** — 0.90 was clearing the guard by under 100 MiB, i.e. by luck. `gmu` does not cap allocation here (KV is pinned; `GPU KV cache size` is identical at 0.88 and 0.90), so it's free margin.
  - **Agent 0.5.770 — the manager was killing its own healthy deploys.** `dgxrun-dropcache.ts` ran blocking `spawnSync("sync; …drop_caches")` every 500 ms; under a ~400 GB NFS weight stream a single `sync` takes seconds (kernel log: 1–7 s gaps), parking the agent's event loop → missed heartbeats **and** a starved `docker inspect`. `inspectDgxrunContainer()` then returned `null` on *any* non-zero exit, and `dgxrun-metrics.ts` read `null` as "container gone" → tore down all four ranks of a deploy that had just finished `torch.compile`. Fixed: async drop + in-flight guard; pure `classifyDockerInspect()` → `found | absent | unknown` (`unknown` skips, `absent` must repeat). 16 tests. **Residual:** the wedge is reduced, not gone — inspect/log-snapshot are still `spawnSync`, so staleness still fires during load; the deploy now *survives* it.
  - **✅ 320K context shipped at DCP2, no image rebuild, no speed traded.** The limiter was never the KV cache or the DCP degree: the b12x sparse-indexer **prewarm** allocates `fold_values+fold_indices = (max_num_batched_tokens × total_slices, topk) × 8 B`, i.e. **workspace ∝ mnbt × max_model_len** — exactly 5.00 GiB @ 2048×256K and 10.00 GiB @ 2048×512K (both observed; the model predicts 5.10). **`--max-num-batched-tokens 2048 → 1024`** halves it, paying for KV 7.00 → 8.50 GiB (333,440 tokens). Validated with a **300,036-token needle prompt** (92% of the window, `finish=stop`, 893 tok/s prefill). Cost ~11% prefill on short prompts; **decode unchanged** (23.8 vs 22.3–23.2). cudagraph FULL + MTP drafter retained.
  - **DCP4 is a dead end on this image** — 3 attempts all OOM-killed on the same prewarm *before* KV allocation, so the multiplier was never measured; `vllm.envs` in `b12x-dcp:probe` contains **no `VLLM_DCP_*` at all (hence the long-standing `Unknown vLLM environment variable: VLLM_DCP_SHARD_DRAFT` warnings). A planned image rebuild for `VLLM_TRITON_MLA_SPARSE_*_CHUNK_SIZE` was abandoned on inspection: those knobs bound the **Triton sparse MLA** backend, not the `B12X_MLA_SPARSE` one the recipe runs.
  - **Reboot ≠ fix.** Fresh-boot free is 116.0–116.4 GiB, *lower* than the 117.5 GiB after a clean teardown. Fragmentation was never the problem.
  - **Two recipes now, one trade.** `@dgxrun/glm-5.2-quanttrio-unpruned-dcp2` stays the **SPEED** daily driver (256K, mnbt 2048, prefill ~698 tok/s); `…-dcp2-320k` is the **CONTEXT** sibling (320K, mnbt 1024, prefill 619.5 tok/s). Decode identical. Everything else — DCP2, tp=4, cudagraph FULL, MTP drafter — is the same in both.
- **🖥️ New eval host `agenthost` @ `192.168.44.15`** — x86_64, 16 cores, 31 GiB RAM, 874 GB free, **no GPU**, passwordless SSH + sudo. Docker **absent**; Python 3.12.3. This is the dedicated benchmark host the accuracy-eval design anticipated (the runner location was deliberately left un-hardwired). GPU nodes stay reserved for model hosting.

## Recent work (July 7–8, 2026)

- **GLM-5.2 unpruned at 256K, 25–30 tok/s — the DCP long-context daily driver shipped.** The DCP-stack arm64 image + the unpruned QuantTrio Int4-Int8Mix on **DCP2 serve at 262144 (256K)**; decode **25.2 tok/s code / 30.5 structured** via a separate MTP drafter (`GLM-5.2-MTP-INT4-aligned`, acceptance 3.5–3.9) + FULL cudagraph. Three load blockers fixed: **earlyoom** was SIGKILLing NetworkManager mid-load (the whole "Marlin hang" was a red herring — now disabled in node-prep), a too-conservative KV profile (`--kv-cache-memory-bytes 7GB`), and an NFS-shared JIT-cache race (moved container-local). Recipe `@dgxrun/glm-5.2-quanttrio-unpruned-dcp2`. Memory `glm52-unpruned-256k-working`.
- **Agent drop-loop — self-contained DCP deploys.** The drop-caches-during-load loop (keeps CUDA-graph-capture headroom in the GB10 unified pool) moved from a scratch orchestrator into the agent `dgxrun` path (start on launch → stop on API-ready → 20-min backstop). Any deploy, dashboard included, now self-covers.
- **Monotonic deploy status.** Extracted the log→phase classifier to a tested `runtime/deploy-phase` module, added the missing **`compiling`** phase, and made reporting **forward-only** — a late "Prefetching checkpoint files" line no longer flips loading→downloading, and a reconnect no longer snaps back to `starting`. Fixes the "downloading while it's compiling" confusion (advances the Phase 2 deploy-status-accuracy item).
- **Agent roll 0.5.738 → 0.5.756** on all 4 GB10 nodes (drop-loop + monotonic status), one-at-a-time worker→head, deploy served throughout — manual SSH swap, not `cmd:update`. Gotcha logged: preserve `/opt/dgx-agent/node-id` or a node without an env `NODE_ID` falls back to the spent join token.
- **buildx installed on the Pi manager** (`apt docker-buildx` 0.20.1 + binfmt QEMU) — clears the long-standing "no buildx on the raspi" blocker; `build-agent-bundles.sh` works for both arches now; built the **amd64 0.5.756 bundle** (ready for aihost01). Agent bundles + `recipes/` now **mounted live** into the server (a rebuilt bundle/recipe is served without re-baking the image).
- **Claude Code 1M-context cap** — the claude-launch snippet reads `max_model_len` from `/v1/models` and emits `CLAUDE_CODE_DISABLE_1M_CONTEXT=1` for sub-1M deploys (Claude Code assumed 1M for `glm-5.2` and overflowed 256K; verified → ~200K).
- **Dashboard: cluster node picker sorted by name** so node 1 (dgx-spark-01) is the default head; **recipe cleanup** to the two keepers + the custom-image overlay **vendored in-repo** (`scripts/glm52-overlay`) for repo-reproducible builds.

## Recent work (July 2026)

- **GLM-5.2 multi-node via the new `dgxrun` runner** — the Ray executor is genuinely broken on this vLLM build (`AttributeError: 'ShmRingBuffer' object has no attribute 'buf'`, even with 61 GB /dev/shm), so sparkrun (Ray-only for multi-node) cannot serve it. Built **dgxrun**: our own agent-per-node `mp`-executor fan-out driven by the manager (`runner: dgxrun` in the recipe, driven by `recipeYaml`/`recipePath`). GLM-5.2-AWQ-INT4 (15pct) validated end-to-end: 85K context @ 0.88 util, **23.7 tok/s warm** with MTP speculative decoding + dual-rail RoCE RDMA (NET/IB). sm12x DSA Triton fallback kernels JIT-compile on first inference (~6 min, memory-heavy, persist to NFS cache). See spec `docs/superpowers/plans/2026-07-04-dgxrun-runner.md` + memory `glm52-shm-and-jit-findings`.
- **GLM-5.2 decode speed investigation — resolved as a benchmark artifact (2026-07-06).** The `pp=2048/tg=256` llama-benchy number (~20 tok/s) is misleadingly low: MTP drafts poorly on generic prose (acceptance ~2.6). On **real work (instruction/reasoning/agentic-coding) decode is ~25–31 tok/s** (acceptance 3.1–3.3), confirmed live — matching the community's "flat 30–35, good for agent coding." Established we're **already on the current community stack** (vLLM `ab6660699` + b12x 0.23.0 + the sm12x DSA kernels = CosmicRaisins `bootstrap.sh` `VLLM_REF`), decode is **memory-bandwidth-bound** (MoE GEMM ~74% of the ~273 GB/s roofline; our 4-node NCCL test: ~19–23 GB/s busbw, comms ~6% of a step), and **every cheap/kernel/firmware lever is a dead end for speed** (draft-TP, num_spec, async-scheduling, NCCL channels all flat/worse at c=1; a faster FP4 kernel gave the community zero decode gain). No arm64 rebuild / no unpruned model needed **for speed**. Memory `glm52-decode-workload-dependent`.
- **✅ SHIPPED 2026-07-08 at 256K (unpruned QuantTrio + DCP2, 25–30 tok/s) — see "Recent work (July 7–8)".** The DCP-stack rebuild + unpruned model landed.
  **UPDATE 2026-07-10 — 320K SHIPPED at DCP2, no image rebuild. The reason is now measured.**
  The limiter is the **sparse-MLA attention workspace**: one allocation of **~20.0 KB/token** (exactly
  `Tried to allocate 5.00 GiB` at `max_model_len 262144`, `10.00 GiB` at `524288`). It is linear in context and is
  **unaffected by `--kv-cache-memory-bytes` and by `--decode-context-parallel-size`**. It OOM-killed 288K, 320K,
  and all three DCP4 attempts (incl. DCP4+PIECEWISE@512K). The config-only lever named below
  (`VLLM_TRITON_MLA_SPARSE_{QUERY,TOPK}_CHUNK_SIZE`) **does not exist in `vllm-node-tf5-glm52-b12x-dcp:probe`** —
  its `vllm.envs` exposes only `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` and `VLLM_USE_B12X_SPARSE_INDEXER`, and **no
  `VLLM_DCP_*` at all** (hence the `Unknown vLLM environment variable: VLLM_DCP_SHARD_DRAFT` warnings — DCP4 is
  effectively unsupported in this build). **Past 256K needs an image rebuild, not a recipe tweak.** Original notes below.
- **Next lever = CONTEXT, for agentic-coding daily-driver use** (opencode / Claude Code need big windows). **Measured 2026-07-06:** reducing the sparse-MLA/indexer workspace chunks (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`, `VLLM_TRITON_MLA_SPARSE_{QUERY,TOPK}_CHUNK_SIZE=128/256`) removes the head's capture/JIT spike and lifts the 15pct **no-DCP ceiling from ~85K to ~159K** (KV-limited @ 0.88; 128K served, 160K exceeded the KV pool) — **config-only, no rebuild**. Shipped as recipe `@dgxrun/glm-5.2-awq-15pct-144k`. (Head-padding was already active via the b12x path — not the lever; the chunk sizes are.) For **>159K** (~200K Claude-parity, or 320K/640K): **DCP** (decode-context-parallel) shards MLA KV across ranks (context × N; decode unaffected ~22; prefill ~linear) — the **DCP-stack arm64 rebuild** (local-inference-lab/vllm `codex/dcp-*` branch + PR#72 + `VLLM_USE_V2_MODEL_RUNNER=1`); unpruned+DCP2 = 320K + better quality. A *context* project, not speed. Pruned-vs-unpruned decided by a quality eval (tool-eval already 100/★★★★★; IFEval TBD).
- **Power control** — agent-primary reboot/shutdown/suspend with SSH fallback (version-gated), force variants for hung nodes, Wake-on-LAN arming before shutdown; compose switched to host networking so WoL packets reach the LAN (memory `wol-broken-docker-bridge`)
- **Node offboarding** — graceful + force offboarding with a 30s timeout, complete FK cleanup on delete
- **Ollama governance** — startup firewall restricts `:11434` to manager + loopback; install without autostart; on-demand service start for Ollama deploys; firewall state reported via self-audit
- **`maxoutmem` recipe flag** — reclaim node memory (stop gdm) before a deploy that needs the full unified-memory budget
- **Metrics fixes** — dgxrun tps folded into node metrics; `vramTotal` now self-heals from every tick via `os.totalmem()` (was a register-only `free -m` parse that returned 0 on a re-onboarded node)
- **Agent v2 Phase 1 shipped** — incident-response core (fork-free diag, audited exec, rich streaming metrics, capability registry, `POST /api/nodes/:id/{diag,exec}` + audit table); deployed on all 4 nodes (see Phase 3.7)
- **Agent v2 Phase 2 shipped — robust self-update** — detached non-blocking updater with atomic swap + health-check + auto-rollback + truthful outcome reporting; fixes the blocking `cmd:update` that wedged `.36` twice. Deployed (0.5.738)
- **Heartbeat-staleness sweep shipped** — node `status` reconciles against `lastSeen` every 10 s, so a dead/half-open node reads offline in ~40 s instead of 76 min; self-heal + agents-map restore on recovery. Live-verified on `.39`
- **`@dgxrun` recipe catalog** — dgxrun recipes now live in-repo (`recipes/dgxrun/*.yaml`), auto-discovered and grouped in the deploy dropdown as `@dgxrun/…` alongside `@sparkrun/…`, so GLM-5.2 is one click (module-relative recipes dir; 404 on a missing recipe)
- **Head-node select at launch** — crown-toggle to pick which node heads a TP=4 cluster deploy (`nodeIds` sent head-first), badged in the deployments list — no more guessing which node becomes head
- **Deploy-status accuracy** — `running` now gates on an `apiReady` probe + reports the `starting→running` transition past the VRAM throttle (fixes serving multi-node deploys stuck reading `starting`)

## Recent work (May–June 2026)

- **Sparkrun deploy backend** — replaced the eugr `run-recipe.sh` path with [sparkrun](https://github.com/spark-arena/sparkrun) (agent-side; head-node runs `sparkrun run`). Recipe catalog from `sparkrun list` registries; inline-`recipeYaml` deploy API (remote-dev, no cluster-fs access); OpenAPI 3 spec + Swagger UI (`/api/openapi.json`, `/api/docs`); live vLLM model-loading logs via a `sparkrun logs` follower; provisioner installs + sets up sparkrun
- **Nemotron-3-Ultra NVFP4 TP=4** — 550B-A55B served across 4 DGX Spark nodes (Ray, engine-isolated vLLM container); MTP speculative decoding enabled via per-MoE-type backend selection
- **Metrics retention** — MetricSnapshot pruning (`METRIC_RETENTION_DAYS`, default 7d) + `(nodeId, timestamp DESC)` index
- **Inference-variant selector** — choose an inference template per recipe on deploy/restart
- **Log catch-up** — deployment + fine-tune logs reconcile on tab-visible / SSE reconnect
- **Node management-IP override** — `NODE_ADVERTISE_IP` for correct multi-node binding
- **Verboseness eval** — thinking-mode response-length probe

---

## Status Matrix

| Area | Server | Agent | Dashboard | Database |
|------|--------|-------|-----------|----------|
| Nodes & Metrics | ✅ | ✅ | ✅ | ✅ |
| Deployments | ✅ | ✅ | ✅ | ✅ |
| Multi-node (dgxrun mp) | ✅ | ✅ | ✅ | ✅ |
| Power Control (reboot/WoL) | ✅ | ✅ | ✅ | ✅ |
| Agent v2 (mgmt plane) | ✅† | ✅† | 🚧 | ✅† |
| Load Balancer | ✅* | — | placeholder | ✅ |
| Models | ✅ | — | placeholder | ✅ |
| Fine-Tuning (single) | ✅ | ✅ | ✅ | ✅ |
| Fine-Tuning (multi-node) | ✅ | ✅ | ✅ | ✅ |
| Merge & Deploy | ✅ | ✅ | ✅ | ✅ |
| Training Metrics | ✅ | ✅ | ✅ | ✅ |
| Agent Bootstrap | ✅ | ✅ | ✅ | ✅ |
| HTTP Agent Updates | ✅ | ✅ | ✅ | ✅ |
| Settings | ✅ | — | ✅ | ✅ |
| SSH Key Exchange | — | — | — | — |
| Heterogeneous Hardware | ✅ | ✅ | ✅ | ✅ |
| Consumer-GPU Recipes | — | — | — | — |
| Datasets | ✅ | — | ✅ | ✅ |
| Evaluation | ✅ | — | ✅ (benchmarks) / partial (eval) | ✅ |
| Benchmarks (throughput / tool-eval / **accuracy**) | ✅ | — | ✅ | ✅ |
| Resume from Checkpoint | ✅ | ✅ | ✅ | ✅ |
| Auth & RBAC | — | — | — | — |
| Multi-Cluster | — | — | — | — |

*\*Load Balancer: rules/endpoints API complete; inference proxy implemented but not mounted; dashboard UI pending.*

*†Agent v2: Phase 1 (fork-free diag / audited exec / rich metrics / capability registry) + Phase 2 (robust self-update) + the heartbeat-staleness sweep are shipped & deployed; declarative provision/restore, image transfer, and Phase 4 (dgxrun on the registry) pending. Dashboard surface for diag/exec still minimal.*

---

*Last updated: July 10, 2026*
