# DGX Manager Roadmap

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

### Known manager bugs (found 2026-07-09/10, all still open)

Each of these cost real debugging time during the GLM-5.2 incident. Ordered by blast radius.

- [ ] **`DELETE /api/deployments/:id` right after `POST` races the launch → orphaned containers.** The agent can process `cmd:undeploy` *before* the queued `cmd:deploy`; undeploy finds nothing, then the launch proceeds. The record walks `removing → stopped` while ~100 GB/node of containers keep running. Symptom: a later deploy is rejected with `Not enough VRAM … only 26 GB free of 122 GB — held by: @dgxrun/…` while `/api/deployments` shows nothing active. A second `DELETE` does **not** help — `routes/deployments.ts` treats `stopped` as terminal. *Fix: tombstone undeployed deploymentIds in the agent so a late `cmd:deploy` for that id is refused; and/or defer undeploy until the deployment leaves `pending`.* Workaround: never stop a dgxrun deploy before it reaches `loading`; clean up with `docker rm -f dgxrun_<deploymentId>` via `POST /api/nodes/:id/exec`.
- [ ] **`POST /api/deployments/:id/restart` ignores the dgxrun runner** and falls back to sparkrun → `Sparkrun launch failed with exit code 1`, launching nothing. Restart is effectively broken for every dgxrun deploy.
- [ ] **A vLLM startup failure never reaches `deployment.error`.** The `ValueError: Free memory on device …` that killed two deploys left `status=stopped, error=null` — a failed deploy is indistinguishable from a stopped one in the UI. This is why the incident began as a mystery instead of a stack trace. *Fix: persist the captured container log's first error line on the failure path (the agent already has `firstErrorLine`).*
- [ ] **`maxoutmem: true` is a silent no-op for dgxrun recipes.** It *is* attempted, then swallowed by the `.catch()` in `routes/deployments.ts` because `maxOutMemoryForDeploy` resolves the recipe from the head node's **sparkrun** cache, where a dgxrun recipe never lives. Harmless today (gdm is already `inactive`), but the flag lies. *Fix: honour it for dgxrun, or drop it from the dgxrun recipes.*

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
  - [ ] **NOT YET VALIDATED END-TO-END.** No real `uvx lm_eval` run has happened. Start with `acc-ifeval-quick` against the live GLM-5.2. Task ids + primary metrics are pinned by intent, not by observation.
  - [ ] Pin `LM_EVAL_VERSION` — it is **not plumbed into `docker-compose.yml`**, so lm-eval floats to latest PyPI, which is exactly where task-id drift bites.
  - [ ] Warm the `uvx` cache for lm-eval in `Dockerfile.server` (only llama-benchy is warmed today, so the first run downloads torch before evaluating anything).
  - [ ] GPQA needs `HF_TOKEN` with access to the gated `Idavidrein/gpqa` (the env var *is* already plumbed through compose).
  - [ ] **Accuracy evals are model-bound, not runner-bound.** First real run (2026-07-10): IFEval `--limit 100` progresses at **~1 prompt/minute** → ~100 min. Cause: the serving recipe pins **`--max-num-seqs 1`** (chosen for decode speed), so there is no batching, and `buildLmEvalArgs` hardcodes `num_concurrent=1` to match. **Moving the runner to `agenthost` will not speed this up.** The lever is a *separate eval deployment* — higher `--max-num-seqs`, no MTP drafter, smaller context — plus making `num_concurrent` a preset field. Consider an `eval` recipe variant alongside SPEED/CONTEXT.
  - [ ] **Move the lm-eval runner to `agenthost` (192.168.44.15).** Today `orchestrator.ts` spawns `uvx lm_eval` on the Pi. The endpoint URL is already a parameter and `LM_EVAL_SPEC` is already env-pinned, so this is a *deployment* change, not a rewrite. Open design question: SSH-exec to `.15`, or onboard it as an **agent-only node** (amd64 bundle 0.5.770 exists) and drive it through the capability registry / `POST /api/nodes/:id/exec` — the latter reuses auth, audit and streaming, and avoids a second remote-exec path. It must **not** become a model-hosting node.
- [ ] **SWE-bench (and other agentic-coding evals) on `agenthost`** — the natural next eval after lm-eval, and the one that actually measures GLM-5.2 as a coding daily driver. Prereqs: **install Docker on `.15`** (absent today; SWE-bench runs each task in a container) and budget disk (874 GB free is enough). Would slot in as a fourth benchmark `kind`, or as its own runner reusing the `BenchmarkRun` row + `accuracyMetrics` JSON breakdown.
- [ ] Track quality metrics over time with per-model and per-run history
- [ ] Compare base models against fine-tuned variants in the dashboard (currently only per-job chart)
- [ ] Dashboard views for benchmark results and regression detection
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

## Recent work (July 9–10, 2026)

- **✅ Accuracy-eval benchmark kind (lm-eval) shipped + merged** — 12 presets, ephemeral `<think>`-strip proxy, two nullable columns, dashboard card/pill/compare. Built spec→plan→subagent-executed with a review gate per task; the final whole-branch review caught two bugs the per-task reviews structurally could not see. **Never yet run against a live endpoint** — that's the top follow-up.
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
