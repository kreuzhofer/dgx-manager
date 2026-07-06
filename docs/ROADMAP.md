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

### Deploy config override UI

- [ ] Deploy form (and fine-tune `Deploy` action) should expose recipe-level config overrides — at minimum `max_model_len`, `gpu_memory_utilization`, `tensor_parallel`. Today recipe defaults (or worse, `recipe.yaml`'s `deploy.max_model_len`) win silently and the only way to override is hand-crafting a `POST /api/finetune/:id/deploy` body with `config: {maxModelLen: …}`. Concrete miss case: the qwen3.6-27b training recipe has `deploy.max_model_len: 4096` as a smoke-test leftover, which shadows `inference.yaml`/`inference-fp8.yaml`'s 128000 default, so every fine-tune deploy lands at 4k context without warning. UI should: (a) show the effective value with provenance (recipe.yaml > inference[ -fp8].yaml > server default), (b) let the user override per-deploy, (c) persist the override on the deployment record so restart preserves it.

### Deploy status accuracy

- [ ] Deployment `status: "running"` should mean vLLM is actually serving, not just that the container started. Today the manager flips to `running` when the agent reports container start (often within 60s), but for a 27B model on 2 nodes, vLLM doesn't bind port 8000 until safetensors load + Ray init + cudagraph capture finish (~5–6 min). A scripted "wait for ready" loop hits a connection-refused window even though the dashboard says "running". Options: add a `loading` sub-state until the `/v1/models` probe succeeds; or have the agent stream readiness events back over the WS and update the deployment row when the API is bound. Either way the UI should show "loading" with progress, not "running". *(June 2026 update: live vLLM model-loading logs now stream via a `sparkrun logs` follower, but `running` still fires at serve-command launch, not at `/v1/models` readiness. Also found: a crash-looping failed deploy keeps re-downloading on each docker restart while the follower narrates progress — so a failed deploy can look alive, and status can stay stale after it stops. Tracked as a status-accuracy follow-up.)* *(July 2026 update: the agent now gates `running` on an `apiReady` probe — dgxrun head checks `/metrics`, sparkrun checks `/v1/models` — AND a `shouldReportStatus` fix reports the `starting→running` transition past the VRAM throttle, which previously dropped it and left a serving multi-node deploy reading `starting` forever. So `running` now means the API answered. Remaining: surface an explicit `loading` sub-state + progress in the UI, and detect the crash-loop-redownload case.)*

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
- [ ] `POST /api/nodes/update-agent-all` — bulk update endpoint that fans out to every online node, optionally serial (default) or parallel, and streams per-node status. Today the dashboard / operator has to iterate per-node, which is awkward after a bundle rebuild.
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
- [ ] **Accuracy-eval benchmark kind (lm-eval-harness) as a first-class GUI option** — add an `accuracy`/`lm-eval` `kind` alongside the shipped `throughput` + `tool-eval` kinds (same `POST /api/benchmarks`, deploy dropdown, and `/benchmarks/compare` view). Design decided 2026-07-06:
  - **Curate, don't wrap all of lm-eval** (4–6 presets): **IFEval** (instruction adherence — the *known* expert-prune failure mode), **MMLU-Pro** + **GPQA** (knowledge tail), **Aider-polyglot** / LiveCodeBench (agentic coding), + the existing SQL eval.
  - **API-mode** against the deployment `/v1` endpoint (scoring is rule-based, no GPU) — run via lm-eval **dispatched to a node** like fine-tune jobs, to keep torch/heavy deps off the Pi; stream results back. (Pi *can* run it — `python3-venv` installed 2026-07-06 — but node-dispatch is the scalable path.)
  - **Reasoning-model handling:** per-preset toggle to strip/parse GLM-5.2 thinking tokens (scores tank otherwise).
  - **Result schema:** reuse the `toolEvalCategories` JSON pattern for per-task/per-subject breakdown — no new columns.
  - **Motivation:** makes the manual "deploy A → eval → deploy B → eval → compare" one-click + dashboard-comparable — directly serves the **15pct-vs-unpruned** prune-quality decision (in progress) and **fine-tune regression** checks. Validate manually first (IFEval on the 15pct), then spec→plan→build.
- [ ] Track quality metrics over time with per-model and per-run history
- [ ] Compare base models against fine-tuned variants in the dashboard (currently only per-job chart)
- [ ] Dashboard views for benchmark results and regression detection
- [ ] Upload fine-tuned models to HuggingFace (requires HF_TOKEN)
- [ ] Multimodal fine-tuning: Gemma 4 as a visual judge (image+text training)
- [ ] **External-facing write-up (blog / LinkedIn post)** — distinct from `docs/gemma4-fine-tuning-on-dgx-spark.md` (which is the deep technical reference). Audience-focused "so what" framing: what the DGX Manager project says about the author's skills to a hiring manager / CTO / practitioner network. Pending audience + platform decisions.

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

## Recent work (July 2026)

- **GLM-5.2 multi-node via the new `dgxrun` runner** — the Ray executor is genuinely broken on this vLLM build (`AttributeError: 'ShmRingBuffer' object has no attribute 'buf'`, even with 61 GB /dev/shm), so sparkrun (Ray-only for multi-node) cannot serve it. Built **dgxrun**: our own agent-per-node `mp`-executor fan-out driven by the manager (`runner: dgxrun` in the recipe, driven by `recipeYaml`/`recipePath`). GLM-5.2-AWQ-INT4 (15pct) validated end-to-end: 85K context @ 0.88 util, **23.7 tok/s warm** with MTP speculative decoding + dual-rail RoCE RDMA (NET/IB). sm12x DSA Triton fallback kernels JIT-compile on first inference (~6 min, memory-heavy, persist to NFS cache). See spec `docs/superpowers/plans/2026-07-04-dgxrun-runner.md` + memory `glm52-shm-and-jit-findings`.
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
| Benchmarks | ✅ | — | ✅ | ✅ |
| Resume from Checkpoint | ✅ | ✅ | ✅ | ✅ |
| Auth & RBAC | — | — | — | — |
| Multi-Cluster | — | — | — | — |

*\*Load Balancer: rules/endpoints API complete; inference proxy implemented but not mounted; dashboard UI pending.*

*†Agent v2: Phase 1 (fork-free diag / audited exec / rich metrics / capability registry) + Phase 2 (robust self-update) + the heartbeat-staleness sweep are shipped & deployed; declarative provision/restore, image transfer, and Phase 4 (dgxrun on the registry) pending. Dashboard surface for diag/exec still minimal.*

---

*Last updated: July 6, 2026*
