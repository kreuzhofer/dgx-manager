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

- Two inference runtimes: vLLM (container-based, YAML recipe-driven) and Ollama (native)
- Solo and multi-node cluster deployments (tensor parallelism × pipeline parallelism via Ray)
- VRAM admission control with safety margins and port conflict detection
- Deployment persistence across agent restarts
- Real-time deployment log streaming and full status lifecycle
- Recipe auto-discovery from the spark-vllm-docker repository
- Server-side load balancer: rules, endpoints, round-robin/first-available strategy, inference proxy
- Dashboard: Deployment creation (runtime toggle, node/recipe selection), log viewer, stop/restart controls, cluster node visualization

### Remaining UI work

The server APIs for these features are complete, but the dashboard pages are still placeholders:

- **Load Balancer UI** — Rule management, endpoint assignment, strategy configuration
- **Models UI** — Model registry browser and management

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

- [ ] **Consumer-GPU recipe repo** — a second vLLM recipe repo (working title `consumer-gpu-vllm-docker`) with curated recipes targeting 32 GB single-GPU amd64 hosts (Nemotron-3-Nano-NVFP4, Qwen3-Coder-Next int4, GLM-4.7-Flash-AWQ, merged finetunes). Slim `Dockerfile` without fastsafetensors/cubin patches, single-node-only `run-recipe.sh`, no cluster plumbing. `VLLM_REPO_PATH` baked per-arch into the agent's systemd unit at install time so each node points at the appropriate repo.
- [ ] **VRAM admission guard** — compare `recipe.vram_required` against `node.vramTotal` at deploy time and reject with a clear error, to prevent wasting a build cycle on models that can't fit.
- [ ] **Heterogeneous cluster guard** — scheduler refuses to form multi-node training/vLLM clusters that mix arches or GPU classes.

## Phase 4: Dataset Management (in progress)

**Goal:** First-class support for training data throughout the fine-tuning workflow.

- Upload datasets via dashboard (multipart file upload to shared storage)
- Register existing NFS paths or HuggingFace dataset IDs
- Auto-detect format from first JSON line (ShareGPT, OpenAI, QA, Instruct)
- Dataset preview (first N rows displayed in dashboard)
- Dataset picker in fine-tune job creation (replaces raw text input)
- [ ] Dataset versioning and lineage tracking
- [ ] CSV/Parquet format support

## Phase 5: Evaluation & Benchmarks 🔜

**Goal:** Measure and track model quality across deployments and fine-tuning runs.

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
- [ ] Run evaluation suites (lm-eval-harness, custom benchmarks) against deployed models
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

## Status Matrix

| Area | Server | Agent | Dashboard | Database |
|------|--------|-------|-----------|----------|
| Nodes & Metrics | ✅ | ✅ | ✅ | ✅ |
| Deployments | ✅ | ✅ | ✅ | ✅ |
| Load Balancer | ✅ | — | placeholder | ✅ |
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
| Evaluation | ✅ | — | partial (in-chart) | ✅ |
| Resume from Checkpoint | ✅ | ✅ | ✅ | ✅ |
| Auth & RBAC | — | — | — | — |
| Multi-Cluster | — | — | — | — |

---

*Last updated: April 14, 2026*
