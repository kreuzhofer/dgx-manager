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

### Monitoring & Observability

- Phase-aware progress tracking: container → downloading → loading → tokenizing → training → eval → saving
- Real-time loss curve visualization (SVG chart, live-updating via SSE)
- Training metrics persisted to DB (TrainingMetric table: step, loss, lr, evalLoss)
- `[TRAIN]` and `[EVAL]` callbacks for explicit progress reporting through Docker pipes
- Smoothed ETA estimation (20-sample rolling average of iteration speed)
- Log file persistence: `train.log` written via Python Tee, survives page refresh and agent restart
- Agent reattaches to running training containers after restart (tails train.log)
- Deployment log persistence to NFS files

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

The current implementation assumes a shared NFS mount (`/mnt/tank`) across all nodes for models, datasets, training scripts, and outputs. Not every deployment will have NFS. Future iterations should:

- Make all storage paths configurable (no hardcoded `/mnt/tank` or `/workspace` assumptions)
- Support alternative shared storage backends (S3/MinIO, Ceph, host-local with rsync)
- Allow dataset and model transfer via the agent when shared storage is unavailable
- Provide a setup wizard or configuration file for storage topology

## Phase 4: Dataset Management

**Goal:** First-class support for training data throughout the fine-tuning workflow.

- Upload and manage training datasets through the dashboard
- Dataset versioning and lineage tracking
- Format validation and preview (JSONL, Alpaca, ShareGPT, etc.)
- Direct integration with fine-tuning jobs from Phase 3

## Phase 5: Evaluation & Benchmarks 🔜

**Goal:** Measure and track model quality across deployments and fine-tuning runs.

- SQL evaluation script — implemented and validated
  - Gemma 4 E2B: base 4% → fine-tuned 22% exact-match accuracy on SQL generation (5.5x improvement)
  - Key lesson: eval prompt format must match training format (no system prompt, same context layout)
  - Better SQL normalization: handles quote style, chat template artifacts, whitespace
- Run evaluation suites (lm-eval-harness, custom benchmarks) against deployed models
- Track quality metrics over time with per-model and per-run history
- Compare base models against fine-tuned variants
- Dashboard views for benchmark results and regression detection

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
| Fine-Tuning | ✅ | ✅ | ✅ | ✅ |
| Merge & Deploy | ✅ | ✅ | ✅ | ✅ |
| Training Metrics | ✅ | ✅ | ✅ | ✅ |
| Datasets | — | — | — | — |
| Evaluation | partial | — | — | — |
| Auth & RBAC | — | — | — | — |
| Multi-Cluster | — | — | — | — |

---

*Last updated: April 9, 2026*
