# Design: README + Documentation Refresh

**Date:** 2026-06-11
**Status:** Approved (pending spec review)
**Author:** Daniel Kreuzhofer (with Claude)

## Problem

`README.md` is a Phase-1/2 snapshot. It documents node metrics and basic vLLM
deployment but omits almost everything the project now does, and parts of it are
factually wrong. A new GitHub visitor reading it today would badly under-estimate
the system. `docs/ROADMAP.md` is ~2 months stale (last updated April 14, 2026).

### Verified gaps (README)

- **Missing capabilities:** full fine-tuning pipeline (LoRA train → merge → deploy,
  multi-node DeepSpeed ZeRO-3, resume-from-checkpoint), benchmarking (llama-benchy
  runner + 3 dashboard pages), Ollama runtime + catalog, multi-node tensor/pipeline-
  parallel vLLM clusters over Ray (Nemotron-3-Ultra 550B NVFP4 on 4 nodes),
  join-token agent bootstrap, HTTP agent auto-update, heterogeneous arm64+amd64,
  datasets, evaluation, load-balanced inference proxy.
- **Factually stale:** "Quick Start" and "Running the Agent" tell you to run
  `npm run dev` / `npm run dev -w packages/agent`. The canonical path (per
  `CLAUDE.md`) is **Docker Compose**, and nodes are onboarded via **SSH provisioning
  or a join-token install script** — not a manual agent process.
- **Incomplete API table:** lists 7 of 13 route groups (missing `training-recipes`,
  `tokens`, `settings`, `ollama-catalog`, `agent` bundle, `datasets`, `benchmarks`).

### Verified facts to preserve accuracy (checked 2026-06-11)

- Dashboard **Models** and **Load Balancer** pages are still **stubs** (placeholder
  text). The load-balancer *server API + inference proxy* are complete; the **UI is
  pending**. Docs must say exactly that — do not claim a Models/LB UI.
- **Benchmarks** (`/benchmarks`, `/benchmarks/compare`, `/benchmarks/[id]`, 282-line
  page) and **Datasets** (493-line page) pages are real and shipped.
- Benchmark presets: `quick-smoke`, `chat-short`, `chat-long`, `code-32k`,
  `throughput` (llama-benchy runner).
- 13 server route groups mounted under `/api`: `nodes`, `models`, `deployments`,
  `finetune`, `lb`, `recipes`, `training-recipes`, `tokens`, `settings`,
  `ollama-catalog`, `agent`, `datasets`, `benchmarks`.

## Goal & Audience

Make a new GitHub visitor understand, in under a minute, the full capability surface
and the engineering behind it. **Primary audience: evaluators** (hiring managers,
CTOs, practitioners browsing the repo) — capability-showcase framing. Operators who
want to run it are served by a dedicated guide, linked at the very top.

## Scope

Three documentation artifacts. **No code changes. No new feature work.**

1. **Rewrite `README.md`** — capability showcase + architecture + feature tour +
   reference. All setup/operator content removed (moved to artifact 2).
2. **New `docs/SELF-HOSTING.md`** — the operator/setup guide. Linked from the very
   top of the README.
3. **Refresh `docs/ROADMAP.md`** — factual catch-up; keep its existing structure.

### Out of scope

- New feature work or code changes of any kind.
- Capturing real screenshots — the README ships with placeholder image links; the
  user supplies images in a later pass (see Screenshot Manifest).
- Forward-looking ROADMAP phases (6 Auth, 7 Multi-cluster) — left as-is.
- Flipping Models / Load Balancer UI status (still genuinely stubs).

## Artifact 1 — README.md structure

Capability-showcase framing, **skimmable** (tight sections that link out to deep-dive
docs rather than inlining them). Setup lives in the self-hosting guide.

1. **Title + one-liner + "what it is"** — self-hosted control plane for a DGX Spark
   cluster: provision over SSH/token, deploy + load-balance inference, fine-tune,
   benchmark — zero cloud dependencies.
2. **Self-hosting callout** *(very top, right under the intro)* — a one-line link:
   "→ Want to run it on your own cluster? See the **[Self-Hosting Guide](docs/SELF-HOSTING.md)**."
3. **Capabilities at a glance** — showcase bullets: real-time GPU telemetry;
   one-click vLLM **and** Ollama deploy; **multi-node tensor/pipeline-parallel
   clusters over Ray** (550B Nemotron-3-Ultra NVFP4 across 4 nodes); load-balanced
   inference proxy *(server-side; UI pending)*; **end-to-end fine-tuning** (LoRA,
   multi-node ZeRO-3, resume-from-checkpoint, merge → deploy); **benchmarking**
   (llama-benchy presets + compare view); evaluation; **zero-touch join-token
   onboarding**; HTTP agent auto-update; heterogeneous arm64 + amd64.
4. **Architecture** — a Mermaid diagram (Dashboard ↔ Server ↔ Agents; SQLite/Prisma;
   NFS shared storage; spark-vllm-docker; vLLM/Ollama runtimes) and a short Mermaid
   **deploy-lifecycle sequence** (dashboard → server → agent → container → serving).
5. **Screenshots** — placeholder image links to `docs/screenshots/` (see manifest).
6. **Feature tour by domain** — Nodes & metrics · Deployments (solo + cluster) ·
   Fine-tuning pipeline · Benchmarks & eval · Load balancer *(API complete, UI
   pending)* · Agent onboarding & updates. Each 3–5 lines, linking to `docs/`
   deep-dives (`gemma4-fine-tuning-on-dgx-spark.md`, `qwen3.6-inference-benchmark.md`,
   etc.) and the ROADMAP.
7. **Tech stack** — TS monorepo, Express 5, Next 15 / React 19 / Tailwind 4, Prisma/
   SQLite, WebSocket hubs, Docker, Ray, DeepSpeed, PEFT, vLLM, Ollama, llama-benchy.
8. **Repo layout** — the three packages + `docs/` + recipes repos.
9. **API surface** — complete table, all 13 route groups.
10. **Project status** — short prose + link to refreshed ROADMAP. Honest about what's
    UI-complete vs API-only.
11. **Related repositories** — spark-vllm-docker, dgx-manager-fine-tune-recipes.

## Artifact 2 — docs/SELF-HOSTING.md structure

Everything an operator needs; this is where the (corrected) setup content lives.

1. **Prerequisites** — host: Docker + Docker Compose, Node 22+ (local dev only), a
   shared NFS mount (default `/mnt/tank`), QEMU binfmt for cross-arch bundle builds;
   nodes: NVIDIA GPU + `nvidia-smi`, Docker, SSH access (or just network for token
   onboarding).
2. **Running the manager (Docker Compose — canonical)** — the `build-agent-bundles.sh`
   step, `MANAGER_ADVERTISE_HOST` / `SSH_USER`, `docker compose up -d --build`, ports
   (server 4000, dashboard 3000), the `dgx-data` volume, SSH-key mount, logs.
3. **Onboarding nodes** — the three tiers from the ROADMAP: (a) full SSH+NFS
   provisioning, (b) SSH-only, (c) **join-token install script** (`POST /api/tokens`
   → `GET /api/agent/install.sh`). What each provisions (Docker,
   nvidia-container-toolkit, Node.js, Ollama, agent systemd service).
4. **HTTP agent updates** — bundle rebuild → `POST /api/nodes/:id/update-agent`;
   version tracking + upgrade detection.
5. **Heterogeneous hardware** — per-arch bundles (amd64/arm64), `Node.arch`, install
   script arch detection.
6. **Local development (no Docker)** — `npm run dev`, `db:push`/`db:generate`,
   `npm test`. Clearly marked as the dev path, not production.
7. **Environment variables** — the full table (server + agent + dashboard build args
   + `SHARED_STORAGE_PATH`, `NODE_ADVERTISE_IP`, `METRIC_RETENTION_DAYS`,
   `LLAMA_BENCHY_VERSION`, `HF_TOKEN`).
8. **vLLM recipes & engine isolation** — spark-vllm-docker integration, `VLLM_REPO_PATH`,
   recipe YAML shape, `POST /api/recipes/refresh`, multiple container images.
9. **Troubleshooting** — a few known gotchas (agent management-IP via
   `NODE_ADVERTISE_IP`, "running ≠ serving" for large models, thin unified-memory
   headroom on Spark).

## Artifact 3 — docs/ROADMAP.md refresh

Keep structure and forward phases. Apply factual catch-up only:

- **Phase 5 (Evaluation & Benchmarks)** — benchmarking has shipped: llama-benchy
  runner, presets (`quick-smoke`/`chat-short`/`chat-long`/`code-32k`/`throughput`),
  and three dashboard pages (`/benchmarks`, `/benchmarks/compare`, `/benchmarks/[id]`).
  Update the heading marker and add a "shipped" subsection. SQL-eval findings stay.
- **Status matrix** — `Evaluation` dashboard: `partial (in-chart)` → reflect the
  benchmark UI. Leave Models / Load Balancer dashboard as `placeholder` (verified
  still true). Add a `Benchmarks` row if not present.
- **Add a "Recent work (May–June 2026)" note** capturing: Nemotron-3-Ultra NVFP4
  TP=4 multi-node serving (550B across 4 nodes); MetricSnapshot retention/pruning +
  index; inference-variant selector; deployment + fine-tune log catch-up on
  tab-visible / SSE reconnect; `NODE_ADVERTISE_IP` override; verboseness eval.
- Update **"Last updated"** to 2026-06-11.

## Screenshot Manifest

**Folder:** `docs/screenshots/` (committed to the repo; referenced by relative path
so images render on GitHub). README ships with these links as placeholders; the user
captures and commits the PNGs in a later pass, then links are verified.

Naming: lowercase, kebab-case, `.png`.

| # | Filename | Page | Shows | Priority |
|---|----------|------|-------|----------|
| 1 | `overview.png` | Overview | Cluster summary + live GPU sparklines + node cards | **Hero (README)** |
| 2 | `deployments.png` | Deployments | Deployment list incl. a **multi-node cluster** deployment with status + cluster-node viz | **Hero (README)** |
| 3 | `deployment-logs.png` | Deployment detail | Streaming vLLM startup log viewer | Hero (README) |
| 4 | `finetune-loss-curve.png` | Fine-tune job detail | **Live loss curve** (train + eval overlay), phase progress | **Hero (README)** |
| 5 | `benchmarks.png` | Benchmarks | Benchmark results list / leaderboard | **Hero (README)** |
| 6 | `benchmarks-compare.png` | Benchmarks compare | Side-by-side run comparison | Optional (feature tour) |
| 7 | `nodes.png` | Nodes | Node management: arch badge, GPU, VRAM, provision health checks | Optional (self-hosting) |
| 8 | `finetune-create.png` | Fine-tune create | Job creation form (recipe/node/dataset/hyperparams) | Optional (feature tour) |
| 9 | `datasets.png` | Datasets | Dataset browser + preview rows | Optional (feature tour) |
| 10 | `settings.png` | Settings | Join-token management + agent bundle version/install command | Optional (self-hosting) |

**Not requested:** Models and Load Balancer pages (still stubs — would misrepresent).

Hero shots (1–5) anchor the README's Screenshots section + feature tour. Optional
shots enrich the feature tour and self-hosting guide. README placeholders use a
consistent form, e.g. `![Overview](docs/screenshots/overview.png)` with an italic
caption; if an image is absent the alt text + caption still convey intent.

## Success Criteria

- README: a visitor grasps the full capability surface + architecture in <1 min;
  zero factual errors; no setup instructions (all in the self-hosting guide);
  self-hosting link at the very top; complete 13-row API table; Mermaid diagrams
  render on GitHub.
- SELF-HOSTING.md: a competent operator can stand up the manager via Docker Compose
  and onboard a node (token or SSH) using only this doc; Docker Compose is canonical;
  no stale `npm run dev` agent instructions presented as production.
- ROADMAP: accurate as of 2026-06-11; benchmarks reflected as shipped; Models/LB UI
  still honestly "placeholder"; recent May–June work captured.

## Verification

- No code changed → `npm test` unaffected (sanity-run anyway).
- All intra-repo doc links resolve to existing files.
- Cross-check every capability claim against the verified-facts list above; nothing
  claims a UI that's a stub.
- Mermaid blocks use GitHub-supported syntax.
