# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DGX Manager is a full-stack system for managing a DGX Spark cluster. It handles node provisioning via SSH, real-time GPU metrics collection, model deployment, inference load balancing, and fine-tuning job orchestration.

## Architecture

npm workspaces monorepo with three packages:

- **`packages/server`** — Express 5 backend (REST API + WebSocket hubs), port 4000
- **`packages/dashboard`** — Next.js 15 frontend (React 19 + Tailwind CSS 4), port 3000
- **`packages/agent`** — Node.js agent deployed on DGX nodes, collects GPU metrics via nvidia-smi

Communication flow:
```
Dashboard <--WS /ws/dashboard--> Server <--WS /ws/agent--> Agent (on DGX node)
                                   |
                              Prisma (SQLite)
```

The server has REST routes at `/api/*` and an inference proxy at `/lb/` for round-robin routing to deployments.

## Running the App

**Always use Docker Compose to run the app:**

```bash
# One-time host setup: register QEMU binfmt handlers so we can cross-build
# the amd64 agent bundle from an arm64 manager (and vice versa).
docker run --privileged --rm tonistiigi/binfmt --install all

# Build per-arch agent bundles (amd64 + arm64) before every compose build.
./scripts/build-agent-bundles.sh

# Start (set your machine's IP and SSH user)
MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d

# Rebuild after code changes (non-disruptive — won't kill active deployments)
./scripts/build-agent-bundles.sh && \
  MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build

# Stop (WARNING: removes docker network, may disrupt active Ray clusters)
docker compose down

# Logs
docker compose logs server -f
docker compose logs dashboard -f
```

- Server runs on port 4000, dashboard on port 3000
- SQLite DB persists in a Docker volume (`dgx-data`)
- SSH keys mounted from host `~/.ssh` for node management
- `NEXT_PUBLIC_*` vars are baked into the dashboard at build time via build args

## Development Commands

```bash
npm run dev              # Run server + dashboard locally (dev mode, no Docker)
npm run dev:server       # Server only (tsx watch, port 4000)
npm run dev:dashboard    # Dashboard only (next dev, port 3000)
npm run build            # Build all packages (tsc + next build)

npm run db:push          # Apply Prisma schema changes to SQLite
npm run db:generate      # Regenerate Prisma client
npm run db:studio        # Open Prisma Studio GUI
```

## Key Source Locations

### Server (`packages/server/src/`)
- `index.ts` — Express setup, WebSocket hub init, route mounting
- `routes/` — REST handlers: nodes, models, deployments, finetune, loadbalancer
- `ws/agent-hub.ts` — Manages agent WebSocket connections, processes metrics
- `ws/dashboard-hub.ts` — Broadcasts updates to connected dashboards
- `ssh/provisioner.ts` — Audits prerequisites, auto-installs packages on nodes
- `routes/agent-bundle.ts` — Serves per-arch agent tarballs + generates the token install script
- `proxy/inference-proxy.ts` — Round-robin request routing to deployments

### Dashboard (`packages/dashboard/`)
- `app/` — Next.js App Router pages (overview, nodes, models, deployments, finetune, loadbalancer)
- `components/` — node-card, metric-gauge, deployment-table, finetune-log
- `lib/api.ts` — Fetch wrapper; `lib/ws.ts` — useWebSocket hook

### Agent (`packages/agent/src/`)
- `index.ts` — WebSocket client, reconnection logic, 5-second metrics loop
- `metrics.ts` — nvidia-smi wrapper for GPU metrics
- `runtime/` — vLLM/Ollama integrations (Phase 2)

## Database

SQLite via Prisma ORM. Schema at `prisma/schema.prisma`. Core models: Node, Model, Deployment, MetricSnapshot, LoadBalancerRule, LoadBalancerEndpoint, FineTuneJob.

## Environment Variables

Copy `.env.example` to set up. Key vars:
- `DATABASE_URL` — Prisma DB path (default: `file:./dev.db`)
- `PORT` / `MANAGER_HOST` — Server bind config
- `NEXT_PUBLIC_API_URL` / `NEXT_PUBLIC_WS_URL` — Dashboard connects to server
- `MANAGER_URL` / `NODE_ID` — Agent connects to server

## Agent Version Bumping — MANDATORY

**Every time you edit ANY file under `packages/agent/src/`, you MUST bump the agent version** by running:
```bash
./scripts/bump-agent-version.sh
```
This increments the patch version in `packages/agent/package.json` (e.g. 0.5.0 → 0.5.1). The dashboard uses this version to detect outdated agents and offer upgrades. Forgetting to bump means agents won't know they need updating.

Do this BEFORE committing. If you edited multiple agent files in one session, bump once at the end.

## Conventions

- TypeScript strict mode, ES modules throughout
- All packages use `tsx` for dev, `tsc` for production builds
- Features organized by domain in directories (routes/, ws/, ssh/, proxy/)
- Dashboard uses App Router with `@/*` path alias


## Development Principles

1. **Test-Driven Development**: Write or update tests first. Do not claim completion unless tests run and pass, or explicitly state why they could not be run.

2. **Small, Reversible, Observable Changes**: Prefer small diffs and scoped changes. Implement user-testable and visible changes before backend changes wherever feasible. Keep changes reversible where possible. Maintain separation of concerns; avoid mixing orchestration, domain logic, and IO unless trivial.

3. **Fail Fast, No Silent Fallbacks**: Validate inputs at boundaries. Surface errors early and explicitly. Assume dependencies may fail. No silent fallbacks or hidden degradation. Any fallback must be explicit, tested, and observable.

4. **Minimize Complexity (YAGNI, No Premature Optimization)**: Implement the simplest solution that meets current requirements and tests. Do not design for speculative future use cases. Optimize only with evidence.

5. **Deliberate Trade-offs: Reusability vs. Fit (DRY with Restraint)**: Apply DRY only to real, stable duplication. Avoid abstractions that increase cognitive load without clear benefit. Prefer fit-for-purpose code unless a second use case is concrete.

6. **Don't Assume—Ask for Clarification**: If requirements are ambiguous or multiple interpretations exist, ask. If proceeding is necessary, state assumptions explicitly and keep changes localized and reversible.

7. **Confidence-Gated Autonomy**: Proceed end-to-end only when confidence is high. Narrow scope and increase checks when confidence is medium. Stop and ask when confidence is low.

8. **Security-by-Default**: Treat all external input as untrusted. Use safe defaults and least privilege. Do not weaken auth, authz, crypto, or injection defenses without explicit instruction. Never introduce secrets into code.

9. **Don't Break Contracts**: Preserve existing public APIs, schemas, and behavioral contracts unless explicitly instructed otherwise. If breaking changes are required, provide migration steps and compatibility tests.

10. **Risk-Scaled Rigor**: Scale rigor with impact: (1) Low risk — unit tests, lint/format. (2) Medium risk — integration tests, edge cases, rollback awareness. (3) High risk (security, auth, money, data loss, core flows) — explicit approval before destructive actions, targeted tests, minimal refactoring.
