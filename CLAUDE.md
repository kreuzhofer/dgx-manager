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
# Start (set your machine's IP and SSH user)
MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d

# Rebuild after code changes (non-disruptive — won't kill active deployments)
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
- `ssh/agent-deployer.ts` — Deploys agent as systemd service on remote nodes
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

## Conventions

- TypeScript strict mode, ES modules throughout
- All packages use `tsx` for dev, `tsc` for production builds
- Features organized by domain in directories (routes/, ws/, ssh/, proxy/)
- Dashboard uses App Router with `@/*` path alias
