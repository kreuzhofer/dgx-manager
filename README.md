# DGX Manager

A full-stack system for managing and monitoring a DGX Spark cluster. Handles node provisioning, real-time GPU metrics, model deployment via vLLM, inference load balancing, and fine-tuning job orchestration.

## Architecture

Three-package TypeScript monorepo using npm workspaces:

- **Server** (`packages/server`) — Express 5 REST API + WebSocket hubs (port 4000)
- **Dashboard** (`packages/dashboard`) — Next.js 15 web UI (port 3000)
- **Agent** (`packages/agent`) — Runs on each DGX node, collects GPU metrics and executes deployments

```
Dashboard <──WS──> Server <──WS──> Agent (on DGX node)
                     │                  │
                 SQLite DB         nvidia-smi
                 (Prisma)          spark-vllm-docker
```

## Prerequisites

- Node.js 22+
- Git
- NVIDIA GPU with `nvidia-smi` (on agent nodes)

## Quick Start

```bash
# Install dependencies
npm install

# Set up environment
cp .env.example .env
# Edit .env as needed (defaults work for local development)

# Initialize the database
npm run db:generate
npm run db:push

# Start server + dashboard
npm run dev
```

- **Dashboard**: http://localhost:3000
- **Server API**: http://localhost:4000/api
- **Health check**: http://localhost:4000/api/health

## Running the Agent

The agent runs on each DGX node and connects back to the manager server:

```bash
NODE_ID=<node-id-from-db> MANAGER_URL=ws://<server-ip>:4000/ws/agent npm run dev -w packages/agent
```

The agent will:
1. Register with the server (GPU model, VRAM)
2. Discover available vLLM recipes from the [spark-vllm-docker](https://github.com/kreuzhofer/spark-vllm-docker) repo and report them to the server
3. Stream GPU metrics every 5 seconds

## vLLM Recipes

The agent integrates with [spark-vllm-docker](https://github.com/kreuzhofer/spark-vllm-docker) for inference. On startup, the agent checks for the repo at `VLLM_REPO_PATH` (default: `/mnt/tank/src/github/spark-vllm-docker`) and clones it if missing.

Recipes are YAML configs in the repo's `recipes/` directory defining one-click vLLM deployments — model, container image, quantization, defaults, etc.

Available recipes are reported to the server via WebSocket and exposed at `GET /api/recipes`.

## Development Commands

```bash
npm run dev              # Server + dashboard in parallel
npm run dev:server       # Server only (tsx watch, port 4000)
npm run dev:dashboard    # Dashboard only (next dev, port 3000)
npm run build            # Build all packages

npm run db:push          # Apply schema changes
npm run db:generate      # Regenerate Prisma client
npm run db:studio        # Prisma Studio GUI
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET/POST | `/api/nodes` | Node management |
| GET/POST | `/api/models` | Model registry |
| GET/POST/DELETE | `/api/deployments` | Deployment management |
| POST | `/api/deployments/:id/restart` | Restart deployment |
| GET | `/api/recipes` | Available vLLM recipes (from agents) |
| GET/POST | `/api/finetune` | Fine-tune jobs |
| GET/POST | `/api/lb` | Load balancer rules |

## WebSocket Channels

- `ws://localhost:4000/ws/agent` — Agent connections (metrics, recipes, deployment status)
- `ws://localhost:4000/ws/dashboard` — Dashboard real-time updates

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `4000` | Server HTTP port |
| `MANAGER_HOST` | `0.0.0.0` | Server bind address |
| `DATABASE_URL` | `file:./dev.db` | Prisma database URL |
| `NEXT_PUBLIC_API_URL` | `http://localhost:4000` | Dashboard → server API |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:4000/ws/dashboard` | Dashboard → server WS |
| `NODE_ID` | — | Agent: node ID (required) |
| `MANAGER_URL` | `ws://localhost:4000/ws/agent` | Agent: server WS URL |
| `VLLM_REPO_PATH` | `/mnt/tank/src/github/spark-vllm-docker` | Agent: path to spark-vllm-docker repo |
