# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DGX Manager is a full-stack system for managing a DGX Spark cluster. It handles node provisioning via SSH, real-time GPU metrics collection, model deployment (via [sparkrun](https://github.com/spark-arena/sparkrun) — the head-node agent runs `sparkrun run`), inference load balancing, and fine-tuning job orchestration.

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
- `routes/` — REST handlers: nodes, models, deployments (recipeFile | recipePath | inline recipeYaml), finetune, loadbalancer, recipes
- `deployments/recipe-path.ts`, `deployments/recipe-inline.ts` — validate the path / inline-YAML deploy sources (security boundary)
- `ws/agent-hub.ts` — Manages agent WebSocket connections, processes metrics
- `ws/dashboard-hub.ts` — Broadcasts updates to connected dashboards
- `ssh/provisioner.ts` — Audits prerequisites, auto-installs packages on nodes (incl. sparkrun install + non-interactive setup)
- `routes/agent-bundle.ts` — Serves per-arch agent tarballs + generates the token install script
- `proxy/inference-proxy.ts` — Round-robin request routing to deployments
- `openapi.ts` — builds the OpenAPI 3 spec (served at `/api/openapi.json`, Swagger UI at `/api/docs`)

### Dashboard (`packages/dashboard/`)
- `app/` — Next.js App Router pages (overview, nodes, models, deployments, finetune, loadbalancer)
- `components/` — node-card, metric-gauge, deployment-table, finetune-log
- `lib/api.ts` — Fetch wrapper; `lib/ws.ts` — useWebSocket hook

### Agent (`packages/agent/src/`)
- `index.ts` — WebSocket client, reconnection logic, 5-second metrics loop; `cmd:deploy`/`cmd:undeploy` handlers + reconnect reconciliation
- `metrics.ts` — nvidia-smi wrapper for GPU metrics
- `recipes.ts` — recipe catalog from `sparkrun list`, mapped to the wire `Recipe` shape
- `runtime/sparkrun.ts` — deploy lifecycle: `sparkrun run`/`stop`, `cluster check-job` liveness, and the `sparkrun logs` follower that streams container logs
- `runtime/sparkrun-args.ts` · `sparkrun-parse.ts` · `sparkrun-metrics.ts` — pure argv builder, CLI output parsers, vLLM `/metrics` scrape
- `runtime/ollama.ts`, `runtime/finetune.ts` — Ollama deploys; fine-tune training/merge (fine-tune deploy also goes through sparkrun)

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


## Testing

The repo uses **Vitest** + **fast-check** + **supertest**, configured at the root in `vitest.config.ts`.

```bash
npm test                # run all tests once (use this before claiming any change is done)
npm run test:watch      # TDD loop
npm run test:ui         # vitest UI in browser
npx vitest run path/to/file.test.ts   # focus a single file
npx vitest run -t "substring"          # focus tests by name substring
```

### Layout

- **Unit + property tests** live next to source as `<name>.test.ts`.
- **Integration tests** that need a real Prisma DB live under `packages/<pkg>/src/__tests__/integration/`. Each suite gets a per-test SQLite via `mkdtempSync` + `DATABASE_URL=file:/tmp/...` (set BEFORE importing prisma) and applies the schema via `npx prisma db push --force-reset`. Wipe between tests in FK-dependency order — see `wipeAll()` in `deployments.vram-admission.test.ts` for the canonical pattern.
- **HTTP routes** are tested via supertest against an Express app that mounts only the router under test, with a stub `agentHub` injected via `app.set("agentHub", …)`. No WebSocket, no port binding.

### When to use what

- **Property test** (`it.prop` from `@fast-check/vitest`) for any pure helper whose correctness is best stated as an invariant over an input space rather than a fixture. Always include a plain-English doc comment above the property describing the invariant — the test should read like a spec.
- **Unit test** for parsers, formatters, and other pure functions where the input space is small and named cases are clearer than properties.
- **Integration test** for HTTP behavior, multi-step flows, or anything where the pure-helper / DB-coupled split matters (e.g. admission checks).
- **Refactor pure logic out** of route files / IO-heavy modules so it can be unit-tested without mocks. The `packages/server/src/admission/vram.ts` split is the example to follow: a pure `computeVramShortfall` for property tests, a Prisma-coupled `checkVllmVramAdmission` for integration tests.

### Risk-tier expectations

- **Low risk** (refactors, log-message tweaks, comment changes): run the existing tests; add one only if a regression would silently degrade behavior.
- **Medium risk** (new endpoints, new config knobs, validation changes): one integration test for the happy path + one for the error path; unit/property test any pure helper added.
- **High risk** (admission control, multi-node coordination, anything affecting data persistence or money): property tests for the invariant + integration test for the failure mode + a hand-picked test mirroring the original incident if there was one.

### Don't claim done without `npm test` green

Principle 1 ("Test-Driven Development") is now enforceable: every change should leave `npm test` passing. If a test cannot be added (e.g. genuinely environmental behavior on a real DGX), say so explicitly in the PR description and explain what manual verification was done instead.

### AI agent + Prisma destructive ops

Prisma 7 refuses `db push --force-reset` and similar destructive operations when invoked by an AI agent unless `PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION` is set with an explicit user-consent string. The integration test sets this on a per-test basis with a fixed consent record — see `deployments.vram-admission.test.ts`. Do **not** set this env var globally; only inside test setup that operates against a per-suite SQLite file.


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
