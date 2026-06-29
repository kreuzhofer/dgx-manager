# UI-Managed Sparkrun Registries — Design

**Date:** 2026-06-29
**Status:** Approved (design), pending implementation plan
**Owner:** Daniel Kreuzhofer

## Problem

The set of sparkrun recipe registries is **external, per-node state**. Each DGX node's
agent runs `sparkrun list --json` (`packages/agent/src/recipes.ts`) and trusts whatever
registries happen to be configured in that node's `~/.config/sparkrun/registries.yaml`
(under the agent user's HOME — the agent systemd unit sets `HOME=/home/<AGENT_USER>`).

dgx-manager has **no awareness** of this file:

- It is not in any git repo, the Prisma DB, or the agent bundle.
- The provisioner never registers a registry (no `registry add` code exists).
- Nothing syncs it between nodes — each file is hand-edited, drift-prone.

Concrete consequence: the self-authored `rtx-recipe-registry`
(`kreuzhofer/rtx-recipe-registry`, ~41 amd64/RTX-5090 recipes) is **not registered**
anywhere, so its recipes never reach `sparkrun list`, never reach `GET /api/recipes`,
and never appear in the deploy form.

## Goal

Let an operator manage the cluster's sparkrun registries from the dgx-manager UI, with
the manager as the single source of truth that pushes config to every node.

Non-goals (v1): private/authenticated registries; per-node registry assignment;
migrating the project to `prisma migrate`.

## Decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Source of truth | **dgx-manager DB owns it** (push model); per-node `registries.yaml` becomes a generated artifact. Full CRUD UI. |
| 2 | Delivery channel | **Agent / WebSocket.** Server broadcasts; agent writes its own file, re-discovers, reports the refreshed catalog back. |
| 3 | Scope | **Single cluster-wide list.** Every node gets the identical set; existing per-recipe arch-filtering (`deriveRecipeArch`) decides what surfaces per node. No per-node matrix. |
| 4 | Auth | **Public registries only.** No secrets stored. |
| 5 | Seeding | **Idempotent boot-time seed-if-empty** of the 7 standard registries. Not a migration (repo has no migration system). |
| 6 | Schema strategy | **Stay on `db push`.** The new table is purely additive (non-destructive). |

## Architecture

### A. Data model (Prisma) — additive table

```prisma
model SparkrunRegistry {
  id               String   @id @default(cuid())
  name             String   @unique   // registry name, e.g. "rtx", "eugr"
  url              String              // git repo sparkrun clones
  subpath          String              // recipes subpath, e.g. "recipes"
  description      String?
  visible          Boolean  @default(true)
  // optional advanced subpaths sparkrun supports
  tuningSubpath    String?
  benchmarkSubpath String?
  modsSubpath      String?
  sortOrder        Int      @default(0)
  createdAt        DateTime @default(now())
  updatedAt        DateTime @updatedAt
}
```

Applied via `npm run db:generate && npm run db:push`. No migration file.

**Deploy ordering caveat:** `db push` is not run at container boot. The schema must be
pushed to the `dgx-data` volume **before** the new server code boots, else the startup
seed hits a missing table. Deploy order: `db:push` → restart server (seed runs) → done.

### B. Seeding — `packages/server/src/registries/seed.ts`

- Exports `seedDefaultRegistries(prisma)`; takes the client as a **parameter** (so
  integration tests pass a per-suite client).
- Logic: `if ((await prisma.sparkrunRegistry.count()) === 0) createMany(DEFAULTS)`.
  Idempotent; never clobbers later edits.
- `DEFAULTS` is a plain constant array of the 7 currently-deployed registries
  (`sparkrun-testing`, `sparkrun-transitional`, `official`, `experimental`,
  `community`, `eugr`, `atlas`) — version-controlled, reviewable, co-located.
  Each entry must reproduce the live config **verbatim**, including its `visible`
  flag (`testing`/`experimental`/`community`/`atlas` are `visible: false`;
  `transitional`/`official`/`eugr` are `visible: true`) and its optional
  `tuning/benchmark/mods` subpaths. Source of truth for the values: the current
  `~/.config/sparkrun/registries.yaml`.
- Called once during server boot in `index.ts`, alongside the existing
  `prisma.benchmarkRun.updateMany(...)` startup block. Defensive: log rather than crash
  boot if the table is absent (schema not yet pushed).

### C. Server — `/api/registries`

- Router mounted in `index.ts`; uses the shared `prisma` singleton (direct import,
  per repo convention) and reads `agentHub` via `req.app.get("agentHub")`.
- Endpoints: `GET /` (list), `POST /` (create), `PATCH /:id` (update), `DELETE /:id`.
- **Pure, testable helper** `validateRegistry(input)` (extracted, no IO):
  - `name` matches `^[a-z0-9-]+$`, unique
  - `url` is a syntactically valid git URL with an allowed scheme (`https`/`http`/`git`/`ssh`)
  - `subpath` non-empty
  - Fail-fast at the boundary; 400 with a clear message on violation.
- On any mutation: load the **full** registry set and broadcast `cmd:set-registries`
  to all online agents via `agentHub`. Payload is the **structured list** — the server
  never string-builds YAML.

### D. Agent — `cmd:set-registries` handler (version bump required)

- New handler in `packages/agent/src/index.ts`:
  1. Receive the structured registry list.
  2. Render `~/.config/sparkrun/registries.yaml` via a **YAML serializer** (no string
     concatenation — injection-safe).
  3. **Atomic write** (temp file + rename).
  4. Re-run `discoverRecipes()`.
  5. Report the refreshed, arch-tagged catalog back up (existing recipe-report path).
- On agent **(re)connect**, the server pushes the current set so the node reconciles to
  truth (handles offline-during-edit nodes).
- **Pure helper** `renderRegistriesYaml(registries)` — property-tested for round-trip
  and escaping.
- **MANDATORY:** run `./scripts/bump-agent-version.sh` after editing agent files.

### E. Dashboard — Registries settings page

- Table: name, url, subpath, `visible` toggle, recipe-count contributed.
- Add / Edit / Delete. Form fields: `name / url / subpath / description / visible`;
  advanced subpaths (`tuning/benchmark/mods`) collapsed.
- After save, surface the per-node refresh result (new recipe count) so the push is
  observable (no silent success).

### F. Data flow

```
UI edit → POST/PATCH/DELETE /api/registries → DB (source of truth)
        → agentHub broadcast cmd:set-registries → each agent writes registries.yaml
        → agent re-runs `sparkrun list` → reports catalog → dashboard recipe list refreshes
agent (re)connect → server pushes current set → agent reconciles
```

## Risk & mitigations

**Medium-high risk.** A bad or empty push overwrites `registries.yaml` cluster-wide and
could drop `eugr`/`official`, breaking *all* deployable recipes at once.

Mitigations:
- Hard `validateRegistry` at the boundary; reject empty/malformed sets.
- Atomic write on the agent (no partial file).
- Agent reports the refreshed catalog back, so breakage is immediately visible in the UI.
- Seed-if-empty guard prevents accidental wipe on first boot.
- Reversible: re-add via UI; no destructive schema change.

## Testing (per CLAUDE.md conventions)

- **Property test** — `renderRegistriesYaml`: serialization round-trip + escaping invariants.
- **Unit tests** — `validateRegistry`: happy path + each rejection (bad name, bad URL
  scheme, empty subpath, dup name).
- **Integration (supertest + per-suite SQLite)** — `/api/registries`:
  - happy: `POST` creates, persists, and broadcasts `cmd:set-registries` (assert against
    a stubbed `agentHub` injected via `app.set`)
  - error: invalid input → 400, no broadcast
  - `GET` lists; `DELETE` removes; duplicate `name` rejected
  - seed: `seedDefaultRegistries` inserts 7 on empty DB, no-ops on populated DB
- **Agent** — `cmd:set-registries`: writes file + triggers re-discovery (render unit-tested
  pure; handler thin, fs/exec mocked).

Risk tier: medium-high (new endpoints, new config knob, writes files cluster-wide) →
integration happy + error paths, property/unit tests on all pure helpers.

## Out of scope (future)

- Private/authenticated registries (token or deploy-key per registry, secure storage).
- Per-node / per-arch registry assignment.
- Drift detection/reporting (agent → manager reconciliation surfacing mismatches).
- Provisioner integration so brand-new nodes get the registry set on first provision.
