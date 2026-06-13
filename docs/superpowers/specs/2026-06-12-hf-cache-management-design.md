# HF Cache Management Page — Design

**Date:** 2026-06-12
**Status:** Approved

## Goal

A dashboard page to inspect and clean up the Hugging Face cache (`HF_HOME`) that
model deployments download into. Users can see what's cached, how big it is, when
it was last deployed, and delete stale repos — without SSHing into nodes.

## Scope

- **In scope:** the HF cache only (`HF_HOME/hub`, i.e. `models--*` and `datasets--*`
  repo dirs). Both topologies are supported: a shared NFS cache (`/mnt/tank/models`)
  *and* per-node local caches (installations without NFS).
- **Out of scope (later):** fine-tune outputs (`$SHARED_STORAGE/outputs`), Ollama
  model storage, per-revision (snapshot-level) deletion. Deletion granularity is
  the whole repo dir.

## Architecture

Approach: **agents scan their own cache with a native Node fs walk** (no
`huggingface-cli`/python dependency, no NFS mount on the manager). Follows the
existing recipes pattern: server fire-and-forgets a command, agent pushes an
inventory message, the hub stores it in memory and broadcasts via SSE. The
filesystem is the source of truth — no Prisma schema changes.

```
Dashboard ──GET /api/hf-cache──> Server (enrich + group)
Dashboard ──POST /api/hf-cache/scan──> Server ──cmd:hf-cache:scan──> ALL agents
Dashboard ──DELETE /api/hf-cache/:cacheId?repoId=…──> Server ──cmd:hf-cache:delete──> one agent in group
Agent ──agent:hf-cache (inventory)──> Server hub (store per nodeId) ──SSE hf-cache:inventory──> Dashboard
```

### Cache identity (shared vs per-node)

Every agent scans its own resolved `HF_HOME` (via the existing `resolveHfHome()`
in `packages/agent/src/runtime/sparkrun.ts:28-30`: explicit `HF_HOME` env, else
`${SHARED_STORAGE}/models`). To avoid rendering one shared NFS cache N times,
the agent reads-or-creates a marker file **`.dgx-cache-id`** (a UUID, one line)
at the HF_HOME root during scan. Nodes on shared storage read the same UUID;
nodes with local disks each get their own. The server groups per-node
inventories by this `cacheId` into *cache groups*. Deterministic — no
filesystem-type heuristics.

## Wire protocol

New agent commands (handled in `packages/agent/src/index.ts` alongside
`cmd:rescan-recipes` etc.):

- `cmd:hf-cache:scan` `{}` — scan and push inventory.
- `cmd:hf-cache:delete` `{ repoId: string, kind: "model" | "dataset" }` —
  validate, `fs.rm` the repo dir, rescan, push fresh inventory. Failures push
  an inventory with `error` set.

New agent → server message:

```ts
{ type: "agent:hf-cache", payload: {
    cacheId: string,        // from .dgx-cache-id marker
    hfHome: string,
    scannedAt: string,      // ISO
    totalBytes: number,
    diskFreeBytes: number,  // fs.statfs(hfHome)
    error?: string,         // scan/delete failure (HF_HOME missing, rm failed…)
    repos: Array<{
      repoId: string,       // "org/name" decoded from models--org--name
      kind: "model" | "dataset",
      sizeBytes: number,    // lstat sum: blobs carry real bytes, snapshot
                            // symlinks count ~0 → no double-counting
      nFiles: number,
      revisions: number,    // snapshot dir count
      lastModified: string  // max mtime across the repo = download/update time
    }>
} }
```

## Agent

New module `packages/agent/src/runtime/hf-cache.ts`:

- `parseRepoDirName(dirName)` → `{ kind, repoId } | null` — decodes
  `models--org--name`; returns null for unrecognized entries (skipped, not fatal).
- `repoDirName(kind, repoId)` — inverse, used by delete.
- `scanHfCache(hfHome)` → inventory (sans cacheId). Walks `hfHome/hub` with
  `lstat`, sums sizes, counts files/snapshots, reads max mtime. Missing
  `hub/` dir → empty repo list (a fresh cache is not an error); missing/
  unreadable `hfHome` itself → throws (caller reports `error`).
- `readOrCreateCacheId(hfHome)` → UUID string from `.dgx-cache-id`, creating it
  (with `crypto.randomUUID()`) if absent.
- `deleteCachedRepo(hfHome, kind, repoId)` — **security boundary.** `repoId`
  must be **1 or 2** segments of `[A-Za-z0-9._-]+` (org-less legacy repos like
  `gpt2` are valid 1-segment ids), with no segment equal to `.` or `..` (note
  `.` and `-` are otherwise legal inside HF names), and the resolved target
  path must be a strict descendant of `hfHome/hub`. `rmSync` uses
  `{ recursive: true, force: true }` so a concurrent download mutating the tree
  can't turn a valid delete into a spurious ENOENT. Deletes the `models--…` or
  `datasets--…` dir per `kind`. Unknown repo → throws (surfaces as inventory
  `error`).

`index.ts` gets the two `cmd:hf-cache:*` handlers; both end by pushing a fresh
`agent:hf-cache`. Agent version bump per CLAUDE.md is mandatory.

## Server

### Hub (`packages/server/src/ws/agent-hub.ts`)

- New case `agent:hf-cache`: store payload in an in-memory
  `Map<nodeId, InventoryWithMeta>` (latest wins), then
  `sseBroadcast({ type: "hf-cache:inventory", payload: { nodeId, ...payload } })`.
- Expose `getHfCacheInventories()` and reuse `sendToAgent` / connected-agent
  enumeration for routes.

### Routes (`packages/server/src/routes/hf-cache.ts`, mounted at `/api/hf-cache`)

- `GET /api/hf-cache` → `{ caches: CacheGroup[] }` where a `CacheGroup` is
  per-node inventories grouped by `cacheId`:
  `{ cacheId, nodes: [{ nodeId, name, connected }], hfHome, scannedAt,
     totalBytes, diskFreeBytes, error?, repos: EnrichedRepo[] }`.
  For groups (shared cache) the newest inventory wins for repo data.
  Enrichment per repo: `lastDeployedAt` (latest matching Deployment.createdAt,
  any status) and `inUse` (any matching deployment with status not in
  `stopped`/`failed` whose head node or cluster nodes intersect the group's
  nodes). All non-terminal states — running, `evicted` (restorable onto the
  GPU), and the transient lifecycle states (pending/launching/deploying/
  stopping/removing) — count as in-use; the bias is deliberate, since a false
  "in use" only annoys whereas a false "deletable" can `rm` weights out from
  under a live container.

  **Matching (the soundness-critical part).** `matchRepoToModels(repoId,
  candidates)` is case-insensitive exact string equality. The guard is only as
  sound as the candidate set, which the route (`loadDeploymentUsage`) assembles
  per deployment from THREE sources, because no single field holds the HF repo
  id for every deploy kind:
  1. `Model.name` + config `modelName` — covers Ollama (name is the pull tag)
     and inline-YAML vLLM (name is the HF id).
  2. The recipe catalog's `model:` resolved from config `recipeFile` via
     `agentHub.getRecipes()` — **required** for registry-ref vLLM deploys,
     where `Model.name` is only the recipe *slug*, not the HF repo id.
  3. `FineTuneJob.baseModel` — a fine-tune deploy still loads its base weights
     from the HF cache, so the base repo must register as in-use.

  **Known matching gaps — residual UNSAFE-direction risk, accepted for now.**
  Two cases can't resolve the HF id and so produce a *missed* match (repo shows
  deletable even if actually loaded): a `recipePath` deploy doesn't persist its
  recipe ref, and a recipe absent from the live catalog can't be resolved.
  Mitigations that already narrow the blast radius: registry-ref (the common
  path), inline-YAML, Ollama, and fine-tune deploys are all covered; deletion
  is a deliberate two-step user action with a confirm dialog; and a deleted
  repo is simply re-downloaded on next deploy (data loss, not corruption).
  Closing these fully would mean persisting the resolved HF id on the
  Deployment row at create time — tracked as follow-up, out of scope here.
- `POST /api/hf-cache/scan` → send `cmd:hf-cache:scan` to **all** connected
  agents (like `/api/recipes/refresh`), `202 { requested: n }`; `503` if none.
- `DELETE /api/hf-cache/:cacheId?repoId=<url-encoded>&kind=model|dataset`
  (`kind` defaults to `model`; repoId travels as a **query parameter** because
  URL-encoded slashes in path segments are unreliable across HTTP stacks) →
  `400` if repoId missing; `409` with blocking deployment names if `inUse`
  (sends nothing to any agent); `404` if cacheId unknown or no repo of that
  `repoId`+`kind` is in the cache; `503` if no agent in the group is connected;
  otherwise send `cmd:hf-cache:delete` `{ repoId, kind }` to any *connected*
  agent in the group, `202`.
- OpenAPI spec (`openapi.ts`) updated for all three.

## Dashboard

Repurpose the stub `packages/dashboard/app/models/page.tsx` (the "Models" nav
entry becomes the cache manager). Follows the deployments-page pattern
(`apiFetch` + SSE + sonner toasts).

- One section per cache group. Single group → no visual grouping overhead;
  multiple → stacked sections titled by node list (e.g. "Cache on spark-3 —
  local" / "Shared cache — spark-1, spark-2").
- Section header: total size, disk free, `hfHome` path, last-scanned time.
  Page-level Rescan button (`POST /scan`).
- Table per group, default sort size-desc, sortable by last-deployed:
  repo id · kind · size · revisions · downloaded · last deployed · in-use
  badge · delete button. Delete disabled with tooltip when in use; otherwise a
  confirm dialog → `DELETE` → wait for the `hf-cache:inventory` SSE event to
  refresh.
- First visit with no inventory auto-triggers one scan. No connected agents →
  explicit empty state ("No agents connected — cache contents unknown"), never
  an infinite spinner. Inventory `error` → inline error banner in that section.

## Error handling

Fail fast, observable everywhere: agent scan/delete failures travel as the
inventory `error` field and render in the UI; the server returns explicit 4xx/5xx
(409 in-use, 404 unknown, 503 no agents); the agent's delete path validates
`repoId` before touching the filesystem.

## Testing (medium-risk + destructive delete)

- **Property tests:** `parseRepoDirName`/`repoDirName` round-trip; delete path
  containment (arbitrary repoId strings never resolve outside `hub/`);
  `matchRepoToModels` case-insensitivity.
- **Unit tests:** `scanHfCache` against a fake cache layout built in
  `mkdtempSync` (sizes, symlink non-double-counting, missing hub dir, marker
  file creation/reuse); cache grouping (shared / per-node / mixed).
- **Integration (supertest, stub agentHub):** GET grouping + enrichment incl.
  `inUse`; DELETE 409 in-use; DELETE happy path sends `cmd:hf-cache:delete` to
  a connected group member; scan with no agents → 503.
- `npm test` green before claiming done.
