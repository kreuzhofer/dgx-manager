# Choose + surface the cluster head node (design)

**Status:** design / approved to plan
**Date:** 2026-07-05
**Scope:** Dashboard-only. No backend / schema / dispatch change.

## Motivation

For a multi-node deploy (dgxrun or sparkrun, TP>1), the head (rank 0) node runs the scheduler + detokenizer + MTP + rank-0, holds the tightest memory, serves the `:8000` API, and is where tps is scraped. Today the deploy form stores the selected cluster nodes in an **unordered `Set`** (`selectedClusterNodes`) and sends `nodeIds = Array.from(set)`, so `nodeIds[0]` silently becomes the head — the operator can neither **choose** which node is head nor **see** which one it is. Discovered live: a deploy auto-picked `.38` as head; the operator (and the assistant) assumed `.36`, and tps looked "missing" purely because it only appears on the head and nothing said which node that was.

The backend already has everything: the deploy route sets `headNodeId = nodeIds[0]`, records `ClusterNode.role = "head"` for it, and links `Deployment.node` to the head. So this is a **dashboard-only** fix: let the operator pick the head (sending it first) and badge it everywhere.

## Goals

- **Choose** the head at deploy time without a new dropdown: a crown / "★ head" toggle on each *selected* cluster node (radio semantics — exactly one head), defaulting to the first selected. On submit, `nodeIds` is built head-first.
- **Surface** the head after deploy: badge the `role:"head"` node (equivalently `Deployment.node`) in the deployments list / cluster-node view.
- Works for both dgxrun and sparkrun multi-node deploys (it reads `role`/`node`, runner-agnostic).

## Non-goals

No backend/API/Prisma/dispatch change. No full worker-rank ordering (workers are symmetric — only the head is operator-relevant). No new dropdown. No change to single-node or Ollama deploys.

## Architecture (all in `packages/dashboard/`)

### 1. Pure helper — head-first `nodeIds`
Extract a tiny pure function (e.g. `lib/cluster-nodes.ts`):
`buildClusterNodeIds(headId: string | null, selected: Iterable<string>): string[]` — returns the selected ids with `headId` first (deduped, order otherwise preserved); if `headId` is absent or not in `selected`, returns the selected ids as-is (first element is the effective head). Unit-tested.

### 2. Deploy form (`app/deployments/page.tsx`)
- Add `clusterHeadId: string` state. When cluster nodes are (de)selected, keep it valid: default to the first selected; if the current head is deselected, fall back to the new first selected.
- In the cluster-node selection list, render a **crown/★ toggle** on each selected node row (unselected rows don't show it). Clicking sets `clusterHeadId` to that node (radio — one head). The head row shows a **"head" badge**, visible before launch.
- On submit (the cluster deploy body), replace `nodeIds: Array.from(selectedClusterNodes)` with `nodeIds: buildClusterNodeIds(clusterHeadId, selectedClusterNodes)`. Apply to both the vLLM cluster deploy and the cluster fine-tune deploy paths that send `nodeIds` from `selectedClusterNodes`.

### 3. Surface the head after deploy
In the deployments list / cluster-node visualization component, badge the node whose `clusterNodes[].role === "head"` (fall back to `deployment.node`) with a crown + "head" label. Pure rendering over data already in `GET /api/deployments`.

## Testing (low risk — UI + one pure helper)
- **Unit** (`lib/cluster-nodes.test.ts`): `buildClusterNodeIds` — head moved to front; head not in selection → unchanged; no head → unchanged; single node → `[id]`; no duplicates.
- **Build:** `npm run build` (dashboard) clean. `npm test` stays green (no server change). No agent change → no agent version bump.
- No dashboard component-test harness exists in the repo; the grouping/badge rendering is verified by build + manual, consistent with the repo's risk-tier norms for presentational changes.

## Isolation / boundaries
- `lib/cluster-nodes.ts` — pure, testable, no React/DOM.
- `deployments/page.tsx` — head state + crown toggle + head-first submit + selection badge.
- deployments-list/cluster-viz — head badge (read-only render).
- Zero coupling to the server; the existing `role`/`node`/`nodeIds[0]=head` contract is unchanged.

## References
- Head already recorded: `routes/deployments.ts` (`headNodeId = nodeIds[0]`, `role: "head"`), `Deployment.node` relation, `ClusterNode.role`.
- Dispatch head-first: `deployments/dgxrun-dispatch.ts` (`masterAddr = clusterNodeIps[0]`).
