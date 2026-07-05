# Choose + Surface Cluster Head Node — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the operator choose which selected cluster node is the head (rank 0) at deploy time (no new dropdown) and badge the head everywhere, fixing the "which node is head / where's tps" confusion.

**Architecture:** Dashboard-only. A pure `buildClusterNodeIds(headId, selected)` helper puts the chosen head first in `nodeIds` (the server already treats `nodeIds[0]` as head). The deploy form gains a `clusterHeadId` + a crown toggle on each selected node; the deployments list badges the `role:"head"` node. No backend/schema/dispatch/agent change.

**Tech Stack:** Next.js 15 (App Router), React 19, TypeScript, Tailwind, Vitest.

## Global Constraints

- TypeScript strict + ESM. No `packages/agent/src/` or `packages/server/` edits → **no agent version bump**, no server changes.
- The server contract is unchanged and relied upon: `nodeIds[0]` becomes the head (`routes/deployments.ts:276` `headNodeId = nodeIds[0]`), recorded as `ClusterNode.role="head"` and `Deployment.node`.
- `selectedClusterNodes` is a `Set<string>` (`deployments/page.tsx:161`); it's sent via `Array.from(...)` at **line 472** (cluster fine-tune deploy) and **line 531** (vLLM cluster deploy).
- Commit after every task, prefix `feat(head-select):`. `npm test` stays green; `npm run build` (dashboard) clean before commits touching the dashboard app.

## File structure

- `packages/dashboard/lib/cluster-nodes.ts` — pure `buildClusterNodeIds`.
- `packages/dashboard/lib/cluster-nodes.test.ts` — unit tests.
- `packages/dashboard/app/deployments/page.tsx` — `clusterHeadId` state + keep-valid effect + crown toggle + head badge + head-first submit.

---

### Task 1: Pure helper — `buildClusterNodeIds`

**Files:**
- Create: `packages/dashboard/lib/cluster-nodes.ts`
- Test: `packages/dashboard/lib/cluster-nodes.test.ts`

**Interfaces:**
- Produces: `export function buildClusterNodeIds(headId: string | null | undefined, selected: Iterable<string>): string[]` — returns the selected ids as an array with `headId` moved to the front (deduped). If `headId` is null/undefined or not present in `selected`, returns the selected ids in their original order (deduped). Never mutates the input.

- [ ] **Step 1: Write the failing test**
```ts
import { describe, it, expect } from "vitest";
import { buildClusterNodeIds } from "./cluster-nodes.js";

describe("buildClusterNodeIds", () => {
  it("moves the head to the front, preserving the rest order", () => {
    expect(buildClusterNodeIds("c", ["a", "b", "c", "d"])).toEqual(["c", "a", "b", "d"]);
  });
  it("returns selection unchanged when head is absent from selection", () => {
    expect(buildClusterNodeIds("z", ["a", "b"])).toEqual(["a", "b"]);
  });
  it("returns selection unchanged when head is null/undefined", () => {
    expect(buildClusterNodeIds(null, ["a", "b"])).toEqual(["a", "b"]);
    expect(buildClusterNodeIds(undefined, ["a", "b"])).toEqual(["a", "b"]);
  });
  it("single node -> [id]", () => {
    expect(buildClusterNodeIds("a", ["a"])).toEqual(["a"]);
  });
  it("dedupes", () => {
    expect(buildClusterNodeIds("b", ["a", "b", "b", "a"])).toEqual(["b", "a"]);
  });
  it("accepts a Set", () => {
    expect(buildClusterNodeIds("b", new Set(["a", "b", "c"]))).toEqual(["b", "a", "c"]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**
Run: `npx vitest run packages/dashboard/lib/cluster-nodes.test.ts`
Expected: FAIL — `buildClusterNodeIds` not exported.

- [ ] **Step 3: Write minimal implementation**
```ts
/**
 * Order selected cluster node ids with the chosen head first, so the deploy
 * API (which treats nodeIds[0] as the rank-0 head) launches the operator's
 * chosen head. Deduped; head absent/unknown -> selection order unchanged.
 */
export function buildClusterNodeIds(
  headId: string | null | undefined,
  selected: Iterable<string>,
): string[] {
  const ids = Array.from(new Set(selected));
  if (!headId || !ids.includes(headId)) return ids;
  return [headId, ...ids.filter((id) => id !== headId)];
}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `npx vitest run packages/dashboard/lib/cluster-nodes.test.ts` — Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add packages/dashboard/lib/cluster-nodes.ts packages/dashboard/lib/cluster-nodes.test.ts
git commit -m "feat(head-select): buildClusterNodeIds pure helper (head first)"
```

---

### Task 2: Deploy form — choose the head (crown toggle + head-first submit)

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx`

**Interfaces:**
- Consumes: `buildClusterNodeIds` (Task 1).
- Produces: a `clusterHeadId` state that always names one of the selected cluster nodes (or `""`), and cluster deploy bodies whose `nodeIds` are head-first.

- [ ] **Step 1: Import the helper + add state.** Near the top imports add `import { buildClusterNodeIds } from "@/lib/cluster-nodes";` (match the file's existing `@/` alias style). After the `selectedClusterNodes` state (line 161) add:
```tsx
  const [clusterHeadId, setClusterHeadId] = useState<string>("");
```

- [ ] **Step 2: Keep `clusterHeadId` valid** — add an effect after the existing head-reselect effect (the one ending ~line 817). It defaults the head to the first selected node and fixes it if the current head gets deselected:
```tsx
  // Keep the chosen head valid: default to the first selected node; if the
  // current head is deselected, fall back to the new first selected.
  useEffect(() => {
    const ids = Array.from(selectedClusterNodes);
    if (ids.length === 0) { if (clusterHeadId) setClusterHeadId(""); return; }
    if (!clusterHeadId || !selectedClusterNodes.has(clusterHeadId)) setClusterHeadId(ids[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedClusterNodes]);
```

- [ ] **Step 3: Add the crown toggle + head badge to each *selected* row.** In the `clusterCandidates.map((n) => { ... })` render (the `<label>` at ~lines 966-996), inside the label after the node-name `<span className="font-medium">{n.name}</span>`, add (only meaningful when `checked`):
```tsx
                            {checked && (
                              <button
                                type="button"
                                title={clusterHeadId === n.id ? "Head node (rank 0)" : "Make head node"}
                                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setClusterHeadId(n.id); }}
                                className={`text-xs leading-none ${clusterHeadId === n.id ? "text-yellow-400" : "text-gray-600 hover:text-yellow-500"}`}
                              >
                                {clusterHeadId === n.id ? "★ head" : "☆"}
                              </button>
                            )}
```
`e.preventDefault()` + `e.stopPropagation()` are REQUIRED so clicking the crown does not toggle the surrounding `<label>`'s checkbox. Adjust the `ml-auto` on the free-GB span if needed so the crown sits between the name and the stats (keep the existing stats span).

- [ ] **Step 4: Send `nodeIds` head-first.** Replace the two submit sites:
  - Line ~472 (cluster fine-tune): `ftBody.nodeIds = Array.from(selectedClusterNodes);` → `ftBody.nodeIds = buildClusterNodeIds(clusterHeadId, selectedClusterNodes);`
  - Line ~531 (vLLM cluster): `nodeIds: Array.from(selectedClusterNodes),` → `nodeIds: buildClusterNodeIds(clusterHeadId, selectedClusterNodes),`

- [ ] **Step 5: Reset on success.** Wherever `setSelectedClusterNodes(new Set())` is called after a successful deploy (e.g. line ~496, ~562), add `setClusterHeadId("");` alongside it.

- [ ] **Step 6: Verify.** Run `npm run build` (from repo root: `npm run build --workspace=packages/dashboard`, or the repo's dashboard build script) — Expected: compiles clean. Run `npm test` — Expected: still green (no server/unit tests touched here). Manual note: selecting cluster nodes shows one `★ head`; clicking `☆` on another moves it; the checkbox still toggles independently.

- [ ] **Step 7: Commit**
```bash
git add packages/dashboard/app/deployments/page.tsx
git commit -m "feat(head-select): pick head node via crown toggle + send nodeIds head-first"
```

---

### Task 3: Surface the head after deploy (crown badge in the deployments list)

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx` (the deployments-by-node render, ~lines 1271-1310)

**Interfaces:**
- Consumes: the existing per-node `role` in the deployments render (`nodeDeps.map(({ deployment: d, role: nodeRole }) => ...)` at ~line 1295) and/or `d.clusterNodes[].role === "head"` / `d.node`.

- [ ] **Step 1: Badge the head node.** In the deployments-by-node render, the outer grouping is keyed by node name (`Array.from(byNode.entries())...map(([nodeName, nodeDeps]) => ...)` ~line 1271). For each node group, determine if that node is the head of any listed deployment — the simplest signal already present is `nodeRole === "head"` on the per-deployment row (line ~1295); render a crown next to that deployment's node label. Add, where the node/role is displayed for a row:
```tsx
                    {nodeRole === "head" && (
                      <span title="Head node (rank 0) — serves the API, scrapes tps" className="ml-1 text-yellow-400 text-[10px]">★ head</span>
                    )}
```
Place it adjacent to the existing node-name/role text for that row so the head is visually marked. If the render shows `nodeRole` already (e.g. a "head"/"worker" label), replace that label's head case with the crowned badge rather than duplicating.

- [ ] **Step 2: Verify.** `npm run build --workspace=packages/dashboard` clean; `npm test` green. Manual note: a running multi-node deploy shows `★ head` on exactly the head node (matches `/api/deployments` `clusterNodes[].role==="head"`).

- [ ] **Step 3: Commit**
```bash
git add packages/dashboard/app/deployments/page.tsx
git commit -m "feat(head-select): badge the head node in the deployments list"
```

---

## Self-review (author checklist — completed)

- **Spec coverage:** pure head-first builder (T1) ✓; choose head via crown, default first, keep-valid, head-first submit on both paths (T2) ✓; surface head badge after deploy (T3) ✓; no backend/schema/dispatch/agent change (all tasks dashboard-only) ✓.
- **Placeholder scan:** helper has full code + tests; JSX tasks give exact snippets + line anchors + the `preventDefault/stopPropagation` gotcha. No TBD.
- **Type consistency:** `buildClusterNodeIds(headId, selected)` (T1) is the exact signature consumed in T2 submit sites; `clusterHeadId: string` state consistent across T2 steps.
- **Note for implementer:** line numbers are anchors from the current file — read the surrounding JSX and integrate; if a line shifted, match by the quoted code, not the number.
