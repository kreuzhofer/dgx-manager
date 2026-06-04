# Training Row Detail Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an on-demand "Details" disclosure to each fine-tune job card in the dashboard's training list, showing the parameters captured at launch time (config keys, dataset, recipe, cluster membership, output dir, timestamps). Collapsed by default; per-row state, not persisted.

**Architecture:** Pure logic (cluster-summary + launch-config parsing) lives in two small modules under `packages/dashboard/lib/` so they're unit-testable from the repo-root vitest config. The page component imports those helpers, holds a `detailsExpanded` map keyed by jobId, and renders a `<button aria-expanded aria-controls>` toggle plus a conditionally-rendered details panel inside each card. Zero new npm dependencies — uses the existing `Button` component pattern and unicode glyphs for the caret.

**Tech Stack:** Next.js 15 App Router client component, React 19, TypeScript strict, Tailwind utility classes (existing palette), root `vitest.config.ts` for unit tests, shadcn/ui `Button` (already in use in this file).

---

## Pre-flight (one-time)

Verify the environment before touching code. These commands are read-only.

```bash
cd /home/daniel/src/github/dgx-manager
git status
ls packages/dashboard/lib/                          # should show api.ts, sse.ts, ws.ts, use-debounced-callback.ts
ls vitest.config.ts                                 # repo-root vitest config exists
grep -n 'interface FineTuneJob' packages/dashboard/app/finetune/page.tsx   # line 37
grep -nE 'jobs\.map\(' packages/dashboard/app/finetune/page.tsx | head -3   # card render site
```

If any of these fail, stop and resolve the environment before proceeding.

---

## File Structure

**New files:**

- `packages/dashboard/lib/cluster-summary.ts` — pure helper, single function `formatClusterSummary(job)`. ~25 lines.
- `packages/dashboard/lib/cluster-summary.test.ts` — vitest unit tests for the above.
- `packages/dashboard/lib/launch-config.ts` — pure helper, `parseLaunchConfig(rawJsonOrNull)` + `LAUNCH_CONFIG_LABELS` constant. ~30 lines.
- `packages/dashboard/lib/launch-config.test.ts` — vitest unit tests for the above.

**Modified files:**

- `packages/dashboard/app/finetune/page.tsx` — import the two helpers, add `detailsExpanded` state, add toggle + panel inside the job card render.

**Why split helpers out of `page.tsx`:** the page is already 1134 lines. Two reasons to extract: (1) pure helpers are unit-testable without React Testing Library, which the dashboard package does not have; (2) keeps the page's React tree readable. These are exactly the kind of "data-shaping" boundaries the writing-plans skill flags for separation.

---

## Task 1: Extract `formatClusterSummary` + unit tests

**Files:**
- Create: `packages/dashboard/lib/cluster-summary.ts`
- Create: `packages/dashboard/lib/cluster-summary.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/dashboard/lib/cluster-summary.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { formatClusterSummary } from "./cluster-summary";

describe("formatClusterSummary", () => {
  it("returns the single node's name for jobs without clusterNodes", () => {
    expect(
      formatClusterSummary({
        nodeId: "n1",
        node: { name: "dgx-spark-01", ipAddress: "10.0.0.1" },
        clusterNodes: [],
      }),
    ).toBe("dgx-spark-01");
  });

  it("falls back to nodeId when node relation is missing", () => {
    expect(
      formatClusterSummary({
        nodeId: "node-abc-12345",
        node: null,
        clusterNodes: [],
      }),
    ).toBe("node-abc-12345");
  });

  it("lists multi-node clusters head-first", () => {
    expect(
      formatClusterSummary({
        nodeId: "n2",
        node: { name: "dgx-spark-02", ipAddress: "10.0.0.2" },
        clusterNodes: [
          { node: { name: "dgx-spark-03", ipAddress: "10.0.0.3" }, role: "worker" },
          { node: { name: "dgx-spark-02", ipAddress: "10.0.0.2" }, role: "head" },
          { node: { name: "dgx-spark-04", ipAddress: "10.0.0.4" }, role: "worker" },
        ],
      }),
    ).toBe("3 nodes: dgx-spark-02 (head), dgx-spark-03, dgx-spark-04");
  });

  it("handles a single clusterNodes entry as N=1", () => {
    expect(
      formatClusterSummary({
        nodeId: "n1",
        node: { name: "dgx-spark-01", ipAddress: "10.0.0.1" },
        clusterNodes: [
          { node: { name: "dgx-spark-01", ipAddress: "10.0.0.1" }, role: "head" },
        ],
      }),
    ).toBe("1 node: dgx-spark-01 (head)");
  });
});
```

- [ ] **Step 2: Run the test — expect FAIL**

```bash
cd /home/daniel/src/github/dgx-manager
npx vitest run packages/dashboard/lib/cluster-summary.test.ts
```

Expected: FAIL with `Cannot find module './cluster-summary'` (or similar resolution error).

- [ ] **Step 3: Implement the helper**

Create `packages/dashboard/lib/cluster-summary.ts`:

```typescript
/**
 * Pure helper: render a FineTuneJob's cluster membership as a single
 * human-readable line for the dashboard "details" panel.
 *
 * Examples:
 *   single-node:     "dgx-spark-01"
 *   single-node, no node relation:  "<nodeId>"
 *   multi-node:      "4 nodes: dgx-spark-01 (head), dgx-spark-02, dgx-spark-03, dgx-spark-04"
 */

interface ClusterNodeRow {
  node: { name: string; ipAddress: string };
  role: "head" | "worker" | string;
}

export interface JobClusterShape {
  nodeId: string;
  node: { name: string; ipAddress: string } | null;
  clusterNodes: ClusterNodeRow[];
}

export function formatClusterSummary(job: JobClusterShape): string {
  if (!job.clusterNodes || job.clusterNodes.length === 0) {
    return job.node?.name ?? job.nodeId;
  }

  // Head first, then workers in their existing order.
  const sorted = [...job.clusterNodes].sort((a, b) => {
    if (a.role === "head" && b.role !== "head") return -1;
    if (b.role === "head" && a.role !== "head") return 1;
    return 0;
  });

  const count = sorted.length;
  const noun = count === 1 ? "node" : "nodes";
  const rendered = sorted
    .map((c) => (c.role === "head" ? `${c.node.name} (head)` : c.node.name))
    .join(", ");
  return `${count} ${noun}: ${rendered}`;
}
```

- [ ] **Step 4: Run the test — expect PASS**

```bash
npx vitest run packages/dashboard/lib/cluster-summary.test.ts
```

Expected: 4/4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/dashboard/lib/cluster-summary.ts packages/dashboard/lib/cluster-summary.test.ts
git commit -m "dashboard: extract formatClusterSummary helper

Pure helper for rendering a fine-tune job's cluster membership as a
single human-readable line in the upcoming training-row details panel.
Sorts head-first; falls back to job.node.name or nodeId for single-node
jobs without a clusterNodes array. Unit tests cover single-node,
single-node-without-node-relation, and multi-node head-first ordering."
```

---

## Task 2: Extract `parseLaunchConfig` + label map + unit tests

**Files:**
- Create: `packages/dashboard/lib/launch-config.ts`
- Create: `packages/dashboard/lib/launch-config.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/dashboard/lib/launch-config.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { parseLaunchConfig, LAUNCH_CONFIG_LABELS } from "./launch-config";

describe("parseLaunchConfig", () => {
  it("returns an empty array when config is null", () => {
    expect(parseLaunchConfig(null)).toEqual([]);
  });

  it("returns an empty array when config is an empty string", () => {
    expect(parseLaunchConfig("")).toEqual([]);
  });

  it("returns an empty array when config is invalid JSON", () => {
    expect(parseLaunchConfig("{not json")).toEqual([]);
  });

  it("returns labeled key-value pairs in canonical order", () => {
    const cfg = JSON.stringify({
      lora_alpha: 32,
      learning_rate: 0.0002,
      max_seq_length: 16384,
      lora_r: 16,
    });
    expect(parseLaunchConfig(cfg)).toEqual([
      { key: "learning_rate", label: "Learning rate", value: 0.0002 },
      { key: "max_seq_length", label: "Max seq length", value: 16384 },
      { key: "lora_r", label: "LoRA rank (r)", value: 16 },
      { key: "lora_alpha", label: "LoRA alpha", value: 32 },
    ]);
  });

  it("skips keys whose value is undefined or null", () => {
    const cfg = JSON.stringify({ learning_rate: null, batch_size: 1, max_steps: undefined });
    // Undefined fields are dropped by JSON.stringify; null fields are
    // returned by JSON.parse but should be filtered out here.
    expect(parseLaunchConfig(cfg)).toEqual([
      { key: "batch_size", label: "Batch size", value: 1 },
    ]);
  });

  it("preserves unknown keys at the end with their raw key as label", () => {
    const cfg = JSON.stringify({ learning_rate: 1e-4, save_steps: 50 });
    expect(parseLaunchConfig(cfg)).toEqual([
      { key: "learning_rate", label: "Learning rate", value: 1e-4 },
      { key: "save_steps", label: "save_steps", value: 50 },
    ]);
  });

  it("LAUNCH_CONFIG_LABELS exposes the canonical label order", () => {
    expect(Object.keys(LAUNCH_CONFIG_LABELS)).toEqual([
      "learning_rate",
      "batch_size",
      "max_seq_length",
      "lora_r",
      "lora_alpha",
      "num_train_epochs",
      "max_steps",
    ]);
  });
});
```

- [ ] **Step 2: Run the test — expect FAIL**

```bash
npx vitest run packages/dashboard/lib/launch-config.test.ts
```

Expected: FAIL with module-resolution error.

- [ ] **Step 3: Implement the helper**

Create `packages/dashboard/lib/launch-config.ts`:

```typescript
/**
 * Pure helper: parse a FineTuneJob.config JSON blob into an ordered list
 * of {key, label, value} entries for the dashboard "details" panel.
 *
 * The canonical order matches the launch form's field order so the
 * round-trip from "what I clicked" to "what was captured" is obvious.
 * Keys present in config but not in LAUNCH_CONFIG_LABELS are kept at the
 * end with their raw key as the label so we never silently hide data.
 *
 * Defensive against:
 *   - config === null (job has no config)
 *   - config === ""   (older rows, edge case)
 *   - malformed JSON  (returns [] rather than throwing)
 *   - value === null  (JSON-stringified explicit nulls — filtered out)
 */

export const LAUNCH_CONFIG_LABELS: Record<string, string> = {
  learning_rate: "Learning rate",
  batch_size: "Batch size",
  max_seq_length: "Max seq length",
  lora_r: "LoRA rank (r)",
  lora_alpha: "LoRA alpha",
  num_train_epochs: "Epochs",
  max_steps: "Max steps",
};

export interface LaunchConfigEntry {
  key: string;
  label: string;
  value: unknown;
}

export function parseLaunchConfig(raw: string | null | undefined): LaunchConfigEntry[] {
  if (!raw) return [];
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return [];
  }
  if (!parsed || typeof parsed !== "object") return [];

  const out: LaunchConfigEntry[] = [];
  // First: canonical-order keys present in the config.
  for (const key of Object.keys(LAUNCH_CONFIG_LABELS)) {
    if (key in parsed && parsed[key] !== null && parsed[key] !== undefined) {
      out.push({ key, label: LAUNCH_CONFIG_LABELS[key]!, value: parsed[key] });
    }
  }
  // Then: any extra keys we don't have labels for, raw.
  for (const key of Object.keys(parsed)) {
    if (key in LAUNCH_CONFIG_LABELS) continue;
    if (parsed[key] === null || parsed[key] === undefined) continue;
    out.push({ key, label: key, value: parsed[key] });
  }
  return out;
}
```

- [ ] **Step 4: Run the test — expect PASS**

```bash
npx vitest run packages/dashboard/lib/launch-config.test.ts
```

Expected: 6/6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add packages/dashboard/lib/launch-config.ts packages/dashboard/lib/launch-config.test.ts
git commit -m "dashboard: extract parseLaunchConfig helper

Pure helper for parsing a FineTuneJob.config JSON blob into an ordered
list of {key,label,value} for the upcoming training-row details panel.
Canonical order mirrors the launch form so operators see the same
sequence of fields they clicked. Defensive against null/empty/malformed
JSON and explicit-null values. Unknown keys are appended at the end
with the raw key as label so nothing silently disappears."
```

---

## Task 3: Wire the toggle + details panel into the job card

**Files:**
- Modify: `packages/dashboard/app/finetune/page.tsx`

This is the meat. Each step is one focused edit.

- [ ] **Step 1: Add imports**

Open `packages/dashboard/app/finetune/page.tsx`. Find the import block at the top (currently ends with `import { LogViewer } from "@/components/log-viewer";` at line 7). Add two new imports immediately after the LogViewer line:

```typescript
import { formatClusterSummary } from "@/lib/cluster-summary";
import { parseLaunchConfig } from "@/lib/launch-config";
```

- [ ] **Step 2: Add detailsExpanded state**

Inside the page component (find the existing `useState` block — look for `setJobs` declaration; the new state should sit next to similar per-row maps). Add:

```typescript
const [detailsExpanded, setDetailsExpanded] = useState<Record<string, boolean>>({});

const toggleDetails = useCallback((jobId: string) => {
  setDetailsExpanded((prev) => ({ ...prev, [jobId]: !prev[jobId] }));
}, []);
```

If `useCallback` is not already imported, the import block already imports `useState` and `useCallback` from "react" — verify the existing import line covers both. If `useCallback` is missing, add it to the existing react import (it should already be there per the file's top).

- [ ] **Step 3: Build to confirm no type errors yet**

```bash
cd /home/daniel/src/github/dgx-manager
npm run build --workspace packages/dashboard 2>&1 | tail -20
```

Expected: build succeeds. State and helpers are imported but not yet rendered — this confirms no broken imports before we touch JSX.

- [ ] **Step 4: Add the Details toggle button inside the card**

Locate the job card render. Find the existing action button row inside the `jobs.map(...)` block — search for the line containing `onClick={() => mergeJob(job.id)}` (around line 906). The action button row has Stop / View Logs / Merge / etc. inside a flex container.

In that same flex container, AFTER the existing buttons but BEFORE the closing `</div>`, add the Details toggle:

```tsx
<button
  type="button"
  onClick={() => toggleDetails(job.id)}
  aria-expanded={!!detailsExpanded[job.id]}
  aria-controls={`details-${job.id}`}
  className="text-xs text-gray-400 hover:text-gray-200 transition-colors px-2 py-1 rounded border border-gray-700 hover:border-gray-500"
>
  {detailsExpanded[job.id] ? "▾ Details" : "▸ Details"}
</button>
```

(Unicode glyphs avoid adding lucide-react dependencies. The button is keyboard-focusable by default; `aria-expanded` and `aria-controls` satisfy the accessibility requirement.)

- [ ] **Step 5: Add the conditional details panel below the action row**

Immediately after the action-button-row closing `</div>` (still inside the per-job card block, BEFORE the existing training-metrics / logs sections), add the panel:

```tsx
{detailsExpanded[job.id] && (
  <div
    id={`details-${job.id}`}
    className="mt-2 border-t border-gray-800 pt-2 space-y-1 text-xs"
  >
    {/* Launch config */}
    <div>
      <span className="text-gray-500">Launch config: </span>
      {(() => {
        const entries = parseLaunchConfig(job.config);
        if (entries.length === 0) {
          return <span className="text-gray-400">(none; recipe defaults applied)</span>;
        }
        return (
          <span className="text-gray-200">
            {entries.map((e, i) => (
              <span key={e.key}>
                {i > 0 && <span className="text-gray-600">, </span>}
                <span className="text-gray-400">{e.label}:</span>{" "}
                <span className="font-mono">{String(e.value)}</span>
              </span>
            ))}
          </span>
        );
      })()}
    </div>

    <div>
      <span className="text-gray-500">Dataset: </span>
      <span className="font-mono text-gray-200 break-all">{job.dataset}</span>
    </div>
    <div>
      <span className="text-gray-500">Recipe: </span>
      <span className="font-mono text-gray-200">{job.recipeFile ?? "—"}</span>
    </div>
    <div>
      <span className="text-gray-500">Base model: </span>
      <span className="text-gray-200">{job.baseModel}</span>
    </div>
    <div>
      <span className="text-gray-500">Method: </span>
      <span className="text-gray-200">{job.method}</span>
    </div>
    <div>
      <span className="text-gray-500">Cluster: </span>
      <span className="text-gray-200">{formatClusterSummary(job)}</span>
    </div>
    <div>
      <span className="text-gray-500">Output dir: </span>
      <span className="font-mono text-gray-200 break-all">{job.outputDir ?? "—"}</span>
    </div>
    <div>
      <span className="text-gray-500">Created: </span>
      <span className="text-gray-200">{new Date(job.createdAt).toLocaleString()}</span>
      {job.startedAt && (
        <>
          <span className="text-gray-600"> · </span>
          <span className="text-gray-500">Started: </span>
          <span className="text-gray-200">{new Date(job.startedAt).toLocaleString()}</span>
        </>
      )}
    </div>
  </div>
)}
```

- [ ] **Step 6: Confirm FineTuneJob type carries the fields the panel touches**

Find `interface FineTuneJob` (around line 37). Make sure it declares — in addition to whatever it already has — these fields. If any are missing, add them. Do NOT remove existing fields.

```typescript
  id: string;
  displayName?: string | null;
  recipeFile?: string | null;
  baseModel: string;
  method: string;
  dataset: string;
  config?: string | null;          // JSON-stringified config from launch
  outputDir?: string | null;
  createdAt: string;
  startedAt?: string | null;
  nodeId: string;
  node?: { name: string; ipAddress: string } | null;
  clusterNodes: Array<{
    node: { name: string; ipAddress: string };
    role: "head" | "worker" | string;
  }>;
```

These shapes match what `formatClusterSummary` and `parseLaunchConfig` expect, and they match what the server already returns (verify with: `curl -sS http://localhost:4000/api/finetune | jq '.[0] | keys'`). If `clusterNodes` is absent from the existing interface, this is the change that adds it — make sure call sites that already iterate it still compile.

- [ ] **Step 7: Build to verify the panel renders without TS errors**

```bash
cd /home/daniel/src/github/dgx-manager
npm run build --workspace packages/dashboard 2>&1 | tail -20
```

Expected: build succeeds with no errors.

- [ ] **Step 8: Run all dashboard-related unit tests to confirm we didn't regress the helpers**

```bash
npx vitest run packages/dashboard/lib/
```

Expected: 10/10 tests pass (4 from Task 1 + 6 from Task 2).

- [ ] **Step 9: Visual smoke**

```bash
cd /home/daniel/src/github/dgx-manager
npm run dev:dashboard
```

Open `http://localhost:3000/finetune`. For at least one job in the list:

1. Confirm a "▸ Details" button is visible in the action row.
2. Click it. Confirm the panel expands inline below the action row.
3. Confirm the caret flips to "▾ Details" and `aria-expanded="true"` is set (inspect via dev tools).
4. Confirm each row of the panel shows the expected fields (config entries, dataset path, recipe, base model, method, cluster summary, output dir, timestamps).
5. Click again — confirm it collapses, caret returns to "▸ Details", panel disappears.
6. Expand two different jobs at once — confirm they're independent (one closing doesn't affect the other).

If any of these fail, the most likely cause is a typo in `aria-controls` / `id` or a missing field on the `FineTuneJob` interface. Fix in place before committing.

- [ ] **Step 10: Commit**

```bash
git add packages/dashboard/app/finetune/page.tsx
git commit -m "dashboard: expand-on-demand training-details panel

Adds a per-row \"Details\" disclosure to the fine-tune list. Toggle is a
proper aria-expanded button with aria-controls pointing at the panel id.
Panel shows: launch config (canonical-order key/value pairs from
job.config), dataset path, recipe file, base model, method, cluster
membership (head-first), output dir, created/started timestamps.
Collapsed by default; per-row state (no persistence). Uses the new
formatClusterSummary + parseLaunchConfig helpers from packages/dashboard/lib."
```

---

## Self-Review

**Spec coverage check:**
- ✅ Disclosure toggle on each card (Task 3 Step 4).
- ✅ Collapsed by default (Task 3 Step 2 initializes empty `detailsExpanded`).
- ✅ Per-row local state, no persistence (Task 3 Step 2).
- ✅ Panel renders launch config in canonical order with human labels (Task 2 + Task 3 Step 5).
- ✅ "(none; recipe defaults applied)" fallback when config is null/empty/malformed (Task 2 implementation + Task 3 Step 5).
- ✅ Dataset, recipe, base model, method, cluster, output dir, timestamps all rendered (Task 3 Step 5, in spec order).
- ✅ Cluster summary head-first, single-node fallback to `job.node?.name ?? job.nodeId` (Task 1 implementation).
- ✅ Accessibility: `<button>` with `aria-expanded` and `aria-controls`; panel has matching `id` (Task 3 Steps 4 & 5).
- ✅ No new npm dependencies — unicode glyphs (Task 3 Step 4).
- ✅ Backwards compatible — `parseLaunchConfig(null)` returns `[]` and panel renders the fallback string (Task 2 test 1 + Task 3 Step 5).
- ✅ Unit tests for the two pure helpers (Tasks 1 & 2); visual smoke for the React surface (Task 3 Step 9).
- ✅ Single PR / single file modified for UI (`page.tsx`) + 2 new lib files + 2 new test files.

**Placeholder scan:** No "TBD"/"add appropriate error handling"/"similar to Task N" — every step is concrete code or command.

**Type consistency:** `JobClusterShape` in `cluster-summary.ts` matches the subset of `FineTuneJob` fields declared at Task 3 Step 6 (`nodeId`, `node`, `clusterNodes`). `LaunchConfigEntry.label/value` matches what Task 3 Step 5 renders. `detailsExpanded`/`toggleDetails` names are consistent across Task 3 Steps 2 / 4 / 5.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-13-training-row-detail-expansion.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
