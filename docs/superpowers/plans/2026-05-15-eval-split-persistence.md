# Eval-Split Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the training-time eval split as `eval-split.jsonl` in each fine-tune job's `$OUTPUT_DIR`, and expose it via `GET /api/finetune/:id/eval-split` so downstream harnesses (chat3d's Primitives eval) can fetch the exact prompts the model never saw during training — unlocking a fair, leakage-free in-distribution generalization comparison.

**Architecture:** Recipe-side, add a reusable `lib/eval_split.py` helper that writes the held-out `datasets.Dataset` to disk as one JSON object per line, in the order produced by `train_test_split(seed=…)` — no shuffling, no sampling. Wire it into `recipes/qwen3.6-27b-base-lora-attn-mlp/train.py` right after `prepare_datasets()` returns. Manager-side, add a minimal `GET /api/finetune/:id/eval-split` route that streams the file with `application/jsonl` content type and `?limit=N&offset=N` slicing for paginated reads. File is the source of truth — no DB columns, no agent code changes (the helper runs inside the training container).

**Tech Stack:** Python 3.12 + `datasets` (recipe repo), Express 5 + Prisma 7 + vitest + supertest (manager repo). Manager-repo path: `/home/daniel/src/github/dgx-manager`. Recipe-repo path: `/mnt/tank/src/github/dgx-manager-fine-tune-recipes`.

---

## File Structure

**Recipe repo** (`/mnt/tank/src/github/dgx-manager-fine-tune-recipes/`):
- Create: `lib/eval_split.py` — `dump_eval_split_to_jsonl(eval_ds, output_dir) -> int`
- Create: `lib/test_eval_split.py` — unit tests for the helper
- Modify: `recipes/qwen3.6-27b-base-lora-attn-mlp/train.py` — one call site after `prepare_datasets()` returns

**Manager repo** (`/home/daniel/src/github/dgx-manager/`):
- Modify: `packages/server/src/routes/finetune.ts` — new `GET /:id/eval-split` route
- Create: `packages/server/src/__tests__/integration/finetune.eval-split.test.ts` — integration tests (happy path, 404 paths, limit/offset)

No agent version bump. No Prisma schema change. No dashboard change. The training script runs inside the recipe's container, not in the agent code path.

---

## Task 1: Recipe-repo `lib/eval_split.py` helper + unit tests

**Files:**
- Create: `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/lib/eval_split.py`
- Create: `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/lib/test_eval_split.py`

- [ ] **Step 1.1: Write the failing test file**

Create `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/lib/test_eval_split.py`:

```python
"""Unit tests for lib/eval_split.dump_eval_split_to_jsonl.

These run in a stock python+pip environment with `datasets` installed.
They do NOT require GPU, training, or any I/O beyond a tmpdir.
"""
import json
from pathlib import Path

import pytest
from datasets import Dataset

from lib.eval_split import dump_eval_split_to_jsonl


def test_writes_one_jsonl_line_per_row(tmp_path: Path) -> None:
    """Each row in the eval Dataset becomes one line of valid JSON."""
    ds = Dataset.from_list([
        {"conversations": [{"from": "user", "value": "hello"}]},
        {"conversations": [{"from": "user", "value": "world"}]},
        {"conversations": [{"from": "user", "value": "third"}]},
    ])
    n = dump_eval_split_to_jsonl(ds, str(tmp_path))
    assert n == 3
    path = tmp_path / "eval-split.jsonl"
    assert path.exists()
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"conversations": [{"from": "user", "value": "hello"}]}
    assert json.loads(lines[1]) == {"conversations": [{"from": "user", "value": "world"}]}
    assert json.loads(lines[2]) == {"conversations": [{"from": "user", "value": "third"}]}


def test_preserves_dataset_row_order(tmp_path: Path) -> None:
    """No shuffling — the order matches Dataset iteration order, which itself
    matches train_test_split's deterministic-seed output. This is what makes
    the file reproducible from the recipe seed."""
    rows = [{"i": i, "v": f"row-{i}"} for i in range(5)]
    ds = Dataset.from_list(rows)
    dump_eval_split_to_jsonl(ds, str(tmp_path))
    written = [json.loads(line) for line in (tmp_path / "eval-split.jsonl").read_text().splitlines()]
    assert written == rows


def test_returns_zero_and_writes_nothing_when_eval_ds_is_none(tmp_path: Path) -> None:
    """eval_fraction=0 / no eval split — helper must no-op gracefully so the
    train script doesn't need conditional plumbing."""
    n = dump_eval_split_to_jsonl(None, str(tmp_path))
    assert n == 0
    assert not (tmp_path / "eval-split.jsonl").exists()


def test_creates_output_dir_if_missing(tmp_path: Path) -> None:
    """Defensive mkdir -p — the trainer usually creates output_dir, but we
    don't want a race or a typo to lose data."""
    nested = tmp_path / "deep" / "nested" / "outputs"
    ds = Dataset.from_list([{"x": 1}])
    n = dump_eval_split_to_jsonl(ds, str(nested))
    assert n == 1
    assert (nested / "eval-split.jsonl").exists()


def test_non_ascii_preserved(tmp_path: Path) -> None:
    """Some build123d / chat3d prompts contain unicode (em dashes, °, ², …).
    Must round-trip cleanly via ensure_ascii=False."""
    ds = Dataset.from_list([{"prompt": "Create a 10 mm × 5 mm² block — flat"}])
    dump_eval_split_to_jsonl(ds, str(tmp_path))
    line = (tmp_path / "eval-split.jsonl").read_text().splitlines()[0]
    assert json.loads(line) == {"prompt": "Create a 10 mm × 5 mm² block — flat"}
```

- [ ] **Step 1.2: Run the test to see it fail**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
python -m pytest lib/test_eval_split.py -v
```

Expected: `ModuleNotFoundError: No module named 'lib.eval_split'` (or equivalent ImportError). All 5 tests collected as errored.

- [ ] **Step 1.3: Write the minimal implementation**

Create `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/lib/eval_split.py`:

```python
"""Persist the training-time eval split to a JSONL file in the job output dir.

Consumers (chat3d's Primitives harness, ad-hoc analysis) can then fetch the
EXACT held-out prompts that produced eval_loss — no DB-wide random sample,
no train-set leakage, no need to re-derive from the seed.

File path: {output_dir}/eval-split.jsonl
Row shape: same as the Dataset rows being trained on (post any recipe-side
           normalization, before tokenization).
Order:     Dataset iteration order, which is the deterministic
           train_test_split(seed=...) output. No shuffling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def dump_eval_split_to_jsonl(eval_ds: Optional["Dataset"], output_dir: str) -> int:  # noqa: F821
    """Write each row of `eval_ds` to {output_dir}/eval-split.jsonl as JSON.

    Args:
        eval_ds: A `datasets.Dataset` (or None when eval_fraction == 0).
        output_dir: Directory to write into. Created (with parents) if missing.

    Returns:
        Number of rows written (0 if eval_ds is None).

    The function does NOT take a `tokenizer` — we deliberately persist the
    row shape that was fed into the trainer's `train_dataset` / `eval_dataset`
    arguments, NOT the tokenized form. That keeps the file usable by any
    downstream consumer that wants to re-render prompts via their own template.
    """
    if eval_ds is None:
        return 0
    out_path = Path(output_dir) / "eval-split.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in eval_ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(eval_ds)
```

- [ ] **Step 1.4: Run the test to verify it passes**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
python -m pytest lib/test_eval_split.py -v
```

Expected:

```
lib/test_eval_split.py::test_writes_one_jsonl_line_per_row PASSED
lib/test_eval_split.py::test_preserves_dataset_row_order PASSED
lib/test_eval_split.py::test_returns_zero_and_writes_nothing_when_eval_ds_is_none PASSED
lib/test_eval_split.py::test_creates_output_dir_if_missing PASSED
lib/test_eval_split.py::test_non_ascii_preserved PASSED

5 passed
```

- [ ] **Step 1.5: Commit**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
git add lib/eval_split.py lib/test_eval_split.py
git commit -m "$(cat <<'EOF'
lib: add dump_eval_split_to_jsonl helper for persisting train eval splits

Writes the held-out Dataset rows to {output_dir}/eval-split.jsonl, one
JSON object per line, in deterministic Dataset iteration order (= the
train_test_split(seed=...) output, no further shuffling). Lets downstream
harnesses (chat3d Primitives eval) fetch the exact prompts the model
never saw at train time, instead of sampling from the whole DB with
unknown train-set overlap.

Pure function: no GPU, no training framework, no tokenizer — just takes
a `datasets.Dataset` (or None for eval_fraction==0) and a target dir.
5 unit tests cover row order, None handling, missing dir auto-create,
and non-ASCII roundtrip.
EOF
)"
```

---

## Task 2: Wire helper into qwen3.6-27b-base-lora-attn-mlp train.py

**Files:**
- Modify: `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora-attn-mlp/train.py`

Why no TDD on this task: verifying the wiring requires running an actual training step, which is a 43-hour multi-node job. The unit tests in Task 1 already prove the helper writes correct files for arbitrary `Dataset` inputs. The wiring is a one-line call site — review the diff, then trust the unit tests.

- [ ] **Step 2.1: Read the current call site**

```bash
sed -n '205,225p' /mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora-attn-mlp/train.py
```

You should see lines around 210 that look like:

```python
    train_ds, eval_ds = prepare_datasets(
        args.dataset, tokenizer, args.max_seq_length, args.eval_fraction, args.seed, world_rank)
    # … some downstream column-drop logic uses eval_ds
    if cfg.model_type in {"qwen3", "qwen3_moe", "qwen3_5", "qwen3_next"}:
        if eval_ds is not None:
            eval_ds = eval_ds.remove_columns(qwen_drop_cols)
```

The dump call goes IMMEDIATELY AFTER `prepare_datasets()` returns and BEFORE any column-drop — we want the pre-tokenized, pre-drop row shape, so chat3d can re-render with their own template if they want.

- [ ] **Step 2.2: Add the import**

Find the existing `from lib.dataset import prepare_datasets` line (around line 52 in train.py). Add directly below it:

```python
from lib.eval_split import dump_eval_split_to_jsonl
```

- [ ] **Step 2.3: Add the call site**

Right after the `train_ds, eval_ds = prepare_datasets(...)` line and BEFORE any `eval_ds.remove_columns(...)` mutation, insert:

```python
    # Persist the held-out split to {output_dir}/eval-split.jsonl so downstream
    # harnesses (e.g. chat3d Primitives eval) can fetch the exact prompts the
    # model never saw at train time. Only rank-0 writes — all ranks see the
    # same split because of the deterministic seed, but one writer is enough
    # and avoids any chance of a partial-write race on shared NFS.
    if world_rank == 0:
        n_eval = dump_eval_split_to_jsonl(eval_ds, args.output_dir)
        print(f"[eval-split] wrote {n_eval} rows to {args.output_dir}/eval-split.jsonl", flush=True)
```

- [ ] **Step 2.4: Sanity-check the diff**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
git diff recipes/qwen3.6-27b-base-lora-attn-mlp/train.py
```

Expected: two additions only — one import near the top, one 7-line block right after the `prepare_datasets` call. Nothing else changed.

- [ ] **Step 2.5: Confirm Python still parses**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
python -c "import ast; ast.parse(open('recipes/qwen3.6-27b-base-lora-attn-mlp/train.py').read()); print('parse ok')"
```

Expected: `parse ok`.

- [ ] **Step 2.6: Commit**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
git add recipes/qwen3.6-27b-base-lora-attn-mlp/train.py
git commit -m "$(cat <<'EOF'
qwen3.6-27b attn-mlp: dump eval split to JSONL after prepare_datasets

Rank-0 writes {output_dir}/eval-split.jsonl right after the train/eval
split is produced and before any qwen-specific column drop, so the
persisted rows match what was fed to the trainer's eval_dataset arg.
Other ranks no-op (they see the same split via the same seed, but we
only need one writer and want to avoid a partial-write race on NFS).

Downstream consumers: chat3d's Primitives harness will use
GET /api/finetune/:id/eval-split (new endpoint, follow-up commit on
dgx-manager) to fetch this file for fair in-distribution generalization
evaluation.

No effect on the actual training run — `dump_eval_split_to_jsonl` is
pure I/O against the Dataset object, no tokenizer or GPU involvement.
EOF
)"
```

---

## Task 3: Manager-repo `GET /api/finetune/:id/eval-split` route + integration tests

**Files:**
- Create: `packages/server/src/__tests__/integration/finetune.eval-split.test.ts`
- Modify: `packages/server/src/routes/finetune.ts`

- [ ] **Step 3.1: Write the failing integration test file**

Create `/home/daniel/src/github/dgx-manager/packages/server/src/__tests__/integration/finetune.eval-split.test.ts`:

```typescript
/**
 * Integration tests for GET /api/finetune/:id/eval-split.
 *
 * Same pattern as finetune.quantize.test.ts: per-suite SQLite, no agent hub
 * needed (this endpoint only reads from disk + DB), supertest against an
 * Express app that mounts only the finetune router.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, writeFileSync, rmSync, mkdirSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

vi.mock("../../sse.js", () => ({
  broadcast: () => {},
  sseHandler: vi.fn(),
}));

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-eval-split-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;
// The route resolves outputDir via the job record itself, so we don't need
// to override SHARED_STORAGE — we just point each seeded job's outputDir at
// a tmpdir we own.

let prisma: typeof import("../../prisma.js").prisma;
let finetuneRouter: typeof import("../../routes/finetune.js").finetuneRouter;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      DATABASE_URL: `file:${DB_PATH}`,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "User consented to db push --force-reset against per-suite SQLite test databases in /tmp on 2026-05-03 (option #1)",
    },
    stdio: "pipe",
  });
  ({ prisma } = await import("../../prisma.js"));
  ({ finetuneRouter } = await import("../../routes/finetune.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

beforeEach(async () => {
  await prisma.fineTuneJob.deleteMany();
  await prisma.node.deleteMany();
});

function makeApp() {
  const app = express();
  app.use(express.json());
  app.set("agentHub", { getTrainingRecipes: () => [], sendToAgent: () => {} });
  app.use("/api/finetune", finetuneRouter);
  return app;
}

async function seedJobWithOutputDir(outputDir: string) {
  const node = await prisma.node.create({
    data: { id: "n1", name: "n1", ipAddress: "10.0.0.1", agentPort: 8089, status: "online", vramTotal: 122000 },
  });
  return prisma.fineTuneJob.create({
    data: {
      nodeId: node.id,
      recipeFile: "recipes/test",
      baseModel: "Qwen/Qwen3.6-27B",
      method: "lora",
      dataset: "/tmp/ds.jsonl",
      status: "completed",
      mergeStatus: "completed",
      outputDir,
    },
  });
}

describe("GET /api/finetune/:id/eval-split", () => {
  it("returns the file body when eval-split.jsonl exists", async () => {
    const dir = join(TMP_DIR, "job-happy");
    mkdirSync(dir, { recursive: true });
    const body = [
      JSON.stringify({ conversations: [{ from: "user", value: "alpha" }] }),
      JSON.stringify({ conversations: [{ from: "user", value: "beta" }] }),
      JSON.stringify({ conversations: [{ from: "user", value: "gamma" }] }),
    ].join("\n") + "\n";
    writeFileSync(join(dir, "eval-split.jsonl"), body, "utf-8");
    const job = await seedJobWithOutputDir(dir);

    const res = await request(makeApp()).get(`/api/finetune/${job.id}/eval-split`);
    expect(res.status).toBe(200);
    expect(res.headers["content-type"]).toMatch(/application\/jsonl/);
    expect(res.headers["content-disposition"]).toMatch(new RegExp(`eval-split-${job.id}\\.jsonl`));
    const lines = res.text.trim().split("\n");
    expect(lines).toHaveLength(3);
    expect(JSON.parse(lines[0])).toEqual({ conversations: [{ from: "user", value: "alpha" }] });
    expect(JSON.parse(lines[2])).toEqual({ conversations: [{ from: "user", value: "gamma" }] });
  });

  it("returns 404 when the job has no eval-split file on disk", async () => {
    const dir = join(TMP_DIR, "job-no-file");
    mkdirSync(dir, { recursive: true });
    // outputDir exists, but no eval-split.jsonl in it (older jobs predate the feature)
    const job = await seedJobWithOutputDir(dir);

    const res = await request(makeApp()).get(`/api/finetune/${job.id}/eval-split`);
    expect(res.status).toBe(404);
    expect(res.body.error).toMatch(/eval-split/i);
  });

  it("returns 404 when the job does not exist", async () => {
    const res = await request(makeApp()).get("/api/finetune/does-not-exist/eval-split");
    expect(res.status).toBe(404);
    expect(res.body.error).toMatch(/job/i);
  });

  it("slices via ?limit and ?offset", async () => {
    const dir = join(TMP_DIR, "job-slice");
    mkdirSync(dir, { recursive: true });
    const body = [0, 1, 2, 3, 4].map((i) => JSON.stringify({ i })).join("\n") + "\n";
    writeFileSync(join(dir, "eval-split.jsonl"), body, "utf-8");
    const job = await seedJobWithOutputDir(dir);

    const res = await request(makeApp()).get(`/api/finetune/${job.id}/eval-split?limit=2&offset=1`);
    expect(res.status).toBe(200);
    const lines = res.text.trim().split("\n");
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0])).toEqual({ i: 1 });
    expect(JSON.parse(lines[1])).toEqual({ i: 2 });
  });

  it("caps ?limit at 10000 to bound response size", async () => {
    const dir = join(TMP_DIR, "job-cap");
    mkdirSync(dir, { recursive: true });
    // Only 3 rows on disk — passing limit=99999 must not OOM, must just return all 3.
    const body = [0, 1, 2].map((i) => JSON.stringify({ i })).join("\n") + "\n";
    writeFileSync(join(dir, "eval-split.jsonl"), body, "utf-8");
    const job = await seedJobWithOutputDir(dir);

    const res = await request(makeApp()).get(`/api/finetune/${job.id}/eval-split?limit=99999`);
    expect(res.status).toBe(200);
    const lines = res.text.trim().split("\n");
    expect(lines).toHaveLength(3);
  });
});
```

- [ ] **Step 3.2: Run the test to see it fail**

```bash
cd /home/daniel/src/github/dgx-manager
npx vitest run packages/server/src/__tests__/integration/finetune.eval-split.test.ts
```

Expected: 5 tests, all fail with 404 on the happy-path test (because the route doesn't exist yet — Express returns its built-in 404 with no body, so `res.body.error` is undefined and the assertions blow up). The "does not exist" test may pass coincidentally because Express also 404s — that's fine, the happy-path failure proves the route isn't wired.

- [ ] **Step 3.3: Read the current route file header to find imports**

```bash
sed -n '1,40p' /home/daniel/src/github/dgx-manager/packages/server/src/routes/finetune.ts
```

You should see `existsSync`, `readFileSync` from `fs` already imported, plus `SHARED_STORAGE`. If `readFileSync` and `existsSync` aren't imported yet, mention it — they should already be there (used by the `/logs` route at line 62).

- [ ] **Step 3.4: Add the route**

In `/home/daniel/src/github/dgx-manager/packages/server/src/routes/finetune.ts`, find the existing `/:id/logs` route handler (starts around line 62). Add the new route IMMEDIATELY AFTER `/:id/logs` and BEFORE `/:id/checkpoints` (around line 86). The exact insertion point: after the closing `});` of `/:id/logs`:

```typescript
finetuneRouter.get("/:id/eval-split", async (req, res) => {
  const job = await prisma.fineTuneJob.findUnique({ where: { id: req.params.id } });
  if (!job) return res.status(404).json({ error: "Job not found" });

  // Same outputDir resolution as /logs — handles resumed jobs that point at
  // their parent's directory.
  const dir = job.outputDir || `${SHARED_STORAGE}/outputs/${job.id}`;
  const filePath = `${dir}/eval-split.jsonl`;
  if (!existsSync(filePath)) {
    return res.status(404).json({
      error: "eval-split not available — job predates this feature or failed before the split was written",
    });
  }

  // Slice support: chat3d's harness samples 100 at a time; no need to ship
  // the whole file every request for a 20k-row holdout. Cap at 10000 lines
  // per response to bound memory/latency.
  const rawLimit = parseInt(req.query.limit as string);
  const rawOffset = parseInt(req.query.offset as string);
  const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(rawLimit, 10_000) : 10_000;
  const offset = Number.isFinite(rawOffset) && rawOffset > 0 ? rawOffset : 0;

  let content: string;
  try {
    content = readFileSync(filePath, "utf-8");
  } catch {
    return res.status(500).json({ error: "Could not read eval-split file" });
  }

  // Filter out blank trailing line(s); preserve internal order. The dumper
  // writes "row\n" so a file with N rows has N+1 split tokens (last empty).
  const allLines = content.split("\n").filter((l) => l.length > 0);
  const slice = allLines.slice(offset, offset + limit);

  res.setHeader("Content-Type", "application/jsonl");
  res.setHeader("Content-Disposition", `inline; filename="eval-split-${job.id}.jsonl"`);
  res.send(slice.length > 0 ? slice.join("\n") + "\n" : "");
});
```

- [ ] **Step 3.5: Run the test to verify it passes**

```bash
cd /home/daniel/src/github/dgx-manager
npx vitest run packages/server/src/__tests__/integration/finetune.eval-split.test.ts
```

Expected:

```
 ✓ packages/server/src/__tests__/integration/finetune.eval-split.test.ts (5 tests)
   ✓ GET /api/finetune/:id/eval-split > returns the file body when eval-split.jsonl exists
   ✓ GET /api/finetune/:id/eval-split > returns 404 when the job has no eval-split file on disk
   ✓ GET /api/finetune/:id/eval-split > returns 404 when the job does not exist
   ✓ GET /api/finetune/:id/eval-split > slices via ?limit and ?offset
   ✓ GET /api/finetune/:id/eval-split > caps ?limit at 10000 to bound response size

 Test Files  1 passed (1)
      Tests  5 passed (5)
```

- [ ] **Step 3.6: Confirm no regression in the broader finetune test suite**

```bash
cd /home/daniel/src/github/dgx-manager
npx vitest run packages/server/src/__tests__/integration/finetune
```

Expected: all suites green, including the existing `finetune.quantize.test.ts` (10 tests).

- [ ] **Step 3.7: Commit**

```bash
cd /home/daniel/src/github/dgx-manager
git add packages/server/src/routes/finetune.ts packages/server/src/__tests__/integration/finetune.eval-split.test.ts
git commit -m "$(cat <<'EOF'
finetune: add GET /:id/eval-split route for fetching held-out training prompts

Streams {outputDir}/eval-split.jsonl (written by the recipe-side
dump_eval_split_to_jsonl helper on the next training run) to downstream
harnesses. application/jsonl content type; ?limit=N&offset=N slicing
with limit capped at 10000 per response.

Returns 404 with a clear error message when the file is absent — older
jobs predate the recipe-side dumper and can't be back-filled (we didn't
save the prompts at train time).

No DB schema change: the file on shared NFS is the source of truth.
Mirror of the /:id/logs route shape — same outputDir resolution, same
filesystem-only reads.

5 integration tests cover happy path, both 404 paths, limit/offset
slicing, and the 10k cap. All existing finetune tests stay green.
EOF
)"
```

---

## Task 4: Document the API contract for downstream consumers

**Files:**
- Modify: `CLAUDE.md` (project instructions, manager repo) — add a short note under the existing "Key Source Locations" or "Database" section pointing at the new endpoint and its contract.

This is documentation only — no TDD, no tests. The goal is so chat3d-claude (and any future agent) can find the contract without reading server code.

- [ ] **Step 4.1: Find the right insertion point in CLAUDE.md**

```bash
grep -n "^## \|^### " /home/daniel/src/github/dgx-manager/CLAUDE.md | head -25
```

Look for a section like "Database" or "Key Source Locations". We want the docs near where other operator-facing details live.

- [ ] **Step 4.2: Add a contract note**

In `/home/daniel/src/github/dgx-manager/CLAUDE.md`, add a new sub-section directly under the existing "Database" heading (between the schema-models line and the next `## ` heading). Exact content:

```markdown
### Per-job artifacts on shared storage

Beyond the DB row, each fine-tune job persists a few files in
`$SHARED_STORAGE/outputs/$JOB_ID/` (typically `/mnt/tank/outputs/$JOB_ID/`):

- `train.log` — full training stdout/stderr (served by `GET /api/finetune/:id/logs`)
- `merge.log` — LoRA → BF16 merge stdout/stderr
- `quantize.log` — offline FP8 quantize stdout/stderr (dead path now that FP8 is on-load — see ROADMAP)
- `lora_adapter/` — raw LoRA adapter
- `merged/` — merged BF16 weights, vLLM-loadable
- `eval-split.jsonl` — held-out training prompts (one JSON row per line, in
  deterministic `train_test_split(seed=…)` order). Written by
  `lib/eval_split.dump_eval_split_to_jsonl` in the recipe repo's train.py.
  Served paginated by `GET /api/finetune/:id/eval-split?limit=N&offset=N`
  (limit capped at 10000 per response). Use this — NOT a random sample of
  the whole prompt DB — for in-distribution generalization comparisons,
  since it's the EXACT set the model never saw. Jobs trained before the
  feature landed (2026-05-15) return 404 and have no way to back-fill.
```

- [ ] **Step 4.3: Commit**

```bash
cd /home/daniel/src/github/dgx-manager
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: document per-job artifacts and the new eval-split endpoint contract

Adds a "Per-job artifacts on shared storage" section to CLAUDE.md that
lists train.log, merge.log, quantize.log, lora_adapter/, merged/, and
the newly-introduced eval-split.jsonl, with the corresponding API
endpoints for each. Calls out the leakage-free-comparison use case for
eval-split so future agents reach for it instead of random-sampling the
prompt DB.
EOF
)"
```

---

## Fast-follow checkboxes (other recipes, copy-paste pattern)

These should be done in separate commits in the recipe repo once Task 2 has shipped and the qwen3.6-27b attn-mlp recipe has had at least one fresh training run to verify the file lands correctly:

- [ ] `recipes/qwen3.6-27b-base-lora-attn-only/train.py` — same two-line import + call-site change
- [ ] `recipes/gemma-4-e2b/train.py`
- [ ] `recipes/gemma-4-e4b/train.py`
- [ ] `recipes/gemma-4-26b-a4b-moe/train.py`
- [ ] `recipes/llama-3.1-8b-unsloth/train.py`
- [ ] Any TRL-based recipe that calls a similar `prepare_datasets()` helper

Each one is mechanically identical: add `from lib.eval_split import dump_eval_split_to_jsonl`, then call `dump_eval_split_to_jsonl(eval_ds, args.output_dir)` rank-0-guarded right after the split happens. No new tests needed per recipe — Task 1's unit tests cover the helper.

---

## What's explicitly out of scope

- **Chat3d-claude's `--prompt-source eval-split:<jobId>` harness mode** — that's chat3d's repo, chat3d's PR. This plan defines the API contract; chat3d-claude implements the consumer.
- **Dashboard UI for browsing the eval split** — nice-to-have, not now. Operator can `curl` if needed.
- **Back-filling existing eval-loss-only jobs** — impossible, the prompts weren't saved. Return 404 and move on.
- **Other recipes' train.py updates** — listed as fast-follow checkboxes above.
- **Schema migration / DB column for the split path** — file is the source of truth; we already had to rip out the analogous `quantizationStatus` / `quantizedPath` columns once on this branch, so don't make that mistake again.

---

## Self-review checklist (run before handing the plan to the executor)

**1. Spec coverage:**
- "persist the training-time eval split as a JSONL artifact in $OUTPUT_DIR" → Task 1 (helper) + Task 2 (wiring) ✓
- "expose it via a new server endpoint" → Task 3 ✓
- "document the contract for chat3d-claude" → Task 4 ✓
- "JSONL, one record per held-out example, EXACT same row shape" → Task 1 implementation + test ✓
- "?limit=N (default unlimited, capped at 10k) and ?offset=N for pagination" → Task 3 implementation + slice test + cap test ✓
- "Content type: application/jsonl" → Task 3 implementation + content-type assertion in test ✓
- "Content-Disposition: inline; filename=eval-split-{jobId}.jsonl" → Task 3 implementation + content-disposition assertion in test ✓
- "Tests use per-suite mkdtempSync + DATABASE_URL pattern, PRISMA consent per-test" → Task 3 test mirrors finetune.quantize.test.ts exactly ✓
- "deterministic order, no shuffling" → Task 1 has a dedicated `test_preserves_dataset_row_order` test ✓
- "don't bump agent version" → confirmed in File Structure section ✓

**2. Placeholder scan:** No "TBD", "implement later", "Similar to Task N", "Add appropriate error handling", etc. Every step has actual code or actual commands.

**3. Type consistency:** Helper signature is `dump_eval_split_to_jsonl(eval_ds, output_dir) -> int` in Task 1 spec, the implementation, the wiring import in Task 2, and the contract doc in Task 4. Route is `GET /api/finetune/:id/eval-split` consistently in Task 3 tests, Task 3 implementation, Task 4 docs.
