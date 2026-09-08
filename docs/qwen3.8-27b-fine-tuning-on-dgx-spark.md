# Fine-Tuning Qwen 3.8-27B on NVIDIA DGX Spark: SQL LoRA

**Status: COMPLETE as of 2026-09-08.** The tuned model has now been evaluated —
**67.0%**, against a **50.0%** base, a **+17 pp** lift. The tuned number had been missing
because the eval was a hand-run script rather than a recorded `BenchmarkRun` (see issue #93);
it was produced by re-running that eval against the merged model on `dgx-spark-01`.

It supersedes nothing. The sibling guide
[`qwen3.6-27b-fine-tuning-on-dgx-spark.md`](qwen3.6-27b-fine-tuning-on-dgx-spark.md) is the
complete worked example; this is its 3.8 counterpart, stopped one step short.

---

## TL;DR

- **Qwen3.8-27B base scores 50.0% on `b-mc2/sql-create-context`** (50/100), measured
  2026-09-04 on `dgx-spark-01`.
- **Do NOT reuse Qwen3.6's 39% as the 3.8 baseline.** The 3.6 guide reports 39% → 76%
  (+37 pp). Quoting 39% against a 3.8-tuned score would **overstate the lift by ~11 points,
  in the flattering direction.** Always measure the base for the model you actually tuned.
- **The 500-step LoRA scores 67.0%** (67/100) — a **+17 pp lift** over the 50.0% base.
  Unpaired 95% CI on the delta **+3.5 … +30.5 pp**, z=2.48, so the lift is real but the
  interval is wide at n=100.
- **Quoted against the wrong baseline it would read as +28 pp.** That is the trap this
  document exists for.
- Two eval traps were checked and cleared for the base measurement — both would have faked a
  *low* base and therefore an *inflated* lift. They are documented below and apply equally to
  the tuned eval when someone runs it.

---

## What was measured

| | dataset | max_tokens | concurrency | score |
|---|---|---|---|---|
| **Qwen3.8-27B base (bf16)** | `b-mc2/sql-create-context`, 100 examples | 2048 | 4 | **50.0%** (50/100) |
| **Qwen3.8-27B + 500-step LoRA** | same | 2048 | 4 | **67.0%** (67/100) |
| Qwen3.6-27B base (for contrast) | same | 2048 | — | 39% |
| Qwen3.6-27B + 500-step LoRA, **single-node (eff. batch 4)** | same | 2048 | — | **73%** |
| Qwen3.6-27B + 500-step LoRA, multi-node (eff. batch 8) | same | 2048 | — | 76% |

Base measured via `@dgxrun/qwen3.8-27b-bf16` on `dgx-spark-01`.

**The 3.8 base is 11 points stronger than the 3.6 base before any tuning** — and the tuned 3.8
lands *lower* than the tuned 3.6 (67% vs 76%).

Read carefully, because two different mistakes are available here:

- **The lift is +17 pp (50 → 67), not +28 pp.** Quoting the 3.6 base of 39% would inflate it by
  11 points. Unpaired 95% CI **+3.5 … +30.5 pp**, z=2.48 — real, but wide at n=100.
- **The LIFT is much smaller than 3.6's (+17 vs +34 pp), while the ENDPOINT is statistically
  indistinguishable (67% vs 73%).** Both facts follow from the same cause: 3.8 starts 11 points
  higher, so there is less headroom for tuning to recover. A smaller lift here is not evidence
  of worse tunability — it is what a stronger base looks like on a benchmark with a ceiling.
- **Compare against 73%, not 76%.** The 3.6 guide's headline 76% is its *multi-node* run at
  effective batch 8. Its **single-node** 500-step result — same topology, same effective batch
  4, same recipe defaults as this run — is **73%**. That is the apples-to-apples number.
- **Do NOT conclude 3.6 tunes better than 3.8.** Like-for-like the gap is 67% vs 73% = 6 pp,
  z=0.93, **not significant at n=100**. Against the multi-node 76% it is 9 pp, z=1.42, also
  not significant. Either way there is no measured difference between the tuned models.

Tuned eval wall time 264.4 s for 100 examples at concurrency 4.

---

## Training setup

Recipe `recipes/qwen3.8-27b-base-lora` — attention-only LoRA targets, **deliberately identical
to the 3.6 sibling so SQL accuracy is comparable**. Framework deepspeed, method lora,
dataset format sharegpt, container `nvcr.io/nvidia/pytorch:25.11-py3`.

| | |
|---|---|
| LoRA rank / alpha / dropout | 16 / 16 / 0.0 |
| target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| learning rate | 2e-4 |
| batch size / grad accum | 1 / 4 |
| max_seq_length | 256 |
| steps | 500 (`num_train_epochs: 1`) |
| seed | 42 |
| eval_fraction | 0.05 |

The recipe carries three GB10 unified-memory mitigations worth knowing about: **Liger fused
linear CE** (~30% speedup, bit-identical loss), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`,
and `per_device_eval_batch_size=1`.

### The run

`qwen3.8-sql-500step-r2`, job `cmtmt9dbd57gi2auhok8cptou`, completed 2026-09-04 12:02Z after
about 1h34m (started 10:28Z). Output at `/mnt/tank/outputs/cmtmt9dbd57gi2auhok8cptou`.

```
step   1  loss 3.7410
step 100  loss 0.5310
step 200  loss 0.5346
step 300  loss 0.5814
step 400  loss 0.4810
step 500  loss 0.4003
```

Adapter 40 MB, merged model 52 GB, checkpoints at 300/400/500.

**500 steps is far less training than it sounds.** Final `epoch` is **0.0268** — the run covered
**2.7% of a single epoch**. At `batch_size 1 x grad_accum 4` on one node, 500 steps is ~2,000
examples against a ~74.6k-row train split. `num_train_epochs: 1` is set but `max_steps: 500`
caps it long before an epoch completes. So **+17 pp came from seeing roughly one fortieth of
the data, once**, and the loss was still falling at the end (0.481 → 0.400 between steps 400
and 500). If 67% is not good enough, a longer run is the obvious lever — more so than anything
about serving configuration.

The 3.6 recipe's defaults are **identical** (`batch_size 1`, `grad_accum 4`, `max_seq_length
256`, `lr 2e-4`, `lora_r/alpha 16`, same attention-only targets), which is what makes the
single-node 3.6 number a fair comparison and the multi-node one not: 2 ranks doubles the
effective batch to 8, so the same 500 steps see twice the data. Per the 3.6 guide that bought
**+3 pp** (73% → 76%) for roughly 4x the wall time.

Two earlier attempts were stopped before completion — `qwen3.8-sql-smoke5` and
`qwen3.8-sql-500step`, both 2026-09-03. `-r2` is the one that finished.

---

## Eval methodology, and two traps that were checked

Both traps would have produced a **falsely low base**, and therefore a **falsely large lift**.
They are cleared for the base run; re-check them for the tuned run, especially if the prompt
shape changes.

### Reasoning does not fire on SQL prompts

Despite the recipe pinning `reasoning_effort: medium`, SQL prompts produced
`reasoning=0c`, `finish=stop`, 119–183 completion tokens. So the empty-answer failure that
bites Qwen3.8 elsewhere — where the model reasons past the token cap and returns
`finish_reason: length` with **both** `content` and `reasoning_content` empty — is **not
active on this prompt shape**. Do not assume that generalises; it is a property of these
prompts, not of the deployment.

### `normalize_sql` handles markdown fences

The base model wraps SQL in ` ```sql ` fences. `normalize_sql` in `scripts/evaluate.py`
strips them and falls back to the last `SELECT`, so the base model's output is extracted
correctly rather than scored as a miss. This matters: an extraction bug that silently scores
correct answers as wrong is exactly what depressed this fleet's GPQA numbers by ~7 points
until an answer-format instruction was added.

### The split is deterministic, so evals can be parallelised

`train_test_split(test_size=0.05, seed=42)`, first N. **Separately-run base and tuned evals
score the identical examples**, so they can run on different nodes at the same time without
losing pairing. That also means the missing tuned number can be produced now and compared
directly against the 50.0% above — no need to re-measure the base.

---

## How the harness parses model output — read this before comparing against another run

**A 1% base score on this dataset is almost always an extraction bug, not a bad model.** The
base model does not emit bare SQL: it wraps output in markdown fences, adds prose, and
sometimes prefixes a label. A strict exact-match scorer sees none of its answers. This section
documents exactly what `scripts/evaluate.py` does, so a run elsewhere can be made comparable.

### The prompt

A single user message, through the model's own chat template:

```
{schema}\n\n{question}
```

**There is no system prompt and no "reply with only SQL" instruction.** That is deliberate — it
matches `lib/dataset.py: format_example()` so the prompt is what the model was trained on — but
it means the harness *must* be lenient about output shape, because nothing told the model to be
terse.

### `normalize_sql()`, applied to BOTH sides

The predicted string and the ground-truth string go through the **same** function, then are
compared with `==`. Order of operations:

1. **`None` → `""`.** A truncated reasoning response (`finish_reason="length"`,
   `content: null`) becomes an empty prediction and scores as a miss instead of crashing.
2. **Closed markdown block wins.** `` ```(?:sql)?\s*\n?(.*?)``` `` with `DOTALL|IGNORECASE`;
   if it matches, only the captured body is kept.
3. **Otherwise, last-`SELECT` fallback.** Find every `\bSELECT\b` case-insensitively, take
   from the **last** one, then hard-cut at the first of `;`, `` ``` ``, `` \n` ``, `\n\nNote`,
   `\n\nThis `, `\n\nExplanation`. Leading backticks stripped. This is what rescues a
   verbose reasoning model that explains itself and then answers.
4. **Chat-template artifacts removed** — split at and discard from the first occurrence of
   `<end_of_turn>`, `<start_of_turn>`, `<|im_end|>`, `<|im_start|>`, `model`, `user`.
5. **Trailing noise stripped** — `.strip().rstrip(";").rstrip("`").strip()`.
6. **Quotes normalised: single → double.** `s.replace("'", '"')`.
7. **Lowercased and whitespace-collapsed** — `" ".join(s.lower().split())`.

### What that means in practice

**Lenient about** — markdown fences (```` ```sql ```` or bare ```` ``` ````), trailing
semicolons, ALL CASE differences, any whitespace/newline/indentation differences,
single-vs-double quotes, prose before the SQL, prose after the SQL, and chat-template tokens.

**Strict about** — everything else. This is **string equality after normalisation, not
semantic SQL equivalence.** There is no AST parse and no execution. So all of these score as
**wrong** even though they are correct SQL:

| ground truth | prediction | scored |
|---|---|---|
| `SELECT a, b FROM t` | `SELECT b, a FROM t` | ✗ wrong |
| `WHERE x = 1 AND y = 2` | `WHERE y = 2 AND x = 1` | ✗ wrong |
| `SELECT COUNT(*) FROM t` | `SELECT COUNT(1) FROM t` | ✗ wrong |
| `FROM table AS t` | `FROM table t` | ✗ wrong |
| `"value"` | `` `value` `` | ✗ wrong — only **single** quotes are normalised, backticks are stripped only at the string ends |

So our 50.0% is a **normalised-exact-match** number. It is directly comparable only to another
normalised-exact-match run using the same rules. If the other side executes the SQL and
compares result sets, expect their number to be **higher** than ours for the same model —
different metric, not a better model.

### Two footguns worth knowing

**Step 4 splits on the bare words `model` and `user`.** Any SQL containing those as an
identifier is truncated there — `SELECT model FROM cars` becomes `SELECT `. Because the *same*
normalisation is applied to ground truth, the truncation is symmetric and usually still
matches, so it does not cause false misses. It can, however, cause **false positives**: two
genuinely different queries that both truncate to the same prefix compare equal. On a dataset
with `model`/`user` columns this inflates rather than depresses the score.

**`--max-tokens` is load-bearing.** We use 2048. A reasoning model that spends its budget
thinking returns `content: null`, which step 1 scores as a miss — indistinguishable from a
wrong answer. On Qwen3.8 reasoning did not fire on these prompts (measured `reasoning=0c`,
119–183 completion tokens), but do not assume that on a different model or prompt shape. If
your base scores near zero, check the null rate before concluding anything.

### To reproduce comparably elsewhere

```
dataset      b-mc2/sql-create-context
split        dataset.train_test_split(test_size=0.05, seed=42)["test"], first 100
prompt       f"{schema}\n\n{question}"  as a single user turn, model's own chat template
             (no system prompt, no format instruction)
max_tokens   2048
metric       normalize_sql(pred) == normalize_sql(gold), rules above, applied to BOTH sides
```

The split is deterministic, so an independent run scores the identical 100 examples.

## Operational notes

- **`evaluate.py` imports matplotlib at module scope even in HTTP mode.** Neither the Pi nor
  agenthost has it by default. Either a throwaway `python:3.11-slim` container with
  `datasets openai requests matplotlib` and `MPLBACKEND=Agg`, or a venv with those four
  packages — the 2026-09-08 run used a venv on the Pi.

- **⚠ THE HF DATASET LOCK IS OWNED BY ROOT AND WILL BLOCK YOUR EVAL.** This bit twice. The
  training job runs as root in a container and leaves a lock file behind in the shared cache:

  ```
  -rw-r--r-- 1 root root 0 Sep  4 14:29
    /mnt/tank/models/datasets/_mnt_tank_models_datasets_b-mc2___sql-create-context_default_0.0.0_<hash>.lock
  ```

  A later eval running as a normal user dies on it:

  ```
  PermissionError: [Errno 13] Permission denied: '/mnt/tank/models/datasets/_mnt_tank_..._.lock'
  ```

  The dataset *directory* is fine (`daniel:gpio`) — it is only the lock. **Workaround, no root
  needed:** the lock filename encodes the cache path, so pointing at a different cache
  generates a different lock name that you can create.

  ```bash
  mkdir -p ~/hfcache
  cp -r /mnt/tank/models/datasets/b-mc2___sql-create-context ~/hfcache/   # ~21 MB
  find ~/hfcache -name '*.lock' -delete
  HF_DATASETS_CACHE=~/hfcache HF_HUB_OFFLINE=1 python evaluate.py ...
  ```

  Keep `HF_HUB_OFFLINE=1` so it uses the copy rather than re-downloading. Do **not** try to
  delete or chmod the shared lock — it is root-owned and other jobs may rely on that cache.

- **⚠ agenthost has NO `/mnt/tank` mount.** It is the eval runner, but it has no NFS mounts at
  all, so the eval script, the dataset cache and the merged model are all invisible to it.
  SWE-bench works there only because its outputs are local (`~/swebench`). **Run this eval from
  the Pi**, which does have shared storage; it is HTTP-driven, so it costs almost no CPU.

- **dgxrun bind-mounts exactly ONE directory into the container**: `weightsDir`
  (`/mnt/tank/models`) at `/cache/huggingface`. There is no volume field in the recipe schema.
  A model under `/mnt/tank/outputs/...` is therefore **invisible to the container**, and a
  symlink dangles. Use a hardlink tree — same filesystem, zero extra space, same inodes:

  ```bash
  cp -al /mnt/tank/outputs/<jobId>/merged /mnt/tank/models/<name>
  # then in the recipe:  model: /cache/huggingface/<name>
  ```

  Verify with `stat -c %i` on both paths — identical inodes prove a hardlink rather than a
  copy that could drift. Note also that inline `recipeYaml` is exempt from the HF-cache guard
  (`deployments.ts:459`), which is why a local path is accepted at all; a `recipeFile` deploy
  would be rejected.
- **A serving replica on GB10 is OOM-killable by a job that uses no GPU.** Unified memory
  means a host-heavy job competes with the replica for one ~121.6 GiB pool; a BF16 deployment
  holds ~104 GB. Do not stage weights or run a merge on a node that is serving the eval
  endpoint. See the warning in `recipes/dgxrun/qwen3.8-27b-bf16.yaml`.

---

## How the tuned number was produced (2026-09-08)

Merged model served on `dgx-spark-01` as deployment `cmtsg22n75d2p2auh8ykhd0ze`, displayName
`qwen38-sql-lora-merged`, via **inline `recipeYaml` derived from `@dgxrun/qwen3.8-27b-bf16`
with only the model path swapped** — so `reasoning_effort=medium`, `max_num_seqs 8`,
`num_speculative_tokens 5`, `max_model_len 262144` and `gpu_memory_utilization 0.88` all match
what the 50.0% base was measured under. No `--dtype` flag: vLLM's `auto` resolved to
`torch.bfloat16` on both sides, so the null `torch_dtype` in the config is a non-event.
Setting it on one side only would have been the confound.

```
dataset      b-mc2/sql-create-context, 100 examples (seed-42 split, first N)
max_tokens   2048
concurrency  4
endpoint     http://192.168.44.36:8000, served name qwen38-sql-lora-merged
result       67.0% (67/100), wall 264.4 s
```

An earlier fine-tune-path deployment of the same merged model
(`cmtmx6ac25dy52auh3yhkwjt0`, config `{localModelPath, artifactVariant}`, no recipe) would
**not** have carried those serving settings, so a number from it would have been quietly
incomparable to the base. That is why the recipe path matters more than it looks.
