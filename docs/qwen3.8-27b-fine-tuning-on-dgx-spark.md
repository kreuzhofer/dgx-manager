# Fine-Tuning Qwen 3.8-27B on NVIDIA DGX Spark: SQL LoRA

**Status: INCOMPLETE.** The base was measured and the LoRA was trained, but **the tuned model
was never evaluated** — or if it was, the number was not recorded anywhere in this repo, on
shared storage, or in the job artefacts. This document exists so the half that *was* measured
is not lost, and so finishing it is a matter of running one eval rather than redoing the work.

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
- A 500-step LoRA was trained successfully (`qwen3.8-sql-500step-r2`), loss 3.74 → 0.40,
  adapter merged. **Its SQL accuracy is unknown.**
- Two eval traps were checked and cleared for the base measurement — both would have faked a
  *low* base and therefore an *inflated* lift. They are documented below and apply equally to
  the tuned eval when someone runs it.

---

## What was measured

| | dataset | max_tokens | concurrency | score |
|---|---|---|---|---|
| **Qwen3.8-27B base (bf16)** | `b-mc2/sql-create-context`, 100 examples | 2048 | 4 | **50.0%** (50/100) |
| **Qwen3.8-27B + 500-step LoRA** | same | — | — | **NOT MEASURED** |
| Qwen3.6-27B base (for contrast) | same | 2048 | — | 39% |
| Qwen3.6-27B + 500-step LoRA | same | 2048 | — | 76% |

Base measured via `@dgxrun/qwen3.8-27b-bf16` on `dgx-spark-01`.

**The 3.8 base is 11 points stronger than the 3.6 base on this task before any tuning.** That
is the headline finding, and it is the reason the comparison has to be re-run rather than
inherited: a lift computed against the wrong baseline is wrong by more than most tuning
effects are large.

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

## Operational notes

- **`evaluate.py` imports matplotlib at module scope even in HTTP mode.** Neither the Pi nor
  agenthost has it. Run the eval in a throwaway `python:3.11-slim` container with
  `datasets openai requests matplotlib` and `MPLBACKEND=Agg`.
- **A serving replica on GB10 is OOM-killable by a job that uses no GPU.** Unified memory
  means a host-heavy job competes with the replica for one ~121.6 GiB pool; a BF16 deployment
  holds ~104 GB. Do not stage weights or run a merge on a node that is serving the eval
  endpoint. See the warning in `recipes/dgxrun/qwen3.8-27b-bf16.yaml`.

---

## To finish this

One eval run against the merged model, using the same harness and settings as the base:

```
dataset      b-mc2/sql-create-context, 100 examples (seed 42 split, first N)
max_tokens   2048
concurrency  4
model        /mnt/tank/outputs/cmtmt9dbd57gi2auhok8cptou/merged
```

Then record the result **here**, next to the 50.0% base, and state the lift against **50.0%**
— not against 39%.

If the tuned score lands near the 3.6 guide's 76%, note that the *lift* is nonetheless much
smaller (+26 pp rather than +37 pp) because the 3.8 base starts 11 points higher. Reporting
the lift against the wrong baseline is the single most likely way this comparison gets
misquoted.
