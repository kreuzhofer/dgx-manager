# Qwen 3.6-27B LoRA Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a working LoRA fine-tune recipe for Qwen 3.6-27B (dense, hybrid GatedDeltaNet+attention, multimodal) and verify it end-to-end through the DGX Manager API: train → merge → deploy → eval. Phase c smoke-tests plumbing; phase a benchmarks SQL accuracy vs base; phase b is gated (separate plan).

**Architecture:** Fork `recipes/qwen3.6-35b-a3b-base-lora/` in `kreuzhofer/dgx-manager-fine-tune-recipes`. Drop MoE `target_parameters` and the MoE capacity check; keep all carryover fixes (NCCL timeout bump, qwen drop-cols, frozen MTP/router loop, gradient checkpointing). Reuse the existing `scripts/merge_qwen3moe.py` (its Case 1 path handles dense 2D LoRA correctly and bypasses the multimodal-wrapper-strip bug). All training and inference operations issued via the DGX Manager HTTP API only.

**Tech Stack:**
- Python 3.11, transformers 4.51+, PEFT 0.17+, TRL 0.16+, DeepSpeed (ZeRO-3), accelerate, torch (bf16)
- vLLM 0.20+ for serving (the FP8-aware spark-vllm-docker container, same as the inference recipe)
- DGX Manager HTTP API at `http://${MANAGER_ADVERTISE_HOST:-192.168.44.36}:4000`
- Two repos:
  - `kreuzhofer/dgx-manager-fine-tune-recipes` at `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/` — recipe code
  - `dgx-manager` at `/home/daniel/src/github/dgx-manager/` — write-up doc only
- Spec: `docs/superpowers/specs/2026-05-04-qwen36-27b-training-design.md` (commits `c5917de` + `f1f85cb`)

**Hardware:** Nodes 3 + 4 only. Nodes 1 + 2 are reserved for active inference workloads. **OOM contingency: pause and ask the user before touching nodes 1+2.**

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/recipe.yaml` | Create | Recipe metadata + defaults |
| `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/train.py` | Create | LoRA training entry point — forked from 35B-A3B |
| `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/entrypoint.sh` | Copy | Container init (unchanged) |
| `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/launch.sh` | Copy | torchrun launcher (unchanged) |
| `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/ds_config.json` | Copy | DeepSpeed ZeRO-3 config (unchanged) |
| `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/scripts/merge_qwen3moe.py` | Reuse | Already handles dense 2D LoRA in Case 1 |
| `/home/daniel/src/github/dgx-manager/docs/qwen3.6-27b-fine-tuning-on-dgx-spark.md` | Create (after phases) | Practitioner write-up parallel to the 35B doc |

The training/merge/deploy code on the manager side (`packages/server`, `packages/agent`) is **unchanged**.

---

## Group A — Pre-flight verification

### Task 1: Verify `q_proj` only appears on full-attention layers

**Files:**
- None modified. Output goes to stdout only.

**Why:** PEFT's `target_modules=["q_proj","k_proj","v_proj","o_proj"]` matches by name suffix. If GatedDeltaNet layers also have `q_proj`-named tensors, LoRA would attach to stateful linear-attn layers — same regression mode as Mamba layers. We must verify the suffix discriminates before training.

- [ ] **Step 1: Write a one-shot verification script**

Create a temporary helper script at `/tmp/verify_qwen27b_layers.py`:

```python
"""Static check: does q_proj appear only on full-attention layers in
Qwen 3.6-27B? If yes, suffix-only target_modules is safe. If q_proj
appears on every layer, train.py needs `layers_to_transform=[…]` to
restrict LoRA to full-attention layer indices."""
import json
from collections import Counter
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

MODEL = "Qwen/Qwen3.6-27B"

idx_path = hf_hub_download(MODEL, "model.safetensors.index.json")
weight_map = json.load(open(idx_path))["weight_map"]

q_proj_layers = sorted(
    {int(k.split(".layers.")[1].split(".")[0])
     for k in weight_map
     if ".q_proj." in k and ".layers." in k}
)
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
total_layers = cfg.num_hidden_layers

print(f"q_proj appears on {len(q_proj_layers)} of {total_layers} layers")
print(f"q_proj layer indices: {q_proj_layers}")

# Scan for linear-attn / GatedDeltaNet markers per layer
markers = ["linear_attn", "gated_delta_net", "in_proj", "out_proj", "A_log", "dt_proj"]
per_layer_markers = {}
for k in weight_map:
    if ".layers." not in k:
        continue
    layer = int(k.split(".layers.")[1].split(".")[0])
    suffix = k.split(".layers.")[1].split(".", 1)[1] if "." in k.split(".layers.")[1] else ""
    for m in markers:
        if m in suffix:
            per_layer_markers.setdefault(layer, set()).add(m)

print("\nLinear-attn markers by layer (first 8 layers shown):")
for layer in sorted(per_layer_markers)[:8]:
    print(f"  layer {layer}: {sorted(per_layer_markers[layer])}")

if len(q_proj_layers) == total_layers:
    print("\n!! q_proj on EVERY layer — train.py MUST use layers_to_transform")
elif len(q_proj_layers) < total_layers:
    full_attn_layers = q_proj_layers
    print(f"\n>> q_proj only on {len(full_attn_layers)} layers — suffix matching is safe.")
    print(f">> full-attention layer indices: {full_attn_layers}")
```

- [ ] **Step 2: Run the script**

Run:
```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
python /tmp/verify_qwen27b_layers.py 2>&1 | tee /tmp/verify_qwen27b_layers.out
```

Expected: prints `q_proj appears on K of N layers`. **Both K==N and K<N are valid outcomes.**

- [ ] **Step 3: Decide based on output**

- If `K == N` (q_proj on every layer): record the full-attention layer indices via the `linear_attn` markers and add `layers_to_transform=[…]` arg in Task 5 below.
- If `K < N` (q_proj only on a subset): suffix matching is safe; no `layers_to_transform` needed. Record the count for the recipe documentation.

Write the conclusion as a one-line note prepended to `/tmp/verify_qwen27b_layers.out`.

**No commit.** This is exploratory analysis; the conclusion drives Task 5.

---

### Task 2: Confirm nodes 3 + 4 are available and idle

**Files:** None.

**Why:** Hardware is the binding constraint. We need both target nodes online and not running other training jobs.

- [ ] **Step 1: List nodes via API**

Run:
```bash
curl -s http://192.168.44.36:4000/api/nodes | jq '.[] | {id,ipAddress,name,status,gpuCount}'
```

Expected: a list including the two nodes designated as "node 3" and "node 4" (verify with the user if names are ambiguous), both with `status: "online"`.

- [ ] **Step 2: Check no active fine-tune jobs on nodes 3+4**

Run:
```bash
curl -s http://192.168.44.36:4000/api/finetune | jq '[.[] | select(.status == "running" or .status == "starting")]'
```

Expected: empty array `[]`. If a running job is on node 3 or 4, **STOP and ask the user** how to proceed.

- [ ] **Step 3: Check no active deployments occupying nodes 3+4**

Run:
```bash
curl -s http://192.168.44.36:4000/api/deployments | jq '[.[] | select(.status == "running" or .status == "starting")]'
```

Expected: any running deployments are on nodes 1+2 only. If node 3 or 4 has a live deployment, **STOP and ask the user**.

**No commit.** This is operational verification.

---

## Group B — Recipe construction

### Task 3: Create the recipe directory and copy unchanged scaffolding

**Files:**
- Create: `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/`
- Copy: `entrypoint.sh`, `launch.sh`, `ds_config.json` from `qwen3.6-35b-a3b-base-lora/`

- [ ] **Step 1: Create directory and copy scaffolding files**

Run:
```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
mkdir -p recipes/qwen3.6-27b-base-lora
cp recipes/qwen3.6-35b-a3b-base-lora/entrypoint.sh recipes/qwen3.6-27b-base-lora/
cp recipes/qwen3.6-35b-a3b-base-lora/launch.sh     recipes/qwen3.6-27b-base-lora/
cp recipes/qwen3.6-35b-a3b-base-lora/ds_config.json recipes/qwen3.6-27b-base-lora/
```

- [ ] **Step 2: Verify copies are byte-identical to source**

Run:
```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
diff recipes/qwen3.6-35b-a3b-base-lora/entrypoint.sh recipes/qwen3.6-27b-base-lora/entrypoint.sh
diff recipes/qwen3.6-35b-a3b-base-lora/launch.sh     recipes/qwen3.6-27b-base-lora/launch.sh
diff recipes/qwen3.6-35b-a3b-base-lora/ds_config.json recipes/qwen3.6-27b-base-lora/ds_config.json
```

Expected: all three diffs return empty output.

- [ ] **Step 3: Don't commit yet** — `recipe.yaml` and `train.py` come in the next tasks. We commit the whole recipe in one go at Task 6.

---

### Task 4: Write `recipe.yaml`

**Files:**
- Create: `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/recipe.yaml`

- [ ] **Step 1: Write the file**

Contents:

```yaml
recipe_version: "1"
name: Qwen3.6-27B-Base-LoRA
description: LoRA fine-tune for Qwen 3.6-27B BASE (dense, hybrid GatedDeltaNet+Gated-Attention, multimodal) on DGX Spark
base_model: Qwen/Qwen3.6-27B
framework: deepspeed
method: lora
dataset_format: sharegpt

container:
  image: nvcr.io/nvidia/pytorch:25.11-py3
  name: dgx-finetune

scripts:
  entrypoint: entrypoint.sh
  train: train.py
  launch: launch.sh
  ds_config: ds_config.json
  # Reuses the 35B-A3B merge script: its Case 1 path
  # (compute_delta with base_tensor.ndim == 2) handles dense 2D LoRA on
  # q/k/v/o_proj. The MoE 3D path is silently skipped for dense models.
  # We don't use scripts/merge.py because it relies on PEFT's
  # merge_and_unload(), which strips Qwen 3.6's multimodal wrapper and
  # produces a model_type=qwen3_5_text leaf config that vLLM can't load.
  merge: scripts/merge_qwen3moe.py

defaults:
  learning_rate: 0.0002
  batch_size: 1
  gradient_accumulation_steps: 4
  max_seq_length: 256
  num_train_epochs: 1
  max_steps: -1
  lora_r: 16
  lora_alpha: 16
  lora_dropout: 0.0
  # Attention projections only. GatedDeltaNet (linear-attention) layers are
  # stateful and not LoRA-friendly — same rule as the 35B-A3B Mamba layers.
  # If Task 1 found q_proj on every layer, see train.py for the
  # layers_to_transform restriction.
  lora_target_modules: q_proj,k_proj,v_proj,o_proj
  seed: 42
  eval_fraction: 0.05

hardware:
  min_nodes: 2
  gpus_per_node: 1
  vram_estimate_mb: 40960   # provisional; refined after phase c measures peak rss

deploy:
  container: vllm-node
  gpu_memory_utilization: 0.85
  max_model_len: 4096
```

- [ ] **Step 2: Verify YAML parses**

Run:
```bash
python3 -c "import yaml; print(yaml.safe_load(open('/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/recipe.yaml')))" | head -5
```

Expected: prints a dict starting with `{'recipe_version': '1', 'name': 'Qwen3.6-27B-Base-LoRA', ...}`. No traceback.

---

### Task 5: Write `train.py`

**Files:**
- Create: `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/train.py`

This file is forked from `recipes/qwen3.6-35b-a3b-base-lora/train.py` with four targeted changes:
1. Drop `target_parameters` from `LoraConfig`
2. Replace MoE capacity check with a dense capacity check (`< 0.001%`)
3. Conditional `layers_to_transform` if Task 1 found q_proj on every layer
4. Updated module docstring

The `frozen` loop and all other patches stay byte-identical.

- [ ] **Step 1: Write the file**

Full contents:

```python
"""Fine-tune Qwen 3.6-27B BASE (dense, hybrid GatedDeltaNet + Gated-Attention,
multimodal) with DeepSpeed ZeRO-3 + LoRA.

Architecture notes (vs the 35B-A3B recipe this was adapted from):
- Dense, not MoE — no fused expert tensors. Drop `target_parameters`;
  rely on `target_modules=q/k/v/o_proj` only.
- Still hybrid: full-attention layers + GatedDeltaNet (linear-attention)
  layers. Same LoRA rule as the 35B's Mamba layers — only target the
  full-attention projections, never linear-attn `in_proj` / `out_proj`.
  Verified via static check that `q_proj` only appears on full-attn
  layers (Task 1 of the impl plan); if that ever changes, switch to
  `layers_to_transform=[…]` enumerating full-attn indices.
- Multimodal wrapper present (vision tower + LM). PEFT will only attach
  to language-model layers because the vision tower's attention
  projections live under different module names.
- Merge happens via scripts/merge_qwen3moe.py (its Case 1 path).
  The generic scripts/merge.py would strip the multimodal wrapper.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lib.patches import apply_all, flush_page_cache, fix_gemma4_use_cache
from lib.dataset import prepare_datasets
from lib.logging import setup_logging, LogMetricsCallback
from lib.tokenizer import setup_tokenizer
from lib.args import add_common_args, add_deepspeed_args

import argparse, gc, torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig

apply_all()

# Increase NCCL timeout BEFORE any process group init.
# ZeRO-3 loading does hundreds of broadcasts during from_pretrained;
# the default 30-min timeout is too short for 27B on DGX Spark.
import datetime
torch.distributed.constants.default_pg_timeout = datetime.timedelta(hours=4)
torch.distributed.constants.default_pg_nccl_timeout = datetime.timedelta(hours=4)


def main():
    p = argparse.ArgumentParser(description="Fine-tune Qwen 3.6-27B with DeepSpeed ZeRO-3 + LoRA")
    add_common_args(p)
    add_deepspeed_args(p)
    args = p.parse_known_args()[0]
    world_rank = int(os.environ.get("RANK", 0))

    setup_logging(args.output_dir)

    if args.ds_config:
        from transformers.integrations.deepspeed import HfDeepSpeedConfig
        HfDeepSpeedConfig(args.ds_config)

    tokenizer = setup_tokenizer(args.model_name)

    train_ds, eval_ds = prepare_datasets(
        args.dataset, tokenizer, args.max_seq_length, args.eval_fraction, args.seed, world_rank)

    # Drop Gemma-only columns the dataset library always emits.
    qwen_drop_cols = [c for c in ("token_type_ids", "mm_token_type_ids") if c in train_ds.column_names]
    if qwen_drop_cols:
        train_ds = train_ds.remove_columns(qwen_drop_cols)
        if eval_ds is not None:
            eval_ds = eval_ds.remove_columns(qwen_drop_cols)

    print(f"[Rank {world_rank}] Loading model: {args.model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, trust_remote_code=True)
    print(f"[Rank {world_rank}] Model loaded.", flush=True)
    gc.collect()
    flush_page_cache()

    # Dense LoRA: just attention projections. PEFT's suffix matcher will
    # automatically skip GatedDeltaNet layers because they don't expose
    # q_proj/k_proj/v_proj/o_proj names (verified in Task 1).
    lora_kwargs = dict(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
    )
    # If Task 1 found that q_proj also appears on linear-attn layers,
    # uncomment and populate the line below with the full-attention layer
    # indices (e.g., every Nth layer).
    # lora_kwargs["layers_to_transform"] = [3, 7, 11, …]

    model = get_peft_model(model, LoraConfig(**lora_kwargs))

    # Defensive freeze (no-op on dense 27B if no router/MTP exists).
    frozen = 0
    for name, p in model.named_parameters():
        if ".gate.weight" in name or "router" in name or "mtp." in name:
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1

    if world_rank == 0:
        model.print_trainable_parameters()
        trainable, total = model.get_nb_trainable_parameters()
        pct = trainable / total * 100
        print(f"[LoRA capacity] {trainable:,} / {total:,} = {pct:.4f}%", flush=True)
        print(f"[Freeze] froze {frozen} router/mtp parameter tensors", flush=True)
        # Dense q/k/v/o LoRA on r=16 lands well under 0.5% (typically
        # 0.02-0.1%). Fail loud only if effectively nothing matched —
        # which would mean the q_proj/k_proj/v_proj/o_proj suffixes
        # didn't hit any module. Threshold chosen to allow normal dense
        # ranges while catching catastrophic mis-targeting.
        if pct < 0.001:
            raise RuntimeError(
                f"LoRA capacity suspiciously low ({pct:.4f}%): "
                f"target_modules likely didn't match any layer. Check "
                f"the model's parameter naming. Aborting."
            )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    fix_gemma4_use_cache(model)  # Generic use_cache=True; safe no-op for non-Gemma archs.

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=train_ds, eval_dataset=eval_ds,
        callbacks=[LogMetricsCallback()],
        args=SFTConfig(
            output_dir=args.output_dir, per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps, num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate, bf16=True, optim="adamw_torch",
            warmup_steps=5, logging_steps=1,
            save_strategy="steps", save_steps=args.save_steps,
            save_total_limit=args.save_total_limit, save_only_model=args.save_only_model,
            eval_strategy="steps" if eval_ds else "no", eval_steps=args.eval_steps,
            seed=args.seed, max_length=args.max_seq_length, packing=False,
            report_to="none", deepspeed=args.ds_config, skip_memory_metrics=True,
            remove_unused_columns=False))

    if world_rank == 0:
        print(f"Starting training: {len(train_ds)} examples, "
              f"max_seq_length={args.max_seq_length}, batch_size={args.batch_size}", flush=True)

    gc.collect()
    flush_page_cache()

    resume = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() == "true":
        resume = True
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(f"{args.output_dir}/lora_adapter")
    if world_rank == 0:
        tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")
        print(f"LoRA adapter saved to {args.output_dir}/lora_adapter", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint-check via py_compile**

Run:
```bash
python3 -m py_compile /mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/train.py
echo "Exit: $?"
```

Expected: `Exit: 0` and no output above it.

- [ ] **Step 3: Apply Task 1 conclusion**

Open the file. If Task 1 concluded `q_proj on every layer`, replace the commented `# lora_kwargs["layers_to_transform"]` line with the actual list discovered there. Otherwise, leave the comment as-is for future readers.

---

### Task 6: Commit the recipe to `kreuzhofer/dgx-manager-fine-tune-recipes`

**Files:**
- 5 new files in `recipes/qwen3.6-27b-base-lora/`

- [ ] **Step 1: Stage and commit**

Run:
```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
git status
git add recipes/qwen3.6-27b-base-lora/
git commit -m "$(cat <<'EOF'
add Qwen 3.6-27B LoRA recipe

Fork of qwen3.6-35b-a3b-base-lora with MoE-specific bits removed:
- target_modules=q/k/v/o_proj only (no target_parameters)
- dense capacity check (<0.001% threshold)
- min_nodes=2

Reuses scripts/merge_qwen3moe.py (Case 1 dense 2D path) to bypass the
multimodal-wrapper-strip bug that affects scripts/merge.py on Qwen 3.6.
EOF
)"
```

- [ ] **Step 2: Verify the commit landed**

Run:
```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
git log -1 --stat
```

Expected: a single new commit listing the 5 files added under `recipes/qwen3.6-27b-base-lora/`.

- [ ] **Step 3: Push (only if user explicitly approves a push)**

Hold off on `git push` until phase c green — keeps the working repo clean of an unverified recipe. Surface this to the user as a question before pushing.

---

## Group C — Phase c: end-to-end smoke

Goal: 5-step run on a tiny SQL slice, then merge → deploy → smoke. **No accuracy claim.** Total wall time budget: ~30 minutes (most of which is `from_pretrained` for 27B over the network).

### Task 7: Verify the manager has discovered the new recipe

**Files:** None.

- [ ] **Step 1: Trigger a recipes-cache refresh**

The agent rescans the recipes repo on agent reconnect; force a refresh by hitting the recipes endpoint:

```bash
curl -s http://192.168.44.36:4000/api/training-recipes | jq '.[] | select(.file | contains("qwen3.6-27b-base-lora")) | {file, base_model, name, method}'
```

Expected: returns a single recipe object referencing the new path. If empty, the agent hasn't seen the commit yet — wait 30 s and retry, or restart the agent on nodes 3+4 via:

```bash
curl -s -X POST http://192.168.44.36:4000/api/nodes/<node-id>/agent-reconnect
```

(check `nodes.ts` for the exact endpoint name; if it doesn't exist, `git pull` on the node's mounted recipes path and the agent will pick up the next time it touches the directory).

---

### Task 8: Start the phase-c smoke run via API

**Files:** None.

- [ ] **Step 1: POST a 5-step training job**

Identify node 3 and node 4 IDs from the API. Then:

```bash
NODE3=<node-3-id>
NODE4=<node-4-id>

curl -s -X POST http://192.168.44.36:4000/api/finetune \
  -H 'Content-Type: application/json' \
  -d @- <<EOF | tee /tmp/phase-c-job.json | jq
{
  "nodeIds": ["${NODE3}", "${NODE4}"],
  "recipeFile": "recipes/qwen3.6-27b-base-lora",
  "dataset": "b-mc2/sql-create-context",
  "config": {
    "max_steps": 5,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_seq_length": 256,
    "save_steps": 5,
    "eval_steps": 5,
    "eval_fraction": 0.001
  }
}
EOF

JOB_ID=$(jq -r '.id' /tmp/phase-c-job.json)
echo "Phase-c job: $JOB_ID"
```

Expected: HTTP 201 with a job object including `id`, `status: "starting"`, `outputDir: "/mnt/tank/outputs/<id>"`.

- [ ] **Step 2: Confirm status moves to `running`**

Run:
```bash
for i in 1 2 3 4 5; do
  STATUS=$(curl -s http://192.168.44.36:4000/api/finetune/$JOB_ID | jq -r '.status')
  echo "[$(date +%T)] status=$STATUS"
  [ "$STATUS" = "running" ] && break
  sleep 30
done
```

Expected: status reaches `running` within ~3 minutes. If status sticks at `starting` for >10 min, check `train.log` for early errors:

```bash
tail -50 /mnt/tank/outputs/$JOB_ID/train.log
```

- [ ] **Step 3: OOM contingency**

If `train.log` shows `CUDA out of memory` or DeepSpeed reports OOM during model loading: **STOP, do NOT auto-escalate to nodes 1+2.** Stop the job and surface to the user:

```bash
curl -s -X POST http://192.168.44.36:4000/api/finetune/$JOB_ID/stop
curl -s -X DELETE http://192.168.44.36:4000/api/finetune/$JOB_ID
```

Then ask the user how to proceed (stop one of nodes 1+2 inference workloads to free a node, or use 8-bit AdamW).

---

### Task 9: Watch `train.log` to completion

**Files:** None.

- [ ] **Step 1: Tail until LoRA adapter saved**

Per memory: tail the file directly, not the API status.

```bash
tail -F /mnt/tank/outputs/$JOB_ID/train.log
```

Expected within ~15 minutes of the run starting:
- `[LoRA capacity] X,XXX / 27,XXX,XXX,XXX = 0.0XX%` (well above 0.001%, well below 1%)
- 5 step lines: `{'loss': X.X, 'grad_norm': X.X, 'learning_rate': ...}`
- `LoRA adapter saved to /workspace/outputs/<id>/lora_adapter`

- [ ] **Step 2: Verify final job status is `completed`**

Run:
```bash
curl -s http://192.168.44.36:4000/api/finetune/$JOB_ID | jq '{status, completedAt}'
```

Expected: `{"status": "completed", "completedAt": "<ISO>"}`. If `failed`, read `train.log` and stop here for the user.

- [ ] **Step 3: Verify adapter file exists on shared storage**

Run:
```bash
ls -la /mnt/tank/outputs/$JOB_ID/lora_adapter/
```

Expected: `adapter_config.json` + `adapter_model.safetensors` (~few hundred MB), tokenizer files.

---

### Task 10: Trigger merge and validate the merged artifact

**Files:** None.

- [ ] **Step 1: POST merge**

Run:
```bash
curl -s -X POST http://192.168.44.36:4000/api/finetune/$JOB_ID/merge | jq
```

Expected: HTTP 200, response indicates merge has started. Server passes `scripts/merge_qwen3moe.py` per the recipe's `scripts.merge` field.

- [ ] **Step 2: Watch merge.log to completion**

Run:
```bash
tail -F /mnt/tank/outputs/$JOB_ID/merge.log
```

Expected within ~10 minutes:
- `Loading adapter config from .../adapter_config.json`
- `rank=16, alpha=16, scaling=1.0`
- `Found N LoRA pairs to merge` where N matches what the LoRA capacity check implied (one per (layer × {q,k,v,o}_proj) on full-attn layers)
- Per-shard `+ merged into model.language_model.layers.X.self_attn.{q,k,v,o}_proj.weight`
- `Merged N / N LoRA pairs` (no unapplied)
- `Output: /mnt/tank/outputs/<id>/merged-clean` or similar

- [ ] **Step 3: Sanity-check merged artifact structure**

Run:
```bash
MERGED=/mnt/tank/outputs/$JOB_ID/merged-clean   # or whatever path merge.log printed
ls "$MERGED" | head -30
jq '.architectures, .model_type, .text_config // {}' "$MERGED/config.json"
ls "$MERGED"/*.safetensors | wc -l
```

Expected:
- `config.json` `architectures` and `model_type` MATCH the base model's (`Qwen3_5ForConditionalGeneration` / `qwen3_5` or whatever Qwen 3.6 uses — verify against `Qwen/Qwen3.6-27B`'s own config).
- Number of `.safetensors` shards matches the base model's shard count.
- `model.safetensors.index.json` is present.

- [ ] **Step 4: Verify no duplicate `lm_head`**

Run:
```bash
python3 - <<'EOF'
import json, os, sys
merged = os.environ["MERGED"]
idx = json.load(open(f"{merged}/model.safetensors.index.json"))["weight_map"]
lm_head_keys = [k for k in idx if "lm_head" in k]
print(f"lm_head-related keys: {lm_head_keys}")
assert len({k.split('.')[-1]: k for k in lm_head_keys}.values()) == len(lm_head_keys), "duplicates!"
print("OK — no duplicate lm_head")
EOF
```

Expected: prints `OK — no duplicate lm_head`. If duplicates exist, the merge produced a broken artifact — stop and investigate `merge_qwen3moe.py`'s Case 1 path.

---

### Task 11: Deploy the merged model and smoke-test

**Files:** None.

- [ ] **Step 1: POST deploy**

Run:
```bash
curl -s -X POST http://192.168.44.36:4000/api/finetune/$JOB_ID/deploy \
  -H 'Content-Type: application/json' \
  -d "{\"nodeId\": \"$NODE3\"}" | tee /tmp/phase-c-deploy.json | jq

DEPLOY_ID=$(jq -r '.id' /tmp/phase-c-deploy.json)
```

Expected: HTTP 201 with a deployment object, `status: "starting"` or `"pending"`.

- [ ] **Step 2: Wait for deployment to be running**

```bash
for i in $(seq 1 20); do
  STATUS=$(curl -s http://192.168.44.36:4000/api/deployments/$DEPLOY_ID | jq -r '.status')
  echo "[$(date +%T)] deploy status=$STATUS"
  [ "$STATUS" = "running" ] && break
  sleep 30
done
```

Expected: reaches `running` within ~10 minutes. If failure, read deployment logs:

```bash
curl -s http://192.168.44.36:4000/api/deployments/$DEPLOY_ID/logs | tail -100
```

- [ ] **Step 3: Resolve the served endpoint URL**

Run:
```bash
ENDPOINT=$(curl -s http://192.168.44.36:4000/api/deployments/$DEPLOY_ID | jq -r '.endpoint // .url // ("http://" + .nodeIp + ":" + (.port|tostring))')
echo "Endpoint: $ENDPOINT"
```

(Adjust the jq path based on whatever shape the deployment object has. If unsure, dump the whole object first: `curl -s http://192.168.44.36:4000/api/deployments/$DEPLOY_ID | jq`.)

- [ ] **Step 4: Smoke-test the endpoint with a SQL prompt**

```bash
curl -sf -X POST "$ENDPOINT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.6-27B",
    "messages": [
      {"role": "user", "content": "CREATE TABLE employees (id INT, name VARCHAR(100), salary INT)\n\nFind employees with salary > 50000"}
    ],
    "max_tokens": 256,
    "temperature": 0.0
  }' | tee /tmp/phase-c-smoke.json | jq '.choices[0].message.content'
```

Expected: HTTP 200 with a non-empty `content` field. Any reasonable response counts — accuracy is not the goal here. If the response is byte-identical to the base model output, that means LoRA didn't actually merge — stop and investigate.

- [ ] **Step 5: If smoke passes, declare phase c green and commit nothing**

Phase c is operational verification, not code. No commit at this step.

---

### Task 12: Clean up the phase-c deployment

**Files:** None.

Per memory `feedback_cleanup_failed_deployments.md`: even successful test deployments should be cleaned to avoid trails.

- [ ] **Step 1: Stop deployment**

```bash
curl -s -X POST http://192.168.44.36:4000/api/deployments/$DEPLOY_ID/stop
```

- [ ] **Step 2: Delete deployment record**

```bash
curl -s -X DELETE http://192.168.44.36:4000/api/deployments/$DEPLOY_ID
```

Expected: HTTP 200 / 204. The phase-c finetune job (`$JOB_ID`) and its outputs stay — they're a useful smoke-baseline.

---

### Task 13: Update `vram_estimate_mb` based on phase-c peak rss

**Files:**
- Modify: `/mnt/tank/src/github/dgx-manager-fine-tune-recipes/recipes/qwen3.6-27b-base-lora/recipe.yaml` (the `hardware.vram_estimate_mb` field)

- [ ] **Step 1: Look up peak rss from the agent metrics**

The manager records `MetricSnapshot` rows during training. Query peak system memory used during the phase-c run:

```bash
curl -s "http://192.168.44.36:4000/api/finetune/$JOB_ID/metrics" | jq '[.[] | .systemMemoryUsedGb // .vramUsedMb // empty] | max'
```

(Exact field name depends on schema — adjust based on what `GET /api/finetune/:id/metrics` returns.)

- [ ] **Step 2: Update the YAML**

If peak RSS was, say, 36 GB → set `vram_estimate_mb: 38912` (a small headroom buffer over the observed peak). If we couldn't measure (e.g., metrics missing): leave at `40960` and add a comment `# unverified — phase c metrics endpoint returned no rows`.

```bash
# Edit recipe.yaml — change vram_estimate_mb line in place.
```

- [ ] **Step 3: Commit**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
git add recipes/qwen3.6-27b-base-lora/recipe.yaml
git commit -m "qwen3.6-27b: refine vram_estimate_mb from phase-c peak rss"
```

---

## Group D — Phase a: SQL benchmark

Same recipe, real runs. **Both runs use nodes 3+4. OOM contingency is the same as Task 8 step 3.**

### Task 14: Start the 50-step training

**Files:** None.

- [ ] **Step 1: POST a 50-step job**

```bash
curl -s -X POST http://192.168.44.36:4000/api/finetune \
  -H 'Content-Type: application/json' \
  -d @- <<EOF | tee /tmp/phase-a-50step.json | jq
{
  "nodeIds": ["${NODE3}", "${NODE4}"],
  "recipeFile": "recipes/qwen3.6-27b-base-lora",
  "dataset": "b-mc2/sql-create-context",
  "config": {
    "max_steps": 50
  }
}
EOF

JOB_50=$(jq -r '.id' /tmp/phase-a-50step.json)
echo "50-step job: $JOB_50"
```

Other defaults (lr, batch, lora_r, etc.) come from the recipe — same as the 35B-A3B baseline for direct comparability.

- [ ] **Step 2: Tail train.log**

```bash
tail -F /mnt/tank/outputs/$JOB_50/train.log
```

Expected: ~50 step lines, monotonically decreasing eval loss, ~2-3 hours wall time (per the 35B-A3B doc, step time was 170s — 27B should be similar or faster).

- [ ] **Step 3: Verify completion**

```bash
curl -s http://192.168.44.36:4000/api/finetune/$JOB_50 | jq '{status, completedAt}'
```

Expected: `status=completed`. If `failed`, read train.log and surface to user.

---

### Task 15: Merge the 50-step adapter and deploy

- [ ] **Step 1: Merge**

```bash
curl -s -X POST http://192.168.44.36:4000/api/finetune/$JOB_50/merge
tail -F /mnt/tank/outputs/$JOB_50/merge.log   # ~10 minutes
```

Expected: same per-shard merge output as phase c, with the same number of LoRA pairs.

- [ ] **Step 2: Deploy**

```bash
curl -s -X POST http://192.168.44.36:4000/api/finetune/$JOB_50/deploy \
  -H 'Content-Type: application/json' \
  -d "{\"nodeId\": \"$NODE3\"}" | tee /tmp/phase-a-50step-deploy.json | jq
DEPLOY_50=$(jq -r '.id' /tmp/phase-a-50step-deploy.json)
```

Wait until deployment status is `running` (poll loop same as Task 11 step 2).

---

### Task 16: Establish 27B base baseline (if not cached)

**Files:** None.

If a base 27B endpoint is not already deployed (the cost-analysis doc mentions one at `Qwen/Qwen3.6-27B-FP8` — that's FP8 inference; the base bf16 might or might not be deployed), deploy one on whichever of nodes 3 or 4 is free.

- [ ] **Step 1: Check existing deployments for a base 27B**

```bash
curl -s http://192.168.44.36:4000/api/deployments | jq '[.[] | select(.modelName | contains("Qwen3.6-27B"))]'
```

If one exists serving the base bf16 (NOT the FP8 quant — eval should be apples-to-apples against tuned bf16), record its endpoint URL.

- [ ] **Step 2: If no base bf16 endpoint, deploy one on node 4**

(This may need to wait until the 50-step deploy on node 3 is running; one deployment per node.)

```bash
curl -s -X POST http://192.168.44.36:4000/api/deployments \
  -H 'Content-Type: application/json' \
  -d "{\"nodeId\":\"$NODE4\",\"model\":\"Qwen/Qwen3.6-27B\",\"container\":\"vllm-node\"}"
```

(Use the existing 27B-base recipe if there is one; otherwise minimal vLLM config.)

---

### Task 17: Run eval — 50 steps vs base, both token budgets

**Files:**
- Create: `/mnt/tank/results/qwen3.6-27b-50step/eval-512.json`
- Create: `/mnt/tank/results/qwen3.6-27b-50step/eval-2048.json`

- [ ] **Step 1: Eval at max_tokens=512 (compresses base verbosity)**

```bash
mkdir -p /mnt/tank/results/qwen3.6-27b-50step

cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
python scripts/evaluate.py \
  --base-endpoint <BASE_27B_ENDPOINT> \
  --base-served-name Qwen/Qwen3.6-27B \
  --tuned-endpoint $(curl -s http://192.168.44.36:4000/api/deployments/$DEPLOY_50 | jq -r '.endpoint') \
  --tuned-served-name Qwen/Qwen3.6-27B \
  --num-examples 100 --concurrency 4 \
  --max-tokens 512 \
  --seed 42 \
  --results-dir /mnt/tank/results/qwen3.6-27b-50step/eval-512
```

Expected: prints accuracy for base and tuned; writes a results JSON.

- [ ] **Step 2: Eval at max_tokens=2048**

Same command as Step 1 but `--max-tokens 2048` and `--results-dir /mnt/tank/results/qwen3.6-27b-50step/eval-2048`.

- [ ] **Step 3: Record the four numbers**

Write to `/tmp/phase-a-results-50.json`:
```json
{
  "base@512": <pct>, "tuned-50@512": <pct>,
  "base@2048": <pct>, "tuned-50@2048": <pct>
}
```

---

### Task 18: Run the 500-step training

- [ ] **Step 1: Stop the 50-step deployment to free a node** (if one of the nodes is needed for the next training)

```bash
curl -s -X POST http://192.168.44.36:4000/api/deployments/$DEPLOY_50/stop
curl -s -X DELETE http://192.168.44.36:4000/api/deployments/$DEPLOY_50
```

(Keep the merged-50 artifact on disk; we may want to re-eval later.)

- [ ] **Step 2: POST a 500-step job**

```bash
curl -s -X POST http://192.168.44.36:4000/api/finetune \
  -H 'Content-Type: application/json' \
  -d @- <<EOF | tee /tmp/phase-a-500step.json | jq
{
  "nodeIds": ["${NODE3}", "${NODE4}"],
  "recipeFile": "recipes/qwen3.6-27b-base-lora",
  "dataset": "b-mc2/sql-create-context",
  "config": { "max_steps": 500 }
}
EOF

JOB_500=$(jq -r '.id' /tmp/phase-a-500step.json)
```

Expected wall time: ~24 hours (matches 35B-A3B at the same step count).

- [ ] **Step 3: Tail train.log**

```bash
tail -F /mnt/tank/outputs/$JOB_500/train.log
```

Watch for normal loss progression (~3.7 → ~0.6 over 500 steps based on 35B-A3B precedent), no NaNs, no spikes.

- [ ] **Step 4: Verify completion**

```bash
curl -s http://192.168.44.36:4000/api/finetune/$JOB_500 | jq '{status, completedAt}'
```

---

### Task 19: Merge, deploy, eval the 500-step run

- [ ] **Step 1: Merge**

```bash
curl -s -X POST http://192.168.44.36:4000/api/finetune/$JOB_500/merge
tail -F /mnt/tank/outputs/$JOB_500/merge.log
```

- [ ] **Step 2: Deploy**

```bash
curl -s -X POST http://192.168.44.36:4000/api/finetune/$JOB_500/deploy \
  -H 'Content-Type: application/json' \
  -d "{\"nodeId\": \"$NODE3\"}" | tee /tmp/phase-a-500step-deploy.json | jq
DEPLOY_500=$(jq -r '.id' /tmp/phase-a-500step-deploy.json)
```

- [ ] **Step 3: Eval at both token budgets**

```bash
# 512 tokens
python /mnt/tank/src/github/dgx-manager-fine-tune-recipes/scripts/evaluate.py \
  --base-endpoint <BASE_27B_ENDPOINT> \
  --base-served-name Qwen/Qwen3.6-27B \
  --tuned-endpoint $(curl -s http://192.168.44.36:4000/api/deployments/$DEPLOY_500 | jq -r '.endpoint') \
  --tuned-served-name Qwen/Qwen3.6-27B \
  --num-examples 100 --concurrency 4 \
  --max-tokens 512 --seed 42 \
  --results-dir /mnt/tank/results/qwen3.6-27b-500step/eval-512

# 2048 tokens
python /mnt/tank/src/github/dgx-manager-fine-tune-recipes/scripts/evaluate.py \
  --base-endpoint <BASE_27B_ENDPOINT> \
  --base-served-name Qwen/Qwen3.6-27B \
  --tuned-endpoint $(curl -s http://192.168.44.36:4000/api/deployments/$DEPLOY_500 | jq -r '.endpoint') \
  --tuned-served-name Qwen/Qwen3.6-27B \
  --num-examples 100 --concurrency 4 \
  --max-tokens 2048 --seed 42 \
  --results-dir /mnt/tank/results/qwen3.6-27b-500step/eval-2048
```

- [ ] **Step 4: Record the four numbers** in `/tmp/phase-a-results-500.json`.

- [ ] **Step 5: Clean up**

```bash
curl -s -X POST http://192.168.44.36:4000/api/deployments/$DEPLOY_500/stop
curl -s -X DELETE http://192.168.44.36:4000/api/deployments/$DEPLOY_500
```

---

## Group E — Documentation

### Task 20: Write `docs/qwen3.6-27b-fine-tuning-on-dgx-spark.md`

**Files:**
- Create: `/home/daniel/src/github/dgx-manager/docs/qwen3.6-27b-fine-tuning-on-dgx-spark.md`

Style: parallel to `docs/qwen3.6-fine-tuning-on-dgx-spark.md` (the 35B-A3B doc). Cite that doc as prerequisite reading.

- [ ] **Step 1: Outline**

Sections:
1. **TL;DR** — accuracy numbers from phase a, what worked, what didn't
2. **The model: Qwen 3.6-27B vs 35B-A3B** — table comparing arch, expert count (none vs 256), LoRA-friendliness
3. **The training setup** — recipe location, hardware (nodes 3+4), step time
4. **The merge — same script, different path** — explain why `merge_qwen3moe.py` Case 1 handles dense
5. **Results** — phase-a numbers in the same shape as the 35B-A3B doc's tables (50/500-step × 512/2048-tokens)
6. **Observations** — anything surprising
7. **Artifacts** — output dir tree, recipe location, eval results paths
8. **Related docs** — link to the 35B-A3B doc and the inference benchmark

- [ ] **Step 2: Write the doc using the actual numbers from `/tmp/phase-a-results-50.json` and `/tmp/phase-a-results-500.json`**

Use the same prose voice and table formatting as `docs/qwen3.6-fine-tuning-on-dgx-spark.md`. Don't speculate beyond what the numbers show.

- [ ] **Step 3: Commit**

```bash
cd /home/daniel/src/github/dgx-manager
git add docs/qwen3.6-27b-fine-tuning-on-dgx-spark.md
git commit -m "$(cat <<'EOF'
docs: Qwen 3.6-27B fine-tuning on DGX Spark — practitioner write-up

End-to-end LoRA on b-mc2/sql-create-context, 50/500-step runs, eval at
max_tokens={512,2048}. Recipe in kreuzhofer/dgx-manager-fine-tune-recipes
(qwen3.6-27b-base-lora). Same multimodal-wrapper-strip merge issue as
35B-A3B; reuses the merge_qwen3moe.py Case 1 path.
EOF
)"
```

---

## Group F — Wrap-up

### Task 21: Push the recipe to its remote (gated)

- [ ] **Step 1: Surface to user**

Ask: "Phase c+a green, doc committed. OK to `git push` the qwen3.6-27b-base-lora recipe to `origin/main` in kreuzhofer/dgx-manager-fine-tune-recipes?"

- [ ] **Step 2: On approval, push**

```bash
cd /mnt/tank/src/github/dgx-manager-fine-tune-recipes
git push origin main
```

---

## Self-review notes

1. **Spec coverage:**
   - Phase c smoke → Tasks 7-12
   - Phase a benchmark (50+500 step) → Tasks 14-19
   - Phase b explicitly NOT in this plan (gated, separate)
   - Recipe construction (recipe.yaml, train.py, scaffolding) → Tasks 3-6
   - Merge script reuse decision → Task 4 (in YAML comment) + Task 10 (validation)
   - GatedDeltaNet layer-naming verification → Task 1
   - OOM contingency (don't auto-escalate to nodes 1+2) → Task 8 step 3
   - vram_estimate_mb refinement → Task 13
   - All ops via API → enforced throughout
   - Doc → Task 20
2. **Placeholder scan:** No "TBD/TODO/implement later" left in any code-emitting step. Two operational TBDs are intentional — peak-RSS measurement (Task 13) and the four eval numbers (Tasks 17 & 19) — both are data-driven and have explicit measurement steps. The `<BASE_27B_ENDPOINT>` placeholder in Tasks 17/19 is resolved by Task 16.
3. **Type consistency:** `JOB_ID` (phase c), `JOB_50`, `JOB_500`, `DEPLOY_ID`, `DEPLOY_50`, `DEPLOY_500` named consistently. Endpoint variable `BASE_27B_ENDPOINT` discovered in Task 16, used in Tasks 17 & 19.
