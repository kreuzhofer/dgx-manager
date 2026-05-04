# Qwen 3.6-27B LoRA training script — design

**Date:** 2026-05-04
**Owner:** Daniel
**Status:** Approved (brainstorming complete) — awaiting user spec review before plan writing.

## Goal

Add a working LoRA fine-tune recipe for **Qwen 3.6-27B** (dense, hybrid GatedDeltaNet + Gated-Attention, multimodal) to `kreuzhofer/dgx-manager-fine-tune-recipes`, validated end-to-end through the DGX Manager API: train → merge → deploy → eval.

The 27B is architecturally different from the 35B-A3B we already trained — **dense, no fused experts** — so most of the MoE-specific scaffolding from the 35B-A3B recipe drops out. The hybrid attention rule (LoRA on full-attention layers only, skip linear-attn/GatedDeltaNet) carries over.

## Phasing

### Phase c — methodology smoke (autonomous)

A 5-step run on a tiny slice of `b-mc2/sql-create-context`. The goal is to push bytes through every step of the API: **container builds → train completes → merge produces a vLLM-loadable artifact → deployment serves a SQL completion**. No accuracy claim.

**Success:** deployed endpoint returns HTTP 200 with a non-empty completion on a SQL-shaped prompt.

### Phase a — SQL benchmark (autonomous, mirrors 35B-A3B)

Two production runs on full `b-mc2/sql-create-context`: **50 steps and 500 steps**. Eval methodology mirrors the 35B-A3B comparison (`scripts/evaluate.py`, 100 held-out examples seed=42, both `max_tokens=512` and `max_tokens=2048`).

**Success:** an accuracy delta vs base 27B at both token budgets, directly comparable to the 35B-A3B `+14 pp` result.

### Phase b — domain task (NOT autonomous, gated)

Deferred. After phase a green, regroup with the user to decide:

- Dataset (text vs multimodal/visual-judge)
- Prompt format
- Recipe variant (`-base-lora` vs `-instruct-lora` vs `-vision-lora`)
- Eval methodology

No phase-b work happens without explicit user direction.

## Recipe layout

New recipe: `recipes/qwen3.6-27b-base-lora/` in `kreuzhofer/dgx-manager-fine-tune-recipes`. Forked from `qwen3.6-35b-a3b-base-lora/`.

```
recipes/qwen3.6-27b-base-lora/
├── recipe.yaml         # changes from 35B-A3B below
├── train.py            # changes from 35B-A3B below
├── entrypoint.sh       # unchanged
├── launch.sh           # likely unchanged; verify in phase c
└── ds_config.json      # unchanged
```

### `recipe.yaml` deltas vs 35B-A3B

| Field | 35B-A3B | 27B |
|---|---|---|
| `name` | `Qwen3.6-35B-A3B-Base-LoRA` | `Qwen3.6-27B-Base-LoRA` |
| `description` | hybrid Mamba+attn MoE, 256 experts | dense, hybrid GatedDeltaNet+Gated-Attention, multimodal |
| `base_model` | `Qwen/Qwen3.6-35B-A3B` | `Qwen/Qwen3.6-27B` |
| `scripts.merge` | `scripts/merge_qwen3moe.py` | **kept** — same script. It handles dense 2D LoRA in `compute_delta()` Case 1 (`base_tensor.ndim == 2`); the MoE 3D branch silently doesn't fire for a dense model. The script name is misleading for a dense model — the 27B `recipe.yaml` will include a comment explaining the reuse. We do NOT use the generic `scripts/merge.py` because it relies on PEFT `merge_and_unload()` which strips Qwen 3.6's multimodal wrapper, producing a `model_type=qwen3_5_text` config that vLLM can't load. |
| `hardware.min_nodes` | `3` | `2` |
| `hardware.vram_estimate_mb` | `49152` | TBD — set after phase c measures peak rss; provisional `40960` |
| `defaults.lora_target_modules` | `q_proj,k_proj,v_proj,o_proj` | unchanged |

Defaults `learning_rate`, `batch_size`, `gradient_accumulation_steps`, `max_seq_length`, `lora_r`, `lora_alpha`, `lora_dropout`, `seed`, `eval_fraction` stay at the 35B-A3B values for like-for-like comparison in phase a.

### `train.py` deltas vs 35B-A3B

1. **Drop `target_parameters`.** 27B is dense — no `experts.gate_up_proj` / `experts.down_proj` to reach. Pass only `target_modules=["q_proj","k_proj","v_proj","o_proj"]` to `LoraConfig`.

2. **Replace the MoE capacity-check with a dense capacity-check.** The 35B-A3B has `if pct < 0.5: raise` because a working 35B run with `target_parameters` should land near 1%. On 27B dense, attention-only LoRA on `r=16` lands much lower (≪0.5%). The dense check should fail loud only on *suspiciously zero* trainable parameter counts (e.g. `< 0.001%`), which would indicate the target-module suffixes didn't match anything.

3. **Keep the `frozen` loop.** It looks for `.gate.weight`, `router`, `mtp.` substrings and freezes any matches. On 27B dense those substrings likely don't match anything (no MoE router, no MTP head) — the loop becomes a no-op, which is fine. Don't remove it; remaining defensive against future Qwen variants is cheap.

4. **Keep all the carryover fixes**, all already in the 35B-A3B `train.py`:
   - 4-hour NCCL timeout (`torch.distributed.constants.default_pg_*_timeout`)
   - `qwen_drop_cols` removal of `token_type_ids` / `mm_token_type_ids`
   - `keep_in_memory=True` (lives in `lib/dataset.py`, no recipe-level change needed)
   - `fix_gemma4_use_cache(model)` (the function name is Gemma-specific but the comment says "generic use_cache=True"; verify it's safe on Qwen 27B in phase c — likely a no-op there)
   - `gradient_checkpointing_enable(use_reentrant=False)`

5. **Linear-attn layer exclusion verification (phase c step 1).**

   The 35B-A3B recipe relies on `q_proj/k_proj/v_proj/o_proj` existing **only** on full-attention layers — PEFT's suffix matcher then automatically skips Mamba layers. We need to verify the same is true for Qwen 3.6-27B's GatedDeltaNet layers. Verification is a 5-minute static check before phase c training:

   ```python
   from transformers import AutoConfig
   from huggingface_hub import hf_hub_download
   import json
   p = hf_hub_download("Qwen/Qwen3.6-27B", "model.safetensors.index.json")
   keys = json.load(open(p))["weight_map"].keys()
   q_proj_layers = sorted({k.split(".layers.")[1].split(".")[0] for k in keys if ".q_proj." in k})
   total_layers = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B").num_hidden_layers
   print(f"q_proj on {len(q_proj_layers)} of {total_layers} layers")
   ```

   - If `q_proj` only appears on a subset (the full-attention layers), suffix matching is safe and we proceed unchanged.
   - If `q_proj` appears on **every** layer (i.e. GatedDeltaNet also has a `q_proj`), we add a `layers_to_transform=[…]` arg to `LoraConfig` enumerating only the full-attention layer indices — same idea as the 35B doc's "scope to `model.language_model.layers`".

   This check is documented as the **first phase-c task** so the design ships even with this unknown.

## Hardware allocation

- **Phase c + a both run on nodes 3 and 4** (DGX Spark, GB10).
- **Nodes 1 and 2 are reserved** for ongoing inference workloads — do NOT stop them.
- **OOM contingency:** if the 27B + ZeRO-3 + bf16 + activations don't fit on 2 × 122 GB, **pause and ask** the user before touching nodes 1+2. Don't auto-escalate to 3 nodes silently.

Memory back-of-envelope **for LoRA, not full FT**: the base 27B model is *frozen*; only LoRA adapters get gradients and optimizer state. So per rank under ZeRO-3:

- Frozen bf16 base weights, sharded: 54 GB / 2 ≈ 27 GB
- LoRA params (`r=16` on q/k/v/o ≈ a few tens of MB) + grads + AdamW state: <1 GB
- Activations with `gradient_checkpointing_enable(use_reentrant=False)` at `max_seq_length=256, bs=1, grad_accum=4`: low single-digit GB
- DeepSpeed buffers + CUDA workspace + tokenizer + python: ~5 GB

Expected per-rank peak: ~35 GB out of 122 GB unified. Comfortably fits on 2 × DGX Spark; matches the 35B-A3B doc's observation that GB10s are *not* compute-saturated and headroom is the rule. The full-FT 416 GB number from the 26B-A4B memory does NOT apply here because the base is frozen. Phase c will confirm with peak-rss observation.

## Operational discipline

These are existing rules in the project's CLAUDE.md / memory; restating because they bind every phase of this work:

1. **All ops via the DGX Manager API.** No `ssh + docker` shortcuts to start/stop containers, restart agents, or peek at training state. If something is broken, fix it in code through the API surface.
2. **Watch `train.log` directly** during runs (`$SHARED_STORAGE/outputs/{jobId}/train.log`), not the API status — status can go stale.
3. **Cleanup discipline:** if a phase-c deployment fails, DELETE the deployment record before retrying. Don't leave a trail.

## Testing strategy

This work spans two repos: code lives in `kreuzhofer/dgx-manager-fine-tune-recipes`; the manager-side train/merge/deploy plumbing in `dgx-manager` is unchanged.

| Risk tier | What gets tested | Where |
|---|---|---|
| Low — recipe YAML | Schema validity (parses, required fields present, `scripts.merge` resolves) | Server-side recipe-loading runs on every train start; phase c exercises it. |
| Low — train.py | Smoke: a 5-step run finishes without import errors and saves an adapter. | Phase c IS this test. No automated unit test (the dependencies — DeepSpeed, NCCL, multi-node — make a unit harness impractical). |
| Medium — `merge_qwen3moe.py` reuse for dense | The script's Case 1 path (`base_tensor.ndim == 2`) handles q/k/v/o LoRA correctly and produces a vLLM-loadable artifact. | Phase c covers this end-to-end. If a dense run surfaces a bug, fix it in code with a regression check rather than forking the script — adding "qwen3 dense" support to the existing script is preferable to a new file. |
| Low — eval | Existing `scripts/evaluate.py` works against any vLLM endpoint; no recipe-specific test. | Phase a IS this test. |

Per CLAUDE.md principle 1: "every change should leave `npm test` passing." This work is in the recipes repo (Python, not in the manager's vitest suite), but the manager-side `npm test` should still pass since we don't touch manager code.

## Out of scope

- Changes to manager (`packages/server`, `packages/agent`, `packages/dashboard`).
- Changes to `lib/` shared helpers in the recipes repo (the existing `qwen_drop_cols` handling at recipe-level keeps `lib/dataset.py` Gemma-shaped).
- Changes to `scripts/merge.py` (we don't use it for the 27B; it remains for Gemma recipes).
- Renaming `scripts/merge_qwen3moe.py` (the name is misleading once it serves a dense recipe, but renaming would force-update the active 35B-A3B recipe — not worth the churn now).
- Phase-b dataset choice or recipe variant.
- TP=4 unblocking (separate upstream vLLM issue tracked in `docs/vllm-issue-draft-tp4-hang.md`).
- FP8 training (we'll evaluate FP8 *inference* of the merged model, but train in bf16).

## Open risks

1. **OOM on 2 nodes.** Mitigation: explicit pause-and-ask before escalating to 3 nodes. Phase c will surface this within minutes of `from_pretrained`.
2. **GatedDeltaNet uses `q_proj` naming.** Mitigation: phase-c step-1 static check (above) before training starts. If true, add `layers_to_transform`.
3. **`merge_qwen3moe.py` Case 1 path is wrong for dense 27B.** The Case 1 branch is the standard 2D LoRA delta computation — well-tested in principle, but never exercised on a real Qwen 3.6 dense base model. Mitigation: phase c's deploy-and-smoke step catches a broken artifact; if needed, fix in `compute_delta()` rather than forking.
4. **`fix_gemma4_use_cache` doing the wrong thing on Qwen 27B.** Mitigation: read the function before phase c, ensure it's a safe no-op for non-Gemma archs.

## Artifacts produced by this design

- `recipes/qwen3.6-27b-base-lora/recipe.yaml`
- `recipes/qwen3.6-27b-base-lora/train.py`
- `recipes/qwen3.6-27b-base-lora/entrypoint.sh` (unchanged copy)
- `recipes/qwen3.6-27b-base-lora/launch.sh` (unchanged copy)
- `recipes/qwen3.6-27b-base-lora/ds_config.json` (unchanged copy)
- A documented phase-c run + phase-a 50/500-step runs with eval results in `docs/qwen3.6-27b-fine-tuning-on-dgx-spark.md` (parallel to the 35B doc).
- `scripts/merge_qwen3moe.py` is **reused, not modified** unless phase c surfaces a Case 1 bug.
