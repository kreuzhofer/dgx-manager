# Fine-Tuning Qwen 3.6-27B on NVIDIA DGX Spark: A Practitioner's Guide

> Running a complete LoRA fine-tuning → merge → deploy pipeline for Alibaba's Qwen 3.6-27B (dense, hybrid GatedDeltaNet + Gated-Attention, multimodal) on DGX Spark, and what we found when comparing it against the 35B-A3B MoE sibling on the same SQL benchmark.

Prerequisite reading: [Fine-Tuning Qwen 3.6-35B-A3B on NVIDIA DGX Spark](./qwen3.6-fine-tuning-on-dgx-spark.md). This doc reuses the recipe scaffolding from that one and inherits the multimodal-wrapper merge fix; we won't re-derive that here.

Companion: [Qwen 3.6-27B FP8 Cost-per-Token Analysis](./qwen3.6-27b-cost-analysis.md) — inference numbers on the same hardware.

---

## TL;DR

Qwen 3.6-27B is a **dense, multimodal, hybrid GatedDeltaNet + attention** model: 64 layers, every 4th a full attention layer (`q/k/v/o_proj`), the other 48 stateful linear-attention (`linear_attn`/`in_proj`/`out_proj`/`A_log`). We fine-tuned it with **LoRA + DeepSpeed ZeRO-3**, single-node on DGX Spark (multi-node fell over on inter-node SSH — see Operational Findings). 500-step run took **~58 minutes** of training (effective batch size 4) on the `b-mc2/sql-create-context` benchmark.

Headline numbers (100 held-out test examples, seed=42, max_tokens=512):

- **27B base (bf16):** _TBD — base eval pending_
- **50-step LoRA:** **21%** (heavily preamble-throttled; only 17/100 outputs reached `</think>` within 512 tokens)
- **500-step LoRA:** _TBD — eval running_

At max_tokens=2048:

- **27B base (bf16):** _TBD_
- **500-step LoRA:** _TBD — eval running_

The methodology comparison with the 35B-A3B MoE sibling is the more interesting story than absolute accuracy — see [Comparison with the 35B-A3B MoE](#comparison-with-the-35b-a3b-moe) below.

---

## The Model

Qwen 3.6-27B is the **dense, smaller** sibling of the 35B-A3B MoE we fine-tuned earlier. Architectural shorthand:

| | 27B | 35B-A3B |
|---|---|---|
| Total params | 27B | 35B |
| Active per token | 27B (dense) | 3B (MoE 256/8) |
| LM layers | 64 | 64 |
| Full-attention layers (LoRA-targetable) | 16 (every 4th: `[3, 7, 11, 15, …, 63]`) | 16 (same pattern) |
| Linear-attention layers | 48 (`linear_attn` / GatedDeltaNet) | 48 (linear-attn / Mamba SSM) |
| Hidden size | 5,120 | 5,120 |
| MoE expert tensors | none | fused `experts.gate_up_proj` / `experts.down_proj` (256 experts) |
| Multimodal wrapper | yes (`Qwen3_5ForConditionalGeneration` + vision tower) | yes (same outer class family) |
| Multi-token prediction (MTP) head | yes (one transformer block under `mtp.layers.0.*`) | (not investigated separately) |
| vLLM arch class | `Qwen3_5ForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` |

**Implications for LoRA on the 27B vs the 35B-A3B:**

- **No `target_parameters` needed.** The 35B-A3B recipe uses PEFT's named-parameter walk to reach fused expert tensors. The 27B is dense — `target_modules=["q_proj","k_proj","v_proj","o_proj"]` is sufficient.
- **GatedDeltaNet layers are skipped automatically by suffix matching.** Static check confirmed `q_proj` only appears on the 16 full-attention layers in the language model, so PEFT's name-suffix matcher never attaches LoRA to the stateful linear-attention layers. No `layers_to_transform=` needed.
- **MTP head's q/k/v/o get matched anyway** — the MTP block's `self_attn.{q,k,v,o}_proj` would silently receive LoRA. The recipe's existing `frozen` loop (filter on `mtp.` substring in parameter name) catches and freezes those post-attach.
- **Vision tower's q/k/v/o get matched too** — by design, kept multimodal-capable for future visual fine-tunes. Wastes a bit of LoRA capacity for text-only tasks but matches the 35B-A3B precedent for like-for-like comparison.
- **Trainable param count:** 10,485,760 / 26,906,484,224 ≈ **0.0390%** at `lora_r=16, alpha=16`. Well within the dense-LoRA expected band; the existing capacity check (`pct < 0.001` raises) catches catastrophic mis-targeting without firing here.

---

## Training Setup

Recipe: `recipes/qwen3.6-27b-base-lora/` in [kreuzhofer/dgx-manager-fine-tune-recipes](https://github.com/kreuzhofer/dgx-manager-fine-tune-recipes), forked from the 35B-A3B recipe with the MoE-specific bits removed.

- **Base model:** `Qwen/Qwen3.6-27B` (bf16)
- **Framework:** DeepSpeed ZeRO-3 + PEFT LoRA (`peft >= 0.17`)
- **Config:** `max_seq_length=256, batch_size=1, grad_accum=4, lora_r=16, lora_alpha=16, dropout=0.0, lr=2e-4, warmup=5`
- **Hardware:** 1× DGX Spark (GB10, 122 GB unified memory) — see Operational Findings for why single-node despite the recipe targeting 2+
- **Dataset:** `b-mc2/sql-create-context` (~78k SQL gen examples, 95/5 train/eval split, seed=42)

**Step time:** ~5.5 s/step on single node, effective batch size 4. A 500-step run is ~46 minutes of pure training plus ~3 min eval and adapter save.

The recipe-level workarounds inherited from the 35B-A3B path (and which still apply unchanged):

- 4-hour NCCL timeout (`torch.distributed.constants.default_pg_*_timeout`) — bf16 ZeRO-3 of 27B does hundreds of broadcasts during `from_pretrained`.
- Drop `token_type_ids` / `mm_token_type_ids` columns from the dataset before training — the dataset library emits them for Gemma 4's multimodal collator and `DataCollatorForLanguageModeling` can't pad variable-length lists.
- `keep_in_memory=True` on `datasets.map()` — avoids NFS Arrow-cache truncation under multi-rank writes.
- `fix_gemma4_use_cache(model)` — generic `model.config.use_cache = True`; safe no-op on Qwen.
- Defensive `frozen` loop catching any parameter with `mtp.`, `router`, or `.gate.weight` in its name — catches the MTP head LoRA mentioned above.

---

## The Merge — Same Bug, Same Fix

Qwen 3.6 — both 27B dense and 35B-A3B MoE — wraps the LM in a multimodal `Qwen3_5ForConditionalGeneration` outer class. PEFT's `merge_and_unload()` strips this wrapper at save time, leaving a `model_type=qwen3_5_text` leaf config that vLLM cannot load.

The 35B-A3B doc derived a **byte-compatible merge that rewrites the base safetensors directly**, bypassing PEFT's runtime entirely (`scripts/merge_qwen3moe.py`). It has two cases:
1. **3D MoE expert tensors** (`base_tensor.ndim == 3`) — per-expert decomposition of the block-diagonal LoRA. Used by the 35B-A3B path.
2. **2D Linear weights** (`base_tensor.ndim == 2`) — standard `delta = (B @ A) * (alpha/r)` add. **This is what the 27B uses.**

The dense 27B recipe simply reuses `scripts/merge_qwen3moe.py`. The 3D path silently doesn't fire (no fused expert tensors exist in the 27B); the 2D path handles all 64 attention LoRA pairs (16 full-attn LM layers × 4 projections). The script name is misleading for a dense model, but renaming it would force-update the active 35B-A3B recipe — not worth the churn now.

Empirical verification on the 27B run: **64 / 64 LoRA pairs merged** across 5 of the 15 base shards (the others hold vision-tower / GatedDeltaNet / embeddings, all of which the LoRA didn't target):

```
shard 06: 27 pairs | shard 07: 5 pairs | shard 13: 26 pairs | shard 14: 5 pairs | shard 15: 1 pair
```

Output config preserves `architectures=["Qwen3_5ForConditionalGeneration"]` and `model_type=qwen3_5` (the multimodal-aware top-level), with the LM-specific config nested under `text_config`. vLLM dispatches it cleanly with no config gymnastics.

---

## Results

_(Filled in after evals complete.)_

### Like-for-like, max_tokens=2048

| Run | Steps | Wall (train) | Final train loss | Final eval loss | SQL accuracy | Δ vs base |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen 3.6-27B | — | — | — | — | _TBD_ | — |
| LoRA 50-step | 50 | ~6 min | _see metrics_ | _1.11_ | _TBD_ | _TBD_ |
| LoRA 500-step | 500 | ~58 min | _0.62_ | _0.81_ | _TBD_ | _TBD_ |

### Max-tokens sensitivity (500-step)

| max_tokens | Base | Tuned (500-step) | Δ |
|---:|---:|---:|---:|
|  512 | _TBD (base @ 512 not run — see Cost-vs-Info Findings)_ | _TBD_ | — |
| 2048 | _TBD_ | _TBD_ | _TBD_ |

### 50-step run — preamble-throttled at 512 tokens

| max_tokens | Tuned (50-step) | Reached `</think>` within budget |
|---:|---:|---:|
| 512 | **21%** (21/100) | 17/100 |
| 2048 | _not run_ (see Cost-vs-Info Findings) | — |

The verbose "Here's a thinking process:" preamble that the 27B base emits before SQL is **not yet suppressed at 50 steps**. 83 of 100 outputs ran out of tokens mid-think. Of the 17 that emerged from `</think>`, additional examples produced correct SQL via the eval's normalize-from-chain-of-thought fallback.

This is the same pattern the 35B-A3B doc described, but more pronounced on the 27B because:
- **Smaller effective batch size in our run** — single-node with `grad_accum=4` gives effective batch 4. The 35B-A3B's 50-step result of 57% at 512 tokens used 3 nodes (effective batch 12). Same step count, ~3× fewer effective examples seen.
- **No mid-run preamble compression** — the dataset has no explicit format-shortening signal; only the gradient through next-token prediction.

---

## Observations

_(Pending eval numbers — to fill.)_

---

## Operational Findings

### Multi-node SSH from node 3 → node 4 hung the launcher

We initially planned phase a on **2 nodes (head=node 3, worker=node 4)**. The training job's agent tries to SSH from the head agent to the worker over port 22 to launch the worker container (`packages/agent/src/runtime/finetune.ts:675`), with a 120s spawnSync timeout. The first SSH (`docker rm -f … ; docker run …`) on the worker hung at the OS level for the full 120s, returning `ETIMEDOUT`. Two retries reproduced.

Symptoms ruled out:
- **Worker reachable.** Ping works, the agent on the worker is online and was reporting `lastSeen` on every poll.
- **First-connection prompt.** Line 675 has `-o StrictHostKeyChecking=no`, so a missing known-hosts entry shouldn't prompt.

We did **not** diagnose the hang directly — the user's playbook forbids `ssh + docker` shortcuts on shared DGX nodes for production-host inspection ([memory: no direct container ops](../.claude/projects/-home-daniel-src-github-dgx-manager/memory/feedback_no_direct_container_ops.md)). The right next step is to verify SSH key trust (`~/.ssh/known_hosts`, `~/.ssh/authorized_keys`) for the agent user on each node pair, possibly through an explicit "agent connectivity test" API on the manager.

For phase a we course-corrected to **single-node on the idle node 4**. ZeRO-3 with `--nproc_per_node=1` is a no-op shard-wise, but a 27B + LoRA + bf16 fits on a single GB10 (122 GB unified memory) — peak observed at deployment time was ~100 GB including vLLM KV cache; training peak is lower.

### Recipe discovery only happens on agent (re)connect

The agent reads its training-recipes directory at WebSocket-handshake time, then sends one `agent:training-recipes` message to the manager. There is no `cmd:rescan-recipes` command. After committing a new recipe to the local recipes repo on shared NFS (visible to all agents), we had to nudge an agent to reconnect — done via `POST /api/nodes/:id/update-agent` on the idle node 4, which re-runs the install and forces a reconnect.

### Cost-vs-info: cut the eval matrix in half

At single-node TP=1 bf16, vLLM serves the 27B at ~17 tok/s aggregate decode (concurrency=4). 100 examples × 512 max_tokens = ~50 min per eval. The full 4-eval matrix (50/500-step × 512/2048-tok) plus a base baseline at both budgets adds up to ~12 hours of node 4 occupancy on top of training.

We dropped:
- **50-step @ 2048** — already know 50 steps is undertrained at our effective batch size, the @2048 number wouldn't change strategy.
- **Base @ 512** — base virtually always 0% at 512 tokens per the 35B-A3B precedent (verbose preamble dominates).

Net wall time savings: ~6 hours; matrix retains the headline 27B-vs-base comparison at @2048 and the LoRA's preamble-suppression progress at @512.

---

## Comparison with the 35B-A3B MoE

_(To be filled in once both runs' eval numbers are settled.)_

The interesting question isn't which model is more accurate at a given step count — they have different parameter counts and different effective batch sizes — but **whether the 27B's smaller-but-dense architecture compresses the verbosity preamble at the same rate as the 35B-A3B's MoE**. Each model has 16 full-attn LM layers × 4 projections under LoRA targeting, so the per-layer adapter capacity is comparable.

---

## Artifacts

```
/mnt/tank/outputs/cmos4vjb613la36s26zp1dvyv/                # phase-c smoke (5 steps)
/mnt/tank/outputs/cmos5lmzk14ch36s21dxy9f78/                # phase-a 50-step
/mnt/tank/outputs/cmosb1vbg1a1836s2j57p9ujm/                # phase-a 500-step
├── lora_adapter/
│   ├── adapter_model.safetensors                            # ~21 MB (10.5M params × bf16)
│   └── adapter_config.json
├── merged/                                                  # ~54 GB, 15 shards
│   ├── config.json                                          # multimodal wrapper preserved
│   ├── model-00001-of-00015.safetensors … (15 shards)
│   └── tokenizer / chat_template / preprocessor_config / video_preprocessor_config
├── train.log
└── merge.log

/mnt/tank/results/qwen3.6-27b-50step/eval-512/results.json   # 21% (preamble-throttled)
/mnt/tank/results/qwen3.6-27b-500step/eval-512/results.json  # TBD
/mnt/tank/results/qwen3.6-27b-500step/eval-2048/results.json # TBD
```

## Recipe location

- `recipes/qwen3.6-27b-base-lora/` (kreuzhofer/dgx-manager-fine-tune-recipes, branch `main`)
- `scripts/merge_qwen3moe.py` (reused as-is — Case 1 dense path handles q/k/v/o LoRA)

## Spec + plan

- Spec: [`docs/superpowers/specs/2026-05-04-qwen36-27b-training-design.md`](./superpowers/specs/2026-05-04-qwen36-27b-training-design.md)
- Plan: [`docs/superpowers/plans/2026-05-04-qwen36-27b-training.md`](./superpowers/plans/2026-05-04-qwen36-27b-training.md)

## Related docs

- [Fine-Tuning Qwen 3.6-35B-A3B on NVIDIA DGX Spark](./qwen3.6-fine-tuning-on-dgx-spark.md) — the MoE sibling, where the multimodal-wrapper-strip merge bug was first diagnosed and fixed.
- [Qwen 3.6-27B FP8 Cost-per-Token Analysis](./qwen3.6-27b-cost-analysis.md) — what the same merged model costs to serve at scale on DGX Spark.
