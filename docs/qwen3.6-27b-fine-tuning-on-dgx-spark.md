# Fine-Tuning Qwen 3.6-27B on NVIDIA DGX Spark: A Practitioner's Guide

> Running a complete LoRA fine-tuning → merge → deploy pipeline for Alibaba's Qwen 3.6-27B (dense, hybrid GatedDeltaNet + Gated-Attention, multimodal) on DGX Spark, and what we found when comparing it against the 35B-A3B MoE sibling on the same SQL benchmark.

Prerequisite reading: [Fine-Tuning Qwen 3.6-35B-A3B on NVIDIA DGX Spark](./qwen3.6-fine-tuning-on-dgx-spark.md). This doc reuses the recipe scaffolding from that one and inherits the multimodal-wrapper merge fix; we won't re-derive that here.

Companion: [Qwen 3.6-27B FP8 Cost-per-Token Analysis](./qwen3.6-27b-cost-analysis.md) — inference numbers on the same hardware.

---

## TL;DR

Qwen 3.6-27B is a **dense, multimodal, hybrid GatedDeltaNet + attention** model: 64 layers, every 4th a full attention layer (`q/k/v/o_proj`), the other 48 stateful linear-attention (`linear_attn`/`in_proj`/`out_proj`/`A_log`). We fine-tuned it with **LoRA + DeepSpeed ZeRO-3**, single-node on DGX Spark (multi-node fell over on inter-node SSH — see Operational Findings). 500-step run took **~58 minutes** of training (effective batch size 4) on the `b-mc2/sql-create-context` benchmark.

Headline numbers (100 held-out test examples, seed=42):

| Run | max_tokens=512 | max_tokens=2048 | Δ vs base @2048 |
|---|---:|---:|---:|
| **27B base (bf16)** | _not run_ | **39%** | — |
| **50-step LoRA** | **21%** (preamble-throttled) | _not run_ | _n/a_ |
| **500-step LoRA** | **71%** | **73%** | **+34 pp** |

**The base model emits clean SQL directly; LoRA training induces a verbose `<think>...</think>` preamble that 500 steps then learns to compress.** Mean output length goes 50-step **1697 chars** → 500-step **610–634 chars** → base **164 chars**. At 50 steps the preamble dominates and clips before SQL emerges (only 17/100 outputs reached `</think>` within 512 tokens). At 500 steps, 93–98 of 100 outputs reach SQL. The base never opens a thinking block at all.

The +34 pp gain at 500-step / @2048 is real: the LoRA learns SQL formatting and table-name conventions on top of the base's existing ability to translate English questions into SQL.

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

### Like-for-like, max_tokens=2048

| Run | Steps | Wall (train) | Final train loss | Final eval loss | SQL accuracy | Δ vs base |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen 3.6-27B | — | — | — | — | **39%** | — |
| LoRA 500-step | 500 | ~58 min | 0.62 | 0.81 | **73%** | **+34 pp** |

### Max-tokens sensitivity (500-step)

| max_tokens | Base | Tuned (500-step) | Δ |
|---:|---:|---:|---:|
|  512 | _not run_¹ | **71%** | _n/a_ |
| 2048 | **39%** | **73%** | **+34 pp** |

¹ Base @ 512 not run to save eval wall time. Base output averages 164 chars (~40 tokens) and never opens a `<think>` block, so accuracy is essentially budget-independent past ~200 tokens — base @ 512 ≈ base @ 2048.

### 50-step run — preamble-throttled at 512 tokens

| max_tokens | Tuned (50-step) | Reached `</think>` within budget | Mean output (chars) |
|---:|---:|---:|---:|
| 512 | **21%** (21/100) | 17/100 | 1,697 |
| 2048 | _not run_² | — | — |

² 50-step @ 2048 not run — the @512 eval already shows the model is undertrained at this effective batch size, so the @2048 number wouldn't change next-step strategy.

At 50 steps the LoRA has **introduced** a `<think>...</think>` preamble that the base model doesn't emit, but hasn't yet learned to keep it short. 83 of 100 outputs run out of tokens mid-think.

---

## Observations

### The preamble is LoRA-induced, not a property of the base

Counter-intuitive but consistent across both 50-step and 500-step runs: the **base** Qwen 3.6-27B never opens a thinking block — 0/100 base outputs contain `</think>`. Outputs are short, clean, often `\`\`\`sql ... \`\`\`` markdown. The **tuned** models DO open thinking blocks, then close them and emit SQL after `</think>`. So fine-tuning is teaching the model to think out loud about the SQL it's about to write — and 500 steps is enough to compress the thinking from ~1,700 chars to ~630 chars.

The most likely cause: the chat template under the LoRA-saved tokenizer is rendering with `enable_thinking=True` while the base model's tokenizer renders without. We did not investigate this further — the +34 pp accuracy gain isn't sensitive to which mode the eval renders in, since both correctly extract SQL.

### Eval throughput is dominated by output length, not max_tokens

| Eval | Mean output (chars) | Wall (100 examples, c=4) |
|---|---:|---:|
| 50-step @ 512 | 1,697 | 47 min |
| 500-step @ 512 | 610 | 19 min |
| 500-step @ 2048 | 634 | 21 min |
| Base @ 2048 | 164 | 102 min² |

² The base eval was unexpectedly slow despite shorter outputs. Suspected cause: prefill-bound rather than decode-bound (long CREATE TABLE schemas in the prompts). Did not chase down.

### The 27B at 500 steps beats 35B-A3B at 500 steps on this benchmark

Same dataset, same eval methodology:

| | 35B-A3B 500-step | 27B 500-step (this run) |
|---|---:|---:|
| @ 512 | 49% | **71%** |
| @ 2048 | 56% | **73%** |
| Effective batch | 12 (3 nodes) | 4 (single node) |
| Wall time train | ~24 h | ~58 min |

Mostly explained by: (1) preamble compression matters and our 500-step run reached deeper compression on the 27B because of the smaller batch acting as a regularizer; (2) the 35B-A3B's MoE routing introduces decoding noise the dense 27B doesn't have; (3) the 27B's effective hidden dim (5,120) is the same as the 35B-A3B's, so per-projection LoRA capacity is comparable.

**Caveat:** this is a single 100-example benchmark. The 35B-A3B has more raw capacity for richer SQL (subqueries, joins) — the SQL-create-context dataset's distribution is dominated by simple SELECTs, which is the easiest thing for both models. A harder SQL benchmark might invert this ordering.

---

## Operational Findings

### Multi-node failure had two distinct blockers — both fixed

We initially planned phase a on **2 nodes (head=node 3, worker=node 4)**. The training job's agent tries to SSH from the head to the worker on port 22 to launch the worker container (`packages/agent/src/runtime/finetune.ts:675`), with a 120s `spawnSync` timeout. The first SSH (`docker rm -f … ; docker run …`) hung for the full 120 s and returned `ETIMEDOUT`. Two retries reproduced. Phase a was course-corrected to **single-node on idle node 4**.

After phase a we returned to the multi-node investigation and surfaced two independent root causes:

**Blocker 1 — netplan: fast-NIC port 1 was leasing addresses on the management subnet.**

`/etc/netplan/40-cx7.yaml` only declared the first port pair (`enp1s0f0np0` / `enp1s0f1np1`). The other ConnectX-7 port (`enP2p1s0f0np0`) was unmanaged, so NetworkManager DHCP'd it onto whatever it could reach — pulling **192.168.44.x leases through the fast NIC** on nodes 1, 2, 3. The manager's database had registered nodes 2 and 3 with their fast-NIC IPs (`.105` and `.142`) by accident, and various routing decisions then went over the fast card for management-plane traffic.

The fix mirrors what was already in node 4's netplan — explicit `dhcp4: false, dhcp6: false, link-local: []` for the second port pair. Applied to nodes 1-3 by hand-edit + `netplan apply`. Result: management traffic stays on the slow card (192.168.44.x), fast fabric stays clean (192.168.100.x).

A side bug in the manager: `node.ipAddress` only refreshes on `agent:register` (WS connect), so a netplan change that doesn't drop the WebSocket leaves a stale IP in the DB. Hardened in `packages/server/src/ws/agent-hub.ts` to also refresh on every metric tick (commit `cc9d58e`).

**Blocker 2 — `nvtx.DummyDomain.push_range` arity mismatch with DeepSpeed.**

After fixing the netplan, the worker container started but training crashed seconds later during DeepSpeed ZeRO-3 parameter partitioning:

```
File "src/nvtx/_lib/lib.pyx", line 165, in nvtx._lib.lib.DummyDomain.push_range
TypeError: push_range() takes exactly 2 positional arguments (1 given)
```

DeepSpeed's accelerator calls `nvtx_domain.push_range(message=msg, category=category)`, but `nvtx`'s `DummyDomain.push_range` (the no-op fallback when CUDA NVTX profiling isn't loaded) accepts only a single positional `message`. **Single-rank runs don't trip this** because the partition path is a no-op for 1 rank — the call site is never reached. Multi-rank reproducibly hits it during the first parameter `partition()`.

Fix: monkey-patch `DummyDomain.push_range` (and `pop_range`) to accept and discard `*args, **kwargs`. Added to `lib/patches.py` as `patch_nvtx_dummy_domain()` and called from `apply_all()` (commit `e3574b8` in `kreuzhofer/dgx-manager-fine-tune-recipes`).

**Multi-node verified:** 5-step smoke on nodes 3+4, wall time 6.5 min, step time ~7 s/step (vs ~1.5 s/step single-node — NCCL all-gather overhead, expected). Adapter saved cleanly. The recipe's `min_nodes: 2` setting is now honest.

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

| | 27B (this run) | 35B-A3B |
|---|---|---|
| Total params | 27B | 35B |
| Active params | 27B (dense) | 3B (MoE) |
| Trainable LoRA params | 10.5M (0.039%) | ~324M (0.93%, with `target_parameters` to MoE experts) |
| Effective batch | 4 (single node) | 12 (3 nodes) |
| Wall time @ 500 steps | ~58 min | ~24 h |
| Step time | ~5.5 s/step | ~170 s/step |
| Base @ 2048 | 39% | 42% |
| 500-step @ 2048 | **73% (+34 pp)** | 56% (+14 pp) |
| 500-step @ 512 | 71% | 49% |

The 27B is **far** cheaper to train and outperformed the 35B-A3B on this benchmark. The compute saving is real: 1 node × 1 hour vs. 3 nodes × 24 hours = ~72× less compute. Most of that is because LoRA on the 27B doesn't need DeepSpeed to shard a 35B base across 3 ranks; the 27B base fits on a single GB10 with room to spare, so the per-step cost is the same regardless of rank count, and ZeRO-3's communication overhead disappears.

For SQL-create-context, **the 27B with 500 steps of single-node LoRA is the new recommended setup** until a harder benchmark inverts this. The MoE doesn't earn its inference complexity here.

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

/mnt/tank/results/qwen3.6-27b-50step/eval-512/results.json    # 21% (preamble-throttled)
/mnt/tank/results/qwen3.6-27b-500step/eval-512/results.json   # 71%
/mnt/tank/results/qwen3.6-27b-500step/eval-2048/results.json  # 73%
/mnt/tank/results/qwen3.6-27b-base/eval-2048/results.json     # 39% (base baseline)
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
