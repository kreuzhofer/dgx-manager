# Fine-Tuning Qwen 3.6-35B-A3B on NVIDIA DGX Spark: A Practitioner's Guide

> Running a complete LoRA fine-tuning → merge → deploy pipeline for Alibaba's Qwen 3.6-35B-A3B (hybrid Mamba + attention MoE) on DGX Spark, and the tooling bugs we had to patch along the way.

Prerequisite reading: [Fine-Tuning Gemma 4 on NVIDIA DGX Spark](./gemma4-fine-tuning-on-dgx-spark.md). This doc assumes you've seen the Gemma 4 26B-A4B MoE LoRA fix (PEFT `target_parameters`) — Qwen 3.6 reuses the same pattern for its fused expert tensors.

Companion benchmark: [Qwen 3.6-35B-A3B Inference Benchmark on DGX Spark](./qwen3.6-inference-benchmark.md).

---

## TL;DR

Qwen 3.6-35B-A3B is a hybrid Mamba + attention Mixture-of-Experts model (256 experts, 8 active per token). We fine-tuned it with LoRA + DeepSpeed ZeRO-3 across 3 DGX Spark nodes on the `b-mc2/sql-create-context` benchmark. The fine-tune improves on the base, but the gain is heavily dependent on the inference token budget and overfits past ~50 steps.

On 100 held-out test examples, seed=42, same endpoints for both sides:

- **Base Qwen 3.6-35B-A3B, max_tokens=2048: 42%**
- **500-step LoRA, max_tokens=2048: 56%** (+14 pp vs base)
- **500-step LoRA, max_tokens=512: 49%** (the base scored **0%** at this budget — more below)
- Training loss dropped cleanly from 3.74 → 0.60 at step 500 and eval loss from ~1.9 → 0.75. **The loss curve is not the benchmark** — the model overfits on the narrow SQL distribution well before loss bottoms out.

Separately from the benchmark improvement, LoRA collapses response length by stripping the "thinking process" preamble the base model emits before SQL. Same "eval-loss down, task-accuracy up-then-down" pattern we saw on Gemma 4 26B-A4B (a 50-step run of this same recipe hit 57% at max_tokens=512). The sweet spot for this recipe on this dataset is tens of steps, not hundreds.

Two things that would have killed the run silently without the workarounds in `scripts/merge_qwen3moe.py`:

1. **PEFT `merge_and_unload()` strips Qwen 3.6's multimodal wrapper**, saving the LM with `model_type=qwen3_5_moe_text` — a leaf-config transformers doesn't know. vLLM can't load the result.
2. **vLLM's runtime `--enable-lora` route silently rejects half the trained adapter** (the `experts.base_layer.lora_*` slot that holds `gate_up_proj` deltas) — and returns byte-identical outputs to the base model with no warning.

Our fix: hand-rolled per-expert LoRA composition against the base's fused tensors, byte-compatible with the base's serving path. Adapter → merged model → vLLM, no config gymnastics.

---

## The Model: Qwen 3.6 vs Gemma 4 26B-A4B

Architecturally Qwen 3.6-35B-A3B sits between Gemma 4 26B-A4B and a pure dense model.

| | Gemma 4 26B-A4B | Qwen 3.6-35B-A3B |
|---|---|---|
| Total params | 26B | 35B |
| Active per token | 4B | 3B |
| Attention | Standard | Hybrid: every 4th layer full-attn; rest are linear-attention (Mamba SSM) |
| Expert count | 64 | 256 |
| Active experts / token | 4 | 8 |
| Expert tensors | Fused 3D (`experts.gate_up_proj`, `experts.down_proj`) | Fused 3D (same naming) |
| vLLM arch class | `Gemma4ForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` |
| Inner LM class | `Gemma4ForCausalLM` | `Qwen3_5MoeForCausalLM` |

**Implications for LoRA:**

- **Mamba layers are not LoRA-friendly.** Stateful SSM updates don't compose with LoRA's low-rank additive trick. Target attention projections on the full-attn layers only — do not add LoRA to `in_proj` / `out_proj` on the linear-attn layers. In the recipe: `target_modules=["q_proj","k_proj","v_proj","o_proj"]`.
- **Expert targeting is exactly the same as Gemma 4 26B-A4B.** `target_parameters=["experts.gate_up_proj","experts.down_proj"]` in `LoraConfig` reaches the fused 3D tensors through PEFT 0.16+'s named-parameter walk. With `lora_r=16`, this gives ~1% trainable on 35B — the normal LoRA range.
- **Freeze the MTP head and the router gates.** The multi-token-prediction head and `mlp.gate.weight` destabilize routing and decoding if LoRA leaks into them. See the `frozen` loop in `recipes/qwen3.6-35b-a3b-base-lora/train.py`.

The `Qwen3_5MoeFor*` class names are real — vLLM resolves Qwen 3.5 and 3.6 to the same model class.

---

## The Training Setup

Recipe: `recipes/qwen3.6-35b-a3b-base-lora/` in [kreuzhofer/dgx-manager-fine-tune-recipes](https://github.com/kreuzhofer/dgx-manager-fine-tune-recipes).

- Base model: `Qwen/Qwen3.6-35B-A3B` (BF16 HF weights)
- Framework: DeepSpeed ZeRO-3 + PEFT LoRA, 3-node cluster
- Config: `max_seq_length=256, batch_size=1, grad_accum=4, lora_r=16, lora_alpha=16, dropout=0.0, lr=2e-4`
- Hardware: 3× DGX Spark (GB10, 128 GB unified memory each)
- Dataset: `b-mc2/sql-create-context` (~78k SQL gen examples, 95/5 train/eval split)

**Step time: ~170 s/step** (effective batch size 12 across the cluster). A 500-step run takes ~24 hours. The 24-h wall time is **all** DeepSpeed communication + gradient accumulation — the GB10s are NOT saturated on compute.

### Increase the NCCL timeout before process-group init

ZeRO-3 does hundreds of broadcasts during `from_pretrained` for 35B. The default 30-minute NCCL collective timeout is not enough — we bump to 4 hours:

```python
import datetime
torch.distributed.constants.default_pg_timeout = datetime.timedelta(hours=4)
torch.distributed.constants.default_pg_nccl_timeout = datetime.timedelta(hours=4)
```

This has to happen **before** any `torch.distributed.init_process_group()` call. The HF trainer will init the group the first time it touches a device — so these two lines go at the top of `train.py`, before `apply_all()` or any other model import that might pull distributed bootstrap in.

### Drop Gemma-only columns from the dataset

`lib/dataset.format_example` unconditionally emits `token_type_ids` and `mm_token_type_ids` (needed by Gemma 4's multimodal collator). Qwen 3.6 doesn't use either, and `DataCollatorForLanguageModeling` can't pad variable-length lists across a batch — eval crashes on the first multi-example batch with `expected sequence of length N at dim 1 (got M)`.

The Qwen recipe drops these columns after `prepare_datasets`:

```python
qwen_drop_cols = [c for c in ("token_type_ids", "mm_token_type_ids") if c in train_ds.column_names]
if qwen_drop_cols:
    train_ds = train_ds.remove_columns(qwen_drop_cols)
    if eval_ds is not None:
        eval_ds = eval_ds.remove_columns(qwen_drop_cols)
```

### HuggingFace Arrow cache: `keep_in_memory=True`

Multi-rank training with HF `datasets.map()` on NFS caches the tokenized output to Arrow files. When three ranks write simultaneously to the same NFS path, the cache shard files get truncated mid-write and pyarrow crashes with SIGBUS the next time it mmaps them.

Fix in `lib/dataset.py`: pass `keep_in_memory=True` to `.map()`. Skips the Arrow cache entirely; tokenization happens once per rank in RAM, no filesystem race.

---

## The Merge Problem (and Why Three Approaches Failed)

This is the part where most of the debugging time went.

### Approach A — vLLM runtime LoRA: **silently does nothing**

vLLM 0.18+ supports `--enable-lora --lora-modules adapter=path`. It loads the adapter, the endpoint returns responses, everything *looks* fine.

Responses are byte-identical to the base model.

vLLM's MoE LoRA loader accepts adapter keys named `experts.lora_A` / `experts.lora_B` (for `down_proj`) and silently rejects keys named `experts.base_layer.lora_A` / `experts.base_layer.lora_B` (the slot PEFT uses for `gate_up_proj`). So half of your trained adapter — specifically the up-projection half, where most of the LoRA signal lives — is never applied at inference. No warning in the logs. No warning in the response.

We verified this by comparing completions between `base` and `base + adapter` endpoints on ~50 SQL examples; outputs were identical token-for-token.

**Do not use vLLM runtime LoRA for Qwen 3.6 adapters until this is fixed upstream.**

### Approach B — PEFT `merge_and_unload()`: **breaks the multimodal wrapper**

The standard PEFT merge path is:

```python
model = AutoModelForCausalLM.from_pretrained(base, ...)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out_dir)
```

For Qwen 3.6 this produces a directory that looks right but fails to load:

- `config.json` has `"model_type": "qwen3_5_moe_text"` — a leaf config that transformers/vLLM doesn't register. You get `KeyError: 'qwen3_5_moe_text'` on load.
- If you patch `model_type` back to `qwen3_5_moe`, vLLM's weight loader throws an AssertionError because `lm_head.weight` appears twice across shards (the generic merge.py adds `fix_clippable_linear_keys`, which copies "all missing keys from base" — fine for Gemma, but duplicates `lm_head` for Qwen where the base's CausalLM wrapper already contains it).

Root cause: `merge_and_unload()` operates on the inner `Qwen3_5MoeForCausalLM`. `save_pretrained()` writes that inner class's config. The outer `Qwen3_5MoeForConditionalGeneration` wrapper — which vLLM needs to dispatch the arch — is gone.

You can work around this by loading with the multimodal class (`AutoModelForImageTextToText`) during training so `merge_and_unload()` operates on the wrapper. We tried that; it trains fine and gives equivalent results (57% at 50 steps vs. our other path's 56%), but PEFT's internal naming of the LoRA parameters still produces an adapter whose keys don't match the vLLM-serving-layer's expectations, so you still need Approach C at the serving step. The extra wrapper class adds nothing.

### Approach C — hand-rolled per-expert merge: **the one that works**

`scripts/merge_qwen3moe.py` does three things:

1. Read the PEFT adapter directly from `adapter_model.safetensors`. No PEFT runtime needed.
2. For each LoRA pair in the adapter, look up the matching parameter in the BASE model's safetensors (under the `model.language_model.layers.N.*` multimodal prefix), compute the delta in the right shape, and add it in.
3. Write the modified shards back out with the **same filenames and tensor keys as the base**. Same `config.json`, same tokenizer.

Output is byte-compatible with the base's serving path. vLLM dispatches it identically to an unmerged checkpoint of the same model — no `model_type` fiddling, no duplicated `lm_head`, no config gymnastics.

**PEFT adapter layout for `target_parameters` on MoE experts** (empirically observed from PEFT 0.19 output):

```
experts.base_layer.lora_A.weight   shape [E*r, hidden]       →   gate_up_proj
experts.base_layer.lora_B.weight   shape [2*moe_inter, E*r]
experts.lora_A.weight              shape [E*r, moe_inter]    →   down_proj
experts.lora_B.weight              shape [hidden, E*r]
```

Note that PEFT names the two halves inconsistently (one with `.base_layer.` infix, one without). The hand-roll script disambiguates based on which slot is present. Per-expert decomposition:

```
A_e = A[e*r:(e+1)*r, :]
B_e = B[:, e*r:(e+1)*r]
delta_e = (B_e @ A_e) * (alpha / r)
```

For `gate_up_proj` the delta has shape `[2*moe_inter, hidden]` matching the base per-expert tensor. For `down_proj` the delta has shape `[hidden, moe_inter]` matching the base. Stack into `[E, out, in]` and add to the base 3D fused tensor.

### Wiring the hand-roll into the agent merge API

The DGX Manager agent's merge command used to hard-code `scripts/merge.py` for every recipe. That's fine for Gemma (generic PEFT merge works) but wrong for Qwen 3.6. We added an optional `scripts.merge` field to the recipe schema:

```yaml
scripts:
  entrypoint: entrypoint.sh
  train: train.py
  launch: launch.sh
  ds_config: ds_config.json
  merge: scripts/merge_qwen3moe.py   # Qwen 3.6 override
```

The server reads this when handling `POST /api/finetune/:id/merge` and passes the repo-relative path to the agent; the agent runs the named script inside a merge container. If the recipe doesn't set it, the agent falls back to `scripts/merge.py` (unchanged behaviour for Gemma recipes).

We also made the merge container write stdout+stderr to `$outputDir/merge.log` alongside `train.log`, so merge failures are diagnosable after the fact instead of vanishing into a WebSocket stream.

### HF cache snapshot picking

`scripts/merge_qwen3moe.py` resolves the base model via `huggingface_hub.snapshot_download(..., local_files_only=True)`. That returns whatever revision is currently HEAD in the local cache — which can be an incomplete "metadata refresh" snapshot containing only `config.json`, not the weights. The script now falls back to scanning the cache for a snapshot that actually contains `.safetensors` if the default snapshot is empty.

---

## Results

Dataset: `b-mc2/sql-create-context`, 95/5 train/eval split, seed=42. Benchmark: 100 held-out test examples, exact-match accuracy (with markdown-fence stripping), concurrency=4. Base and tuned served via vLLM solo (TP=1) on separate DGX Spark nodes.

### Like-for-like (max_tokens=2048, same 100 examples, both models live)

| Run | Steps | Wall | Final train loss | Final eval loss | SQL accuracy | Δ vs base |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen 3.6-35B-A3B | — | — | — | — | **42%** | — |
| LoRA 500-step | 500 | ~24 h | 0.60 | 0.75 | **56%** | **+14 pp** |

### Max-tokens sensitivity (500-step tuned vs base)

| max_tokens | Base | Tuned | Δ |
|---:|---:|---:|---:|
|  512 | **0%** | **49%** | +49 pp |
| 2048 | **42%** | **56%** | +14 pp |

**The base model's verbosity dominates low-token evals.** At `max_tokens=512` the base emits a multi-paragraph "thinking process" preamble and runs out of tokens before the SQL. Its accuracy drops from 42% to 0% just by tightening the budget. The fine-tuned model drops only 7 pp (56 → 49) over the same budget change — LoRA has taught it to compress the response, but it's not fully purged. Downstream inference cost and latency are directly affected.

### Observations

- **50 steps probably wins on accuracy at low token budgets.** A prior 50-step run of this recipe scored 57% at `max_tokens=512` — essentially matching the 500-step's 56% at 2048 and beating its 49% at 512. More steps → shorter response preamble stripping is already saturated, and the model starts overfitting the SQL surface patterns of the training data.
- **Eval loss keeps going down the whole time.** From ~1.9 (warmup) → 0.75 at step 500. The loss curve and the task benchmark tell opposite stories. Trust the benchmark.
- **Training was healthy by every internal metric.** No loss spikes, no NaNs, no gradient blow-ups, `mean_token_accuracy` steadily rising, all 19 expected eval points landed. It's just that the thing the optimizer is minimizing is not the thing we actually care about.
- **The +14 pp at 500 steps is a real improvement** — repeatable across two independent evals (49% vs 0% at 512 tokens; 56% vs 42% at 2048). Measurement methodology matters: `max_tokens` is as important as the training config.

Recommendation: for production SQL agents, stop training at ~50 steps AND set a generous `max_tokens` budget (or enforce `/no_think`-style response compression at the prompt level). For a methodology benchmark (is Qwen 3.6 LoRA-trainable end-to-end?), the answer is yes — both runs cleanly beat the base and round-trip through merge → deploy → eval on the API.

---

## The API-Only Workflow

Every step in this guide runs through the DGX Manager HTTP API. No `ssh + docker` shortcuts. The merge and deploy flows needed fixes before that was true:

- **Agent reattach after restart.** The agent restarts mid-training (e.g. `cmd:update` picked up a new version) used to leave the job stuck at `status=starting` forever — `reattachFinetuneJobs` re-found the container but never re-attached the completion detection. Fixed in 0.5.281: tail the training log after reattach and emit `agent:finetune:complete` on `LoRA adapter saved`. Verified on this 24-h run.
- **Recipe-driven merge script selection.** Added `scripts.merge` to the recipe schema; server passes the path in `cmd:finetune:merge`; agent falls back to `scripts/merge.py` if unset. 0.5.287.
- **Merge log persistence.** Agent now tees merge stdout+stderr to `$outputDir/merge.log` before streaming to the server, so failures after the container exits are still debuggable. 0.5.291.

All three fixes were driven by real failures on this run. Each one eliminates a case where the only way to diagnose was to go around the API with direct docker — which is explicitly against the playbook.

---

## Artifacts

```
/mnt/tank/outputs/cmobis8d3003v36ra3e7anka7/         # 500-step run
├── lora_adapter/
│   ├── adapter_model.safetensors                     # ~1.85 GB
│   └── adapter_config.json
├── merged/                                            # 67 GB
│   ├── config.json                                    # same as base
│   ├── model-00001-of-00026.safetensors … (26 shards)
│   └── tokenizer / chat_template / generation_config
├── train.log
├── merge.log
├── worker-192_168_44_37.log
├── worker-192_168_44_38.log
└── eval/
    ├── results-500step.json                           # tuned-only, max_tokens=512
    ├── results-500step-vs-base.json                   # like-for-like, max_tokens=512
    └── results-500step-vs-base-2048tok.json           # like-for-like, max_tokens=2048
```

## Recipe location

- `recipes/qwen3.6-35b-a3b-base-lora/` (kreuzhofer/dgx-manager-fine-tune-recipes, `main`)
- `scripts/merge_qwen3moe.py` (same repo)

## Related docs

- [Fine-Tuning Gemma 4 on NVIDIA DGX Spark](./gemma4-fine-tuning-on-dgx-spark.md) — same tooling, different model family; the `target_parameters` fix for fused MoE experts originated there.
- [Qwen 3.6-35B-A3B Inference Benchmark on DGX Spark](./qwen3.6-inference-benchmark.md) — BF16 vs FP8, cluster vs solo throughput.
