# Does any training framework know `glm5_next`?

> Research for [#67](https://github.com/kreuzhofer/dgx-manager/issues/67). Part of the
> [GLM-5.3-Flash wayfinder map](https://github.com/kreuzhofer/dgx-manager/issues/65).
>
> **Everything below was verified against primary sources on 2026-08-31, 19:00–22:30 UTC.**
> `zai-org/GLM-5.3-Flash` was published **2026-08-25**; `transformers` support merged
> **2026-08-26 14:26 UTC**; the one framework that supports it merged **2026-08-28**.
> This surface is **six days old**, and three of the facts below were established within the last
> 72 hours. Treat every "no" as a snapshot, not a law, and re-check before acting on it.
>
> Every claim is labelled **verified** (read out of the source, the API or the vendor's own
> docs myself), **derived** (arithmetic over verified numbers) or **untested** (follows from
> the code but nobody, including me, has run it).

---

## Recommendation

**The blocker did not materialise. Something can build a trainable
`Glm5NextForConditionalGeneration` — and it is not the framework anyone would have guessed.**

**Use `NVIDIA-NeMo/Automodel` at or after commit `9228f33c` (PR
[#3699](https://github.com/NVIDIA-NeMo/Automodel/pull/3699), merged 2026-08-28), starting from
`examples/vlm_finetune/glm5_next/glm5_3_flash_medpix_packed2k_ep72_cp2_100steps.yaml`, with a
`peft:` block copied from `examples/llm_finetune/glm/glm_5.2_lora.yaml`.**

That recipe already does, as shipped and validated, four of the five things the map decided it
wanted:

- **image + text training on `zai-org/GLM-5.3-Flash`** — a real VLM recipe against the real
  checkpoint, not a text-only port;
- **`freeze_vision_tower: true`** and `freeze_embeddings: true` — the map's exact freeze
  decision, as first-class config flags;
- **sequence packing that respects document boundaries** (`packing_format: neat`, 2K packs) —
  which, notably, upstream `transformers` *cannot* do on this model (§2.4c);
- **288-expert MoE handled properly** — expert parallelism with `dispatcher: hybridep`, four
  routed experts per GPU at EP72.

The fifth — **parameter-efficient** rather than full fine-tune — is the only piece not already
composed. But both halves exist and are validated separately: `FinetuneRecipeForVLM` reads a
`peft:` block (`nemo_automodel/recipes/vlm/finetune.py` L507-509, L537-538), and
`glm_5.2_lora.yaml` shows the LoRA target syntax for a GLM MoE — including `"*.mlp.experts"`,
backed by a real `GroupedExpertsLoRA` that injects into the grouped-GEMM expert forward rather
than materialising the expert stack. **Composing them is a YAML change, not a code change.**
It is nonetheless *untested*, and that untested composition is the single highest-value probe
this map can run next.

**Everything else says no.** LLaMA-Factory, ms-swift, Axolotl, torchtune and Unsloth-on-CUDA
cannot train this model today, and two of them fail *silently* rather than loudly. Bare
`transformers` + `peft` + `trl` technically can — every training hook is wired — but it is the
worse route for three concrete reasons, all in §4: PEFT's `all-linear` reaches **2.49% of the
parameters** while mis-targeting a frozen vision tower and a no-gradient indexer; reaching the
experts costs **1.42 B trainable parameters at r=8** and a **~13.5 GiB materialisation per MoE
layer forward**; and packing is impossible.

### This also answers the map's venue question, and the answer is Nebius

Three independent, verified facts, any one of which is decisive:

1. **`zai-org/GLM-5.3-Flash-BF16` is 321.32 B parameters of BF16 = 598.5 GiB of weights**, against
   498 GB of pooled memory across all four Sparks. The map's charting estimate was right; it is now
   measured, from the published BF16 repo.
2. **NeMo's own path rules the hardware out.** Its coverage doc states *"Full-model single-GPU
   checkpoint loading and training are not supported"*; EP must evenly divide 288 routed experts;
   the validated topologies are **EP72 on 9 nodes / 72 H100s** and **EP144 on 18 nodes / 144
   H100s**, at **57.68 GiB peak per GPU**. **Derived:** four Sparks would be EP4 → 72 experts/GPU
   → 155.8 GB of BF16 expert weights per 124.5 GB GB10, before optimizer state.
3. **The kernels are not there on GB10.** `transformers`' own kernel table carries the comment
   *"GB10/SM121 GDN fast path (no fla/causal_conv1d build there)"*, and NeMo's coverage doc says
   *"KDA layers use Flash Linear Attention (FLA) kernels"* and the cuDNN sparse path *"requires
   SM90 or later"* — sm_121 is not sm_90. Without FLA, 34 of 45 layers fall back to a
   pure-PyTorch chunked loop (Appendix B).

The Sparks are not a venue for training this model at any method. That is a firmer no than the
map assumed, and it does not depend on the cost model that #49 has already been told not to reuse.

### The stack, concretely

| | |
|---|---|
| Framework | `NVIDIA-NeMo/Automodel`, `main` at ≥ `9228f33c` (2026-08-28). *Not* `NVIDIA/NeMo` — that repo now redirects to `NVIDIA-NeMo/Speech` and has no LLM collection. |
| Recipe | `examples/vlm_finetune/glm5_next/glm5_3_flash_medpix_packed2k_ep72_cp2_100steps.yaml` + a `peft:` block from `examples/llm_finetune/glm/glm_5.2_lora.yaml` |
| Base | `zai-org/GLM-5.3-Flash` (FP8), loaded via the distributed-checkpoint path with `dequantize_base_checkpoint: true` |
| `transformers` | **`==5.12.1`** — NeMo owns the implementation and does *not* need 5.16 |
| Extra build | FlashMLA at `b7643bd5` from source (not on PyPI), for the cuDNN backend; or `backend.attn: sdpa` and lose 13.3% throughput |
| Venue | **Nebius H100s, ≥ 8 GPUs, realistically 9+ nodes.** Not the Sparks. |
| Fallback | `transformers==5.16.1` + `peft>=0.20.0` + `trl` if NeMo's operational cost proves too high — see §4.5 for the adapter config that avoids the traps |

---

## 1 · Support matrix

**As of 2026-08-31.** "LoRA" means *the framework's own LoRA path resolves target modules on this
architecture* — not merely "PEFT exists".

| Stack | Text training | Vision training | LoRA | Status / blocker |
|---|:---:|:---:|:---:|---|
| **NeMo Automodel** ≥ `9228f33c` | **✅ validated** | **✅ validated** (image; video not) | **⚠️ untested here** | Native `glm5_next` implementation + a published, numerically-validated VLM SFT recipe. Generic `GroupedExpertsLoRA` exists and `glm_5.x_lora.yaml` recipes exist, but **no `glm5_next` LoRA recipe**. No TP, no PP, no single-GPU. |
| `transformers` 5.16.1 + `peft` 0.20.0 (+ `trl` / own `Trainer`) | ✅ **wired, untested** | ✅ **wired, untested** | ⚠️ **partial** | All training hooks present. But `all-linear` reaches 2.49% and mis-targets; experts need `target_parameters` at 1.42 B trainable + 13.5 GiB/layer; **packing impossible**; no FlashAttention. |
| `transformers` **≤ 5.16.0** | ❌ | ❌ | ❌ | Class does not exist. `glm5_next` is in **v5.16.1 only** — merged 24 minutes before that patch release. |
| **Axolotl** `0.19.0.dev0` | ⚠️ generic fallback | ⚠️ generic fallback | ⚠️ generic | Pins **`transformers==5.16.1`** exactly, and its multimodal allowlist is *derived from* `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`, so `glm5_next` is in it automatically. But **zero** `glm5_next` references anywhere in the repo, no processing strategy, no chat template, no example. Its deep GLM-5.2 DSA kernel work does **not** carry over. |
| **Unsloth** (CUDA) | ❌ | ❌ | ❌ | Pins **`transformers<=5.5.0`** — eleven minors below the floor. Cannot construct the class at all. |
| **Unsloth** (MLX / Apple Silicon) | ✅ | ✅ | ✅ | Real support, merged [unsloth-zoo#1120](https://github.com/unslothai/unsloth-zoo/pull/1120) 2026-08-28. Irrelevant to this fleet — no Apple Silicon, and the model does not fit one. |
| **LLaMA-Factory** (`hiyouga/LlamaFactory`) | ❌ | ❌ | ❌ **silently wrong** | Zero matches in registry, templates or visual plugin. Pins `transformers<=5.8.0`, re-checked at runtime. Has never supported **any** GLM-5.x model. Unregistered multimodal types **no-op instead of raising** — `--freeze_vision_tower` is silently ignored and LoRA attaches to the vision tower. |
| **ms-swift** | ❌ **fail-fast** | ❌ | ❌ | Zero matches. Registers `GlmMoeDsaForCausalLM` (GLM-5/5.1/5.2), a *text-only* arch. Its generic fallback **explicitly raises** for multimodal configs. `transformers<5.17.0` — the pin is fine; only registration is missing. |
| **torchtune** | ❌ | ❌ | ❌ | **Deprecated 2025-07-15.** Never had any GLM model. No `transformers` dependency at all. |
| **Megatron-Bridge**, **NeMo-RL** | ❌ | ❌ | ❌ | Zero matches. Both pin `transformers<=5.12.1` *and* lack the implementation. `glm_moe_dsa` covers GLM-5.1/5.2/5.3 **text**, not Flash. |
| *(context)* **vLLM `main`** | — | — | — | `Glm5NextForConditionalGeneration` is **not in vLLM `main`** either. We serve it from a side image. See §6. |

---

## 2 · `transformers` — the reference implementation

### 2.1 Which version, exactly

**Verified.** `glm5_next` landed in
[huggingface/transformers#48342](https://github.com/huggingface/transformers/pull/48342)
("[Glm 5.3 Flash] GLM 5.3 Flash Support", by `Dovis01`, 130 commits, +7910/−110), **merged to
`main` at 2026-08-26T14:26:41Z**.

| Tag | Published | `src/transformers/models/glm5_next/` present? |
|---|---|:---:|
| `v5.16.0` | 2026-08-26T12:35:15Z | **no** — the API returns 404 for that path at that ref |
| `v5.16.1` | 2026-08-26T14:50:01Z | **yes** |

A **24-minute** gap between merge and patch release. PyPI `transformers==5.16.1` was uploaded
2026-08-26T14:48:55. **`5.16.1` is the floor** — and because it is a patch release, anything
pinning `~=5.16.0`, `==5.16.0` or `<=5.16.0` will not see the model. The checkpoint's own
`config.json` says `"transformers_version": "5.16.0"`, which is wrong by one patch.

The model dir on `main` is a complete, non-stub implementation:

```
__init__.py                        1 182 B
configuration_glm5_next.py        14 202 B
image_processing_glm5_next.py     13 325 B   Glm5NextImageProcessor(TorchvisionBackend)
image_processing_pil_glm5_next.py 12 571 B   PIL fallback
modeling_glm5_next.py            102 009 B   2 375 lines
modular_glm5_next.py              95 314 B
processing_glm5_next.py            9 598 B   Glm5NextProcessor
video_processing_glm5_next.py     15 884 B
```

Two later commits touched it — `83d46aa2` (2026-08-28, #47625) and `3cf87d87` (2026-08-31,
#48367) — and **neither is in a release yet**, so `main` ≠ `5.16.1`.

### 2.2 There is no `trust_remote_code` path, and that is good news

**Verified.** `https://huggingface.co/zai-org/GLM-5.3-Flash/tree/main` contains **no
`modeling_*.py`, no `configuration_*.py`, no `image_processing_*.py`**. The complete non-weight
file list is:

```
.gitattributes  LICENSE  README.md  chat_template.jinja  config.json
generation_config.json  model.safetensors.index.json  processor_config.json
tokenizer.json  tokenizer_config.json
```

(plus 62 `.safetensors` shards). The repo relies entirely on native support. The failure mode the
ticket anticipated — *"remote code that works for inference often breaks for training"* — **does
not apply here, because there is no remote code**. The equivalent question becomes "does the
native implementation train", and the answer is yes, with caveats.

### 2.3 The implementation is training-shaped

All **verified** in `modeling_glm5_next.py` on `main`:

| Signal | Line | Value |
|---|---|---|
| `supports_gradient_checkpointing` | 1336 | `True` |
| Decoder layer base class | 1259 | `GradientCheckpointingLayer` |
| Vision block base class | 1670 | `GradientCheckpointingLayer` |
| Loss | 2174-2176 | `labels` → `self.loss_function(...)` |
| MoE aux loss | 1983, 2179-2188 | `load_balancing_loss_func`, `router_aux_loss_coef = 0.001` |
| `pixel_values` | 2108 | first-class `forward` argument |
| Training-mode dropout | 1209, 1634, 1657 | `0.0 if not self.training else self.attention_dropout` |
| `_no_split_modules` | 1345 | `["Glm5NextTextDecoderLayer", "Glm5NextVisionBlock"]` — FSDP/`device_map` ready |
| `_keep_in_fp32_modules_strict` | 1358 | `["e_score_correction_bias", "conv1d", "dt_bias", "A_log"]` |

And `Glm5NextModelTest(VLMModelTest)` in `tests/models/glm5_next/test_modeling_glm5_next.py`
**does not skip** the inherited `test_training`, `test_training_gradient_checkpointing`,
`test_training_gradient_checkpointing_use_reentrant_false` or `..._true`
(`tests/test_modeling_common.py` L1890-1920). Gradient-checkpointed training runs in CI — on a
tiny random config, but it runs.

It does set `test_all_params_have_gradient = False  # MoE`, an explicit acknowledgement that some
parameters never receive gradients. Which brings us to the traps.

### 2.4 Five traps

#### (a) `AutoModelForCausalLM` will not load it

**Verified**, `src/transformers/models/auto/modeling_auto.py` on `main`:

| Mapping | Entry |
|---|---|
| `MODEL_MAPPING_NAMES` (L211-213) | `glm5_next → Glm5NextModel`, `glm5_next_text → Glm5NextTextModel`, `glm5_next_vision → Glm5NextVisionModel` |
| `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` (L1111) | `glm5_next → Glm5NextForConditionalGeneration` |
| `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` | **absent** |

`__all__` is exactly `["Glm5NextPreTrainedModel", "Glm5NextTextModel", "Glm5NextVisionModel",
"Glm5NextModel", "Glm5NextForConditionalGeneration"]` — **there is no `Glm5NextForCausalLM`.**

This lands directly on `dgx-manager-fine-tune-recipes`, whose `train.py` loads
`AutoModelForCausalLM`. That call raises for this model. Use `AutoModelForImageTextToText` or the
class directly, even for a text-only tune — just don't pass `pixel_values`.

#### (b) The DSA indexer runs under `@torch.no_grad()`

**Verified**, L773: `Glm5NextTextIndexer.forward` is decorated `@torch.no_grad()`. The indexer
owns `nn.Linear`s (`wq_b`, `wk`, `weights_proj`) plus an `nn.LayerNorm` across 12 layers.

`target_modules="all-linear"` **will** attach adapters to them — PEFT's
`_maybe_include_all_linear_layers` matches `isinstance(module, (nn.Linear, Conv1D))` and only
excludes the output embedding (`peft/tuners/tuners_utils.py` L2390-2440). Those adapters sit there
receiving zero gradient for the whole run. Not a crash; a silently dead slice of your adapter and
a misleading `trainable params` count.

#### (c) Sequence packing is unsupported — by the model's own test suite

**Verified**, `test_modeling_glm5_next.py` L579-585:

```python
@unittest.skip("Fundamentally incompatible with indexer - indexer has no boundary offset telling sequences apart")
def test_eager_padding_matches_padding_free_with_position_ids(self): ...

@unittest.skip("Fundamentally incompatible with indexer - indexer has no boundary offset telling sequences apart")
def test_sdpa_padding_matches_padding_free_with_position_ids(self): ...
```

Padding-free / `position_ids`-based packing is the standard SFT efficiency trick (TRL
`packing=True`, LLaMA-Factory `neat_packing`). It is **wrong** on this model under transformers:
the DSA indexer selects top-k keys across document boundaries. One example per sequence, padded.

**This is exactly what NeMo solved.** Its recipe packs at 2K with `packing_format: neat` and its
coverage doc says *"FLA carries the recurrent state across contiguous sequence shards while
preserving packed document boundaries"* — i.e. NeMo's own implementation is boundary-aware where
transformers' is not. It is one of the strongest arguments for the NeMo route.

#### (d) No FlashAttention, and O(S²) masks even though attention is "sparse"

**Verified**, L1339-1341: `_supports_flash_attn = False`, `_supports_sdpa = True`,
`_supports_flex_attn = False`, with in-source reasons *"needs index based kernel"* and *"needs per
layer creation, too expensive"*.

`build_attention_mask_from_topk` (L1218) turns the indexer's top-k indices into a mask by
allocating `selected_counts` of shape `(batch, q_len, kv_len)` in `int32`, then a bool mask. Its
docstring: *"Only supporting SDPA and Eager as we have a 3D dependency which cannot be mapped to
FA."*

**Derived:** at training time (no cache) `q_len == kv_len == S`, so each of the 12 DSA layers
transiently allocates `S²×4` bytes int32 + `S²×1` bytes bool:

| S | int32 + bool, per DSA layer |
|---:|---:|
| 8 192 | 268 MiB + 67 MiB |
| 16 384 | 1.07 GiB + 268 MiB |
| 32 768 | 4.29 GiB + 1.07 GiB |

Under gradient checkpointing it is re-allocated on recompute. DSA saves *compute*, not *mask
memory*. **Untested** — worth being among the first things a probe measures if you take the
transformers route.

#### (e) The MTP head is discarded on load

**Verified** twice over:

- `_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", r"layers\.\d+\.shared_head\."]` (L1359).
- `docs/source/en/model_doc/glm5_next.md`, first line of the overview:
  *"The implementation in transformers does not include an MTP layer."*

The checkpoint does ship it: `model.safetensors.index.json` contains
`model.language_model.layers.45.eh_proj.weight`, `.enorm.weight`, `.hnorm.weight` and
`layers.*.shared_head.norm.weight`, and there are 46 `input_layernorm` / 46 `o_proj` tensors
against `num_hidden_layers: 45`. NeMo's recipe sets `num_nextn_predict_layers: 0` explicitly,
i.e. it drops MTP too.

**This answers the map's open "does the MTP head survive a tune?" question.** A LoRA *adapter*
never touches layer 45, so **base + adapter served on the vLLM side image keeps MTP intact**. But
anything that round-trips through `save_pretrained()` — a merge, a re-quantise — **drops the MTP
head**, and with it the 1.57–1.82× decode the map values it at. If you merge, layer 45's shards
must be copied across by hand. That is now answerable, not speculative.

---

## 3 · The trainer ecosystem

### 3.1 NVIDIA NeMo Automodel — the one that works

**First, a path correction: `NVIDIA/NeMo` is no longer the LLM repo.** The GitHub API redirects
`repos/NVIDIA/NeMo` → **`NVIDIA-NeMo/Speech`**, whose `nemo/collections/` holds only
`asr audio common speechlm2 tts`; `nemo/collections/llm/gpt/model/` does not exist. Grep for
`glm5_next|Glm5Next|GLM-5.3` across that repo: **zero hits**. The training work lives in
**`NVIDIA-NeMo/Automodel`**.

**Verified**, `nemo_automodel/_transformers/registry.py`:

```python
# L91-96
("Glm5NextForConditionalGeneration",
 ("nemo_automodel.components.models.glm5_next.model", "Glm5NextForConditionalGeneration")),
# L346
"glm5_next": ("nemo_automodel.components.models.glm5_next.config", "Glm5NextConfig"),
```

**Verified**, `nemo_automodel/components/models/glm5_next/` — a complete native implementation,
not a passthrough:

```
config.py 13 697 B   cp.py 13 491 B (context-parallel)   image_processing.py 8 807 B
layers.py 41 101 B   model.py 18 500 B   processing.py 4 552 B
state_dict_adapter.py 11 517 B   vision.py 11 999 B
```

Landed in [**PR #3699**](https://github.com/NVIDIA-NeMo/Automodel/pull/3699),
*"feat(glm5-next): add GLM-5.3-Flash training support"*, created 2026-08-27, **merged
2026-08-28T07:27:29Z**, 25 files, +4646/−275 (commit `9228f33c`). Follow-ups: `cdc9147a`
(2026-08-29, docs #3744), `37ff7fea` (2026-08-28, router parity #3635).

**It does not need `transformers` 5.16.** Automodel pins `transformers==5.12.1` and owns the
implementation. Its own `config.py` header:

> *"The released checkpoint requires Transformers 5.16, while AutoModel's current runtime baseline
> predates the upstream `glm5_next` config. These classes keep the checkpoint field protocol
> stable and allow `AutoConfig` to resolve the model without remote code or a dependency bump."*

#### The published recipe — read it, it is the map's own decision written in YAML

`examples/vlm_finetune/glm5_next/glm5_3_flash_medpix_packed2k_ep72_cp2_100steps.yaml`,
**verified** by reading it in full. The load-bearing lines:

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: zai-org/GLM-5.3-Flash
  torch_dtype: bfloat16
  text_config:
    num_nextn_predict_layers: 0        # MTP head off for training
  backend:
    attn: cudnn                        # FlashMLA fwd + cuDNN Frontend bwd, SM90+
    experts: torch_mm
    dispatcher: hybridep
distributed:
  strategy: fsdp2
  tp_size: 1 ; cp_size: 2 ; pp_size: 1 ; ep_size: 72
  activation_checkpointing: true
freeze_config:
  freeze_embeddings: true
  freeze_vision_tower: true            # <- the map's decision, as a flag
  freeze_language_model: false
packed_sequence:
  max_length: 2048 ; pack_size: 2048 ; packing_format: neat
  balance_media_tokens: true
checkpoint:
  dequantize_base_checkpoint: true     # FP8 -> BF16 on load
dataset:
  path_or_dataset: mmoukouba/MedPix-VQA
optimizer: AdamW, lr 5.0e-6, betas [0.9, 0.95], weight_decay 0.1
```

Validation figures from `docs/model-coverage/vlm/thudm/glm-5-3-flash.mdx` (**verified**):

| Check | Result |
|---|---|
| HF logit parity (4-layer KDA/KDA/KDA/DSA, S=4096, full 154 880 vocab) | cosine **0.99993**, top-1 agreement **98.34%**, mean KL 2.48e-4; all 135 tensors loaded clean |
| Packed CP1 vs CP8 parity, 100 steps, EP144 | 18 nodes / 144 H100; peak **38.89 / 41.04 GiB**; final loss 1.2344 / 1.2328; mean abs Δ 0.0019 |
| cuDNN vs SDPA, EP72/CP2, 100 steps | 9 nodes / 72 H100; **9 489 vs 8 378 TPS** (+13.26%); peak **57.78 / 57.68 GiB**; loss 1.2337 / 1.2359 |

#### What it does *not* do

From the same doc, verbatim:

- *"Image training is supported; video training is not."*
- *"TP and PP are not supported for this model."*
- *"Full-model single-GPU checkpoint loading and training are not supported. Use the distributed
  checkpoint initialization path."*
- *"An EP size must evenly divide the model's 288 routed experts and fit the available GPU memory."*
- *"The cuDNN sparse-attention path requires SM90 or later."*
- The recipe is **full-parameter SFT of the language backbone**. It contains **no `peft:` block**,
  and **there is no `glm5_next` LoRA recipe** in the repo.

#### But the PEFT half exists, and the VLM recipe already accepts it

**Verified.** `nemo_automodel/components/_peft/` contains `lora.py` (38 KB), **`lora_experts.py`
(26 KB)**, `lora_kernel.py`, `lora_mlp.py`, `module_matcher.py`. `lora_experts.py` defines
`GroupedExpertsLoRA(GroupedExperts)` and `GroupedExpertsDeepEPLoRA(GroupedExpertsDeepEP)` which
**inject LoRA into the grouped-GEMM expert forward directly** — no full expert-stack
materialisation, which is precisely PEFT's weakness (§4.4).

`examples/llm_finetune/glm/glm_5.2_lora.yaml` shows the target syntax for a GLM MoE:

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules:
    - "*.self_attn.q_a_proj" ; "*.self_attn.q_b_proj"
    - "*.self_attn.kv_a_proj_with_mqa" ; "*.self_attn.kv_b_proj" ; "*.self_attn.o_proj"
    - "*.mlp.gate_proj" ; "*.mlp.up_proj" ; "*.mlp.down_proj"
    - "*.mlp.shared_experts.gate_proj" ; "*.mlp.shared_experts.up_proj" ; "*.mlp.shared_experts.down_proj"
    - "*.mlp.experts"                       # <- the 288-expert stack, first-class
  dim: 32 ; alpha: 64 ; dropout: 0.0
  use_memory_efficient_lora: true
```

And `nemo_automodel/recipes/vlm/finetune.py` reads `cfg.peft` at L507-509 and threads it through
alongside `freeze_config` at L537-538 — **so `FinetuneRecipeForVLM` already supports a `peft:`
block.**

**The gap is exactly one composition that nobody has published: the validated glm5_next VLM
recipe, plus the glm_5.2 LoRA block.** That composition is *untested* — the KDA attention module
names differ from GLM-5.2's MLA-only stack (glm5_next has `q_proj/k_proj/v_proj/f_a_proj/g_a_proj/
b_proj` on its 34 KDA layers), so the target-module globs need adjusting, and `module_matcher.py`
behaviour on the VLM module tree is unverified. **This is the probe to run.**

### 3.2 Axolotl — generic fallback only, but the pin is right

HEAD `40ea5813` (2026-08-31), version `0.19.0.dev0`.

- **Verified:** `grep -rnE 'glm5_next|Glm5Next|GLM-5\.3|5\.3-Flash'` over the **entire repo tree** →
  **zero hits**. Issue and PR searches for `GLM-5.3`, `glm5_next`, `"GLM-5.3-Flash"` → **zero
  results each**.
- **Verified**, `pyproject.toml`: `"transformers==5.16.1"` (L20), `"peft==0.20.0"` (L18),
  `"trl==1.9.0"` (L23). The transformers bump landed in `536a2899` /
  [PR #3957](https://github.com/axolotl-ai-cloud/axolotl/pull/3957), merged 2026-08-28 — a routine
  chore bump whose body does not mention GLM at all. **It is nonetheless the exact release that
  carries `glm5_next`**, so Axolotl on `main` can construct the class.
- **Verified**, `src/axolotl/loaders/constants.py` — the multimodal allowlist is *derived*, not
  hand-maintained: `MULTIMODAL_AUTO_MODEL_MAPPING = dict(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)`
  plus two manual additions. So `glm5_next` is in it automatically, `cfg.is_multimodal` becomes
  `True`, and `loaders/model.py` picks `AutoModelForImageTextToText`.
- **What is missing:** no processing strategy. `get_processing_strategy` (`processing_strategies.py`
  L1461) dispatches through the model-support registry, then a hardcoded `chat_template_type`
  chain, then processor-`isinstance` checks (including `Glm4vProcessor`, `Glm46VProcessor`) — none
  match. It falls through to the **base** `ProcessingStrategy`, which has no glm5_next-aware
  image-token expansion or label masking. `docs/multimodal.qmd` lists 22 supported VLMs; GLM-5.3-Flash
  is not among them, and the page is labelled BETA with "limited" feature parity.
- Axolotl *does* carry deep GLM-**5.2** work — `src/axolotl/integrations/kernels/libs/glm_dsa/`
  (DSA / Lightning-Indexer training kernels), `adapters/glm_moe_dsa.py`,
  `examples/glm_moe_dsa/glm-5.2-nvfp4-lora.yaml`. **None of it carries over**: GLM-5.3-Flash is a
  different architecture (KDA + KPool-DSA + mHC, multimodal).

### 3.3 Unsloth — MLX only; the CUDA path is hard-blocked

**Verified**, `unsloth/pyproject.toml` L171 and L720 (and `unsloth-zoo` L55/L100) carry the
identical string ending **`,<=5.5.0`**. Eleven minor versions below the 5.16.1 floor. Under a
supported install, CUDA Unsloth cannot construct `Glm5NextForConditionalGeneration` at all.

Real support does exist — on **MLX / Apple Silicon**. `glm5_next` appears at
`unsloth_zoo/mlx/loader.py` L2638 (`_VLM_TEXT_PATH_MODEL_TYPES`) and `unsloth_zoo/mlx/utils.py`
L3109 (`_VLM_ARRAY_GRID_MODEL_TYPES`), via
[**unsloth-zoo#1120**](https://github.com/unslothai/unsloth-zoo/pull/1120) (merged 2026-08-28),
whose body says *"Neither trained before this"* and *"both `qwen4_exp` and `glm5_next` are absent
from transformers, and no transformers version can make them loadable when mlx-vlm is what builds
them"*. Its own caveat: *"`glm5_next`, whose full checkpoint does not fit locally, was verified
against a shrunken but otherwise real one rather than by argument."*

Corroborating the CUDA block: [unsloth#9878](https://github.com/unslothai/unsloth/pull/9878)
(merged 2026-08-27) is titled *"fix(studio): stop offering a transformers upgrade where it cannot
load anything"*. `unsloth/import_fixes.py` L3403-3406 maps `glm4_moe`, `glm4_moe_lite`,
`glm4v_moe`, `glm_moe_dsa` to the `qwen2_moe` patch family — **`glm5_next` is absent**, and
`grep -i glm unsloth/models/mapper.py` returns nothing.

Irrelevant to this fleet regardless: no Apple Silicon, and 598.5 GiB does not fit a Mac.

### 3.4 LLaMA-Factory — no, and it fails silently

Repo renamed to **`hiyouga/LlamaFactory`** (the old name 422s on `gh search`). HEAD `d6bb97dd`
(2026-08-31).

- **Verified:** `grep -E "glm5|glm_5|5\.3|5_3|Glm5"` over `src/llamafactory/extras/constants.py`
  (3 688 lines), `src/llamafactory/data/template.py` (2 503 lines) and
  `src/llamafactory/model/model_utils/visual.py` → **zero matches in all three**. Highest GLM
  registered is `GLM-4.7-Flash` and `GLM-4.6V`. Repo-wide code search for `glm5_next` and `GLM-5`
  → **zero results** (control: `glm4_7` → 2 results, so the index works).
- **It has never supported any GLM-5.x model** — no PR, no issue, for GLM-5, 5.1 or 5.2.
- **Verified**, `pyproject.toml` L43: `"transformers>=4.55.0,<=5.8.0,!=4.57.0,!=5.6.0"`, re-checked
  **at runtime** in `src/llamafactory/extras/misc.py` L97. Eight minors below the floor.
  (Bypassable via `DISABLE_VERSION_CHECK=1`, and the pin is demonstrably stale — open issue
  #10788 / PR #10789, both 2026-08-25, show maintainers adapting to transformers 5.15.)
- **The silent-failure risk is the reason to avoid it even as a hack.** `visual.py` L177 and L201
  both guard on `if model_type in COMPOSITE_MODELS`; for an unregistered type they **no-op instead
  of raising**. So `--freeze_vision_tower` is silently ignored and `patch_target_modules` returns
  targets unfiltered — **LoRA attaches to the vision tower the map decided to freeze**, with no
  error. Cadence context: GLM-4.7-Flash took 21 days from HF release to merged support; the entire
  GLM-5 line was skipped.

### 3.5 ms-swift — no, but it fails loudly, and it is the one to watch

HEAD `5e43bdff` (2026-08-31). Note `swift/llm/` no longer exists; the tree is now
`swift/model/models/glm.py`, `swift/model/constant.py`, `swift/template/templates/glm.py`.

- **Verified:** `grep -E "glm5_next|Glm5Next|GLM-5\.3"` over `swift/model/models/glm.py`,
  `swift/model/constant.py`, `swift/template/constant.py`, `swift/template/templates/glm.py` →
  **zero matches**. Repo-wide code search → zero. Issue search for `GLM-5.3` → `total_count: 0`.
- What *is* registered (`swift/model/models/glm.py` L498-518): GLM-5, GLM-5.1, GLM-5.2 under
  `LLMModelType.glm_moe_dsa` with `architectures=['GlmMoeDsaForCausalLM']` — a **text-only**
  architecture that cannot absorb `Glm5NextForConditionalGeneration`.
- **The pin is not the blocker:** `requirements/framework.txt` L35 is `transformers>=4.33,<5.17.0`,
  which admits 5.16.1. Only registration is missing.
- **The generic fallback is explicitly closed to multimodal models.** `swift/model/model_meta.py`
  L283-289: when no `model_type` matches, `if model_info.is_multimodal: raise ValueError(...)`,
  else fall back to `template='dummy'`. And `HfConfigFactory.is_multimodal` returns `True` for any
  config carrying `text_config` or `vision_config` — GLM-5.3-Flash has both. A manual
  `--model_type` also raises (L233-234). **Fail-fast, not silent** — better behaviour than
  LLaMA-Factory, but equally a no.
- **Cadence says this could flip fast.** `zai-org/GLM-5.2` was published 2026-06-16T07:39; ms-swift
  PR #9581 was created 2026-06-17T03:16 and merged 2026-06-17T05:59 — **~22 hours**. GLM-5.3-Flash
  at six days with no PR, no issue and no commit is a conspicuous gap, most plausibly because
  `glm5_next` needs a new MLLM `ModelMeta` + loader + multimodal template rather than the one-line
  `ModelGroup` append that GLM-5.2 was. It also has a real escape hatch: `register_model` +
  `register_template` loaded via `--external_plugins`, documented in
  `docs/source_en/BestPractices/MLLM-Registration.md`.

### 3.6 torchtune — deprecated

`pytorch/torchtune` redirects to **`meta-pytorch/torchtune`**. `main`'s HEAD is `bd2a0fc7`, dated
**2026-04-23**, whose commit message is literally *"docs: note torchtune wind-down in README"*.
README line 1:

> ⚠️ **Torchtune is no longer actively maintained:** torchtune development wound down in 2025 — see
> [The future of torchtune](https://github.com/meta-pytorch/torchtune/issues/2883).

Issue #2883 (open, created **2025-07-15**): *"we are stopping active development on torchtune,
effective immediately… No new features will be added to the library."* `torchtune/models/` and
`recipes/configs/` contain **no `glm*` directory of any generation**, and the repo has **no
`transformers` dependency at all** — so it cannot even inherit support. Dead end.

### 3.7 Megatron-Bridge and NeMo-RL — no

**Verified:** `NVIDIA-NeMo/Megatron-Bridge` (HEAD `99c79f44`) → zero `glm5_next` hits;
`src/megatron/bridge/models/` has `glm`, `glm_moe_dsa`, `glm_vl`, where `glm_moe_dsa` is the
GLM-5.1/5.2/5.3 **text** line. Pin `transformers>=5.8,<=5.12.1`. `NVIDIA-NeMo/RL` (HEAD `e09e977b`)
→ zero hits, pin `transformers>=5.5.0,<=5.12.1`.

### 3.8 Global cross-check

`gh api -X GET search/code -f q='glm5_next'` returns ~1 028 matches. Distinct repos in the first
100 results:

```
1CatAI/1Cat-vLLM   Blaizzy/mlx-vlm   JustVugg/colibri   NVIDIA-NeMo/Automodel
RobTand/prismaquant   asher/gmlx   guqiong96/Lsglang   huggingface/transformers
jjang-ai/jangq   jjang-ai/vmlx   jundot/omlx   modular/modular   mudler/vllm.cpp
wkljohn/ds4-strix-halo-tp-odinlink   wtdcode/vllm-backport
yhfgyyf/vllm-deepseek-v4-sm89   zai-org/GLM-5   ztxz16/fastllm
```

**`NVIDIA-NeMo/Automodel` is the only training framework in that list.** `huggingface/transformers`
is the reference implementation, `Blaizzy/mlx-vlm` the Apple-Silicon one; everything else is
inference, serving or deployment configs. Caveat: `unslothai/unsloth-zoo` is **absent** from the
index despite carrying `glm5_next` at two verified locations — GitHub code search lags, so treat
this as a lower bound.

---

## 4 · PEFT

### 4.1 No `glm5_next` entry, and nobody has asked

**Verified.** `grep -n "glm5\|Glm5"` over `peft/utils/constants.py` (the
`TRANSFORMERS_MODELS_TO_*_TARGET_MODULES_MAPPING` tables) and `peft/tuners/lora/config.py` on
`main` → **zero matches**; the newest GLM entry is `chatglm`.
`gh api "search/issues?q=repo:huggingface/peft+glm5_next"` → **`total_count: 0`**. A search across
the whole `huggingface` org for `glm5_next` returns **only the five transformers PRs** — nothing in
`peft`, `trl` or `accelerate`.

Absence from `constants.py` is not fatal; it only means you must pass `target_modules` explicitly.
The `all-linear` shorthand still works. It is just wrong here.

### 4.2 The question was backwards: `all-linear` does *not* target the 288 experts

The ticket asked whether `all-linear` "catastrophically targets 288 experts". **It does the
opposite — it cannot see them at all.**

**Verified**, `modeling_glm5_next.py` L107-133:

```python
@use_experts_implementation
class Glm5NextTextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""
    def __init__(self, config):
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj    = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
```

The router (L144-155) is likewise a bare `nn.Parameter` (`self.weight`) plus an `nn.Buffer`
(`e_score_correction_bias`). PEFT's `all-linear` matches
`isinstance(module, (torch.nn.Linear, Conv1D))` only; bare `nn.Parameter`s are invisible to it.

So the router is safe by accident — good, you do not want a LoRA on a MoE router — and the experts
are unreachable.

**Layout note:** on disk the checkpoint stores experts *unfused*, as 12 384 separate
`model.language_model.layers.N.mlp.experts.M.{gate,up,down}_proj.weight` tensors per projection
(43 sparse layers × 288 experts). transformers fuses them into 3-D `nn.Parameter`s at load.
Adapter names are against the **fused runtime** layout, not the on-disk one.

### 4.3 What `all-linear` actually reaches: 2.49%

**Derived** from `config.json` (`hidden_size 4096`, `moe_intermediate_size 2048`,
`intermediate_size 12288`, `n_routed_experts 288`, `n_shared_experts 1`; 3 dense + 42 sparse MLP
layers + 1 MTP layer; 34 KDA + 11 DSA attention layers; vision `depth 24 / hidden 1024`) and
cross-checked against the HF API's `321 323 031 390` total — the model below accounts for
**320.93 B of 321.32 B (99.88%)**.

| Block | Module kind | Params | Share | Reachable by `all-linear`? |
|---|---|---:|---:|:---:|
| Routed experts (288 × 43) | 3-D `nn.Parameter` | **311.65 B** | **96.99%** | **no** — needs `target_parameters` |
| KDA attention projections | `nn.Linear` | 4.56 B | 1.42% | yes |
| MLA attention projections | `nn.Linear` | 1.41 B | 0.44% | yes |
| `embed_tokens` + `lm_head` | `nn.Embedding` / `nn.Linear` | 1.27 B | 0.39% | `lm_head` excluded by PEFT |
| `shared_experts` (43 layers) | `nn.Linear` | 1.08 B | 0.34% | yes |
| Dense MLP (layers 0-2) | `nn.Linear` | 0.45 B | 0.14% | yes |
| Vision tower (24 blocks) | `nn.Linear` | 0.41 B | 0.13% | yes — **but frozen by decision** |
| DSA indexer (12 layers) | `nn.Linear` | 0.09 B | 0.03% | yes — **but `@torch.no_grad()`** |

- `all-linear` reachable: **8.01 B = 2.49%** of the model.
- After removing the vision tower (frozen by the map's own decision) and the no-grad indexer:
  **7.51 B = 2.34%**.

An adapter over 2.34% of a 320 B model, none of it in the FFN capacity that holds most of the
knowledge, is a plausible way to teach *format and style* and an implausible way to teach *new
visual grounding*. That is a live quality risk for chat3d's use case and belongs on the map —
and it is a further argument for NeMo, whose `"*.mlp.experts"` target reaches the other 97%
without PEFT's cost structure.

### 4.4 Reaching the experts with PEFT works, but is not cheap

**Verified.** `peft` **0.20.0** (PyPI, 2026-07-28, current release) has
`LoraConfig.target_parameters`, added for exactly this shape. From `peft/tuners/lora/config.py`
L588-600:

> *"in many mixture of expert (MoE) layers in HF Transformers, instead of using `nn.Linear`, an
> `nn.Parameter` is used… to apply LoRA to that parameter, it needs to be targeted with
> `target_parameters`. As an example, for Llama4, you can pass
> `target_parameters=['feed_forward.experts.gate_up_proj', 'feed_forward.experts.down_proj']`."*

The glm5_next analogue is `target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]`.
**Untested** on this model, but the mechanism is generic and the names match.

Two costs, both **derived** from the config and PEFT's implementation:

**(i) Per-expert adapters — trainable parameters scale with 288.** `ParamWrapper.__init__`
(`peft/tuners/lora/layer.py` L2267+) builds `lora_A = nn.Linear(in_features, r * num_experts)` and
`lora_B = nn.Linear(r * num_experts, out_features)` — a *separate rank-r adapter per expert*:

| rank | trainable params | Adam fp32 states | BF16 adapter file |
|---:|---:|---:|---:|
| r=4 | 710 M | 7.9 GiB | 1.32 GiB |
| **r=8** | **1 420 M** | **15.9 GiB** | 2.65 GiB |
| r=16 | 2 841 M | 31.7 GiB | 5.29 GiB |

For comparison, #49's Qwen3.8-27B LoRA had **10.5 M** trainable parameters. This is two orders of
magnitude larger, and not "parameter-efficient" in the sense the map's method decision assumed.

**(ii) A full expert-stack materialisation per MoE layer forward.** PEFT applies the adapter via
`torch.nn.utils.parametrize.register_parametrization` with a `_LoraFactorsProxy` whose entire
`forward` is `torch.baddbmm(W, lhs, rhs, alpha=scaling)` (L2237-2255). Its docstring is explicit
that `baddbmm` avoids materialising a *second* full-size tensor — it still materialises **one**:

| Targeted parameter | Shape | BF16 |
|---|---|---:|
| `experts.gate_up_proj` | `(288, 4096, 4096)` | 9.00 GiB |
| `experts.down_proj` | `(288, 4096, 2048)` | 4.50 GiB |
| | **per MoE layer forward** | **13.50 GiB** |

With gradient checkpointing the peak *should* stay around one layer's worth rather than 43×, but
that is **untested** and is exactly the sort of assumption that turns into a 600 GiB OOM if wrong.
**This is the single sharpest reason to prefer NeMo**, whose `GroupedExpertsLoRA` injects into the
grouped-GEMM expert forward instead of reconstructing the stack.

Two smaller constraints from the same class: `ParamWrapper` **raises** if `lora_dropout != 0`
(*"It's not possible to factor out x from `lora_B(lora_A(dropout(x)))`"*), and multiple adapters
must all target the same parameter set.

### 4.5 If you take the transformers route anyway: the adapter config

Do **not** pass `target_modules="all-linear"`. Name the modules:

```python
LoraConfig(
    r=16, lora_alpha=32,
    lora_dropout=0.0,   # MUST be 0 if you add target_parameters
    target_modules=[
        # MLA — 11 DSA layers (+ the MTP layer's twin, which is dropped on load)
        "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
        # KDA linear attention — 34 layers
        "q_proj", "k_proj", "v_proj",
        # everywhere
        "o_proj",
        # shared_experts + the 3 dense MLP layers
        "gate_proj", "up_proj", "down_proj",
    ],
    # Optional, expensive, and the only way to touch 97% of the model:
    # target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
)
```

Explicitly excluded, and why: `indexer.*` (`@torch.no_grad()` — dead adapters), `visual.*`
(frozen by decision), `lm_head` (PEFT excludes it under `all-linear` anyway), `mlp.gate` (the MoE
router — an `nn.Parameter`, and you do not want to perturb routing during a small tune).

---

## 5 · The multimodal path

The map's decision — mixed text+image data, vision tower frozen — is **supported by the code**,
with one gap that lands on the data pipeline rather than the model.

### 5.1 Processor: complete and torch-native

**Verified.** `Glm5NextProcessor` (`processing_glm5_next.py`) wires `image_processor`, `tokenizer`
and `video_processor`. The HF repo's `processor_config.json` declares
`"processor_class": "Glm5NextProcessor"`, `"image_processor_type": "Glm5NextImageProcessor"`,
`"video_processor_type": "Glm5NextVideoProcessor"`, with `min_image_tokens: 16`,
`max_image_tokens: 8000`, `patch_size: 14`, `merge_size: 2` and CLIP normalisation constants.

`Glm5NextImageProcessor` subclasses **`TorchvisionBackend`** — a fast, torch-native processor with
`smart_resize` (`image_processing_pil_glm5_next.py` is the PIL fallback). Good for training
throughput.

`Glm5NextProcessorKwargs._defaults` sets `"return_mm_token_type_ids": True`, and
`Glm5NextForConditionalGeneration.forward` accepts `mm_token_type_ids` — the placeholder-mask
plumbing VLM trainers need is present and default-on.

### 5.2 `pixel_values` reaches the loss

**Verified**, `forward` signature L2100-2117: `pixel_values`, `pixel_values_videos`,
`image_grid_thw`, `video_grid_thw`, `mm_token_type_ids` and `labels` are all top-level arguments on
the same call. `Glm5NextModel.get_image_features` / `get_placeholder_mask` (L1859, L1873) splice
vision features into `inputs_embeds`, and there is a dedicated test
`test_image_and_video_placeholder_masks_are_disjoint`. This is a genuine end-to-end VLM training
path, not a bolt-on.

Freezing the tower is a plain `requires_grad_(False)` over `model.model.visual` (parameter prefix
`model.visual.*` in the checkpoint) — or `freeze_vision_tower: true` under NeMo.

### 5.3 The chat template renders image turns — but has no `{% generation %}` block

**Verified.** `chat_template.jinja` L51-68 defines
`{%- macro emit_image() -%}<|begin_of_image|><|image|><|end_of_image|>{%- endmacro -%}` and
dispatches on `item.type in ['image', 'image_url']` inside `visible_text()`.
`Glm5NextProcessor.replace_image_token` then expands `<|image|>` to
`image_grid_thw.prod() // merge_size**2` copies. **Rendering is correct for training.**

What is missing: **the template contains no `{% generation %}` block** (grep count: 0). That is the
marker `apply_chat_template(..., return_assistant_tokens_mask=True)` needs to emit an
assistant-token mask. Without it, **loss masking must be built by hand** — render prompt-only and
prompt+response and diff the lengths, or locate the `<|assistant|>` token id.

This lands squarely on the map's known gap
([#55](https://github.com/kreuzhofer/dgx-manager/issues/55): `lib/dataset.py` has no image path).
Whoever writes that path owns the masking. NeMo sidesteps it — its VLM dataset builders
(`nemo_automodel/components/datasets/vlm/`) do their own masking.

### 5.4 It is a reasoning model, and the template will bite

**Verified**, template tail:

```jinja
{%- if add_generation_prompt -%}
    <|assistant|>{{- '<think>' -}}
{%- endif -%}
```

and in the assistant branch, when no `reasoning_content` is present the template emits
`{{ '<think></think>' }}`. Per `zai-org/GLM-5`'s README, `reasoning_effort` accepts `low`/`high`/
`max` and **defaults to `max`**.

So every training target must carry an explicit `<think>…</think>` segment, empty or real. Train
on assistant turns rendered without one and you teach the model that an *empty* reasoning block is
the correct response prefix. Given memory `qwen38-reasoning-effort-xhigh-default` — the same class
of default-reasoning-effort bug already bit this fleet once on Qwen3.8 — this deserves an explicit
assertion in the dataset builder.

---

## 6 · The asymmetry, sharpened

The ticket's premise was that inference support runs far ahead of training support. **It is
stranger than that: `Glm5NextForConditionalGeneration` is not in vLLM `main` either.**

**Verified** via `gh api repos/vllm-project/vllm/git/trees/main?recursive=1` — the only `glm5`
paths in the entire tree are
`benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa_glm5.yaml` and
`vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py`.
`vllm/model_executor/models/registry.py` lists `glm`, `glm4`, `glm4_moe`, `glm4_moe_lite`,
`GlmMoeDsa`, `glm4v`, `glm_ocr`, `glmasr` — **no `Glm5Next`**. Meanwhile there are dozens of open
vLLM issues about `glm5next` runtime behaviour (#54591, #54458, #54451, #54359, #54317 …), so it
plainly runs — just not from `main`.

The timeline, all **verified**:

| Time (UTC) | Event |
|---|---|
| 2026-08-25 06:43 | `zai-org/GLM-5.3-Flash` published on HF |
| 2026-08-26 09:46:30 | vLLM **v0.28.0** released — *without* the class |
| **2026-08-26 14:10:08** | Docker tag **`vllm/vllm-openai:glm53-flash-arm64-cu130`** pushed (9.71 GB) |
| 2026-08-26 14:26:41 | transformers **#48342** merged to `main` |
| 2026-08-26 14:50:01 | transformers **v5.16.1** released |
| 2026-08-27 | NeMo Automodel **#3699** opened |
| **2026-08-28 07:27:29** | NeMo Automodel **#3699** merged — first training support anywhere |
| 2026-08-28 | Unsloth-zoo **#1120** merged — MLX training support |
| 2026-08-28 07:02:10 | Axolotl bumps to `transformers==5.16.1` (incidentally, not for GLM) |

Serving was in a pullable image **16 minutes before the model class existed in transformers `main`**.
Training support followed **two days later**, in exactly one framework — which is fast by any
historical standard, and is the reason this ticket's headline answer is "yes" rather than "no".

**One vendor claim to discount.** `zai-org/GLM-5`'s README has a *"Fine-tuning GLM-5 Series
Models"* section naming Slime (v0.3.0+) and ms-swift (v4.4.0+). **It is stale.** `git log` on that
file shows the section was added in commit `431634c1` on **2026-07-09** — seven weeks before
GLM-5.3-Flash existed — and the three GLM-5.3 update commits (`f6adb53a`, `153ca465`, `c9d77905`,
2026-08-26/27) did not touch it; fetching the README at `25206af8` (2026-08-11, pre-Flash) shows a
byte-identical section. It is a claim about the GLM-5 dense-MoE line, not about `glm5_next`, and
§3.5 confirms ms-swift has no `glm5_next` registration. By contrast the **serving** list *was*
updated for the release (SGLang, vLLM, TokenSpeed, transformers, KTransformers, Unsloth) — Unsloth
appears there under *serving*, and its GLM-5.3 page covers GGUF and llama.cpp only, with no
fine-tuning section.

---

## 7 · What would have to land

**Nothing that would add training support to a framework currently lacking it is in flight.** The
two `glm5_next` training PRs anywhere — NeMo Automodel #3699 and unsloth-zoo #1120 — are both
already **merged**; the only open `glm5_next` PRs found are unsloth-zoo #1126 (MoE tensor names for
GGUF *export*) and #1137 (MLX gated-delta VJP test hardening). LLaMA-Factory, ms-swift, Axolotl and
torchtune have **zero** open issues or PRs mentioning it. But the gap between here and a runnable
PEFT job is small and well-defined.

| # | What | Where | Who can do it | Effort |
|---|---|---|---|---|
| **1** | **A `glm5_next` LoRA recipe** — the validated VLM recipe plus a `peft:` block, with target globs adapted from `glm_5.2_lora.yaml` to glm5_next's KDA module names | `NVIDIA-NeMo/Automodel`, `examples/vlm_finetune/glm5_next/` | **us** — it is a YAML file | **hours**, plus a multi-node run to validate |
| 2 | Confirm `module_matcher.py` + `GroupedExpertsLoRA` bind correctly on the VLM module tree (VLM recipes have only ever been run full-FT; LoRA recipes have only ever been run on text models) | same | us, or an upstream issue | the actual risk in (1) |
| 3 | ms-swift registration: `MLLMModelType.glm5_next`, a `Glm5NextLoader`, `register_model(... architectures=['Glm5NextForConditionalGeneration'], requires=['transformers>=5.16.1'])`, a `TemplateType.glm5_next`, and a `_deepspeed_set_z3_leaf_modules` branch | `modelscope/ms-swift` | upstream, or **us via `--external_plugins`** (documented in `docs/source_en/BestPractices/MLLM-Registration.md`) | days |
| 4 | Axolotl: a `ModelSupport` descriptor via `@register_model_support` with a `processing_strategy_cls`, plus a chat template and example | `axolotl-ai-cloud/axolotl` | upstream, or out-of-tree — the registry explicitly supports plugin registration | days |
| 5 | A `{% generation %}` block in the chat template, so `return_assistant_tokens_mask=True` works | `zai-org/GLM-5.3-Flash` on HF | vendor only | unknowable |
| 6 | An FLA / `causal_conv1d` build for sm_121, so KDA has a real kernel on GB10 | `kernels-community/fla`, or `Atlas-Inference/gdn`-style special case | upstream | **irrelevant** — 598.5 GiB still does not fit four Sparks |
| 7 | LLaMA-Factory: raise the `<=5.8.0` pin, register the model group, template, mm_plugin **and** `_register_composite_model` (without the last, LoRA silently hits the vision tower) | `hiyouga/LlamaFactory` | upstream | weeks; it has skipped the whole GLM-5 line |

**Watch-list, in priority order:** (a) `NVIDIA-NeMo/Automodel` `examples/vlm_finetune/glm5_next/`
for a LoRA recipe landing upstream; (b) `modelscope/ms-swift` — its pin already admits 5.16.1, it
shipped GLM-5.2 within 22 hours, and only registration is missing; (c)
`axolotl-ai-cloud/axolotl` `src/axolotl/model_support/`.

---

## Appendix A — the model, by the numbers

All **verified** from the HF API and `config.json`.

| | `zai-org/GLM-5.3-Flash` | `zai-org/GLM-5.3-Flash-BF16` |
|---|---|---|
| Total parameters | 321 323 031 390 | 321 323 031 390 |
| dtype split | 314.40 B `F8_E4M3` + 6.93 B `BF16` + 0.30 M `F32` | 321.32 B `BF16` + 0.30 M `F32` |
| On disk | 328.4 GB / 305.8 GiB | ~642.6 GB / **598.5 GiB** |
| Shards | 62 | — |
| Downloads (30 d) | 379 271 | 8 648 |

Architecture (`config.json`): `Glm5NextForConditionalGeneration`, `model_type: glm5_next`;
`text_config.model_type: glm5_next_text`, `vision_config.model_type: glm5_next_vision`.
45 layers = 34 `linear_attention` (KDA) + 11 `deepseek_sparse_attention`; `mlp_layer_types` =
3 dense + 42 sparse; `n_routed_experts: 288`, `num_experts_per_tok: 8`, `n_shared_experts: 1`,
`moe_intermediate_size: 2048`, `hidden_size: 4096`, `vocab_size: 154 880`,
`max_position_embeddings: 1 048 576`, `mla_use_nope: true`, `qk_rope_head_dim: 0`,
`index_topk: 2048`, `index_kpool: 4`, `num_nextn_predict_layers: 1` (the MTP head), `mhc: true`,
`hc_mult: 4`, `router_aux_loss_coef: 0.001`, `scoring_func: sigmoid`, `topk_method: noaux_tc`.
Vision: `depth 24`, `hidden_size 1024`, `out_hidden_size 4096`, `image_size 448`, `patch_size 14`,
`spatial_merge_size 2`.

**The 598.5 GiB BF16 figure is the venue-deciding number.** The map's charting note said
"~640 GB against 498 GB of pooled unified memory across all four nodes"; that is now verified
exactly, from the published BF16 repo, not estimated. Weights alone do not fit the Spark cluster —
before a single byte of optimizer state, activation, or PEFT materialisation.

**Derived, for scale:** at NeMo's EP72 the 311.65 B of routed-expert weights are spread over 72
GPUs = 4.33 B/GPU ≈ 8.7 GB, and measured peak is 57.68 GiB/GPU (full-parameter SFT, so optimizer
state dominates). At EP4 on four Sparks the same experts would be 77.9 B/GPU ≈ **155.8 GB of
weights per 124.5 GB GB10** — over budget before anything else exists.

## Appendix B — the kernel situation on GB10

**Verified**, `src/transformers/integrations/hub_kernels.py` on `main`, in the default kernel
mapping (L156):

```python
# GB10/SM121 GDN fast path (no fla/causal_conv1d build there); dense and MoE share it.
"Qwen3_5GatedDeltaNet": {
    Device(type="cuda", properties=CUDAProperties(min_capability=121, max_capability=121)):
        LayerRepository(repo_id="Atlas-Inference/gdn", ...),
},
```

That special case exists for Qwen3.5's gated delta net. **There is no sm_121 entry for `chunk_kda`
or `fused_recurrent_kda`** — both map only to `kernels-community/fla` under a generic `"cuda"` key
(L497-524). `use_kernel_func_from_hub_with_fallback` (L822+) documents its priority as *"1. HF
kernels (if requested) 2. Original package 3. Torch only path"*, with hub kernels off unless the
user passes `use_kernels=True`.

**So on a DGX Spark, `chunk_kimi_delta_attention` runs the pure-PyTorch fallback** at
`modeling_glm5_next.py` L479-580: a Python loop over `total_sequence_length // chunk_size` chunks,
each containing an inner `for i in range(1, chunk_size)` of 64 clone-and-matmul steps, all building
autograd graph in fp32, across **34 of 45 layers**. It is differentiable and it is correct. It is
not a training path to schedule around.

NeMo's coverage doc states the same constraint from the other side: *"KDA layers use Flash Linear
Attention (FLA) kernels"*, and *"the cuDNN sparse-attention path requires SM90 or later"* — sm_121
is not sm_90. Combined with Appendix A, this is the second independent reason the Sparks are the
wrong venue for this model.

## Appendix C — reproducing these checks

```bash
# transformers: which release first carries glm5_next
gh api "repos/huggingface/transformers/contents/src/transformers/models/glm5_next?ref=v5.16.0"  # 404
gh api "repos/huggingface/transformers/contents/src/transformers/models/glm5_next?ref=v5.16.1"  # 200
gh api repos/huggingface/transformers/pulls/48342 --jq '{merged_at, state, user: .user.login}'

# HF repo: no remote code; the BF16 base and its size
curl -s "https://huggingface.co/api/models/zai-org/GLM-5.3-Flash" \
  | python3 -c "import json,sys;[print(s['rfilename']) for s in json.load(sys.stdin)['siblings'] if not s['rfilename'].endswith('.safetensors')]"
curl -s "https://huggingface.co/api/models/zai-org/GLM-5.3-Flash-BF16?expand[]=safetensors"

# the traps, straight out of the source
curl -sL https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/glm5_next/modeling_glm5_next.py \
  | grep -n "supports_gradient_checkpointing\|_supports_flash_attn\|torch.no_grad\|_keys_to_ignore_on_load_unexpected\|nn.Parameter(torch.empty"
curl -sL https://raw.githubusercontent.com/huggingface/transformers/main/tests/models/glm5_next/test_modeling_glm5_next.py | grep -n "unittest.skip"
curl -sL https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/integrations/hub_kernels.py | grep -n "SM121"

# NeMo Automodel: the one framework that has it
curl -sL https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/nemo_automodel/_transformers/registry.py | grep -n glm5_next
curl -sL https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/examples/vlm_finetune/glm5_next/glm5_3_flash_medpix_packed2k_ep72_cp2_100steps.yaml
curl -sL https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/examples/llm_finetune/glm/glm_5.2_lora.yaml
curl -sL https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/docs/model-coverage/vlm/thudm/glm-5-3-flash.mdx

# the pins that decide who is even in the running
curl -sL https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/pyproject.toml       | grep -n "transformers\|peft\|trl"
curl -sL https://raw.githubusercontent.com/unslothai/unsloth/main/pyproject.toml              | grep -n "transformers>="
curl -sL https://raw.githubusercontent.com/hiyouga/LlamaFactory/main/pyproject.toml           | grep -n transformers
curl -sL https://raw.githubusercontent.com/modelscope/ms-swift/main/requirements/framework.txt | grep -n transformers

# vLLM: still not in main
gh api "repos/vllm-project/vllm/git/trees/main?recursive=1" --jq '.tree[].path' | grep -i glm5
```
