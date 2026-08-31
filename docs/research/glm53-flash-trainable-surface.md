# The trainable surface of GLM-5.3-Flash

> Research for [#66](https://github.com/kreuzhofer/dgx-manager/issues/66). Part of the
> [Can we fine-tune GLM-5.3-Flash, and where? wayfinder map](https://github.com/kreuzhofer/dgx-manager/issues/65).
>
> **Every number in the tables below was read off a primary artefact on 2026-08-31**: the
> `config.json`, the `model.safetensors.index.json`, and the **safetensors headers of all 62
> shards** of `zai-org/GLM-5.3-Flash` (and all 121 shards of `LibertAIDAI/GLM-5.3-Flash-NVFP4`),
> fetched by HTTP range request so that every tensor's dtype and shape is ground truth rather
> than an inference from config dims. Module semantics are read from
> `transformers/models/glm5_next/modeling_glm5_next.py` on `main`.
>
> Rows marked **derived** are arithmetic over those readings. Rows marked **inferred** are the
> few places where I am reasoning rather than reading, and they say so.

---

## Headline

Five things a downstream memory or method calculation must not get wrong:

1. **321.32 B parameters exactly**, of which **94.7% are routed MoE experts**. The published
   "320B/18B" is honest rounding: computed active is **17.38 B**.
2. **The native checkpoint is FP8, not BF16.** `zai-org/GLM-5.3-Flash` ships block-quantized
   FP8 at 305.8 GiB. There *is* an official BF16 sibling — `zai-org/GLM-5.3-Flash-BF16`, **598.5
   GiB** — which is where any full-precision workflow has to start.
3. **The language model's attention is SPLIT, but only for 34 of 45 layers.** The 34 KDA layers
   have `q_proj`/`k_proj`/`v_proj`/`o_proj`. The 11 sparse-MLA layers **do not** — they are
   LoRA-factorised into `q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`. The textbook
   PEFT target list silently covers 34/45 layers' QKV.
4. **The vision tower's attention is FUSED (`attn.qkv`)** — the exact Qwen3.8-27B trap from
   [#47](https://github.com/kreuzhofer/dgx-manager/issues/47) is present again. But here it
   works *in our favour*, because the map already decided the vision tower is frozen. The
   **real** hazard is the mirror image: the vision tower's MLP and the VL projector use
   `gate_proj`/`up_proj`/`down_proj`, the same leaf names as the language model, so an
   MLP-targeting adapter **silently trains the vision path we decided to freeze**.
5. **Three module families are named differently in the checkpoint than in the live module
   tree.** Most importantly the 288 routed experts are stored per-expert in the checkpoint but
   held as two **3-D `nn.Parameter`s** at runtime — they are not `nn.Linear`, so stock PEFT
   **cannot target them at all**. And `transformers` hard-codes
   `_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", …]`, i.e. **it discards the MTP head
   on load**.

Points 3–5 are the ones that would have cost a week if found empirically.

---

## 1. Parameter counts

Computed by summing `numel(shape)` over every tensor in all 62 safetensors headers of
`zai-org/GLM-5.3-Flash`, excluding the 37,338 quantization scale tensors
(`*.weight_scale_inv`). **All figures below are read, not estimated.**

### Total

| Quantity | Value | Status |
|---|---:|---|
| Total parameters | **321,323,031,390** (321.32 B) | **measured** — sum over 38,770 non-scale tensors; matches the HF API's own `safetensors.total` byte-for-byte |
| Tensors in checkpoint | 76,108 (38,770 params + 37,338 scales) | measured |
| Published claim ("320B total") | 320 B | model card — rounds our 321.32 B |
| Active parameters / token, text-only | **17,376,721,726** (17.38 B) | **derived** — dense backbone + 8/288 of routed experts + both embeddings |
| Published claim ("18B active") | 18 B | model card — see reconciliation below |

**Reconciling 17.38 B against the published 18 B.** Our computation switches the MTP head off
and the vision tower off. Adding the MTP head's per-token active share (0.386 B) gives 17.76 B;
adding the vision tower on top gives 18.33 B. Either convention rounds to "18B". *Which*
convention Z.ai used is **not stated anywhere I could find** — treat "18B active" as
±0.6 B rather than a hard number, and use **17.38 B** for a text-only FLOP model.

### By component

Every row sums exactly to the 321,323,031,390 total.

| Component | Tensors | Parameters | % of total | Where it lives |
|---|---:|---:|---:|---|
| **MoE routed experts** (42 sparse layers × 288) | 36,288 | **304,405,807,104** | **94.74%** | `…layers.{3..44}.mlp.experts.{0..287}.{gate,up,down}_proj` |
| **MTP head @ layer 45** (all of it) | 889 | **7,432,592,416** | **2.31%** | `…layers.45.*` — see §5 |
|  ├ its own 288-expert MoE | 864 | 7,247,757,312 | 2.26% | `…layers.45.mlp.experts.*` |
|  ├ its MLA attention + indexer | 14 | 124,914,432 | 0.04% | `…layers.45.self_attn.*` |
|  ├ `eh_proj` + `enorm`/`hnorm`/`shared_head.norm` + 2 layernorms | 6 | 33,574,912 | 0.01% | the four tensors unique to a nextn layer |
|  └ shared expert + router | 5 | 26,345,760 | 0.01% | `…layers.45.mlp.{shared_experts,gate}` |
| **KDA linear attention** (34 layers) | 510 | **4,682,897,792** | 1.46% | `…self_attn.{q,k,v,o}_proj`, `b_proj`, `f_a/f_b/g_a/g_b_proj`, 3 × `conv1d`, `A_log`, `dt_bias`, `o_norm` |
| **Sparse-MLA attention** (11 layers) | 77 | **1,291,868,160** | 0.40% | `…self_attn.{q_a,q_b,kv_a_proj_with_mqa,kv_b,o}_proj` + 2 layernorms |
| **MoE shared expert** (42 × 1) | 126 | 1,056,964,608 | 0.33% | `…mlp.shared_experts.{gate,up,down}_proj` |
| **Output embedding** `lm_head` | 1 | 634,388,480 | 0.20% | `lm_head.weight` — **untied** (`tie_word_embeddings: false`) |
| **Input embedding** `embed_tokens` | 1 | 634,388,480 | 0.20% | `model.language_model.embed_tokens.weight` |
| **Dense MLP** (layers 0–2, `first_k_dense_replace: 3`) | 9 | 452,984,832 | 0.14% | `…layers.{0,1,2}.mlp.{gate,up,down}_proj` |
| **Vision encoder** | 339 | **404,231,168** | 0.13% | `model.visual.patch_embed` + 24 × `blocks` + `post_layernorm` |
| **Vision-language projector** | 8 | **159,395,840** | 0.05% | `model.visual.downsample` + `model.visual.merger` |
| **Sparse-MLA indexer** (11 layers) | 77 | 82,190,592 | 0.03% | `…self_attn.indexer.{wk,wq_b,weights_proj,k_norm,…}` |
| **MoE router** (42 layers) | 84 | 49,557,312 | 0.02% | `…mlp.gate.weight` + `e_score_correction_bias` |
| **mHC hyper-connections** (45 layers) | 270 | 35,391,870 | 0.01% | `…hc_{attn,ffn}_{fn,base,scale}` |
| Layer norms (45 layers, in + post-attn) | 90 | 368,640 | 0.00% | |
| Final norm | 1 | 4,096 | 0.00% | |
| **TOTAL** | **38,770** | **321,323,031,390** | 100% | |

**Roll-ups a memory model actually wants** (all **derived** from the rows above):

| Roll-up | Parameters | BF16 bytes |
|---|---:|---:|
| Everything **except** routed experts and the MTP head | 9,484,631,870 | 17.67 GiB |
| Vision encoder + projector (the frozen half) | **563,627,008** | **1.05 GiB** |
| MTP head (layer 45), whole | 7,432,592,416 | 13.84 GiB |
| Routed experts only (42 backbone layers) | 304,405,807,104 | 567.00 GiB |
| Whole model, BF16 | 321,323,031,390 | **598.5 GiB** |

That last row is worth stating plainly against the map's Notes: **BF16 weights alone (642.7 GB)
exceed the entire 4-node Spark pool (~498 GB)** — confirmed, not assumed. And the per-node
figure for the NVFP4 base is **90.6 GiB/rank at TP2** (88.6 GiB with the MTP head excluded),
not the "~85 GiB/rank" our recipe comment claims; see §2's footnote.

### Architecture dimensions the counts rest on

Read from `config.json` (`text_config` unless noted).

| Key | Value | Note |
|---|---|---|
| `architectures` | `Glm5NextForConditionalGeneration` | confirms the fleet note |
| `model_type` | `glm5_next` / `glm5_next_text` / `glm5_next_vision` | |
| `num_hidden_layers` | **45** | so layer index 45 is *one beyond* the stack |
| `hidden_size` | 4096 | |
| `layer_types` | 34 × `linear_attention`, 11 × `deepseek_sparse_attention` | confirms 34 KDA + 11 MLA |
| `linear_attn_config.full_attn_layers` | `[3,7,11,15,19,23,27,31,35,39,43]` | every 4th layer |
| `linear_attn_config` | `num_heads: 64`, `head_dim: 128`, `short_conv_kernel_size: 4` | KDA qkv_dim = 8192 |
| `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts` | **288** / 8 / 1 | |
| `moe_intermediate_size` / `intermediate_size` | 2048 / 12288 | expert vs dense MLP |
| `mlp_layer_types` | 42 × `sparse`, 3 × `dense` | `first_k_dense_replace: 3` |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 | **why MLA has no `q_proj`** |
| `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` | 256 / **0** / 256 | NoPE MLA — confirmed |
| `mla_use_nope` | `true` | |
| `num_nextn_predict_layers` | **1** | the MTP head, see §5 |
| `vocab_size` / `max_position_embeddings` | 154,880 / 1,048,576 | |
| `mhc` | `true` | Manifold-Constrained Hyper-Connections |
| `vision_config` | `depth: 24`, `hidden_size: 1024`, `num_heads: 16`, `intermediate_size: 4096`, `patch_size: 14`, `image_size: 448`, `spatial_merge_size: 2`, `out_hidden_size: 4096`, `projection_intermediate_size: 10240`, `attention_bias: true` | |

**NoPE, proven from tensor shapes rather than the flag** *(derived)*:
`kv_a_proj_with_mqa.weight` is `(512, 4096)` and `kv_lora_rank` is 512 — the output width is
*exactly* the latent rank with **zero** extra rotary dimensions appended. On a RoPE-MLA model
this tensor is `(kv_lora_rank + qk_rope_head_dim, hidden)`. Likewise `q_b_proj` is
`(16384, 1536)` = 64 heads × 256 `qk_nope_head_dim`, with nothing added.

---

## 2. Checkpoint inventory

All sizes read from the HF tree API (`?recursive=1&expand=1`, paginated) on 2026-08-31.
"Tensor bytes" is my own sum over the safetensors headers and excludes the per-shard JSON
header overhead, which is why it is a few MB under the repo figure.

| Repo | Format | Shards | Repo bytes | GiB | Params | Vision tower | MTP head (layer 45) |
|---|---|---:|---:|---:|---:|---|---|
| **`zai-org/GLM-5.3-Flash`** | **FP8** block 128×128, `activation_scheme: dynamic` | 62 | 328,337,455,672 | **305.8** | 321.32 B | **BF16, entirely** | **present**; its MoE + MLA q/kv/o are FP8, `eh_proj` + all norms + `kv_b_proj` BF16 |
| **`zai-org/GLM-5.3-Flash-BF16`** | **BF16** (no `quantization_config`) | 120 | 642,652,070,880 | **598.5** | 321.32 B | BF16 | **present**, entirely BF16 |
| `unsloth/GLM-5.3-Flash` | BF16 mirror — safetensors total is **byte-identical** to the above (642,652,070,880) | 120 | 642,652,070,880 | 598.5 | 321.32 B | BF16 | present, BF16 |
| **`LibertAIDAI/GLM-5.3-Flash-NVFP4`** — *what we serve* | **NVFP4** (`modelopt` 0.45.0), group_size 16, **routed experts only** | 121 | 194,665,046,744 | **181.3** | 321.32 B | **BF16, entirely** (`model.visual.*` in `ignore`) | **present**; its 288 experts are NVFP4, everything else BF16 |
| `RedHatAI/GLM-5.3-Flash-NVFP4` | `compressed-tensors` **mixed-precision**: NVFP4 on `layers.3–44` experts, FP8 elsewhere | 11 | 197,843,812,476 | 184.3 | 188.1 B *(as the index's own `total_parameters` reports — packed)* | BF16 (every `model.visual.*` module listed in `ignore`) | present; layer-45 experts **excluded from the NVFP4 target regex** — kept at the FP8 group |
| `wtdcode/GLM-5.3-Flash-AWQ-W4A16` | AWQ W4A16, routed experts only | 13 | 190,810,457,040 | 177.7 | not read | BF16 (`re:.*visual\..*` ignored) | present; recipe ignores **`re:.*layers\.45\..*`** outright |
| `unsloth/GLM-5.3-Flash-FP8` | FP8 | — | not measured | — | — | — | — |

Also on the Hub, **not inspected** (listed for completeness, all community): a large GGUF
family (`unsloth`, `antirez`, `AesSedai`, …), MLX 2/4/6/8-bit (`pipenetwork`, `Vontra`,
`orcarouter`), EXL3 (`turboderp`, `davidsyoung`), MXFP4 (`OneNexus`), plus abliterated/uncensored
re-quants. None of these are training bases.

### Dtype by tensor class — `zai-org/GLM-5.3-Flash` (native FP8)

Read off the shard headers. This is the answer to "which tensor classes are actually FP8".

| Tensor class | dtype | Bytes in ckpt |
|---|---|---:|
| Routed experts `{gate,up,down}_proj.weight` (incl. layer 45's) | `F8_E4M3` + `F32` `weight_scale_inv` | **290.32 GiB** |
| **All 34 KDA layers'** `q/k/v/b/f_*/g_*/o_proj`, 3 × `conv1d`, `o_norm` | **`BF16`** (`A_log`, `dt_bias` are `F32`) | **8.72 GiB** |
| `embed_tokens` + `lm_head` | `BF16` | 2.36 GiB |
| MLA `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj` (12 layers) | `F8_E4M3` + `F32` scale | 1.13 GiB |
| **Whole vision tower + projector** | **`BF16`** | **1.05 GiB** |
| Shared experts `{gate,up,down}_proj` (43) | `F8_E4M3` + `F32` scale | 1.01 GiB |
| Dense MLP, layers 0–2 | `F8_E4M3` + `F32` scale | 0.42 GiB |
| MLA `kv_b_proj` (12) | **`BF16`** — no scale tensor exists | 0.38 GiB |
| Sparse-MLA indexer (`wk`, `wq_b`, `weights_proj`, `k_norm`, kpool) | `BF16` | 0.17 GiB |
| MoE router `gate.weight` + `e_score_correction_bias` | `BF16` + `F32` | 0.09 GiB |
| mHC `hc_*` | `BF16` + `F32` | 0.07 GiB |
| MTP-specific `eh_proj`/`enorm`/`hnorm`/`shared_head.norm` | **`BF16`** | 0.06 GiB |
| Layernorms + final norm | `BF16` | ~0 |
| **Total** | 314.40 B F8 + 6.93 B BF16 + 19.5 M F32 | **305.78 GiB** |

`quantization_config.modules_to_not_convert` is a 1,509-entry explicit list and it corroborates
every "BF16" row above, including `visual.*`, `model.visual`, `lm_head`, `model.layers.N.eh_proj`,
`model.layers.N.enorm`/`hnorm`/`shared_head.norm`, all 34 KDA projections, and `kv_b_proj`.

### Dtype by tensor class — `LibertAIDAI/GLM-5.3-Flash-NVFP4` (what we serve)

| Tensor class | dtype | Bytes |
|---|---|---:|
| Routed experts `{gate,up,down}_proj.weight` | **`U8`** (two FP4 values per byte) | 145.1 GiB |
| ↳ `weight_scale` (per group of 16) | `F8_E4M3` | 18.1 GiB |
| ↳ `weight_scale_2`, `input_scale` (per-tensor) | `F32` scalars | negligible |
| **Everything else** — all attention (both types), shared experts, dense MLPs, router, indexer, `lm_head`, `embed_tokens`, `eh_proj`, **the whole vision tower** | **`BF16`** | 18.0 GiB |
| **Total** | 155.83 B U8 + 19.48 B F8 + 9.67 B BF16 | **181.28 GiB** |

The `ignore` list confirms it explicitly: `model.visual.*`, `visual.*`, `*.visual.*`, `lm_head`,
`*.embed_tokens`, every attention projection by name, `*.mlp.shared_experts.*`, `*.mlp.gate`,
and `*.eh_proj`. **Only the routed experts are quantized.**

> **Correction to `recipes/dgxrun/glm-5.3-flash-libertai-nvfp4-2x.yaml`** *(derived, low stakes)*:
> the header says "the weights are ~182 GB, i.e. ~85 GiB per rank at TP2". 194.67 **GB** =
> **181.28 GiB** — so the "182" is a GiB figure labelled GB, and per rank at TP2 it is
> **90.6 GiB** of weight bytes (88.6 GiB if the MTP head is not loaded), not 85. The recipe's
> own measured "weights + non-torch 95.87–96.34 GiB" per rank is consistent with 90.6 + overhead.
> This does not change any conclusion in that recipe; it just makes the arithmetic close.

**Two structural facts hold across every safetensors checkpoint above:** the tensor *names* are
identical (the NVFP4 and BF16 indices normalise to exactly the same module tree as the FP8
base, differing only in which scale tensors accompany a weight), and **no quantizer touches the
vision tower** — four independent parties (Z.ai's own FP8, LibertAI's modelopt NVFP4, Red Hat's
compressed-tensors, wtdcode's AWQ) all put `visual.*` in the exclusion list. Vision stays BF16
everywhere.

---

## 3. Tensor naming, and the fused-vs-split verdict

This is the section [#47](https://github.com/kreuzhofer/dgx-manager/issues/47) exists to make us
write down.

### Verdict

| Where | Attention QKV | Attention output | MLP |
|---|---|---|---|
| **LM, 34 KDA layers** | **SPLIT** — `q_proj`, `k_proj`, `v_proj` | `o_proj` | (MoE, see below) |
| **LM, 11 sparse-MLA layers** | **NEITHER** — LoRA-factorised: `q_a_proj`→`q_b_proj`, and `kv_a_proj_with_mqa` (itself a fused KV-latent projection) → `kv_b_proj` | `o_proj` | (MoE) |
| **LM, MTP layer 45** | same as sparse-MLA | `o_proj` | (MoE) |
| **Vision tower, 24 blocks** | **FUSED** — `attn.qkv` (weight *and* bias) | `attn.proj` | **SPLIT** — `mlp.{gate,up,down}_proj` (+ biases) |
| **VL projector** | n/a | `merger.proj` | **SPLIT** — `merger.{gate,up,down}_proj` |

So: **the Qwen3.8 fused-`qkv` trap is present in the vision tower, identically.** A PEFT
`target_modules=["q_proj","k_proj","v_proj","o_proj"]` cannot see the vision tower at all.
Given the map froze the vision tower, that is a *convenience* here rather than a defect — but
it must be asserted, not assumed, and the next table shows why the assumption breaks anyway.

### Leaf-name collision table

Every leaf module name in the checkpoint, and where it occurs. This is what a PEFT suffix
matcher actually sees.

| Leaf name | LM backbone | MTP (layer 45) | Vision | Collision |
|---|---:|---:|---:|---|
| `q_proj`, `k_proj`, `v_proj` | 34 each | — | — | LM-only, **KDA layers only** |
| `o_proj` | 45 | 1 | — | LM-only (all 45 layers) |
| `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj` | 11 each | 1 each | — | LM-only, MLA layers only |
| `wk`, `wq_b`, `weights_proj` (indexer) | 11 each | 1 each | — | LM-only |
| `b_proj`, `f_a_proj`, `f_b_proj`, `g_a_proj`, `g_b_proj` | 34 each | — | — | KDA-only |
| **`gate_proj`, `up_proj`, `down_proj`** | **12,141 each** (12,096 routed experts + 42 shared + 3 dense) | 289 each | **49 each** (24 weights + 24 biases + `merger`) | **⚠ LM *and* vision *and* projector** |
| **`k_norm`** | 22 (indexer, w + b) | 2 | **24** (vision attn) | **⚠ LM *and* vision** |
| `qkv` | — | — | 48 (24 w + 24 b) | vision-only |
| `proj` | — | — | 51 | vision-only (`attn.proj`, `merger.proj`, `patch_embed.proj`) |
| `q_norm`, `norm1`, `norm2`, `post_layernorm`, `downsample`, `post_projection_norm` | — | — | 1–24 | vision-only |
| `gate` (MoE router) | 42 | 1 | — | LM-only |
| `eh_proj`, `enorm`, `hnorm`, `shared_head.norm` | — | 1 each | — | **MTP-only** |

### What common `target_modules` lists actually hit

Counting only 2-D weights (i.e. `nn.Linear`-shaped tensors) in the checkpoint. LoRA parameter
counts at **r = 16**, both A and B.

| `target_modules` | Modules matched | LoRA params (r=16) | Verdict |
|---|---:|---:|---|
| `["q_proj","k_proj","v_proj","o_proj"]` | **148** (147 LM + 1 MTP) | 30.7 M | **Under-reaches.** Covers QKV for only the 34 KDA layers; the 11 sparse-MLA layers contribute their `o_proj` and nothing else. Vision untouched (correct, by luck). |
| `+ ["q_a_proj","q_b_proj","kv_a_proj_with_mqa","kv_b_proj"]` | 196 | 42.5 M | **The correct "all LM attention" list.** All 45 layers + MTP. Vision still untouched. |
| `["gate_proj","up_proj","down_proj"]` | **37,365** | **3.67 B** | **Catastrophic over-reach.** 36,288 routed-expert + 867 MTP + 135 shared/dense + **75 vision** modules — it trains the tower the map froze. (And the 36,288 are not wrappable anyway; see below.) |
| `["q_proj",…,"gate_proj","up_proj","down_proj"]` (the "all-linear" habit) | 37,513 | 3.70 B | same, plus attention |
| `["qkv","proj"]` | 49 | 2.5 M | the *only* way to reach the vision tower — nothing else does |

**The actionable rule:** for a language-only PEFT tune of GLM-5.3-Flash, target
`["q_proj","k_proj","v_proj","o_proj","q_a_proj","q_b_proj","kv_a_proj_with_mqa","kv_b_proj"]`
and, if MLP adaptation is wanted, reach the shared experts by *full path* regex
(`re:.*mlp\.shared_experts\.(gate|up|down)_proj$`) rather than by leaf name. Never target
`gate_proj`/`up_proj`/`down_proj` by suffix.

### Checkpoint name ≠ runtime module path — three divergences

Read from `modeling_glm5_next.py`. This is the part that cannot be discovered from the tensor
index alone, and it is where the real hazard lives, because **PEFT matches against the live
module tree, not the checkpoint.**

| Checkpoint tensor name | Live `transformers` module path | Consequence |
|---|---|---|
| `…mlp.experts.{0..287}.{gate,up,down}_proj.weight` | `…mlp.experts.gate_up_proj` and `…mlp.experts.down_proj`, **two 3-D `nn.Parameter`s** of shape `[288, 4096, 4096]` and `[288, 4096, 2048]` inside `Glm5NextTextExperts` | **Stock PEFT cannot target the routed experts.** LoRA wraps `nn.Linear`/`nn.Embedding`/`Conv1D`; a bare 3-D `nn.Parameter` is not wrappable. Also note `gate` and `up` are **fused** in the live parameter even though the checkpoint splits them. |
| `…self_attn.{f_a_proj,f_b_proj,dt_bias}` | `…self_attn.forget_gate.{f_a_proj,f_b_proj,dt_bias}` (class `Glm5NextTextForgetGate`) | an extra path segment; leaf names survive, full-path regexes do not. Red Hat's own `ignore` list uses the `forget_gate.` form — independent corroboration. |
| `…self_attn.{q,k,v}_conv1d.weight` (three tensors) | `…self_attn.conv1d`, **one** depthwise `nn.Conv1d` over `3 × 8192` channels | three checkpoint tensors, one live module |

Add to this that vLLM fuses further still at load time: the FP8 config's
`modules_to_not_convert` and the NVFP4 `ignore` list both name `self_attn.qkv_proj`,
`self_attn.fused_qkvbfg_a_proj`, `self_attn.fused_qkv_a_proj_with_mqa`, `self_attn.qkv_conv1d`
and `mlp.gate_up_proj` — **names that appear in no checkpoint index**. They are the serving
engine's fused forms, listed defensively by the quantizer authors. *(inferred: I did not read
vLLM's `glm5_next` loader; the names' absence from all three indices is measured, their being
vLLM-side fusions is the only reading that fits.)*

---

## 4. The vision tower and the projector

| | Module path | Params | Shape facts |
|---|---|---:|---|
| Patch embedding | `model.visual.patch_embed.proj` | 1,205,248 | `Conv3d` `(1024, 3, 2, 14, 14)` — `temporal_patch_size: 2`, `patch_size: 14` |
| 24 encoder blocks | `model.visual.blocks.{0..23}` | 403,024,896 | per block ≈ 16.79 M: `attn.qkv (3072,1024)+bias`, `attn.proj (1024,1024)+bias`, `attn.{q,k}_norm (64,)`, `mlp.{gate,up}_proj (4096,1024)+bias`, `mlp.down_proj (1024,4096)+bias`, `norm1`, `norm2` |
| Post-norm | `model.visual.post_layernorm` | 1,024 | |
| **Encoder subtotal** | | **404,231,168** | |
| Spatial merge | `model.visual.downsample` | 16,781,312 | `Conv2d (4096, 1024, 2, 2)` + bias — `spatial_merge_size: 2`, and this is where 1024 → 4096 |
| Projector MLP | `model.visual.merger.{proj, gate_proj, up_proj, down_proj, post_projection_norm}` | 142,614,528 | `proj (4096,4096)`, `gate_proj`/`up_proj (10240,4096)`, `down_proj (4096,10240)` — class `Glm5NextVisionPatchMerger` |
| **Projector subtotal** | | **159,395,840** | |
| **Vision total** | | **563,627,008** | **0.175% of the model; 1.05 GiB at BF16** |

**"Frozen" can now be asserted rather than assumed**, on three independent grounds:

1. The tower is **0.175%** of the parameters. Freezing it costs nothing and gains everything.
2. Its attention is `attn.qkv` — a standard `q_proj/k_proj/v_proj/o_proj` adapter *cannot*
   reach it even by accident.
3. **But** its MLP and the projector use `gate_proj`/`up_proj`/`down_proj`, which *any*
   MLP-targeting adapter reaches by accident. So freezing must be *enforced* — either by
   excluding `model.visual.` by full path, or (cleanly) by
   `model.model.visual.requires_grad_(False)` — and then **verified** by printing the matched
   module list, not trusted.

Note also that the tower is BF16 in **every** checkpoint, including the NVFP4 one we serve. A
QLoRA-style run over the NVFP4 base therefore has a full-precision vision path already; nothing
needs dequantizing to keep image features exact.

---

## 5. The MTP head

**Confirmed: it sits at layer index 45, one beyond `num_hidden_layers: 45`.** Five independent
lines of evidence, four of them primary:

1. `config.json` → `num_nextn_predict_layers: 1`. **measured**
2. The tensor index contains layer indices **0..45 inclusive** — 46 layers where the config
   declares 45. **measured**
3. Layer 45 carries four tensors that exist **nowhere else in the model**:
   `eh_proj`, `enorm`, `hnorm`, `shared_head.norm` — the canonical DeepSeek-V3 nextn signature.
   `eh_proj` is `(4096, 8192)`: it projects `concat(hidden, embedding)` back to `hidden`.
   **measured**
4. `transformers` hard-codes `_keys_to_ignore_on_load_unexpected = [r"layers\.45\.",
   r"layers\.\d+\.shared_head\."]` in `Glm5NextPreTrainedModel` — **the literal string `45`**.
   **measured, first-party**
5. Two independent third-party quantizers treat layer 45 as special: wtdcode's AWQ recipe
   ignores `re:.*layers\.45\..*` outright, and Red Hat's NVFP4 target regex is
   `layers\.(?:[3-9]|[1-3][0-9]|4[0-4])` — deliberately stopping at 44. A community re-quant is
   even named `…-mtp-l45`. **measured**

**Grepping the tensor names for `nextn`, `mtp`, `draft` or `speculat` returns ZERO hits** —
confirmed on all three indices (`zai-org` FP8, `zai-org` BF16, `LibertAI` NVFP4). GLM names the
head positionally and nothing else. Our recipe's warning about this is correct and worth
keeping.

### What the MTP head actually contains

`model.language_model.layers.45.*` — 889 tensors, **7,432,592,416 parameters**. It is a *full
decoder layer*, not a small head:

| Tensors | Params | |
|---|---:|---|
| `mlp.experts.{0..287}.{gate,up,down}_proj` | 7,247,757,312 | its **own** 288-expert MoE — 97.5% of the head |
| `self_attn.{q_a,q_b,kv_a_proj_with_mqa,kv_b,o}_proj`, `q_a_layernorm`, `kv_a_layernorm`, `indexer.*` | 124,914,432 | a sparse-MLA layer, complete with indexer |
| `eh_proj` `(4096, 8192)`, `enorm` `(4096,)`, `hnorm` `(4096,)`, `shared_head.norm` `(4096,)`, `input_layernorm`, `post_attention_layernorm` | 33,574,912 | the nextn-specific parts |
| `mlp.shared_experts.{gate,up,down}_proj`, `mlp.gate.weight`, `mlp.gate.e_score_correction_bias` | 26,345,760 | |

**It has no `hc_*` tensors** (those stop at 45 entries, i.e. layers 0–44) and **no output head
of its own** — only `shared_head.norm`, so the vocabulary projection is shared with the main
`lm_head`. *(inferred, high confidence: the absence of a `shared_head.head.weight` tensor is
measured; that this implies tying to `lm_head` is the standard DeepSeek-MTP arrangement and the
only structure the remaining tensors permit.)*

Active cost when speculating is small — 0.386 B/token, because only 8 of its 288 experts fire.

### The finding that matters for this map

**`transformers` does not load the MTP head.** `_keys_to_ignore_on_load_unexpected` silences
every `layers.45.*` key, and `Glm5NextTextModel` builds exactly
`range(config.num_hidden_layers)` = 45 decoder layers. So:

- a PEFT/`transformers` fine-tune sees a **45-layer** model;
- `save_pretrained` after such a run emits a checkpoint with **no layer 45**;
- the 1.57–1.82× decode speedup our recipe measured from native MTP is **not carried by the
  training path** — it has to be re-attached from the original checkpoint, or lost.

That is squarely the map's open question *"whether the MTP head survives a tune"*. This ticket
does not decide it — but the answer is no longer "unknown": on the `transformers` path the head
is not merely at risk, it is **absent from the very first `from_pretrained` call**. Re-attaching
it means copying the 889 weight tensors (1,760 files-worth including FP8 scales; 3,481 in the
NVFP4 checkpoint) from the base into the output — mechanically easy, but it means the head is
**never updated** to match the tuned backbone, and whether a stale drafter still achieves 45%
acceptance against a shifted target is a measurement nobody here has made.

---

## 6. Confidence

| Claim | Status |
|---|---|
| 321,323,031,390 total params; every per-component row; every shape and dtype | **Measured** — summed over the safetensors headers of all 62 shards (and all 121 NVFP4 shards), fetched by range request 2026-08-31. Cross-checks: my byte total equals the index's `metadata.total_size` exactly for all three repos, and my "BF16 if dequantized" figure (642,646,653,816 B) equals the independently-published `zai-org/GLM-5.3-Flash-BF16` index total to the byte. |
| Fused-vs-split, for LM and vision; the leaf-name collision table | **Measured** — read off the tensor index keys, and independently confirmed in `modeling_glm5_next.py` (`Glm5NextVisionAttention.qkv`, `Glm5NextTextLinearAttention.{q,k,v}_proj`, `Glm5NextTextAttention.{q_a,q_b}_proj`) |
| Routed experts are 3-D `nn.Parameter`s, not `nn.Linear`; `forget_gate.` and fused `conv1d` paths | **Measured** — read from `modeling_glm5_next.py` on `transformers` `main`. The *consequence* ("stock PEFT cannot target them") is **derived** from how `peft.tuners.lora` dispatches on module type; I did not run it. |
| `transformers` drops `layers.45.*` on load | **Measured** — the literal regex in `Glm5NextPreTrainedModel._keys_to_ignore_on_load_unexpected` |
| Layer 45 is the MTP head | **Measured**, five ways (§5) |
| Active parameters = 17.38 B | **Derived** — dense backbone + 8/288 of routed experts + both embeddings. The published "18B" uses an unstated convention; ±0.6 B. |
| Vision tower is BF16 in every checkpoint | **Measured** for `zai-org` FP8, `zai-org` BF16 and `LibertAI` NVFP4 (header dtypes). **Read from the quantizer recipe, not the headers**, for RedHatAI and wtdcode. |
| Whether the model *trains* at all under `transformers`/PEFT/DeepSpeed | **Not investigated** — that is the framework-support ticket, not this one. `supports_gradient_checkpointing = True` and `_no_split_modules = ["Glm5NextTextDecoderLayer", "Glm5NextVisionBlock"]` are present, which is necessary but far from sufficient. |
| Whether a stale MTP head re-attached after a tune still drafts usefully | **Unknown, and unmeasurable from a checkpoint.** Needs a run. |

### What this changes downstream

- **Memory tickets** should use 598.5 GiB (BF16) / 305.8 GiB (FP8) / 181.3 GiB (NVFP4) and the
  0.175% vision share, not the "~640 GB" and "~320B" round numbers.
- **Method tickets** must treat "expert-subset tuning" and "adapter-on-router" as **blocked on a
  custom module wrapper**, not as available PEFT configurations — the experts are not
  `nn.Linear` in `transformers`.
- **Servability tickets** inherit a concrete, sharpened risk: the MTP head is dropped at load,
  so the artifact will not serve at 22–26 tok/s unless something re-attaches it.

---

## Sources

Everything below was fetched directly on **2026-08-31**.

- `https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json` — all architecture dims, the 1,509-entry `modules_to_not_convert`
- `https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/model.safetensors.index.json` (8.4 MB, 76,108 keys) and the **safetensors headers of all 62 shards**, by HTTP `Range` request against `/resolve/main/model-000NN-of-00062.safetensors`
- `https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/raw/main/config.json` and its index + **all 121 shard headers**
- `https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/raw/main/config.json` + index (identical to the FP8 config minus `quantization_config`; identical tensor names minus the scales)
- `https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4/raw/main/{config.json,recipe.yaml}` and index metadata
- `https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16/raw/main/recipe.yaml`
- `https://huggingface.co/api/models/{repo}` and `.../tree/main?recursive=1&expand=1` (paginated via the `Link` header) — file sizes, dtype parameter counts, timestamps
- `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/glm5_next/{modeling,configuration}_glm5_next.py` — `Glm5NextTextExperts`, `Glm5NextTextLinearAttention`, `Glm5NextTextAttention`, `Glm5NextVisionAttention`, `Glm5NextVisionPatchMerger`, `Glm5NextPreTrainedModel`
- `zai-org/GLM-5.3-Flash` model card — the "320B total / 18B active" claim, the hybrid-attention and mHC description
- `recipes/dgxrun/glm-5.3-flash-libertai-nvfp4-2x.yaml` (branch `qwen3.8-27b-and-staging-design`) — our measured serving figures: 320K window, MTP at 22.4–26.0 tok/s, 45.4% draft acceptance, `weights + non-torch 95.87–96.34 GiB`/rank
- Memory `glm53-flash-is-a-different-model` — the fleet facts this ticket was asked to verify. **All of them hold**: `glm5_next` / `Glm5NextForConditionalGeneration`, 45 layers, hidden 4096, 288 experts, 34 KDA + 11 sparse-MLA, NoPE MLA (`qk_rope_head_dim: 0`), natively multimodal, NVFP4 ≈ 182 (GiB, not GB), MTP at layer 45, zero `nextn`/`mtp`/`draft` grep hits.
