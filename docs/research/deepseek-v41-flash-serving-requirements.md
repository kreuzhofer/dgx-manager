# Serving DeepSeek-V4.1-Flash on the GB10 Spark cluster

**Research date:** 2026-09-14 · **Asked as:** *"What would it take to serve DeepSeek V4.1 Flash, and
what changes relative to DeepSeek-V4-Flash-0731?"* · with the mid-task correction *"treat it as a
heavily modified architecture, not a point release."*

**Method.** Primary sources only. Every weight number here was read from the **safetensors headers
themselves** — I range-fetched the 8-byte header length plus the JSON header of all 48 shards of
`deepseek-ai/DeepSeek-V4.1-Flash` and summed real `data_offsets`, so byte counts are measured, not
rounded claims from a card. The architecture comes from DeepSeek's own reference implementation
(`inference/model.py`, `convert.py`, `kernel.py`, `encoding/encoding.py` in the model repo), not from
prose. vLLM facts come from a local clone of `main` at `e85c8826ce2a81…` (2026-09-14 18:00 UTC) plus
the GitHub API for PR state and merge ancestry. Three third-party deployment records are used where
they are the *owner* of the thing asked about, and are attributed inline.

**Labelling.** Numbers that are my arithmetic rather than someone's measurement are marked
`[inference]`. Third-party measurements are attributed to their source. Gaps are listed in §7.

---

## BOTTOM LINE

**2026-09-14.** DeepSeek-V4.1-Flash **exists as open weights** — `deepseek-ai/DeepSeek-V4.1-Flash`,
MIT, published 2026-09-10 — and the correction you were given is right: **this is a new architecture,
not a point release.** New `architectures` string, new `model_type`, a nested text+vision config, a
**causal encoder-decoder** layout, a rebuilt sparse-attention scheme that shares one compressed KV
across groups of layers instead of one per layer, a **196 B-parameter Engram n-gram memory** absent
from V4, **3 DSpark draft stages with their own 128-expert MoE** instead of 1, dense FP8 scales at
**32×32 instead of 128×128**, and a changed prompt format. **Roughly two-thirds of our recipe
changes.** The dominant constraint is memory: 475.24 GiB of weights of which **189.13 GiB is
Engram**, and on GB10 the usual "offload it to host RAM" escape is worthless because host memory *is*
the GPU pool. **But the answer to "is it servable on four Sparks" is yes, and it is already being
done** — three independent parties have published working 4×GB10 recipes since 2026-09-10, with
measured numbers, and all three solve the same way: **keep the Engram tables on disk and never make
them resident.** The reference record loads **81.36 GiB per rank** at TP4 with DSpark and vision on
and serves at 73.8 tok/s single-stream on code. The cost is real: a **custom container built from a
pinned vLLM commit plus seven bind-mounted patch files**, a FlashInfer upgrade, ~48 GB of local disk
per worker for staged Engram rows, and **the whole fleet** for one model. **TP2 does not fit, either
way.** Verdict: **servable with about a week of work, at TP4 across all four Sparks, on a path we do
not currently run.**

| Claim / question | Verdict |
|---|---|
| **"DeepSeek V4.1 Flash" exists as open weights** | **Yes.** [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), created 2026-09-10T02:17:58Z, **MIT**. No dated `-0xxx` snapshot; no non-Flash `DeepSeek-V4.1` (404). (§1.1) |
| **Same architecture class as V4-Flash-0731** | **No.** `DeepseekV41ForCausalLM` / `deepseek_v41` vs `DeepseekV4ForCausalLM` / `deepseek_v4`; config shape went flat → nested. (§1.2) |
| **Multimodal** | **Yes — new.** 32-layer DeepSeek-ViT + 2-layer aligner; `image-text-to-text`. V4-Flash-0731 was text-only. (§1.2) |
| **Size** | **475.24 GiB** measured. 763.21 B logical params = 551.88 B backbone + **196.61 B Engram** + 14.23 B DSpark + 0.49 B vision. Card's "552B / 196B" reproduce exactly. (§1.3) |
| **Fits at TP2 on two Sparks** | **No.** 239.99 GiB/rank `[inference]`; 145.6 GiB/rank even with Engram entirely off-GPU. Tech2Wild states flatly: *"TP2 does not fit either way."* (§3.3) |
| **Fits at TP4 on four Sparks** | **Yes — only with Engram on disk.** 121.70 GiB/rank resident `[inference]` → **81.36 GiB/rank measured** with the disk patch, DSpark and vision on. (§3.3, §3.5) |
| **Is the Engram table TP-sharded or replicated?** | **Row-sharded**, confirmed in DeepSeek's `ParallelEngramEmbedding` and `convert.py`, and again in a practitioner's per-rank row ranges. Replication costs only 3.40 GiB/rank. (§3.2) |
| **`B12X_MLA_SPARSE` still valid** | **No.** Upstream vLLM has no such literal at all; b12x's V4.1 config uses `B12X_MLA_SPARSE_DSV41`; upstream uses `FLASHMLA_SPARSE_DSV41` / `FLASHINFER_MLA_SPARSE_DSV41`. (§2.3) |
| **`deepseek_v4` tokenizer / tool / reasoning parsers still valid** | **No, all three.** Distinct `deepseek_v41` for each, and the DSML tags literally changed (`<｜DSML｜tool_calls>` → `<｜DSML｜ calls>`, with a space). (§2.4) |
| **`dspark` still the right method** | **Method name only.** 3 stages not 1, its own 128-expert MoE, new sub-keys, draft in-checkpoint under `mtp.*`, and vLLM **hard-rejects `mtp`**. (§2.5) |
| **`--load-format instanttensor` still needed** | **No — actively harmful.** b12x recorded it failing: the 91.56 GiB Engram tensors cannot be staged. (§2.6) |
| **Supported in vLLM in-tree** | **Yes, `main` only** — #56228/#56208/#56214 merged 2026-09-10/11. **In no published release** (latest v0.29.0, 2026-09-09). Official recipe says `min_vllm_version: 0.30.0`, which does not exist. (§5.1) |
| **Works on sm_121 out of the box** | **No.** [#56461](https://github.com/vllm-project/vllm/issues/56461) — "cannot serve on SM120/SM121 (GB10)". Needs **7 patch files + FlashInfer v0.7.0rc1**. (§5.3) |
| **Official hardware support** | `h200 / gb200 / gb300 / mi350x`. sm_121 not listed; recipe states `vram_minimum_gb: 614`. (§5.2) |
| **Our container works** | **No.** No spark-arena registry has a V4.1 recipe; a new build from a **pinned vLLM commit** is required. (§6) |
| **Anyone running it on 4× GB10?** | **Yes — three independent parties**, vLLM ×2 and SGLang ×1, all since 2026-09-10, one in "LIVE production". (§5.5) |
| **Biggest blocker** | **The 189.13 GiB Engram table**, and the build discipline needed to get a working sm_121 engine. (§7) |

---

## §1 — Identity

### 1.1 It exists, and the name is unambiguous

Probed against the HF API directly:

```
deepseek-ai/DeepSeek-V4.1-Flash          -> 200   created 2026-09-10T02:17:58Z
deepseek-ai/DeepSeek-V4-Flash-0731       -> 200   created 2026-07-31T07:30:24Z   (what we run)
deepseek-ai/DeepSeek-V4-Flash            -> 200   created 2026-04-22T06:04:20Z
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp -> 200   created 2026-08-31T06:16:18Z
deepseek-ai/DeepSeek-V4.1                -> 401   (does not exist)
deepseek-ai/DeepSeek-V4.1-Flash-Base     -> 401   (does not exist)
```

The name-confusion traps you warned about mostly **do not** apply: there is no dated `-0xxx` snapshot
of V4.1 and no non-Flash open-weight `DeepSeek-V4.1`. Exactly one repo. The only adjacent model that
could be mistaken for it is `DeepSeek-V4-Flash-Vision-Exp` (2026-08-31), which is a **V4** checkpoint
registered separately by vLLM as `DeepseekV4ForConditionalGeneration`. License is **MIT**.

### 1.2 What it is, from `config.json`

[Raw config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json):

```json
"architectures": ["DeepseekV41ForCausalLM"],
"model_type": "deepseek_v41",
"quantization_config": {
  "quant_method": "fp8", "activation_scheme": "dynamic",
  "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"
},
"text_config":   { "model_type": "deepseek_v41_text",   ... },
"vision_config": { "model_type": "deepseek_v41_vision", ... }
```

| field | V4.1-Flash | V4-Flash-0731 |
|---|---|---|
| `architectures` | `DeepseekV41ForCausalLM` | `DeepseekV4ForCausalLM` |
| `model_type` | `deepseek_v41` (nested) | `deepseek_v4` (flat) |
| `num_hidden_layers` | **40** (20 encoder + 20 decoder) | 43 |
| `hidden_size` | **5120** | 4096 |
| `n_routed_experts` / `num_experts_per_tok` | **384** / 6 (+1 shared) | 256 / 6 (+1 shared) |
| `moe_intermediate_size` | **2304** | 2048 |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | **1280** / 1024 / 8 | 1024 / 1024 / 8 |
| `rms_norm_eps` | **1e-20** | 1e-6 |
| `compress_ratios` | **0,0, 2×18, 1×20, 0,0,0** | 0,0, alternating **4,128**, 0,0,0 |
| `kv_source_layer_ids` | **[2, 8, 14, 20]** (new key) | — (each layer compressed its own) |
| `index_source_layer_ids` | **[2,8,14,20,24,28,32,36]** (new key) | — |
| `index_n_heads` | **32** | 64 |
| `candidate_source_layer_id` / `_topk_blocks` / `_block_size` | **20 / 2048 / 8** (all new) | — |
| `engram_*` | **present** (`engram_layer_ids [1,14]`) | absent (`num_hash_layers: 3`, different mechanism) |
| `num_nextn_predict_layers` | **3** | 1 |
| `dspark_n_routed_experts` / `_num_experts_per_tok` | **128 / 3** (new) | — (draft reused the 256-expert MoE) |
| `weight_block_size` (dense FP8) | **[32, 32]** | [128, 128] |
| vision | **32 layers, dim 1024, patch 14, downsample 3** | none |
| `max_position_embeddings` / `vocab_size` | 1,048,576 / 129,280 | same |

### 1.3 Parameter count, decomposed from the actual tensors

The HF API's `safetensors.parameters.total = 763,205,315,794` is confusing on its own (it counts FP4
expert tensors as fp4 pairs and excludes the E8M0 scale planes). Decomposing the real headers:

| group | logical params | matches the card? |
|---|---:|---|
| backbone (attention, MoE, mHC, norms, embed, head) | **551.88 B** | card: "552B backbone" ✓ |
| **Engram tables** (`layers.{1,14}.engram.embed`) | **196.61 B** | card: "Engram … (196B parameters)" ✓ |
| DSpark draft (`mtp.0/1/2.*`) | 14.23 B | — |
| vision tower + aligner | 0.49 B | — |
| **total** | **763.21 B** | |

Active parameters, per the card: **8 B during prefill, 16 B during decode** — the asymmetry is the
whole point of the encoder-decoder split (§2.1).

---

## §2 — The delta

Per the correction, I established the architecture on its own terms first and then tested each line
of our recipe against it. The architecture genuinely is substantially new. On your three-way
classification, the honest answer is: **not "bump a flag", and not "needs upstream work that does not
exist" — it is squarely "needs a new container build", plus seven source patches that already exist
and are published.**

### 2.1 What actually changed, mechanically

**(a) Causal Encoder-Decoder (CED).** From the
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/README.md):

> a 40-layer Transformer organized as a 20-layer causal encoder followed by a 20-layer decoder. With
> CED, the decoder's global KV cache is projected from the final encoder hidden states rather than
> derived from each decoder layer's own hidden states.

Visible in the config: `compress_ratios` is `2` for layers 2–19 and `1` for 20–39, and
`kv_source_layer_ids` is `[2, 8, 14, 20]` — layer 20 is the boundary and the single source for all
twenty decoder layers.

**(b) CSA2 — compressed sparse attention with layer sharing.** V4-Flash-0731's index has
`attn.compressor.*` on **41 of 43 layers** and `attn.indexer.*` on **21**. V4.1 has **4 compressors
and 8 indexer query projections**, and the indexer no longer owns a compressor — it derives index
keys from the main compressor's latent through new `indexer.wk` / `indexer.k_norm` tensors that do
not exist in V4. The reference names the three resulting modes directly:

```python
self.owns_k = layer_id in args.kv_source_layers
self.is_candidate_source = layer_id == args.candidate_source_layer
self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
```

**(c) Hierarchical sparse indexer.** New. Layer 20 scores all compressed positions and publishes a
block mask (`select_candidate_blocks`, `candidate_topk_blocks: 2048`, `candidate_block_size: 8`);
layers 24–36 score only inside it. No analogue in V4.

**(d) Engram.** Two tables of ~384 M rows × 256 dims, FP8 with E8M0 scales, read by n-gram hash and
gated into the hyper-connection residual. 39.8 % of the checkpoint, and the single most important
fact in this document.

### 2.2 Verification that I read it right

Four independent cross-checks, all exact:

1. **KV cache size.** Deriving from the reference's cache shapes and `compress_ratios` gives
   **890 B/token**, against the card's "**890 bytes per token**" (§3.4).
2. **Engram parameter count.** Measured 196.61 B vs the card's "196B".
3. **Per-rank memory.** My TP4 arithmetic gives 121.70 GiB/rank, of which **47.21 GiB is Engram**;
   b12x's instrumented run reports `per_rank_pinned_bytes: 50,689,508,232` = **47.21 GiB**
   ([ram_engram.json](https://github.com/eugr/b12x/blob/master/validation/deepseek_v41/ram_engram.json)).
4. **Engram row ranges.** A practitioner's TP4 staging script uses
   `1:96000564:192001740` for rank 1 — i.e. rows ⌈384,006,168/4⌉ onward, exactly the ceil-shard in
   `convert.py`
   ([prepare-engram.sh](https://huggingface.co/0xTank/DeepSeek-V4.1-Flash-vLLM-4x-GB10-Recipe/raw/main/tools/prepare-engram.sh)).

### 2.3 `B12X_MLA_SPARSE` — **invalidated**

This was flagged as the load-bearing question; the answer is clean.

- **Upstream vLLM has no `B12X_MLA_SPARSE` literal at all.** Grepping the whole `main` tree returns
  nothing. The only b12x member of `AttentionBackendEnum` is
  `B12X = "vllm.v1.attention.backends.b12x.B12xPagedAttentionBackend"`
  ([registry.py L117](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py#L117)).
  So the value in our current recipe is a **fork-only literal** — worth knowing independently of V4.1.
- **b12x's own V4.1 engine config uses a new name**: `"attention_backend": "B12X_MLA_SPARSE_DSV41"`
  ([engine_tp4_ssd.json](https://github.com/eugr/b12x/blob/master/validation/deepseek_v41/engine_tp4_ssd.json)).
- Upstream's equivalents are new and V4.1-specific, with a comment that says exactly why:

  ```python
  # DeepSeek V4.1 sparse MLA backends (model-driven; selected via the V4.1
  # layer). Separate names from DSV4 so a V4.1 model never resolves the
  # V4.0 backend classes through this enum.
  FLASHMLA_SPARSE_DSV41 = ...
  FLASHINFER_MLA_SPARSE_DSV41 = ...
  ```
  ([registry.py L107-L116](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py#L107-L116))

The b12x README describes the new cache contract, and its numbers match the checkpoint: *"288-byte
main rows (E2M1 plus per-16 E4M3 scales)"* — precisely the record I derive in §3.4.

**The sparse-attention mechanism changed enough to invalidate the backend, yes — but the kernels
exist.** b12x ships `attention.compressed_sparse_mla`, `attention.mla_compress` "for the
nonoverlapping ratio-1/ratio-2 compressor", and an MXFP4 DSA recipe with "bounded candidate indices
for **hierarchical reindexing**" ([b12x README](https://github.com/eugr/b12x/blob/master/README.md)).

### 2.4 `deepseek_v4` tokenizer / tool / reasoning parsers — **all three invalidated**

vLLM `main` registers a distinct `deepseek_v41` for each, and **auto-selects the tokenizer**:

```python
elif arch == "DeepseekV41ForCausalLM":
    self.tokenizer_mode = "deepseek_v41"
```
([config/model.py L709-L710](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py#L709-L710))

- tokenizer: `"deepseek_v41": ("deepseek_v41", "DeepseekV41Tokenizer")` ([tokenizers/registry.py](https://github.com/vllm-project/vllm/blob/main/vllm/tokenizers/registry.py))
- reasoning: `"deepseek_v41": ("deepseek_v41_engine_reasoning_parser", "DeepSeekV41ParserReasoningAdapter")` ([reasoning/__init__.py](https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/__init__.py))
- tool: `"deepseek_v41": ("deepseekv41_engine_tool_parser", "DeepSeekV41EngineToolParser")` ([tool_parsers/__init__.py](https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/__init__.py))

**Why the V4 parser fails silently rather than erroring.** DeepSeek documents three prompt-format
changes ([encoding/README.md](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/encoding/README.md)),
and the rename is real in vLLM's source:

| | `deepseek_v4_encoding.py` | `deepseek_v41_encoding.py` |
|---|---|---|
| tool block | `tool_calls_block_name = "tool_calls"` | `tool_calls_block_name = " calls"` |
| invoke tag | `invoke` (literal) | `tool_call_tag_name = " invoke"` |
| parameter tag | `parameter` (literal) | `tool_parameter_tag_name = " parameter"` |

Note the **leading space**. The model emits `<｜DSML｜ calls>`; a V4 parser looks for
`<｜DSML｜tool_calls>`. No exception — tool calls simply arrive as prose.

There is no `deepseek_v4_1` (underscore) spelling anywhere. **NOT FOUND** — don't guess it.

Reasoning effort is now a **numeric budget 1–100** delivered via `chat_template_kwargs`, not V4's
verbose natural-language blocks. See the silent-failure table for a mapping discrepancy.

### 2.5 `dspark` — **method name survives, nothing else does**

`"dspark"` is still a first-class `SpeculativeMethod`
([config/speculative.py L72-L82](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py#L72-L82)),
and the draft still ships **inside the target checkpoint** under `mtp.*` — no separate draft repo to
stage. But: **3 stages** (`num_nextn_predict_layers: 3`); the draft has **its own 128-expert MoE**
(V4 reused the backbone's 256); new `markov_head.{embed,head}` and `confidence_head` tensors (V4 had
differently-shaped `markov_wN`); and vLLM **hard-rejects** `mtp` with a message that names the
tensors:

```
DeepSeek V4.1 has no classic-MTP draft: its checkpoints ship DSpark stages under mtp.*
(main_proj/markov_head/confidence_head) and carry no e_proj/h_proj/enorm/hnorm/hc_head weights.
Use speculative method 'dspark' instead of 'mtp'.
```
([speculative.py L1345-L1354](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py#L1345-L1354))

New sub-keys we do not set: `rejection_sample_method` (`standard|synthetic|block`) and
`enable_adaptive_verification` — the card's "confidence-scheduled verification", backed by
`confidence_head`, dspark-only.

**The practitioner consensus is to turn adaptive verification OFF on sm_121**, and the reason is
specific: *"it forces variable-length decode graphs with padded rows, the #5015 trigger"* — padded
speculative batches can hang SM120 sparse MLA
([RECIPE.md](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/docs/RECIPE.md)).
Separately, [#56771](https://github.com/vllm-project/vllm/issues/56771) reports illegal memory access
with DSpark on prompts ≳4 K on sm_120, and [#56797](https://github.com/vllm-project/vllm/issues/56797)
reports mean acceptance of only **2.82** at k=5. One 4×Spark record runs **k=3** deliberately; another
runs k=5 with `draft_sample_method: probabilistic` and `rejection_sample_method: block`.

### 2.6 `--load-format instanttensor` — **drop it**

`instanttensor` is genuinely upstream (in `LoadFormats`, maps to `DefaultModelLoader`), but for this
model b12x recorded it failing, structurally:

> `fastsafetensors` — "Sorting the live weight iterator retained GPU payloads; global Engram tensors
> are about **91.56 GiB each** and cannot be staged alongside the resident model."
> … `selected_loader: "safetensors"`, `safetensors_load_strategy: "lazy"`
> ([loader_selection.json](https://github.com/eugr/b12x/blob/master/validation/deepseek_v41/loader_selection.json))

"91.56 GiB each" is exactly my measured `layers.14.engram.embed.weight` = 91.557 GiB. Any loader that
materialises a whole tensor before sharding dies here. The 4×Spark recipes go further and **patch
`weight_utils.py` so the loader skips the Engram tables entirely** — "203 GB never read at load".
Our `instanttensor-hybrid-draft-loader` mod also targets a *separate* draft checkpoint, which V4.1
does not have.

### 2.7 The `VLLM_USE_B12X_*` toggles — superseded

Those seven env vars are not upstream vLLM variables; they belong to eugr's fork. b12x's V4.1 configs
do not set them — backends are selected structurally via
`"kernel_config": {"moe_backend": "b12x", "linear_backend": "b12x"}`. Two of our env settings are
**inverted** relative to every working V4.1 record: `VLLM_USE_BREAKABLE_CUDAGRAPH` must be `1` (we set
`0`), and `VLLM_USE_FLASHINFER_SAMPLER` must be `0` (we set `1`) — the latter "so the first request
does not JIT-compile one".

The kernel work that matters does exist, dated **2026-09-12**: `gemm.block_fp8_linear` "also accepts
V4.1's **32x32** E4M3/UE8M0 weight blocks with per-32 activation quantization" (§4.1), and
`sequence.engram` implements "V4.1's compressed-token, DEAD-bounded n-gram hashing and row-sharded
FP8/group-32 lookup. **It is not an alias for Qwen PLE.**"

### 2.8 Line-by-line verdict on our recipe

`recipes/dgxrun/deepseek-v4-flash-0731-2x.yaml`, every line. "Reference" below means
[Tech2Wild's `docs/RECIPE.md`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/docs/RECIPE.md),
the most complete 4×GB10 record.

| line in our recipe | verdict | replacement / note |
|---|---|---|
| model `…/DeepSeek-V4-Flash-0731` | **change** | `deepseek-ai/DeepSeek-V4.1-Flash` |
| TP2 / 2 Sparks | **change** | **TP4, all four Sparks.** TP2 does not fit (§3.3) |
| container `dgx-vllm-eugr-nightly-b12x:latest` | **change** | new build from a **pinned** vLLM commit + FlashInfer 0.7.0rc1 (§6) |
| mod `instanttensor-hybrid-draft-loader` | **remove** | no separate draft; loader fails on Engram (§2.6) |
| `--load-format instanttensor` | **change** | default loader + a `weight_utils.py` patch that skips Engram |
| `--moe-backend b12x` | **carries over** | b12x `kernel_config.moe_backend: "b12x"` |
| `--linear-backend b12x` | **carries over** | b12x `kernel_config.linear_backend: "b12x"` |
| `--attention-backend B12X_MLA_SPARSE` | **change** | `B12X_MLA_SPARSE_DSV41` (b12x) / `FLASHINFER_MLA_SPARSE_DSV41` (upstream) |
| `--tokenizer-mode deepseek_v4` | **change** | `deepseek_v41` — auto-selected, so the flag can be dropped |
| `--tool-call-parser deepseek_v4` | **change** | `deepseek_v41` + `--enable-auto-tool-choice`. Silent failure if left (§2.4) |
| `--reasoning-parser deepseek_v4` | **change** | `deepseek_v41` |
| `--reasoning-config '{…deepseek_v4…}'` | **remove** | effort is numeric 1–100 via `chat_template_kwargs` |
| `--kv-cache-dtype fp8` | **remove** | cache records are structural (528 B SWA / 288 B main), not a dtype knob |
| `--block-size 256` | **change → 128** | Reference: `--block-size 128` **required**; vLLM otherwise picks 64 and the V4 indexer backend refuses it at KV init |
| `--gpu-memory-utilization 0.85` | **change → 0.80** | Reference: target graphs 1.85 GiB, draft graphs 0.56 GiB, KV pool 4.84 GiB |
| `--max-model-len auto` | **change** | resolves to 1,048,576. Reference serves **300,000**; 1 M proven only in eager mode |
| `--max-num-seqs 8` | **carries over** | Reference uses 8 |
| `--max-num-batched-tokens 8192` | **carries over** | Reference uses 8192 (b12x used 1024; **512 segfaults** on sm_120, [#56837](https://github.com/vllm-project/vllm/issues/56837)) |
| `--enable-prefix-caching` | **carries over** | |
| `--trust-remote-code` | **remove** | vLLM ships `DeepseekV41Config` (§5.4) |
| `--max-cudagraph-capture-size 64` | **change** | Reference pins an explicit `cudagraph_capture_sizes` list so DSpark batches are never padded |
| `--compilation-config FULL_AND_PIECEWISE, custom_ops all` | **partly carries** | keep `FULL_AND_PIECEWISE`, but add the capture-size list; CUDA graphs are *the* throughput fix (eager decode is host-bound, ~200 ms/step) |
| `--speculative-config {…}` | **rewrite** | `{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}` |
| **new: `--enable-expert-parallel`** | **add** | set in every working V4.1 config |
| **new: `--engram-config '{"cpu_offload": false}'` + `DSV41_ENGRAM_DISK=1`** | **add** | the flag that makes TP4 fit (§3.5) |
| **new: `--limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1`** | **add** | multimodal; encoder costs ~0.22 GiB/rank |
| **new: `--default-chat-template-kwargs '{"thinking": false}'`** | **consider** | thinking off by default, on per request |
| env `CUTE_DSL_ARCH=sm_121a` | **carries over** | correct for GB10 |
| env `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | **invert → 1** | must be set on every node; the decorator binds at model import |
| env `VLLM_USE_FLASHINFER_SAMPLER=1` | **invert → 0** | native sampler, avoids a first-request JIT |
| env `VLLM_USE_B12X_*` (6 vars) | **remove** | superseded by `kernel_config` (§2.7) |
| env `VLLM_USE_AOT_COMPILE`, `_MEGA_AOT_ARTIFACT`, `_MEMORY_PROFILE_INCLUDE_ATTN` | **unknown** | not set in any V4.1 record I found |
| **new: `MAX_JOBS=2`, `FLASHINFER_NVCC_THREADS=1`** | **add** | so a stray runtime compile cannot OOM the host |
| **new: `VLLM_ENGINE_READY_TIMEOUT_S=3600`** | **add** | official `base_env`; load takes 10–18 min/node |

---

## §3 — Fit

### 3.1 Measured checkpoint composition

Summing real `data_offsets` across all 48 shards — **510,286,023,000 bytes = 475.24 GiB**, matching
`metadata.total_size` exactly:

| category | GiB | % | dtypes |
|---|---:|---:|---|
| MoE routed experts | 268.95 | 56.6 | FP4 (E2M1, packed in `I8`) + E8M0 |
| **Engram tables** | **189.13** | **39.8** | E4M3 + E8M0 |
| MTP / DSpark draft | 7.39 | 1.6 | mixed |
| attention (q/kv/o LoRA) | 4.72 | 1.0 | E4M3 + E8M0 |
| MoE shared expert | 1.32 | 0.3 | E4M3 |
| embed / lm head | 1.23 / 1.23 | 0.5 | BF16 |
| vision tower + aligner | 0.90 | 0.2 | BF16 |
| router, mHC, indexer, compressor, norms | 0.37 | 0.1 | F32/BF16 |
| **TOTAL** | **475.24** | | |

The two Engram tensors individually — and note they live in **shards 47 and 48**, which is what makes
the disk-streaming patches possible:

```
layers.1.engram.embed.weight    F8_E4M3  [384006168, 256]   91.554 GiB   (shard 47)
layers.1.engram.embed.scale     F8_E8M0  [384006168,   8]    2.861 GiB
layers.14.engram.embed.weight   F8_E4M3  [384016682, 256]   91.557 GiB   (shard 48)
layers.14.engram.embed.scale    F8_E8M0  [384016682,   8]    2.861 GiB
```

### 3.2 Sharded vs replicated — the question that bit us before

You asked specifically because a 51.2 B embedding table on another model turned out to be TP-sharded.
Here it is unambiguous, from DeepSeek's own code
([inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/model.py)):

```python
class ParallelEngramEmbedding(nn.Module):
    """The n-gram hash table, sharded over its rows. Stays fp8: rows are dequantized on lookup."""
    self.part_num_embeddings = (num_embeddings + world_size - 1) // world_size
    ...
        if world_size > 1:
            dist.all_reduce(values)
```

and `convert.py` does the same with ceil-division plus padding:

```python
elif ".engram.embed." in name:
    shard_size = (param.size(0) + mp - 1) // mp
    new_param = param[i * shard_size : (i + 1) * shard_size].contiguous()
```

**Sharded**: `embed`(0), `wq_b`(0), `wo_a`(0), `wo_b`(1), `head`(0), `attn_sink`(0),
`weights_proj`(0), `engram.embed`(rows), and routed experts by **expert parallelism** (whole experts
per rank). **Replicated on every rank** — everything else, totalling only **3.40 GiB**:

| replicated | GiB |
|---|---:|
| MoE shared expert | 1.419 |
| vision tower + aligner | 0.904 |
| engram `wkv`/`q_weight`/`k_weight` | 0.293 |
| attention `wq_a` | 0.263 |
| mHC coefficients | 0.157 |
| MoE router + `bias_vl` | 0.150 |
| attention `wkv` | 0.105 |
| DSpark `main_proj` / confidence | 0.073 |
| KV compressor, indexer `wk`, norms | 0.035 |
| **total** | **3.400** |

**So replication is not the problem this time** — it costs 3.4 GiB/rank. The problem is sheer size.

Hard ceiling worth recording: **TP cannot exceed 8**, because `o_groups = 8` and
`n_local_groups = o_groups // world_size` must be ≥ 1. Divisibility also needs
`n_routed_experts % TP == 0` (384) and `vocab_size % TP == 0` (129,280) — TP 2/4/8 all fine. TP3 is
**not** natively supported but one party built a TP3 path (virtual attention heads 64→72, a TP3 vocab
split, 3-rank Engram rows, and a fix to load the draft's 128 experts on three ranks).

### 3.3 Per-rank arithmetic

Applying `convert.py`'s rules to the measured bytes (including its dequantisation of `wo_a` from FP8
to BF16, ~+1.4 GiB):

| TP | sharded | expert-parallel | replicated | **per rank** | × TP |
|---:|---:|---:|---:|---:|---:|
| 1 | 197.51 | 275.67 | 3.40 | **476.58** | 476.58 |
| 2 | 98.76 | 137.83 | 3.40 | **239.99** | 479.98 |
| **4** | 49.38 | 68.92 | 3.40 | **121.70** | 486.78 |
| 8 | 24.69 | 34.46 | 3.40 | **62.55** | 500.39 |

`[inference]`. At TP4 the 121.70 GiB splits as **68.92 GiB experts + 47.21 GiB Engram + 5.57 GiB
everything else**.

**Against a Spark's usable memory.** Tech2Wild measures "128 GB unified memory each, about
**121.7 GiB visible to the OS**", and with `--gpu-memory-utilization 0.80` the practical weight budget
lands near 85–90 GiB.

| configuration | per rank | fits? |
|---|---:|---|
| TP2, native checkpoint, everything resident | 239.99 GiB `[inference]` | **No**, by 2.3× |
| TP2, native checkpoint, Engram entirely off-GPU | 145.57 GiB `[inference]` | **No**, by ~40 GiB |
| TP2, **EXL3 2.0 bpw** pack, Engram on NVMe | third-party claim only | claimed yes — unverified (§4.2) |
| TP4, everything resident | 121.70 GiB `[inference]` | **No** — leaves nothing for KV, graphs or activations |
| **TP4, Engram on disk, no DSpark** | **78.79 GiB measured** | **Yes** |
| **TP4, Engram on disk, DSpark + vision** | **81.36–81.58 GiB measured** | **Yes** |
| TP3 (3 Sparks), EXL3 experts, Engram on disk | 84.2 GiB measured | Yes (custom TP3 path) |
| TP8 | 62.55 GiB `[inference]` | fits, but we have 4 Sparks |
| RTX 5090 (32 GiB discrete) | 476.58 GiB | **No**, by ~15× |

The practitioner states the conclusion in one line: *"The routed experts (296 GB MXFP4) split four
ways fit. The two Engram n-gram tables (203 GB FP8) do not: stock vLLM puts them in host memory, and
**on GB10 host memory is the GPU pool**. … **TP2 does not fit either way.**"*

### 3.4 KV cache — a rounding error, and the card's claim verified

From the reference's cache declarations: main compressed KV is FP4 (E2M1) with one E4M3 scale per 16
channels over `head_dim = 512`; the indexer K is FP4 with one E8M0 scale per 32 channels over
`index_head_dim = 128`. Exactly four of each, at the `kv_source_layer_ids`:

| source layer | ratio | main B/entry | main B/token | indexer B/entry | indexer B/token |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 288 | 144 | 68 | 34 |
| 8 | 2 | 288 | 144 | 68 | 34 |
| 14 | 2 | 288 | 144 | 68 | 34 |
| 20 | 1 | 288 | 288 | 68 | 68 |
| | | | **720** | | **170** |

**890 B/token**, exactly the card's figure, and 288 B/entry matches b12x's "288-byte main rows"
independently. At 1,048,576 tokens that is **0.87 GiB per sequence** `[inference]`.

The sliding-window KV is **not** persistent — a 128-slot ring per layer, which is what "SWA Bounded
Replay" means. So 1 M context is cheap, and `--max-model-len` is not the lever it was on GLM-5.2.
Measured KV pools confirm the scale: **1,070,168 tokens in 4.84 GiB/rank** at 300 K context with
vision and tools on; a 1 M-context proof boot reached a **1,078,380-token** pool.

The real long-context cost is elsewhere: *"the indexer prefill buffer grows with max context, about
**5.2 GiB at 1M**, so 1M needs the extra memory from eager mode and gmu 0.80."*

### 3.5 Why "CPU offload" does not save us, and disk does

Upstream vLLM's answer to a huge n-gram table is pinned host memory, **and it is the default**:

```python
cpu_offload: bool = Field(default_factory=_default_cpu_offload)
"""Store embedding weights in pinned CPU memory for UVA lookup.
Defaults to VLLM_PLE_CPU_OFFLOAD, which is enabled by default."""
```
([vllm/config/engram.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/engram.py);
`VLLM_PLE_CPU_OFFLOAD: bool = True` at [envs.py L157](https://github.com/vllm-project/vllm/blob/main/vllm/envs.py))

**On GB10 this frees nothing.** CPU and GPU share one pool, so a 47.21 GiB table costs 47.21 GiB
either way — the same trap as our 126 GiB-download OOM. And because it is the default you get no
error, just a machine 47 GiB short.

The escape is to never make the table resident. Every stack that targets Spark-class hardware built
exactly that, independently:

- **Tech2Wild/0xTank (vLLM)**: `patch/engram.py` reads rows with `preadv` from shards 47/48 and
  dequantizes on CPU; `patch/model_state.py` stages rows in `prepare_inputs` **before the forward**,
  so the forward has no host round trip and stays CUDA-graph-capturable; `patch/weight_utils.py`
  makes the loader skip the tables entirely.
- **b12x**: `sequence._shared.disk_table` with "aligned **O_DIRECT** reads, block
  deduplication/coalescing", selected by `--engram-config '{"table_memory":"disk"}'`. **This key does
  not exist upstream** — grepping `main` for `table_memory` returns nothing.
- **SGLang lane (bertholomus)**: an NVMe Engram cache, `DSV41_CACHE_GIB=4` / `DSV41_CACHE_WAYS=16`,
  measured **74 % aggregate hit rate**. Notably *"12 GiB **rejected** — head host-RAM exhaustion ~90 s
  into weight load"*.
- **ds4**: "Engram rows are read directly from the file as needed in every mode, never loaded as a
  resident table."

**Practical consequence for us:** each worker needs the Engram rows for *its own rank* on **local**
disk — about **48 GB per worker** — not the whole checkpoint. The rest of the weights can come over
NFS read-only from one head node holding the 510 GB checkpoint; nothing else is copied. Load takes
~10 min on the head and 10–18 min on workers over NFS. Our `/mnt/tank` NFS mount is fine for the bulk
weights but **not** for the Engram rows.

---

## §4 — Quantization

### 4.1 The native checkpoint is already aggressively quantized

Unusually, DeepSeek shipped the *native* checkpoint pre-quantized, so the usual "find a smaller
community quant" move mostly does not apply. Measured from the headers:

| component | stored as | scale granularity |
|---|---|---|
| routed + DSpark experts | **FP4 (E2M1)**, two per `I8` byte | E8M0, one per **1×32** |
| dense weights | **FP8 E4M3** | E8M0, one per **32×32** |
| Engram tables | **FP8 E4M3** | E8M0, one per **1×32** |
| embed, lm head, vision, compressor, indexer `wk`, norms | BF16 | — |
| mHC coefficients, attention sinks, router bias | FP32 | — |

At 475.24 GiB for 763.21 B logical params, that is **0.67 bytes per parameter**.

**The V4→V4.1 granularity change is a genuine kernel-level break**, measured on both checkpoints:

| | V4-Flash-0731 | V4.1-Flash |
|---|---|---|
| dense FP8 block | `[128, 128]` | **`[32, 32]`** |
| FP4 expert group | 16 | **32** |
| scale format | UE8M0 | UE8M0 (unchanged) |

A 32×32 block scale is a 16× finer grid than the 128×128 every stock blockwise-FP8 GEMM assumes.

**Upstream vLLM does handle it — but not through the blockwise-FP8 path.** `vllm/models/deepseek_v41/quant_config.py`
defines `DeepseekV4FP8Config` (`quant_method: "deepseek_v4_fp8"`), whose `override_quantization_method`
claims `model_type in (deepseek_v4, deepseek_v41, …)` and then routes 32-wide `ue8m0` FP8 as **MXFP8**:

```python
if isinstance(layer, LinearBase) and self.weight_block_size == [32, 32] and self.is_scale_e8m0:
    return ModelOptLinearMethod(QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
                                CkptCtx(scale_block_size=(32, 32)))
```

with `expert_dtype == "fp4"` → `Mxfp4MoEMethod`. (The *generic* FP8 MoE oracle also accepts refined
blocks ≥ 32, but only for Triton kernels — `refine_fp8_moe_block_shape` rejects
FLASHINFER_CUTLASS/DEEPGEMM/CUTLASS automatically, because they "require the native 128x128 blocks".)

**Two consequences worth knowing.** First, because `wo_a_fp8_gemm_enabled` still requires `[128,128]`,
`attn.wo_a` is **dequantized to BF16 at load** — which is exactly what DeepSeek's own `convert.py`
does ("convert.py dequantizes it to bf16; an fp8 grouped GEMM would halve the memory"), and is the
+1.4 GiB in my §3.3 arithmetic. Second, getting this wrong does not raise: llama.cpp's still-draft
conversion PR [#28696](https://github.com/ggml-org/llama.cpp/pull/28696) says it plainly — *"V4
hardcodes 128, and **reusing that value rescales every dequantized weight without raising**."*

### 4.2 Derived checkpoints that exist (sizes measured via `?blobs=true`)

| repo | method | weights | note |
|---|---|---:|---|
| `deepseek-ai/DeepSeek-V4.1-Flash` | native FP8 + FP4 experts | **475.24 GiB** | the baseline |
| [`RedHatAI/DeepSeek-V4.1-Flash`](https://huggingface.co/RedHatAI/DeepSeek-V4.1-Flash) | re-upload | 475.3 GiB | no change |
| [`bot-lab-21/…EXL3-3.5bpw-Pollard`](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard) | EXL3 experts, rest FP8 | 428.4 GiB | **the default serving lane** in the Tech2Wild record since 2026-09-11 |
| [`Mia-AiLab/…EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) | EXL3 | 196.1 GiB | smaller; unverified on Spark |
| [`diffbot/…EXL3-2.0bpw-2x-RTX-PRO-6000`](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000) | EXL3 | 333.5 GiB | targets discrete SM120 |
| [`lvkaokao/…MXFP4-Engram-AutoRound`](https://huggingface.co/lvkaokao/DeepSeek-V4.1-Flash-MXFP4-Engram-AutoRound) | **Engram quantized to MXFP4** | 383.7 GiB | the only real "shrink the Engram" artifact |
| [`lvkaokao/…W4A16-Engram-AutoRound`](https://huggingface.co/lvkaokao/DeepSeek-V4.1-Flash-W4A16-Engram-AutoRound) | W4A16 | 420.6 GiB | larger than MXFP4 |
| [`caiovicentino1/…HLWQ-Engram-Q4`](https://huggingface.co/caiovicentino1/DeepSeek-V4.1-Flash-HLWQ-Engram-Q4) | Engram Q4 | 380.8 GiB | |
| [`GreenBitAI/…4bit-paged`](https://huggingface.co/GreenBitAI/DeepSeek-V4.1-Flash-4bit-paged) | 4-bit paged | 303.1 GiB | |
| [`LibertAIDAI/…REAP-256E`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E) | expert pruning 384→256 | **see trap below** | claims 385.5 GiB |
| [`antirez/deepseek-v4.1-flash-gguf`](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf) `Q2` | GGUF imatrix | 340.60 GiB | ds4 only, not vLLM |
| [`LibertAIDAI/…NVFP4`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4) | NVFP4 | 399.9 GiB | |
| [`s-zaizen/…NVFP4`](https://huggingface.co/s-zaizen/DeepSeek-V4.1-Flash-NVFP4) · [`AtomicChat/…NVFP4-nvidia`](https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia) | NVFP4 | **491.1 GiB** | **LARGER than the native FP8 checkpoint** |

**The "quant is bigger than FP8" trap you warned about is present here, exactly.** Three NVFP4 builds
(`s-zaizen`, `AtomicChat`, `Solstice-AI`) are **491.1 GiB against the native 475.24 GiB**. The
per-shard accounting shows why: the FP4 expert payload is unchanged, but expert scales moved from
`F8_E8M0` at block 32 (0.396 GiB/shard) to `F8_E4M3` at group 16 (0.942 GiB/shard) — NVFP4's
`group_size=16` costs **2× the scale bytes** — while the 189 GiB Engram is untouched. These are
scale-format conversions for kernel compatibility, **not** size reductions. Do not reach for NVFP4
on this model.

**Two of the smaller NVFP4 builds will not load at all**, per vLLM's own allowlist
`_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")`:

- `LibertAIDAI/…-NVFP4` declares `"expert_dtype": "nvfp4"` → `ValueError: Unsupported DeepSeek V4
  expert_dtype=…`. It also uses keys vLLM never reads (`expert_block_size`, `engram_dtype`,
  `expert_global_scale`).
- `msuiche/…-NVFP4` (386.4 GiB, the smallest *complete* safetensors build) keeps
  `weight_block_size:[32,32]` + `scale_fmt:"ue8m0"` in config, so vLLM selects the MXFP8 linear method
  expecting `.weight` E4M3 `[512,5120]` + `.scale` `[16,160]` — but the files hold `.weight` **U8
  `[512,2560]`** plus `weight_scale`/`weight_scale_2`. **The shapes do not match the declared format.**

The 491 GiB trio are, ironically, the internally *consistent* ones: they set `moe_quant_algo:"NVFP4"`,
the one key vLLM reads for the NVFP4 MoE path.

Additional single- and dual-Spark packs that do fit, both with their own runtimes rather than stock
vLLM — I did **not** verify either claim:

| repo | weights | claim |
|---|---:|---|
| [`sayyidfareed/…-Next-DGX-Spark-512K`](https://huggingface.co/sayyidfareed/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K) | 289.2 GiB | **one** 128 GB Spark; K154 expert pruning (154/384) + Engram on NVMe; claims 524,337-token request, 27.79 tok/s warm decode |
| [`sfxnz/DeepSeek-V4.1-Flash-EXL3`](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3) rev `2.0bpw-mcg` | 333.5 GiB | **two** Sparks TP2, Engram on NVMe. Card states: *"Native MXFP4/MXFP8 is about 511 GB and does not fit 2× Spark UMA."* |
| [`apetersson/…-MixedQ2-GGUF`](https://huggingface.co/apetersson/DeepSeek-V4.1-Flash-MixedQ2-GGUF) | 157.3 GiB | smallest artifact of any kind; GGUF |

So the nuance on §3.3's "TP2 does not fit": that is true of the **native** checkpoint, which is what
every working vLLM recipe uses. A 2.0 bpw EXL3 pack claims two Sparks. **GGUF is not usable upstream
yet** — llama.cpp's conversion PR #28696 is still a draft.

**Trap: `REAP-256E` does not contain a complete checkpoint.** Its README claims "475.2 GiB →
385.5 GiB … the Engram tables are untouched", and its index declares `total_size` 413,974,276,440
(= 385.5 GiB) with `layers.{1,14}.engram.embed.*` mapped to `model-00047-of-00048` and
`model-00048-of-00048`. **Those two shards are not in the repo** — only 46 of 48 are uploaded,
totalling 196.4 GiB. So the index resolves to files that do not exist. *(It is plausible one could
point a disk-Engram reader at the base repo's shards 47/48 and use REAP's other 46 — the shard
numbering lines up — but that is `[inference]`, untested, and nobody documents it.)*

Its README does contain one genuinely useful upstream fact: **vLLM's fused MoE router dispatches on a
fixed table of expert counts** — `{1,2,4,8,16,32,64,128,192,256,320,384,448,512,576}` — so a 272-expert
prune fails at startup with `Unsupported expert number: 272`.

### 4.3 sm_120 / sm_121 landmines for this model family

| issue | what it breaks | state |
|---|---|---|
| [#56461](https://github.com/vllm-project/vllm/issues/56461) | **GB10 sm_121 specifically.** SWA cache `block_size=32` vs SM120 decode page 64; ratio-1 indexer `block_kv=128` vs DeepGEMM sm120 (64 only). Fails in warmup. | open |
| [#56837](https://github.com/vllm-project/vllm/issues/56837) | sm_120: no FlashInfer kernel for `(num_heads=64, topk=1152)`; `FLASHMLA_SPARSE_DSV41` is SM90a/SM100f only. `--max-num-batched-tokens 512` segfaults deterministically. | open |
| [#56771](https://github.com/vllm-project/vllm/issues/56771) | DSpark + prompts ≳4 K → illegal memory access in SM120 sparse-MLA prefill. | open |
| FlashInfer #5015 (via Tech2Wild) | **Padded speculative batches can hang SM120 sparse MLA** — the reason adaptive verification stays off and capture sizes are pinned. | referenced |
| `persistent_topk` on GB10 (via Tech2Wild) | "**oversubscribes GB10's 48 SMs on long rows and kills the engine**"; the generic kernel matches `torch.topk` and is 1.6–3.6× faster. Patched out. | patched |
| [#56702](https://github.com/vllm-project/vllm/issues/56702) | `--language-model-only` ignored by the prefill SWA index width → topk 1152, unsatisfiable on sm_120. | closed |

The root cause under most of these is one table: FlashInfer's SM120 DSV4 decode dispatch instantiates
`topk ∈ {128, 192, 256, 512, 1024}` and V4.1 asks for **1152**. The fix in practice is **FlashInfer
v0.7.0rc1** — "FlashInfer 0.6.18's SM120 sparse-MLA decode lacks V4.1's topk of 1152." Note that 1152
is **not architecture**: it is `sliding_window(128) + vision_max_n_token(1024)`, which is why
text-only serving still widens SWA rows unless [#56623](https://github.com/vllm-project/vllm/pull/56623)
is applied. The upstream fixes ([#56509](https://github.com/vllm-project/vllm/pull/56509), #56623,
[flashinfer#5174](https://github.com/flashinfer-ai/flashinfer/pull/5174)) are **all still open** as of
2026-09-14 — which is why the working recipes bind-mount their own patches.

### 4.4 Checking our four specific historical landmines

| our prior scar | status for this model class |
|---|---|
| **Blockwise FP8 on sm_120 ([#51884](https://github.com/vllm-project/vllm/issues/51884))** | **Open, but effectively fixed.** `vllm/utils/deep_gemm.py` gates on `is_device_capability_family(120)`, which **includes GB10's sm_121**, and the thread contains a 2× GB10 reproduction on DeepSeek-V4-Flash-0731 at TP2. Fixed by [#52035](https://github.com/vllm-project/vllm/pull/52035) (merged 2026-08-12, DeepGEMM@8b1392b) — **present in v0.28.0+, absent from v0.27.1**. The issue is simply not closed. Root cause tracked separately in [#54125](https://github.com/vllm-project/vllm/issues/54125). |
| **NVFP4 KV failing silently on GB10** | **CONFIRMED, with GB10 measurements.** [#50084](https://github.com/vllm-project/vllm/issues/50084): NVFP4 KV writes V block scales in the SM100 trtllm-gen swizzle unconditionally, corrupting the SM120 FlashInfer XQA route. The fix PR [#55976](https://github.com/vllm-project/vllm/pull/55976) is **open, not merged**, and states it plainly — *"the cache is written in a layout its reader does not expect, and **V dequantizes to garbage**"* — with **GB10 / DGX Spark sm_121: 24 failed / 0 passed on main, 24 passed with the fix.** Use `fp8` KV, as both working 4×Spark recipes do. |
| **Marlin WNA16 MoE hang with Int8-mix experts** | **NOT FOUND upstream** for sm_121/GB10, in either vLLM or SGLang. Consistent with our own later finding that the original incident was earlyoom, not Marlin. What *is* confirmed on GB10: [#49546](https://github.com/vllm-project/vllm/issues/49546) — Marlin W4A8-FP8 **silently corrupts output** on sm_121a (repeated `</think>` loop at temperature 0, and ~2.5 % *faster*); and [#56064](https://github.com/vllm-project/vllm/issues/56064) — `moe_wna16_marlin_gemm` illegal memory access at M=256, clean at M≤128. |
| **AWQ/GPTQ larger than FP8** | **Present, as NVFP4** — see §4.2. |

Two more GB10 hazards that apply to our *current* V4-Flash deployment as much as to V4.1:

- [#40969](https://github.com/vllm-project/vllm/issues/40969) — DeepSeek-V4-Flash **hangs after ~6
  requests** with `cudagraph_mode=FULL_AND_PIECEWISE` + chunked prefill on SM 12.x (GB10). Open.
- [#56824](https://github.com/vllm-project/vllm/issues/56824) — GB10 unified memory: **29 GiB vanishes
  in 9 s**, and `MemAvailable` reports 22.5 GiB free when the driver already returns `NV_ERR_NO_MEMORY`.
  **No host-side signal can guard it.** This is the mechanism behind our own "a download OOM-killed a
  serving replica" memo.

---

## §5 — vLLM support

### 5.1 In-tree on `main`, in no release

| PR | what it added | merged |
|---|---|---|
| [#56228](https://github.com/vllm-project/vllm/pull/56228) | the model package + attention-backend enum entries | 2026-09-10 |
| [#56208](https://github.com/vllm-project/vllm/pull/56208) | tokenizer, renderer, reasoning + tool parsers, `DeepseekV41Config`, Rust frontend | 2026-09-10 |
| [#56214](https://github.com/vllm-project/vllm/pull/56214) | **the `registry.py` entry** — makes the arch resolvable — plus `vllm/config/engram.py` | 2026-09-11 |
| [#56741](https://github.com/vllm-project/vllm/pull/56741) | rename `deepseek_v4_1` → `deepseek_v41` (44 files) | 2026-09-14 |

Registered in `_MULTIMODAL_MODELS`, pointing at a **package** rather than the usual flat module —
`vllm/models/deepseek_v41/`, 27 files, ~12,300 lines. The draft is registered separately as
`DSparkV41DraftModel`. Tracking issue [#56400](https://github.com/vllm-project/vllm/issues/56400) is
open with a long tail of kernel, Engram, pipeline-parallel and ROCm work.

**No published release contains it.** Latest Release is **v0.29.0, 2026-09-09** — before every merge;
compare-API against `v0.29.0` gives `diverged / ahead_by 468–526` for all three SHAs. Tag `v0.29.1rc0`
contains it but has no Release object. It is also **entirely undocumented**: `grep -rn "deepseek_v41"
docs/` on `main` returns zero hits.

### 5.2 Official hardware support does not include us

From [`vllm-project/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml):

```yaml
  hardware: { h200: verified, gb200: verified, gb300: verified, mi350x: verified }
  min_vllm_version: "0.30.0"
variants: { default: { precision: fp8, vram_minimum_gb: 614 } }
```

`vram_minimum_gb: 614` against our ~487 GiB of fleet-visible memory — which is why the disk-Engram
patch is not optional.

### 5.3 Does not start on GB10 without patches

[#56461](https://github.com/vllm-project/vllm/issues/56461) is from a **DGX Spark, GB10, capability
(12, 1)** against the very commit that enabled the arch. The model takes the SM12x route by
construction (`device_capability.major == 12 → DeepseekV4FlashInferSM120Attention`) and dies in
`flashinfer_sparse_mla_warmup` because `attention.py` hardcodes `DeepseekV4SWACache(..., block_size=32)`
while FlashInfer's SM120 decode path only has `page_block_size == 64`.

**The published fix is seven bind-mounted whole-file patches** plus a four-stage image build:

| patch file | over | what it does |
|---|---|---|
| `engram.py` | `models/deepseek_v4_1/common/engram.py` | Engram on disk via `preadv` from shards 47/48; rank-offset fix; one shared read pool; `EngramDiskStager` |
| `model_state.py` | `models/deepseek_v4_1/nvidia/model_state.py` | stages rows in `prepare_inputs` **before** the forward, so it stays CUDA-graph-capturable |
| `weight_utils.py` | `model_executor/model_loader/weight_utils.py` | loader skips the two Engram tables (203 GB never read) |
| `attention.py` | `models/deepseek_v4_1/attention.py` | SM12x page sizes; indexer cache 64 states/page, because DeepGEMM's paged MQA logits only takes 32 or 64 |
| `flashinfer_sparse.py` | `models/deepseek_v4_1/nvidia/flashinfer_sparse.py` | 64-state pages + a 64-token SWA backend, the only page size FlashInfer's SM120 kernels are built for |
| `sparse_swa.py` | `v1/attention/backends/mla/sparse_swa.py` | the `get_swa_block_size()` hook |
| `sparse_attn_indexer.py` | `model_executor/layers/sparse_attn_indexer.py` | SM12x decode top-k; avoids `persistent_topk`, which kills the engine on GB10 |

### 5.4 Flags: required, absent, and surprising

- **`--trust-remote-code` is NOT needed.** vLLM ships `DeepseekV41Config`. Drop it.
- **`--enable-expert-parallel`** is set in every working config.
- **`--block-size 128` is required** on this path — vLLM otherwise picks 64 and the V4 indexer backend
  refuses it at KV init.
- **`--tokenizer-mode` can be omitted** (auto-selected).
- The official recipe notes **"DeepseekV41ForCausalLM does not support torch.compile"**, which is why
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1` matters.
- **CUDA graphs are the throughput fix, not an optimisation**: "eager decode on this model is
  host-bound (about 200 ms per step, GPUs nearly idle)."

### 5.5 Three independent 4×GB10 deployments already exist

| who | stack | config | measured |
|---|---|---|---|
| [tonyd2wild / Tech2Wild](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) (65★, created 2026-09-10) | **vLLM** TP4 + DSpark k=5 + CUDA graphs, vision + tools on, 300 K ctx, gmu 0.80 | 81.36 GiB/rank, KV pool 1,070,168 tok | single-stream decode **code 73.8**, JSON 52.1, math 50.9, prose 24.4 tok/s; 6 streams **131.9 tok/s** aggregate; cold prefill 902–1,539 tok/s; 7/7 vision + tool checks pass. Serving **7 hours after the model dropped**. |
| same, EXL3 lane (default since 2026-09-11) | vLLM TP4 on EXL3 3.5 bpw experts | 56.6 / 68.9 GiB/rank (uneven 512/640/640/512 split), KV pool **3,304,863 tok** | C6 aggregate 141.2 tok/s; 2026-09-14 speedrun: six-stream **189.97 tok/s**, cold prefill ~2,000 tok/s, TTFT 0.22 s, 1 M context proven |
| same, **TP3 lane on three Sparks** | vLLM TP3 (custom path: virtual heads 64→72, TP3 vocab split) | 84.2 GiB/rank, KV pool 678,950 tok | C6 aggregate **152.9 tok/s** — leaves one Spark free |
| [0xTank](https://huggingface.co/0xTank/DeepSeek-V4.1-Flash-vLLM-4x-GB10-Recipe) (builds on the above) | vLLM TP4 over RoCE, DSpark **k=3**, FULL_AND_PIECEWISE | **600,000-token** ceiling | 2026-09-11: prose decode **36.13 tok/s** cold / 36.67 warm, TTFT 0.385–0.733 s |
| [0xTank, FP8-R37 variant](https://huggingface.co/0xTank/DeepSeek-V4.1-Flash-FP8-R37-4x-GB10) | vLLM, original weights, 4× GB10 | 600 K context, DSpark k=3 | **50.50 tok/s** single-stream prose, **229.28 tok/s at 8 concurrent**, ~3,100–3,300 input tok/s prefill |
| [bertholomus](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-DSpark-TP4-4xGB10-Recipe) | **SGLang** TP4, 2-rail CX7, EP_SIZE 2, NVMe Engram cache | 1,048,576 tok/request, 8 M-token shared KV pool, FP8 KV | **"LIVE production"**; DSPARK_BLOCK_SIZE 3 gave +7–10 % at concurrency 2–4; Engram cache hit rate 74 % |

**Caveat on the SGLang lane: V4.1 is not merged into SGLang.** `python/sglang/srt/models/` has
`deepseek_v4.py` but **no `deepseek_v41.py`**, and `EntryClass = [DeepseekV4ForCausalLM]` only. PR
[#38798](https://github.com/sgl-project/sglang/pull/38798) is open with no CI run, and SGLang's own
cookbook says *"DeepSeek-V4.1 Flash support has **not shipped in an SGLang release yet**"* — there is
only a preview image, `lmsysorg/sglang:dev-dsv41`. Two open SGLang bugs are specifically on 4× DGX
Spark: [#39173](https://github.com/sgl-project/sglang/issues/39173) (the Engram profiled SPS table
dies in CUDA-graph capture; removing the table boots and serves) and
[#39226](https://github.com/sgl-project/sglang/issues/39226) (`--moe-runner-backend deep_gemm` is
accepted and then dies in graph capture — **the working flag is `--moe-runner-backend
flashinfer_mxfp4`**). SGLang also refuses `--enable-deterministic-inference` on this backend, and
states output is **not bitwise stable across batch composition**.

A fourth, non-vLLM path exists: [`antirez/ds4`](https://github.com/antirez/ds4) (DwarfStar) added V4.1
CUDA text support on 2026-09-13 and documents **two Sparks over RoCE at 21.9 tok/s** single session
with the Q2 GGUF (81 GiB/rank), but **no vision, no DSpark, and sessions served in order** on CUDA.
Its images are published for GB10 as `ghcr.io/spark-arena/dgx-ds4:<version>-sm121a-cu131` — though
`sparkrun` has **no `ds4` runtime** in `src/sparkrun/runtimes/` (I checked the full 720-entry tree),
so dgx-manager would need to bridge that itself.

---

## §6 — Container

**Our image will not work**, on three grounds:

1. `ghcr.io/spark-arena/dgx-vllm` is a **mirror** of eugr's image (latest commit "Mirror 2026091302",
   2026-09-13), and eugr's `spark-vllm-docker/recipes/` contains `deepseek-v4-flash-0731.yaml`,
   `deepseek-v4-flash-vision-exp.yaml` and `deepseek-v4-flash.yaml` — **no V4.1 recipe**.
2. **No spark-arena registry has a V4.1 recipe.** I walked the full git trees of
   `spark-arena/recipe-registry`, `community-recipe-registry` and `eugr-recipes`: every DeepSeek path
   is `deepseek4-flash` / `deepseek-v4-flash-*`. Zero V4.1 matches.
3. The reference build is **four stacked overlays**, none of which our image has:

| overlay | what | why |
|---|---|---|
| 1 | `vllm/vllm-openai:nightly-8a728663…` + `_C_stable_libtorch` rebuilt for **sm_121a** | the branch's kernel changes all live in that extension |
| 3 | **FlashInfer v0.7.0rc1** with pinned submodules; the stale 0.6.18 jit-cache/cubin packages removed | 0.6.18's SM120 sparse-MLA decode lacks V4.1's topk 1152 |
| 4 | prebuild `mxfp8_gemm_cutlass_sm120` with `MAX_JOBS=2` | its runtime compile "exhausted host memory on all four nodes at once" |
| 5 | rebuild `sparse_mla_sm120` under the exact runtime env; `verify5.py` checks nothing compiles at runtime | this is the serving image |

**The `ENTRYPOINT ["vllm","serve"]` trap applies here.** The reference builds on
`vllm/vllm-openai:nightly-…`, which is exactly the family that sets that entrypoint, so under dgxrun
this needs our usual entrypoint-less overlay layer. (It does *not* apply to the eugr/spark-arena
images, which are built for this runner.)

So: **new build required, not a version bump** — and the build must target a *pinned commit*, not a
branch head. See the silent-failure table for why that is not a stylistic preference.

---

## §7 — Verdict

**Servable with about a week of work, at TP4 across all four Sparks.** The model is real, open, MIT,
and genuinely strong (it beats V4-Pro on most agentic benchmarks at 8–16 B active parameters). It is
*not* blocked on missing upstream work: the arch is in vLLM `main`, the sm_121 kernel gaps have
published fixes, and three independent parties are already serving it on our exact hardware. What it
costs us:

1. **A custom container** from a pinned vLLM commit + FlashInfer v0.7.0rc1 + two prebuilt kernel
   extensions, plus seven bind-mounted patch files (§6, §5.3).
2. **The whole fleet.** TP4 on four Sparks, ~81 GiB/rank. TP2 does not fit either way. *(The TP3 lane
   is the interesting hedge — 3 Sparks, one free, and it benchmarks slightly faster per stream.)*
3. **Storage work.** One node holds the 510 GB checkpoint on local NVMe and NFS-exports it read-only;
   each worker needs ~48 GB of **local** disk for its Engram row slice. `/mnt/tank` serves the bulk
   weights but must not be in the Engram path.
4. **Recipe rewrite.** About two-thirds of our lines change (§2.8).

**The single biggest blocker is the 189.13 GiB Engram table** — and specifically that on GB10 unified
memory, the default `cpu_offload: true` "solution" frees nothing while looking like it should.

**Five operational rules that fall straight out of the evidence:**

1. **Serve the native 475.24 GiB checkpoint.** Every working 4×GB10 recipe does. No vLLM-loadable
   quant is both complete and trustworthy (§4.2).
2. **Never build on vLLM < v0.28.0** — v0.27.1 lacks the #52035 DeepGEMM fix that bit a two-Spark
   DeepSeek-V4 deployment, and pin a *commit*, not a branch.
3. **Keep KV at `fp8`, not NVFP4** — NVFP4 KV is 24/24 failing on GB10 and the fix is unmerged.
4. **Validate with a ≥100 K needle test, never a short prompt** — two of the three confirmed
   wrong-output bugs are invisible to smoke tests, and one was measured on 4× DGX Spark with exactly
   this checkpoint.
5. **Do not trust `MemAvailable`** while staging weights on these nodes.

**If the goal is to hear the model talk quickly** rather than to advance our vLLM stack, `antirez/ds4`
with `ds41f-q2` on two Sparks is a much cheaper experiment (81 GiB/rank, 21.9 tok/s measured, GB10
images published) — at the cost of no vision, no DSpark, and a runtime sparkrun does not yet have.

### Flags that fail silently

Nothing in this table raises an error. Every row is a wrong-output or silent-degradation path.

| flag / setting | what happens | evidence |
|---|---|---|
| **Building the engine from the vLLM branch *head* instead of the pinned commit `e47aa780bccf…`** | *"an engine that boots without a single error, passes profiling and graph capture, and **emits one repeated garbage token from the first position**, with DSpark accepting nothing."* The worst failure mode in this document. | [RECIPE.md §3](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/docs/RECIPE.md) |
| `--tool-call-parser deepseek_v4` on V4.1 | V4.1 emits `<｜DSML｜ calls>` (**leading space**); the V4 parser matches `<｜DSML｜tool_calls>`. Tool calls arrive as prose, no exception. | `tool_calls_block_name` `"tool_calls"` vs `" calls"` in vLLM's two encoding modules; [encoding/README.md](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/encoding/README.md) |
| **`reasoning_effort` aliases mean different numbers in vLLM than in DeepSeek's reference** | vLLM: `low=25, high=50, xhigh=75, max=100`. DeepSeek: `low=50, high=75, max=100` (no `xhigh`). Both default to `"high"` — so **vLLM's default renders `Reasoning Effort: 50` where DeepSeek's renders `75`.** Quieter reasoning, no warning. Card benchmarks used **100**. | [vllm `deepseek_v41_encoding.py` L183-L190](https://github.com/vllm-project/vllm/blob/main/vllm/tokenizers/deepseek_v41_encoding.py) vs [DeepSeek `encoding.py` L444-L449](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/encoding/encoding.py) |
| `engram_config.cpu_offload` — **default `true`** — on GB10 | "Offloads" 47.21 GiB/rank to pinned host memory. On unified memory that frees **zero**; you OOM later with a misleading cause. | [`vllm/config/engram.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/engram.py); `VLLM_PLE_CPU_OFFLOAD: bool = True` |
| Engram rank-offset bug (pre-patch) | *"without it, ranks 1-3 read **rank 0's rows**"* — every rank looks up the wrong n-gram memory. Serves fine, answers worse. | [boot3-wedge-and-engram-offset.md](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/docs/boot3-wedge-and-engram-offset.md) |
| `enable_adaptive_verification: true` on sm_121 | Forces variable-length decode graphs with padded rows — the FlashInfer #5015 trigger; **padded speculative batches can hang SM120 sparse MLA**. | [RECIPE.md §5](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/docs/RECIPE.md) |
| Leaving `cudagraph_capture_sizes` to defaults with DSpark k=5 | Decode batches are multiples of 6 target / 5 draft tokens; without exact graphs they get **padded**, same #5015 trigger. | same |
| `--attention-backend B12X_MLA_SPARSE` (V4.0 name) | vLLM's own comment says the DSV41 names exist "so a V4.1 model never resolves the V4.0 backend classes" — reaching a V4.0 class with V4.1's layer-sharing gives wrong attention, not a crash. | [registry.py L107-L116](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py#L107-L116) |
| **`DEEPGEMM_SCALE_UE8M0=false` on GB10** | The `wo_a` absorb GEMM is **silently ~25 % wrong**: DeepGEMM's assert covers the *weight* scale but not the *activation* scale, so it truncates the mantissa and returns a wrong answer. Measured max relative error **2.52e-01** vs 2.50e-03 correct — **on 4× DGX Spark SM121 with exactly this checkpoint's quant config.** | [SGLang #39193](https://github.com/sgl-project/sglang/issues/39193) |
| **Misreading the FP8 KV block layout** | It is `[64 tok × 128 B fp8 ‖ 64 tok × 4 B fp32 scale]` — **block-appended, not per-token interleaved**. A wrong interpretation **passes short-prompt smoke tests and only fails at ≥100 K needle retrieval.** Validate with a long-context needle test, never a smoke prompt. | [vLLM #56700](https://github.com/vllm-project/vllm/issues/56700) |
| `fuse_norm_quant` / `fuse_act_quant` (**default-enabled**) | Garbled output on GB10 even at temperature 0, on DeepSeek-V4-Flash. Applies to our *current* deployment too. | [vLLM #50773](https://github.com/vllm-project/vllm/issues/50773) |
| **NVFP4 KV cache on sm_121** | "V dequantizes to garbage." GB10: **24 failed / 0 passed** on main. A binary with no SASS for the device can also fail `cudaErrorNoKernelImageForDevice` and, unchecked, **write nothing** — surfacing downstream as NaN "quantization" rather than an error. | [#50084](https://github.com/vllm-project/vllm/issues/50084), [#55976](https://github.com/vllm-project/vllm/pull/55976) |
| `VLLM_MARLIN_INPUT_DTYPE=fp8` on GB10 | Silently corrupts output on sm_121a — repeated `</think>` loop at temperature 0, and the kernel runs **~2.5 % faster**, so throughput metrics look fine. | [vLLM #49546](https://github.com/vllm-project/vllm/issues/49546) |
| Reusing the V4 `[128,128]` FP8 block size on a V4.1 checkpoint | **"Rescales every dequantized weight without raising."** | [llama.cpp #28696](https://github.com/ggml-org/llama.cpp/pull/28696) |
| `--moe-runner-backend deep_gemm` on SGLang / 4× Spark | Accepted at parse time, then dies in CUDA-graph capture. The working value is `flashinfer_mxfp4`. | [SGLang #39226](https://github.com/sgl-project/sglang/issues/39226) |
| Trusting `MemAvailable` on GB10 | 29 GiB can vanish in 9 s; the host reports 22.5 GiB free while the driver already returns `NV_ERR_NO_MEMORY`. **No host-side signal can guard it.** | [vLLM #56824](https://github.com/vllm-project/vllm/issues/56824) |
| `launch/dsv41-tp4.sh` defaults | Ships `TEXT_ONLY=1, PARSERS=0` — **vision and tool calling silently off** unless you set them, as `boot10-go.sh` does. The repo warns "Watch the defaults." | [RECIPE.md §5](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/docs/RECIPE.md) |
| No Jinja chat template in the repo | `tokenizer_config.json` has **no `chat_template`**. A stack that falls back to a generic template produces a subtly wrong prompt (missing `<think>`, missing effort prefix) and still generates fluent text. | [tokenizer_config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/tokenizer_config.json) |
| `REAP-256E` as a drop-in checkpoint | Index declares 48 shards and maps Engram to `model-00047/00048`; **only 46 shards are uploaded**. Resolves to files that do not exist. | measured via `?blobs=true` + its `model.safetensors.index.json` |
| NVFP4 community builds | Two of three are **491.1 GiB — larger than the 475.24 GiB native checkpoint**, because the experts are already FP4 and the Engram is untouched. | measured via `?blobs=true` |
| ds4 `download_model.sh` help calling `ds41f-q2` "Metal only" | Contradicts `MODELS.md` / `DGX_SPARK.md`, which document CUDA Spark usage and publish CUDA measurements. Likely stale help text. | [download_model.sh](https://github.com/antirez/ds4/blob/main/download_model.sh) vs [DGX_SPARK.md](https://github.com/antirez/ds4/blob/main/docs/DGX_SPARK.md) |

### What I could NOT determine

| gap | what would settle it |
|---|---|
| Whether `dgx-vllm-eugr-nightly-b12x:latest` already contains PR #56214 and the 2026-09-12 b12x V4.1 commits | `docker pull` and check `vllm.__version__` / commit label and `pip show b12x`. I did not pull images. |
| Whether b12x's own TP4+SSD path works on **sm_121a** | Its record is `CUTE_DSL_ARCH=sm_120a` on a "discrete SM120 platform". Only a GB10 run settles it. (The Tech2Wild path *is* GB10-proven, but it is not the b12x path.) |
| The ~2× gap between my derived 890 B/token KV and b12x's measured ~1.74 KiB/token | Read vLLM's V4.1 `cache_utils.py` page layout, or instrument a run. Budget from the measured figure. |
| Whether `draft_sample_method` should be `probabilistic` or `greedy`, and k=5 vs k=3 | Records disagree (official recipe + Tech2Wild: probabilistic k=5; b12x: greedy; 0xTank and bertholomus: k=3). An A/B on acceptance length. |
| Whether REAP-256E's missing shards can be satisfied from the base repo | Shard numbering lines up, but nobody documents it and I did not test. |
| Whether the EXL3 3.5 bpw lane holds up on quality | It is the practitioner's *default* on throughput and KV grounds; I found no quality comparison against the release checkpoint. |
| Whether `--max-cudagraph-capture-size`, `VLLM_USE_AOT_COMPILE`, `VLLM_USE_MEGA_AOT_ARTIFACT` carry over | Not set in any V4.1 record I found. Needs a run. |
| Long-run stability on our fleet | Every published number is hours-to-days old on someone else's hardware, and one record explicitly flags GPU "slow-state" phases and clock latch as recurring hazards. |
