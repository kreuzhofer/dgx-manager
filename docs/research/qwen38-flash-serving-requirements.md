# Serving "Qwen3.8-Flash" on the GB10 Spark cluster

**Research date:** 2026-09-06 · **Asked as:** *"Qwen3.8-Flash, the larger sibling of Qwen3.8-27B"*

**Method.** Primary sources only. Hugging Face model cards and raw `config.json` /
`chat_template.jinja`, with every byte count taken from `huggingface.co/api/models/<repo>?blobs=true`
— these are real file sizes, not rounded claims. vLLM source on `main` and the GitHub API for
issues, PRs and **commit ancestry** (whether a fix is in an image was checked with
`repos/vllm-project/vllm/compare/<merge-sha>...<image-sha>`, not by comparing dates). Docker and
GHCR registry manifests read directly, so entrypoints and `TORCH_CUDA_ARCH_LIST` come from image
configs rather than documentation. Plus one third-party working record whose measurements are on
our exact hardware, and the community reports it links.

**Labelling.** Numbers that are my arithmetic rather than someone's measurement are marked
`[inference]`. Third-party measurements are attributed. Two claims I was given as premises turned
out to be **misattributed**, and are corrected in §3.2 — read those before quoting anything from
elsewhere on this model.

---

## BOTTOM LINE

**The name is wrong; the model exists under a different name; it fits our fleet far more easily
than the single-Spark heroics circulating in the community suggest, because we have four Sparks and
they have one; and the container we need is already published under a tag we already pull. The real
risk is not fit — it is a cluster of sm_121 correctness bugs in the sparse-attention indexer, and a
documented tendency for multi-node Sparks to wedge above ~95–100 K context.**

| Claim / question | Verdict |
|---|---|
| **"Qwen3.8-Flash" exists as open weights** | **No.** `Qwen/Qwen3.8-Flash` 404s. Qwen3.8-Flash is the **Qwen Cloud API product**. The open-weight model is **`Qwen/Qwen3.8-Flash-Next`** (2026-08-24/26), Qwen's *"experimental preview of the architecture that will underpin Qwen4"*. (§1.1) |
| **It is the larger sibling of Qwen3.8-27B** | **Yes, and it is a different architecture, not a scaled 27B.** 27B is `Qwen3_5ForConditionalGeneration`, 51.75 GiB. Flash-Next is `Qwen4ExpForConditionalGeneration`, **180.0 B params / 335.28 GiB at BF16**. (§1.2) |
| **Parameter count is 125 B** | **Partly.** 125 B is the transformer. The checkpoint holds **179,999,981,459** params: 125 B body + **51.2 B n-gram (PLE) table** + 4 B MTP. The table is 28 % of the weights and drives every fit decision. (§1.3, §2.2) |
| **Multimodal; can it take our 8 images per call** | **Yes.** 27-layer vision tower, images and video. vLLM's `limit_mm_per_prompt` defaults to **999 per modality**, so 8 images needs no flag. ~576 tokens per 768 px image `[inference]`. (§1.4) |
| **Fits on one Spark** | **Only via an unmerged PR.** Smallest published NVFP4 build is 123.6 GiB against ~105 GiB of budget. One Spark works — 36.5 tok/s, third-party measured — but needs **vllm#53899 (PLE CPU offload), still open, not on `main`** (`vllm/v1/ple_offload/` 404s, `VLLM_PLE_CPU_OFFLOAD` absent from `envs.py`). (§2.4, §4.4) |
| **Fits on 2 or 4 Sparks** | **Yes, and it is already being done.** The n-gram table is a `VocabParallelEmbedding` — **sharded by TP, not replicated** (verified in source, and confirmed by a dual-Spark ledger showing 62.5 GB/node for a 125.9 GiB checkpoint). TP2 FP8 = **86.4 GiB/rank**; TP4 BF16 = **83.8**; TP4 FP8 = **43.2**. (§2.3, §2.4) |
| **Anyone running it multi-Spark already?** | **Yes — at least six independent reports.** vLLM TP2 on 2× GB10, vLLM TP4+EP on 4× GB10, SGLang TP2 on 2× GB10 at **64 tok/s single-stream**. Three of the six are bug reports, but in all of them the engine comes up and serves. (§2.6) |
| **Fits on the RTX 5090** | **No, by 3×.** Smallest complete checkpoint on HF is 101.68 GiB against 32 GiB of discrete VRAM, and the offload trick that rescues one Spark depends on unified memory. (§2.7) |
| **AWQ / GPTQ-Int4 is the small option** | **Backwards.** Every W4A16/AWQ build on HF is **167–175 GiB — larger than official FP8** — because they leave the 51.2 B table in BF16 (95.4 GiB alone). The whole field uses **NVFP4 (W4A4)**. No Int4 result on sm_121 exists at all. (§3.1) |
| **Supported in vLLM in-tree** | **Yes, since 2026-08-31 (PR #53896) — `main` only.** Not in any release: v0.28.0 shipped 2026-08-26. **Not in `docs/models/supported_models.md`** (zero hits). No `--trust-remote-code` — vLLM ships its own `Qwen4ExpConfig`. (§4.1, §4.2) |
| **We need a fresh container build** | **No.** `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2026090501` is vLLM `0.28.1rc1.dev441+g2902ca17e`, **309 commits ahead of the #53896 merge, 0 behind**, arm64, `TORCH_CUDA_ARCH_LIST=12.1a`, entrypoint `nvidia_entrypoint.sh` — so the `vllm/vllm-openai` `ENTRYPOINT ["vllm","serve"]` trap does not apply. (§5) |
| **Pipeline parallel as an alternative to TP** | **Hard-refused in code**, not merely unsupported: `NotImplementedError` in `models/config.py:879`. TP only, and **TP must be a power of two** (TP3 fails on the vision encoder's 16 heads). (§4.3) |
| **Biggest blocker** | **Not fit. The QSA indexer top-k on sm_121, and multi-node long-context stability.** Greedy output is non-deterministic above `indexer_budget` on 2× Spark (#54521); `persistent_topk` silently returns wrong values (#51782); 4× GB10 TP4+EP wedges around the third >100 K-token request (#54629); dual GB10 dies from ~95 K context. (§6) |

**Verdict: servable with about a day of work, at TP2 on two Sparks.** First shot:
`Qwen/Qwen3.8-Flash-Next-FP8`, TP2, on `:2026090501`, **MTP off**, **prefix caching off**, KV
`bfloat16`, `--mamba-cache-mode align`, `--tool-call-parser qwen3_xml`, `reasoning_effort` pinned to
`medium`, and a **32–64 K window on the first boot, not 262 K**.

---

## §1 — Identity

### 1.1 The name is wrong, and the correction matters

There is no open-weight `Qwen3.8-Flash`. Checked against the HF API:

```
Qwen/Qwen3.8-Flash           -> HTTP 401 (does not exist)
Qwen/Qwen3.8-Flash-Next      -> exists, created 2026-08-24, modified 2026-08-27
Qwen/Qwen3.8-Flash-Next-FP8  -> exists, modified 2026-08-31
```

The Qwen org publishes exactly six `Qwen3.8-*` repos: `27B`, `27B-FP8`, `2.4T-A95B`,
`2.4T-A95B-FP8`, `Flash-Next`, `Flash-Next-FP8`.

Qwen's model card explains the naming, verbatim
([source](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/README.md)):

> **Qwen3.8-Flash** is the official version based on Qwen3.8-Flash-Next with more production
> features, e.g., 1M context length by default, official built-in tools.

with a link to Qwen Cloud. **"Qwen3.8-Flash" is the hosted API product; "Qwen3.8-Flash-Next" is the
weights.** Everything below concerns Flash-Next.

*(Aside: the genuinely largest open Qwen3.8 sibling is `Qwen/Qwen3.8-2.4T-A95B` — 2.446 T
parameters, **4,556 GiB** of BF16 safetensors. Not a candidate for anything we own.)*

### 1.2 What it is, from `config.json`

[Raw config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json):

```json
"architectures": ["Qwen4ExpForConditionalGeneration"],
"model_type": "qwen4_exp",
```

| field | value |
|---|---|
| `num_hidden_layers` | 48 |
| `hidden_size` | 2560 |
| `layer_types` | 12 × (3 × `linear_attention` → 1 × `full_attention`), `full_attention_interval: 4` |
| `num_experts` / `num_experts_per_tok` | **512 / 10** routed, + 1 shared |
| `moe_intermediate_size` / `shared_expert_intermediate_size` | 640 / 640 |
| `num_attention_heads` / `num_key_value_heads` | 24 / **2** |
| `head_dim` | 256 (`partial_rotary_factor` 0.25 → 64 rotary dims) |
| GDN | `linear_num_value_heads` 48, `linear_num_key_heads` 16, head dims 128, `linear_conv_kernel_dim` 4, `mamba_ssm_dtype` float32 |
| QSA indexer | `indexer_n_heads` 4, `indexer_kv_heads` 1, `indexer_head_dim` 128, **`indexer_budget` 2048**, `indexer_compress_ratio` 4 |
| n-gram (PLE) | `ngram_vocab_size_base` **20,000,000**, `ngram_size` 3, `heads_per_ngram` 8, `split_ngram_parts` 128, `ple_layer_ids` **[2]**, `ple_embed_dim` 2560 |
| hyper-connections | `hc_count` 4, `hc_lowrank` 320 |
| MTP | `mtp_num_hidden_layers` 1, `hybrid: true`, one `full_attention` layer |
| `max_position_embeddings` | **262,144** (YaRN to 1 M) |
| `vocab_size` | 248,320 |
| `output_gate_type` | **`sigmoid`** — see §5.2, the default in the C++ op is `silu` |

Vision config: 27 layers, `hidden_size` 1152, `patch_size` **16**, `spatial_merge_size` 2,
`out_hidden_size` 2560, `num_heads` **16** (this is what breaks TP3, §4.3).

The card's own layout line:

> Hidden Layout: 12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))

### 1.3 Parameter count — the number everyone gets wrong

The card says *"125B with 6B activated, plus 51B n-gram embedding and 4B MTP"*. The safetensors
index gives the number that actually matters for memory:

```
"safetensors": {"parameters": {"BF16": 179999981424, "I64": 35}, "total": 179999981459}
```

**180.0 B parameters on disk.** The n-gram table alone is 20,000,000 × 2560 = **51.2 B parameters**
— `[inference]`, but it reconciles to the byte in §2.2, so treat it as established.

### 1.4 Multimodality and our 8-images-per-call requirement

The card documents `image_url` and `video_url` message content with working OpenAI-client examples,
and states **no per-request image limit**.

`vllm/config/multimodal.py` on `main`:

```python
limit_per_prompt: MMDummyOptions = Field(default_factory=dict)
"""The maximum number of input items and options allowed per prompt for each modality.

Defaults to 999 for each modality.
```

**8 images per request needs no flag.** Practical notes:

- At 768 px a square image is `(768/32)² = 576` tokens `[inference]` — 8 of them ≈ **4,608 prompt
  tokens** before any text. That matches the ~580 tokens/view measured for Qwen3.8-27B in this repo,
  which uses the same patch/merge geometry.
- `--limit-mm-per-prompt '{"image":8}'` is still worth setting to *bound* the startup profiling run.
- **The vision-encoder warmup allocates outside the profiled budget.** That is how the
  GLM-5.3-Flash deploy in this repo got a worker SIGKILLed at `gmu 0.89` with 0.24 GiB of slack.
  Same shape here.
- One field-reported image-request trap: *"this is a reasoning model and thinking consumes
  `max_tokens` first — if `content` is empty with `finish_reason: "length"`, just raise `max_tokens`
  (2–4 k covers most image QA)"*. Detect empty content by `finish_reason`, **never** by counting
  characters.
- `--mm-encoder-tp-mode data` appears in at least one published Spark launch line; untested here.
- Vision is a **vLLM-path-only** capability: one field report records *"333 vision tensors present
  and 0.967 on their image eval, against a GGUF with none"*.

### 1.5 Behavioural defaults to pin now

From `chat_template.jinja`:

```jinja
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
... 'Supported types are xhigh (default), medium, and low.'
```

**`reasoning_effort` defaults to `xhigh`** — the same trap this fleet already documented for
Qwen3.8-27B (memory note *"Qwen3.8 defaults reasoning_effort to xhigh"*; both 27B recipes pin
`medium`). Pin it. Note also the field finding that `chat_template_kwargs.thinking_budget` is **not
honoured** by these builds — `reasoning_effort` is the only working lever.

The tool-call format in the template is **XML** (`<tool_call><function=…><parameter=…>`), so the
parser is `qwen3_xml`, not the `qwen3_coder` our 27B recipes use. Caveat worth knowing: the field
record notes that in their build `vllm/tool_parsers/__init__.py` maps *both* names to the same
`Qwen3EngineToolParser`, so the load-bearing flag is `--enable-auto-tool-choice`, not the parser
name. Without both, **every request carrying `tools` returns HTTP 400** — a capability failure no
throughput benchmark can see.

---

## §2 — Fit: the arithmetic

### 2.1 What one Spark actually gives us

Not the nominal 128 GB. From this repo's own measured numbers:

| source | figure |
|---|---|
| `zai-glm-5.3-flash-libertai-nvfp4-2x.yaml` (measured on our fleet) | CUDA-visible **total ≈ 121.6 GiB**, **free at launch 108.49 GiB**, `gmu 0.87` → **105.82 GiB** budget, **2.67 GiB** warmup slack |
| same file | `gmu 0.89` → 0.24 GiB slack → worker **SIGKILLed** during the multimodal warmup |
| `qwen3.8-27b-nvfp4.yaml` (measured) | `gmu 0.88`, `vramActual` 102,892 MiB = 100.5 GiB |
| third-party GB10 (Flash-Next, TP1) | total **121.63 GiB**, free at startup **114.3 GiB**, runs `gmu 0.90` |

**Working budget: ~105 GiB per Spark**, and for a multimodal model leave ≳2.7 GiB on top of it.
Four Sparks → **~420 GiB aggregate**.

One field warning worth carrying: *lowering* `gpu-memory-utilization` to avoid host freezes is
**refuted** — 0.70 was the worst recorded outcome; the cause is absolute free memory at launch, not
the ratio.

### 2.2 Weight bytes — measured, not estimated

Each row is the summed size of that repo's `.safetensors` blobs, including quantization scales,
which is what has to be resident.

| checkpoint | dtype composition (params) | **GiB** |
|---|---|---:|
| `Qwen/Qwen3.8-Flash-Next` | BF16 180.0 B | **335.28** |
| `Qwen/Qwen3.8-Flash-Next-FP8` | F8_E4M3 174.5 B + BF16 5.49 B | **172.78** |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | U8 60.40 GB packed + F8 53.7 B + BF16 5.49 B | **123.57** |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | U8 60.40 GB + F8 **51.20 B** + BF16 8.00 B | **125.91** |
| `provsalt/…-NVFP4-PLE-NVFP4` | U8 87.26 GB + BF16 5.49 B | **101.68** |
| `Intel/…-W4A16-AutoRound` | I32 15.2 B + BF16 **59.20 B** | 168.75 |
| `wtdcode/…-AWQ-W4A16` | I32 120.8 B + BF16 **59.20 B** | 168.31 |
| `cyankiwi/…-AWQ-INT4` | I32 120.8 B + BF16 **59.20 B** | 175.36 |
| `Inferact/…-NVFP4` | U8 61.66 GB + BF16 **56.69 B** | 170.23 |
| `primitive-ai/…-NVFP4` | U8 60.40 GB + BF16 **59.20 B** | 173.59 |

The `RadixArk` row settles §1.3 exactly: its FP8 group is **51,200,245,760 bytes** = 51.2 B
one-byte values — **the n-gram table and only the n-gram table**.

**The pattern that decides everything.** The 51.2 B-parameter table is 95.37 GiB in BF16,
47.68 GiB in FP8, 23.84 GiB in NVFP4 `[inference]`. Checkpoints that quantize the *body* but leave
the *table* in BF16 — every AWQ/W4A16 build, plus `Inferact` and `primitive-ai` — land at
167–175 GiB and are **larger than official FP8**. Only ModelOpt NVFP4 builds with a quantized table
get under 130 GiB.

### 2.3 Does the n-gram table shard across TP ranks? — Yes. Verified twice.

This is the load-bearing question: a *replicated* 47.7 GiB table would sink TP2 outright.

**In source** — `vllm/models/qwen4_exp/common/ple.py` on `main`
([link](https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/common/ple.py)):

```python
class PLEVocabParallelEmbedding(VocabParallelEmbedding):
    """Vocab-parallel embedding that accepts checkpoint row shards."""
    ...
        copy_ple_embedding_shard_(
            param, loaded_weight,
            checkpoint_start=checkpoint_start,
            tp_start=self.shard_indices.org_vocab_start_index,
            tp_end=self.shard_indices.org_vocab_end_index,
        )
```

Each rank materialises only its `[org_vocab_start_index, org_vocab_end_index)` slice of the 20 M
rows. **Per-rank weight bytes = checkpoint bytes / TP.**

**Empirically** — the MiaAI-Lab dual-Spark deployment (SGLang TP2, `RadixArk` NVFP4, 125.91 GiB)
publishes a per-node ledger of **GPU weights ~62.5 GB**. 125.91/2 = 62.96. The arithmetic and the
hardware agree.

*(An open PR, [#54371](https://github.com/vllm-project/vllm/pull/54371) "[Qwen4] Support UVA
PLE-offload and N-gram parallelism", suggests further parallelism work is coming; it does not change
what `main` does today.)*

### 2.4 The fit table

Budget **~105 GiB/rank**.

| placement | precision | GiB/rank | headroom | verdict |
|---|---|---:|---:|---|
| 1 Spark (TP1) | BF16 | 335.3 | −230 | no |
| 1 Spark | FP8 | 172.8 | −68 | no |
| 1 Spark | NVFP4 (nvidia / RadixArk) | 123.6 / 125.9 | −19 / −21 | **no as published** |
| 1 Spark | NVFP4 + NVFP4 PLE (provsalt) | 101.7 | +3 | arithmetically marginal, nothing left for KV → no |
| 1 Spark | NVFP4 **+ PLE offload** | **76.6 resident** (measured) + table paged to host/swap | — | **yes — but needs unmerged vllm#53899** (§4.4) |
| 2 Sparks (TP2) | BF16 | 167.6 | −63 | no |
| **2 Sparks (TP2)** | **FP8** | **86.4** | **+19** | **yes** |
| 2 Sparks (TP2) | NVFP4 | 61.8 / 63.0 | +42 | yes, roomy — and matches a shipped dual-Spark ledger |
| **4 Sparks (TP4)** | **BF16** | **83.8** | **+21** | **yes** |
| **4 Sparks (TP4)** | **FP8** | **43.2** | **+62** | **yes, very roomy** |
| 4 Sparks (TP4) | NVFP4 | 30.9 / 31.5 | +74 | yes |
| RTX 5090 (32 GiB discrete) | anything | ≥101.7 | −70 | **no, by 3×** |

TP3 and other non-powers-of-two are excluded independently — §4.3.

### 2.5 Context budget

KV is unusually cheap here: only **12 of 48 layers** hold a KV cache, with **2 KV heads**.

`[inference]` KV bytes/token at BF16 = 12 × 2 × 256 × 2 (K,V) × 2 B = **24,576 B = 24 KiB**. A full
262,144-token window is **6.0 GiB for one sequence**.

Cross-checked against a real GB10 measurement (TP1, NVFP4 + PLE offload, `--max-model-len 262144`):
the engine reported a **1,077,542-token** BF16 pool = **24.7 GiB** at 24 KiB, i.e. 4.11× concurrency
at the full window. Structure confirmed.

At TP2/TP4 each rank holds 1 KV head → **12 KiB/token/rank** `[inference]`, ~3.0 GiB/rank for 262 K.
The ~19 GiB/rank left over at TP2-FP8 would buy roughly **6× concurrency at 262 K** `[inference]`.

**Other consumers you must not forget:**

- **GDN (Mamba) state is per *sequence*.** `[inference]` 36 linear-attention layers × 48 value heads
  × 128 × 128 × 4 B (`mamba_ssm_dtype: float32`) ≈ **108 MiB/sequence** at TP1 — ~27 GiB at
  `--max-num-seqs 256`, ~1.7 GiB at 16. **Keep `max-num-seqs` low on a Spark.** It shards with TP.
- **Speculation costs KV pool, hard.** Measured on GB10: MTP doubles the attention block size
  (800 → 1600) because *"attention page size must be >= mamba page size"*, and the pool falls
  **−36 % at k=2**, **−56 % at n=6**. `ngram n=4` costs only −17 %.
- **Real KV pools measured on one GB10**, for calibration: `mml 8192` → 30.99 GiB;
  `mml 32768, gmu 0.90` → 985,006 tok; `mml 262144` bf16 → 1,077,542 tok.

### 2.6 Multi-Spark is not theoretical — it is already reported

Six independent reports of this model on more than one GB10. Three are bug reports, but in every
one the engine comes up and serves.

| deployment | stack | result |
|---|---|---|
| [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | **SGLang TP2, 2× Spark**, RadixArk NVFP4 | **64 tok/s single-stream.** Per node: GPU weights ~62.5 GB, pinned host PLE ~11 GB, KV (956,800 tok) 9.0 GB, free ~17.5 GB |
| `ryangu00/dell-pro-max-gb10` | **vLLM TP2, 2× GB10** | 2,321 tok/s prefill @30 k — **but a worker dies from ~95 k context on sm_121 dual-node** |
| [vllm#54521](https://github.com/vllm-project/vllm/issues/54521) | **vLLM TP2, 2× Spark**, `--enforce-eager` | serves; greedy non-determinism above `indexer_budget` (§3.2 Q7) |
| [vllm#54629](https://github.com/vllm-project/vllm/issues/54629) | **vLLM TP4 + `--enable-expert-parallel`, 4× GB10** | *"serves normally until roughly the third request whose prompt exceeds ~100k tokens… That request never completes. A worker rank's GPU stream stops advancing"* |
| [vllm#54919](https://github.com/vllm-project/vllm/issues/54919) | **vLLM TP2 + EP, 2× Spark** | *"Aggregate decode throughput falls to approximately 0.5-5 tok/s for about 3-7 minutes"* during long prefills |
| sglang#36797 / SGLang #37995 | **SGLang TP2, 2× Spark** | KV-dtype A/B (44.0 nvfp4 vs 56.8–58.6 fp8 vs 54–59 bf16 tok/s); a cookbook for NVFP4 TP2 **with PLE offload off**, which is exactly our shape |

**Read this two ways.** The good news: TP2 and TP4 across Sparks demonstrably work, and one of them
reaches 64 tok/s single-stream — well above the 36.5 tok/s single-Spark ceiling. The bad news: the
two long-context failures (#54629, ryangu00) both bite between ~95 K and ~100 K tokens on multi-node
sm_121. **Do not open at 262 K.**

### 2.7 The RTX 5090

Not close, and not rescuable. The smallest complete checkpoint is 101.68 GiB against 32 GiB. The
body alone at NVFP4 (~78 GiB in the RadixArk split) is 2.4× the card. And PLE offload depends on
**unified memory** — on a discrete card the 16 scattered 160-byte lookups per token would cross
PCIe every step. Off the table for vLLM.

*(For scale, the closest sm_120 success anyone reports is a single **RTX PRO 6000 with 96 GiB** —
`mratsim/sglang-qwen38fn-sm120-turbo`, ~200 tok/s per stream, 1,170 aggregate at 8 — i.e. three
times our 5090's memory. GGUF/MLX builds exist for llama.cpp and Macs; different runtime, out of
scope for this fleet.)*

---

## §3 — Quantization

### 3.1 What exists (vLLM-loadable safetensors only; GGUF/MLX omitted)

| repo | format | GiB | note |
|---|---|---:|---|
| `Qwen/Qwen3.8-Flash-Next-FP8` | official FP8 E4M3, table included | 172.78 | the reference quant; upstream-validated on GB300/GB200/H200/H100/MI355X |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | ModelOpt v0.46 MIXED_PRECISION: routed experts **W4A4 NVFP4 g16, MSE-calibrated**; everything else in the main model **BF16**; MTP experts FP8 128×128 block-scaled; PLE per-tensor FP8 | 123.57 | NVIDIA's own. Needs vLLM ≥ `d4d703ca` (= #54882). Sample command is TP8. Reported accuracy FP8→NVFP4 within noise on nine benchmarks (GPQA 92.0→91.5, MMMU Pro 77.1→78.3, Terminal-Bench 2.1 83.3→82.9). **Cannot load with MTP today** — [#55496/#55498](https://github.com/vllm-project/vllm/issues/55496), `Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'` |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | NVFP4 body (dense projections left BF16) + FP8 table | 125.91 | what most of the GB10 field uses. **Needs a one-line gate change** — see 3.3 |
| `provsalt/…-NVFP4-PLE-NVFP4` | NVFP4 body **and** NVFP4 table | **101.68** | smallest anywhere; loads via plugin `VLLM_PLUGINS=qwen38_nvfp4_ple` and a public container. See 3.4 |
| `Inferact/…-NVFP4` | NVFP4 body, BF16 table | 170.23 | the default variant of the official `vllm-project/recipes` file. **GB10 not in its verified list** |
| `primitive-ai/…-{NVFP4,mixed-NVFP4-FP8}` | NVFP4, BF16 table | 173.6 / 171.1 | actively updated (2026-09-06). Also ships a separate `…-PLE-quant` repo with FP8/INT4/NVFP4 tables served **memory-mapped from disk** |
| `Intel/…-W4A16-AutoRound`, `wtdcode/…-AWQ-W4A16`, `VnimanieAI/…-W4A16`, `cyankiwi/…-AWQ-INT4` | Int4 W4A16 | 167–175 | table BF16 |
| `unsloth/…-FP8` | FP8 mirror | — | |

**Headline: AWQ/GPTQ-Int4 is the wrong axis on this model.** Every Int4 build is larger than
official FP8, and — searching the whole GB10 field record — **there is no W4A16 or AWQ result on
sm_121 at all**. The only W4A16 datapoints are a 4×3090 build (*"Ampere, so W4A16 is forced and the
quant choice does not transfer"*) and a Strix Halo build. The field standard is **NVFP4 (W4A4) for
routed experts, FP8 or MXFP8 for the dense side path**.

### 3.2 Known-broken and known-degrading on sm_120 / sm_121

Each entry names the mechanism, because that is what predicts whether it bites *our* config.
**[SILENT]** = healthy-looking server, wrong output.

| # | Symptom | Mechanism | Applies to us? |
|---|---|---|---|
| **Q1** | Engine dies in `profile_run`, `CUDA error: unspecified launch failure` in `deep_gemm.fp8_gemm_nt` | [vllm#54125](https://github.com/vllm-project/vllm/issues/54125) (open): *"`support_deep_gemm()` returns **True on sm_121**… The gate accepts the whole `120` capability family, and GB10 is in that family without being a device DeepGEMM actually runs on."* Its sm_120 twin is [#51884](https://github.com/vllm-project/vllm/issues/51884) — *"Unknown SF transformation"* at load on an RTX 5090. **Same defect, both sides of the 12.x line.** A field follow-up narrows it further: on sm_121 `is_deep_gemm_e8m0_used()` also returns True, selecting an **E8M0 kernel variant whose scale format the checkpoint does not supply** | **Yes for any blockwise-FP8 checkpoint.** `VLLM_USE_DEEP_GEMM=0` → `CutlassFp8BlockScaledMMKernel`. Also open: [#54600](https://github.com/vllm-project/vllm/pull/54600) "Exclude SM121 from DeepGEMM support" |
| **Q2** | Fluent garbage, whole checkpoint, server healthy **[SILENT]** | ModelOpt `MIXED_PRECISION` checkpoints can declare per-layer `quant_algo: "FP8_PB_WO"`. `ModelOptMixedPrecisionConfig.get_quant_method()` dispatches FP8/NVFP4/W4A16_NVFP4/MXFP8 then **falls through to `UnquantizedLinearMethod()`**, loading packed FP8 bytes into BF16 parameters. Also [#54126](https://github.com/vllm-project/vllm/issues/54126): the method *"supports exactly one of the two `FP8_PB_WO` export conventions that ModelOpt produces"* | **Yes for any mixed-precision checkpoint.** *A format the runtime does not recognise is more dangerous than one it rejects.* Gate offline: enumerate the `quant_algo` strings the runtime dispatches, intersect with the checkpoint's `quantized_layers` (**not** `config_groups`, and read `config.json`'s embedded `quantization_config`, **not** `hf_quant_config.json`) |
| **Q3** | Prefix caching → `CUBLAS_STATUS_INTERNAL_ERROR` then illegal memory access inside the model's `forward` | [vllm#54173](https://github.com/vllm-project/vllm/issues/54173), GB10 sm_121 + NVFP4, open. Triggered by *"prompts of differing lengths that share a prefix"* | **Directly against the official recipe's `--enable-prefix-caching`.** Start with it **off**; turn it on as a separate, tested step |
| **Q4** | `NotImplementedError` on prefix caching mode | `models/qwen4_exp/nvidia/model.py:641` — *"Qwen4Exp currently does not support 'all' prefix caching, please use `--mamba-cache-mode=align`"* | **Yes.** If you enable prefix caching, `--mamba-cache-mode align` is mandatory |
| **Q5** | First token correct, then `!!!!` forever | FlashInfer **trtllm-gen decode kernels are SM100-only and silently emit garbage on SM121** (reported against SGLang) | Not on our path if `auto` picks `FLASHINFER_CUTLASS`, which it does. The **class** is live |
| **Q6** | MoE illegal memory access on sm_121 | `flashinfer_b12x` faults — [vllm#50189](https://github.com/vllm-project/vllm/issues/50189), open since 2026-07-28 | **Do not set `--moe-backend flashinfer_b12x`.** `triton`/`cutlass` hit the SM120/121 **99 KiB (101,376 B)** shared-memory limit against a 228 KiB "Blackwell" assumption (that figure is SM100/B200 only, NVIDIA/cutlass#3144) and assert at `nvfp4_blockwise_moe.cuh:78`. Leave `--moe-backend` unset |
| **Q7** | Greedy decoding not reproducible **[SILENT for any identity-based eval]** | Three converging causes, all in the QSA indexer: [#54521](https://github.com/vllm-project/vllm/issues/54521) — **on 2× DGX Spark GB10, tp 2**, *"five byte-identical requests at `temperature=0` return five different completions — but only when the context exceeds `indexer_budget`"*; [#51782](https://github.com/vllm-project/vllm/issues/51782) — `persistent_topk` *"silently returns wrong results… drops genuine top-k candidates"* when `bin_pop > 16384` (**reported on sm103, but `persistent_topk` is called from `qwen4_exp/nvidia/ops/qsa_indexer.py`, so it is on our path**); [#54945](https://github.com/vllm-project/vllm/issues/54945) — on GB10, *"it is the FlashInfer CUTLASS NVFP4 MoE's fused finalize, which reduces the top-k expert outputs with atomics"* | **Yes.** Any eval resting on text identity at `temperature=0` is invalid here. Fixes in flight and **unreviewed**: [#55122](https://github.com/vllm-project/vllm/pull/55122) (deterministic `persistent_topk`), [#54948](https://github.com/vllm-project/vllm/pull/54948) (`VLLM_FLASHINFER_MOE_FUSED_FINALIZE`) |
| **Q8** | Long prompts hang forever | PDL used in `_build_qsa_metadata_kernel`, where `is_arch_support_pdl()` is just `major >= 9`. **Diagnosed on sm120** (2× RTX PRO 6000) in a comment on [vllm#53960](https://github.com/vllm-project/vllm/issues/53960), not on GB10. flashinfer#3170's audit table lists `device_support_pdl → True` for SM121, so the hazard is structurally present | Unconfirmed on GB10. **Test with a >8 k prompt on first boot** |
| **Q9** | Box-wide OOM (SSH included) during JIT | Unbounded ninja fan-out on a shared 128 GB pool. **flashinfer#4757** removed `SM121a` from the **aarch64** cu130/cu134 JIT-cache arch lists; it is commit `89aa0feb4` on the **v0.6.18** tag, and **vllm#54313 pinned vLLM to 0.6.18 on 2026-08-30**. The 0.6.18 release notes: *"CUDA 13 AArch64 wheels also drop native SM121a cubins (**DGX Spark keeps running via SM120 family cubins**)"* — so this is a **JIT-cost regression, not a correctness one** | `MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1`; warm the JIT cache at low `gpu-memory-utilization` first. cf. flashinfer#3634 (JIT of SM120/121a CUTLASS FP4 OOMs on first forward) |
| **Q10** | Multi-node long-context wedge | [#54629](https://github.com/vllm-project/vllm/issues/54629) (4× GB10 TP4+EP, third >100 k request never completes) and `ryangu00` (dual GB10, worker dies from ~95 k) | **Yes.** This, not fit, is the ceiling on our 262 K ambition |
| **Q11** | Malformed non-English output | [#54739](https://github.com/vllm-project/vllm/issues/54739), 2× DGX Spark tp2: *"systematically malformed Thai… The same model at the same effective QSA budget under llama.cpp is clean"* — split out of #54521 as a **separate** defect | Unknown scope. Worth a non-English smoke test |

#### Two premises I was given that are **misattributed** — correct these before quoting

- **"sglang#36806 is a retracted SM121 QSA kernel-guard fix."** **Wrong, and inverted.** #36806
  (*"fix(qsa): route exact SM120 to FlashInfer sparse decode"*, **merged 2026-08-28**) is the PR that
  **excludes** SM121 from the corrupting path: *"The existing `is_sm120_supported()` helper matches
  compute-capability major 12, including SM121/GB10. That family-wide predicate is unsafe here:
  real-weight SM121 testing reports token-0 corruption rising from 1/4 runs at 120K tokens to 4/4 at
  210K tokens when this decode path is forced."* The **retracted** item is **sglang#36649**, merged
  2026-08-27 and superseded a day later; the story resolves in **sglang#36845**, which puts SM121 on
  a narrow Triton kernel and passes NIAH at 120 k / 190 k / 210 k. **This arc is our strongest
  independent evidence of the 12.0-vs-12.1 hazard** — a decode path clean on sm120 that silently
  corrupts long-context output on sm121, fixed by replacing a family predicate with exact-capability
  checks.
- **"vllm#53960 is the PDL long-prompt hang."** **Partly.** The issue itself is a **PLE-offload
  rendezvous deadlock at TP=1** (*"The warmup forward needs n-gram/PLE rows to make progress. The
  worker that would supply them is parked in `Queue.get(block=True, timeout=None)`"*), still open,
  workaround `--distributed-executor-backend mp`. The PDL diagnosis is a *comment* on it, from sm120
  hardware. Both matter; they are not the same bug, and neither is confirmed on GB10 without offload.

### 3.3 The RadixArk load failure (and why it is not a checkpoint problem)

`RadixArk/…-NVFP4` ships an NVFP4 body with an FP8 PLE table plus one global BF16
`ngram_embedding.weight_scale`. `_get_ple_embedding_quant_method()` gated on `Fp8Config`, so with
`quant_config = modelopt_fp4` the embedding was built unquantized and load died with
`ValueError: There is no module or parameter named 'ngram_embedding.weight_scale'`. Upstream issue
[#54765](https://github.com/vllm-project/vllm/issues/54765) (reported on **SM120**, 2× RTX PRO 6000).
The field record's verdict: *"corrects the published checkpoint tables that list it as not loading on
vLLM: it loads and serves with a one-line gate change."* Check whether our nightly already carries
that widening before assuming a patch is needed.

### 3.4 On quantizing the PLE table further

`provsalt/…-NVFP4-PLE-NVFP4` (101.68 GiB) is tempting for single-Spark work. Two cautions:

- **Quality.** Field measurement of worst-shard relative error: **FP8 0.0345 vs NVFP4 0.1493 —
  4.3× worse** on the table specifically. (FP8 PLE vs BF16 measures cosine 0.999635.)
- **It buys memory, not speed.** The lookup reads 16 rows of 160 bytes per token; each row costs a
  4 KiB page either way, so halving the row does not reduce the fault count.
- **NIAH cannot gate this** — the field record reports *"thirteen consecutive 5/5 passes across every
  config today, including the worst-performing one, means it discriminates nothing."*

---

## §4 — vLLM support

### 4.1 In-tree, since when

`vllm/model_executor/models/registry.py` on `main`:

```python
"Qwen4ExpForCausalLM":              ("vllm.models.qwen4_exp", "Qwen4ExpForCausalLM"),
"Qwen4ExpForConditionalGeneration": ("vllm.models.qwen4_exp", "Qwen4ExpForConditionalGeneration"),
"Qwen4ExpMTP":                      ("vllm.models.qwen4_exp", "Qwen4ExpMTP"),
```

Landed in [PR #53896 "[Model] Support Qwen3.8-Flash-Next"](https://github.com/vllm-project/vllm/pull/53896),
**merged 2026-08-31T05:57:56Z**, 124 files, +19,425/−451, merge commit `e126687a`. (The package was
renamed `qwen3_8_flash_next` → `qwen4_exp` just before merge, so pre-merge source paths in field
write-ups no longer resolve.)

**In no release.** v0.28.0 shipped 2026-08-26 — five days early. The vLLM recipe page states
*"Minimum vLLM: 0.29.0+"* (does not exist) and *"PyPI installation not supported"*. **`main`/nightly
only.**

**Not in the docs.** `docs/models/supported_models.md` on `main` has **zero** hits for `Qwen4Exp` or
`Qwen3.8`; #53896 touched no `docs/` file.

The PR's own validation matrix:

> - Without offload: BF16, FP8, NVFP4. Platforms: **GB300, GB200, H200, and MI355X** (BF16 only).
>   Parallel configurations: **TP2, TP4, TEP4**.
> - With N-gram embedding offload: BF16 and FP8 (**NVFP4 is not currently supported**). Platform:
>   GB200. Parallel: TP2, TP4, DEP4.

**No sm_120 or sm_121 anywhere in it.** GB300/GB200 are datacenter Blackwell (sm_100/sm_103).

### 4.2 `--trust-remote-code`

**Not required.** `vllm/models/qwen4_exp/config.py` defines `Qwen4ExpConfig` /
`Qwen4ExpTextConfig` / `Qwen4ExpVisionConfig`, and `vllm/transformers_utils/config.py` registers
them:

```python
qwen4_exp="Qwen4ExpConfig",
qwen4_exp_text="Qwen4ExpTextConfig",
```

so the checkpoint's `transformers_version: "5.8.0.dev0"` does not gate us. (The docs say nothing
either way; the registry entry is the evidence.)

### 4.3 Parallelism: TP only, powers of two only

**Pipeline parallel is hard-refused**, not merely unsupported —
`vllm/model_executor/models/config.py:879`:

```python
if text_config.ple_layer_ids and parallel_config.pipeline_parallel_size > 1:
    raise NotImplementedError(
        "Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 "
        "because non-first pipeline ranks do not receive the raw input_ids "
        "it needs. Please run with PP=1.")
```

duplicated per-rank in `nvidia/model_state.py:35`, and stated in the official recipe: *"N-gram
Embedding does not initially support pipeline parallelism; use single-node TP or TEP instead."*
Two open issues argue the gate is over-broad ([#54709](https://github.com/vllm-project/vllm/issues/54709),
[#55515](https://github.com/vllm-project/vllm/issues/55515)) — for our config `ple_layer_ids = [2]`,
one layer — but today it raises.

**TP must be a power of two.** [#55517](https://github.com/vllm-project/vllm/issues/55517): TP3
fails because the **vision encoder has 16 attention heads** and `divide()` asserts. TP2 and TP4
satisfy every assert in the model, checked against `config.json`:

| assert (source) | TP2 | TP4 |
|---|---|---|
| `nvidia/qsa.py:206` — 24 Q heads divisible by TP | 12 ✓ | 6 ✓ |
| `nvidia/qsa.py:215` — 2 KV heads: divisible, else TP divisible by them | exact ✓ | replicated branch (`4 % 2 == 0`) ✓ |
| `qwen_gdn_linear_attn.py:463` — `divide(48 v-heads, tp)`, 16 k-heads | 24 / 8 ✓ | 12 / 4 ✓ |
| 512 experts, `moe_intermediate` 640 | 256 / 320 ✓ | 128 / 160 ✓ |
| `ple_layer.py:276` — `ngram_heads = (3−1)×8 = 16`, `2560 % 16 == 0` | ✓ (TP-independent) | ✓ |
| `config.py:157` — `indexer_budget / compress_ratio ∈ {512, 2048}` → 512 | ✓ | ✓ |

Other hard refusals in `qwen4_exp`: `NotImplementedError("Qwen4Exp HC does not support
sequence-parallel MoE")` (`nvidia/model.py:167,192`); no dual-batch overlap/microbatching
(`models/config.py:874`); spec-decode limited to `{"mtp","ngram","ngram_gpu"}` (`:894`).

**Expert parallel** is supported (`Qwen4ExpMixtureOfExperts(MixtureOfExperts)`, EPLB protocol) and
recommended by the official recipe on datacenter parts — but note that **both multi-Spark failures
in §2.6 involved EP**, and the field record traces UMA freezes partly to *"EP all-to-all buffers"*.
**Leave `--enable-expert-parallel` off on the first boot.**

### 4.4 PLE offload — the single-Spark route, and why to skip it

`VLLM_PLE_CPU_OFFLOAD=1` keeps the n-gram table off the GPU. It is **not on `main`**, verified:

```
repos/vllm-project/vllm/contents/vllm/v1/ple_offload   -> 404
grep PLE_ in main:vllm/envs.py                         -> zero matches
```

It lives in **[PR #53899](https://github.com/vllm-project/vllm/pull/53899)**: open as of 2026-09-06,
`mergeable_state: blocked`, +3673/−57, **no human review submitted** (38 reviewers requested, none
responded; only bot comments). No maintainer has commented on timing. It additionally requires:

- **`--cap-add=SYS_PTRACE`** in Docker (or `AmbientCapabilities=CAP_SYS_PTRACE` under systemd). The
  offload worker's `rebuild_cuda_tensor` needs `pidfd_getfd`, and `kernel.yama.ptrace_scope=1` (the
  DGX OS / Ubuntu default) refuses it between **sibling** processes. Without it the engine dies
  ~10 minutes in with only `Failed core proc(s): {}`. *(The field record's first published
  explanation — Docker seccomp — was **wrong** and was corrected upstream; it fails identically on
  bare metal.)*
- **`--kv-cache-memory-bytes`**, because KV profiling counts the offload process's memory on
  unified-memory parts (`Available KV cache memory: -11.06 GiB` at `gmu 0.80`).
- Accepting demand paging. The table is *not* fully resident: measured **major faults per token 16.0
  at c=1 falling to 3.6 at c=48** (a resident table would fault zero times), bounded at ~5 % of the
  token budget at c=1 and ~1 % at c=16. A 160-byte row costs a 4 KiB page — **26× read
  amplification** on an address pattern that is `hash(ngram) mod size`, uniform by construction.
- The author's own matrix lists **GB200, not GB10**; and rebasing onto post-#54517 `main` conflicts
  in four places.

**With two Sparks we need none of this.** SGLang #37995 documents exactly that shape — NVFP4 TP2 on
2× Spark **with PLE offload off**.

### 4.5 MTP / speculative decoding — real, but not on day one

Measured on one GB10 (NVFP4 + FP8-head checkpoint, TP1, `max-num-seqs 16`):

| arm | c=1 decode | c=1 TTFT | c=16 aggregate | c=16 TTFT |
|---|---:|---:|---:|---:|
| MTP off | 26.4 tok/s | 1.87 s | 96.6 | 9.71 s |
| MTP k=2 | **38.0 tok/s** | 1.82 s | 99.1 | **6.79 s** |

Five reasons to enable it *second*:

1. **[SILENT] The MTP corruption fix is not in our newest nightly.** With speculation configured,
   prefills after batch row 0 wrote their PLE conv state into request 0's blocks — a strided
   `state_indices` view read with unit stride, symptom *"repeated tokens such as
   `Theductductduct...`"*. Fixed by [vllm#55375](https://github.com/vllm-project/vllm/pull/55375),
   **merged 2026-09-05T14:02Z**. Our newest image is commit `2902ca17e`, dated **2026-09-05T09:56Z**;
   I checked ancestry, and `#55375 … 2902ca17e` compares as **`behind`**. The independent duplicate
   (#55467, self-withdrawn) quantifies it: *"stock 30–50 % of cells with a corrupted request…; with
   this change: 0 corrupted of 20 cells."* **MTP on `:2026090501` is a known-corrupting config.**
2. **`num_speculative_tokens = 5..8` is unreachable** — [#54552](https://github.com/vllm-project/vllm/issues/54552)
   shows the QSA ring assert (`qsa_cache.py:836`) is *one-sided*, so k=5–8 raise while **k=0–4 and
   9–12 are legal** at `block_size 848`. It is a hole, not a ceiling; k=4 and the 9–12 band appear
   never to have been tried. Fix in flight: [#54912](https://github.com/vllm-project/vllm/pull/54912).
3. **The optimum depth inverts between decode benchmarks and agent turns.** The field record
   published k=2 as optimal, then **withdrew it**: on a fixed-work agent loop k=2 was the *worst* arm
   (48.6 ms/tok) against no speculation (43.6), while n=5 was best (31.9).
4. **It costs −36 % of the KV pool at k=2** (§2.5).
5. `nvidia/…-NVFP4` **cannot load with MTP at all** today (#55496).

Structural, from the engine's own log **[SILENT]**:

```
Fused multi-step draft decode is not supported by attention backend(s)
QWEN38_FLASH_NEXT_EXP_QSA_STATE; falling back to rebuilding attention metadata between draft steps.
```

a genuine gap whose cost grows with `k`. And `SpeculativeConfig.moe_backend` — the documented way to
give an unquantized drafter its own MoE backend — **is inert on the V2 runner**, which #53896 pins
this model to: the engine accepts it, echoes it in the config dump, and ignores it.

**Never combine MTP with `--async-scheduling`** **[SILENT]**: `_prepare_ngram_context` reads the CPU
token mirror while it still holds speculation's `-1` placeholders, giving a wrong n-gram context.

### 4.6 Flag table

**[SILENT]** = wrong behaviour without an error.

| flag / env | value | why |
|---|---|---|
| `--enable-auto-tool-choice --tool-call-parser qwen3_xml` | **required** | without it every `tools` request returns **HTTP 400**. Not `qwen3_coder` |
| `--reasoning-parser qwen3` | **required** | else the reasoning trace and a stray `</think>` leak into `content` **[SILENT]** |
| `--default-chat-template-kwargs '{"reasoning_effort":"medium"}'` | strongly recommended | template default is `xhigh`; at `xhigh` the model reasons past generous caps and returns **empty content** **[SILENT]** |
| `--max-model-len` | **32768 to start**, ≥32768 always | 8192 *cannot hold the model's own reasoning* — one code task emitted 31,115 characters of thinking before 12,931 of content. And do **not** open at 262 K: §2.6 Q10 |
| `--max-num-seqs` | **16** on a Spark | GDN state is per-sequence (§2.5). The official 256 is a datacenter value |
| `--max-num-batched-tokens` / `--enable-chunked-prefill` | 4096 / on | field default; 8192 was tried and reversed at n=3 |
| `--enable-prefix-caching` | **off for the first boot** | #54173 (§3.2 Q3). When you do enable it, **`--mamba-cache-mode align` is mandatory** (Q4) |
| `--no-enable-flashinfer-autotune` | on | in every official config |
| `--moe-backend` | **unset** | `auto` → `FLASHINFER_CUTLASS`. `flashinfer_b12x` faults on sm_121; `triton`/`cutlass` overflow the 99 KiB SMEM budget. (`--moe-backend triton` in the official H100/H200 recipes is a **Hopper** requirement) |
| `--kv-cache-dtype` | **`auto`/`bfloat16` only** | `common/qsa_cache.py` declares `supported_kv_cache_dtypes = ["auto", "bfloat16"]` and enforces it at config time. FP8/NVFP4 KV are open ([#54426](https://github.com/vllm-project/vllm/issues/54426), [#54846](https://github.com/vllm-project/vllm/pull/54846), [#55557](https://github.com/vllm-project/vllm/pull/55557)). Patched-fp8 field result: **×1.72 pool, +2.6 % decode** — buys admission, not latency, because QSA is sparse. **NVFP4 KV is closed as a lever** — three independent GB10 measurements, and it fails silently |
| `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | worth trying second | measured on GB10 vs PIECEWISE: −2 % ms/tok, +4 % deep decode, **+17 % KV pool**; 36 s → 142 s startup. `FULL` is strictly worse. n=1 per arm — replicate |
| `--enable-expert-parallel` | **off** | §4.3 |
| `--distributed-executor-backend mp` | set | TP=1 uniproc selection hangs (#53960); ours is TP2 anyway |
| pipeline parallel | **impossible** | §4.3 |
| `VLLM_USE_DEEP_GEMM=0` | set | Q1 |
| `VLLM_GDN_DECODE_KERNEL=triton` | set | the default CUDA GDN decode kernel deterministically hangs the engine at c≈32 with FP8 GDN projections — **no error, requests just stall** |
| `CUTE_DSL_ARCH=sm_121a` | set | required for the FlashInfer CuteDSL path on GB10 |
| `MAX_JOBS=2`, `FLASHINFER_NVCC_THREADS=1` | set | Q9 |

### 4.7 Measurement hygiene, inherited

- **Noise floor: 6.9 % for decode, ±20 % for prefill** (1,633–2,367 tok/s within one config). Nothing
  under ~10 % is callable from one run.
- `usage.prompt_tokens_details.cached_tokens` is **inert** (0 on confirmed hits) — read
  `vllm:prefix_cache_hits_total` from `/metrics`. And prefix caching does not hit until the **second**
  repetition, so a two-request probe measures a defect that is not there.
- Detect empty content by `finish_reason: "length"`, never by character count — that let a
  determinism probe call five empty strings "identical", twice.
- **Accept-length pinned at maximum is a corruption signature, not health** (one case read 3.00/3
  while GSM8K scored 0/10).
- Verify downloads with HF's `lfs.sha256`, **never file sizes** — `aria2` preallocates, so a
  byte-corrupt shard is size-correct and produces *fluent garbage invariant to every configuration
  you change*.

---

## §5 — Container

### 5.1 We already have one

All read from registry image configs.

| image | vLLM | arch list | entrypoint | has #53896? |
|---|---|---|---|---|
| `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2026081501` *(current fleet default)* | `0.27.2rc1.dev113+g5cecfc013.d20260815` | `12.1a` | `nvidia_entrypoint.sh` | **no** — 16 days early |
| **`…dgx-vllm-eugr-nightly:2026090501`** | **`0.28.1rc1.dev441+g2902ca17e.d20260905`** | **`12.1a`** | `nvidia_entrypoint.sh` | **yes — 309 ahead of `e126687a`, 0 behind** |
| `vllm/vllm-openai:qwen38-flash-next-arm64-cu130` (upstream day-0) | preview build, 2026-08-26 | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` — **no 12.1** | **`["vllm","serve"]`** | preview of #53896+#53899 |

**`:2026090501` is the right base**, on three counts:

1. Built with `TORCH_CUDA_ARCH_LIST=12.1a` — **native sm_121a cubins**, which upstream's
   `qwen38-flash-next` image is not (its list stops at 12.0).
2. Entrypoint is `/opt/nvidia/nvidia_entrypoint.sh`, so the documented dgxrun trap
   (`ENTRYPOINT ["vllm","serve"]` → container dies instantly) **does not apply** and no
   entrypoint-clearing overlay is needed — unlike the GLM-5.3-Flash build, which had to add one.
3. It is vLLM `main` from 2026-09-05, carrying the post-merge fix train.

Fixes verified **by ancestry**, not by date:

| PR | merged | in `:2026090501`? |
|---|---|---|
| #53896 Support Qwen3.8-Flash-Next | 08-31 | ✅ |
| #54513 separate QSA indexer prefill/decode | 09-02 | ✅ |
| #54517 fuse Qwen4Exp PLE kernels | 09-02 | ✅ |
| #54722 validate FP8 PLE weight scale | 09-02 | ✅ |
| #54813 Rust frontend Qwen4-exp multimodal | 09-01 | ✅ |
| #54882 fix FP8 PLE loading in mixed ModelOpt checkpoints | 09-03 | ✅ |
| #55054 optimize PLE MTP metadata transfers | 09-03 | ✅ |
| #54915 compact indexer logits workspace | 09-04 | ✅ |
| #54873 improve QSA sparse GQA for prefill | 09-04 | ✅ |
| #54687 reuse HC combine-norm for MTP input | 09-04 | ✅ |
| **#55375 PLE conv state-index strides (MTP corruption)** | **09-05 14:02Z** | ❌ **behind** |
| #53899 PLE CPU offload | not merged | ❌ |

So: **serve on `:2026090501` with MTP off**; pick up a nightly ≥ `2026090601` before enabling
speculation. (`2026090601` did not exist when I checked — HTTP 404.)

### 5.2 If you ever do need to build

A Python-only overlay is **not** sufficient here. #53896 changed a **C++ op signature** in
`csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu`:

```diff
-    double scale, double norm_eps);
+    double scale, double norm_eps, const std::string& output_gate_activation);
```

A prebuilt wheel registers the *old* schema. Worse, the new default is wrong for this model —
`output_gate_type: sigmoid` in the config, and silently accepting the `silu` default computes the
**wrong activation** rather than failing. The op is gated on `num_spec_decodes > 0`, so it is
unreachable without speculation — another reason the first boot should be MTP-off. The remaining
new kernels (hyper-connection, QSA) are Triton and JIT at runtime.

If you pin FlashInfer yourself: **0.6.17 keeps prebuilt aarch64 sm_121a cubins; 0.6.18 does not**
(flashinfer#4757, on the v0.6.18 tag; vllm#54313 pinned vLLM to 0.6.18 on 2026-08-30). Per the
0.6.18 release notes, DGX Spark *"keeps running via SM120 family cubins"* — so the cost is
first-use JIT, not breakage, but on a shared 128 GB pool that JIT is exactly what has taken a box
down before.

---

## §6 — Verdict and a starting configuration

### 6.1 Verdict

**Servable with work — roughly a day, not a project.** Nothing structural blocks it:

- in vLLM in-tree since 2026-08-31;
- an arm64 sm_121a image containing it already sits in the registry we already pull from;
- the n-gram table shards across TP, so two Sparks are enough and four are comfortable;
- multi-Spark TP2 and TP4 of this exact model are **already reported working** by others;
- no `--trust-remote-code`, no source build, no entrypoint overlay.

**The single biggest blocker is correctness in the QSA indexer on sm_121, not fit.** Three open
issues converge on it — greedy non-determinism above `indexer_budget` on 2× Spark (#54521),
`persistent_topk` silently returning wrong values (#51782), and an atomics-based MoE finalize
(#54945) — with all three fixes unreviewed. Downstream of that sits the SGLang arc (§3.2), which is
a *documented* case of a path that is clean on sm_120 and silently corrupts long context on sm_121.

**Second: multi-node long-context stability.** 4× GB10 TP4+EP wedges around the third >100 K
request (#54629); a dual GB10 dies from ~95 K (ryangu00); 2× TP2+EP starves decode for 3–7 minutes
under long prefills (#54919). Both wedges involved expert parallel. **Start at TP2 without EP, at
32–64 K, and walk the window up.**

**Third: everything about MTP.** The one fix that matters landed four hours after our newest nightly
was cut.

**Not a blocker, contrary to the framing I started with:** whether it fits. TP2-FP8 at 86.4 GiB/rank
leaves ~19 GiB of headroom per node, and a shipped dual-Spark deployment already reports 62.5 GB/node
for the NVFP4 build.

### 6.2 A first configuration

Not a recipe yet — deliberately. It is the shape a recipe should take, with reasons attached.

```
model:     Qwen/Qwen3.8-Flash-Next-FP8          # 172.78 GiB -> 86.4 GiB/rank at TP2
container: ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2026090501
nodes:     2 Sparks, tensor_parallel: 2

vllm serve {model} \
  --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --no-enable-flashinfer-autotune \
  --gpu-memory-utilization 0.87 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --limit-mm-per-prompt '{"image":8}'
# NO --speculative-config      (§4.5 — #55375 is not in this image)
# NO --enable-prefix-caching   (§3.2 Q3; when you add it, add --mamba-cache-mode align)
# NO --enable-expert-parallel  (both multi-Spark wedges involved EP)
# NO --kv-cache-dtype          (QSA backend accepts auto/bfloat16 only)
# NO --moe-backend             (auto -> FLASHINFER_CUTLASS)

env:
  VLLM_USE_DEEP_GEMM: "0"
  VLLM_GDN_DECODE_KERNEL: "triton"
  CUTE_DSL_ARCH: "sm_121a"
  MAX_JOBS: "2"
  FLASHINFER_NVCC_THREADS: "1"
```

`gpu_memory_utilization: 0.87`, not 0.90: this is a **multimodal** model, and the GLM-5.3-Flash
recipe in this repo records the vision-encoder warmup allocating **outside** the profiled budget and
SIGKILLing a worker at 0.89 with 0.24 GiB of slack. Same failure shape.

`--max-model-len 32768` on the first boot, not 262144. The window is affordable
(§2.5) but multi-node sm_121 has two documented wedges between 95 K and 100 K (§2.6). Raise it
deliberately, with a needle probe at each step.

**First-boot checklist — capabilities before speed.** Every item has a documented silent failure
behind it:

1. `sha256sum -c` against HF's `lfs.sha256`, **never file sizes**.
2. Offline, before serving: enumerate the `quant_algo` values the runtime dispatches and intersect
   with the checkpoint's `quantized_layers` (§3.2 Q2). This is a ten-second check that prevents a
   day-long hunt.
3. A `tools`-bearing request must return **200, not 400**.
4. A generation long enough to clear the thinking block, asserting `completion_tokens > 0` and
   checking `finish_reason` — an all-empty comparison is not a comparison.
5. A **>8 k** prompt, to confirm the PDL hazard (Q8) does not bite this build.
6. A non-English prompt (Q11).
7. Coherence across a handful of prompts — but **not** by text identity at `temperature=0`; greedy
   is not reproducible here (Q7). Use needle-in-a-haystack retrieval.
8. Read `/metrics` (`vllm:prefix_cache_hits_total`), not `usage`.

### 6.3 What I could NOT determine

- **Whether it boots on *our* Sparks.** No hardware was available for this note. Upstream validated
  GB300/GB200/H200/H100/MI355X; the sm_121 evidence is all third-party, and the multi-Spark
  third-party evidence is mostly bug reports.
- **Multi-node TP throughput over our RoCE fabric.** The only multi-Spark throughput figure anywhere
  is MiaAI-Lab's 64 tok/s on **SGLang** TP2. No vLLM multi-Spark tok/s exists. The model is 6 B
  active, so decode should be latency- rather than bandwidth-bound at TP2 `[inference]`, but that is
  arithmetic, not data.
- **Whether `Qwen/Qwen3.8-Flash-Next-FP8` contains blockwise (`FP8_PB_WO`) layers**, which decides
  whether Q1/Q2 bite. I did not download and parse its `hf_quant_config.json` / `config.json`
  `quantized_layers`. §6.2 step 2 is the check.
- **Whether our nightly already carries the widened PLE quant-config gate** (§3.3), i.e. whether an
  NVFP4-body/FP8-table checkpoint loads unpatched on `:2026090501`.
- **Which FlashInfer version `:2026090501` ships.** The image label records the FlashInfer *git hash*
  (`18e5811d…`), not a version — and 0.6.17 vs 0.6.18 decides whether GB10 gets prebuilt sm_121a
  cubins (Q9). Resolve before any JIT-heavy first launch.
- **Per-rank vision-tower and warmup memory.** The tower is small (27 layers, hidden 1152), but the
  GLM-5.3-Flash precedent says the warmup allocation is what kills a boot, and no figure exists for
  this model. No first-party image test on a Spark exists anywhere in the field record either — no
  image counts, no per-image latency.
- **Real quality on our workloads.** Qwen's numbers (DeepSWE 1.1 58.7, SWE-bench Pro 62.5,
  SWE-bench Multilingual 81.0, CoWorkBench 73.9 — against 42.2 / 61.7 / 73.8 / 70.7 for
  Qwen3.8-27B) are the vendor's. And note the field warning that `gsm8k_metrics.json` /
  `aime26_metrics.json` are **byte-identical across four different HF quant repos**, latency
  included to ten decimal places — *treat any metric you did not generate as provenance, not
  evidence*.

---

## Sources

- **Qwen (primary):** [`Qwen/Qwen3.8-Flash-Next` model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
  [`config.json`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json),
  `generation_config.json`, `chat_template.jinja`; file manifests via
  `huggingface.co/api/models/<repo>?blobs=true`; `Qwen/Qwen3.8-27B` and `Qwen/Qwen3.8-2.4T-A95B`
  configs for comparison.
- **vLLM source (`main`):** `model_executor/models/registry.py`,
  [`models/qwen4_exp/common/ple.py`](https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/common/ple.py),
  `models/qwen4_exp/config.py`, `models/qwen4_exp/common/qsa_cache.py`,
  `models/qwen4_exp/nvidia/{qsa.py,model.py,ple_layer.py,ops/ple.py}`,
  `model_executor/models/config.py`, `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`,
  `config/multimodal.py`, `transformers_utils/config.py`, `envs.py`, `docker/Dockerfile`,
  `docs/models/supported_models.md`.
- **vLLM issues/PRs:** #50189, #51782, #51884, #53896, #53899, #53908, #53960, #54125, #54126,
  #54129, #54173, #54313, #54371, #54426, #54521, #54552, #54600, #54629, #54709, #54739, #54765,
  #54846, #54873, #54882, #54912, #54915, #54919, #54928, #54945, #54948, #55054, #55122, #55180,
  #55375, #55394, #55430, #55467, #55496, #55515, #55517, #55557, #55572.
- **Other upstreams:** flashinfer-ai/flashinfer #3170 (DGX Spark SM121 support audit), #4757, #3634;
  sgl-project/sglang #36497, #36545, #36556, #36558, #36649, #36797, #36806, #36845, #37995.
- [vLLM recipe: Qwen/Qwen3.8-Flash-Next](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) and
  `vllm-project/recipes` → `models/Qwen/Qwen3.8-Flash-Next.yaml`.
- **Registry manifests:** `vllm/vllm-openai:{qwen38-flash-next,qwen38-flash-next-arm64-cu130,qwen38-arm64-cu130,glm53-flash-arm64-cu130,v0.28.0}`;
  `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:{2026081501,2026090401,2026090501}`.
- **First-hand GB10 field record:** [`jschmied/qwen38-flash-next-gb10`](https://github.com/jschmied/qwen38-flash-next-gb10)
  (Apache-2.0) — a working record of serving this model on one DGX Spark (sm_121, 128 GB unified,
  aarch64), with per-claim data files, an explicit list of its own withdrawn claims, and the author
  of vllm#55394/#55430. Community reports it links and checks: MiaAI-Lab (single and dual Spark),
  blazux, provsalt, primitive-ai, 0xBakeer, DJLougen, dolf3131, mratsim, and others named inline.
- **This repo:** `recipes/dgxrun/qwen3.8-27b-nvfp4.yaml`, `recipes/dgxrun/qwen3.8-27b-bf16.yaml`,
  `recipes/dgxrun/zai-glm-5.3-flash-libertai-nvfp4-2x.yaml`, `scripts/build-glm53-flash-image.sh`.
