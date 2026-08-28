# Qwen 3.8 on the Spark fleet — versions, quantizations, and what it takes to serve them

**Research date:** 2026-08-15. **Hardware results added 2026-08-28 (§11).** Sources for
§§1–10 are primary throughout: HuggingFace raw `config.json` and the HF model API, the
vLLM / SGLang / llama.cpp repos via the GitHub API, and the `spark-arena` registry repos.
**§§1–10 contain no results measured on this fleet** — any such claim there is attributed to a
third party. **§11 is the opposite**: everything in it was measured on dgx-spark-01.

**Scope:** `Qwen/Qwen3.8-27B`. The 2.4T-A95B "Max" is out of reach and is covered in one line
(§2).

---

## TL;DR

1. **Qwen 3.8 needed no new vLLM support.** `Qwen/Qwen3.8-27B` declares
   `architectures: ["Qwen3_5ForConditionalGeneration"]` — the *same* class Qwen3.5 and Qwen3.6
   declare, registered since **vLLM v0.17.0 (2026-03-07)**. Its `config.json` is
   field-for-field identical to Qwen3.6-27B's. This is a weights refresh on the Qwen3.5
   architecture. Both of our container images already carry it.
2. **An official sparkrun recipe already exists** — `@official/qwen3.8-27b-fp8-mtp-vllm`,
   committed upstream 2026-08-14. Our local catalog lacks it only because the registry clone is
   stale. **`sparkrun registry update` is the entire fix.** No sparkrun upgrade needed.
3. **For benchmarking against Muse Glimmer, use BF16, not that recipe.** 51.7 GiB fits one
   Spark, which means we can measure reference weights with no quantization confound — the same
   argument `muse-glimmer-30b.yaml` already makes for itself. That needs a small dgxrun recipe
   (§6).
4. **GPQA-Diamond is the one clean comparison** — Qwen3.8-27B **89.2** vs Muse Glimmer **83.5**.
   The SWE-bench comparison is a trap (§5).
5. **MTP ships inside the checkpoint**, so unlike Muse Glimmer's DFlash there is no separate
   drafter to stage on NFS.

---

## 1. What shipped

Four first-party checkpoints ([HF API](https://huggingface.co/api/models?author=Qwen), queried
2026-08-15):

| Repo | Arch string | Size | Modality |
|---|---|---|---|
| [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | `Qwen3_5ForConditionalGeneration` | **51.7 GiB** BF16 | vision-language |
| [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `Qwen3_5ForConditionalGeneration` | **28.7 GiB** | vision-language |
| [`Qwen/Qwen3.8-2.4T-A95B`](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | `Qwen3_5MoeForCausalLM` | 4556 GiB | text-only |
| `Qwen/Qwen3.8-2.4T-A95B-FP8` | `Qwen3_5MoeForCausalLM` | 2325 GiB | text-only |

Sizes are exact, summed from HF blob metadata, not estimated.

**Timeline** (inferred — see §8): Max announced ~2026-08-03, 27B repo created 08-05, Max weights
08-09, 27B card finalized 08-12→13, FP8 published 08-13. There is **no open-weight Qwen 3.7** —
`Qwen3.7-Plus`/`-Max` appear only as comparison columns, i.e. API-only. Qwen3.8 is the direct
open-weight successor to Qwen3.6.

### The 27B lineup is exactly two repos

No Instruct/Thinking split (thinking is a *runtime* toggle via `reasoning_effort`), no Base, no
Coder, **no text-only variant**, and no separate drafter repo. The vision tower is fused into
the same checkpoint (333 `model.visual.*` tensors). License is **Apache 2.0**.

### Two corrections to widely-repeated secondary claims

- **The open-weight Max is text-only.** Its card: *"Qwen3.8-2.4T-A95B is a text-only model…
  Multimodal inputs are not supported."* Vision belongs to the hosted API product. **The 27B is
  the multimodal one, not the Max.**
- **Native context is 262,144, not 1M.** `max_position_embeddings: 262144`, stock
  `rope_type: "default"`. 1M requires an explicit YaRN `--hf-overrides`, and Qwen warns that
  because every open framework implements *static* YaRN, enabling it *"potentially impacts
  performance on shorter texts."*

---

## 2. The Max does not fit

`Qwen3.8-2.4T-A95B` is **4556 GiB at BF16 / 2325 GiB at FP8** against a ~398 GiB usable pool at
tp=4 — roughly 6× over even at FP8. The lowest unpruned quant is a 1-bit GGUF at 370 GiB, which
leaves no room for KV cache. Only heavily *pruned* derivatives fit, and those are no longer the
model Qwen benchmarked. Out of scope.

---

## 3. The decisive architecture fact

Diffing [Qwen3.8-27B `config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)
against [Qwen3.6-27B's](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json)
field-by-field: **structurally identical.** 64 layers, hidden 5120, head_dim 256,
`full_attention_interval: 4`, vocab 248320, mrope `[11,11,10]`, `mtp_num_hidden_layers: 1`,
`vision_config` present.

Hybrid attention: **only 16 of 64 layers are full attention**; the other 48 are linear
(gated-delta / GDN, `linear_conv_kernel_dim: 4`). That matters for KV sizing — a 262K window is
far cheaper here than the number suggests, since the linear layers carry a constant-size
recurrent state instead of a growing cache.

**Registry evidence.** `Qwen3_5ForConditionalGeneration` is at
[`registry.py:592`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py),
added by [vllm#34110](https://github.com/vllm-project/vllm/pull/34110) **merged 2026-02-09**,
and verified present in `registry.py` at every tag from **v0.17.0** through v0.27.1. This is
**not main-only** — the "week-old release, must pin a commit" scenario does not apply.

The multimodal processor is **native in-tree** (`Qwen3_5ProcessingInfo` in
[`qwen3_5.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5.py)),
so **no `--trust-remote-code`** — the official 3.8 recipe drops the flag the 3.6 recipe carried.

**Parsers exist, exact names confirmed in source:** `--reasoning-parser qwen3`
(`vllm/reasoning/__init__.py:127`), `--tool-call-parser qwen3_coder` (or `qwen3_xml`) — note the
tool parsers moved to `vllm/tool_parsers/`, no longer `entrypoints/openai/tool_parsers/`. No
parser lag, because these are the Qwen3-family parsers and predate 3.8.

### The `transformers_version: 5.8.0.dev0` stamp is a non-issue

It reads like a requirement on an unreleased library. It isn't — it is merely what Qwen had
installed when saving. transformers **5.8.1 shipped 2026-05-13**, stable is 5.15.0, and vLLM
pins `>= 5.5.3`. Direct proof it round-trips: `unsloth/Qwen3.8-27B-NVFP4` re-saves the same
architecture stamping `transformers_version: 5.14.1`. The one real transformers-5.x item,
[vllm#50704](https://github.com/vllm-project/vllm/pull/50704), merged 2026-08-02.

---

## 4. Quantizations, and why BF16 is the right pick for us

Everything fits one Spark. Selected rows:

| Repo | Method | GiB | Downloads | Note |
|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | **BF16** | **51.7** | — | reference weights, no confound |
| `Qwen/Qwen3.8-27B-FP8` | blockwise FP8 `[128,128]` | 28.7 | — | vision tower left unquantized |
| `unsloth/Qwen3.8-27B-NVFP4` | NVFP4 | 21.8 | 90.9k | most-used NVFP4 |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | AWQ W4A16 | 19.6 | **3** | created 2026-08-15 — unvalidated |
| `dbirks/Qwen3.8-27B-W4A16-AutoRound` | AutoRound int4 | 18.1 | — | smallest safetensors int4 |
| `unsloth/Qwen3.8-27B-GGUF` | GGUF IQ2_XXS→BF16 | 8.4–50.9 | 868k | most-downloaded 3.8 artifact anywhere |

**Publishers we normally rely on:** ❌ no `Intel`, ❌ no `nvidia` first-party NVFP4, ❌ no
`QuantTrio`, ❌ no `CosmicRaisins`, ❌ **no first-party int4 at all**. (Precedent: Qwen shipped
`Qwen3.5-*-GPTQ-Int4` ~2 months after the 3.5 base release, so a first-party int4 may yet come.)
✅ `cyankiwi` and `unsloth` are present.

**All community repacks keep the MTP head** — I checked each `model.safetensors.index.json`:
BF16 15 `mtp.*` keys, FP8 22, unsloth NVFP4 15, cyankiwi AWQ-INT4 15. So the `-MTP` suffix some
repos carry is redundant labelling, not a distinct capability.

**sm_121 kernel reality:** NVFP4 is **not** capability-gated out —
`compressed_tensors_w4a4_nvfp4.py` returns `get_min_capability → 75`, and W4A16_NVFP4 falls back
to FP4 Marlin off SM100. **MXFP4 does not load on NVIDIA** at all (missing linear-method
support) — treat all `*-MXFP4` repos as blocked.

### The apples-to-apples argument

`muse-glimmer-30b.yaml` states its own rationale: *"BF16 fits one Spark, so we measure the
REFERENCE weights: no quantization confound, unlike GLM-5.2 (INT4) or DeepSeek V4 Flash (fp8).
A gap against Meta's numbers is therefore harness or serving, never 'our quantization is
worse'."*

That argument applies unchanged to Qwen3.8-27B, and we get to keep it: **51.7 GiB fits one
Spark outright**, leaving ~47 GiB for KV and capture at gmu 0.88. Benchmarking the FP8 or NVFP4
checkpoint against a BF16 Muse Glimmer would reintroduce exactly the confound the Glimmer recipe
was written to remove — and would do so in the direction that flatters Glimmer, making any Qwen
win understated and any Qwen loss unattributable.

BF16 also sidesteps every quantization-kernel question on sm_121 at once: the blockwise-FP8
DeepGEMM family-12 gate, NVFP4 Marlin fallback, MXFP4. For a first boot that is the lowest-risk
path as well as the most rigorous one.

**Use FP8 for serving; use BF16 for measuring.** They are different jobs.

---

## 5. Benchmarks — and the comparability trap

From the [27B model card](https://huggingface.co/Qwen/Qwen3.8-27B). Muse Glimmer-30B appears as
a baseline column in Qwen's own table, listed at exactly its published 83.5 GPQA-D — a good sign
their harness is comparable.

| Benchmark | Qwen3.8-27B | Qwen3.6-27B | Muse Glimmer-30B |
|---|---|---|---|
| **GPQA Diamond** | **89.2** | 87.8 | **83.5** |
| **LiveCodeBench v6** | 90.3 | 83.9 | — |
| SWE-bench **Pro** | 61.7 | 53.5 | 51.2 |
| Terminal Bench 2.1 | 73.0 | 63.4 | 51.7 |
| IFBench | 79.5 | 69.1 | 77.0 |
| HLE | 30.8 | 24.0 | 22.0 |

Qwen3.8-27B beats Muse Glimmer on every benchmark where both are scored.

**What we can actually reproduce:** GPQA-Diamond, LiveCodeBench v6, HLE, IFBench — static, open
harnesses. **What we cannot:** every coding number of note (SWE-bench Pro, Terminal Bench,
DeepSWE, NL2Repo, FrontierSWE) runs **through the Claude Code harness** at temp=1.0/top_p=0.95
with multi-hour timeouts (5h Terminal Bench, 8h QwenSWEBench, 12h PaperBench). `QwenSWEBench`,
`CoWorkBench`, `JobBench` are in-house and unreleased.

### Three traps

- **SWE-bench Pro ≠ SWE-bench Verified.** Muse Glimmer publishes **76.0 Verified**; Qwen
  publishes **61.7 Pro** and lists Glimmer at **51.2** on Pro. Setting 61.7 beside 76.0 is
  meaningless; like-for-like on Pro is 61.7 vs 51.2, which *reverses* the ranking. Worse, Qwen
  states they **corrected "problematic tasks" in SWE-bench Pro and re-evaluated all baselines**
  on their refined variant — so their Pro numbers don't match anyone else's Pro numbers either.
- **No official IFEval and no AIME.** Qwen publishes IFBench (79.5), a different and harder
  instruction-following benchmark. There is **nothing to place beside GLM-5.2's 0.83 IFEval** —
  we'd have to run IFEval ourselves.
- **GPQA-Diamond is the clean number**, and this fleet already runs that harness.

---

## 6. Classification: served / small recipe / blocked

| Variant | Class | Reason | Fork from |
|---|---|---|---|
| `Qwen3.8-27B-FP8` | **(a) already served** | `@official/qwen3.8-27b-fp8-mtp-vllm` upstream 2026-08-14. Run `sparkrun registry update` | — use directly |
| **`Qwen3.8-27B` BF16** | **(b) small dgxrun recipe** | 51.7 GiB on one Spark; arch supported since v0.17.0; no upstream recipe exists; sidesteps all quant-kernel risk | **`muse-glimmer-30b.yaml`** |
| `unsloth/Qwen3.8-27B-NVFP4` | (b), with risk | Kernel path clear (Marlin fallback), keeps MTP. But sglang#34895 reports a dropped `lm_head weight_scale` on *this exact repo* → degenerate repetition | muse-glimmer-30b.yaml |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | (b), unvalidated | Well-trodden dense W4A16→Marlin path, keeps MTP, but 3 downloads. Our prior Int4-Int8Mix stall was **MoE**-specific and later re-attributed to earlyoom, so it doesn't indict dense W4A16 | muse-glimmer-30b.yaml |
| GGUF / llama.cpp | (b) | Works as `qwen35` hybrid. Known: MTP CUDA lockups under `--split-mode tensor`, chat-template friction with agentic harnesses | `@sparkrun-transitional/*-llama-cpp` |
| SGLang FP8 | (b), custom image | Day-0 support, **validated on one GB10** (below). Needs `lmsysorg/sglang:qwen38-27b` | `@sparkrun-transitional/qwen3.5-27b-fp8-sglang` |
| MXFP4 | **(c) blocked** | Does not load on NVIDIA; vllm#52347 is an open feature request, not a fix | — |
| AMD Quark INT4 | **(c) blocked** | vllm#52454, open, filed 2026-08-15 | — |
| DSpark drafting | **(c) blocked** | [vllm#52197](https://github.com/vllm-project/vllm/pull/52197) **open, unmerged**. Would need a `--vllm-ref` pin | — |
| DFlash drafting | **(c)** | No Qwen3.8-27B DFlash drafter exists on HF at all. Use in-checkpoint MTP | — |

### The one third-party GB10 datapoint

[sglang#34872](https://github.com/sgl-project/sglang/issues/34872), closed 2026-08-14:
`Qwen3.8-27B-FP8` validated on **one DGX Spark / SM121** at `--mem-fraction-static 0.70`,
FlashInfer full-attention, Triton hybrid-GDN kernels, chunked-prefill 8192, native MTP 3 steps.
72/72 requests, max prompt 32,768, concurrency 8. Notably the reporter flags that the cookbook's
*generated* DGX Spark cells (`0.95 --disable-prefill-cuda-graph`) did **not** match what worked,
and that 0.75 blew a 512 MiB swap guard during CUDA-graph capture.

---

## 7. Container images — a correction to our own recipe comments

Per the [spark-arena/dgx-vllm README](https://github.com/spark-arena/dgx-vllm), **`-tf5` is now a
deprecated alias of `-nightly` — identical digest.** The upstream `--tf5` build flag no longer
produces a separate lineage. The comment block in `recipes/dgxrun/muse-glimmer-30b.yaml` treats
tf5 as a distinct thing; that is now only historically true.

| Image | vLLM | `Qwen3_5*` registered? |
|---|---|---|
| `dgx-vllm-eugr-nightly` (= `-tf5`) | `0.27.2rc1.dev113+g5cecfc013` — real vLLM **main** | ✅ |
| `dgx-vllm-eugr-nightly-b12x` | `0.1.dev20003+gad848fc41` — diverged branch (215 ahead / 216 behind) | ✅ (8 entries, checked at that SHA) |

**Both can load Qwen3.8-27B.** b12x is unnecessary — the 27B is dense, so none of its sparse-MLA
/ MoE / DeepSeek-indexer kernels apply. Use the plain nightly, and **pin a dated tag rather than
`:latest`**: [vllm#52147](https://github.com/vllm-project/vllm/pull/52147) (merged 2026-08-13)
touched `qwen3_5.py`, and a `ParallelLMHead` regression
([#52434](https://github.com/vllm-project/vllm/issues/52434)) was filed against main 2026-08-15.

---

## 8. Live risks for a single-Spark deploy

| Issue | State | Applies? |
|---|---|---|
| [#52244](https://github.com/vllm-project/vllm/pull/52244) hybrid-GDN prefix-cache hits collapse to **zero** under MTP for prompts a multiple of the hash unit | **open** | **Yes** — perf only, not correctness, but `@official` enables both prefix-caching and MTP |
| [#51884](https://github.com/vllm-project/vllm/issues/51884) blockwise FP8 fails in `process_weights_after_loading` on capability family 12 (DeepGEMM "Unknown SF transformation") | **open** | FP8 only. Workaround: `VLLM_USE_DEEP_GEMM=0` + `VLLM_MOE_USE_DEEP_GEMM=0`. Counter-evidence: `@official` ships no override and the identically-schemed Qwen3.6-27B-FP8 recipe already works here |
| [#51987](https://github.com/vllm-project/vllm/issues/51987) revert FlashInfer XQA decode on SM12x | open | Watch — `@official` sets `--attention-backend flashinfer` |
| [#52030](https://github.com/vllm-project/vllm/pull/52030) fix packed **GDN** decode launch | merged 2026-08-13 | Directly on this model's linear-attention path — already in the nightly |
| [#51921](https://github.com/vllm-project/vllm/issues/51921) engine stalls after ~1 min idle on 4-node TP=4 GB10 | open | **No** — TP=1. (This report appears to be from *this very fleet*: GLM-5.2, dual-rail RoCEv2) |
| [#52291](https://github.com/vllm-project/vllm/issues/52291) FlashInfer autotune deadlocks multi-node TP startup | open | **No** — TP=1 |

### Text-only serving is a first-class config path

The vision tower is fused and no text-only repo exists, but vLLM can decline to use it.
In [`vllm/config/multimodal.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/multimodal.py),
`get_limit_per_prompt()` returns 0 for **every** modality when `language_model_only` is set
(lines 487–493); `skip_mm_profiling` (line 217) skips the encoder's profiling pass. A
third-party repro (vllm#52454) passes `--language-model-only --skip-mm-profiling
--limit-mm-per-prompt '{"image": 0}'` together on a Qwen3.8-27B checkpoint. For agentic coding
this skips the ViT and sidesteps the whole vision bug class. *Unconfirmed:* the argparse
spelling at our pinned image tag — check `vllm serve --help` in the container first.

---

## 9. Recommended next steps

1. **`sparkrun registry update`** — picks up `@official/qwen3.8-27b-fp8-mtp-vllm` for serving.
   Zero work, no sparkrun upgrade (v0.3.4 latest; our `>=0.3.3` floor is fine).
2. **Add `recipes/dgxrun/qwen3.8-27b.yaml`** for the BF16 reference weights — the
   apples-to-apples benchmark target against Muse Glimmer. A full sketch forked from
   `muse-glimmer-30b.yaml`, with every flag justified and unknowns marked TODO rather than
   guessed, is in
   `scratchpad/qwen38-serving-findings.md` §8.
3. **Run GPQA-Diamond first** — the one directly comparable number (89.2 claimed vs Glimmer's
   83.5, which our fleet has the harness for).
4. **Fix the stale `muse-glimmer-30b.yaml` comment** about tf5 being a distinct image lineage.

---

## 10. What could not be confirmed

1. **The official blog post body and exact announcement date.** `qwen.ai/blog?id=qwen3.8` is a
   JS SPA with no server-rendered content; `qwenlm.github.io/blog/` renders **stale** (newest
   post Sept 2025); all dated permalinks 404; the atom feed 404s. The ~2026-08-03 date is
   inferred from the Max card's *"as of August 3, 2026"* leaderboard footnote plus HF repo
   timestamps.
2. **No Qwen3.8 technical report on arXiv** and **no `QwenLM/Qwen3.8` GitHub repo** exist as of
   2026-08-15.
3. **Whether blockwise FP8 actually fails on GB10.** #51884 is against sm_120 (RTX 5090) with a
   compressed-tensors checkpoint; ours would be a native `fp8` checkpoint on sm_121. Same
   capability-family gate, different repro.
4. **Nothing was run on the cluster** — no load, no tok/s, no VRAM measurement. The only
   third-party GB10 datapoint is sglang#34872 (FP8), which does not validate the BF16 or NVFP4
   cells.
5. **`--load-format instanttensor`** is an eugr-image extension, not upstream vLLM; not
   confirmed present in the tag recommended for pinning.
6. **Quality of the community quants** — sizes and MTP-head survival were verified from the
   safetensors index, but not numerics. The cyankiwi repos were created the same day as this
   research with 0–3 downloads.

---

## 11. Measured on hardware — dgx-spark-01, 2026-08-28

First boot of `@dgxrun/qwen3.8-27b-bf16`. Everything below is measured on this fleet, not
quoted.

| Property | Result |
|---|---|
| Loads and serves | ✅ `running`, 103,778 MiB VRAM, graph capture 5 s / 0.13 GiB |
| Context | KV cache **716,119 tokens** at `max_model_len 262144` (2.73× concurrency) |
| MTP | ✅ Engaged, **no separate drafter** — *"Detected MTP model. Sharing target model embedding weights with the draft model."* |
| MTP acceptance | 56.2% of drafted tokens; mean 1.68 accepted per draft → **2.68 tokens per forward pass**. Per position 77% / 53% / 38% |
| Decode | **~9–12 tok/s** single stream |
| Vision | ✅ Works and is accurate (below) |
| Image cost | **677 prompt tokens** for one 768×768 image |

The predictions from §3 held: the `Qwen3_5ForConditionalGeneration` architecture loaded with no
new vLLM support, no `--trust-remote-code`, transformers 5.15.0 against a config stamped
`5.8.0.dev0`, and MTP came from inside the checkpoint.

### Vision is real

A 768×768 synthetic render of a machined bracket with a hairline crack was described correctly:
the part (rectangular plate, two symmetric bores in the upper half) and then the defect — *"a
crack … originates at the bottom edge, roughly centered horizontally (between the two bores),
and propagates upward with an angular/zigzag bend, ending partway up the face."* Position, path
shape, and termination all correct.

### ⚠️ Two findings that change how this must be benchmarked

**1. Decode is slower than Muse Glimmer.** ~9–12 tok/s versus Glimmer's measured 17.6 tok/s on
one Spark — and Qwen3.8-27B is the *smaller* model with working speculative decoding. Not yet
explained. Candidates worth an A/B: `--attention-backend flashinfer` (unset here, vllm#51987
open), and prefix caching interacting with hybrid GDN under MTP (vllm#52244).

**2. The default reasoning effort can return a COMPLETELY EMPTY response.** When thinking
overruns the token cap, the API returns `finish_reason: "length"`, `completion_tokens` at the
cap, and **both `content` and `reasoning_content` empty**. Not truncated — empty. Measured on
*"Write a clear 250-word explanation…"*:

| Setting | Tokens | Result |
|---|---|---|
| default (`xhigh`) | >4000, `finish=length` | **empty** |
| `reasoning_effort: low` | 564, `finish=stop` | full answer |
| `enable_thinking: false` | 356, `finish=stop` | full answer |

The `acc-gpqa-diamond-full` preset uses `maxGenToks: 4096` with `reasoning: true`. Any GPQA item
whose reasoning overruns returns empty and scores **wrong**, with no error recorded anywhere —
indistinguishable from the model simply being bad. **Raise the cap well above 4096 or pin
`reasoning_effort` before trusting any accuracy number from this model.** This is a plausible
contributor to the kind of silent-zero result that is easy to misread as a model comparison.

