# Prompt caching, prefill, and decode

**Research date:** 2026-08-31. Sources are vLLM `main` (HEAD `fdbf2ddb`), tagged releases
v0.9–v0.28.0, upstream issues/PRs, LMCache/SGLang/llm-d/Dynamo docs, and Anthropic/OpenAI
API docs. Our own measurements are marked **MEASURED**.

---

## 1. The short answer

**Prompt caching is not a layer on top of the engine, and it is not a model capability.**
It is the KV-cache manager reusing blocks it already holds. Nothing in a model's `forward()`
participates. What the model architecture decides is not *whether the feature exists* but
*whether a lookup can find anything*.

It is **already on** in every recipe we run — vLLM defaults `enable_prefix_caching = True`
(`vllm/config/cache.py`), so omitting the flag does not disable it; only
`--no-enable-prefix-caching` does.

And the part that matters for the goal that prompted this: **it accelerates prefill only.**
vLLM's own doc is unambiguous — APC "only reduces the time of processing the queries (the
prefilling phase) and does not reduce the time of generating new tokens (the decoding
phase)." Token *generation* is a different toolbox (§7). Anyone promising a cache that
speeds up decode is describing something else.

---

## 2. What it actually is

### The mechanism

vLLM hashes each **full** KV block and keeps a map from hash to block:

```python
BlockHash(hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys)))
```
— `hash_block_tokens()`, `vllm/v1/core/kv_cache_utils.py`

- **Chained**: each block's hash includes its parent's, so a match means "this entire prefix
  is identical", not "this block is identical".
- **Exact token ids**, never embeddings or semantics. One differing token invalidates
  everything after it.
- **`extra_keys`** carry LoRA name, multimodal item hashes, prompt-embeds digest, and
  `cache_salt` (first block only).
- **Full blocks only.** A 15-token shared prefix at block size 16 hits *zero*.

On a hit the scheduler simply subtracts those tokens: `num_new_tokens = request.num_tokens -
num_computed_tokens`. The hit tokens never enter the forward pass, so the whole prefill cost
for them — QKV projections, MLP/MoE, norms, attention — disappears. What does *not*
disappear: the surviving suffix still attends over the full reloaded prefix.

Storage is the GPU KV pool itself, indexed by a hash map; eviction is LRU over a
preallocated block pool. It costs **no extra memory** and, per vLLM's V1 blog, "less than 1%
decrease in throughput even when the cache hit rate is 0%". That is why it is on by default.

### Three properties that surprise people

1. **Per-instance and in-GPU-memory.** It does not survive an engine restart, and replica A's
   cache is invisible to replica B.
2. **Best-effort.** No pinning, no reservation, no TTL. Any prefix can be evicted between two
   requests.
3. **A 100% hit still costs one block.** `max_cache_hit_length = request.num_tokens - 1` —
   the last token must be recomputed to produce logits.

### Engine feature vs model feature

Mechanically: pure engine. But vLLM gates the *default* on a model property,
`ModelConfig.is_prefix_caching_supported`:

| `attn_type` | default |
|---|---|
| `decoder` (generative) | **on** |
| `hybrid` (Mamba/GDN + attention) | **on** — flipped by [#50991](https://github.com/vllm-project/vllm/pull/50991), merged 2026-08-04 |
| `attention_free` (pure SSM) | off |
| `encoder_decoder` | off (cross-attention `find_longest_cache_hit` raises `NotImplementedError`) |

So: the *engine* implements it; the *architecture* determines the search algorithm
(full-attention scans left-to-right, sliding-window right-to-left, Mamba matches a single
checkpoint state) and, through the attention backend, the **block size** — which is the
granularity floor below which nothing hits.

### The commercial APIs are the same thing

Anthropic's and OpenAI's "prompt caching" is this mechanism plus a business contract. vLLM's
own design doc says prefix caching "has been widely used by many public endpoints (e.g.
OpenAI, Anthropic, etc.)". What the vendors add: a **billing split** (writes 1.25–2× input,
reads 0.1×), a **TTL guarantee** (5 min / 1 h / 30 min) instead of "whatever LRU does",
**explicit breakpoints** (`cache_control`, max 4), and **routing affinity** across a fleet.
Self-hosted, the billing is free, the TTL and the affinity are ours to build, and breakpoints
are unnecessary because vLLM caches every full block automatically.

The inverse also exists: vLLM's **`cache_salt`** *partitions* the cache per tenant, mitigating
the TTFT timing side-channel of CVE-2025-46570 (prefix-guessing is "nearly perfectly
distinguishable, ROC AUC 0.99, at prefix lengths of just 8 tokens").

---

## 3. What it accelerates — and what it does not

| | effect |
|---|---|
| **TTFT / prefill** | Drops roughly in proportion to the hit fraction. This is the entire win. |
| **Decode / ITL / tok-s** | **Unchanged.** APC never touches the decode loop. |
| **Effective concurrency** | Indirect win — shared blocks are refcounted, not copied, so N requests on one system prompt pay for its KV once. |
| **Cascade attention** | The one decode-side effect of a shared prefix. Requires `common_prefix_len ≥ 256`, `num_reqs ≥ 8`, and **`dcp_world_size == 1`**. Opt-in since v0.23 (`--no-disable-cascade-attn`). Unavailable on every DCP deployment we run, and MLA backends never implement it. |

**Consequence for our workloads.** A reasoning model generating 2–8K thinking tokens from a
short unique prompt gets ~nothing from prompt caching. An agentic coding session resending a
growing 100K transcript gets almost everything. Same engine, same flag.

---

## 4. The layer map

Five distinct things get called "caching". Layers 1, 2 and 4 are the same physical mechanism
at different scopes; layer 5 is a different thing entirely.

| Layer | Stores | Lives | Saves | Our status |
|---|---|---|---|---|
| **1. In-engine prefix cache** (vLLM APC, SGLang RadixAttention) | KV blocks in HBM | inside the engine | prefill FLOPs | **on everywhere, free** |
| **2. KV offload / multi-tier** (`OffloadingConnector`, LMCache) | KV blocks in DRAM / NVMe / object store | beside the engine | prefill FLOPs, paid in PCIe bytes | not used |
| **3. Prefix-aware routing** (llm-d, Dynamo, SGLang router, production-stack) | *nothing* — it is placement | in front of N replicas | makes 1 and 2 actually hit | **not implemented — see §6** |
| **4. API prompt caching** (Anthropic/OpenAI) | KV blocks, provider-side | inside the provider | money | n/a, we self-host |
| **5. Semantic/response cache** (GPTCache-style) | the generated **response** | in the app | the entire inference | not used, and not appropriate |

**Layer 2 — when it pays.** vLLM's `OffloadingConnector` reports TTFT gains of 2×–22× loading
KV from CPU, but measured with *GPU prefix caching disabled* to isolate the effect. LMCache's
most transferable benchmark (739 real Claude Code traces, 93–97% prefix reuse across turns)
shows 3.0× lower mean TTFT at 100K context / 32 users. **The negative result in that same
post is the one to internalise**: at 16K contexts with HBM headroom still available, LMCache
*cost* 10–17% throughput. Their conclusion — "cache hit rate alone is not enough" — means
layer 2 only pays when the working set genuinely exceeds HBM.

Note LMCache ships an official GLM-5.2 recipe (vLLM 0.23.0 + LMCache 0.4.7) that runs with
`--no-enable-prefix-caching`, routing all reuse through LMCache instead. It is TP-only; **PP
and DCP are not addressed in their docs**, so DCP + external KV is unverified. The KV
connector API is still labelled experimental upstream.

**Layer 5 — why not.** A semantic cache returns a *previously generated answer to a different
question*, accepting false positives by design. Fine for FAQ deflection; wrong for anything
agentic, tool-calling, or freshness-sensitive.

---

## 5. Our stack, model by model

### GLM-5.2 (MLA + DCP) — works, at coarse granularity

APC and DCP compose. `MLAAttentionSpec` extends `FullAttentionSpec`, so MLA rides the normal
full-attention path; and DCP does not disable caching, it **multiplies the match granularity**:

```python
if dcp_world_size > 1:
    # DCP shards each block's KV across ranks; hashes must be viewed at
    # the sharded block size.
    block_size *= dcp_world_size
```
— `FullAttentionManager.find_longest_cache_hit`, `vllm/v1/core/single_type_kv_cache_manager.py`

`B12xMLASparseBackend` supports block size 64, so:

- **DCP2 → 128-token match granularity**
- **DCP4 → 256-token match granularity**

Nothing to fix; that is just the floor. Our `--dcp-kv-cache-interleave-size 1` is the safest
value (`block_size % interleave == 0` is enforced). The sparse indexer's own cache is
`CircularBufferSpec.prefix_cacheable = False`, but it does not poison the MLA group.

Historical note: DCP + APC returned **the previous request's answer** on v0.10.2.x
([#26672](https://github.com/vllm-project/vllm/issues/26672)); fixed in v0.11.0. We are well
past that.

### Qwen3.8-27B BF16 (hybrid GDN + native MTP) — **works; the earlier zero was a length artifact**

**MEASURED 2026-08-31**, spark-01, this recipe unchanged, image `2026081501`, MTP active at
nst=5. Hits follow an exact law with hash unit **U = 816 tokens**, losing exactly one unit to
the MTP/Eagle last-block drop:

```
hits(L) = max(0, floor((L-1)/U) - 1) * U          U = 816
```

| prompt tokens | cached | measured | TTFT cold -> cached |
|---|---|---|---|
| 166 / 322 / 622 / 1222 | 0 | 0 | — |
| 2,422 | 816 | 816 (33.7%) | — |
| 4,822 | 3,264 | 3,264 (67.7%) | 4.54 s -> 2.35 s |
| 19,222 | 17,952 | 17,952 (93.4%) | 15.56 s -> 2.10 s |
| 48,022 | 46,512 | 46,512 (96.8%) | 33.01 s -> 2.62 s |

Every point fits exactly. The **zero-reuse threshold is 2U = 1,632 tokens**; above it the
unreused remainder is bounded between U and 2U *regardless of context length*, i.e. a fixed
~1.7 s of prefill at ~980 tok/s.

**The earlier finding is retracted.** The 2026-08-30 measurement used a **1,342-token** prompt
and swept only `prompt_tokens % 16` — varying the remainder while holding the length below the
cliff. It could only ever report zero. "MTP and prefix caching are mutually exclusive" does not
hold on this stack, and the break-even arithmetic derived from it (`P ~= 121 x G`) is withdrawn:
it assumed a full re-prefill per turn. Corrected, MTP wins at every context length, including
the long-context/short-reply case the old comment carved out as its exception (100K in / 200
out: ~20 s with MTP vs ~43 s without, not 121 s vs 43 s).

The 816-token unit is the inflated attention block covering the GDN mamba page — the same
mechanism as [#53749](https://github.com/vllm-project/vllm/issues/53749). Not tunable from the
recipe.

### Qwen3.8-27B NVFP4 on v0.28.0 — **still zero, and not explained by the law**

The sibling deployment (`qwen3.8-27b-nvfp4-rtx`, aihost01, vLLM **v0.28.0**) reports
**0 hits over 49M queried tokens** with MTP active — where the BF16 engine on the older image
hits 88.7% lifetime. Same model family; different image, quantization and hardware. The law
above does not explain a flat zero at that query volume.

Two further facts about that deployment:

- **It never opted in.** [#50991](https://github.com/vllm-project/vllm/pull/50991) shipped in
  v0.28.0 under "New defaults — prefix caching enabled by default for Mamba models". The recipe
  passes no caching flag either way, so it runs align-mode APC by default.
- That puts it in [#50188](https://github.com/vllm-project/vllm/issues/50188)'s exact
  configuration (RTX 5090, NVFP4, MTP k=3, APC), where 3 of 6 byte-identical repeats returned
  malformed tool markup in 1.4-3.5 s instead of 20-65 s. Discriminator: set `cache_salt` on the
  failing request; if it comes back correct, it is that bug.

A controlled sweep on that endpoint is the obvious next measurement; it was serving traffic
when this was written.

### Muse Glimmer 30B (dflash drafter) — **MEASURED working, 2026-08-31**

Engine `0.27.2rc1.dev113+g5cecfc013` on dgx-spark-04, `--enable-prefix-caching` explicit,
`dflash` drafter with a separate assistant model, `num_speculative_tokens: 15`.

| call | prompt_tokens | cached | wall |
|---|---|---|---|
| 1 (cold) | 1251 | 0 | 46.6 s |
| 2 (exact repeat) | 1251 | **1088 (87%)** | 8.2 s |
| 3 (exact repeat) | 1251 | **1088 (87%)** | 40.2 s |

Wall times are **confounded** — a benchmark was running at 8 concurrent, so these are
queueing times, not prefill times. Do not quote them. The hit counts are sound.

The 163 unreused tokens are consistent with the documented `prompt_len - 1` cap plus the
Eagle-family one-unit drop plus block alignment — i.e. correct behaviour, not a defect.

**Two measurement traps found the hard way:**

- **The counter lags the response.** Call 3's hit did not appear in `/metrics` until after the
  HTTP response returned; scraping immediately made it look like a miss. Scrape a few seconds
  late.
- **`prompt_tokens_details` was `null`.** vLLM *does* populate `cached_tokens`, but it is
  gated behind **`--enable-prompt-tokens-details`**, off by default
  (`vllm/entrypoints/launchers/cli_args.py`). No recipe of ours sets it.

Note this is a *third* config with a speculative drafter, and it hits. Combined with
GLM-5.3-Flash, the split is not "spec decode kills APC" — it is the hybrid-GDN interaction
above.

### GLM-5.3-Flash — hits, then evicts almost immediately

Our recipe already warns that the "~21,000 tok/s" on a repeated long prompt "is prefix cache,
not prefill" — correct. But [#54458](https://github.com/vllm-project/vllm/issues/54458) is
the thing to know: block-size inflation to cover the mamba page produces a **7,808-token
attention block** and ~23 KV groups, a fixed per-request footprint regardless of prompt
length. Measured there: a cached 35K prompt is **fully evicted after 3 interleaved 35K
admissions** (23,808 hits → 0, TTFT 2.3 s → 7.1 s). Capping `--max-model-len` does not help;
the pool shrinks proportionally.

### Benchmarks — deliberately cache-free, correctly

`packages/server/src/benchmarks/presets.ts` sets `enablePrefixCaching: false` on all five
throughput presets, so llama-benchy varies each prompt. **MEASURED today**: during a live run,
`prefix_cache_queries_total` climbed 12,228 → 21,549 with hits frozen at 2,176. That is the
intended behaviour and it keeps runs comparable — but it means our benchmark numbers measure
**cold prefill**, which is not what interactive traffic experiences. vLLM's own benchmarking
doc warns of the converse: re-running against a live server inflates throughput from leftover
cache entries. Relevant given the 20% SWE-bench noise floor we already recorded.

---

## 6. The gap: no cache-aware routing, no cache observability

**Routing.** `packages/server/src/gateway/selection.ts` picks least-outstanding, ties broken
round-robin. There is no session affinity, no prompt hash, no client identity (auth and cookie
headers are explicitly dropped). `publishedName` has no unique constraint, and five GLM
recipes hard-code `served_model_name: glm-5.2` — so two concurrent GLM deployments form one
pool and consecutive turns of the *same conversation* round-robin between two engines with
disjoint KV caches. Every turn is then a cold prefill.

The industry numbers for fixing this are large. llm-d's published benchmark (8 vLLM pods, 150
tenants × 6,000-token contexts): P90 TTFT **92.55 s random → 0.54 s** with a precise
prefix-aware router; throughput 4,429 → 8,730 tok/s. SGLang's router reports hit rate 20% →
75%. All four implementations (llm-d, Dynamo, SGLang, production-stack) expose a knob for the
same tension — routing purely for cache locality creates hotspots.

The seam here is clean: `selectLeastOutstanding` is a pure function with one call site
(`gateway/router.ts:164`), and the body is already buffered and JSON-parsed there for `model`,
so a prefix hash over `messages` costs no extra I/O. `gateway/rotation.ts` is the precedent
for per-published-name gateway state.

**Observability.** `packages/agent/src/runtime/sparkrun-metrics.ts` scrapes
`num_requests_running`, `kv_cache_usage_perc`, `generation_tokens_total` — but **not**
`prefix_cache_queries_total` / `prefix_cache_hits_total`, the exact counters every
prefix-caching claim in this repo rests on. All of them are hand-run one-offs.

---

## 7. Decode is a different problem

Prompt caching cannot help token generation. What can, on our hardware:

- **Speculative decoding** — already everywhere (MTP, dflash, dspark). This is *the* decode
  lever, and on Qwen3.8 it currently costs us the prefix cache outright.
- **Bandwidth.** Our own prior finding stands: GLM-5.2 decode is memory-bandwidth-bound on
  GB10 and already on the community stack. There is no cache that changes this.
- **Cascade attention** would be the one shared-prefix decode win, and it is unavailable to us
  — disabled under DCP, unimplemented on MLA backends.
- **Batching.** Higher `max_num_seqs` raises aggregate throughput, not single-stream latency.

---

## 8. Silent-failure catalogue

Everything here fails **without an error, a warning, or a log line**. This is the dominant
risk in this area — far more than misconfiguration.

| Failure | Trigger | Signal |
|---|---|---|
| **MTP + DCP + FULL cudagraphs → silent output corruption** ([#45425](https://github.com/vllm-project/vllm/issues/45425), open) | attention backend lacking varlen-decode under DCP; DCP forces `reorder_batch_threshold = 1`, so MTP's `q_len=2` verify rows reclassify as prefills while the full-graph capture expects decode | truncation after 1–2 chars, or looping. **PIECEWISE, eager, no-MTP or no-DCP each fix it.** |
| Hybrid GDN + spec decode → 0 hits ([#54360](https://github.com/vllm-project/vllm/issues/54360)) | our Qwen3.8 config exactly | `hits_total` flat at 0 while queries climb |
| Prefix shorter than one block → 0 hits ([#53749](https://github.com/vllm-project/vllm/issues/53749)) | block inflation gives real deployments blocks of 816–7,808 tokens | nothing names the threshold anywhere |
| SWA hits collapse at ~25% pool occupancy ([#48435](https://github.com/vllm-project/vllm/issues/48435)) | Gemma-class models; LRU evicts tail blocks, exactly what an SWA hit needs | hit rate → 0 under mild load |
| MTP + APC corrupts tool calls ([#47194](https://github.com/vllm-project/vllm/issues/47194), open) | hybrid + MTP3 | **83.9% hit rate maintained** while tool calls go 2/10 and needle recall 0/10; raw `<tool_call>` XML leaks as text; recovers only on eviction or restart |
| NVFP4 + MTP + APC → malformed markup ([#50188](https://github.com/vllm-project/vllm/issues/50188), open) | closest match to our `qwen3.8-27b-nvfp4-rtx` recipe | 3 of 6 byte-identical repeats return garbage in 1.4–3.5 s instead of 20–65 s |
| GB10 GDN crash ([#54173](https://github.com/vllm-project/vllm/issues/54173)) | sm_121 + APC, **prompts of differing lengths sharing a prefix** — nine identical 50K prompts run clean | CUBLAS illegal memory access |

**APC is also not bit-deterministic.** Cache-on vs cache-off diverge at T=0 from floating-point
non-associativity — measured max logprob delta 4.6e-02 on H100
([#39389](https://github.com/vllm-project/vllm/issues/39389),
[#40896](https://github.com/vllm-project/vllm/issues/40896)). The design doc's "won't change
model outputs" is prose, not a guarantee. Validate at score level, never token level.

**The upstream docs are actively wrong here.** `docs/usage/v1_guide.md` still says prefix
caching "is not yet supported for any of the above models" (false since 2025-10);
`hybrid_kv_cache_manager.md` still calls Mamba support "work in progress" (stale ~10 months);
and `features/compatibility_matrix.html` **404s** — there is no published vLLM compatibility
matrix. This is why the above came from source.

---

## 9. What to do, ranked

1. **Add `--enable-prompt-tokens-details` to the recipes.** One flag; every client and every
   benchmark then reports `prompt_tokens_details.cached_tokens` per request. Turns every
   future caching question into an observation instead of an expedition.
2. **Scrape the two prefix-cache counters** in `sparkrun-metrics.ts` and store them. Cheap,
   and it makes regressions like #54360 visible the day an image bumps rather than months
   later.
3. **A/B the DCP2 + MTP + `cudagraph_mode: FULL` pair against PIECEWISE on long outputs.**
   `glm-5.2-quanttrio-unpruned-dcp2.yaml` and `-dcp2-320k.yaml` sit exactly on #45425's
   shape. Counter-evidence: those recipes have served real agentic sessions and scored
   normally on benchmarks, which argues against gross corruption — so this is a check to
   close the question, not an alarm. Our DCP4 recipes already use PIECEWISE.
4. **DONE 2026-08-31 for BF16 — repeat it for the NVFP4/v0.28.0 endpoint.** The sweep must
   vary total prompt *length* (not just the remainder) and cross 2U, or it reports zero for
   reasons that have nothing to do with the bug. `scripts/`-able version of the probe is in
   the session scratchpad; the shape is: shared prefix, divergent tail, exact repeat, at
   ~1k/2.5k/5k/19k/48k tokens, scraping `/metrics` ~4 s *after* each response.
5. **Prefix-aware routing in the gateway** — only worth it once we routinely run ≥2 replicas
   of one published name. Until then it is speculative complexity. The seam is identified in
   §6 if and when it is.
6. **KV offloading (LMCache / `OffloadingConnector`)** — not yet. It pays only when the working
   set exceeds HBM, and costs 10–17% throughput when it does not. Revisit if long multi-turn
   agentic sessions start thrashing the pool, which is exactly GLM-5.3-Flash's #54458 shape.

---

## 10. Key sources

- vLLM APC design: `docs/design/prefix_caching.md`; feature doc `docs/features/automatic_prefix_caching.md`
- Source: `vllm/v1/core/{kv_cache_utils,block_pool,kv_cache_manager,single_type_kv_cache_manager,kv_cache_coordinator}.py`
- DCP: `docs/serving/context_parallel_deployment.md` (accurate), PR [#23734](https://github.com/vllm-project/vllm/pull/23734)
- Hybrid default-on: PR [#50991](https://github.com/vllm-project/vllm/pull/50991)
- Spec-decode collapse: [#52244](https://github.com/vllm-project/vllm/pull/52244), [#54360](https://github.com/vllm-project/vllm/issues/54360)
- Security: CVE-2025-46570 / [GHSA-4qjh-9fv9-r85r](https://github.com/vllm-project/vllm/security/advisories/GHSA-4qjh-9fv9-r85r), `cache_salt`
- Layers: [LMCache docs](https://docs.lmcache.ai/), [vLLM KV offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/), [llm-d KV-cache routing](https://llm-d.ai/blog/kvcache-wins-you-can-see), [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- API contracts: [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
