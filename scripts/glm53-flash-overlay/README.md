# GLM-5.3-Flash sm_121 overlay

Eight thin layers that make `vllm/vllm-openai:glm53-flash-arm64-cu130` — vLLM's
own day-0 arm64 image for GLM-5.3-Flash — actually run on a GB10 Spark, plus one
layer of ours that makes the result usable as a dgxrun container.

## Why an overlay at all

GLM-5.3-Flash is `glm5_next`, not `glm_moe_dsa`. It is a different model from
GLM-5.2 in every way our stack cares about — 45 layers instead of 78, hybrid KDA
linear attention on 34 of them, and **NoPE MLA** (`qk_rope_head_dim = 0`). None
of the GLM-5.2 machinery applies: not the CosmicRaisins sparse-MLA kernels, not
b12x, not `build-glm52-image.sh`. Do not try to reuse them.

The good news is that nothing here recompiles CUDA. Upstream ships a working
arm64 image; every layer below is a Python source patch or a pip pin, so a build
is minutes rather than the hour `build-glm52-image.sh` costs.

The core defect the stack fixes: the day-0 image's only capability-12 sparse-MLA
backend is `FLASHINFER_MLA_SPARSE_SM120`, whose packed `fp8_ds_mla` layout
hard-requires DeepSeek's `pe_dim = 64` and dies in warmup. The SM90 NoPE backend
handles `kpe = 0` but gates itself to capability 9. Layer v1 opens that gate;
v3–v8 fix what then breaks.

## Layers

| Layer | Kind | What it does |
|---|---|---|
| `v1` | vLLM py | Offer `FLASHINFER_MLA_SPARSE_SM90` on capability 12; gate it `major in (9,12)`; pick FA2 off-Hopper; scope the FlashInfer ≥0.6.18 check to the fp8-KV path it actually guards |
| `v2` | vLLM py | `GLM53_NAN_DEBUG=1` forward hooks that name the first module emitting non-finite output. Inert unless the env var is set |
| `v3` | pip | FlashInfer → `0.6.18.dev20260819`; drop the AOT jit-cache. 0.6.17's FA2 MLA kernel returns NaN for 64–256-row batches on sm_121 |
| `v4` | pip | Re-pin `nvidia-nccl-cu13==2.30.7` — the FlashInfer nightly silently downgrades it and `ncclCommInitRank` then fails on the IB fabric |
| `v5` | pip | Re-pin `nvidia-cutlass-dsl==4.6.2` — the nightly leaves a mixed 4.7.0/4.6.2 state that ICEs CuTeDSL warmup |
| `v6` | vLLM py | Gate Programmatic Dependent Launch to `major in (9,10)`. On sm_121 it races the KDA state kernels and boots NaN depending on launch timing |
| `v7` | vLLM py | Indexer hardening: top-k pool ids `torch.empty` → `torch.full(-1)`, plus a Triton bounds clamp. Uninitialised pool ids made MLA gather uninitialised KV |
| `v8` | FlashInfer py + `.cuh` | fp8 KV on the FA2 NoPE path: cap `EFF_CTA_TILE_KV` at 32 instead of forcing it (GB10 has ~101 KB smem, not Hopper's 228 KB), and accept major 12 in FlashInfer's gate |
| `dgxrun-entrypoint` | ours | `ENTRYPOINT []` — see below |

Every upstream layer asserts on the exact source text it expects and refuses to
patch otherwise, so a base-image bump fails the build loudly instead of silently
producing an image that serves NaN. That property is the reason these are
vendored verbatim; do not "tidy" the assertions away.

`.cuh` in v8 is a header FlashInfer JIT-compiles at runtime, so editing it is
still a text patch — there is no build-time compile step anywhere in this stack.

### The one layer that is ours

`Dockerfile.dgxrun-entrypoint` resets `ENTRYPOINT` to `[]`. The official
`vllm/vllm-openai` image entrypoints into the server, but dgxrun passes the
whole serve argv as the container **command** (and appends `--nnodes`,
`--node-rank`, `--master-addr`, `--master-port`, `--headless` to it). Without the
reset the recipe's `vllm serve …` lands as arguments *to* `vllm serve`. Every
other dgxrun container here is entrypoint-less; this makes GLM-5.3-Flash behave
the same rather than making the recipe a special case.

## What is deliberately NOT here

Upstream's ninth layer ports DFlash2 speculative decoding (vLLM PR #52816, still
unmerged) and pairs it with `incoai/GLM-5.3-Flash-DFlash2`. We skip it for now:
the draft checkpoint is **CC-BY-NC-ND-4.0** where the target is MIT, and the
model carries a native MTP head that costs no extra licence. See the recipe for
where speculative decoding stands.

## Provenance

| Source | Upstream | Commit | Vendored |
|---|---|---|---|
| `Dockerfile.sm121-v1..v8`, `patch_v7.py`, `patch_v8_fp8.py` | [barrydeen/glm53-flash-dgx-spark](https://github.com/barrydeen/glm53-flash-dgx-spark) `docker/` | `c267f9f` (2026-08-28) | 2026-08-29 |

Upstream is MIT licensed — Copyright (c) 2026 Barry; the copy is at
`LICENSE.upstream`. The vendored files are **unmodified copies**, including
their internal `glm53:sm121-vN` FROM chain — `build-glm53-flash-image.sh` builds
that chain under those local tags and only re-tags the final image, so nothing
here needs rewriting on the way in.

Vendored rather than referenced for the same reason as `mods/`: a deployment
whose behaviour depends on someone else's HEAD is not reproducible. The cost is
that upstream fixes do not reach us on their own — when re-pulling, update the
commit above.
