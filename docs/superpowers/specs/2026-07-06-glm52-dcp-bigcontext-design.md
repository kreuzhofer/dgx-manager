# GLM-5.2 big-context via DCP — arm64 DCP-stack build (design)

**Status:** design / awaiting execution (a focused build session)
**Date:** 2026-07-06
**Scope:** An **arm64/sm_121 vLLM image build** on the DCP-stack, + a new `@dgxrun` recipe with DCP enabled, to serve GLM-5.2 at **~200K–320K context** for an agentic-coding daily driver (opencode / Claude Code). Ops/build project — **not** a manager code change. **Out of scope:** the speed question (resolved — decode is ~25–31 on real content, memory-bandwidth-bound; see memory `glm52-decode-workload-dependent`).

## Motivation

Decode speed is a solved problem (workload-dependent ~25–31 tok/s on code; we're on the current community stack). The remaining lever for a **daily-driver coding agent is CONTEXT**. Measured 2026-07-06: the config-only chunk-size fix lifts the 15pct no-DCP ceiling from ~85K to **~159K** (KV-limited at gmu 0.88), shipped as `@dgxrun/glm-5.2-awq-15pct-144k`. That covers a large codebase + long session but is **short of Claude Code's ~200K** and well short of the community's 320K/640K.

Past ~159K the limiter is the **KV pool size per rank** — MLA replicates the latent KV across TP ranks, so each rank holds the *full* cache and the head runs out first. **DCP (decode-context-parallel) shards the MLA KV across the N ranks** (per-rank KV ÷ N ⇒ context × N), which is the only clean way past ~159K. Decode is unaffected (~22–31); prefill pays ~linearly with DCP degree (community: ~720 → ~600 t/s at DCP2). This needs a **different vLLM** than our current legacy build.

## Goals

- Serve GLM-5.2 at **≥200K** (target DCP2 → ~320K) on the 4×GB10 cluster.
- Keep decode ~25–31 on code content (DCP doesn't hurt decode).
- Prefill viable for interactive coding via **prefix caching** (only new tokens prefilled per turn).
- Ship as a first-class `@dgxrun` recipe + a **new image tag** (don't overwrite `b12x:probe`).

## Non-goals

No speed optimization (resolved). No manager code change (recipe + env + a serve flag only). Not the 640K DCP4 config (prefill too slow for interactive use — DCP2 is the target; DCP4 is a documented option, not the daily driver).

## Architecture — the DCP stack (exact pins)

Reuse the documented arm64 build (`docs/glm-5.2-custom-image-build.md` + CosmicRaisins `bootstrap.sh`) but on the **DCP branch** instead of the legacy `VLLM_REF`:

| Component | Current (`b12x:probe`) | DCP-stack target |
|---|---|---|
| vLLM ref | `ab6660699` (legacy, `bootstrap.sh` VLLM_REF) | **local-inference-lab/vllm branch `codex/dcp-globaltopk-sharddraft-defaults-20260622` @ `e232d26`** |
| DCP draft patches | — | **PR#72:** `pr72-1-draft-dcp-config-propagation.patch` + `pr72-2-glm-dcp-draft-path.patch` (without these the drafter crashes under DCP: `requires topk_scores_buffer`) |
| b12x | 0.23.0 | 0.23.0 (verify master ≥ `80eb49b` for the fp8 decode path) |
| DSA kernels | CosmicRaisins sm12x (10 files) | latest CosmicRaisins `kernels/` (DCP-stack revision) + `deep_gemm` re-bind patch |
| Build infra | eugr `spark-vllm-docker` `build-and-copy.sh --vllm-ref <ref> -t <tag> --tf5 --copy-to` | same, with the DCP `--vllm-ref` |
| Runtime env | — | **`VLLM_USE_V2_MODEL_RUNNER=1`** (the DCP+MTP draft path lives only in the V2 runner; V1 drops DCP from the draft config) |
| Serve flag | — | **`--decode-context-parallel-size 2`** (DCP2 → ~320K) |

Everything else matches the existing recipe (NCCL/RDMA stanza, fp8_ds_mla KV, MTP k=3, the reduced sparse-MLA chunk sizes from the 144K recipe, `--enable-prefix-caching`).

### Model choice (decide before/at build)
- **15pct + DCP2** — no download (reuse what we have), smaller weights ⇒ *more* KV headroom ⇒ possibly >320K. Quality: tool-eval already 100/★★★★★; IFEval TBD.
- **Unpruned QuantTrio Int4-Int8Mix + DCP2** — the community reference (320K), best quality (community dropped the prune for instruction adherence), but a **~400 GB download** to `/mnt/tank`.
- **Default recommendation:** build DCP-stack first, deploy **15pct + DCP2** (zero download) to validate the stack + hit ~320K; pull the unpruned model only if a quality eval says the prune hurts *your* coding.

## Build approach

1. Clone the DCP vLLM branch into the eugr `spark-vllm-docker` clone (`/mnt/tank/src/github/spark-vllm-docker`); apply the PR#72 patches.
2. `build-and-copy.sh --vllm-ref <DCP-ref> -t vllm-node-tf5-glm52-b12x-dcp:probe --tf5 --copy-to` — the ~1 h arm64 CUDA build on a GB10 node; verify `torch.cuda.is_available()` (torch must NOT be `+cpu`).
3. Overlay: copy the latest CosmicRaisins `kernels/` into the MLA backend + ops dirs; apply `patch_deep_gemm.py`; `pip install --no-deps b12x==0.23.0`.
4. Stage the NCCL 2.30.4 aarch64 lib (already on NFS) for `LD_PRELOAD`.
5. Write `@dgxrun/glm-5.2-awq-15pct-dcp2.yaml`: the new image tag + `VLLM_USE_V2_MODEL_RUNNER=1` env + `--decode-context-parallel-size 2` + `max_model_len` ~262144–327680.

## Risks (this is an arm64 CUDA build — where things silently break)

- **The DCP branch building on arm64/sm_121** — it's a fork; verify it compiles (biggest risk).
- **cu130 vs cu132** — our current image is cu130-derived; the community's DCP images are cu132. The DCP branch may require cu132 (⇒ verify a torch+cu132 **aarch64** wheel exists — the recurring GB10 blocker).
- **PR#72 patches applying** to that branch revision (rebase drift).
- **V2 runner + DCP + MTP + our sm12x kernels** interacting correctly on sm_121 (the community runs exactly this, so it's known-feasible, but it's the integration surface).
- **b12x compatibility** with the DCP branch.

## Success gate

Build succeeds → deploy **15pct + DCP2** → **serves at ≥262K** → decode ~25–31 on code content → a real agentic-coding session via opencode/Claude Code works end-to-end (large codebase loaded, multi-turn, prefix-cache amortizes prefill). If all green: this is the daily driver. If the prune quality disappoints, repeat with the unpruned model (after the ~400 GB pull).

## Testing / validation

- **Startup:** clears capture/JIT at the target context (like the 144K probe).
- **Correctness at long context:** a needle-in-haystack retrieval at ~200K + a coding task that references early context — output stays coherent (DCP KV-sharding must not corrupt retrieval).
- **Decode:** ~25–31 on code content at ≥200K (read the vLLM `SpecDecoding metrics` for MTP acceptance).
- **Prefill:** TTFT at a few depths with prefix caching on (the interactive-coding cost).
- **Daily-driver acceptance:** opencode / Claude Code integration against the `/v1` endpoint.

## Rollback / coexistence

Build to a **new image tag** (`…-dcp:probe`); keep `b12x:probe` + the `144k` recipe untouched as the fallback (config-only ~159K, no DCP). Both recipes remain available in the `@dgxrun` catalog.

## Open questions for the build session

1. Does the DCP branch build on arm64, and does it need **cu132** (is there a torch+cu132 aarch64 wheel)? — resolve first; it gates everything.
2. **Model:** 15pct+DCP2 (no download) vs unpruned+DCP2 (~400 GB, better quality) — default to 15pct first.
3. **DCP degree / max_model_len:** DCP2 @ ~262–327K is the target; confirm the head fits at that context with DCP sharding + the reduced chunks.
4. Does **prefix caching** compose with DCP (the per-turn prefill amortization that makes big context interactive)?

## References

- Memory `glm52-decode-workload-dependent` (speed resolved; the ~159K no-DCP ceiling), `glm52-shm-and-jit-findings`.
- Community: CosmicRaisins/glm-5.2-gb10 (README + `bootstrap.sh` + `CHANGES.md`), local-inference-lab/vllm (`codex/dcp-*` branch + PR#72), forum thread 374125.
- Current arm64 build: `docs/glm-5.2-custom-image-build.md`. Shipped config-only big-context recipe: `recipes/dgxrun/glm-5.2-awq-15pct-144k.yaml`.
