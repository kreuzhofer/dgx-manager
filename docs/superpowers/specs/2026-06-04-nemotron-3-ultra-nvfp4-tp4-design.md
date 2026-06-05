# Nemotron 3 Ultra — NVFP4, TP=4 deployment recipe

**Date:** 2026-06-04
**Status:** Design approved; revised during execution — see "Execution-time revisions" at the end.

## Goal

Add a deployment recipe so DGX Manager can serve **NVIDIA Nemotron 3 Ultra (NVFP4
quant)** across **4 Spark nodes with tensor-parallel = 4**, and stand up one
verified deployment through the dgx-manager API.

## Context (what is already in place)

- **Multi-node cluster deployment is tested and working.** `recipes/4x-spark-cluster/`
  already holds functioning TP=4 recipes (`qwen3.5-397b-a17B-fp8.yaml`,
  `minimax-m2.5.yaml`). Ray + IB/RDMA interconnect + per-arch image sync are proven.
- A **working Nemotron 3 Super NVFP4** recipe exists
  (`recipes/nemotron-3-super-nvfp4.yaml`, TP=2). It is the reference for the
  Nemotron-family vLLM flags.
- **Recipe discovery recurses into subdirectories.** `packages/agent/src/recipes.ts`
  `scanDir()` walks directories and records the relative path as `Recipe.file`, so a
  recipe at `recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml` is discovered with
  that path. No agent change is required for discovery.
- **Models are stored on NFS.** The recipe references the Hugging Face model id; vLLM
  downloads into the NFS-backed shared cache that all 4 nodes mount. No pre-staging
  step.
- The server already computes `needed = tensor_parallel * pipeline_parallel` and
  rejects the deploy if fewer idle nodes are online
  (`packages/server/src/routes/deployments.ts`).

## Non-goals

- No proxy/dry-run model. We go straight to Nemotron Ultra.
- No dgx-manager schema, UI, or admission changes.
- No BF16/FP8 variants — NVFP4 only.
- No manual `run-recipe.sh` / `ssh` / `docker`. Deployment is **exclusively** via the
  dgx-manager REST API.

## Deliverable

A single new recipe file:

**`recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml`** in the
`spark-vllm-docker` repo.

Agent/launch code changes are **contingency only** — applied solely if the bring-up
exposes a genuine launch bug. Any such change requires `./scripts/bump-agent-version.sh`
and a unit/property test for pure logic (per CLAUDE.md risk tiers).

## The recipe

Derived as a **diff of the working Super NVFP4 recipe**, bumped to TP=4, placed in the
4-node cluster directory.

### Carried over from Super (Nemotron hybrid-Mamba / MoE essentials)

These are the starting template. **Validate against Ultra's model card / `config.json`
before locking** — Ultra's architecture may differ (parser name, MoE backend, mamba
cache, attention backend):

- `--moe-backend cutlass`
- `--mamba_ssm_cache_dtype float32`
- `--reasoning-parser nemotron_v3`
- `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice`
- `--kv-cache-dtype fp8`
- `--enable-prefix-caching`
- `--load-format fastsafetensors`
- `--trust-remote-code`
- `--attention-backend TRITON_ATTN`
- `--distributed-executor-backend ray`
- env: `VLLM_FLASHINFER_ALLREDUCE_BACKEND: trtllm`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN: 1`

### Changed for Ultra / TP=4

- `model:` → the **Ultra NVFP4 HF id** (verify exact string; Super is
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`).
- `tensor_parallel: 4`
- `cluster_only: true` (Ultra is too large for a single 128 GB node)
- Add SPREAD placement like the 4x minimax recipe, forcing one rank per node:
  `env: VLLM_DISTRIBUTED_EXECUTOR_CONFIG: '{"placement_group_options":{"strategy":"SPREAD"}}'`
- **Capacity, conservative first launch** (Ultra > Super 120B):
  `gpu_memory_utilization: 0.8`, `max_model_len: 32768`. Raise `max_model_len` toward
  Super's 262144 once it serves cleanly.

### Sketch (final values resolved during implementation)

```yaml
recipe_version: "1"
name: Nemotron-3-Ultra-NVFP4
description: vLLM serving Nemotron 3 Ultra (NVFP4) on a 4-node TP=4 cluster
model: nvidia/NVIDIA-Nemotron-3-Ultra-<size>-NVFP4   # VERIFY exact id
container: vllm-node
cluster_only: true
env:
  VLLM_FLASHINFER_ALLREDUCE_BACKEND: trtllm
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: 1
  VLLM_DISTRIBUTED_EXECUTOR_CONFIG: '{"placement_group_options":{"strategy":"SPREAD"}}'
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 4
  gpu_memory_utilization: 0.8
  max_model_len: 32768
command: |
  vllm serve nvidia/NVIDIA-Nemotron-3-Ultra-<size>-NVFP4 \
    --trust-remote-code \
    --moe-backend cutlass \
    --mamba_ssm_cache_dtype float32 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3 \
    --attention-backend TRITON_ATTN \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --max-model-len {max_model_len} \
    --host {host} --port {port} \
    --tensor-parallel-size {tensor_parallel} \
    --distributed-executor-backend ray
```

## Bring-up flow

Plumbing is proven, so this is lightweight.

1. **Verify model identity** — confirm the exact Ultra NVFP4 HF id and read its model
   card / `config.json` to validate the carried-over flags (arch, recommended parsers).
2. **Write the recipe** with the validated values.
3. **Preflight** — recipe appears in `agentHub.getRecipes()`; ≥4 idle nodes online.
   (IB is trusted; no heavy NCCL probe.)
4. **Deploy via API** — `POST /api/deployments` with `nodeIds: "auto"`,
   `recipeFile: "recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml"`,
   `runtime: "vllm"`. No other launch path.
5. **Tail the deploy log directly** through milestones: image sync → Ray cluster forms
   → model download (NFS) → weights load → NVFP4/CUTLASS kernels init →
   `Application startup complete`.

## Risk buckets

With plumbing proven, risk collapses to two:

- **Model / capacity** — OOM at weight load or KV-cache allocation → tune
  `gpu_memory_utilization` / `max_model_len`.
- **Flag carry-over** — an Ultra-specific arch difference makes a Super flag wrong
  (e.g. parser name, moe/attention backend) → adjust the recipe flags.

A genuine launch bug in the agent is unlikely given the proven plumbing; if one
appears, fix in `packages/agent/src/`, bump the agent version, add a test for pure
logic.

## Definition of done

- Recipe is discovered by the agent and selectable via the API.
- TP=4 deploy launched via `POST /api/deployments` reaches `Application startup
  complete` with a 4-rank tensor-parallel group across the 4 nodes.
- A `POST http://<head-node-ip>:<port>/v1/chat/completions` request — straight to the
  deployment on its assigned node/port — returns a coherent completion. (The `/lb/`
  inference proxy in `packages/server/src/proxy/inference-proxy.ts` is **not mounted**
  in `index.ts`, and deployments are not auto-registered as LB endpoints; wiring the
  proxy is a separate dgx-manager code change, out of scope here. Direct-to-node is the
  verification path.)
- If any agent code was touched: `npm test` green and agent version bumped. (Expected:
  no code changes — recipe only.)

---

## Execution-time revisions (2026-06-04)

Investigation during execution changed several assumptions. The goal and the
"deploy via dgx-manager API" requirement are unchanged; the path to get there grew.

1. **Model identity & architecture.** Target is
   `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` (550B total / 55B active),
   architecture **LatentMoE** — Mamba-2 + MoE + Attention hybrid with **Multi-Token
   Prediction (MTP)**. Quant is ModelOpt MIXED_PRECISION (Mamba proj FP8, MoE experts
   NVFP4 g16), KV-cache FP8 — auto-detected from `hf_quant_config.json`, so no explicit
   `--quantization`. NVIDIA's model card recommends **TP=4** for Ultra.

2. **vLLM version is the real gate.** Ultra needs vLLM **≥ 0.22** (LatentMoE +
   `nemotron_h_mtp` MTP). The shared `vllm-node` container shipped vLLM **0.18.1**
   (March prebuilt aarch64 wheel) — too old. eugr (`eugr/spark-vllm-docker`, our
   upstream) publishes nightly prebuilt sm_121a wheels; the `prebuilt-vllm-current`
   release is **0.22.1rc1** (2026-06-03, the day Ultra dropped).

3. **Fork sync.** Our `spark-vllm-docker` fork was 28 commits behind upstream; merged
   `upstream/main` (clean) and pushed to origin so the wheel-download logic + Super
   reference are current.

4. **Engine isolation (key decision).** Because every recipe shares `container:
   vllm-node`, bumping it to 0.22.1 would move *all* recipes off the tested 0.18.1
   engine (flag-surface drift, esp. our untested fork-local recipes). Decision: Ultra
   runs on its **own image** `container: vllm-node-eugr022` (built from the 0.22.1rc1
   wheel); `vllm-node` stays 0.18.1 and existing recipes are untouched. The 0.18.1
   wheel is preserved under `wheels/.preserve-0.18.1/` so `vllm-node` can be rebuilt.

5. **Recipe format = v1, in the fork.** Maintainer guidance (PR #165 comment) is to move
   new recipes to **sparkrun / v2** in `spark-arena/community-recipe-registry`. But
   sparkrun is not integrated into dgx-manager (grep finds nothing), and our agent only
   runs v1 via `run-recipe.sh`. To keep "deploy via dgx-manager API," we author **v1**
   at `recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml`; the v2/sparkrun transition
   is deferred to its own effort.

6. **Recipe flags = Super's Spark-adapted baseline + Ultra deltas.** On sm121/GB10,
   FlashInfer NVFP4 is unreliable, so Super uses CUTLASS MoE + TRITON attention +
   fastsafetensors (not NVIDIA's datacenter FlashInfer flags). Ultra adds TP=4,
   `cluster_only`, `--enable-expert-parallel`, and MTP
   `--speculative-config '{"method":"nemotron_h_mtp","num_speculative_tokens":5}'`.
   Capacity seeds conservative (`gpu_memory_utilization: 0.8`, `max_model_len: 65536`).
   Flags marked uncertain (`--moe-backend`, `--mamba-ssm-cache-dtype`, MTP) are
   validated against the built engine (`vllm serve --help`) and the deploy loop.

7. **Container build/distribution path.** The agent builds a *missing* image on deploy
   via `run-recipe.sh`/`build-and-copy.sh`, and `syncContainerImage()` copies the head's
   image to workers (`docker save | ssh docker load`) on layer mismatch. We pre-build
   `vllm-node-eugr022` on the head node (`dgx-spark-01`) with
   `./build-and-copy.sh -t vllm-node-eugr022 --force-vllm-download` so flags can be
   verified before the live deploy; the agent then syncs it to the 3 workers.

## Outcome (2026-06-05) — DEPLOYED & VERIFIED ✅

TP=4 deployment reached `Application startup complete` across all 4 nodes and returned a
coherent `/v1/chat/completions` reply (served name `nemotron-3-ultra`, ~95 GB/node VRAM,
GPU KV cache 776,601 tokens, 11.85× concurrency @ 65 k ctx). Five real bugs were found and
fixed through the deploy loop (each redeploy got strictly further):

1. **JSON-in-command** → `run-recipe.py` does single-pass `str.format()`, so inline
   `--speculative-config '{"method":...}'` braces were misread as `{placeholders}`. Fix:
   put the JSON in a `defaults` value, reference as one `{speculative_config}` token.
2. **`--enable-expert-parallel`** → CUTLASS NVFP4 MoE doesn't support EP (`ep_size>1`) on
   sm121. Fix: drop it; experts shard via TP (Super's pattern).
3. **`--load-format instanttensor`** → `OSError: [Errno 24] Too many open files` opening
   the 113 shards. Fix: `--load-format safetensors`.
4. **`--speculative-config` (MTP)** → MTP draft layers carry an *unquantized* MoE, and
   `--moe-backend cutlass` is quantized-only ("not supported for unquantized MoE"). A
   single global `--moe-backend` can't serve both NVFP4-main (needs cutlass on sm121) and
   unquantized-MTP (needs triton/flashinfer/aiter). **RESOLVED (MTP now enabled):** drop
   `--moe-backend` (unquantized MTP MoE auto-selects its own backend — observed FlashInfer
   CUTLASS) and set env `VLLM_USE_FLASHINFER_MOE_FP4=0` (read only by the FP4 oracle, so it
   strips FlashInfer and the NVFP4 main experts deterministically land on `VLLM_CUTLASS`).
   Verified: main=`VLLM_CUTLASS`, MTP=FlashInfer CUTLASS, coherent completion.
5. **`--load-format fastsafetensors`** removed in vLLM 0.22 → use `safetensors`.

Each failed deployment record was DELETED before retrying (no trail). Known follow-ups:
raise `max_model_len`/`gpu_memory_utilization` once stable; tune MTP token budget
(`max_num_batched_tokens` for the draft slots); revisit `instanttensor` (needs higher
container `nofile` ulimit); the agent marks `status: running` *before* vLLM actually serves
(verify via the HTTP endpoint / "startup complete", not the status field); reconcile the
`/mnt/tank` agent repo with origin.
