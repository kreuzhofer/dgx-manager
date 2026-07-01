# Building a GLM-5.2 vLLM Image for DGX Spark (arm64 / sm_121)

Date: 2026-07-01
Hardware: DGX Spark cluster, 4× NVIDIA GB10 (124 GB unified memory each, sm_121)
Result image: `vllm-node-tf5-glm52-b12x:probe`

## Why a custom image is required

GLM-5.2 is a `GlmMoeDsaForCausalLM` model — a DeepSeek-Sparse-Attention (DSA) MoE. Its
sparse-attention **indexer** calls DeepGEMM's `fp8_fp4_mqa_logits` family of kernels. The
stock `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest` image **does not ship the
sm_121 DeepGEMM kernels** — its vendored `deep_gemm/include/deep_gemm/impls/` is empty, so
the JIT fails at warmup with:

```
RuntimeError: Assertion error (DeepGEMM/.../jit/include_parser.hpp:69):
  Failed to open: .../deep_gemm/include/deep_gemm/impls/sm121_fp8_mqa_logits.cuh
```

Upstream `deepseek-ai/DeepGEMM` has no sm_121 mqa_logits kernels (PR #324 adds sm_120 only).
The community (CosmicRaisins / m9e / dark-devotion) solved this by **building a complete
image** with the DSA kernels + a `deep_gemm.py` patch that routes those functions to Triton
"sm12x" fallbacks. Their prebuilt images are amd64-only; the arm64 Spark image was built
locally and never published. This doc reproduces that build for arm64 from public sources.

## Sources (all public)

- **vLLM** `vllm-project/vllm@ab666069935c1f23e8ef56038b4659ac9e8f19f8`
  ("[bugfix] Indexer init skip and MTP TopK share for iteration" #45895)
- **eugr build infra** `spark-vllm-docker` (our clone at `/mnt/tank/src/github/spark-vllm-docker`)
- **DSA kernels** `github.com/CosmicRaisins/glm-5.2-gb10` → `kernels/*.py` (10 files) + `launch.sh`
- **b12x** `pip install --no-deps b12x==0.23.0` (decode kernel; makes cudagraph FULL capture-safe)

## Step 1 — Base image (~1 h arm64 compile, on the head node)

```bash
cd /mnt/tank/src/github/spark-vllm-docker
./build-and-copy.sh \
  --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 \
  -t vllm-node-tf5-glm52-b12x:base --tf5
```

Produces `vllm-node-tf5-glm52-b12x:base` (~9 GB, vLLM `0.23.1rc1.dev190+gab6660699`,
`GPU_ARCH_LIST=12.1a`, transformers ≥ 5).

### Build fix: the PR #35568 patch step

The eugr `Dockerfile` applies vLLM PR #35568 ("Fix SM121 exclusion from Marlin/CUTLASS FP8
paths") via `curl ...35568.diff | git apply`. At `ab666069` this step fails two ways — patch
both in the Dockerfile before building:

1. **curl rate-limiting** — `patch-diff.githubusercontent.com` throttles; add retries:
   `curl -fsL --retry 10 --retry-all-errors --retry-delay 4 ...`
2. **context drift** — the diff doesn't apply cleanly at this ref; make the apply tolerant:
   `(git apply --3way -v --exclude="tests/*" pr35568.diff || git apply -v --reject ... || echo "continuing")`

## Step 2 — Overlay (fast, no CUDA recompile)

Bake the DSA kernels + b12x + the deep_gemm re-bind onto the base. Build context lives at
`/mnt/tank/src/glm52-overlay/` (`kernels/` = the 10 CosmicRaisins files, `patch_deep_gemm.py`,
`Dockerfile.glm52-overlay`).

`Dockerfile.glm52-overlay`:

```dockerfile
FROM vllm-node-tf5-glm52-b12x:base
ARG VLLM=/usr/local/lib/python3.12/dist-packages/vllm
# 10 kernels → vLLM tree (paths from CosmicRaisins launch.sh KMOUNTS)
COPY kernels/sparse_mla_kernels.py kernels/sparse_mla_env.py \
     kernels/sm12x_sparse_mla_attn.py kernels/patch_flashmla_ops.py \
     kernels/flashmla_sparse.py                 ${VLLM}/v1/attention/backends/mla/
COPY kernels/sm12x_deep_gemm_fallbacks.py kernels/sm12x_mqa.py \
     kernels/b12x_sparse_helpers.py             ${VLLM}/v1/attention/ops/deepseek_v4_ops/
COPY kernels/sparse_attn_indexer.py            ${VLLM}/model_executor/layers/
COPY kernels/deepseek_v2.py                    ${VLLM}/model_executor/models/
RUN pip install --no-deps b12x==0.23.0
COPY patch_deep_gemm.py /tmp/patch_deep_gemm.py
RUN python3 /tmp/patch_deep_gemm.py && \
    python3 -c "import ast; ast.parse(open('${VLLM}/utils/deep_gemm.py').read()); print('OK')"
```

```bash
cd /mnt/tank/src/glm52-overlay
docker build -f Dockerfile.glm52-overlay -t vllm-node-tf5-glm52-b12x:probe .
```

### The deep_gemm sm12x re-bind (the crux)

The CosmicRaisins mod's `run.sh` (which patches `vllm/utils/deep_gemm.py`) is **not public** —
it was reconstructed. Every DeepGEMM function the DSA path needs is impl-gated: on sm_121 the
impl is `None`, so the function does `return _missing()` → `RuntimeError`. There are **6** of
them, hit progressively as warmup goes deeper. `patch_deep_gemm.py` appends a block to
`deep_gemm.py` that, for `current_platform.is_device_capability_family(120)`, re-binds all 6
before any `from vllm.utils.deep_gemm import …` consumer runs:

| deep_gemm function | sm12x binding |
|---|---|
| `fp8_fp4_mqa_logits` | → `_fp8_mqa_logits_sm12x` (signatures match) |
| `tf32_hc_prenorm_gemm` | → `_tf32_hc_prenorm_gemm_sm12x` (signatures match) |
| `fp8_fp4_paged_mqa_logits` | → wrapper dropping stock `schedule_metadata`/`clean_logits` args, then `_fp8_paged_mqa_logits_sm12x` |
| `get_paged_mqa_logits_metadata` | → `torch.zeros(())` stub (the sm12x paged kernel ignores schedule metadata) |
| `get_mk_alignment_for_contiguous_layout` | → `[128, 128]` (sm12x kernels use 128-blocks) |
| `get_col_major_tma_aligned_tensor` | → passthrough (Triton kernels need no TMA alignment) |

Plus `has_deep_gemm()` / `is_deep_gemm_supported()` → `True`.

The `_*_sm12x` dispatchers live in the public `sm12x_deep_gemm_fallbacks.py`. `deepseek_v4_ops/`
does not exist in stock vLLM — `COPY` creates it; it imports fine as a namespace package (no
`__init__.py` needed).

Verify on a GPU:

```
GLM52: sm12x DSA fallback re-bind active (mqa_logits/paged/metadata/tf32/mk_align/tma + has_deep_gemm)
fp8_fp4_mqa_logits -> ...sm12x_deep_gemm_fallbacks._fp8_mqa_logits_sm12x
has_deep_gemm: True  mk_align: [128, 128]
```

## Step 3 — Distribute + deploy

sparkrun re-syncs the image from the head at deploy time (it flags workers "stale"), so a
manual copy is optional. Deploy via the dgx-manager API against the recipe
`@community-kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer` (`container: vllm-node-tf5-glm52-b12x:probe`).

### Deploy gotchas

- **sparkrun run-cache is not refreshed by `POST /api/recipes/refresh`.** That updates
  `sparkrun list` only. After editing the recipe, `git pull` the run-cache clone on every node:
  `~/.cache/sparkrun/registries/community-kreuzhofer/`.
- **Pass `displayName: "glm-5.2"` on the deploy** so the dgx-manager benchmark's model name
  matches vLLM (the benchmark uses `deployment.displayName ?? deployment.model.name`, not the
  recipe's `served_model_name` — a server-side bug worth fixing).
- **Startup is ~45 min:** ~12 min shard load + **~21 min silent AWQ/MoE weight-processing**
  (GPU 0 %, looks wedged but isn't) + cudagraph FULL capture, then `Application startup complete`.

## Working serve config (recipe)

`-tp 4 --pipeline-parallel-size 1`, `--distributed-executor-backend ray`,
`--kv-cache-dtype fp8_ds_mla`, `--reasoning-parser glm45 --tool-call-parser glm47
--enable-auto-tool-choice`, `--max-num-seqs 1 --max-num-batched-tokens 4096`,
`--compilation-config '{"cudagraph_mode":"FULL"}'`.

| knob | value | why |
|---|---|---|
| `gpu_memory_utilization` | **0.88** | GB10 CUDA-reported free mem ≈ 109.8 GiB < `0.93×121.63 = 113.1` → vLLM aborts at 0.93 |
| `max_model_len` | **32768** | at 0.88 the KV budget fits only ~62 K tokens; 262144 aborts ("13.17 GiB KV needed > available") |
| `cudagraph_mode` | **FULL** | b12x's decode kernel is capture-safe; without b12x use PIECEWISE |

env: `GLM52_BIND_HOST_TRITON=1 GLM52_MQA_LOGITS_TRITON=1 GLM52_PAGED_MQA_TRITON=1
GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192 GLM52_B12X_MLA=1
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 LD_PRELOAD=/cache/huggingface/nccl-2.30.4/libnccl.so.2`.

## Results

See [glm-5.2-inference-benchmark.md](./glm-5.2-inference-benchmark.md).
