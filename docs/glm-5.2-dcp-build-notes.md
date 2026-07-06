# GLM-5.2 DCP (long-context) build notes

Status as of **2026-07-06**: the DCP image **builds and the stack fully initializes** (B12X_MLA_SPARSE backend + DCP2 grid + all 4 workers + NCCL groups + KV cache). It failed **only in graph capture** on a unified-memory OOM at `gpu_memory_utilization 0.90` / `max_model_len 327680`. **Retry pending at 0.89 / 256K** (recipe already updated). Not yet served/validated.

Build with `scripts/build-glm52-image.sh dcp`; deploy `recipes/dgxrun/glm-5.2-awq-15pct-dcp2.yaml`. Design/plan: `docs/superpowers/{specs,plans}/2026-07-06-glm52-dcp-bigcontext*`.

## Resolved pins (the hard-won facts)

| Thing | Value | Why |
|---|---|---|
| vLLM ref | `e232d262…` on **fork** `local-inference-lab/vllm` (`codex/dcp-globaltopk-sharddraft-defaults-20260622`) | DCP + MTP top-k score-buffer (PR#72) already in-branch |
| torch | keep base **2.11** (`use_existing_torch.py`) | no bump needed |
| **kernel overlay** | **NONE** | e232d26's native `B12X_MLA_SPARSE` ships its kernels in-tree; overlaying the legacy CosmicRaisins kernels **regresses it** |
| b12x | **`0.30.0`** (PyPI) | has `index_topk_fp8(out_scores=…)`; `0.23.0` (legacy) does not. README pins git `9cd63a7`; PyPI 0.30.0 has the same signature + is trusted |
| CUDA | cu130 (base default) | no cu132 needed |

Legacy `b12x:probe` build is the opposite: it **does** overlay 10 kernels + `b12x==0.23.0` (`scripts/build-glm52-image.sh legacy`). The two variants share the script; only the DCP branch is self-contained.

## The 5 deploy config walls (each fails at init one-by-one)

Fixed by matching CosmicRaisins `recipes/glm52-quanttrio-unpruned-dcp2-320k.yaml`:

1. `ImportError use_b12x_sparse_indexer` → don't overlay the 3 in-tree files (⇒ no kernel overlay).
2. `Sparse Attention Indexer requires DeepGEMM` → `VLLM_USE_B12X_SPARSE_INDEXER=1`.
3. `No valid attention backend` → `--attention-backend B12X_MLA_SPARSE` (not `FLASHMLA_SPARSE`).
4. `B12X_MLA_SPARSE requires index_topk config` → `--hf-overrides '{"index_topk_pattern":"FFFSSS…"}'` (78 chars).
5. `index_topk_fp8() unexpected kwarg out_scores` → b12x `0.30.0`.

Plus env `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_DCP_GLOBAL_TOPK=1`, `VLLM_DCP_SHARD_DRAFT=1`, `TORCH_CUDA_ARCH_LIST=12.1a`; flags `--decode-context-parallel-size 2 --dcp-kv-cache-interleave-size 1 --async-scheduling`; spec-config `draft_attention_backend:B12X_MLA_SPARSE`. Drop the legacy `GLM52_*` / reduced-chunk env (DCP handles memory by sharding).

## The last mile — unified-memory OOM

`0.90/320K` spiked to **116/124 GB in capture → OOM → wedged all 4 nodes** (ping-only; needed a power-cycle). Key facts:
- **GB10 `/dev/shm` is in the unified memory pool** — do NOT raise `--shm-size`; it steals GPU budget and makes OOM worse. The shm-broadcast timeout was a *symptom* of the OOM.
- We run ~5 GB tighter than the (unpruned) reference because we use a **separate MTP drafter** (~5 GB) vs their self-drafting model.
- Fix: **0.89 / 256K first**, climb toward 320K once capture survives. `TRITON_CACHE_DIR` persists the sm12x JIT → faster recapture.

## Next-session checklist
1. Nodes online (verify via API, **not SSH** — see memory `ssh-hammering-wedges-nodes`).
2. Deploy `glm-5.2-awq-15pct-dcp2.yaml` (0.89/256K). If OOM in capture → drop to 0.87 / ~200K; if it serves → climb toward 320K.
3. Validate: 200K needle-in-haystack retrieval + decode/MTP-acceptance on code.
4. Unpruned `QuantTrio/GLM-5.2-Int4-Int8Mix` (406 GB) downloading to NFS for a later quality comparison (self-drafts → no separate MTP → more memory headroom, likely reaches 320K cleaner).
5. Commit the recipe + finish the eugr-fork retirement (pin upstream `08c34dd8`, delete `kreuzhofer/spark-vllm-docker`).
