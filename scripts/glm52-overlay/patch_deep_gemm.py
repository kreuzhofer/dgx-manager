import sys
F = "/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py"
MARK = "# --- GLM52 SM12x DSA fallback re-bind ---"
s = open(F).read()
if MARK in s:
    print("deep_gemm already patched"); sys.exit(0)
block = '''

# --- GLM52 SM12x DSA fallback re-bind ---
# GB10 (sm_121, capability family 120) lacks the DeepGEMM mqa_logits kernels.
# Route the DeepGEMM-only DSA entry points to the public sm12x fallbacks and
# stop gating on has_deep_gemm(). Runs at module import, before any
# `from vllm.utils.deep_gemm import ...` consumer binds these names.
try:
    from vllm.platforms import current_platform as _glm_cp
    if _glm_cp.is_cuda() and _glm_cp.is_device_capability_family(120):
        import torch as _glm_torch
        from vllm.v1.attention.ops.deepseek_v4_ops import (
            sm12x_deep_gemm_fallbacks as _glm_sm12x,
        )
        # signatures match the sm12x dispatchers directly
        fp8_fp4_mqa_logits = _glm_sm12x._fp8_mqa_logits_sm12x
        tf32_hc_prenorm_gemm = _glm_sm12x._tf32_hc_prenorm_gemm_sm12x

        # stock paged sig has extra schedule_metadata + clean_logits args the
        # sm12x dispatcher does not take; the sm12x paged kernel schedules
        # internally, so drop them.
        def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens,
                                     block_tables, schedule_metadata,
                                     max_model_len, clean_logits):  # noqa: F811
            return _glm_sm12x._fp8_paged_mqa_logits_sm12x(
                q, kv_cache, weights, context_lens, block_tables, max_model_len,
            )

        # the indexer fills scheduler_metadata_buffer[:] = this(); the sm12x
        # paged path ignores schedule_metadata, so a zero scalar is harmless and
        # broadcasts into the pre-allocated buffer regardless of its shape.
        def get_paged_mqa_logits_metadata(context_lens, block_size, num_sms):  # noqa: F811
            return _glm_torch.zeros((), dtype=_glm_torch.int32,
                                    device=context_lens.device)

        # DeepGEMM layout helpers also _missing on sm_121; the sm12x kernels
        # use 128-element blocks and need no TMA alignment.
        def get_mk_alignment_for_contiguous_layout():  # noqa: F811
            return [128, 128]

        def get_col_major_tma_aligned_tensor(x):  # noqa: F811
            return x

        def is_deep_gemm_supported() -> bool:  # noqa: F811
            return True

        def has_deep_gemm() -> bool:  # noqa: F811
            return True

        import vllm.logger as _glm_log
        _glm_log.init_logger(__name__).info(
            "GLM52: sm12x DSA fallback re-bind active "
            "(mqa_logits/paged/metadata/tf32/mk_align/tma + has_deep_gemm)"
        )
except Exception as _glm_e:  # pragma: no cover
    import vllm.logger as _glm_log
    _glm_log.init_logger(__name__).warning("GLM52 sm12x re-bind failed: %s", _glm_e)
'''
open(F, "w").write(s + block)
print("deep_gemm patched: 4 DSA fns + gate re-bound")
