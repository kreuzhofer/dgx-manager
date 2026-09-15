#!/usr/bin/env python3
"""
Widen vLLM's PLE quant-method gate so an NVFP4-body / FP8-PLE Qwen4Exp
checkpoint loads.

THE BUG. `_get_ple_embedding_quant_method()` in
`vllm/models/qwen4_exp/nvidia/ple_layer.py` returns an FP8 embedding method only
for `ModelOptMixedPrecisionConfig` (when the prefix resolves to FP8) or a
serialized `Fp8Config`. A checkpoint whose BODY is NVFP4 yields
`ModelOptNvFp4Config`, matches neither, and the PLE is built UNQUANTIZED — so
loading dies with:

    ValueError: There is no module or parameter named
    'ngram_embedding.weight_scale' in Qwen4ExpNGramEmbedding

Upstream: https://github.com/vllm-project/vllm/issues/54765

WHY THIS IS CORRECT, NOT A WORKAROUND. Verified against the staged checkpoint
`RadixArk/Qwen3.8-Flash-Next-NVFP4` by reading the safetensors headers directly:
128 `...ngram_embedding.shard_N.weight` tensors are **F8_E4M3**, plus one BF16
`ngram_embedding.weight_scale` of shape [1] — a single global scale. They live
in a shard family literally named `model-plefp8-*`. The table IS FP8, so
`Qwen4ExpPLEFp8EmbeddingMethod` is the right method for it.

The checkpoint's `exclude_modules: ["*.ple.*"]` means "not NVFP4" — not
"unquantized". vLLM reads the NVFP4 config, sees the PLE excluded, and wrongly
concludes there is nothing to dequantize.

OPT-IN ON PURPOSE. Gated behind `QWEN4EXP_PLE_FP8_GATE=1` rather than widening
the branch unconditionally: an NVFP4 checkpoint with a genuinely BF16 PLE would
then have that table misread as FP8, which produces fluent, plausible, WRONG
output rather than an error. Only a recipe that has verified its checkpoint's
PLE dtype should set the variable.
"""
import re
import sys

TARGET = sys.argv[1]
MARKER = "# [qwen4exp-ple-fp8-gate]"

src = open(TARGET, encoding="utf-8").read()

if MARKER in src:
    print("[qwen4exp-ple-fp8-gate] already applied — no-op")
    sys.exit(0)

# 1. import ModelOptNvFp4Config alongside ModelOptMixedPrecisionConfig
imp_old = ("from vllm.model_executor.layers.quantization.modelopt import (\n"
           "    ModelOptMixedPrecisionConfig,\n)")
imp_new = ("from vllm.model_executor.layers.quantization.modelopt import (\n"
           "    ModelOptMixedPrecisionConfig,\n    ModelOptNvFp4Config,\n)")
if imp_old not in src:
    print("[qwen4exp-ple-fp8-gate] FAILED: import block not found as expected", file=sys.stderr)
    sys.exit(1)
src = src.replace(imp_old, imp_new, 1)

# 2. insert the opt-in branch at the top of the gate body
anchor = '    """Select global-scale FP8 only for quantized PLE checkpoint shards."""\n'
if anchor not in src:
    print("[qwen4exp-ple-fp8-gate] FAILED: gate docstring not found", file=sys.stderr)
    sys.exit(1)
branch = anchor + (
    "\n"
    "    " + MARKER + " NVFP4 body with an FP8 PLE table (vllm#54765).\n"
    "    # Opt-in: the caller must have verified the PLE tensors really are FP8.\n"
    "    import os as _os\n"
    "    if _os.environ.get(\"QWEN4EXP_PLE_FP8_GATE\") == \"1\" and isinstance(\n"
    "        quant_config, ModelOptNvFp4Config\n"
    "    ):\n"
    "        return Qwen4ExpPLEFp8EmbeddingMethod()\n"
)
src = src.replace(anchor, branch, 1)

open(TARGET, "w", encoding="utf-8").write(src)
print("[qwen4exp-ple-fp8-gate] patch applied")
