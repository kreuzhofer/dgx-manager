#!/bin/bash
set -euo pipefail

PREFIX="[qwen4exp-ple-fp8-gate]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON_ROOT="/usr/local/lib/python3.12/dist-packages"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-$DEFAULT_PYTHON_ROOT}}"
TARGET="$PYTHON_ROOT/vllm/models/qwen4_exp/nvidia/ple_layer.py"

echo "=== Qwen4Exp PLE FP8 quant-gate mod (vllm#54765) ==="

if [[ ! -f "$TARGET" ]]; then
    echo "$PREFIX FAILED: $TARGET not found — wrong image or vLLM layout changed." >&2
    exit 1
fi

python3 "$MOD_DIR/patch_ple_gate.py" "$TARGET"

# Verify the module still imports; a syntactically broken patch must fail HERE,
# loudly, rather than deep inside engine startup where it reads as a model bug.
python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('_ple_check', '$TARGET')
if spec is None or spec.loader is None:
    sys.exit('$PREFIX FAILED: cannot load patched module spec')
" || { echo "$PREFIX FAILED: patched file does not parse" >&2; exit 1; }
python3 -m py_compile "$TARGET" || { echo "$PREFIX FAILED: py_compile rejected the patched file" >&2; exit 1; }

if [[ "${QWEN4EXP_PLE_FP8_GATE:-}" != "1" ]]; then
    echo "$PREFIX patch installed but INERT (QWEN4EXP_PLE_FP8_GATE is not 1)."
else
    echo "$PREFIX ACTIVE — NVFP4-body checkpoints will build the PLE as FP8."
fi
