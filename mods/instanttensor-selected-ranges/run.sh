#!/bin/bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Locate without importing vLLM or initializing CUDA during container setup.
if [[ -z "${VLLM_PACKAGE_ROOT:-}" ]]; then
    if [[ -n "${VLLM_SITE_PACKAGES:-}" ]]; then
        VLLM_PACKAGE_ROOT="$VLLM_SITE_PACKAGES/vllm"
    elif [[ -n "${PYTHON_ROOT:-}" ]]; then
        VLLM_PACKAGE_ROOT="$PYTHON_ROOT/vllm"
    else
        VLLM_PACKAGE_ROOT=$(python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("[instanttensor-selected-ranges] vLLM package not found")
print(next(iter(spec.submodule_search_locations)))
PY
        )
    fi
fi

python3 "$MOD_DIR/patch_weight_utils.py" \
    "$VLLM_PACKAGE_ROOT/model_executor/model_loader/weight_utils.py"
