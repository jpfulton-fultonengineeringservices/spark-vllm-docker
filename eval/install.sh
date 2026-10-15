#!/bin/bash
#
# eval/install.sh - folder-resident uv venv for the lm-evaluation-harness rig
#
# Creates eval/.venv and installs the harness (editable, from the sibling
# submodule gpu-cluster-forks/lm-evaluation-harness) with the api, tasks and
# zeno extras, plus transformers (tokenizers, without the torch-pulling hf
# extra) and pillow (MMMU image decode).
#
# Usage:
#   eval/install.sh [--refresh]
#
# EVAL_WITH_VLLM=1 additionally installs the [vllm] extra - only works on a
# Linux/NVIDIA host (vLLM ships no macOS/arm64 wheels); on this macOS
# workstation the venv runs evals over the network via HTTP backends.
#
# Idempotent: skips install when .venv/bin/lm-eval exists unless --refresh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
EVAL_DIR="$SCRIPT_DIR"
VENV="$EVAL_DIR/.venv"
HARNESS_DIR="$(cd "$EVAL_DIR/../.." && pwd -P)/lm-evaluation-harness"

REFRESH=0
if [ "${1:-}" = "--refresh" ]; then REFRESH=1; fi

command -v uv >/dev/null 2>&1 || {
    echo "error: uv not found; install it first (curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
    exit 1
}
[ -d "$HARNESS_DIR" ] || {
    echo "error: harness submodule missing at $HARNESS_DIR" >&2
    echo "       run: git submodule update --init gpu-cluster-forks/lm-evaluation-harness" >&2
    exit 1
}

if [ "$REFRESH" -eq 0 ] && [ -x "$VENV/bin/lm-eval" ]; then
    echo "[install.sh] venv already present: $VENV (use --refresh to reinstall)"
    exit 0
fi

echo "[install.sh] creating venv $VENV"
uv venv --python 3.12 "$VENV"

echo "[install.sh] installing harness extras: api,tasks,zeno (+transformers, pillow, psycopg)"
uv pip install --python "$VENV/bin/python" \
    -e "$HARNESS_DIR[api,tasks,zeno]" \
    transformers pillow "psycopg[binary]"

if [ "${EVAL_WITH_VLLM:-0}" = "1" ]; then
    echo "[install.sh] EVAL_WITH_VLLM=1: installing [vllm] extra (Linux/NVIDIA only)"
    uv pip install --python "$VENV/bin/python" -e "$HARNESS_DIR[vllm]"
fi

"$VENV/bin/python" -c "import lm_eval; print('[install.sh] lm_eval', lm_eval.__version__)"
echo "[install.sh] done: $VENV/bin/lm-eval"
