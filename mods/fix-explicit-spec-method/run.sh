#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/fix-explicit-spec-method.patch"
PREFIX="[fix-explicit-spec-method]"

if ! command -v git >/dev/null 2>&1; then
  echo "$PREFIX git is required to apply this mod." >&2
  exit 1
fi

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "$PREFIX vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

cd "$PYTHON_ROOT"

if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "$PREFIX Patch is already applied; skipping."
elif git apply --check "$PATCH_FILE"; then
  git apply "$PATCH_FILE"
  echo "$PREFIX Applied explicit-spec-method detection guard."
else
  echo "$PREFIX Patch could not be applied to installed vLLM." >&2
  echo "$PREFIX The speculative.py auto-detect anchor moved; re-derive the hunk." >&2
  exit 1
fi

echo "=====> Explicit speculative method is now honored; auto-detect runs only when method is unset."
