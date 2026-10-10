#!/bin/bash
# [mooncake-worker] Thin launcher for the host-staged Mooncake store connector
# installer (see pyproject.toml + src/mooncake_worker/).
#
# Locates the vLLM package root (no vLLM import needed), then delegates to the
# package CLI. Mirrors the house mod pattern (fix-qwen3-next-autoround and the
# exl3-pack uv package).
#
# Usage: run.sh [install | verify | selftest | manifest]   (default: install)
#
# Set VLLM_SITE_PACKAGES to override the dist-packages root.

set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-install}"

export PYTHONPATH="${MOD_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [ "$CMD" = "selftest" ] || [ "$CMD" = "manifest" ]; then
    exec python3 -m mooncake_worker.cli "$CMD"
fi

if [ -z "${VLLM_PACKAGE_ROOT:-}" ]; then
    VLLM_PACKAGE_ROOT="$(python3 <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    sys.exit("vLLM not importable (is this the vllm-node container?)")
print(next(iter(spec.submodule_search_locations)))
PY
)"
fi

exec python3 -m mooncake_worker.cli "$CMD" --site "${VLLM_PACKAGE_ROOT}"
