#!/bin/bash
# [mooncake-worker] Thin launcher for the host-staged Mooncake connector
# installers (see pyproject.toml + src/mooncake_worker/).
#
# Locates the vLLM **site root** (parent of the vllm package — find_spec
# returns the package dir itself, so take its parent), then delegates to the
# package CLI. Mirrors the house mod pattern (fix-qwen3-next-autoround and
# the exl3-pack uv package).
#
# Usage: run.sh [install | verify | install-p2p | verify-p2p | selftest | manifest]
#   (default: install — the store set, THEN the P2P set unless
#   MOONCAKE_WORKER_P2P=skip; install-p2p is idempotent and gates fail
#   closed, so the auto-add is safe for non-PD store users too)
#
# Set VLLM_PACKAGE_ROOT to override the dist-packages root.

set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-install}"

export PYTHONPATH="${MOD_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "${VLLM_PACKAGE_ROOT:-}" ]; then
    VLLM_PACKAGE_ROOT="$(python3 <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    sys.exit("vLLM not importable (is this the vllm-node container?)")
import pathlib
pkg = pathlib.Path(next(iter(spec.submodule_search_locations)))
print(pkg.parent)  # site root: parent of the vllm package dir
PY
)"
fi

# P2P MooncakeConnector set: mods/pd-disagg dispatches P2P KV on its mooncake
# backend, and the stock image ships the pre-decomposition monolith connector
# (no host_staging). Applied alongside the store set by default — idempotent
# (install_p2p short-circuits on p2p_ported state) and fail-closed on gate
# rejection. Opt out with MOONCAKE_WORKER_P2P=skip (store-only users).
if [ "$CMD" = "install" ] && [ "${MOONCAKE_WORKER_P2P:-auto}" != "skip" ]; then
    echo "[mooncake-worker] applying P2P set (MOONCAKE_WORKER_P2P=${MOONCAKE_WORKER_P2P:-auto})"
    python3 -m mooncake_worker.cli install-p2p --site "${VLLM_PACKAGE_ROOT}" || {
        echo "[mooncake-worker] FAIL: install-p2p rejected (gates); aborting" >&2
        exit 1
    }
fi

if [ "$CMD" = "selftest" ] || [ "$CMD" = "manifest" ]; then
    exec python3 -m mooncake_worker.cli "$CMD"
fi

exec python3 -m mooncake_worker.cli "$CMD" --site "${VLLM_PACKAGE_ROOT}"
