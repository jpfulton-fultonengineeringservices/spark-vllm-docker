#!/bin/bash
# [mooncake-worker] Install the host-staged Mooncake **store connector package**
# into the running vLLM tree.
#
# Source: Fulton-Engineering-Services/vllm @ cuda13.3-aarch64-gb10-glm5next-modular
# tip (4e79d6be6f) — the full store/ package (8 files) as a matched set:
#   __init__.py connector.py coordinator.py data.py metrics.py protocol.py
#   scheduler.py worker.py
#
# Why the WHOLE package, not just worker.py: the FES worker references
# ReqMeta.partial_tail_offloads, but the image's stock data.py uses
# boundary_state_offloads (b12x/DiffKV lineage). A lone-worker swap would
# AttributeError. worker.py + data.py (+ the coordinator/scheduler that build
# ReqMeta) are one coherent set and must move together.
#
# SHA-gated against the KNOWN stock image package:
#   - all files already == this mod's set  → idempotent skip
#   - stock worker.py == b180493c...       → stock image (valid target)
#   - anything else                        → image drift; fail closed
#
# Install: copy each file with atomic rename (same filesystem).

set -euo pipefail

PREFIX="[mooncake-worker]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
STORE_DIR="${PYTHON_ROOT}/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store"

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${MOD_DIR}/store"

# Stock image (vllm-node-b12x:latest, vLLM 0.1.dev21546+g502d6cb5a) worker
# hash — the whole-package replacement is gated on this.
MD5_STOCK_WORKER="b180493c964225f6a9282b30a47fb967"
# This mod's worker hash (branch tip 4e79d6be6f).
MD5_MOD_WORKER="b11bd659baf724298b2f347dad503b4e"

FILES=(__init__.py connector.py coordinator.py data.py metrics.py protocol.py scheduler.py worker.py)

md5_of() { md5sum "$1" 2>/dev/null | awk '{print $1}' || md5 -q "$1"; }

[ -d "$STORE_DIR" ] || { echo "$PREFIX FAIL: $STORE_DIR missing (vLLM not installed?)" >&2; exit 1; }
[ -d "$SRC_DIR" ]   || { echo "$PREFIX FAIL: $SRC_DIR missing" >&2; exit 1; }

# Integrity: every mod file present and parses.
for f in "${FILES[@]}"; do
    [ -f "$SRC_DIR/$f" ] || { echo "$PREFIX FAIL: mod missing store/$f" >&2; exit 1; }
    python3 -c "import ast; ast.parse(open('$SRC_DIR/$f').read())" \
        || { echo "$PREFIX FAIL: mod store/$f does not parse" >&2; exit 1; }
done
[ "$(md5_of "$SRC_DIR/worker.py")" = "$MD5_MOD_WORKER" ] \
    || { echo "$PREFIX FAIL: mod worker.py md5 mismatch (expected $MD5_MOD_WORKER)" >&2; exit 1; }

CUR_WORKER="$(md5_of "$STORE_DIR/worker.py")"

if [ "$CUR_WORKER" = "$MD5_MOD_WORKER" ]; then
    echo "$PREFIX OK: already installed (worker md5=$CUR_WORKER)"
    exit 0
fi

if [ "$CUR_WORKER" != "$MD5_STOCK_WORKER" ]; then
    echo "$PREFIX FAIL: installed worker md5=$CUR_WORKER is not a known base" >&2
    echo "$PREFIX       expected one of: $MD5_STOCK_WORKER (stock image)" >&2
    echo "$PREFIX                             $MD5_MOD_WORKER (already ported)" >&2
    echo "$PREFIX       action: re-port mods/mooncake-worker/store/ against the new" >&2
    echo "$PREFIX               image's store/ package; do NOT blind-overwrite." >&2
    exit 1
fi

# Install the matched set atomically.
for f in "${FILES[@]}"; do
    tmp="$(mktemp "$STORE_DIR/.$f.XXXXXX")"
    cp "$SRC_DIR/$f" "$tmp"
    mv -f "$tmp" "$STORE_DIR/$f"
done

NEW_WORKER="$(md5_of "$STORE_DIR/worker.py")"
[ "$NEW_WORKER" = "$MD5_MOD_WORKER" ] \
    || { echo "$PREFIX FAIL: post-install worker md5=$NEW_WORKER mismatch" >&2; exit 1; }
echo "$PREFIX OK: installed host-staged store package (8 files; worker $MD5_STOCK_WORKER -> $MD5_MOD_WORKER)"
