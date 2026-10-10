#!/bin/bash
# [mooncake-worker] Install the host-staged Mooncake store connector worker
# into the running vLLM tree.
#
# Source: mods/mooncake-worker/worker.py — the store/worker.py from
# Fulton-Engineering-Services/vllm @ cuda13.3-aarch64-gb10-glm5next-modular
# tip (4e79d6be6f), md5 b11bd659baf724298b2f347dad503b4e (2574 lines).
#
# SHA-gated against the KNOWN image versions:
#   - installed == b11bd659...  → already this worker (skip)
#   - installed == 13aa4784...  → stock e93769fd9b connector shipped in
#     vllm-node-b12x (valid replacement target)
#   - anything else             → image drift; fail closed with instructions
#
# Replace-in-place via atomic rename (same filesystem).

set -euo pipefail

PREFIX="[mooncake-worker]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
TARGET="${PYTHON_ROOT}/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${MOD_DIR}/worker.py"

MD5_TARGET_FINAL="b11bd659baf724298b2f347dad503b4e"   # branch tip (this mod)
MD5_STOCK_E937="13aa4784d38f8c1ff0d9284d0033a335"     # shipped in image (e93769fd9b)

[ -f "$SRC" ] || { echo "$PREFIX FAIL: $SRC missing" >&2; exit 1; }
[ -f "$TARGET" ] || { echo "$PREFIX FAIL: $TARGET missing (vLLM not installed?)" >&2; exit 1; }

md5_of() { md5sum "$1" 2>/dev/null | awk '{print $1}' || md5 -q "$1"; }

CUR="$(md5_of "$TARGET")"
SRC_MD5="$(md5_of "$SRC")"

[ "$SRC_MD5" = "$MD5_TARGET_FINAL" ] || { echo "$PREFIX FAIL: mod worker.py md5=$SRC_MD5, expected $MD5_TARGET_FINAL (mod corrupted?)" >&2; exit 1; }

if [ "$CUR" = "$MD5_TARGET_FINAL" ]; then
    echo "$PREFIX OK: already installed ($CUR)"
    exit 0
fi

if [ "$CUR" != "$MD5_STOCK_E937" ]; then
    echo "$PREFIX FAIL: installed worker md5=$CUR is not a known base" >&2
    echo "$PREFIX       expected one of: $MD5_STOCK_E937 (stock image)" >&2
    echo "$PREFIX                             $MD5_TARGET_FINAL (already ported)" >&2
    echo "$PREFIX       action: re-port mods/mooncake-worker/worker.py against the" >&2
    echo "$PREFIX               new image's store/worker.py; do NOT blind-overwrite." >&2
    exit 1
fi

# Verify the patched source parses before clobbering the live file.
python3 -c "import ast; ast.parse(open('$SRC').read())" \
    || { echo "$PREFIX FAIL: $SRC does not parse" >&2; exit 1; }

TMP="$(mktemp "$(dirname "$TARGET")/.worker.XXXXXX")"
cp "$SRC" "$TMP"
mv -f "$TMP" "$TARGET"

NEW="$(md5_of "$TARGET")"
[ "$NEW" = "$MD5_TARGET_FINAL" ] || { echo "$PREFIX FAIL: post-install md5=$NEW mismatch" >&2; exit 1; }
echo "$PREFIX OK: installed host-staged worker (was $MD5_STOCK_E937, now $MD5_TARGET_FINAL)"
