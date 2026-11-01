#!/usr/bin/env bash
# fix-mfu-metrics (v1): port the FES vllm fork's MFU corrections into the
# installed b12x vLLM tree (vllm/v1/metrics/perf.py).
#
# Fixes (FES fork commits e1f55c271, 20bf55f2c, dbf11e81f, 48f9cf884,
# 142e8949f, branch mooncake-host-staging):
#   1. Attention/FFN quant parsers raised InvalidComponent for
#      modelopt_mixed -> attention/FFN/unembed dropped from MFU entirely.
#   2. Hybrid Qwen3.5 models (48x linear_attention + 16x full_attention)
#      billed all 64 layers as attention; GDN layers contributed zero.
#   3. _QUANT_WEIGHT_BYTE_SIZE lacked modelopt_mixed (+4 newer methods);
#      UnembedMetrics.weight_byte_size was strictly int.
#
# patch_perf.py holds the edit list. Idempotent via a sentinel line
# embedded in perf.py; anchor-count-gated (drift aborts with the anchor
# name); py_compile post-check runs on a temp copy and the installed file
# is swapped in atomically, so it is never left partially patched.
#
# Opt-in: inert unless FIX_MFU_METRICS=1 is in the mod's calling env or the
# recipe's exec-script references FIX_MFU_METRICS. apply_mod_to_container runs
# with WORKSPACE_DIR=$PWD = the container WORKDIR (/workspace/vllm), NOT the
# exec-script's location, so grep the canonical /workspace/exec-script.sh path
# directly. run-recipe.py renders the recipe env block into that script
# before launch-cluster.sh applies mods, so the flag is visible here.
set -euo pipefail

PREFIX="[fix-mfu-metrics]"
fail() { echo "$PREFIX FAIL $*" >&2; return 1; }
info() { echo "$PREFIX $*"; }

EXEC_SCRIPT="/workspace/exec-script.sh"
if [ "${FIX_MFU_METRICS:-0}" != "1" ] \
   && { [ ! -f "$EXEC_SCRIPT" ] || ! grep -q 'FIX_MFU_METRICS' "$EXEC_SCRIPT" 2>/dev/null; }; then
    echo "$PREFIX inert (set FIX_MFU_METRICS=1 to activate)"
    return 0 2>/dev/null || exit 0
fi

cd "$(dirname "$0")"
python3 patch_perf.py || fail "op=patch_perf_metrics"

info "ok (MFU corrections active)"
