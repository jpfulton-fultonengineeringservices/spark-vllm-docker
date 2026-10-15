#!/bin/bash
#
# [mamba-spec-relocate-guard] Make the Mamba-"align" speculative-block
# relocation tolerant of a hashed/shared block.
#
# Problem (observed live 2026-10-15 on qwen3.8-27b-coder390, DFlash2 spec
# decode + --mamba-cache-mode align + a KV connector): on the first external
# (store) prefix hit the scheduler re-allocates blocks for an already-running
# request, and single_type_kv_cache_manager._relocate_speculative_block
# asserts that each speculative scratch block is exclusively owned and
# unhashed:
#
#     assert self.block_pool.is_block_writable(block), (
#         "Speculative Mamba blocks must be exclusively owned and unhashed "
#         "before relocation")
#
# A KV connector (e.g. MooncakeStoreConnector) can expose blocks that are
# hashed/shared at exactly that point, so the assert fires and kills
# EngineCore -> HTTP 500 EngineDeadError. The core path predates the
# connector and never hits it without one.
#
# Fix: when the block is not writable, hand the request a FRESH scratch block
# instead of asserting. The relocation's contract (append one block, null the
# slot, keep len(req_blocks) growth at +1) is preserved; the request simply
# re-derives that speculative scratch on the current step. Idempotent,
# anchor-count-gated, and a no-op if the target code is absent.
#
# Opt-in: inert unless MAMBA_SPEC_RELOCATE_GUARD=1 (the store-L2 recipe's env
# block sets it). Mods run before the exec-script, so the launcher also
# injects it as a container -e for determinism at mod-apply time.

set -euo pipefail

PREFIX="[mamba-spec-relocate-guard]"
fail() { echo "$PREFIX FAIL $*" >&2; return 1; }
info() { echo "$PREFIX $*"; }

# Opt-in: the recipe env is exported only in the exec-script, which a mod
# cannot read as its own env (separate process). Match the house pattern:
# self-activate from the exec-script's reference to the guard, or from the
# calling env (launcher -e / debug).
EXEC_SCRIPT="${WORKSPACE_DIR:-/workspace}/exec-script.sh"
if [ "${MAMBA_SPEC_RELOCATE_GUARD:-0}" = "1" ] \
   || { [ -f "$EXEC_SCRIPT" ] && grep -q 'MAMBA_SPEC_RELOCATE_GUARD' "$EXEC_SCRIPT" 2>/dev/null; }; then
    :
else
    echo "$PREFIX inert (exec-script does not reference MAMBA_SPEC_RELOCATE_GUARD; export MAMBA_SPEC_RELOCATE_GUARD=1 to force)"
    return 0 2>/dev/null || exit 0
fi

# --- patch the installed vLLM tree in-container ------------------------------
# Mirrors mods/lmcache-offload/run.sh's python-heredoc patch style: locate the
# module by import, apply an anchor-count-gated string replacement, idempotent.
python3 - <<'PYEOF' || fail "op=patch_mamba_relocate"
import importlib.util
import sys

spec = importlib.util.find_spec("vllm.v1.core.single_type_kv_cache_manager")
if spec is None or not spec.origin:
    sys.exit("vllm.v1.core.single_type_kv_cache_manager not importable")

path = spec.origin
src = open(path).read()

MARK = "# [mamba-spec-relocate-guard]"
if MARK in src:
    print(f"[mamba-spec-relocate-guard] already patched: {path}")
    sys.exit(0)

anchor = (
    "        assert self.block_pool.is_block_writable(block), (\n"
    '            "Speculative Mamba blocks must be exclusively owned and unhashed "\n'
    '            "before relocation"\n'
    "        )\n"
)
if src.count(anchor) != 1:
    sys.exit(
        f"anchor count {src.count(anchor)} != 1 in {path} "
        "(vLLM core drifted; re-anchor the guard)"
    )

replacement = (
    "        if not self.block_pool.is_block_writable(block):\n"
    f"            {MARK} a hashed/shared block here is a cached-prefix\n"
    f"            {MARK} block a KV connector exposed mid-flight, not\n"
    f"            {MARK} exclusive scratch. Asserting kills EngineCore;\n"
    f"            {MARK} hand the request a fresh scratch block instead.\n"
    "            block = self.block_pool.get_new_blocks(1)[0]\n"
)
src = src.replace(anchor, replacement)
open(path, "w").write(src)
print(f"[mamba-spec-relocate-guard] patched {path}")
PYEOF

info "ok (guard active)"
