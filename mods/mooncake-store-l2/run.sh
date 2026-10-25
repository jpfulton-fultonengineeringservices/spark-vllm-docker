#!/bin/bash
#
# [mooncake-store-l2] Standalone Mooncake store as a durable L2 KV cache.
#
# Runs in-container on head node AND every worker before the serve command
# (mods convention: dir run.sh, no args, executed by launch-cluster.sh's
# apply_mod_to_container()). Installs the store connector set, the
# mooncake-transfer-engine + cupy runtime deps the connector imports in-process,
# and renders the `standalone-store` MOONCAKE_CONFIG_PATH JSON that
# MooncakeStoreConnector reads. The store cohort (docker/mooncake-store, brought
# up by cluster-config's launch_mooncake_daemons) owns the shared pool and the
# NVMe SSD tier, so computed KV survives a serving-process restart and is
# re-served by content-hash lookup.
#
# Scope: NON-PD. The recipe serves a single instance (kv_role=kv_both), so there
# is no P2P topology and the P2P connector set is deliberately NOT installed.
#
# Why this mod and not LMCache: the model is a Qwen3_5 hybrid (FullAttentionSpec
# + MambaSpec, HMA required). LMCacheConnectorV1 lacks SupportsHMA and
# LMCacheMPConnector hard-requires HMA OFF (single uniform layout), so both
# boot-crash this model. MooncakeStoreConnector subclasses SupportsHMA and
# handles the Mamba group in align mode -- it is the mamba-capable corridor.
#
# Opt-in: inert unless MOONCAKE_L2_ENABLED=1 (launcher -e) or the exec-script
# already references MOONCAKE_CONFIG_PATH (re-apply on an existing container).
#
# Env (all optional unless noted):
#   MOONCAKE_L2_ENABLED=1             activate (launcher injects it; see
#                                     cluster-config spark-serve.sh).
#   MOONCAKE_MASTER_SERVER_ADDRESS    REQUIRED. Rail B IP of rank 0 (launcher -e).
#   MOONCAKE_MASTER_PORT              master metadata port (default 50051).
#   MOONCAKE_LOCAL_BUFFER_GIB         per-rank TransferEngine buffer (default 4).
#   MOONCAKE_CONFIG_PATH              rendered config path (default
#                                     /tmp/mooncake-store.json; MUST equal the
#                                     recipe env value).
#   MOONCAKE_L2_INSTALL_DEPS          1 (default) installs transfer-engine+cupy.
#   MOONCAKE_L2_TP                    TransferEngine protocol override (default
#                                     rdma); device_name is RDMA-fixed.

set -euo pipefail

PREFIX="[mooncake-store-l2]"
fail() { echo "$PREFIX MOONCAKE_L2_FAIL $*" >&2; return 1; }
info() { echo "$PREFIX $*"; }

# Opt-in. The recipe env is NOT visible here: mods run BEFORE the exec-script
# that exports recipe env, in a separate process. The mod self-activates from
# the exec-script it can read in-container (launch-cluster.sh copies
# /workspace/exec-script.sh before applying mods) by grepping for the config
# pointer the recipe pins. MOONCAKE_L2_ENABLED=1 in the calling env (injected by
# the launcher -e, or exported for bare-mod testing) forces activation.
EXEC_SCRIPT="${WORKSPACE_DIR:-/workspace}/exec-script.sh"
if [ "${MOONCAKE_L2_ENABLED:-0}" = "1" ] \
   || { [ -f "$EXEC_SCRIPT" ] && grep -q 'MOONCAKE_CONFIG_PATH' "$EXEC_SCRIPT" 2>/dev/null; }; then
    :
else
    echo "$PREFIX inert (recipe exec-script does not reference MOONCAKE_CONFIG_PATH; export MOONCAKE_L2_ENABLED=1 to force)"
    return 0 2>/dev/null || exit 0
fi

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOONCAKE_WORKER_DIR="${MOD_DIR}/../mooncake-worker"

# Validate the launcher-injected store endpoint BEFORE the heavy installs, so a
# misconfigured launch fails in seconds instead of after a pip wheel download.
: "${MOONCAKE_MASTER_SERVER_ADDRESS:?MOONCAKE_MASTER_SERVER_ADDRESS required (launcher -e: rank-0 Rail B IP)}"

# --- 1. Install the store connector set (mooncake-worker, store-only) --------
[ -f "${MOONCAKE_WORKER_DIR}/run.sh" ] \
    || fail "reason=missing_mooncake_worker dir=${MOONCAKE_WORKER_DIR} hint='recipe must list mods/mooncake-worker before mods/mooncake-store-l2'"
# Force the store-only install: this mod is non-PD by definition, so the P2P
# facade is dead weight. Overriding here (not relying on the recipe env, which
# cannot reach a mod) keeps the install deterministic.
MOONCAKE_WORKER_P2P=skip bash "${MOONCAKE_WORKER_DIR}/run.sh" install \
    || fail "reason=store_set_install_failed"

# --- 2. Install the transfer-engine + cupy runtime deps ----------------------
# The connector imports `mooncake.store` in-process and _StagingSlotPool needs
# cupy (pinned slot allocs). Neither ships in vllm-node-b12x. Block copied
# verbatim from mods/pd-disagg/run.sh; the pin is load-bearing for GB10: 0.3.13's
# rdma_context gates RNIC open on GPU-RDMA support (DMA-BUF / nvidia-peermem)
# and GB10 has neither, so every RNIC is disabled and engine init fails.
# libnuma1: the mooncake-transfer-engine wheel dlopens it (engine.so/store.so
# DT_NEEDED); the derived coder390 image ships without it, so `import
# mooncake.store` died with "libnuma.so.1: cannot open shared object file"
# AFTER pip install (observed live 2026-10-25 on the qwen3.8-27b-coder390-2x
# store-L2 apply). Install before the import probes; -y + update-tolerant so
# the base apt lists being stale does not wedge the mod.
: "${MOONCAKE_L2_INSTALL_NUMA:=1}"
if ! ldconfig -p 2>/dev/null | grep -q "libnuma.so.1"; then
    if [ "${MOONCAKE_L2_INSTALL_NUMA}" = "1" ] && command -v apt-get >/dev/null 2>&1; then
        info "libnuma.so.1 missing; apt-get installing libnuma1"
        apt-get update >/dev/null 2>&1 || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libnuma1 >/dev/null \
            || fail "op=install_libnuma reason=apt_failed hint='set MOONCAKE_L2_INSTALL_NUMA=0 if the image pre-bakes libnuma, or bake libnuma1 into the image'"
    fi
    ldconfig -p 2>/dev/null | grep -q "libnuma.so.1" \
        || fail "op=verify_libnuma reason=absent hint='libnuma.so.1 must be resolvable for mooncake-transfer-engine'"
    info "libnuma1 installed"
else
    info "libnuma.so.1 already present"
fi
: "${MOONCAKE_L2_INSTALL_DEPS:=1}"
: "${MOONCAKE_L2_MOONCAKE_SPEC:=mooncake-transfer-engine-cuda13==0.3.12.post1}"
if ! python3 -c 'import mooncake' >/dev/null 2>&1; then
    if [ "${MOONCAKE_L2_INSTALL_DEPS}" = "1" ]; then
        info "mooncake module missing; installing '${MOONCAKE_L2_MOONCAKE_SPEC}'"
        python3 -m pip install --no-cache-dir "${MOONCAKE_L2_MOONCAKE_SPEC}" >&2 \
            || fail "op=install_mooncake spec='${MOONCAKE_L2_MOONCAKE_SPEC}' reason=pip_failed"
    fi
    python3 -c 'import mooncake' >/dev/null 2>&1 \
        || fail "op=import_mooncake reason=absent hint='set MOONCAKE_L2_INSTALL_DEPS=1 or bake ${MOONCAKE_L2_MOONCAKE_SPEC} into the image'"
    info "mooncake module installed"
else
    info "mooncake module already present"
fi

if ! python3 -c 'import cupy' >/dev/null 2>&1; then
    if [ "${MOONCAKE_L2_INSTALL_DEPS}" = "1" ]; then
        info "cupy missing; installing 'cupy-cuda13x<14.1.0'"
        python3 -m pip install --no-cache-dir 'cupy-cuda13x<14.1.0' >&2 \
            || fail "op=install_cupy reason=pip_failed"
    fi
    python3 -c 'import cupy' >/dev/null 2>&1 \
        || fail "op=import_cupy reason=absent hint='set MOONCAKE_L2_INSTALL_DEPS=1 or bake cupy-cuda13x into the image'"
    info "cupy installed"
else
    info "cupy already present"
fi

# Fail closed if the connector itself is not importable after install: a boot
# that silently falls back to no connector would look alive but never persist.
python3 -c 'from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector' >/dev/null 2>&1 \
    || fail "reason=connector_missing hint='mooncake-worker install did not place the store set'"

# --- 3. Render the standalone-store config -----------------------------------
# MooncakeStoreConfig.load_from_config() reads MOONCAKE_CONFIG_PATH
# (vendor/store/worker.py) and validates mode=standalone-store =>
# global_segment_size == 0 (the external mooncake client owns the pool and its
# SSD tier). Body mirrors infra/cluster-config/templates/mooncake-store.json.tmpl.
MASTER_PORT="${MOONCAKE_MASTER_PORT:-50051}"
BUFFER_GIB="${MOONCAKE_LOCAL_BUFFER_GIB:-4}"
PROTOCOL="${MOONCAKE_L2_TP:-rdma}"
CONFIG_FILE="${MOONCAKE_CONFIG_PATH:-/tmp/mooncake-store.json}"

cat > "$CONFIG_FILE" <<EOF
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "${MOONCAKE_MASTER_SERVER_ADDRESS}:${MASTER_PORT}",
  "global_segment_size": 0,
  "local_buffer_size": "${BUFFER_GIB}GB",
  "protocol": "${PROTOCOL}",
  "device_name": "roceP2p1s0f1",
  "enable_offload": true
}
EOF

echo "$PREFIX ok config=${CONFIG_FILE} master=${MOONCAKE_MASTER_SERVER_ADDRESS}:${MASTER_PORT} buffer=${BUFFER_GIB}G p2p=skip"
