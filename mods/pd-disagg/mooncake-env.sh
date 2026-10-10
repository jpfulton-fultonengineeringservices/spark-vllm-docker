#!/bin/bash
# [pd-disagg] Mooncake store-only env computation for vLLM ranks.
#
# Sourced by mods/pd-disagg/run.sh. Renders /tmp/pd-disagg/mooncake-store.json
# (the MooncakeStoreConnector config consumed via MOONCAKE_CONFIG_PATH) and
# exports the node-local steering env. Mirrors the proven qwen38 store-only
# deployment (infra/cluster-config/templates/mooncake-store.json.tmpl) with
# two MiMo-specific values:
#   - global_segment_size=0  (vLLM ranks are pure requesters; the DRAM+SSD
#     pool is owned by the sibling mooncake-client container, NOT in-process)
#   - enable_offload=false   (SSD tier is client-owned; vLLM rank offload
#     stays off)
#
# NO UCX_TLS / UCX_NET_DEVICES exports here: Mooncake's engine.so links
# libmlx5 directly (mooncake-store Dockerfile) — UCX is not on the store
# data path, unlike NIXL.
#
# PYTHONHASHSEED=0 is a silent-correctness requirement: hash-based KV dedup
# across ranks/instances must not vary between processes (store connector
# usage doc + store-only prior plan).

set -euo pipefail

PREFIX="[pd-disagg-mooncake]"

fail() {
    echo "$PREFIX PD_DISAGG_FAIL: $1" >&2
    exit 1
}

# --- Required env -----------------------------------------------------------
# MOONCAKE_MASTER_SERVER_ADDRESS: rank-0 host's Rail B IP (serve.sh -e).
: "${MOONCAKE_MASTER_SERVER_ADDRESS:?op=env reason=MOONCAKE_MASTER_SERVER_ADDRESS required (rank-0 Rail B IP)}"
: "${MOONCAKE_MASTER_PORT:?op=env reason=MOONCAKE_MASTER_PORT required (50051)}"

MOONCAKE_CLIENT_PORT="${MOONCAKE_CLIENT_PORT:-50053}"
MOONCAKE_LOCAL_BUFFER_GIB="${MOONCAKE_LOCAL_BUFFER_GIB:-4}"

# --- Node-local Rail B IP ----------------------------------------------------
# VLLM_HOST_IP is set per-rank by launch-cluster.sh (get_env_flags).
THIS_RAILB_IP="${VLLM_HOST_IP:-}"
if [ -z "$THIS_RAILB_IP" ]; then
    THIS_RAILB_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
: "${THIS_RAILB_IP:?op=env reason=cannot resolve node Rail B IP}"

# --- Render mooncake-store.json ---------------------------------------------
MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-/tmp/pd-disagg/mooncake-store.json}"
mkdir -p "$(dirname "$MOONCAKE_CONFIG_PATH")"

cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "${MOONCAKE_MASTER_SERVER_ADDRESS}:${MOONCAKE_MASTER_PORT}",
  "global_segment_size": 0,
  "local_buffer_size": "${MOONCAKE_LOCAL_BUFFER_GIB}GB",
  "protocol": "rdma",
  "device_name": "roceP2p1s0f1",
  "enable_offload": false
}
EOF

# --- Exports -----------------------------------------------------------------
export MOONCAKE_CONFIG_PATH
export MOONCAKE_PREFERRED_SEGMENT="${THIS_RAILB_IP}:${MOONCAKE_CLIENT_PORT}"
export MOONCAKE_REQUESTER_LOCAL_HOSTNAME="${THIS_RAILB_IP}"
export PYTHONHASHSEED=0

echo "$PREFIX mooncake-store.json=$MOONCAKE_CONFIG_PATH"
echo "$PREFIX preferred_segment=$MOONCAKE_PREFERRED_SEGMENT requester_host=$MOONCAKE_REQUESTER_LOCAL_HOSTNAME"
echo "$PREFIX MOONCAKE_ENV_OK master=${MOONCAKE_MASTER_SERVER_ADDRESS}:${MOONCAKE_MASTER_PORT} client=${MOONCAKE_CLIENT_PORT}"
