#!/bin/bash
#
# [pd-disagg] Role-aware PD launcher for MiMo-V2.6 EXL3 prefill + source decode.
#
# Invoked by recipes/pd-disagg-mimo-exl3.yaml as the recipe command. The
# launcher (run-recipe.py -> launch-cluster.sh) appends the outer-engine
# multi-node args to this script on every node (see make_node_script /
# exec_no_ray_cluster):
#
#     dispatch.sh --nnodes N --node-rank R --master-addr HEAD --master-port P [--headless]
#
# Role map (3-node Config C, tmp/3node-serving-config-proposals.md):
#   rank 0          -> EXL3 prefill instance  (tensor-parallel 1, kv_producer)
#   ranks 1..N-1    -> source decode instance (tensor-parallel 2 sub-group,
#                                               kv_consumer; DFlash drafter decode-side)
#   shared node     -> toy_proxy_server router (rank 0, see PD_ROUTER_*)
#
# TP-TRIMMING FOOTGUN: launch-cluster.sh's parse_parallelism_from_text()
# scans the recipe *command text* for literal '-tp N' / '--tensor-parallel-size N'
# and trims the node count (last value wins). This script therefore NEVER
# carries those flags with inline integers in the recipe command; TP values
# travel exclusively via env (PD_TP_PREFILL / PD_TP_DECODE) and variables.
#
# Requires the env published by mods/pd-disagg/run.sh (UCX_NET_DEVICES ->
# RoCE HCAs, VLLM_NIXL_SIDE_CHANNEL_PORT). Inert unless PD_DISAGG_ENABLED=1.

set -euo pipefail

PREFIX="[pd-disagg]"

# --- 0. opt-in gate (mirror run.sh) -------------------------------------
if [ "${PD_DISAGG_ENABLED:-0}" != "1" ]; then
    echo "$PREFIX inert (set PD_DISAGG_ENABLED=1 to enable)"
    exit 0
fi

# --- 1. load env published by mods/pd-disagg/run.sh ---------------------
ENV_FILE="${PD_DISAGG_ENV_FILE:-/tmp/pd-disagg.env}"
if [ -f "$ENV_FILE" ]; then
    # shellcheck source=/dev/null
    . "$ENV_FILE"
else
    echo "$PREFIX PD_DISAGG_FAIL op=env reason=missing_file file=$ENV_FILE (mods/pd-disagg/run.sh must run first)" >&2
    exit 1
fi

# --- 2. parse launcher-appended outer-engine args -----------------------
NNODES=""
NODE_RANK=""
MASTER_ADDR=""
MASTER_PORT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --nnodes) NNODES="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --headless) shift ;;
        *)
            echo "$PREFIX PD_DISAGG_FAIL op=argparse reason=unknown_arg arg=$1" >&2
            exit 1
            ;;
    esac
done
: "${NNODES:?op=argparse reason=missing_--nnodes}"
: "${NODE_RANK:?op=argparse reason=missing_--node-rank}"

# --- 3. TP values: variables/env only (no literal -tp N) ----------------
TP_PREFILL="${PD_TP_PREFILL:-1}"
TP_DECODE="${PD_TP_DECODE:-2}"

# This node's routable IP (launcher always sets VLLM_HOST_IP per node).
NODE_IP="${VLLM_HOST_IP:-${MASTER_ADDR:-}}"
if [ -z "$NODE_IP" ]; then
    NODE_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
: "${NODE_IP:?op=env reason=no_host_ip}"

# NIXL side channel: bind on this node's own IP (prefill and decode each
# run their side-channel server; docs: VLLM_NIXL_SIDE_CHANNEL_HOST=<own IP>).
export VLLM_NIXL_SIDE_CHANNEL_HOST="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_PORT="${PD_NIXL_PORT:-5600}"

KV_PRODUCER='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
KV_CONSUMER='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

# Backend command (default plain vllm; override, e.g. "b12x-kcache exec vllm serve").
VLLM_SERVE=(${PD_VLLM_SERVE:-vllm serve})

# --- 4. rank 0: EXL3 prefill instance (TP=1, kv_producer) ---------------
if [ "$NODE_RANK" -eq 0 ]; then
    echo "$PREFIX role=prefill rank=$NODE_RANK tp=$TP_PREFILL kv_role=kv_producer nixl_host=$NODE_IP:$VLLM_NIXL_SIDE_CHANNEL_PORT"

    # Router on the shared node (rank 0): background start after a delay so
    # the prefill engine is up before toy_proxy_server dials it.
    if [ "${PD_ROUTER_ENABLED:-0}" = "1" ]; then
        ROUTER_SCRIPT="${PD_ROUTER_SCRIPT:-/workspace/toy_proxy_server.py}"
        if [ -f "$ROUTER_SCRIPT" ]; then
            (
                sleep "${PD_ROUTER_START_DELAY:-30}"
                exec python3 "$ROUTER_SCRIPT" \
                    --port "${PD_ROUTER_PORT:-8000}" \
                    --prefiller-hosts "$NODE_IP" \
                    --prefiller-ports "${PD_PREFILL_PORT:-8100}" \
                    --decoder-hosts "${PD_DECODE_NODE_IPS:-${PD_DECODE_MASTER_ADDR:-}}" \
                    --decoder-ports "${PD_DECODE_PORT:-8200}"
            ) >> /tmp/pd-router.log 2>&1 &
            echo "$PREFIX router boot scheduled (delay=${PD_ROUTER_START_DELAY:-30}s, log=/tmp/pd-router.log)"
        else
            echo "$PREFIX WARN router enabled but script not found: $ROUTER_SCRIPT (set PD_ROUTER_SCRIPT)" >&2
        fi
    fi

    exec "${VLLM_SERVE[@]}" "${PD_EXL3_MODEL:-XiaomiMiMo/MiMo-V2.6-Flash-RL-EXL3}" \
        --host 0.0.0.0 \
        --port "${PD_PREFILL_PORT:-8100}" \
        --tensor-parallel-size "$TP_PREFILL" \
        --quantization exl3 \
        --trust-remote-code \
        --max-model-len "${PD_MAX_MODEL_LEN:-1048576}" \
        --attention-backend B12X \
        --linear-backend b12x \
        --block-size "${PD_BLOCK_SIZE:-128}" \
        --kv-cache-dtype fp8 \
        --kv-cache-dtype-skip-layers sliding_window \
        --max-num-seqs "${PD_MAX_NUM_SEQS:-32}" \
        --max-num-batched-tokens "${PD_MAX_NUM_BATCHED_TOKENS:-16384}" \
        --gpu-memory-utilization "${PD_PREFILL_GMU:-0.72}" \
        --enable-chunked-prefill \
        --async-scheduling \
        --enable-prefix-caching \
        --kv-cache-metrics \
        --enable-mfu-metrics \
        --kv-transfer-config "$KV_PRODUCER"
fi

# --- 5. ranks 1..N-1: source decode instance (TP=2 sub-group, kv_consumer)
NODE_COUNT="${NNODES:-0}"
if [ "$NODE_COUNT" -lt 3 ]; then
    echo "$PREFIX PD_DISAGG_FAIL op=topology reason=pd_needs_3_nodes nnodes=$NODE_COUNT rank=$NODE_RANK" >&2
    exit 1
fi
DECODE_NNODES=$(( NODE_COUNT - 1 ))
if [ "$DECODE_NNODES" -ne "$TP_DECODE" ]; then
    echo "$PREFIX PD_DISAGG_FAIL op=topology reason=decode_nodes_must_equal_tp decode_nnodes=$DECODE_NNODES tp_decode=$TP_DECODE" >&2
    exit 1
fi
DECODE_RANK=$(( NODE_RANK - 1 ))

# Decode sub-group master = rank 1 (this node's own IP when DECODE_RANK=0);
# decode workers need the master's address via PD_DECODE_MASTER_ADDR.
DECODE_MASTER_ADDR="${PD_DECODE_MASTER_ADDR:-}"
if [ "$DECODE_RANK" -eq 0 ]; then
    DECODE_MASTER_ADDR="$NODE_IP"
fi
: "${DECODE_MASTER_ADDR:?op=env reason=PD_DECODE_MASTER_ADDR_required_on_decode_workers}"

echo "$PREFIX role=decode rank=$NODE_RANK decode_rank=$DECODE_RANK tp=$TP_DECODE kv_role=kv_consumer nixl_host=$NODE_IP:$VLLM_NIXL_SIDE_CHANNEL_PORT master=$DECODE_MASTER_ADDR"

SPEC_CFG=$(printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"attention_backend":"B12X"}' \
    "${PD_DFLASH_MODEL:-/workspace/MiMo-V2.6-Flash-RL-dflash}" \
    "${PD_NUM_SPECULATIVE_TOKENS:-7}")

DECODE_CMD=("${VLLM_SERVE[@]}" "${PD_DECODE_MODEL:-XiaomiMiMo/MiMo-V2.6-Flash-RL}"
    --host 0.0.0.0
    --port "${PD_DECODE_PORT:-8200}"
    --tensor-parallel-size "$TP_DECODE"
    --nnodes "$DECODE_NNODES"
    --node-rank "$DECODE_RANK"
    --master-addr "$DECODE_MASTER_ADDR"
    --master-port "${PD_DECODE_MASTER_PORT:-29502}"
    --trust-remote-code
    --max-model-len "${PD_MAX_MODEL_LEN:-1048576}"
    --attention-backend B12X
    --linear-backend b12x
    --moe-backend b12x
    --load-format b12x
    --block-size "${PD_BLOCK_SIZE:-128}"
    --kv-cache-dtype fp8
    --kv-cache-dtype-skip-layers sliding_window
    --max-num-seqs "${PD_MAX_NUM_SEQS:-32}"
    --max-num-batched-tokens "${PD_MAX_NUM_BATCHED_TOKENS:-16384}"
    --gpu-memory-utilization "${PD_DECODE_GMU:-0.80}"
    --enable-chunked-prefill
    --async-scheduling
    --enable-prefix-caching
    --speculative-config "$SPEC_CFG"
    --generation-config vllm
    --reasoning-parser mimo
    --tool-call-parser mimo
    --enable-auto-tool-choice
    --kv-cache-metrics
    --enable-mfu-metrics
    --kv-transfer-config "$KV_CONSUMER"
)
if [ "$DECODE_RANK" -gt 0 ]; then
    DECODE_CMD+=(--headless)
fi
exec "${DECODE_CMD[@]}"