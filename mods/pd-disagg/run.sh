#!/bin/bash
#
# [pd-disagg] Prefill/Decode disaggregation runtime prerequisite mod.
#
# Runs in-container on the head node AND every worker before the serve
# command (mods convention: dir with run.sh, no args, executed by
# launch-cluster.sh apply_mod_to_container()).
#
# Responsibilities
#   1. Validate (optionally install) the `nixl` python package that backs
#      vLLM's NixlConnector. Fail loud if absent — a silent miss makes PD
#      look like it worked while KV never moves (nixl_bytes_transferred 0).
#   2. Pin the NIXL/UCX transport to the RoCE HCAs. launch-cluster.sh
#      force-sets UCX_NET_DEVICES=$ETH_IF for inter-node gloo/NCCL, which is
#      the WRONG device class for NIXL RDMA (see docs/NETWORKING.md). We
#      override it to the RoCE twin HCAs and fail loud if they are absent.
#   3. Publish the resulting env to a file that dispatch.sh sources, because
#      launch-cluster.sh execs `docker exec bash -c <cmd>` (non-login), so
#      /etc/profile.d is never sourced.
#
# Opt-in: inert unless PD_DISAGG_ENABLED=1 (recipe env sets it).
#
# Env (all optional unless noted):
#   PD_DISAGG_ENABLED=1            REQUIRED to do anything.
#   PD_DISAGG_INSTALL_NIXL=1       pip-install nixl when it is not importable.
#   PD_DISAGG_NIXL_SPEC=nixl       pip spec to install (default "nixl").
#   PD_UCX_NET_DEVICES             RoCE devices, default "rocep1s0f1:1,roceP2p1s0f1:1".
#   PD_UCX_TLS                     default "rc,ud,sm,self,^cuda_ipc".
#   PD_NIXL_PORT                   NIXL side-channel port, default 5600.
#   PD_DISAGG_ALLOW_TCP_FALLBACK=1 downgrade a missing HCA to a warning.
#   PD_DISAGG_ENV_FILE             env file path, default /tmp/pd-disagg.env.
#
# GB10 note: on this UMA platform the UCX half of the NIXL fix ships ONLY
# inside the `nixl-runtime:24.04-cu13.3-sm121` image (tmp/perf-concepts/
# gb10-nixl-prior-art/). A pip wheel alone re-opens NIXL_ERR_BACKEND; baking
# nixl into the image is the durable path and this mod defers to it.

set -euo pipefail

PREFIX="[pd-disagg]"
ENV_FILE="${PD_DISAGG_ENV_FILE:-/tmp/pd-disagg.env}"
fail() { echo "$PREFIX PD_DISAGG_FAIL $*" >&2; return 1; }
info() { echo "$PREFIX $*"; }

# --- 0. opt-in gate ------------------------------------------------------
# Source-safety (B1): dispatch.sh sources this file in-process when the env
# file is missing, so the inert gate must return, not exit (a stray exit would
# kill dispatch mid-gate). Bare invocation keeps exit 0.
if [ "${PD_DISAGG_ENABLED:-0}" != "1" ]; then
    echo "$PREFIX inert (set PD_DISAGG_ENABLED=1 to enable)"
    return 0 2>/dev/null || exit 0
fi

# --- 1. Mooncake store env (replaces NIXL/UCX) ---------------------------
# MooncakeStoreConnector reads its config via MOONCAKE_CONFIG_PATH; the
# sibling mooncake-client container owns the DRAM segment + SSD offload
# tier, so vLLM ranks render a requester-only JSON (global_segment_size=0,
# enable_offload=false). No UCX exports: Mooncake's engine.so links libmlx5
# directly (mooncake-store Dockerfile) — UCX is not on the store data path.
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1b. Host-staging connector patch --------------------------------------
# The deployed image's MooncakeStoreConnector registers the GPU KV buffer
# directly (worker.py `register_buffer(gpu_addr)`), which dies on GB10 with
# ibv_reg_mr EFAULT ("Bad address [14]", num_segments=0). The Fulton vllm
# fork's host-staging branches add _StagingSlotPool (D2H -> pinned host
# slots -> RDMA) keyed off kv_connector_extra_config['host_staging'].
# The image predates it, so stage the fork's mooncake/ connector package at
# the pinned SHA and overlay it in-place (same tree the qwen38 systemd path
# bind-mounts via SERVE_VLLM_PATCH_CONNECTOR_DIR; mods are COPIED into the
# container at apply time, not mounted, so the in-place overlay is the
# mod-pipeline equivalent). Idempotent: skipped when host_staging already
# present.
if ! grep -q 'host_staging' \
    "${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py" 2>/dev/null; then
    info "connector lacks host_staging; staging fork package"
    bash "$MOD_DIR/mooncake-connector-patch.sh" stage \
        "${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}/vllm/distributed/kv_transfer/kv_connector/v1/mooncake"
else
    info "connector host_staging already present"
fi

# The connector imports `mooncake` in-process (worker.py:
# `from mooncake.store import ...`). The vllm-node-b12x image does not ship
# the wheel; install it at boot if absent (mirrors the retired NIXL install
# path). RDMA runtime deps (libmlx5/libibverbs) are already in the image.
# Pin to the same version the sibling mooncake-store image runs.
: "${PD_MOONCAKE_SPEC:=mooncake-transfer-engine-cuda13==0.3.12.post1}"
if ! python3 -c 'import mooncake' >/dev/null 2>&1; then
    if [ "${PD_DISAGG_INSTALL_MOONCAKE:-1}" = "1" ]; then
        info "mooncake module missing; installing '${PD_MOONCAKE_SPEC}'"
        python3 -m pip install --no-cache-dir "${PD_MOONCAKE_SPEC}" >&2 \
            || fail "op=install_mooncake spec='${PD_MOONCAKE_SPEC}' reason=pip_failed"
    fi
    python3 -c 'import mooncake' >/dev/null 2>&1 \
        || fail "op=import_mooncake reason=absent hint='set PD_DISAGG_INSTALL_MOONCAKE=1 or bake ${PD_MOONCAKE_SPEC} into the image'"
    info "mooncake module installed"
else
    info "mooncake module already present"
fi

# cupy-cuda13x: required by the host-staged _StagingSlotPool (pinned slot
# allocs via cupy.cuda); the b12x image does not ship it. Mirrors vLLM's
# kv_connectors.txt pin (<14.1.0 until cupy.testing regression fixed).
if ! python3 -c 'import cupy' >/dev/null 2>&1; then
    if [ "${PD_DISAGG_INSTALL_CUPY:-1}" = "1" ]; then
        info "cupy missing; installing 'cupy-cuda13x<14.1.0'"
        python3 -m pip install --no-cache-dir 'cupy-cuda13x<14.1.0' >&2 \
            || fail "op=install_cupy reason=pip_failed"
    fi
    python3 -c 'import cupy' >/dev/null 2>&1 \
        || fail "op=import_cupy reason=absent hint='set PD_DISAGG_INSTALL_CUPY=1 or bake cupy-cuda13x into the image'"
    info "cupy installed"
else
    info "cupy already present"
fi

source "$MOD_DIR/mooncake-env.sh"

# --- 2. publish env for dispatch.sh -------------------------------------
pd_num_speculative_tokens="${PD_NUM_SPECULATIVE_TOKENS:-7}"
pd_block_size="${PD_BLOCK_SIZE:-128}"
pd_max_model_len="${PD_MAX_MODEL_LEN:-1048576}"
pd_max_num_seqs="${PD_MAX_NUM_SEQS:-32}"
pd_max_num_batched_tokens="${PD_MAX_NUM_BATCHED_TOKENS:-16384}"

# n3 single-source: canonical parity payload lives ONLY here (run.sh).
# dispatch.sh consumes the published PD_PARITY_PAYLOAD (from the env file)
# and never re-types the format string. Both roles hash identical bytes.
PD_PARITY_PAYLOAD="$(printf 'num_speculative_tokens=%s\nblock_size=%s\nkv_cache_dtype=fp8\nkv_cache_dtype_skip_layers=sliding_window\nmax_model_len=%s\nmax_num_seqs=%s\nmax_num_batched_tokens=%s\n' \
    "$pd_num_speculative_tokens" "$pd_block_size" "$pd_max_model_len" "$pd_max_num_seqs" "$pd_max_num_batched_tokens")"
PD_PARITY_SHA256="$(printf '%s\n' "$PD_PARITY_PAYLOAD" \
    | { sha256sum 2>/dev/null || shasum -a 256; } | awk '{print $1}')"
export PD_PARITY_PAYLOAD PD_PARITY_SHA256
{
    echo "# generated by mods/pd-disagg/run.sh — do not edit"
    # Shared PD parity vars: consumed by dispatch.sh on EVERY role; these are
    # the ONLY inputs (plus fixed fp8/sliding_window KV layout) to
    # PD_PARITY_SHA256. Every role must log an identical hash.
    echo "export PD_NUM_SPECULATIVE_TOKENS=\"$pd_num_speculative_tokens\""
    echo "export PD_BLOCK_SIZE=\"$pd_block_size\""
    echo "export PD_MAX_MODEL_LEN=\"$pd_max_model_len\""
    echo "export PD_MAX_NUM_SEQS=\"$pd_max_num_seqs\""
    echo "export PD_MAX_NUM_BATCHED_TOKENS=\"$pd_max_num_batched_tokens\""
    echo "export PD_PARITY_PAYLOAD=\"$PD_PARITY_PAYLOAD\""
    echo "export PD_PARITY_SHA256=\"$PD_PARITY_SHA256\""
    echo "export PD_DISAGG_ENV_READY=1"
} > "$ENV_FILE.tmp.$$"
mv -f "$ENV_FILE.tmp.$$" "$ENV_FILE"
chmod 0644 "$ENV_FILE"
{
    echo "export MOONCAKE_CONFIG_PATH=\"$MOONCAKE_CONFIG_PATH\""
    echo "export MOONCAKE_PREFERRED_SEGMENT=\"$MOONCAKE_PREFERRED_SEGMENT\""
    echo "export MOONCAKE_REQUESTER_LOCAL_HOSTNAME=\"$MOONCAKE_REQUESTER_LOCAL_HOSTNAME\""
    echo "export PYTHONHASHSEED=\"$PYTHONHASHSEED\""
    echo "export MOONCAKE_MASTER_SERVER_ADDRESS=\"$MOONCAKE_MASTER_SERVER_ADDRESS\""
    echo "export MOONCAKE_MASTER_PORT=\"$MOONCAKE_MASTER_PORT\""
    echo "export MOONCAKE_CLIENT_PORT=\"$MOONCAKE_CLIENT_PORT\""
} >> "$ENV_FILE"

info "mooncake env published to $ENV_FILE:"
info "  MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH"
info "  MOONCAKE_PREFERRED_SEGMENT=$MOONCAKE_PREFERRED_SEGMENT"
info "PD_PARITY_SHA256=$PD_PARITY_SHA256 payload={num_speculative_tokens=$pd_num_speculative_tokens,block_size=$pd_block_size,max_model_len=$pd_max_model_len,max_num_seqs=$pd_max_num_seqs,max_num_batched_tokens=$pd_max_num_batched_tokens,kv=fp8/sliding_window}"
info "PD_DISAGG_OK op=prepare env_file=$ENV_FILE"
