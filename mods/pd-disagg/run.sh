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

# Backend gate: the mooncake/cupy boot installs below are only needed by the
# MooncakeStoreConnector path (vLLM imports `mooncake` in-process and the
# host-staged _StagingSlotPool needs cupy). PD_KV_BACKEND=lmcache attaches to
# the MP server instead and needs neither.
if [ "${PD_KV_BACKEND:-mooncake}" = "mooncake" ]; then

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

fi  # PD_KV_BACKEND=mooncake

# LMCache backend: the vLLM rank imports the LMCacheMPConnector shim, which
# does `from lmcache.integration.vllm.utils import mla_enabled` — so the
# `lmcache` package (NOT just the MP server image) must be importable in the
# engine image. The vllm-node-b12x image does not ship it; install the
# GB10/sm_121 wheel at boot (same dgx-spark-wheels release the lmcache-server
# image uses). --no-deps: torch and the C++ extensions' runtime libs are
# already in the image.
if [ "${PD_KV_BACKEND:-mooncake}" = "lmcache" ]; then
    : "${PD_LMCACHE_WHEEL_URL:=https://github.com/Fulton-Engineering-Services/dgx-spark-wheels/releases/download/lmcache-mooncake-v0.5.5rc1-cu13.3/lmcache-0.5.5rc1+cu13.3torch2.13.glibc239-cp312-cp312-linux_aarch64.whl}"
    if ! python3 -c 'from lmcache.integration.vllm.utils import mla_enabled' >/dev/null 2>&1; then
        if [ "${PD_DISAGG_INSTALL_LMCACHE:-1}" = "1" ]; then
            info "lmcache module missing; installing from ${PD_LMCACHE_WHEEL_URL##*/}"
            python3 -m pip install --no-cache-dir --no-deps "${PD_LMCACHE_WHEEL_URL}" >&2 \
                || fail "op=install_lmcache url='${PD_LMCACHE_WHEEL_URL}' reason=pip_failed"
            # Runtime deps from the canonical requirements list (bundled beside
            # this script; same set the lmcache-server image installs).
            # Excludes: torch (image has the from-source cu13.3 build),
            # cufile-python (GDS unsupported on GB10), setuptools* (build deps),
            # pytest (test dep), numpy (image pins its own), nixl (already in
            # the image as nixl_cu13). Installed WITH resolution.
            req="${MOD_DIR}/lmcache-requirements-common.txt"
            if [ -f "$req" ]; then
                info "installing lmcache runtime deps from ${req##*/}"
                grep -v '^\s*#' "$req" | grep -v '^\s*$' \
                    | grep -vE '^(torch|numpy|cufile-python|pytest|setuptools|setuptools_scm|nixl)([<>=!].*)?$' \
                    > /tmp/lmcache-reqs-filtered.txt
                python3 -m pip install --no-cache-dir -r /tmp/lmcache-reqs-filtered.txt >&2 \
                    || fail "op=install_lmcache_deps reason=pip_failed hint='see ${req##*/}'"
                rm -f /tmp/lmcache-reqs-filtered.txt
            fi
        fi
        python3 -c 'from lmcache.integration.vllm.utils import mla_enabled' >/dev/null 2>&1 \
            || fail "op=import_lmcache reason=absent hint='set PD_DISAGG_INSTALL_LMCACHE=1 or bake the lmcache wheel into the engine image'"
        info "lmcache module installed"
    else
        info "lmcache module already present"
    fi
fi  # PD_KV_BACKEND=lmcache

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
    echo "export PD_KV_BACKEND=\"$PD_KV_BACKEND\""
    echo "export PYTHONHASHSEED=\"$PYTHONHASHSEED\""
    if [ "$PD_KV_BACKEND" = "lmcache" ]; then
        # LMCache MP mode: the MP server owns the store client; the vLLM rank
        # only needs the MP endpoint. globals are published for logging parity.
        echo "export LMCACHE_MP_HOST=\"$LMCACHE_MP_HOST\""
        echo "export LMCACHE_MP_PORT=\"$LMCACHE_MP_PORT\""
        echo "export MOONCAKE_MASTER_SERVER_ADDRESS=\"$MOONCAKE_MASTER_SERVER_ADDRESS\""
        echo "export MOONCAKE_MASTER_PORT=\"$MOONCAKE_MASTER_PORT\""
    else
        echo "export MOONCAKE_CONFIG_PATH=\"$MOONCAKE_CONFIG_PATH\""
        echo "export MOONCAKE_PREFERRED_SEGMENT=\"$MOONCAKE_PREFERRED_SEGMENT\""
        echo "export MOONCAKE_REQUESTER_LOCAL_HOSTNAME=\"$MOONCAKE_REQUESTER_LOCAL_HOSTNAME\""
        echo "export MOONCAKE_MASTER_SERVER_ADDRESS=\"$MOONCAKE_MASTER_SERVER_ADDRESS\""
        echo "export MOONCAKE_MASTER_PORT=\"$MOONCAKE_MASTER_PORT\""
        echo "export MOONCAKE_CLIENT_PORT=\"$MOONCAKE_CLIENT_PORT\""
    fi
} >> "$ENV_FILE"

info "mooncake env published to $ENV_FILE:"
if [ "$PD_KV_BACKEND" = "lmcache" ]; then
    info "  PD_KV_BACKEND=lmcache LMCACHE_MP_HOST=$LMCACHE_MP_HOST LMCACHE_MP_PORT=$LMCACHE_MP_PORT"
else
    info "  MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH"
    info "  MOONCAKE_PREFERRED_SEGMENT=$MOONCAKE_PREFERRED_SEGMENT"
fi
info "PD_PARITY_SHA256=$PD_PARITY_SHA256 payload={num_speculative_tokens=$pd_num_speculative_tokens,block_size=$pd_block_size,max_model_len=$pd_max_model_len,max_num_seqs=$pd_max_num_seqs,max_num_batched_tokens=$pd_max_num_batched_tokens,kv=fp8/sliding_window}"
info "PD_DISAGG_OK op=prepare env_file=$ENV_FILE"
