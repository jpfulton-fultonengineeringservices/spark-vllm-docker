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
DEFAULT_UCX_DEVS="rocep1s0f1:1,roceP2p1s0f1:1"

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

# --- 1. nixl python package ---------------------------------------------
if python3 -c 'import nixl' >/dev/null 2>&1; then
    nixl_ver="$(python3 -c 'import importlib.metadata as m; print(m.version("nixl"))' 2>/dev/null || echo unknown)"
    info "nixl present (version=$nixl_ver)"
else
    if [ "${PD_DISAGG_INSTALL_NIXL:-0}" = "1" ]; then
        info "nixl missing; installing '${PD_DISAGG_NIXL_SPEC:-nixl}'"
        python3 -m pip install --no-cache-dir ${PD_DISAGG_NIXL_SPEC:-nixl} >&2 \
            || fail "op=install_nixl spec='${PD_DISAGG_NIXL_SPEC:-nixl}' reason=pip_failed"
    fi
    python3 -c 'import nixl' >/dev/null 2>&1 \
        || fail "op=import_nixl reason=absent hint='bake nixl+nixl-runtime:24.04-cu13.3-sm121 into the image, or set PD_DISAGG_INSTALL_NIXL=1'"
    info "nixl installed"
fi

# --- 2. RoCE HCA pinning -------------------------------------------------
ucx_devs="${PD_UCX_NET_DEVICES:-$DEFAULT_UCX_DEVS}"
# UCX_TLS must keep a CUDA memory transport: NixlConnector registers the KV
# buffer on the GPU (kv_buffer_device=cuda), and GB10 has no GDR -- cuda_copy
# is the transport that maps CUDA memory. Without it UCX reports "CUDA support
# was not found", host-registers the GPU VA, and register_memory dies with
# NIXL_ERR_BACKEND. The image default is cuda_copy,rc,sm,self,tcp (Dockerfile.nixl);
# this default keeps cuda_copy and only excludes cuda_ipc (incompatible with
# the RoCE two-HCA layout).
ucx_tls="${PD_UCX_TLS:-cuda_copy,rc,ud,sm,self,^cuda_ipc}"

missing=""
old_ifs="$IFS"
IFS=','
for dev in $ucx_devs; do
    hca="${dev%%:*}"
    if [ ! -d "/sys/class/infiniband/$hca" ]; then
        missing="$missing $hca"
    fi
done
IFS="$old_ifs"

if [ -n "$missing" ]; then
    if [ "${PD_DISAGG_ALLOW_TCP_FALLBACK:-0}" = "1" ]; then
        info "WARN op=roce_hca reason=absent devices='$missing' (PD_DISAGG_ALLOW_TCP_FALLBACK=1; enabling tcp TLS)"
        case ",$ucx_tls," in
            *,tcp,*) : ;;
            *) ucx_tls="$ucx_tls,tcp" ;;
        esac
    else
        fail "op=roce_hca reason=absent devices='$missing' hint='check ibdev2netdev; set PD_UCX_NET_DEVICES, or PD_DISAGG_ALLOW_TCP_FALLBACK=1 to accept TCP'"
    fi
else
    info "RoCE HCAs present: $ucx_devs"
fi

# --- 3. publish env for dispatch.sh -------------------------------------
nixl_port="${PD_NIXL_PORT:-5600}"
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
    echo "export UCX_NET_DEVICES=\"$ucx_devs\""
    echo "export UCX_TLS=\"$ucx_tls\""
    echo "export VLLM_NIXL_SIDE_CHANNEL_PORT=\"$nixl_port\""
    echo "export NIXL_LOG_LEVEL=\"${NIXL_LOG_LEVEL:-WARN}\""
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

export UCX_NET_DEVICES="$ucx_devs"
export UCX_TLS="$ucx_tls"
export VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl_port"

info "nixl/RoCE env published to $ENV_FILE:"
info "  UCX_NET_DEVICES=$UCX_NET_DEVICES"
info "  UCX_TLS=$UCX_TLS"
info "  VLLM_NIXL_SIDE_CHANNEL_PORT=$VLLM_NIXL_SIDE_CHANNEL_PORT"
info "PD_PARITY_SHA256=$PD_PARITY_SHA256 payload={num_speculative_tokens=$pd_num_speculative_tokens,block_size=$pd_block_size,max_model_len=$pd_max_model_len,max_num_seqs=$pd_max_num_seqs,max_num_batched_tokens=$pd_max_num_batched_tokens,kv=fp8/sliding_window}"
info "PD_DISAGG_OK op=prepare env_file=$ENV_FILE"
