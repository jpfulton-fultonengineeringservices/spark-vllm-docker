#!/bin/bash
# [pd-disagg] Mooncake connector package staging (host-staging fork subtree).
#
# The deployed vllm-node-b12x image (vLLM dev21546, baked 2026-10-03) ships a
# MooncakeStoreConnector that registers the GPU KV buffer directly
# (`self.store.register_buffer(gpu_addr)`); on GB10 that is the documented
# dead end — ibv_reg_mr returns EFAULT, worker.py logs
# "register_buffer failed ... Bad address [14]" and registers num_segments=0,
# so no KV ever reaches the store.
#
# The Fulton vllm fork's host_staging branches fix exactly this: the vLLM
# ranks D2H-copy GPU KV into pinned host slots and RDMA-write from there
# (_StagingSlotPool, worker.py:280; `host_staging` / `staging_buffer_size_mb`
# / `staging_num_slots` read from kv_connector_extra_config, worker.py:1780+).
#
# Rebuilding the image is hours of work; the proven alternative (mirrors
# cluster-config/scripts/setup-mooncake-pd.sh:333-355) is to fetch the fork's
# mooncake/ connector PACKAGE at the host-staging SHA and bind-mount it over
# the image's stale copy at container start. This script stages that tree on
# the shared FS so a mod apply can --bind-mount it.
#
# Sourced by mods/pd-disagg/run.sh (variant=stage) and consumed by dispatch.sh
# (variant=apply, which prints the -v mount pair). Both run INSIDE the
# container, so "stage" writes to a bind-mounted host dir and "apply" just
# reports the mount source the caller already mounted.

set -euo pipefail

PREFIX="[pd-disagg-mooncake-patch]"

# Pinned to the fork's host-staging head. Bump when the fork branch moves.
# e93769fd9b = "fix(store): add claim() timeout, handle queue.Empty in put/get
# chunk" on cuda13.3-aarch64-gb10-glm53-flash (same tree the qwen38 systemd
# path pins in setup-mooncake-pd.sh).
: "${PD_MOONCAKE_PATCH_SHA:=e93769fd9be0a0bcd1065eea20181e232ebedcf3}"
: "${PD_MOONCAKE_PATCH_REPO:=Fulton-Engineering-Services/vllm}"
# Container path the connector package is bind-mounted to (over the image's
# own vllm/distributed/.../mooncake dir). Keep in lockstep with the mount.
: "${PD_MOONCAKE_PATCH_MOUNT:=/opt/mooncake-connector-patch/mooncake}"

# The in-container package dir the mount shadows (must match the image's vLLM).
VLLM_SITE="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
CONNECTOR_DIR="$VLLM_SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake"

stage_connector_patch() {
    local dst="$1"
    if [ -f "$dst/worker.py" ] && grep -q 'host_staging' "$dst/worker.py"; then
        echo "$PREFIX already staged (host_staging present): $dst"
        return 0
    fi
    local tmp tarball sub
    tmp="$(mktemp -d)"
    tarball="$tmp/fork.tar.gz"
    echo "$PREFIX fetching ${PD_MOONCAKE_PATCH_REPO}@${PD_MOONCAKE_PATCH_SHA}"
    curl -fsSL "https://codeload.github.com/${PD_MOONCAKE_PATCH_REPO}/tar.gz/${PD_MOONCAKE_PATCH_SHA}" -o "$tarball" \
        || { echo "$PREFIX PD_DISAGG_FAIL: fetch failed" >&2; rm -rf "$tmp"; return 1; }
    sub="vllm-${PD_MOONCAKE_PATCH_SHA}/vllm/distributed/kv_transfer/kv_connector/v1/mooncake"
    tar -xzf "$tarball" -C "$tmp" "$sub" || { echo "$PREFIX PD_DISAGG_FAIL: extract failed ($sub)" >&2; rm -rf "$tmp"; return 1; }
    [ -f "$tmp/$sub/worker.py" ] || { echo "$PREFIX PD_DISAGG_FAIL: no worker.py at $sub" >&2; rm -rf "$tmp"; return 1; }
    if ! grep -q 'host_staging' "$tmp/$sub/worker.py"; then
        echo "$PREFIX PD_DISAGG_FAIL: fetched tree lacks host_staging (wrong SHA?)" >&2
        rm -rf "$tmp"; return 1
    fi
    for f in "$tmp/$sub"/*.py; do
        python3 -c "compile(open('$f').read(), '$f', 'exec')" || {
            echo "$PREFIX PD_DISAGG_FAIL: $f does not compile" >&2; rm -rf "$tmp"; return 1; }
    done
    mkdir -p "$(dirname "$dst")"
    rm -rf "$dst"
    mv "$tmp/$sub" "$dst"
    rm -rf "$tmp"
    echo "$PREFIX PD_DISAGG_OK variant=stage dst=$dst sha=$PD_MOONCAKE_PATCH_SHA"
}

# When bind-mounted, $CONNECTOR_DIR already IS the patched tree — detect and
# report readiness (this is the apply/verify variant run by dispatch.sh).
verify_connector_patch() {
    if grep -q 'host_staging' "$CONNECTOR_DIR/worker.py" 2>/dev/null; then
        echo "$PREFIX PD_DISAGG_OK variant=verify host_staging=present dir=$CONNECTOR_DIR"
        return 0
    fi
    echo "$PREFIX PD_DISAGG_FAIL: connector lacks host_staging at $CONNECTOR_DIR (mount ${PD_MOONCAKE_PATCH_MOUNT} not applied?)" >&2
    return 1
}

case "${1:-stage}" in
    stage)  stage_connector_patch "${2:-$CONNECTOR_DIR}" ;;
    verify) verify_connector_patch ;;
    mount-src) printf '%s' "$PD_MOONCAKE_PATCH_MOUNT" ;;
    *) echo "usage: $0 {stage [dst]|verify|mount-src}" >&2; exit 2 ;;
esac
