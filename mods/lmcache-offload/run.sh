#!/bin/bash
#
# [lmcache-offload] In-process LMCacheConnectorV1 KV offload + durability mod.
#
# Runs in-container on head node AND every worker before the serve command
# (mods convention: dir run.sh, no args, executed by launch-cluster.sh's
# apply_mod_to_container()). NO env exports survive to the serve exec (separate
# process), so the LMCACHE_CONFIG_FILE path here MUST match the recipe env
# (recipes/*-lmcache.yaml). Installs the lmcache wheel the vllm-node-b12x image
# lacks and renders a requester-only LMCache YAML config: L1 = host DRAM
# (local_cpu), L2 = fs_native on the shared NFS (/cluster-shared/lmcache-l2,
# per-node subdir) + Valkey on Dell. No PD, no MP server, no RDMA transfer.
#
# Opt-in: inert unless LMCACHE_OFFLOAD_ENABLED=1 (recipe env sets it).
#
# Env (all optional unless noted):
#   LMCACHE_OFFLOAD_ENABLED=1   REQUIRED to do anything.
#   LMCACHE_WHEEL_URL           lmcache wheel (default: dgx-spark-wheels
#                               lmcache-mooncake-v0.5.5rc1-cu13.3 aarch64).
#   LMCACHE_LOCAL_CPU_GIB       L1 host DRAM cap in GiB (default 8).
#   LMCACHE_LOCAL_DISK_GIB      L2 fs_native cap in GiB (default 50).
#   LMCACHE_CONFIG_FILE         config file path (default /tmp/lmcache-config.yaml;
#                               MUST equal the recipe env value).

set -euo pipefail

PREFIX="[lmcache-offload]"
fail() { echo "$PREFIX LMCACHE_OFFLOAD_FAIL $*" >&2; return 1; }
info() { echo "$PREFIX $*"; }

# Opt-in: inert unless the recipe sets LMCACHE_OFFLOAD_ENABLED=1.
if [ "${LMCACHE_OFFLOAD_ENABLED:-0}" != "1" ]; then
    echo "$PREFIX inert (set LMCACHE_OFFLOAD_ENABLED=1 to enable)"
    return 0 2>/dev/null || exit 0
fi

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. Install the lmcache wheel (mirrors mods/pd-disagg/run.sh:112-142) ----
# vllm-node-b12x image does not ship it; install the GB10/sm_121 wheel at boot
# (same dgx-spark-wheels release the lmcache-server image uses). --no-deps:
# torch and the C++ extensions' runtime libs are already in the image.
: "${LMCACHE_WHEEL_URL:=https://github.com/Fulton-Engineering-Services/dgx-spark-wheels/releases/download/lmcache-mooncake-v0.5.5rc1-cu13.3/lmcache-0.5.5rc1+cu13.3torch2.13.glibc239-cp312-cp312-linux_aarch64.whl}"
if ! python3 -c 'from lmcache.integration.vllm.utils import mla_enabled' >/dev/null 2>&1; then
    info "lmcache module missing; installing from ${LMCACHE_WHEEL_URL##*/}"
    python3 -m pip install --no-cache-dir --no-deps "${LMCACHE_WHEEL_URL}" >&2 \
        || fail "op=install_lmcache url='${LMCACHE_WHEEL_URL}' reason=pip_failed"
    # Runtime deps from the canonical requirements list (bundled beside this
    # script; same set the lmcache-server image installs). Excludes: torch
    # (image has the from-source cu13.3 build), cufile-python (GDS unsupported
    # on GB10), setuptools* (build deps), pytest (test dep), numpy (image pins
    # its own), nixl (already in the image as nixl_cu13). Installed WITH resolution.
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
    python3 -c 'from lmcache.integration.vllm.utils import mla_enabled' >/dev/null 2>&1 \
        || fail "op=import_lmcache reason=absent hint='bake the lmcache wheel into the engine image'"
    info "lmcache module installed"
else
    info "lmcache module already present"
fi

# The valkey:// L2 connector needs the synchronous GLIDE client (the async
# valkey-glide package is a different module namespace).
if ! python3 -c 'import glide_sync' >/dev/null 2>&1; then
    info "installing valkey-glide-sync for the valkey:// L2 connector"
    python3 -m pip install --no-cache-dir 'valkey-glide-sync>=2.3.0' >&2 \
        || fail "op=install_valkey_glide reason=pip_failed"
fi

# --- 2. Render the LMCache config file ----------------------------------------
# Requester-only, in-process. L1 = host DRAM (local_cpu). L2 = fs_native on the
# shared NFS under a PER-NODE subdir (two ranks never share a shard dir; the
# bare /cluster-shared/lmcache-l2 is deliberately NOT used -- the PD MP-server
# fleet writes there). L2 remote = Valkey on Dell (no PD, no MP server, no RDMA).
L1_GIB="${LMCACHE_LOCAL_CPU_GIB:-8}"
DISK_GIB="${LMCACHE_LOCAL_DISK_GIB:-50}"
CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/tmp/lmcache-config.yaml}"
L2_DIR="/cluster-shared/lmcache-l2/${HOSTNAME}"
mkdir -p "$L2_DIR"

cat > "$CONFIG_FILE" <<EOF
# Rendered by mods/lmcache-offload/run.sh -- LMCacheConnectorV1 offload+durability.
chunk_size: 256
local_cpu: true
max_local_cpu_size: ${L1_GIB}
local_disk: ${L2_DIR}
max_local_disk_size: ${DISK_GIB}
save_unfull_chunk: false
save_decode_cache: true
remote_storage_plugins:
  - "valkey://10.10.10.53:6379"
EOF

echo "$PREFIX ok config=${CONFIG_FILE} l1=${L1_GIB}G l2_disk=${DISK_GIB}G dir=${L2_DIR} l2_valkey=10.10.10.53:6379"
