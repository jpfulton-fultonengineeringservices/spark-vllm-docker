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

# Opt-in. The recipe env (LMCACHE_OFFLOAD_ENABLED=1) is NOT visible here: mods
# run BEFORE the exec-script that exports recipe env, in a separate process.
# So the mod self-activates from the exec-script it can read in-container
# (launch-cluster.sh copies /workspace/exec-script.sh before applying mods):
# grep for the config-file pointer the recipe pins. LMCACHE_OFFLOAD_ENABLED=1
# in the calling env still forces activation (debug / bare-mod testing).
EXEC_SCRIPT="${WORKSPACE_DIR:-/workspace}/exec-script.sh"
if [ "${LMCACHE_OFFLOAD_ENABLED:-0}" = "1" ] \
   || { [ -f "$EXEC_SCRIPT" ] && grep -q 'LMCACHE_CONFIG_FILE' "$EXEC_SCRIPT" 2>/dev/null; }; then
    :
else
    echo "$PREFIX inert (recipe exec-script does not reference LMCACHE_CONFIG_FILE; export LMCACHE_OFFLOAD_ENABLED=1 to force)"
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
DISK_GIB="${LMCACHE_LOCAL_DISK_GIB:-1024}"
CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/tmp/lmcache-config.yaml}"
L2_DIR="/cluster-shared/lmcache-l2/${HOSTNAME}"
L2_DIR_2="${LMCACHE_L2_PATH_2:-/nas-1/lmcache-l2/${HOSTNAME}}"
mkdir -p "$L2_DIR"
mkdir -p "$L2_DIR_2"

cat > "$CONFIG_FILE" <<EOF
# Rendered by mods/lmcache-offload/run.sh -- LMCacheConnectorV1 offload+durability.
chunk_size: 256
local_cpu: true
max_local_cpu_size: ${L1_GIB}
local_disk:
  - ${L2_DIR_2}
  - ${L2_DIR}
max_local_disk_size: ${DISK_GIB}
save_unfull_chunk: false
save_decode_cache: true
remote_url: "valkey://10.10.10.53:6379"
EOF

# --- 3. Fork patch: uniformity probe must not raise on mixed windows ----------
# With --disable-hybrid-kv-cache-manager, get_kv_cache_groups() first runs
# unify_hybrid_kv_cache_specs(), which probes uniformity via
# is_kv_cache_spec_uniform(). That probe calls specs[0].merge(all) and catches
# ONLY AssertionError -- but FullAttentionSpec.merge's merge_window_sizes()
# raises ValueError on a mixed window set (MiMo: SWA layers window=128 vs
# full-attention layers window=None). Observed live 2026-10-11:
#   ValueError: All attention layers in the same KV cache group must have the
#   same window size. (via _promote_local_kv_cache_specs -> probe)
# Catching ValueError lets the probe correctly report "not uniform"; the
# per-layer promotion then yields FullAttentionSpec(128)/FullAttentionSpec(None),
# which FullAttentionSpec.merge unifies by filtering None windows -> ONE group,
# the single-group invariant the LMCache v1 adapter requires. Idempotent.
python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"
with open(path) as f:
    src = f.read()

# 3a. is_kv_cache_spec_uniform: catch ValueError from merge_window_sizes
old = """    except AssertionError:
        return False
    return True"""
new = """    except (AssertionError, ValueError):
        return False
    return True"""
if new in src:
    print("[lmcache-offload] kv_cache_utils uniformity-probe patch already applied")
elif old in src:
    with open(path, "w") as f:
        f.write(src.replace(old, new, 1))
    print("[lmcache-offload] patched is_kv_cache_spec_uniform to catch ValueError (mixed-window probe)")
else:
    print("[lmcache-offload] WARN: uniformity-probe patch target not found (fork drift?); continuing", file=sys.stderr)

# 3b. Normalize mixed sliding_window values after promotion to None.
# The normalization catches layers (e.g. DFlash drafter) that are already
# FullAttentionSpec with a non-None window, which the drop in 3a doesn't
# cover.  Code identical in structure to the vendored-source fix in
# gpu-cluster-forks/vllm/vllm/v1/core/kv_cache_utils.py _promote_local_kv_cache_specs.
fallback_promote = '''    if not (
        is_kv_cache_spec_uniform(promoted_specs)
        or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs)
    ):
        raise ValueError("Failed to promote local KV cache specs to one unified type.")'''
fallback_after = '''    if not (
        is_kv_cache_spec_uniform(promoted_specs)
        or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs)
    ):
        for name, spec in promoted_specs.items():
            if isinstance(spec, FullAttentionSpec) and spec.sliding_window is not None:
                promoted_specs[name] = dataclasses.replace(spec, sliding_window=None)
    if not (
        is_kv_cache_spec_uniform(promoted_specs)
        or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs)
    ):
        raise ValueError("Failed to promote local KV cache specs to one unified type.")'''
if fallback_after in src:
    print("[lmcache-offload] promote-local normalise-fallback patch already applied")
elif fallback_promote in src:
    with open(path, "w") as f:
        f.write(src.replace(fallback_promote, fallback_after, 1))
    print("[lmcache-offload] patched _promote_local_kv_cache_specs to normalise mixed windows")
else:
    print("[lmcache-offload] WARN: promote-local fallback patch target not found (fork drift?); continuing", file=sys.stderr)

PYEOF

# --- 4. Fork patch: GPU connector device ordinal must clamp to device_count() ---
# LMCacheConnectorV1 reads metadata.local_worker_id and passes it directly to
# torch_dev.set_device(local_worker_id). In cross-host TP=2, rank1 has
# local_worker_id=1 (correct per-host TP index) but node carries one GPU
# (device_count=1, so only device 0 exists). The set_device(1) raises
# CUDA error: invalid device ordinal, putting that rank's LMCache into
# degraded recompute mode. Clamp: local_worker_id % device_count.
python3 - <<'PYEOF'
import sys
path = "/usr/local/lib/python3.12/dist-packages/lmcache/v1/gpu_connector/__init__.py"
with open(path) as f:
    src = f.read()
old = "local_worker_id = metadata.local_worker_id\n        torch_dev.set_device(local_worker_id)"
new = "local_worker_id = metadata.local_worker_id % max(1, torch_dev.device_count())\n        torch_dev.set_device(local_worker_id)"
if new in src:
    print("[lmcache-offload] gpu_connector device-ordinal patch already applied")
elif old in src:
    src = src.replace(old, new, 2)  # two occurrences: SGLang + vLLM paths
    with open(path, "w") as f:
        f.write(src)
    print("[lmcache-offload] patched gpu_connector to clamp device ordinal to device_count() (cross-host TP)")
else:
    print("[lmcache-offload] WARN: gpu_connector device-ordinal target not found (wheel drift?); continuing", file=sys.stderr)
# Also fix the device string following (same pattern, second occurrence)
old2 = 'torch_dev.set_device(local_worker_id)\n        device = torch.device(f"{torch_device_type}:{local_worker_id}")'
new2 = 'torch_dev.set_device(local_worker_id)\n        device = torch.device(f"{torch_device_type}:{safe_id}")'
# Check if already fixed
if 'safe_id' in src:
    print("[lmcache-offload] gpu_connector device-string patch already applied")
else:
    # Just fix set_device line, device line uses local_worker_id directly
    # Wrap both with safe_id
    import re
    # The second occurrence (line 169-171 in source)
    count = 0
    def fix_second(m):
        global count
        count += 1
        if count == 2:
            return 'safe_id = metadata.local_worker_id % max(1, torch_dev.device_count())\n        torch_dev.set_device(safe_id)\n        device = torch.device(f"{torch_device_type}:{safe_id}")'
        return m.group(0)
    new_src = re.sub(
        r'local_worker_id = metadata\.local_worker_id\n\s+torch_dev\.set_device\(local_worker_id\)\n\s+device = torch\.device\(f"\{torch_device_type\}:\{local_worker_id\}"\)',
        fix_second, src
    )
    if new_src != src:
        with open(path, "w") as f:
            f.write(new_src)
        print("[lmcache-offload] patched gpu_connector device string (second occurrence) for safe_id")
    else:
        print("[lmcache-offload] WARN: gpu_connector device-string regex target not found; continuing", file=sys.stderr)
PYEOF

echo "$PREFIX ok config=${CONFIG_FILE} l1=${L1_GIB}G l2_disk=${DISK_GIB}G dir=${L2_DIR} l2_remote=valkey://10.10.10.53:6379"
