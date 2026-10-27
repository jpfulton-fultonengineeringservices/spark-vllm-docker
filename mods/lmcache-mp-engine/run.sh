#!/bin/bash
#
# [lmcache-mp-engine] Engine-side LMCache MP connector runtime mod.
#
# Runs in-container on head node AND every worker before the serve command
# (mods convention: dir run.sh, no args, executed by launch-cluster.sh's
# apply_mod_to_container()). Installs the lmcache wheel the vllm-node-b12x
# image lacks -- the engine side of the MP corridor (LMCacheMPConnector).
# The MP server side runs from the cluster-config lmcache-server image
# (units installed by cluster-config's launch_lmcache_mp_servers_external).
#
# This mod does NOT render a config file and does NOT install valkey-glide:
# the MP connector takes lmcache.mp.* via --kv-transfer-config extra_config
# (pinned in the recipe), and L2 adapters (fs_native/valkey) live entirely
# server-side in the lmcache-server image.
#
# Opt-in: inert unless LMCACHE_MP_ENABLED=1 (cluster-config's lmcache-ext
# launcher arm injects it) OR the exec-script already references
# LMCacheMPConnector (re-apply on an existing container).
#
# Env (all optional unless noted):
#   LMCACHE_MP_ENABLED=1   activate (launcher -e; see cluster-config
#                          spark-serve.sh's lmcache-ext arm).
#   LMCACHE_MP_WHEEL_URL   lmcache wheel (default: dgx-spark-wheels
#                          lmcache-mooncake-v0.5.5rc1-cu13.3 aarch64).
#
# Fork patch (device-ordinal clamp): cross-host TP=2 gives rank1
# local_worker_id=1 while the node carries ONE GPU (device_count=1); a
# direct set_device(1) raises "invalid device ordinal". The in-process
# lmcache-offload mod patches lmcache/v1/gpu_connector/__init__.py for
# this. The MP connector derives its device via the same metadata, so the
# same clamp is applied to the MP connector file IF the vulnerable pattern
# is present in the installed wheel; if the pattern is absent (wheel
# drift), the mod logs and continues rather than shipping a no-op edit.

set -euo pipefail

PREFIX="[lmcache-mp-engine]"
fail() { echo "$PREFIX LMCACHE_MP_FAIL $*" >&2; return 1; }
info() { echo "$PREFIX $*"; }

# Opt-in. The recipe env is NOT visible here: mods run BEFORE the
# exec-script that exports recipe env, in a separate process. Self-activate
# from the exec-script (launch-cluster.sh copies /workspace/exec-script.sh
# before applying mods) by grepping for the MP connector the recipe pins.
EXEC_SCRIPT="${WORKSPACE_DIR:-/workspace}/exec-script.sh"
if [ "${LMCACHE_MP_ENABLED:-0}" = "1" ] \
   || { [ -f "$EXEC_SCRIPT" ] && grep -q 'LMCacheMPConnector' "$EXEC_SCRIPT" 2>/dev/null; }; then
    :
else
    echo "$PREFIX inert (exec-script does not reference LMCacheMPConnector; export LMCACHE_MP_ENABLED=1 to force)"
    return 0 2>/dev/null || exit 0
fi

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. Install the lmcache wheel (mirrors mods/pd-disagg/run.sh:146-172) ---
# --no-deps: torch and the C++ extensions' runtime libs are already in the
# image (same from-source cu13.3 build the wheel was compiled against).
: "${LMCACHE_MP_WHEEL_URL:=https://github.com/Fulton-Engineering-Services/dgx-spark-wheels/releases/download/lmcache-mooncake-v0.5.5rc1-cu13.3/lmcache-0.5.5rc1+cu13.3torch2.13.glibc239-cp312-cp312-linux_aarch64.whl}"
if ! python3 -c 'from lmcache.integration.vllm.utils import mla_enabled' >/dev/null 2>&1; then
    info "lmcache module missing; installing ${LMCACHE_MP_WHEEL_URL##*/}"
    python3 -m pip install --no-cache-dir --no-deps "${LMCACHE_MP_WHEEL_URL}" >&2 \
        || fail "op=install_lmcache url='${LMCACHE_MP_WHEEL_URL}' reason=pip_failed"
    # Deps from the canonical requirements list (bundled beside this script;
    # same set the lmcache-server image installs). Excludes: torch (image
    # ships the from-source cu13.3 build), numpy/cufile-python/pytest/
    # setuptools/nixl per the pd-disagg filter.
    req="${MOD_DIR}/lmcache-requirements-common.txt"
    grep -vE '^(torch|numpy|cufile-python|pytest|setuptools|setuptools_scm|nixl)([<>=!].*)?$' \
        "$req" > /tmp/lmcache-reqs-filtered.txt
    python3 -m pip install --no-cache-dir -r /tmp/lmcache-reqs-filtered.txt >&2 \
        || fail "op=install_lmcache_deps reason=pip_failed hint='see ${req##*/}'"
    rm -f /tmp/lmcache-reqs-filtered.txt
fi
python3 -c 'from lmcache.integration.vllm.utils import mla_enabled' >/dev/null 2>&1 \
    || fail "op=import_lmcache reason=absent"
info "lmcache importable (MP connector engine side ready)"

# --- 2. Fork patch: MP connector device-ordinal clamp ------------------------
# Cross-host TP=2: rank1 metadata.local_worker_id=1 vs device_count=1.
# Patch ONLY when the vulnerable direct set_device(local_worker_id) pattern
# exists in the installed wheel's connector surface; log-and-continue
# otherwise (never a silent no-op edit).
python3 - <<'PYEOF'
import glob
import re
import sys

targets = [
    "/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/lmcache_mp_connector.py",
]
# The in-process gpu_connector factory is patched by the lmcache-offload mod
# when that corridor runs; here we only consider the MP surface, but include
# the factory file IF the MP connector imports device selection from it.
import importlib.util
spec = importlib.util.find_spec("lmcache")
if spec is None:
    print("[lmcache-mp-engine] WARN: lmcache not importable at patch time", file=sys.stderr)
    sys.exit(0)

patched_any = False
for path in targets:
    if not glob.glob(path):
        print(f"[lmcache-mp-engine] WARN: target missing: {path}", file=sys.stderr)
        continue
    with open(path) as f:
        src = f.read()
    if "set_device(local_worker_id) % " in src or "% max(1, torch_dev.device_count())" in src:
        print(f"[lmcache-mp-engine] device-ordinal clamp already present: {path}")
        patched_any = True
        continue
    old = "set_device(local_worker_id)"
    if old not in src:
        print(f"[lmcache-mp-engine] device-ordinal clamp not applicable (no direct set_device(local_worker_id) in {path})")
        continue
    new = "set_device(local_worker_id % max(1, torch_dev.device_count()))"
    new_src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(new_src)
    print(f"[lmcache-mp-engine] patched device-ordinal clamp in {path}")
    patched_any = True

if not patched_any:
    print("[lmcache-mp-engine] no device-ordinal patch needed (no vulnerable pattern found)")
PYEOF

echo "$PREFIX ok engine-side MP connector ready"

# --- 3. Fork patch: timeline-semaphore handle struct vs cuda-python bindings ---
# The pinned cuda-python bindings ship cudaIpcMemHandle_t/CUipcMemHandle as
# opaque objects exposing getPtr() (address of the internal 64-byte buffer),
# NOT a `.reserved` byte-array field. The lmcache timeline-semaphore event
# backend (used by the isolated-IPC / lmcache_driven corridor) reads/writes
# `.reserved` directly and raises "'cudaIpcMemHandle_t' object has no
# attribute 'reserved'" at KV-cache registration. Shim both sites via
# ctypes (the object IS ctypes-backed; getPtr() is a live buffer pointer,
# verified live on the engine container).
# Idempotent; log-and-continue when the pattern is absent (wheel drift).
python3 - <<'PYEOF'
import glob
import re

# The pinned cuda-python bindings ship opaque getPtr()-backed handle objects
# WITHOUT .reserved. THREE sites in two files touch it: the timeline-semaphore
# event backend (export + import) and the RawCudaIPCWrapper KV-tensor path
# (reserved-blob stash + restore). Shim all of them via ctypes — getPtr()
# returns the live 64-byte buffer pointer (verified live on the engine
# container). Idempotent per site; log-and-continue on wheel drift.

def _ensure_ctypes(src):
    if re.search(r"^import ctypes$", src, re.M) is None:
        src = src.replace("import struct\n", "import struct\nimport ctypes\n", 1)
    return src

for path in glob.glob(
    "/usr/local/lib/python3.12/dist-packages/lmcache/v1/platform/cuda/timeline_semaphore_event_ipc.py"
) + glob.glob(
    "/usr/local/lib/python3.12/dist-packages/lmcache/v1/platform/cuda/ipc_wrapper.py"
):
    with open(path) as f:
        src = f.read()
    orig_src = src
    src = src.replace(
        "ipc_handle.reserved = handle_bytes",
        "ctypes.memmove(ipc_handle.getPtr(), bytes(handle_bytes), 64)",
        1,
    )
    src = src.replace(
        "bytes(handle.reserved)",
        "ctypes.string_at(handle.getPtr(), 64)",
        1,
    )
    src = src.replace(
        "self._ipc_handle_reserved = bytes(ipc_handle.reserved)",
        "self._ipc_handle_reserved = ctypes.string_at(ipc_handle.getPtr(), 64)",
        1,
    )
    src = src.replace(
        "handle.reserved = self._ipc_handle_reserved",
        "ctypes.memmove(handle.getPtr(), bytes(self._ipc_handle_reserved), 64)",
        1,
    )
    if src == orig_src:
        print(f"[lmcache-mp-engine] handle shim already applied or no vulnerable pattern: {path}")
        continue
    src = _ensure_ctypes(src)
    with open(path, "w") as f:
        f.write(src)
    print(f"[lmcache-mp-engine] patched handle shim in {path}")
PYEOF