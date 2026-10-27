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

# --- 4. Fork patch: suppress flat block-id load failures (multi-group models) --
# vLLM's scheduler hard-raises for models with >1 KV cache group when a
# connector reports a flat `invalid_block_ids` set:
#   "A KV connector reported block-level load failures (invalid_block_ids)
#    which is not supported for models with multiple KV cache groups.
#    Connectors must report failed requests via
#    KVConnectorTransferResults.failed_recving instead."
# (vllm/v1/core/sched/scheduler.py:_handle_invalid_blocks). A mamba-align
# hybrid ALWAYS has >1 group, and LMCache's MP adapter routes a failed
# retrieve (partial L2 hit after a restart, unhealthy server) into
# `error_block_ids`, which vLLM surfaces as invalid_block_ids -> EngineCore
# dies.
#
# Fix: stop emitting block-level failures entirely. Under vLLM's default
# kv_load_failure_policy=recompute a request whose KV never arrives simply
# recomputes its prefix, which is exactly the desired degraded behavior; the
# request is still released through finished_recving by the normal path
# (upstream's own contract: "Failed receives also appear in finished_recving
# so the scheduler can release the request from its transfer wait state").
# Reporting per-request failures as well was tried and rejected: vLLM's
# _update_from_kv_xfer_finished asserts the id is still in scheduler.requests
# and dies for requests that already completed.
python3 - <<'PYEOF'
import glob

target = "/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/vllm_multi_process_adapter.py"
paths = glob.glob(target)
if not paths:
    print("[lmcache-mp-engine] multi-group failure patch not applicable (adapter absent)")
    raise SystemExit(0)

path = paths[0]
with open(path) as f:
    src = f.read()
orig = src

src = src.replace(
    """        errors = self.error_block_ids.copy()
        self.error_block_ids.clear()
        return errors""",
    """        # Flat block ids are ambiguous across KV cache groups and vLLM
        # hard-rejects them for multi-group models; failures are handled by
        # the recompute policy via the normal finished_recving path.
        self.error_block_ids.clear()
        return set()""",
    1,
)

if src == orig:
    print("[lmcache-mp-engine] multi-group failure patch already applied or no anchor")
else:
    with open(path, "w") as f:
        f.write(src)
    print(f"[lmcache-mp-engine] suppressed block-level load failures in {path}")
compile(open(path).read(), path, "exec")
print("[lmcache-mp-engine] adapter compiles OK")
PYEOF

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
        # Files differ: timeline_semaphore_event_ipc.py imports struct;
        # ipc_wrapper.py imports only threading/torch. Insert after the
        # module's first plain import so both are covered.
        m = re.search(r"^(import \w+\n)", src, re.M)
        if m is None:
            m = re.search(r"^(from __future__ .*)\n", src, re.M)
        src = src[: m.end()] + "import ctypes\n" + src[m.end():]
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

# --- 5. Fork patch: exclude the speculative drafter's KV groups ---------------
# A speculative drafter builds its OWN KV during decode; LMCache must never
# store or load it. Measured on qwen3.8-27b-coder390 (2x TP2, DFlash2,
# num_speculative_tokens=7), identical 400-token generations:
#   control, no LMCache           : mean accept len 3.32-4.10, pos0 76-83%
#   with LMCache, L2 empty (fresh): mean accept len 4.32,      pos0 ~78%
#   with LMCache, L2 populated    : mean accept len 2.6-3.4,   pos0 ~43%
# i.e. acceptance is fine until L2 holds data, then collapses -- consistent
# with the drafter's KV being served back from L2 and corrupting it. The
# mooncake store-l2 path documents the same failure and already guards it
# (mods/mooncake-worker/.../store/worker.py:_drafter_group_ids, "spec-decode
# acceptance collapses to zero, live 2026-10-25"). The MP path had no guard:
# the MP server log shows the drafter registered as its own kernel group
#   KernelGroupInfo(layers=5, indices=64-68, nh=4, hs=256, uint8)
# alongside target groups (nh=2, hs=512, int8) and
#   "Registered KV cache for GPU ID ... with 69 layers"   # 64 target + 5 drafter
#
# DETECTION IS CONFIG-FREE ON PURPOSE. The obvious approach -- read
# vllm_config.speculative_config -- does NOT work here: get_current_vllm_config()
# raises "Current vLLM config is not set" at this point in startup, which an
# earlier revision of this patch swallowed, making it silently inert (measured:
# drafter still registered with 69 layers). Instead the boundary is derived
# from the registered layer NAMES, which are disjoint by construction:
#   target  ...language_model.model.layers.N   N in [0, 64)
#   drafter ...model.layers.N                  N in [64, 69)
# A vLLM group whose indices all sit at/above the largest "compact prefix"
# boundary of the other group is the drafter's. Concretely: take the minimum
# index over all groups that start at 0 (the target, always), and treat any
# group whose indices are all >= that target's max+1 as the drafter.
#
# Fail-safe: every failure path prints to STDOUT (not a logger that may be
# unconfigured) and leaves grouping untouched, so a target group is never
# dropped, and an inert patch is always visible in `docker logs vllm_node`.
python3 - <<'PYEOF'
import glob

target = "/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/kv_cache_groups.py"
paths = glob.glob(target)
if not paths:
    print("[lmcache-mp-engine] drafter exclusion: kv_cache_groups.py absent, skipping")
    raise SystemExit(0)

path = paths[0]
with open(path) as f:
    src = f.read()

MARK = "lmcache-mp-engine: exclude speculative-drafter"
if MARK in src:
    print("[lmcache-mp-engine] drafter exclusion: already applied")
    raise SystemExit(0)

anchor = "    return [\n        EngineGroupInfo(\n"
if src.count(anchor) != 1:
    print(
        "[lmcache-mp-engine] drafter exclusion: ANCHOR NOT UNIQUE (%d) - skipping, grouping unchanged"
        % src.count(anchor)
    )
    raise SystemExit(0)

inject = (
    "    # --- lmcache-mp-engine: exclude speculative-drafter KV groups ---\n"
    "    # Config-free by design: get_current_vllm_config() is NOT valid here.\n"
    "    if per_layer_group_idx is not None and vllm_groups:\n"
    "        _gi = []\n"
    "        for _g in vllm_groups:\n"
    "            _s = set()\n"
    "            for _ln in getattr(_g, 'layer_names', ()) or ():\n"
    "                try:\n"
    "                    _s.add(int(str(_ln).rsplit('.', 1)[-1]))\n"
    "                except Exception:\n"
    "                    pass\n"
    "            _gi.append(_s)\n"
    "        # The target group always starts at 0; its max index bounds the\n"
    "        # target's layer range. Anything entirely above that is drafter.\n"
    "        _starts_at_zero = [_s for _s in _gi if _s and 0 in _s]\n"
    "        if _starts_at_zero:\n"
    "            _boundary = max(max(_s) for _s in _starts_at_zero) + 1\n"
    "            _marked = 0\n"
    "            for _s, _g in zip(_gi, vllm_groups):\n"
    "                if _s and min(_s) >= _boundary:\n"
    "                    for _ln in _g.layer_names:\n"
    "                        if _ln in layer_to_idx:\n"
    "                            per_layer_group_idx[layer_to_idx[_ln]] = EXCLUDED_ENGINE_GROUP\n"
    "                            _marked += 1\n"
    "            print(\n"
    "                '[lmcache-mp-engine] drafter exclusion: boundary=%d marked=%d layers'\n"
    "                % (_boundary, _marked)\n"
    "            )\n"
    "        else:\n"
    "            print('[lmcache-mp-engine] drafter exclusion: no group starts at 0 - skipped')\n"
    "    # --- end lmcache-mp-engine patch ---\n"
    "\n"
)

src = src.replace(anchor, inject + anchor, 1)
with open(path, "w") as f:
    f.write(src)
compile(open(path).read(), path, "exec")
print("[lmcache-mp-engine] drafter exclusion: patched into", path)
PYEOF
