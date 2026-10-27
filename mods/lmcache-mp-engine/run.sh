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
#   control, no LMCache            : mean accept len 3.32-4.10, pos0 76-83%
#   with LMCache, L2 empty (fresh) : mean accept len 4.32,      pos0 ~78%
#   with LMCache, L2 populated     : mean accept len 2.6-3.4,   pos0 ~43%
# Acceptance is fine until L2 holds data, then collapses -- consistent with the
# drafter's KV being served back from L2. The mooncake store-l2 path documents
# the same failure and already guards it
# (mods/mooncake-worker/.../store/worker.py:_drafter_group_ids, "spec-decode
# acceptance collapses to zero, live 2026-10-25"). The MP path had no guard:
# the MP server log shows the drafter as its own kernel group
#   KernelGroupInfo(layers=5, indices=64-68, nh=4, hs=256, uint8)
# beside target groups (nh=2, hs=512, int8) and
#   "Registered KV cache for GPU ID ... with 69 layers"   # 64 target + 5 drafter
#
# DETECTION IS CONFIG-FREE AND NAME-FORMAT-FREE. Two earlier revisions failed:
#   (a) reading vllm_config.speculative_config raised
#       "Current vLLM config is not set" at group-construction time;
#   (b) assuming layer names end in ".layers.<N>" and that the target group
#       starts at index 0 -- live log said "no group starts at 0 - skipped".
# So this revision derives the boundary from the DATA: group the registered
# layer names by their prefix (everything before ".layers."), take the prefix
# carrying the MOST layers as the target, and treat any vLLM group whose names
# use a different prefix as the drafter's. That holds regardless of how the
# names are actually spelled, and needs no vLLM config.
#
# Every path prints to STDOUT (not a logger that may be unconfigured), and the
# first few real layer names are printed once, so an inert or misaimed patch is
# always visible in `docker logs vllm_node`.
# Fail-safe: on any failure grouping is left untouched, so a target group is
# never dropped.
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
        "[lmcache-mp-engine] drafter exclusion: ANCHOR NOT UNIQUE (%d) - skipping"
        % src.count(anchor)
    )
    raise SystemExit(0)

inject = (
    "    # --- lmcache-mp-engine: exclude speculative-drafter KV groups ---\n"
    "    # Config-free and name-format-free: majority prefix == target.\n"
    "    if per_layer_group_idx is not None and vllm_groups:\n"
    "        def _pfx(_n):\n"
    "            _s = str(_n)\n"
    "            _i = _s.rfind('.layers.')\n"
    "            return _s[:_i] if _i >= 0 else _s\n"
    "        _by_pfx = {}\n"
    "        for _g in vllm_groups:\n"
    "            for _ln in getattr(_g, 'layer_names', ()) or ():\n"
    "                _by_pfx.setdefault(_pfx(_ln), 0)\n"
    "                _by_pfx[_pfx(_ln)] += 1\n"
    "        if len(_by_pfx) < 2:\n"
    "            print(\n"
    "                '[lmcache-mp-engine] drafter exclusion: only %d layer prefix(es) %r - no drafter, skipped'\n"
    "                % (len(_by_pfx), sorted(_by_pfx))\n"
    "            )\n"
    "        else:\n"
    "            _tgt = max(_by_pfx.items(), key=lambda kv: kv[1])[0]\n"
    "            _marked = 0\n"
    "            for _g in vllm_groups:\n"
    "                _names = list(getattr(_g, 'layer_names', ()) or ())\n"
    "                if _names and all(_pfx(_n) != _tgt for _n in _names):\n"
    "                    for _ln in _names:\n"
    "                        if _ln in layer_to_idx:\n"
    "                            per_layer_group_idx[layer_to_idx[_ln]] = EXCLUDED_ENGINE_GROUP\n"
    "                            _marked += 1\n"
    "            print(\n"
    "                '[lmcache-mp-engine] drafter exclusion: prefixes=%r target=%r marked=%d layers'\n"
    "                % (sorted(_by_pfx), _tgt, _marked)\n"
    "            )\n"
    "            _sample = [str(_n) for _g in vllm_groups for _n in (getattr(_g, 'layer_names', ()) or ())]\n"
    "            print('[lmcache-mp-engine] sample layer names: %r' % (_sample[:3],))\n"
    "    # --- end lmcache-mp-engine patch ---\n"
    "\n"
)

src = src.replace(anchor, inject + anchor, 1)
with open(path, "w") as f:
    f.write(src)
compile(open(path).read(), path, "exec")
print("[lmcache-mp-engine] drafter exclusion: patched into", path)
PYEOF

# --- 4b. Drop excluded groups from the REGISTERED set -------------------------
# The patch above only marks drafter layers EXCLUDED_ENGINE_GROUP, which keeps
# them out of the engine-group list. It does NOT remove them from the
# kv_caches dict handed to worker_adapter.register_kv_caches(), so the server
# still registers every layer (observed live 2026-10-27: "Registered KV cache
# for GPU ID ... with 69 layers" = 64 target + 5 drafter, while the group dump
# stops at index 63). A 69-wide registration against a 64-wide group map shifts
# every restored block by 5 slots, so L2 hits return KV belonging to the wrong
# layer -> DFlash's context selector reads garbage -> 0% draft acceptance at
# every position (Mean acceptance length pinned at 1.00).
#
# Same root cause as the PD-path fix (commit 1a580fb: "exclude
# speculative-drafter KV from P2P registration") -- that one patched the
# mooncake worker; this is the MP-corridor equivalent.
#
# Fail-safe: if the anchor is missing or already patched, log and continue
# (never drop target layers).
python3 - <<'PYEOF'
import glob

target = "/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/lmcache_mp_connector.py"
paths = glob.glob(target)
if not paths:
    print("[lmcache-mp-engine] register-drop: lmcache_mp_connector.py absent, skipping")
    raise SystemExit(0)

path = paths[0]
with open(path) as f:
    src = f.read()

MARK = "lmcache-mp-engine: drop excluded layers from registration"
if MARK in src:
    print("[lmcache-mp-engine] register-drop: already applied")
    raise SystemExit(0)

anchor = (
    "        self.worker_adapter.register_kv_caches(\n"
    "            kv_caches,\n"
    "            engine_group_infos=engine_group_infos,\n"
)
if src.count(anchor) != 1:
    print(
        "[lmcache-mp-engine] register-drop: ANCHOR NOT UNIQUE (%d) - skipping"
        % src.count(anchor)
    )
    raise SystemExit(0)

inject = (
    "        # --- lmcache-mp-engine: drop excluded layers from registration ---\n"
    "        # engine_group_infos only cover layers that belong to a group (the\n"
    "        # drafter layers are EXCLUDED_ENGINE_GROUP and form none). The raw\n"
    "        # kv_cache_config still lists every layer name, so grouping by it\n"
    "        # changes nothing -- key off the engine group layer_indices instead.\n"
    "        # Registering beyond the grouped width widens the transfer map past\n"
    "        # the group map and shifts every restored block.\n"
    "        try:\n"
    "            _grouped = set()\n"
    "            for _gi in (engine_group_infos or ()):\n"
    "                for _li in (getattr(_gi, 'layer_indices', ()) or ()):\n"
    "                    _grouped.add(int(_li))\n"
    "            _names = list(kv_caches)\n"
    "            if _grouped and len(_grouped) < len(_names):\n"
    "                _kept = [_n for _i, _n in enumerate(_names) if _i in _grouped]\n"
    "                _gone = [_n for _i, _n in enumerate(_names) if _i not in _grouped]\n"
    "                print(\n"
    "                    '[lmcache-mp-engine] register-drop: %d registered -> %d '\n"
    "                    '(dropped %d: %r)'\n"
    "                    % (len(_names), len(_kept), len(_gone), _gone[:6])\n"
    "                )\n"
    "                kv_caches = {_n: kv_caches[_n] for _n in _kept}\n"
    "            else:\n"
    "                print(\n"
    "                    '[lmcache-mp-engine] register-drop: nothing to drop '\n"
    "                    '(%d grouped, %d registered)'\n"
    "                    % (len(_grouped), len(_names))\n"
    "                )\n"
    "        except Exception as _e:\n"
    "            print('[lmcache-mp-engine] register-drop: skipped (%r)' % (_e,))\n"
    "        # --- end lmcache-mp-engine register-drop ---\n"
)

src = src.replace(anchor, inject + anchor, 1)
with open(path, "w") as f:
    f.write(src)
compile(open(path).read(), path, "exec")
print("[lmcache-mp-engine] register-drop: patched into", path)
PYEOF

# --- 4c. Store-only L2: keep saves, suppress loads/lookups --------------------
# L2 restore corrupts this hybrid stack's speculative drafter: every load
# collapses draft acceptance to ~0 (MAL 1.00, per-position 0.000 x7,
# drafted>0/accepted=0) for the rest of the boot, starting in the same 10 s
# window as the first retrieve. Measured live 2026-10-27 on
# qwen3.8-27b-coder390-2x-lmcache-l2 with FS-NVMe + FS-NAS + valkey adapters
# all healthy and registration already reconciled to 64 layers:
#
#   L2 off (L1-only)     -> MAL 3.96-5.43, draft acceptance 42-64%
#   any single L2 restore -> MAL 1.00-1.30, draft acceptance  1-7%
#   (--separate-object-groups did not help: 1648-slot attention chunks and
#    16-slot GDN state still share an object group.)
#
# Same signature and same remedy the sibling Mooncake recipes took in
# a87632d7a0 ("spec-decode acceptance collapses to zero for the rest of a
# boot after the first store load"): keep the durable stores, turn the load
# probe off. The MP corridor has no enable_lookup knob, so this patch adds the
# equivalent as lmcache.mp.store_only in --kv-transfer-config.
#
# Default false: nothing changes unless a recipe opts in. Fail-safe: an
# unparseable extra_config logs and leaves load suppression OFF, so loads are
# never silently disabled by accident.
python3 - <<'PYEOF'
import glob

target = "/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/lmcache_mp_connector.py"
paths = glob.glob(target)
if not paths:
    print("[lmcache-mp-engine] store-only: lmcache_mp_connector.py absent, skipping")
    raise SystemExit(0)

path = paths[0]
with open(path) as f:
    src = f.read()

MARK = "lmcache-mp-engine: store-only L2"
if MARK in src:
    print("[lmcache-mp-engine] store-only: already applied")
    raise SystemExit(0)

# Anchor 1: the eager_prefetch parse in __init__ (sanctioned accessor).
anchor_init = (
    "        self._eager_prefetch: bool = bool(\n"
    "            vllm_config.kv_transfer_config.get_from_extra_config(\n"
    '                "lmcache.mp.eager_prefetch", False\n'
    "            )\n"
    "        )\n"
)
# Anchor 2: start_load_kv method signature. Deliberately the `def` line, not
# the body: "metadata = self._get_connector_metadata()" +
# "assert isinstance(metadata, LMCacheMPConnectorMetadata)" appears TWICE in
# this wheel (start_load_kv:604 and wait_for_save:678), so a body anchor is
# not unique and would make the whole patch skip.
anchor_load = (
    '    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:\n'
)
# Anchor 3: on_new_request body (eager lookup submission).
anchor_new = (
    "        if not self._eager_prefetch or request.resumable:\n"
    "            return\n"
)

# Anchor 4: get_num_new_matched_tokens signature. This is the load DECISION
# path -- it submits the lookup (maybe_submit_lookup_request, ~line 902) and
# reports the matched-token count back to vLLM, which is what actually drives
# the WAITING_FOR_REMOTE_KVS state and the subsequent retrieve. Guarding only
# start_load_kv/on_new_request left this path live (observed live 2026-10-27:
# lookup_requests_total and prefetch_load_submitted_requests_total still
# advanced with store_only=true). Returning (0, False) means "no external
# tokens", so vLLM computes the prompt locally and never waits on a load.
anchor_matched = (
    "    def get_num_new_matched_tokens(\n"
    "        self,\n"
    '        request: "Request",\n'
    "        num_computed_tokens: int,\n"
    "    ) -> tuple[int | None, bool]:\n"
)

for name, a in (
    ("init", anchor_init),
    ("start_load_kv", anchor_load),
    ("on_new_request", anchor_new),
    ("get_num_new_matched_tokens", anchor_matched),
):
    if src.count(a) != 1:
        print(
            "[lmcache-mp-engine] store-only: ANCHOR %s NOT UNIQUE (%d) - skipping"
            % (name, src.count(a))
        )
        raise SystemExit(0)

inject_init = (
    "        # --- lmcache-mp-engine: store-only L2 ---\n"
    "        # lmcache.mp.store_only=true keeps L2 saves (durable prompt-KV\n"
    "        # recording) and suppresses every load/lookup, because an L2\n"
    "        # restore corrupts this hybrid stack's speculative drafter.\n"
    "        try:\n"
    "            self._lmcache_store_only: bool = bool(\n"
    "                vllm_config.kv_transfer_config.get_from_extra_config(\n"
    '                    "lmcache.mp.store_only", False\n'
    "                )\n"
    "            )\n"
    "        except Exception as _e:\n"
    "            self._lmcache_store_only = False\n"
    "            print('[lmcache-mp-engine] store-only: extra_config parse failed (%r) - loads stay ENABLED' % (_e,))\n"
    "        if self._lmcache_store_only:\n"
    "            print('[lmcache-mp-engine] store-only: L2 loads disabled, stores active (config-driven)')\n"
    "        # --- end lmcache-mp-engine store-only ---\n"
)

inject_load = (
    '    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:\n'
    "        # --- lmcache-mp-engine: store-only L2 (suppress load) ---\n"
    "        if getattr(self, '_lmcache_store_only', False):\n"
    "            return\n"
    "        # --- end lmcache-mp-engine store-only ---\n"
)

inject_new = (
    "        # --- lmcache-mp-engine: store-only L2 (suppress eager lookup) ---\n"
    "        if getattr(self, '_lmcache_store_only', False):\n"
    "            return\n"
    "        # --- end lmcache-mp-engine store-only ---\n"
)

inject_matched = (
    "    def get_num_new_matched_tokens(\n"
    "        self,\n"
    '        request: "Request",\n'
    "        num_computed_tokens: int,\n"
    "    ) -> tuple[int | None, bool]:\n"
    "        # --- lmcache-mp-engine: store-only L2 (no external tokens) ---\n"
    "        # Report zero matched tokens so vLLM computes locally and never\n"
    "        # enters WAITING_FOR_REMOTE_KVS. This is the load-decision path;\n"
    "        # guarding start_load_kv alone leaves the lookup/retrieve live.\n"
    "        if getattr(self, '_lmcache_store_only', False):\n"
    "            return 0, False\n"
    "        # --- end lmcache-mp-engine store-only ---\n"
)

src = src.replace(anchor_init, anchor_init + inject_init, 1)
# inject_load already repeats the signature line, so replace (not prepend).
src = src.replace(anchor_load, inject_load, 1)
src = src.replace(anchor_new, inject_new + anchor_new, 1)
src = src.replace(anchor_matched, inject_matched, 1)

with open(path, "w") as f:
    f.write(src)
compile(open(path).read(), path, "exec")
print("[lmcache-mp-engine] store-only: patched into", path)
PYEOF
