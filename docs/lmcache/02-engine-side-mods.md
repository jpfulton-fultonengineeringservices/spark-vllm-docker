# Engine-Side LMCache Mods

**Scope** - This document describes the two engine-side LMCache mods that run **before** the recipe `exec-script.sh` is sourced: `mods/lmcache-mp-engine` (MP corridor) and `mods/lmcache-offload` (in-process KV offload). Both mods are invoked by `launch-cluster.sh` via `apply_mod_to_container()` with **no arguments**. The mods self-activate by grepping for their own activation marker in the copied `exec-script.sh` (see § 1.1). Because the recipe `env:` is invisible to the mod process, any runtime-required variable must be supplied **three-ways** (see § 1.2). For corridor topology, CUDA-IPC wiring, and the six failure modes, see the sibling document [`01-architecture-and-corridors.md`](./01-architecture-and-corridors.md).

---

## 1. Mod contract & self-activation

### 1.1 Guard logic (exact source)

*MP-engine* - `mods/lmcache-mp-engine/run.sh` lines 42-53:

```bash
# Opt-in. recipe env NOT visible here: mods run BEFORE
# exec-script exports recipe env, in separate process. Self-activate
# the exec-script (launch-cluster.sh copies /workspace/exec-script.sh
# before applying mods) by grepping MP connector recipe pins.
EXEC_SCRIPT="${WORKSPACE_DIR:-/workspace}/exec-script.sh"
if [ "${LMCACHE_MP_ENABLED:-0}" = "1" ] \
|| { [ -f "$EXEC_SCRIPT" ] && grep -q 'LMCacheMPConnector' "$EXEC_SCRIPT" 2>/dev/null; }; then
:
else
echo "$PREFIX inert (exec-script does not LMCacheMPConnector; export LMCACHE_MP_ENABLED=1 to force)"
return 0 2>/dev/null || exit 0
fi
```

*Off-load* - `mods/lmcache-offload/run.sh` lines 37-44:

```bash
EXEC_SCRIPT="${WORKSPACE_DIR:-/workspace}/exec-script.sh"
if [ "${LMCACHE_OFFLOAD_ENABLED:-0}" = "1" ] \
   || { [ -f "$EXEC_SCRIPT" ] && grep -q 'LMCACHE_CONFIG_FILE' "$EXEC_SCRIPT" 2>/dev/null; }; then
    :
else
    echo "$PREFIX inert (recipe exec-script does not reference LMCACHE_CONFIG_FILE; export LMCACHE_OFFLOAD_ENABLED=1 to force)"
    return 0 2>/dev/null || exit 0
fi
```

Both guards run **before** the recipe `env:` values are exported, so the mod process can never read them. Self-activation therefore relies on the marker the recipe pins in the copied `exec-script.sh` (`LMCacheMPConnector` for the MP corridor, `LMCACHE_CONFIG_FILE` for the offload corridor). The explicit env override (`LMCACHE_MP_ENABLED=1` / `LMCACHE_OFFLOAD_ENABLED=1`) still forces activation for debug or bare-mod testing.

### 1.2 Runtime-required env-chain (cautionary tale)

| Channel | What it carries | Where it is set | Example value |
|---------|----------------|----------------|---------------|
| **Recipe env** (`env:`) | `LMCACHE_CHUNK_SIZE` - the KV-cache block size the **engine-side connector** reads at serve-time | Inside the recipe YAML (`LMCACHE_CHUNK_SIZE: "1648"`) | `1648` |
| **Launcher `-e`** | Same `LMCACHE_CHUNK_SIZE` is passed to the **container** so the MP connector can read it from `os.getenv` | `spark-serve.sh` → `docker run -e LMCACHE_CHUNK_SIZE=$LMCACHE_CHUNK_SIZE ...` | `-e LMCACHE_CHUNK_SIZE=1648` |
| **MP-server CLI** | The **MP server** reads `--chunk-size` from its command line (it never looks at the recipe env) | `lmcache-server-entrypoint.sh` → `./lmcache_server --chunk-size $LMCACHE_CHUNK_SIZE ...` | `--chunk-size 1648` |

If any one of these three channels is missing, the server defaults to a safe (tiny) block size and the corridor degrades to recompute-only behavior. The three-channel requirement is documented here to avoid silent performance loss.

---

## 2. `mods/lmcache-mp-engine` - MP corridor (engine side)

### 2.1 Wheel install (exact snippet, lines 60-76)

```bash
: "${LMCACHE_MP_WHEEL_URL:=https://github.com/Fulton-Engineering-Services/dgx-spark-wheels/releases/download/lmcache-mooncake-v0.5.5rc1-cu13.3/lmcache-0.5.5rc1+cu13.3torch2.13.glibc239-cp312-cp312-linux_aarch64.whl}"
if ! python3 -c 'from lmcache.integration.vllm.utils import mla_enabled' >/dev/null 2>&1; then
    info "lmcache module missing; installing from ${LMCACHE_MP_WHEEL_URL##*/}"
    python3 -m pip install --no-cache-dir --no-deps "${LMCACHE_MP_WHEEL_URL}" >&2 \
        || fail "op=install_lmcache url='${LMCACHE_MP_WHEEL_URL}' reason=pip_failed"
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
```

*Exclusion rationale* (four items, table):

| Excluded | Rationale |
|----------|-----------|
| **torch** | Already built into the `vllm-node-b12x` image (from-source cu13.3); reinstalling would collide with the engine build. |
| **numpy** | The image pins its own compatible version; a second numpy would break ABI. |
| **cufile-python / pytest** | GDS is unsupported on GB10; pytest is only a test dependency. |
| **setuptools / setuptools_scm / nixl** | Already present in the image (build deps / `nixl_cu13`); re-installing invites drift. |

The import gate `from lmcache.integration.vllm.utils import mla_enabled` guarantees the wheel is importable before the mod proceeds.

### 2.2 Device-ordinal clamp (lines 85-126)

The patch targets `lmcache/integration/vllm/lmcache_mp_connector.py`. It is **INERT** on the pinned wheel: the MP connector derives its device from registered tensors and contains no direct `set_device(local_worker_id)` call, so the patcher logs *"device-ordinal clamp not applicable"* and continues. (The device-ordinal bug is live only in the in-process corridor - see § 3.4.)

```python
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
```

### 2.3 Multi-group block-id suppression (lines 152-184)

Exact replacement from the patcher:

```python
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
```

**Reasoning chain** (from the mod comments, lines 130-151):
1. vLLM's scheduler hard-raises for models with > 1 KV cache group when a connector reports a flat `invalid_block_ids` set: *"A KV connector reported block-level load failures (invalid_block_ids) which is not supported for models with multiple KV cache groups."* (vllm/v1/core/sched/scheduler.py:`_handle_invalid_blocks`).
2. A mamba-align hybrid **always** has > 1 group, and LMCache's MP adapter routes a failed retrieve (partial L2 hit after restart, unhealthy server) into `error_block_ids`, which vLLM surfaces as `invalid_block_ids` → EngineCore dies.
3. Fix: stop emitting block-level failures entirely. Under vLLM's default `kv_load_failure_policy=recompute`, a request whose KV never arrives simply recomputes its prefix - the desired degraded behavior - and is still released through `finished_recving` by the normal path.
4. Reporting per-request failures via `failed_recving` was tried and **rejected**: vLLM's `_update_from_kv_xfer_finished` asserts `req_id in self.requests` and dies for requests that already completed.

**Residual risk** (plain): a request that genuinely cannot be served from L2 will always recompute its prefix; the patch masks the scheduler abort but does not eliminate the latency cost of a cold/unhealthy L2.

### 2.4 ctypes handle shim (lines 198-255)

The pinned cuda-python bindings expose `cudaIpcMemHandle_t`/`CUipcMemHandle` as opaque objects with only `getPtr()` (address of the internal 64-byte buffer), no `.reserved` attribute. The shim rewrites four sites across **both** files - `lmcache/v1/platform/cuda/timeline_semaphore_event_ipc.py` (export + import) and `lmcache/v1/platform/cuda/ipc_wrapper.py` (`RawCudaIPCWrapper` stash + restore) - because both files touch `.reserved`:

```python
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
```

`import ctypes` must anchor on the file's first plain import (`re.search(r"^(import \w+\n)", src, re.M)`) because `ipc_wrapper.py` imports neither `struct` nor `ctypes`, while `timeline_semaphore_event_ipc.py` imports `struct`. The patcher is idempotent per site and log-and-continues on wheel drift.

---

## 3. `mods/lmcache-offload` - in-process corridor

### 3.1 Wheel install & valkey-glide-sync (lines 52-84)

Same wheel URL and `--no-deps` exclusion filter as the MP mod (§ 2.1), plus a **second import gate** for the synchronous GLIDE client required by the `valkey://` L2 connector (the async `valkey-glide` package is a different module namespace):

```bash
# The valkey:// L2 connector needs the synchronous GLIDE client (the async
# valkey-glide package is a different module namespace).
if ! python3 -c 'import glide_sync' >/dev/null 2>&1; then
    info "installing valkey-glide-sync for the valkey:// L2 connector"
    python3 -m pip install --no-cache-dir 'valkey-glide-sync>=2.3.0' >&2 \
        || fail "op=install_valkey_glide reason=pip_failed"
```

### 3.2 Rendered config (`/tmp/lmcache-config.yaml`)

Lines 92-108 render the requester-only YAML consumed by the in-process `LMCacheConnectorV1`. Per-node L2 subdir under `/cluster-shared/lmcache-l2/${HOSTNAME}` avoids the bare NFS dir the PD MP-server fleet writes to; `remote_url` points at Valkey on Dell. No PD, no MP server, no RDMA. (Contrast: the MP server corridor takes L2 adapters via the old `--l2-adapter` JSON CLI blocks in `lmcache-server-entrypoint.sh`, not via this YAML.)

```yaml
# Rendered by mods/lmcache-offload/run.sh -- LMCacheConnectorV1 offload+durability.
chunk_size: 256
local_cpu: true
max_local_cpu_size: ${L1_GIB}
local_disk: ${L2_DIR}
max_local_disk_size: ${DISK_GIB}
save_unfull_chunk: false
save_decode_cache: true
remote_url: "valkey://10.10.10.53:6379"
```

The path defaults to `/tmp/lmcache-config.yaml` and **must equal** the recipe's `LMCACHE_CONFIG_FILE` env value (the offload mod cannot read that env itself - § 1).

### 3.3 Uniformity-probe patch (lines 123-178)

Two sub-patches to `vllm/v1/core/kv_cache_utils.py`:

**(a) Catch `ValueError` in the uniformity probe** - with `--disable-hybrid-kv-cache-manager`, `get_kv_cache_groups()` runs `unify_hybrid_kv_cache_specs()`, whose `is_kv_cache_spec_uniform()` probe calls `specs[0].merge(all)` and catches **only** `AssertionError`. `FullAttentionSpec.merge`'s `merge_window_sizes()` raises `ValueError` on a mixed window set (MiMo: SWA layers window=128 vs full-attention layers window=None). Catching `ValueError` lets the probe correctly report *not uniform*:

```python
old = """    except AssertionError:
        return False
    return True"""
new = """    except (AssertionError, ValueError):
        return False
    return True"""
```

**(b) Normalize mixed sliding windows after promotion** - the normalization catches layers (e.g. DFlash drafter) that are already `FullAttentionSpec` with a non-`None` window, which the drop in (a) doesn't cover. Code is identical in structure to the vendored-source fix in `gpu-cluster-forks/vllm/vllm/v1/core/kv_cache_utils.py` `_promote_local_kv_cache_specs`:

```python
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
```

Both sub-patches are guarded: applied → log "already applied"; anchor present → apply; anchor absent (fork drift) → WARN + continue.

### 3.4 GPU-connector clamp (lines 186-230)

Targets `lmcache/v1/gpu_connector/__init__.py`. This patch is **LIVE** in the in-process corridor (unlike § 2.2): `LMCacheConnectorV1` reads `metadata.local_worker_id` and passes it directly to `torch_dev.set_device(local_worker_id)`. In cross-host TP=2, rank1 has `local_worker_id=1` but the node carries one GPU (`device_count=1`) → `CUDA error: invalid device ordinal`.

```python
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
if 'safe_id' in src:
    print("[lmcache-offload] gpu_connector device-string patch already applied")
else:
    import re
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
```

---

## 4. Patch-writing conventions (used by both mods)

- **Heredoc vs `-c`** - multi-line patches use `python3 - <<'PYEOF' ... PYEOF`; the whole patcher runs as one stdin script. Simple probes use `python3 -c '...'` one-liners.
- **Applied vs. skipped** - every patcher checks `if new in src` (or anchor-absent) first, prints an explicit *"already applied"* / *"not applicable"* / *"WARN ... continuing"* line, and exits 0 on skip. **Log-and-continue** is the rule on wheel drift (expected across image rebuilds); a patch aimed at the wrong file is worse than none, so each patcher dumps its target path list before editing.
- **Fail-hard only for critical steps** - `fail` (non-zero exit) is reserved for wheel install / import-gate failure. Non-critical mismatches are logged and ignored.
- **Idempotence** - every replace is guarded by a pre-check or a `src == orig` comparison, so re-running the mod on an existing container is a no-op.

---

## 5. Execution order diagram

```mermaid
flowchart TD
    A[launch-cluster.sh] --> B[Copy exec-script.sh into container]
    B --> C[apply_mod_to_container() - runs each mod/run.sh]
    C --> D[mods read env-markers in exec-script (self-activate)]
    D --> E[Serve command (vllm-node-b12x) sources recipe env]
    E --> F[LMCacheConnectorV1 (MP) / LMCacheConnectorV1 (offload) reads config]
    F --> G[Model inference]
```

---

**See also:** `01-architecture-and-corridors.md` (corridor topology, CUDA-IPC story, ownership matrix) and `05-failure-modes-and-patches.md` (the six defects referenced in § 1.2).

---

*All quoted snippets, line numbers, and file paths verified against `mods/lmcache-mp-engine/run.sh` (255 lines) and `mods/lmcache-offload/run.sh` (232 lines). No other files changed.*
