# Qwen3.8-27B-Coder390 on `vllm-node-b12x`: the B12X attention gap

Status: **open**. The `qwen3.8-27b-coder390-{2x,4x}` recipes serve again, but
without B12X paged attention. This document records what broke, what was
fixed, and the paths to restore the measured B12X win.

Applies to image `vllm-node-b12x:latest` (`a1fc4f18d3cb`, built 2026-10-16,
`vllm 0.32.0rc1.dev301+g1ac942555.d20261016`, `flashinfer-python 0.7.2`).

## 1. Summary

Four independent defects surfaced in sequence while bringing this recipe up.
Each one masked the next, so the launch looked like a single failure.

| # | Defect | Effect | State |
|---|---|---|---|
| 1 | `b12x` vendored inside flashinfer; mods keyed on the old layout | setup aborted at `apply-mod` | **fixed** |
| 2 | `b12x` dist metadata gone; kernel-cache key resolution silently degraded | stale kernel archives could be restored across b12x versions | **fixed** |
| 3 | `--gdn-decode-kernel` removed, `--gdn-prefill-backend b12x` renamed | engine died at argparse | **fixed** |
| 4 | `--load-format b12x` is Flash-Next-only | engine died at weight load | **fixed** |
| 5 | B12X attention + hybrid mamba block alignment are mutually exclusive | engine died at KV block-size validation | **worked around** (B12X disabled) |

## 2. Defect 1 — b12x is now vendored inside flashinfer

Deployed images no longer ship a standalone `b12x` distribution. Site-packages
`b12x/` contains only `__init__.py` (1727 bytes), a `sys.meta_path` aliaser;
the implementation lives in `flashinfer/experimental/b12x/`. There is no `b12x`
dist metadata at all — `importlib.metadata.version("b12x")` raises
`PackageNotFoundError`.

Anything resolving the package by path broke:

```
b12x-reclaim-gate: Missing
  /usr/local/lib/python3.12/dist-packages/b12x/preparation/session.py
```

`mods/b12x-reclaim-gate/run.sh` (v3) exited 1 here, which aborted the whole
recipe apply before the engine ever started.

Resolution: `run.sh` v4 discovers the target in this order —

1. `B12X_SESSION_PATH` (explicit override, must exist),
2. `python3 -c 'import importlib.util as u; u.find_spec("b12x.preparation.session").origin'`
   — portable across both layouts,
3. legacy `$PYTHON_ROOT/b12x/preparation/session.py`,
4. otherwise fail, naming both layouts and the probe.

While there, v4 also fixed two latent bugs in the gate itself:

- v3 called `logger.warning_once(...)`. The vendored module's logger is a plain
  `logging.Logger`; vLLM patches `warning_once` only onto loggers it creates via
  `init_logger`, so v3 raised `AttributeError` the moment the gate fired. v4 uses
  `logger.warning(...)`.
- The sweep exists at **two** sites, not one. `PreparationSession._reclaim_programs`
  was gated; `PreparationJob._run`'s finishing sweep — the same unguarded
  retained-owner walk — was not. v4 gates both.

Site 2 needs care because `keep` is consumed downstream:

```python
# PreparationJob._run, after the gated block
reclaim_device_memory(keep, stack_limit=self._stack_limit)
```

and `program_cache.reclaim_device_memory` rewrites the CUDA stack limit even for
an empty `keep`:

```python
def reclaim_device_memory(keep, *, stack_limit):
    gc.collect()
    if stack_limit is None:
        return
    limit = stack_limit
    for program in keep: ...          # max over retained kernels' local size
    for value in (0, limit):
        _cudaDeviceSetLimit(cudaLimitStackSize, value)
```

Passing the baseline with an empty set would *shrink* the per-thread stack below
what retained kernels need. v4 therefore rewrites the call to
`stack_limit=None if not keep else self._stack_limit`, taking the documented
host-device path (`gc.collect()` only), which still unloads losing-candidate
libraries without touching the CUDA stack. `self._program_cache.clear()` stays
unconditional above the gate: it is an O(1) dict drop, not the expensive walk.

## 3. Defect 2 — kernel-cache key silently hashed a failure string

`mods/b12x-kernel-cache/lib/key.sh` resolved the b12x revision from
`/workspace/b12x-source-commit` then `pyver b12x`. Both are absent now, so the
field came back empty. The guard was structurally broken:

```bash
F_B12X_REV=$(_require b12x_rev "$v")     # die() exits only the subshell
```

`die` ran inside command substitution, so it killed the subshell and its stdout
(the log line) was captured as the revision. The key still computed — rc 0 —
and that constant text is identical across every b12x version, so archives could
be restored across incompatible kernel builds.

Resolution:

- every identity field is validated in the current shell before assignment
  (`[ -n "$v" ] || die b12x_rev; F_X=$v`), so a failed lookup aborts with rc 3;
- revision precedence is `$B12X_KCACHE_ID_B12X_REV` →
  `/workspace/b12x-source-commit` → `flashinfer-python` version, since the
  vendored tree's identity *is* the flashinfer release;
- the key schema moved to `b12x-kcache/v2`, so every `v1` archive (all of which
  hashed the bogus string) can never match again. They age out via `prune`.

Observed on gx10-node1: old key `...tp1__fa6feac8b24be5a7` →
new key `...tp1__cf350b85ee72be40`, rc 0, no `cannot resolve` line.

## 4. Defect 3 — GDN flag removals

```
vllm serve: error: argument --gdn-prefill-backend: invalid choice: 'b12x'
  (choose from 'flashinfer', 'triton', 'cutedsl', 'aiter_flydsl')
```

- `--gdn-decode-kernel` **does not exist** in this build. Decode selection is the
  env var `VLLM_GDN_DECODE_KERNEL`, `Literal["cuda","triton"]`, default `cuda`
  (`vllm/envs.py:131`), consumed at `qwen_gdn_linear_attn.py:672`.
- `--gdn-prefill-backend b12x` → `cutedsl`, the CuteDSL kernel the b12x arm
  derived from (`qwen_gdn_linear_attn.py:255` maps `"cutedsl": "CuteDSL"`).

Fixed in all five GDN recipes (the three coder390 recipes and both
`qwen3.8-flash-next-nvfp4` recipes). No env var was added: `cuda` is already the
default.

## 5. Defect 4 — the b12x loader is Flash-Next-only

```
ValueError: the regular b12x loader currently supports Flash Next only
```

`vllm/model_executor/model_loader/b12x_loader.py:119-126` allow-lists exactly
`{Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration, Qwen4ExpMTP}`; the
class docstring is *"Opt-in Flash Next loader for Spark, including its embedded
MTP draft."* The coder390 checkpoint is
`Qwen3_5ForConditionalGeneration` (confirmed from the staged
`config.json`), so `--load-format b12x` could never load it.

Fixed by switching the three coder390 recipes to
`--load-format instanttensor`, which routes through `DefaultModelLoader`'s
safetensors iterator (`default_loader.py:342`, `weight_utils.py:1357`) with a
CUDA-resident read path and **no architecture allow-list** — the same choice the
sibling `qwen3.8-27b-nvfp4-dflash2` recipe already makes for this family.

The two `qwen3.8-flash-next-nvfp4` recipes keep `--load-format b12x`: their
checkpoint is `Qwen4ExpForConditionalGeneration`, which is exactly what the
loader supports.

## 6. Defect 5 — B12X attention vs hybrid mamba alignment (OPEN)

This is the remaining gap, and it is a genuine conflict inside the image, not a
recipe bug.

```
ValueError: Aligned KV cache block size 1664 is not supported by
attention backend(s): B12X.
```

Chain of causation:

1. `BasePlatform.update_block_size_for_backend` runs phase 2 for hybrid models:
   `if model_config.is_hybrid: cls._align_hybrid_block_size(...)`
   (`vllm/platforms/interface.py:785-788`).
2. `_align_hybrid_block_size` computes the per-token attention page size and the
   mamba page size, then:

   ```python
   attn_block_size = kernel_block_alignment_size * cdiv(
       mamba_page_size,
       kernel_block_alignment_size * attn_page_size_1_token,
   )
   ```

   With fp8 KV at TP=2 the mamba page is large enough that this lands on 1664
   tokens. (Observed log: *"Setting attention block size to 1664 tokens to ensure
   that attention page size is >= mamba page size"*, then *"Padding mamba page
   size by 1.71%"*.)
3. `_check_aligned_block_size` then validates the rounded size against every
   non-SSM backend (`interface.py:797-822`) and fails, because
   `B12xPagedAttentionBackend.get_supported_kernel_block_sizes()` returns
   `[64, 128]` (`vllm/v1/attention/backends/b12x.py:47`).

`--mamba-cache-mode` does not help: this build logs *"`--mamba-cache-mode` is
deprecated and will be removed in v0.33.0. It has no effect: Mamba prefix
caching follows `--enable-prefix-caching`."* With prefix caching on (required
for this workload) the alignment runs regardless.

Verification that the conflict is real and not a misconfiguration: driving
`EngineArgs.create_engine_config()` + `update_block_size_for_backend()` inside
the image with `--nnodes 2 --node-rank 0 --tensor-parallel-size 2` and the
recipe's full argv yields `pre-block: 64`, `post-block: 64` at `--max-model-len
4096`, but the real 262144-token launch lands at 1664 — the page-size arithmetic
depends on the final cache geometry, so short-context probes under-report it.

### What was done now

`--attention-backend B12X` and `--block-size 64` were removed from both coder390
recipes; they now use the image defaults (FlashInfer + `--block-size 16`), which
is exactly the pre-`caaba0f` configuration and matches both sibling recipes.
This restores a bootable serve at the cost of the perf arm.

### Cost

Commit `caaba0f` measured this arm on gx10-node1 TP=2 (C1/60k probe, 8
sequential prompts, temp 0, `ignore_eos`):

- per-stream decode **32.2 vs 26.4 tok/s (+22.1%)**
- mean TPOT **31.1 vs 37.9 ms (−18.0%)**
- accept length 3.04 vs 3.19 (kernel-driven, not acceptance-driven)
- cold-60k TTFT 66.6 vs 67.5 s (−1.3%)
- KV pool 3,346,412 tokens (−1.1%, within the 10% gate)

All of that is currently unrealized.

## 7. Resolution paths for defect 5

Ranked by preference. Each needs an image rebuild or a new mod, so none is a
recipe-only change.

### 5a. Widen B12X's accepted page sizes (preferred)

`_B12X_SUPPORTED_PAGE_SIZES = (64, 128)` is a Python-level allow-list, not
obviously a kernel constraint. If the paged kernels genuinely handle any
sufficiently large page (they already gate internally on
`key_page_size in _B12X_SUPPORTED_PAGE_SIZES` at `b12x.py:92`), then admitting
`MultipleOf(64)` — as most backends do via
`get_supported_kernel_block_sizes() -> [MultipleOf(1)]` — removes the conflict
outright.

Risk: the `b12x.py:92` runtime check would also have to accept the new sizes,
and if the CuTe kernels carry a real 64/128 constraint this fails at kernel
launch instead of at validation. Needs a source read of the paged-attention
kernel's page-size handling before committing.

### 5b. Make the hybrid aligner respect backend page-size limits

`_align_hybrid_block_size` rounds by `kernel_block_alignment_size` and the mamba
page size only; `_check_aligned_block_size` is a post-hoc assertion. Rounding to
a size that satisfies *both* the mamba page and the backend's supported set (or
failing with a clearer message that names the two constraints) is the more
upstream-correct fix, and would benefit any hybrid model on a page-size-limited
attention backend.

Risk: a fork-local vLLM patch, so it needs the fork-and-prove workflow rather
than a one-line recipe change. It also cannot always succeed — if no size in the
backend's set is >= the mamba page, the model genuinely cannot use that backend.

### 5c. Shrink the mamba page so alignment lands on 64/128

The page size scales with model shard count and KV dtype. Levers that could move
it: drop `--kv-cache-dtype fp8` (larger page — wrong direction), drop DFlash2
spec decode (removes the drafter's cache group), or change TP. Each is a
deliberate perf/quality choice, and none has been tried; the TP=2, fp8 configuration
is what `caaba0f` measured. Treat as a last resort.

### 5d. Accept the regression

Stay on FlashInfer + block 16 permanently. Lowest risk, gives up +22.1%
per-stream decode. Reasonable only if B12X on this hybrid architecture proves
unstable for other reasons.

## 8. Reproducing the diagnosis

All commands run on gx10-node1 through the `fulton-remote-shell` MCP
(`shell_open` host `gx10-node1`, then `shell_exec`); never raw `ssh`.

```bash
# 1. b12x layout
docker run --rm --entrypoint bash vllm-node-b12x:latest -c '
  ls /usr/local/lib/python3.12/dist-packages/b12x
  python3 -c "import importlib.util as u; \
    print(u.find_spec(\"b12x.preparation.session\").origin)"
  python3 -c "import importlib.metadata as m; m.version(\"b12x\")"   # PackageNotFoundError'

# 2. GDN flag table
docker run --rm --gpus all --entrypoint bash vllm-node-b12x:latest -c \
  'vllm serve --help=all 2>&1 | grep -E "gdn|linear-backend"'

# 3. b12x loader allow-list
docker run --rm --entrypoint bash vllm-node-b12x:latest -c '
  grep -n "Flash Next only" -B 8 \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/b12x_loader.py'

# 4. B12X page sizes vs the hybrid aligner
docker run --rm --entrypoint bash vllm-node-b12x:latest -c '
  grep -n "_B12X_SUPPORTED_PAGE_SIZES" \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/b12x.py
  grep -n "_align_hybrid_block_size" -A 20 \
    /usr/local/lib/python3.12/dist-packages/vllm/platforms/interface.py'
```

## 9. Related code

- `mods/b12x-reclaim-gate/` — v4 gate, discovery, second sweep site
- `mods/b12x-kernel-cache/lib/key.sh` — fail-closed identity fields, v2 schema
- `docker/patch_b12x_cache_integrity.py` — layout-aware root discovery
- `recipes/qwen3.8-27b-coder390-{2x,4x,2x-store-l2}.yaml`
- `recipes/qwen3.8-27b-nvfp4-dflash2.yaml` — the same-family precedent
- `tests/test_b12x_reclaim_gate_mod.sh`, `tests/test_b12x_kernel_cache.sh`,
  `tests/test_b12x_cache_integrity_patch.py`
