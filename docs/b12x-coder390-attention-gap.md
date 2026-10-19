# Qwen3.8-27B-Coder390 on `vllm-node-b12x`: the B12X attention gap

Status: **open**. The `qwen3.8-27b-coder390-{2x,4x}` recipes serve again, but
without B12X paged attention. This document records what broke, what was
fixed, and the paths to restore the measured B12X win.

To test the arm end to end, follow
[`b12x-attention-arm-testing-runbook.md`](./b12x-attention-arm-testing-runbook.md)
— unit tests, patcher-vs-image anchoring, the two gates, and the live boot with
its acceptance criteria and failure triage.

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

### Root cause: the storage size is validated where a kernel page size belongs

The conflict is **not** a kernel constraint. vLLM already implements virtual
block splitting: `prepare_kernel_block_sizes` (`v1/worker/utils.py:630`) asks
`select_common_block_size` for a kernel page that divides the manager block, and
the resulting `kernel_block_size` is threaded into the metadata builder with
block-table remapping (`AttentionGroup.create_metadata_builders`,
`map_to_kernel_block_table`). Measured directly against the image:

```
select_common_block_size(1664, [B12xPagedAttentionBackend]) -> 128   # 13 kernel pages
```

So storage-1664 was always meant to be consumed as 128-token kernel pages, and
every `page_size not in (64, 128)` gate in the flashinfer kernel layer sees 128,
never 1664. That is why the 13 enumerated gates are *not* the blocker.

Two call sites validate the **storage** size against the **kernel page** set:

1. `Platform._check_aligned_block_size` (`platforms/interface.py:797`) calls
   `B12xPagedAttentionBackend.supports_block_size(1664)` → `False` → the
   `Aligned KV cache block size 1664 is not supported by attention backend(s)`
   error.
2. `B12xPagedAttentionImpl.__init__` (`v1/attention/backends/b12x.py:399`) does
   `default_block_size = int(cache_config.block_size)` and raises if it is not in
   `(64, 128)` — again comparing a storage size to the kernel page set.

Both treat `_B12X_SUPPORTED_PAGE_SIZES` as if it described storage blocks. It
describes kernel pages. The fix is therefore plumbing, not widening.

Gate 2 was confirmed **reachable and live**, not just read. Constructing
`B12xPagedAttentionImpl` with a config whose `cache_config.block_size` is 1664
(exactly what the aligner leaves after the gate-1 fix) raises:

```
cache_config.block_size = 1664
IMPL RAISED ValueError: b12x requires --block-size in (64, 128), got 1664.
```

Ordering matters: gate 1 runs inside
`Platform.update_block_size_for_backend` (`multiproc_executor.py:701`), which is
called *before* any layer is constructed, so gate 1 fires first on an unpatched
build and hides gate 2. Both must be fixed together.

With the corrected predicate applied to both, verified in-image:

```
storage= 1664 gate1=True  kernel=128   gate3=ok
storage=   64 gate1=True  kernel= 64   gate3=ok
storage=  128 gate1=True  kernel=128   gate3=ok
storage=  256 gate1=True  kernel=128   gate3=ok
storage= 2688 gate1=True  kernel=128   gate3=ok
storage= 1648 gate1=False kernel=None  gate3=RAISE   <- still fails closed
```

### Completeness checks for the 5a patch

Three things could have made the two-site fix insufficient; each was checked
against the shipped image.

**No other consumer of the storage size.** `grep 'cache_config.block_size'` in
`b12x.py` returns exactly two hits (lines 398 and 403, both inside the patched
guard). The flashinfer kernel layer never reads it — its only `block_size` hits
are `query_block_size` in `reference.py`, an unrelated knob. So nothing
downstream re-applies the kernel-page set to the storage size.

**Splitting is not silently disabled by packing.** `prepare_kernel_block_sizes`
has a bypass: if any tensor in the group is block-outermost packed
(`block_stride > layer_stride`) it keeps the manager block and never splits.
Splitting also hard-requires dense, unpadded pages
(`compute_layer_kv_cache_shape_bytes`) and names `LBNHC` as the layout that
satisfies it. The live boot reports `Using LBNHC KV cache layout`, i.e. the
layer-compact path, so the bypass is not taken for this model.

**`--installed` is reachable.** `importlib.util.find_spec("vllm")` resolves to
`/usr/local/lib/python3.12/dist-packages/vllm`, and
`v1/attention/backends/b12x.py` exists there, so the wheel-stage patcher does
not fall through to its "absent; skipping" branch.

**The split really reaches the kernel's page geometry.** The gap between
"`select_common_block_size` returns 128" and the kernel layer's own
`_kv_page_size` check (`b12x.py:92`, `key_page_size in (64, 128)`) is closed by
`compute_layer_kv_cache_shape_bytes`, which uses `spec.block_size // kernel_block_size`
pages per manager block. Measured in-image with the patched backend:

```
backend: vllm.v1.attention.backends.b12x.B12xPagedAttentionBackend
selected kernel block: 128
block_stride_alignment: MultipleOf(128)
page dim with storage block: 1664
page dim with kernel block : 128
kernel accepts page dim: True
```

So the KV tensors the kernels index are 128-token pages, not 1664, and the 13
`page_size not in (64, 128)` gates in the flashinfer layer are satisfied without
edits.

**The fix is not tuned to 1664.** `_align_hybrid_block_size` derives
`kernel_block_alignment_size` from the backend's smallest kernel page
(`max(min(sizes), block_size)` = 64 for B12X) and `backend_block_alignment` as
an `lcm` of those minimums, so every B12X-aligned storage size is a multiple of
64. Since 64 is itself a supported kernel page, *every* such size is splittable:

```
  storage=   64 -> kernel=64
  storage=  128 -> kernel=128
  storage= 1600 -> kernel=64
  storage= 1664 -> kernel=128
  storage= 1728 -> kernel=64
non-splittable multiples of 64: NONE
```

Only a size with no supported divisor keeps failing closed, which is what the
FlashInfer quantum-16 path produces (1648 = 16 x 103, 103 prime). That is the
desired asymmetry: the B12X arm becomes bootable, the misaligned FlashInfer
alignment keeps its honest error.

### Why the aligned size is not stable at 1664

`_align_hybrid_block_size` rounds the result to the backend's alignment quantum
(`kernel_block_alignment_size`, `platforms/interface.py:127`), which is 16 for
FlashInfer and 64 for B12X. The live FlashInfer boot logs:

```
[interface.py:1105] Setting attention block size to 1648 tokens
[kv_cache_utils.py:887] kv lcm block sizes 1648
```

1648 = 16 × 103, and 103 is prime, so 1648 is **not** divisible by 64 or 128.
The aligned size is a function of the chosen backend, not a fixed property of the
model. This matters for any fix: only the B12X quantum (64) yields a
128-splittable storage size.

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

### Live boot with the 5a patch (defect 5 verified end to end)

The 5a patch was exercised on a real 2-node TP=2 launch rather than only in
unit tests: the patcher was applied inside the running container by a
verification mod, and the recipe was driven with
`--attention-backend B12X --block-size 64`. Engine log, node1 rank 0:

```
[b12x-storage-block-split] supports_block_size(1664)= True
[b12x.py:735] Using b12x with q_heads=12 kv_heads=2 head_dim_qk=256
              head_dim_vo=256 window_left=-1 planned_page_sizes=(64, 128)
[interface.py:1105] Setting attention block size to 1664 tokens to ensure
                    that attention page size is >= mamba page size.
[kv_cache_utils.py:887] kv lcm block sizes 1664
[kv_cache_utils.py:1205] GPU KV cache size: 3,056,670 tokens,
                         Maximum concurrency for 262,144 tokens per request: 11.66x
```

This is the path unit tests could not reach: `_find_non_ssm_backends` resolved
B12X from real model layers, the planner built page sizes `(64, 128)`, the
aligned storage block stayed 1664, and the KV manager split it. Confirmed
further:

- CUDA-graph capture completed (`Capturing CUDA graphs (FULL): 100%`) with
  **zero** `device-side` / `AcceleratorError` / `IllegalInstruction` hits, so
  the remapped kernel block table replays cleanly.
- The endpoint served a real chat completion (HTTP 200), so the arm is not just
  bootable but functional.

Both ranks applied the patch at boot (`Accepted splittable hybrid storage
blocks in B12X` on 10.100.171.3 and .4).

Not yet measured: decode throughput against `caaba0f`'s 32.2 vs 26.4 tok/s.
The image itself was not rebuilt for this run — the patcher was injected
in-container — so the Dockerfile wiring is still unexercised by a real build.

### The same verification on the production nodes (with a KV-pool caveat)

A second run, on gx10-cb11/gx10-f1d8 (Rail B `.3` and `.4`) at the recipe's
GMU 0.75, reproduced the same acceptance evidence with a healthy KV pool:

```
Available KV cache memory: 70.51 GiB
GPU KV cache size: 3,190,974 tokens, Maximum concurrency for 262,144 tokens per request: 12.17x
Capturing CUDA graphs (FULL): 100%
CUDA errors (device-side / AcceleratorError / IllegalInstruction): 0
endpoint: HTTP 200 chat completion
```

An earlier attempt on the same shape at **GMU 0.25** failed *after* the B12X
gates passed — worth recording so it is not mistaken for a patch defect:

```
Setting attention block size to 1664 tokens          <- gate 1 + gate 2 passed
Available KV cache memory: -28.74 GiB
ValueError: No available memory for the cache blocks.
  (kv_cache_planning.py:56 _check_enough_kv_cache_memory)
```

Two lessons:

- `gpu_memory_utilization` sets the budget the engine must fit weights +
  activations + CUDA-graph profiling **and** KV blocks into. Measured on this
  model: GMU 0.25 left `-28.74 GiB` for KV; GMU 0.75 left `+70.51 GiB`. Those
  two runs were on different nodes with different co-tenant load, so a single
  "non-KV footprint" figure should not be inferred from them — read the
  `Available KV cache memory` line on the node you actually intend to use
  rather than computing a value.
- Defect 5's acceptance evidence (1664 accepted, planner `(64, 128)`, graph
  capture 100%, zero CUDA errors) is independent of the KV-pool outcome — it
  appeared *before* the cache-sizing failure on the starved boot.

### Rebuilding this image: two traps that wasted a day

The 5a patch is committed and proven, but shipping it in an image failed twice.
Both failures are orthogonal to the patch; record them so the next rebuild does
not repeat them.

**Trap 1 — the NCCL deb was mounted but never installed.** The final stage had
`--mount=type=bind,from=base,source=/workspace/vllm/nccl/build/pkg/deb`, but the
`apt install` line never referenced it, so the CUDA base image's *held*
libnccl 2.28.3 survived. NCCL 2.28.3 predates `ncclCommResume`, which torch 2.13
links against, so every start died at import:

```
ImportError: libtorch_cuda.so: undefined symbol: ncclCommResume
```

Fixed by installing `/workspace/nccl-pkg/*.deb` in the same RUN that mounts it
(`--allow-downgrades --allow-change-held-packages`, then `apt-mark hold`) and
pinioning the source build to `v2.32.3-1` instead of unpinned `master`. Verify
with `dpkg -l | grep libnccl2` (expect 2.32.3) and `python3 -c "import torch"`.

**Trap 2 — `dev/karmic-kraken` HEAD needs an unreleased InstantTensor.**
Building the `--exp-b12x` profile clones `local-inference-lab/vllm` at that
branch, whose tip is now `8244d1c44` (vLLM `0.1.dev21830+g8244d1c44`). Its weight
loader gates on an InstantTensor capability introduced by `de475def9`
("Bound InstantTensor staging…", 2026-09-22):

```
RuntimeError: Installed InstantTensor does not expose the metadata layout
needed for index-aware loading (missing: _determine_buffer_size)
  (weight_utils.py:1679, required_attrs list)
```

No published instanttensor exposes `_determine_buffer_size` — 0.2.2 is the
newest on PyPI. So the branch head is unbuildable with any released dependency.

**Recommended rebuild ref: `50cbc118b`** — the direct parent of `de475def9`,
verified to contain **zero** references to `_determine_buffer_size` (the gate
commit has 2). Use:

```bash
./build-and-copy.sh --exp-b12x --rebuild-vllm --b12x-vllm-ref 50cbc118b \
  -t vllm-node-b12x --copy-to 10.100.171.4
```

Note `--exp-b12x` rejects `--use-wheels` and `--vllm-ref` (use
`--b12x-vllm-ref`); a cached-wheel run skips the recompile that bakes the patch
in. Pin `--b12x-vllm-ref` to a commit, never to the moving branch, or the build
silently changes vLLM underneath the recipe.

### Cost

Commit `caaba0f` measured this arm on gx10-node1 TP=2 (C1/60k probe, 8
sequential prompts, temp 0, `ignore_eos`):

- per-stream decode **32.2 vs 26.4 tok/s (+22.1%)**
- mean TPOT **31.1 vs 37.9 ms (−18.0%)**
- accept length 3.04 vs 3.19 (kernel-driven, not acceptance-driven)
- cold-60k TTFT 66.6 vs 67.5 s (−1.3%)
- KV pool 3,346,412 tokens (−1.1%, within the 10% gate)

That measurement predates the image these recipes now pin, so it is not a
baseline this configuration ever ran at. `caaba0f` landed 2026-10-17 and its
recipe still passed `--gdn-prefill-backend b12x`, which does not exist in the
pinned image (built 2026-10-16); the numbers came from an earlier image whose
`b12x` loader still accepted `Qwen3_5ForConditionalGeneration`. Defects 3 and 4
are exactly that drift, surfaced by re-running against the current image. So the
figures are a real measurement of the B12X arm on a *previous* image, not of a
configuration that ever booted on the pinned one — which is why section 7 ranks
restoring the arm as future work rather than treating its loss as a regression
from a working baseline. All of it is currently unrealized either way: no
configuration on the pinned image can boot this arm yet (defect 5).

## 7. Resolution paths for defect 5

Ranked by preference. Each needs an image rebuild or a new mod, so none is a
recipe-only change.

### 5a. Let the two storage-facing validators accept a splittable block (preferred)

Not a widening of the kernel page set. Two sites compare a *storage* block size
against the *kernel page* set; both should instead accept storage sizes that a
supported kernel page divides:

- `B12xPagedAttentionBackend.supports_block_size` — used by
  `_check_aligned_block_size`.
- `B12xPagedAttentionImpl.__init__` (`b12x.py:399`) — validates
  `cache_config.block_size` instead of the kernel block size.

Leave `get_supported_kernel_block_sizes()` returning the exact `[64, 128]`. This
matters: declaring `MultipleOf(64)` is actively wrong, because
`_block_size_is_supported` then reports the *storage* size as supported,
`select_common_block_size` short-circuits and returns 1664, splitting is
disabled, and the kernels then get a 1664-token page. Measured in the image:

```
current exact [64,128]: select_common_block_size(1664) -> 128   (split, correct)
MultipleOf(64)        : select_common_block_size(1664) -> 1664  (no split, breaks)
```

With the corrected predicate (`bs in (64,128) or any(bs % p == 0 for p in
(64,128))`) the behaviour is:

```
storage=1664 supported=True kernel=128 split=True
storage=  64 supported=True kernel= 64 split=False
storage= 128 supported=True kernel=128 split=False
storage= 256 supported=True kernel=128 split=True
storage=2688 supported=True kernel=128 split=True
storage=1648 supported=False  -> still fails closed (not divisible by 64/128)
```

Note 1648 (the FlashInfer-aligned size) still fails, which is correct — it
genuinely cannot be split. The fix only unblocks B12X's own quantum-64 result.

Cost: the backend precompiles decode/verify/extend plans per page size, so this
adds no new page sizes and no extra warmup. All `== 64` / `== 128` fast paths
keep matching, because the kernel page stays 128.

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
