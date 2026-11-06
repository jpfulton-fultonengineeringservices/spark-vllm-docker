# InstantTensor selected-range compatibility

Adapts the index-aware loader in the coder390 image
`sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02`
to InstantTensor 0.2.2. The old vLLM helper requires the removed
`_determine_buffer_size` API and leaves the original `tensor_keep` mask behind
when compacting physical checkpoint ranges.

The mod changes only that helper's required metadata attributes and its final
selected-layout update:

- Require `tensor_keep` and `_finalize_buffer_size`.
- Rebuild the native keep mask with one `True` per compacted tensor.
- Invoke the installed `_finalize_buffer_size` with
  `instanttensor._impl.env_buffer_size()`. Unset means `None` and native
  automatic sizing. Explicit `INSTANTTENSOR_BUFFER_SIZE` keeps the prior
  helper's environment semantics; the constructor's all-shard buffer size is
  not reused after selection.

Index ownership, prefixes, priority exclusions, CPU fallback scheduling, and
coordinated process groups remain in vLLM. No public `tensor_filter` is used:
it rejects a process group and cannot identify canonical same-name overlay
shards. No InstantTensor methods or buffer arithmetic are replaced. Selection
with no GPU tensors retains the iterator's existing no-native-open path.

The patch compiles the complete result before an atomic write. Exact required
attribute/layout anchors must either both be original or both be patched;
missing, ambiguous, and partial layouts fail closed. Repeat application is
unchanged. Runtime source is discovered without importing vLLM or CUDA;
`VLLM_PACKAGE_ROOT`, `VLLM_SITE_PACKAGES`, and `PYTHON_ROOT` support fixtures.
Only recipes listing this mod activate it.

## Regression test

From the repository root:

```sh
python3 tests/test_instanttensor_selected_ranges_mod.py
bash -n mods/instanttensor-selected-ranges/run.sh
```

The CPU test executes captured, unchanged image selection/iterator functions
and actual 0.2.2 `_finalize_buffer_size`, `_open`, and typed tensor iteration.
Its native transport seam checks mask/range cardinality and reads selected
byte intervals from tiny on-disk safetensors-format files. It checks canonical
overlays, fragmented ranges, TP communicator forwarding, automatic/explicit
sizing, priority exclusions, mixed/all-CPU fallback, repeat application, and
anchor drift. The original image helper reproduces the missing-method failure.
CUDA allocation, distributed transport, and constructor setup are substituted;
this is not evidence of live GPU loading.
