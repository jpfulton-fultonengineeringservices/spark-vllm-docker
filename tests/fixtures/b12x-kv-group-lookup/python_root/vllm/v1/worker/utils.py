"""Stub of vllm/v1/worker/utils.py for the b12x-kv-group-lookup mod test.

Carries the exact allocate_kv_cache lookup block the mod anchors on; the
rest of the real function (torch allocation) is replaced by an inert
bookkeeping tail so the lookup can run without torch.
"""


class KVCacheTensor:
    def __init__(self, layers):
        self.layers = layers


class KVCacheConfig:
    def __init__(self, kv_cache_tensors, kv_cache_groups):
        self.kv_cache_tensors = kv_cache_tensors
        self.kv_cache_groups = kv_cache_groups


class KVCacheGroupSpec:
    def __init__(self, layer_names, kv_cache_spec):
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec


def allocate_kv_cache(
    kv_cache_config: "KVCacheConfig",
    device,
    layout,
    kernel_block_sizes=None,
) -> dict:
    """Allocate the KV cache and view it per layer (fragment under test)."""
    if not kv_cache_config.kv_cache_tensors:
        return {}

    kv_caches: dict[str, torch.Tensor] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        layer_name = tensor.layers[0]
        group_id, group = next(
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if layer_name in group.layer_names
        )
        spec = group.kv_cache_spec
        kv_caches.update((name, ("buf", group_id)) for name in tensor.layers)
    return kv_caches
