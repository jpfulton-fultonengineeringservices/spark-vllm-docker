"""Install a fake vllm package tree and alias the vendored store modules
into it so that ``import vllm.distributed.kv_transfer.kv_connector.v1.mooncake
.store.worker`` (and friends) resolves without a real vLLM install.

Must be called exactly once before importing any vendored store module.
Idempotent guard: raises ``AssertionError`` if the stub is already installed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# BlockHash: subclass of bytes so .hex() works and it's hashable.
# ---------------------------------------------------------------------------

BlockHash = bytes  # alias: the coordinator's ExternalCachedBlockPool calls
                   # bytes(block_hash); the scheduler's db.key_for calls
                   # chunk_hash.hex().  bytes satisfies both.


def _make_module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _make_package(name: str, **subattrs: Any) -> types.ModuleType:
    mod = _make_module(name, **subattrs)
    mod.__path__ = []  # mark as package
    return mod


def install_vendor_stub() -> None:
    """Register fake vllm modules + alias vendored store files.

    Call this at the top of test_connector_behavior.py before any
    ``from vllm...store.worker import ...`` line.
    """
    assert "vllm" not in sys.modules, "stub already installed or real vllm present"

    # The vendored files are aliased into the fake vllm tree via exec_module;
    # suppress .pyc emission so they don't pollute the vendor dir and trip the
    # gate's PACKAGE_FILES completeness check (test_gates).
    sys.dont_write_bytecode = True

    # --- vllm package + sub-modules --------------------------------------
    _make_package("vllm")
    _make_module("vllm.logger", init_logger=lambda _: types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    ))
    _make_module("vllm.envs",
        VLLM_MOONCAKE_STORE_TIER_LOG=False,
        VLLM_PREFIX_CACHE_RETENTION_INTERVAL=None,
        VLLM_MOONCAKE_LOAD_RECV_THREADS=4,
        VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO=0.5,
        VLLM_RPC_BASE_PATH="/tmp",
    )
    _make_module("vllm.config", VllmConfig=type("VllmConfig", (), {}))
    _make_module("vllm.utils.math_utils", cdiv=lambda a, b: -(-a // b))
    from unittest.mock import MagicMock as _MagicMock
    _make_module("vllm.utils.network_utils",
        get_ip=lambda: "127.0.0.1",
        make_zmq_socket=lambda *a, **k: _MagicMock(),
    )
    _make_module("vllm.model_executor.models.utils",
        extract_layer_index=lambda x: 0,
    )
    _make_module("vllm.forward_context", ForwardContext=type("ForwardContext", (), {}))
    _make_module("vllm.v1.attention.backend",
        AttentionMetadata=type("AttentionMetadata", (), {}),
    )
    _make_module("vllm.v1.attention.backends.utils", NULL_BLOCK_ID=0)
    _make_module("vllm.v1.outputs",
        KVConnectorOutput=type("KVConnectorOutput", (), {}),
    )
    _make_module("vllm.v1.request",
        Request=type("Request", (), {}),
    )

    # --- distributed ----------------------------------------------------
    _make_package("vllm.distributed",
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 2,
        get_pcp_group=lambda: types.SimpleNamespace(world_size=1, rank_in_group=0),
        get_dcp_group=lambda: types.SimpleNamespace(world_size=1, rank_in_group=0),
    )
    _make_module("vllm.distributed.kv_events",
        BlockStored=type("BlockStored", (), {"__init__": lambda self, **kw: None}),
        KVCacheEvent=type("KVCacheEvent", (), {}),
        KVConnectorKVEvents=type("KVConnectorKVEvents", (), {}),
        KVEventAggregator=type("KVEventAggregator", (), {"__init__": lambda self, n: None}),
    )

    # --- kv_connector.v1.base / .metrics ---------------------------------
    _make_package("vllm.distributed.kv_transfer")
    _make_package("vllm.distributed.kv_transfer.kv_connector")
    _make_package("vllm.distributed.kv_transfer.kv_connector.v1",
        SupportsHMA=type("SupportsHMA", (), {}),
    )
    _make_module("vllm.distributed.kv_transfer.kv_connector.v1.base",
        KVConnectorBase_V1=type("KVConnectorBase_V1", (), {"__init__": lambda self, **kw: None}),
        KVConnectorMetadata=type("KVConnectorMetadata", (), {}),
        KVConnectorWorkerMetadata=type("KVConnectorWorkerMetadata", (), {}),
        KVConnectorRole=type("KVConnectorRole", (), {"SCHEDULER": "scheduler", "WORKER": "worker"}),
        SupportsHMA=type("SupportsHMA", (), {}),
        KVConnectorTransferResults=type("KVConnectorTransferResults", (),
            {"__init__": lambda self, **kw: self.__dict__.update(kw)}),
    )
    _make_module("vllm.distributed.kv_transfer.kv_connector.v1.metrics",
        KVConnectorPromMetrics=type("KVConnectorPromMetrics", (), {}),
        KVConnectorStats=type("KVConnectorStats", (), {"__init__": lambda self, **kw: None}),
        PromMetric=type("PromMetric", (), {}),
        PromMetricT=type("PromMetricT", (), {}),
    )

    # --- mooncake package + rdma_utils ----------------------------------
    _make_package("vllm.distributed.kv_transfer.kv_connector.v1.mooncake",
        rdma_utils=_make_module(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.rdma_utils",
            get_configured_preferred_segment=lambda extra_cfg: None,
            get_requester_local_hostname=lambda ip: "127.0.0.1",
        ),
    )

    # --- v1.core / v1.kv_cache_interface --------------------------------
    class _KVCacheBlock:
        def __init__(self, block_id: int = 0):
            self.block_id = block_id
        def __eq__(self, o: object) -> bool:
            return isinstance(o, _KVCacheBlock) and o.block_id == self.block_id
        def __hash__(self) -> int:
            return hash(self.block_id)

    from collections import namedtuple as _namedtuple
    # Coordinator unpacks SpecGroup as a 4-tuple: (spec, group_ids, manager_cls, use_eagle).
    _SpecGroup = _namedtuple("_SpecGroup", ["spec", "group_ids", "manager_cls", "use_eagle"])

    _make_module("vllm.v1.core.block_pool",
        BlockPool=type("BlockPool", (), {}),
    )
    _make_module("vllm.v1.core.kv_cache_coordinator", SpecGroup=_SpecGroup)
    _make_module("vllm.v1.core.kv_cache_utils",
        BlockHash=BlockHash,
        BlockHashListWithBlockSize=type(
            "BlockHashListWithBlockSize", (), {"__init__": lambda self, *a, **k: None}
        ),
        KVCacheBlock=_KVCacheBlock,
        resolve_kv_cache_block_sizes=lambda kv_cache_config, vllm_config: (16, 16),
        maybe_convert_block_hash=lambda h: h,
    )
    _make_module("vllm.v1.core.kv_cache_manager",
        KVCacheBlocks=type("KVCacheBlocks", (), {}),
    )
    _make_module("vllm.v1.core.sched.output",
        NewRequestData=type("NewRequestData", (), {}),
        SchedulerOutput=type("SchedulerOutput", (), {}),
        CachedRequestData=type("CachedRequestData", (), {}),
    )

    class _KVCacheGroupSpec:
        def __init__(self, kv_cache_spec: Any = None, is_eagle_group: bool = False):
            self.kv_cache_spec = kv_cache_spec
            self.is_eagle_group = is_eagle_group

    class _MambaSpec:
        def __init__(self, block_size: int = 16, mamba_cache_mode: str = "align",
                     num_heads: int = 1, head_dim: int = 1):
            self.block_size = block_size
            self.mamba_cache_mode = mamba_cache_mode
            self.num_heads = num_heads
            self.head_dim = head_dim

    class _FullAttentionSpec:
        def __init__(self, block_size: int = 16, num_heads: int = 1, head_dim: int = 1):
            self.block_size = block_size
            self.num_heads = num_heads
            self.head_dim = head_dim

    class _UniformTypeKVCacheSpecs:
        def __init__(self, **kw: Any):
            self.kv_cache_specs = kw

    _make_module("vllm.v1.kv_cache_interface",
        KVCacheConfig=type("KVCacheConfig", (), {"__init__": lambda self, **kw: None}),
        KVCacheGroupSpec=_KVCacheGroupSpec,
        KVCacheSpec=type("KVCacheSpec", (), {}),
        MambaSpec=_MambaSpec,
        FullAttentionSpec=_FullAttentionSpec,
        MLAAttentionSpec=type("MLAAttentionSpec", (), {}),
        SlidingWindowMLASpec=type("SlidingWindowMLASpec", (), {}),
        UniformTypeKVCacheSpecs=_UniformTypeKVCacheSpecs,
        CrossAttentionSpec=type("CrossAttentionSpec", (), {}),
    )
    _make_module("vllm.v1.kv_cache_spec_registry",
        KVCacheSpecRegistry=type("KVCacheSpecRegistry", (), {
            "get_manager_class": staticmethod(lambda spec: _make_manager_cls(spec)),
        }),
    )

    # --- mooncake.store namespace: alias the vendored files --------------
    _make_package("vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store")
    store_pkg = sys.modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store"
    ]
    store_pkg.__path__ = []  # ensure it looks like a package for relative imports

    vendor_dir = (
        Path(__file__).resolve().parents[1] / "src" / "mooncake_worker" / "vendor" / "store"
    )
    alias_order = ["protocol", "data", "metrics", "coordinator", "worker", "scheduler", "connector"]
    for fname in alias_order:
        fpath = vendor_dir / f"{fname}.py"
        spec = importlib.util.spec_from_file_location(
            f"vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.{fname}",
            str(fpath),
        )
        assert spec is not None and spec.loader is not None, f"spec for {fname}"
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store"
        sys.modules[mod.__name__] = mod
        spec.loader.exec_module(mod)
        setattr(store_pkg, fname, mod)


def _make_manager_cls(spec: Any) -> type:
    """Return a manager class whose ``find_longest_cache_hit`` walks the
    ExternalCachedBlockPool over the group's spec.  Used by the coordinator's
    single-group fast-path and by the per-group loop in _find_hit_blocks."""

    class _Manager:
        supports_fine_grained_hash_lookup = False

        @staticmethod
        def find_longest_cache_hit(
            block_hashes,
            max_length: int,
            kv_cache_group_ids: list[int],
            block_pool: Any,
            kv_cache_spec: Any,
            drop_eagle_block: bool = False,
            alignment_tokens: int = 16,
        ) -> tuple[list, int]:
            # hit_blocks[g] is a list of KVCacheBlock, one per hash-chunk
            # (matching the upstream find_longest_cache_hit return shape).
            # The coordinator iterates: "for blk in blocks" where blocks is
            # the per-group list of KVCacheBlock.
            hash_block_size = 16
            num_hash_blocks = max(0, max_length // hash_block_size)
            per_group_blocks: list[list] = []
            hit_len = 0
            for gid in kv_cache_group_ids:
                g_blocks: list = []
                for i in range(num_hash_blocks):
                    blocks = block_pool.get_cached_block(block_hashes[i], [gid])
                    if blocks is not None:
                        g_blocks.append(block_pool._present_block)
                    else:
                        g_blocks.append(block_pool.null_block)
                per_group_blocks.append(g_blocks)
            # Hit length = longest prefix where every group has present blocks.
            for i in range(num_hash_blocks):
                if all(g[i] is not block_pool.null_block for g in per_group_blocks):
                    hit_len = (i + 1) * hash_block_size
                else:
                    break
            if drop_eagle_block and hit_len > 0:
                hit_len = max(0, hit_len - hash_block_size)
                # Truncate to the eagle-pruned length.
                num_keep = hit_len // hash_block_size
                for g in per_group_blocks:
                    g[:] = g[:num_keep]
            return per_group_blocks, hit_len

        @staticmethod
        def reachable_block_mask(
            block_hashes, max_length: int, *, retention_interval: int | None = None,
            num_prompt_tokens: int | None = None,
        ) -> list[bool]:
            return [True] * (max_length // 16)

    return _Manager
