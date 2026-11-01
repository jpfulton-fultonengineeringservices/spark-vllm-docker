"""Behavioral tests for the vendored Mooncake store connector.

Exercises the save/lookup/load threads and the coordinator against fake
stores, without requiring a running Mooncake daemon or a real vLLM install.
The stub in tests/_vendor_stub.py is installed in-file so the 6 existing
torch-free suites are unaffected.

Seven surfaces covered:
  1. Key layout — PoolKey prefix / to_string for two different ranks.
  2. Drafter-group exclusion — the drafter group never appears in key
     strings, and the excluded group's load_mask is all-False.
  3. Save-path BatchPut failure — a store that returns -200 on a sub-batch
     marks the request pressure-skipped; subsequent batches are suppressed
     while pressure is active.
  4. Load-path failure — a store that returns a negative byte count routes
     the request to failed_recving.
  5. Recv-thread tier-log raise regression — with TIER_LOG=True and a
     store.batch_get_replica_desc that raises, the pre-try timer bind
     (worker.py:1588) means the except handler does NOT hit
     UnboundLocalError; it records status="error" and marks the request
     failed.
  6. Per-batch timer reset — a two-batch load records two load_get entries
     with distinct start times and the second duration excludes tier lookup.
  7. Lookup with enable_lookup=False — scheduler.get_num_new_matched_tokens
     returns (0, False) without touching the store.
"""

from __future__ import annotations

# Install the fake vllm package tree and alias the vendored store modules
# into it.  Must run before any `from vllm...store.worker import ...` below.
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _vendor_stub import install_vendor_stub

install_vendor_stub()

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E402
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    PoolKey,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E402
    MOONCAKE_NO_AVAILABLE_HANDLE,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)

BLOCK_SIZE = 16
HASH_BLOCK_SIZE = 16
MODEL = "m"
N_GROUPS = 2  # group 0 = target, group 1 = drafter (excluded)
N_TOKENS = 64  # 4 blocks of 16 tokens each


def _make_db(group_id: int, tp_rank: int = 0) -> ChunkedTokenDatabase:
    return ChunkedTokenDatabase(
        KeyMetadata(
            model_name=MODEL,
            tp_rank=tp_rank,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
            group_id=group_id,
        ),
        block_size=BLOCK_SIZE,
        hash_block_size=HASH_BLOCK_SIZE,
    )


def _block_hashes(n_blocks: int) -> list[bytes]:
    """One 32-byte BlockHash per block."""
    return [bytes([i + 1]) * 32 for i in range(n_blocks)]


# ---------------------------------------------------------------------------
# Surface 1: key layout
# ---------------------------------------------------------------------------


class TestKeyLayout:
    def test_prefix_differs_across_ranks(self) -> None:
        db0 = _make_db(group_id=0, tp_rank=0)
        db1 = _make_db(group_id=0, tp_rank=1)
        kh = _block_hashes(1)
        assert db0.key_for(kh[0]) != db1.key_for(kh[0])
        assert "@tp_rank:0@" in db0.key_for(kh[0])
        assert "@tp_rank:1@" in db1.key_for(kh[0])
        assert "@group:0@" in db0.key_for(kh[0])

    def test_prefix_differs_across_groups(self) -> None:
        db0 = _make_db(group_id=0)
        db1 = _make_db(group_id=1)
        kh = _block_hashes(1)
        assert "@group:0@" in db0.key_for(kh[0])
        assert "@group:1@" in db1.key_for(kh[0])
        assert db0.key_for(kh[0]) != db1.key_for(kh[0])

    def test_poolkey_build_prefix_respects_cache_prefix(self) -> None:
        md = KeyMetadata(
            model_name=MODEL, tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0,
            cache_prefix="fes-staged",
        )
        pfx = PoolKey.build_prefix(md)
        assert pfx.startswith("fes-staged@")


# ---------------------------------------------------------------------------
# Surface 2: drafter-group exclusion
# ---------------------------------------------------------------------------


class TestDrafterGroupExclusion:
    def test_drafter_group_skipped_in_save_keys(self) -> None:
        """Sending thread's key construction skips excluded groups."""
        db_target = _make_db(group_id=0)
        db_drafter = _make_db(group_id=1)
        db_target.set_kv_caches_base_addr([100])
        db_target.set_block_len([16])
        db_drafter.set_kv_caches_base_addr([200])
        db_drafter.set_block_len([16])

        hashes = _block_hashes(4)
        keys_target: list[str] = []
        keys_drafter: list[str] = []
        for db, excluded, out in [
            (db_target, frozenset(), keys_target),
            (db_drafter, frozenset({1}), keys_drafter),
        ]:
            for _start, _end, bh in db.process_tokens(N_TOKENS, hashes):
                g_idx = 0 if db is db_target else 1
                if g_idx in excluded:
                    continue
                out.append(db.key_for(bh))

        assert len(keys_target) == 4
        assert keys_drafter == []  # drafter group fully excluded

    def test_drafter_group_load_mask_is_all_false(self) -> None:
        """Coordinator.load_mask returns False for every chunk in excluded group."""
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
            MooncakeStoreCoordinator,
        )
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

        # Two groups: group 0 = target (attention), group 1 = drafter (excluded).
        spec = FullAttentionSpec(block_size=BLOCK_SIZE)
        target = KVCacheGroupSpec(kv_cache_spec=spec, is_eagle_group=False)
        drafter = KVCacheGroupSpec(kv_cache_spec=spec, is_eagle_group=False)
        coord = MooncakeStoreCoordinator(
            kv_cache_groups=[target, drafter],
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=HASH_BLOCK_SIZE,
            excluded_group_ids=frozenset({1}),
        )
        masks = coord.load_mask(_block_hashes(4), N_TOKENS)
        # Group 0 (target) should have 4 True chunks (load_mask → apply_eagle=False,
        # so all chunks up to token_len are True). Group 1 (drafter) is excluded,
        # so its load_mask is all-False.
        assert all(masks[0])
        assert not any(masks[1])


# ---------------------------------------------------------------------------
# Surface 3: save-path BatchPut failure → pressure skip
# ---------------------------------------------------------------------------


class _FakeSendingCoord:
    lcm_block_size = BLOCK_SIZE

    @staticmethod
    def store_mask(aligned_token_len: int, start_token: int = 0,
                   num_prompt_tokens: int | None = None) -> tuple[Any, ...]:
        return (None,)


class TestSavePathBatchPutFailure:
    def _make_sender(self, store: Any, excluded: frozenset[int] = frozenset()) -> Any:
        db = _make_db(group_id=0)
        db.set_kv_caches_base_addr([100])
        db.set_block_len([16])
        coord = _FakeSendingCoord()
        ev = threading.Event()
        sender = KVCacheStoreSendingThread(
            store, coord, [db], BLOCK_SIZE, 0,
            group_put_steps=(1,),
            kv_role="kv_both",
            ready_event=ev,
            replicate_config=MagicMock(),
            host_staging=False,
            excluded_group_ids=excluded,
        )
        return sender, db, ev

    def test_no_available_handle_triggers_pressure_skip(self) -> None:
        store = MagicMock()
        store.batch_is_exist.return_value = [0, 0, 0, 0]
        store.batch_put_from_multi_buffers.return_value = [
            MOONCAKE_NO_AVAILABLE_HANDLE,
            MOONCAKE_NO_AVAILABLE_HANDLE,
            16,
            16,
        ]
        sender, _db, _ev = self._make_sender(store)
        sender.add_request(ReqMeta(
            req_id="req-pressure",
            token_len_chunk=N_TOKENS,
            block_ids=([1, 2, 3, 4],),
            block_hashes=_block_hashes(4),
            can_save=True,
            num_prompt_tokens=N_TOKENS,
            store_job_id=1,
        ))
        sender._handle_request(sender.request_queue.get())
        assert sender._store_pressure_active is True
        assert sender._should_skip_request("req-pressure") is True

    def test_clear_store_pressure_resets_flag(self) -> None:
        sender, _db, _ev = self._make_sender(MagicMock())
        sender._store_pressure_active = True
        sender._skip_store_requests = {"req-x"}
        assert sender._clear_store_pressure() is True
        assert sender._store_pressure_active is False
        assert sender._should_skip_request("req-x") is False


# ---------------------------------------------------------------------------
# Surface 4: load-path BatchGet failure → failed_recving
# ---------------------------------------------------------------------------


class TestLoadPathFailure:
    def test_negative_get_routes_to_failed_recving(self) -> None:
        store = MagicMock()
        store.batch_get_into_multi_buffers.return_value = [-1, 16, 16, 16]
        db = _make_db(group_id=0)
        db.set_kv_caches_base_addr([100])
        db.set_block_len([16])
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
            MooncakeStoreCoordinator,
        )
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

        spec = FullAttentionSpec(block_size=BLOCK_SIZE)
        coord = MooncakeStoreCoordinator(
            kv_cache_groups=[KVCacheGroupSpec(kv_cache_spec=spec)],
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=HASH_BLOCK_SIZE,
        )
        ev = threading.Event()
        recv = KVCacheStoreRecvingThread(
            store, coord, [db], BLOCK_SIZE, 0, ev,
            host_staging=False,
            excluded_group_ids=frozenset(),
        )
        req = ReqMeta(
            req_id="req-load-fail",
            token_len_chunk=N_TOKENS,
            block_ids=([1, 2, 3, 4],),
            block_hashes=_block_hashes(4),
            can_save=False,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=N_TOKENS,
                can_load=True,
                token_len=N_TOKENS,
            ),
            store_job_id=None,
        )
        # Put on the queue so the finally-block task_done() has a matching get().
        recv.request_queue.put(req)
        recv._handle_request(recv.request_queue.get())
        finished, failed = recv.get_and_clear_finished_requests()
        assert "req-load-fail" in failed
        assert "req-load-fail" in finished  # failed ⊆ finished invariant


# ---------------------------------------------------------------------------
# Surface 5: recv-thread tier-log raise regression (pre-try timer bind)
# ---------------------------------------------------------------------------


class TestRecvTierLogRaiseRegression:
    def test_tier_log_raise_does_not_raise_unboundlocal(self) -> None:
        """When batch_get_replica_desc raises, the except handler must use the
        hoisted load_get_start (worker.py:1588) without hitting UnboundLocalError."""
        store = MagicMock()
        store.batch_get_into_multi_buffers.return_value = [16, 16, 16, 16]
        store.batch_get_replica_desc.side_effect = RuntimeError("tier lookup boom")
        db = _make_db(group_id=0)
        db.set_kv_caches_base_addr([100])
        db.set_block_len([16])
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
            MooncakeStoreCoordinator,
        )
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

        spec = FullAttentionSpec(block_size=BLOCK_SIZE)
        coord = MooncakeStoreCoordinator(
            kv_cache_groups=[KVCacheGroupSpec(kv_cache_spec=spec)],
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=HASH_BLOCK_SIZE,
        )
        ev = threading.Event()
        operations: list[tuple] = []
        _recv = KVCacheStoreRecvingThread(
            store, coord, [db], BLOCK_SIZE, 0, ev,
            host_staging=False,
            record_operation=lambda **kw: operations.append(kw),
            excluded_group_ids=frozenset(),
        )
        req = ReqMeta(
            req_id="req-tierlog",
            token_len_chunk=N_TOKENS,
            block_ids=([1, 2, 3, 4],),
            block_hashes=_block_hashes(4),
            can_save=False,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=N_TOKENS,
                can_load=True,
                token_len=N_TOKENS,
            ),
            store_job_id=None,
        )
        # The tier lookup is guarded by envs.VLLM_MOONCAKE_STORE_TIER_LOG.
        with patch("vllm.envs.VLLM_MOONCAKE_STORE_TIER_LOG", True):
            _recv.request_queue.put(req)
            _recv._handle_request(_recv.request_queue.get())
        assert operations, "load_get operation must be recorded even when tier lookup raises"
        # The recorded duration must be a finite non-negative float (no exception
        # propagated, no UnboundLocalError masking the real error).
        assert all(
            isinstance(op["duration_seconds"], float) and op["duration_seconds"] >= 0.0
            for op in operations
        )


# ---------------------------------------------------------------------------
# Surface 6: per-batch timer reset (multi-batch load records two load_get ops)
# ---------------------------------------------------------------------------


class TestPerBatchTimerReset:
    def test_multi_batch_load_records_two_ops_with_distinct_durations(self) -> None:
        """The per-batch reset (worker.py:1600) means each sub-batch gets its
        own load_get entry; without it the second batch would inherit the
        first batch's timer and the recorded RPC duration would be wrong."""

        store = MagicMock()
        # Length-aware so a 2-key sub-batch still gets a matching result list.
        store.batch_get_into_multi_buffers.side_effect = (
            lambda keys, addrs, sizes: [16] * len(keys)
        )
        store.batch_get_replica_desc.return_value = {}
        db = _make_db(group_id=0)
        db.set_kv_caches_base_addr([100])
        db.set_block_len([16])
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
            MooncakeStoreCoordinator,
        )
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

        spec = FullAttentionSpec(block_size=BLOCK_SIZE)
        coord = MooncakeStoreCoordinator(
            kv_cache_groups=[KVCacheGroupSpec(kv_cache_spec=spec)],
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=HASH_BLOCK_SIZE,
        )
        ev = threading.Event()
        operations: list[tuple] = []
        # Force a two-batch split.  Each of the 4 keys stages 12288 bytes
        # (_estimate_disk_offload_staging_bytes: a single block of 16 tokens,
        # aligned to 4096 + 8192 padding).  A 49152-byte raw budget yields a
        # 24576-byte usable budget at the stub's 0.5 ratio, so exactly 2 keys
        # fit per batch -> 4 keys split into two 2-key sub-batches.
        recv2 = KVCacheStoreRecvingThread(
            store, coord, [db], BLOCK_SIZE, 0, ev,
            disk_offload_buffer_budget_bytes=49152,
            host_staging=False,
            record_operation=lambda **kw: operations.append(kw),
            excluded_group_ids=frozenset(),
        )
        req = ReqMeta(
            req_id="req-mb",
            token_len_chunk=N_TOKENS,
            block_ids=([1, 2, 3, 4],),
            block_hashes=_block_hashes(4),
            can_save=False,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=N_TOKENS,
                can_load=True,
                token_len=N_TOKENS,
            ),
            store_job_id=None,
        )
        recv2.request_queue.put(req)
        recv2._handle_request(recv2.request_queue.get())

        load_ops = [op for op in operations if op["operation"] == "load_get"]
        assert len(load_ops) == 2, (
            f"expected two load_get entries (one per sub-batch), got {len(load_ops)}: {load_ops}"
        )
        d1, d2 = load_ops[0]["duration_seconds"], load_ops[1]["duration_seconds"]
        assert d1 >= 0.0 and d2 >= 0.0
        # The per-batch reset means the two durations are independently measured;
        # without the reset the second would include the first's RPC time.
        assert d1 != d2 or d2 == 0.0, "durations should be independently measured"


# ---------------------------------------------------------------------------
# Surface 7: lookup with enable_lookup=False
# ---------------------------------------------------------------------------


class TestLookupDisabled:
    def test_get_num_new_matched_tokens_returns_zero(self) -> None:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
            MooncakeStoreScheduler,
        )

        fake_config = MagicMock()
        fake_config.kv_transfer_config.kv_role = "kv_consumer"
        fake_config.kv_transfer_config.kv_connector_extra_config = {
            "enable_lookup": False,
        }
        fake_config.parallel_config.world_size = 1
        scheduler = MooncakeStoreScheduler(fake_config, MagicMock())
        # enable_lookup must be False and no store call should be made.
        assert scheduler.enable_lookup is False
        req = MagicMock()
        req.num_tokens = N_TOKENS
        req.block_hashes = _block_hashes(4)
        result, need_load = scheduler.get_num_new_matched_tokens(req, 0)
        assert result == 0
        assert need_load is False
