"""Static gates for the store connector's per-group addressing and drafter exclusion.

Torch-free, per the house rule (gates.py and these tests stay importable on host
and container). The behaviour under guard is grounded in the live stack probe of
2026-10-25 (see gates.py's MD5_MOD_WORKER note):

  * ONE backing storage, 39,919,042,560 bytes, 5 physical layer planes;
  * 15 overlay groups x 5 layers, layer_stride = 4731 blocks x 1,687,552 page;
  * target layers `language_model.model.layers.0..63`, drafter `model.layers.64..68`;
  * flat hf_config.num_hidden_layers is None (nested under text_config).

The two regressions these guard are both silent-corruption classes: a
single-segment collapse that stores only 1 of 5 layer slots, and a store load
that sweeps the speculative drafter's KV.
"""

from __future__ import annotations

from pathlib import Path

from mooncake_worker import gates

WORKER_SRC = (gates.vendor_dir() / "worker.py").read_text()


def test_per_group_segment_addressing_present() -> None:
    """Registration must derive per-group segments from KVCacheTensor metadata."""
    assert "def _register_group_segments()" in WORKER_SRC
    assert "tensor.block_stride" in WORKER_SRC
    assert "tensor.layer_stride" in WORKER_SRC
    # The per-group list must be installed per DB, not one global list for all.
    assert "db.set_kv_caches_base_addr(g_addrs)" in WORKER_SRC
    assert "db.set_block_len(g_blens)" in WORKER_SRC


def test_legacy_single_segment_only_as_fallback() -> None:
    """The region//num_blocks collapse may survive only behind `not _per_group`."""
    assert "if not _per_group:" in WORKER_SRC
    # The global-list assignment must not run unconditionally.
    assert WORKER_SRC.count("db.set_kv_caches_base_addr(addrs)") == 1
    assert "if not _per_group:\n            # Per-group addressing" in WORKER_SRC


def test_drafter_group_excluded_from_all_io_paths() -> None:
    """Every put/get/lookup loop must skip the excluded drafter groups.

    Four paths: the partial-tail save, the block-aligned save, the load, and
    the prefix lookup.
    """
    assert "def _drafter_group_ids(" in WORKER_SRC
    # Registration, save, load and lookup all consult the exclusion set.
    assert WORKER_SRC.count("self._excluded_group_ids") >= 6
    assert WORKER_SRC.count("if g_idx in self._excluded_group_ids:") == 4


def test_drafter_detection_handles_nested_text_config() -> None:
    """num_hidden_layers is None on Qwen3.5 VL checkpoints; text_config carries it."""
    assert "def _target_num_hidden_layers(" in WORKER_SRC
    assert 'getattr(text_cfg, "num_hidden_layers", None)' in WORKER_SRC


def test_drafter_detection_is_fail_safe() -> None:
    """A detection failure must keep every group (never drop a target layer)."""
    idx = WORKER_SRC.index("def _drafter_group_ids(")
    body = WORKER_SRC[idx : WORKER_SRC.index("    def register_cross_layers_kv_caches(", idx)]
    assert "return set()" in body
    assert "except Exception:" in body


def test_obsolete_layer_name_detector_removed() -> None:
    """The name-based detector was a no-op here (num_hidden_layers None)."""
    assert "_drafter_layer_names" not in WORKER_SRC
    assert "MOONCAKE_STORE_LAYOUT_PROBE" not in WORKER_SRC


COORD_SRC = (gates.vendor_dir() / "coordinator.py").read_text()


def test_coordinator_excludes_groups_from_hit_convergence() -> None:
    """Excluded groups must not participate in prefix-hit convergence.

    They never hold stored keys, so including them makes every lookup converge
    to zero — measured live as 686,878 external queries with 0 hits.
    """
    assert "excluded_group_ids" in COORD_SRC
    assert "self.excluded_group_ids = frozenset(excluded_group_ids)" in COORD_SRC
    # The skip must precede spec/manager resolution in _verify_and_split.
    idx = COORD_SRC.index("def _verify_and_split_kv_cache_groups")
    body = COORD_SRC[idx : COORD_SRC.index("def find_longest_cache_hit", idx)]
    assert "if i in self.excluded_group_ids:" in body
    assert body.index("if i in self.excluded_group_ids:") < body.index(
        "spec = _unwrap_spec(g.kv_cache_spec)"
    )


def test_worker_resolves_exclusion_before_coordinator() -> None:
    """The set must exist before the coordinator is constructed, or it goes stale."""
    i_excl = WORKER_SRC.index("self._excluded_group_ids = self._drafter_group_ids()")
    i_coord = WORKER_SRC.index("self.coord = MooncakeStoreCoordinator(")
    i_kwarg = WORKER_SRC.index("excluded_group_ids=self._excluded_group_ids,")
    assert i_excl < i_coord < i_kwarg
