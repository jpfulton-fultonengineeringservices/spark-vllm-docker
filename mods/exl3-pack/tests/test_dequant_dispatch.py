"""Dispatch tests for the per-group-vs-flat FP8 scale-grid dequant.

`_dequant_fp8_block_dispatch` is the riskiest ported logic: a flat expansion of
an interleaved fused-projection scale grid "misaligns every scale row after
group 0 and silently corrupts the tensor." These tests exercise it with
synthetic tensors (no real checkpoint needed).

They need torch, so they SKIP on a torch-free host and run inside the image
(`exl3-pack selftest` invokes pytest here, or run `pytest` in-image directly).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from exl3pack.assemble import (  # noqa: E402
    _dequant_fp8_block,
    _dequant_fp8_block_dispatch,
    _dequant_fp8_block_per_group,
)

BLOCK = (4, 4)
GROUPS = 4


def _make(weight_rows: int, n: int, grid_rows: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(weight_rows, n, generator=g, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.rand(grid_rows, -(-n // BLOCK[1]), generator=g, dtype=torch.float32) + 0.5
    return weight, scale


def test_flat_grid_selects_flat_and_matches_reference() -> None:
    # m=16, block_out=4 -> flat rows = 4; rows_per_group=4 is an exact multiple
    # of block_out, so flat and per-group coincide -> flat is used.
    m, n = 16, 8
    weight, scale = _make(m, n, grid_rows=4)
    out, used_pg = _dequant_fp8_block_dispatch("t.flat", weight, scale, BLOCK, GROUPS)
    assert used_pg is False
    assert torch.equal(out, _dequant_fp8_block(weight, scale, BLOCK))


def test_interleaved_grid_selects_per_group_and_differs_from_flat() -> None:
    # m=12, groups=4 -> rows_per_group=3, ceil(3/4)*4 = 4 grid rows; the flat
    # formula gives ceil(12/4)=3. Grid of 4 rows must pick per-group.
    # scale grid rows must equal 4 (per-group), NOT 3 (flat).
    m, n = 12, 8
    weight, scale = _make(m, n, grid_rows=4)
    out, used_pg = _dequant_fp8_block_dispatch("t.interleaved", weight, scale, BLOCK, GROUPS)
    assert used_pg is True
    assert torch.equal(out, _dequant_fp8_block_per_group(weight, scale, BLOCK, GROUPS))


def test_per_group_round_trips_and_is_not_the_flat_result() -> None:
    # Build a genuinely interleaved grid where flat != per-group, prove the
    # dispatch result equals the per-group reference AND differs from flat
    # (the silent-corruption case).
    m, n, groups, bs = 12, 8, 4, 4
    rows_per_group = m // groups  # 3
    srg = -(-rows_per_group // bs)  # 1
    assert srg * groups == 4  # per-group grid rows
    g = torch.Generator().manual_seed(7)
    weight = torch.randn(m, n, generator=g).to(torch.float8_e4m3fn)
    scale = torch.rand(4, n // bs, generator=g) + 0.5
    per_group = _dequant_fp8_block_per_group(weight, scale, (bs, bs), groups)
    flat = _dequant_fp8_block(weight, scale, (bs, bs))
    out, used_pg = _dequant_fp8_block_dispatch("t.rt", weight, scale, (bs, bs), groups)
    assert used_pg is True
    assert torch.equal(out, per_group)
    assert not torch.equal(per_group, flat)  # the corruption the dispatch prevents


def test_unmatched_grid_is_a_hard_error() -> None:
    m, n = 12, 8
    weight, scale = _make(m, n, grid_rows=99)  # matches neither flat nor per-group
    with pytest.raises(SystemExit):
        _dequant_fp8_block_dispatch("t.bad", weight, scale, BLOCK, GROUPS)


def test_single_group_forces_flat() -> None:
    m, n = 12, 8
    weight, scale = _make(m, n, grid_rows=3)
    out, used_pg = _dequant_fp8_block_dispatch("t.g1", weight, scale, BLOCK, num_groups=1)
    assert used_pg is False
    assert torch.equal(out, _dequant_fp8_block(weight, scale, BLOCK))
