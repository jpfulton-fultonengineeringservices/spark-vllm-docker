"""Host tests for the torch-free shard merger."""

from __future__ import annotations

import json

import pytest

from exl3pack.merge import merge, parse_layer_spec


def _make_shard(base, layer_nums, shared_names=()) -> None:
    qt = base / "qtensors"
    qt.mkdir(parents=True, exist_ok=True)
    for n in layer_nums:
        (qt / f"model.layers.{n}.safetensors").write_bytes(f"layer-{n}".encode())
    for name in shared_names:
        (qt / name).write_bytes(f"shared-{name}".encode())


def test_merge_disjoint_ok(tmp_path) -> None:
    d0 = tmp_path / "node0"
    d1 = tmp_path / "node1"
    _make_shard(d0, [0, 1], ["model.embed_tokens.safetensors"])
    _make_shard(d1, [2, 3], ["model.mtp.layers.0.weight"])
    out = tmp_path / "out"
    receipt = merge(
        [d0, d1],
        out,
        expected_layers={0, 1, 2, 3},
        required_shared=["model.embed_tokens.safetensors"],
    )
    assert set(receipt["layers"]) == {
        "model.layers.0.safetensors",
        "model.layers.1.safetensors",
        "model.layers.2.safetensors",
        "model.layers.3.safetensors",
    }
    assert "model.embed_tokens.safetensors" in receipt["shared"]
    assert (out / "merge-receipt.json").is_file()
    assert (out / "qtensors" / "model.layers.2.safetensors").is_file()
    assert json.loads((out / "merge-receipt.json").read_text())["schema"] == "merge-receipt/v1"


def test_merge_overlap_fails(tmp_path) -> None:
    d0 = tmp_path / "node0"
    d1 = tmp_path / "node1"
    _make_shard(d0, [0, 1])
    _make_shard(d1, [1, 2])
    with pytest.raises(SystemExit, match="overlap"):
        merge([d0, d1], tmp_path / "out", expected_layers={0, 1, 2})


def test_merge_gap_fails(tmp_path) -> None:
    d0 = tmp_path / "node0"
    d1 = tmp_path / "node1"
    _make_shard(d0, [0])
    _make_shard(d1, [2])
    with pytest.raises(SystemExit, match="gap"):
        merge([d0, d1], tmp_path / "out", expected_layers={0, 1, 2})


def test_merge_duplicate_shared_fails(tmp_path) -> None:
    d0 = tmp_path / "node0"
    d1 = tmp_path / "node1"
    _make_shard(d0, [0], ["model.embed_tokens.safetensors"])
    _make_shard(d1, [1], ["model.embed_tokens.safetensors"])
    with pytest.raises(SystemExit, match="overlap"):
        merge([d0, d1], tmp_path / "out", expected_layers={0, 1})


def test_merge_missing_shared_fails(tmp_path) -> None:
    d0 = tmp_path / "node0"
    d1 = tmp_path / "node1"
    _make_shard(d0, [0])
    _make_shard(d1, [1])
    with pytest.raises(SystemExit, match="missing shared"):
        merge(
            [d0, d1],
            tmp_path / "out",
            expected_layers={0, 1},
            required_shared=["model.embed_tokens.safetensors"],
        )


def test_merge_ignores_non_tensor_artifacts(tmp_path) -> None:
    d0 = tmp_path / "node0"
    d1 = tmp_path / "node1"
    _make_shard(d0, [0], ["model.embed_tokens.safetensors"])
    _make_shard(d1, [1])
    # Each shard writes its own manifest/recipe JSON; these are not tensors and
    # must be skipped (not promoted, not overlap-aborted), even when duplicated.
    for d in (d0, d1):
        (d / "qtensors" / "manifest.json").write_text('{"layers": 1}')
        (d / "qtensors" / "recipe.yaml").write_text("tensors: {}\n")
    receipt = merge([d0, d1], tmp_path / "out", expected_layers={0, 1})
    assert set(receipt["shared"]) == {"model.embed_tokens.safetensors"}
    assert not (tmp_path / "out" / "qtensors" / "manifest.json").exists()
    assert not (tmp_path / "out" / "qtensors" / "recipe.yaml").exists()


def test_parse_layer_spec() -> None:
    assert parse_layer_spec("0-2,5") == {0, 1, 2, 5}
