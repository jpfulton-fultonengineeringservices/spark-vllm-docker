"""Host tests for the distributed-quantization shared contract.

The bulk of this module is torch-free: it exercises :mod:`exl3pack.dist_types`
without importing torch or safetensors at all. The H-file round-trip tests at
the bottom need ``safetensors.torch``; each of those calls
``pytest.importorskip`` at the top of the test body so they skip cleanly on a
torch-free host while keeping the module import torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exl3pack.dist_types import (
    ShardSpec,
    WorkerEndpoint,
    cfg_hash,
    coverage_assert,
    load_h_file,
    read_done_marker,
    save_h_file,
    sha256_file,
    write_atomic,
    write_done_marker,
)

_CFG_KWARGS = {
    "exllamav3_version": "0.3.1",
    "wheel_sha": "abc123",
    "bits": 3,
    "codebook": "default",
    "recipe_strategy": "auto",
    "seed_basis": "fixed",
    "devices": (0, 1),
    "apply_out_scales": True,
    "source_index_hash": "deadbeef",
    "weights_digest": "cafebabe",
}


def _spec(**overrides: object) -> ShardSpec:
    base: dict[str, object] = {
        "job_id": "job-1",
        "module_idx": 12,
        "module_key": "model.layers.12",
        "shard_idx": 2,
        "linear_keys": ("model.layers.12.mlp.experts.0.up_proj",),
        "qmaps": ("model.layers.12.mlp.experts.0",),
        "h_dir": "/nfs/h/job-1/model.layers.12",
        "weights_source": "/nfs/ckpt/index.json",
        "result_uri": "/nfs/out/job-1/shard-2.safetensors",
        "cfg_hash": cfg_hash(**_CFG_KWARGS),  # type: ignore[arg-type]
    }
    base.update(overrides)
    return ShardSpec(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Torch-free: ShardSpec JSON round-trip
# ---------------------------------------------------------------------------


def test_shard_spec_json_round_trip_empty_and_unicode() -> None:
    spec = _spec(
        job_id="",
        module_key="model.layers.12.mlp.experts.Ω.up_proj",
        linear_keys=("", "ünïcødé-🔑", "plain"),
        qmaps=("κόσμε",),
        h_dir="/tmp/∅",
    )
    text = spec.to_json()
    back = ShardSpec.from_json(text)
    assert back == spec
    # Canonical form: sorted keys, compact separators.
    assert json.loads(text) == json.loads(spec.to_json())
    assert back.linear_keys == spec.linear_keys
    assert back.qmaps == spec.qmaps


def test_shard_spec_json_round_trip_preserves_tuple_types() -> None:
    spec = _spec(linear_keys=(), qmaps=())
    back = ShardSpec.from_json(spec.to_json())
    assert isinstance(back.linear_keys, tuple)
    assert isinstance(back.qmaps, tuple)
    assert back.linear_keys == ()
    assert back.qmaps == ()


def test_shard_spec_to_json_is_stable_for_key_order() -> None:
    a = _spec().to_json()
    b = _spec().to_json()
    assert a == b


def test_shard_spec_from_json_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        ShardSpec.from_json("[1,2,3]")


def test_shard_spec_from_json_rejects_missing_field() -> None:
    d = json.loads(_spec().to_json())
    del d["cfg_hash"]
    with pytest.raises(ValueError):
        ShardSpec.from_json(json.dumps(d))


def test_worker_endpoint_is_frozen() -> None:
    ep = WorkerEndpoint(node="gx10-node4", inbox="/nfs/inbox", device=0)
    assert ep.node == "gx10-node4"
    assert ep.device == 0
    with pytest.raises(AttributeError):
        ep.node = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Torch-free: cfg_hash / sha256_file
# ---------------------------------------------------------------------------


def test_cfg_hash_is_deterministic() -> None:
    a = cfg_hash(**_CFG_KWARGS)  # type: ignore[arg-type]
    b = cfg_hash(**_CFG_KWARGS)  # type: ignore[arg-type]
    assert a == b
    assert len(a) == 64
    assert all(c in "0123456789abcdef" for c in a)


def test_cfg_hash_sensitive_to_every_field() -> None:
    base = cfg_hash(**_CFG_KWARGS)  # type: ignore[arg-type]
    variants: list[dict[str, object]] = [
        {**_CFG_KWARGS, "exllamav3_version": "0.3.2"},
        {**_CFG_KWARGS, "wheel_sha": "abc124"},
        {**_CFG_KWARGS, "bits": 4},
        {**_CFG_KWARGS, "codebook": "other"},
        {**_CFG_KWARGS, "recipe_strategy": "manual"},
        {**_CFG_KWARGS, "seed_basis": "random"},
        {**_CFG_KWARGS, "devices": (0,)},
        {**_CFG_KWARGS, "apply_out_scales": False},
        {**_CFG_KWARGS, "source_index_hash": "deadbeee"},
        {**_CFG_KWARGS, "weights_digest": "cafebabf"},
    ]
    for v in variants:
        assert cfg_hash(**v) != base, v  # type: ignore[arg-type]


def test_cfg_hash_canonical_json_ignores_dict_order() -> None:
    a = cfg_hash(**_CFG_KWARGS)  # type: ignore[arg-type]
    shuffled = dict(reversed(list(_CFG_KWARGS.items())))
    b = cfg_hash(**shuffled)  # type: ignore[arg-type]
    assert a == b


def test_sha256_file(tmp_path: Path) -> None:
    f = tmp_path / "blob.bin"
    payload = b"hello exl3" * 1000
    f.write_bytes(payload)
    import hashlib

    assert sha256_file(f) == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Torch-free: write_atomic / done-marker
# ---------------------------------------------------------------------------


def test_write_atomic_writes_and_leaves_no_tmp(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "data.bin"

    def _w(p: Path) -> None:
        p.write_bytes(b"payload")

    write_atomic(target, _w)
    assert target.read_bytes() == b"payload"
    leftovers = [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []


def test_write_atomic_replaces_existing(tmp_path: Path) -> None:
    target = tmp_path / "data.bin"
    target.write_bytes(b"old")

    def _w(p: Path) -> None:
        p.write_bytes(b"new")

    write_atomic(target, _w)
    assert target.read_bytes() == b"new"


def test_write_atomic_cleans_tmp_on_writer_failure(tmp_path: Path) -> None:
    target = tmp_path / "data.bin"

    def _w(p: Path) -> None:
        p.write_bytes(b"partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_atomic(target, _w)
    assert not target.exists()
    leftovers = [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []


def test_done_marker_round_trip(tmp_path: Path) -> None:
    done = tmp_path / "shard.done"
    write_done_marker(done, "a" * 64)
    assert read_done_marker(done) == "a" * 64


def test_done_marker_no_tmp_left(tmp_path: Path) -> None:
    done = tmp_path / "shard.done"
    write_done_marker(done, "b" * 64)
    leftovers = [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []


def test_read_done_marker_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_done_marker(tmp_path / "nope.done")


def test_read_done_marker_empty_raises(tmp_path: Path) -> None:
    done = tmp_path / "empty.done"
    done.write_text("")
    with pytest.raises(ValueError):
        read_done_marker(done)


def test_read_done_marker_whitespace_only_raises(tmp_path: Path) -> None:
    done = tmp_path / "ws.done"
    done.write_text("  \n\t ")
    with pytest.raises(ValueError):
        read_done_marker(done)


def test_write_done_marker_rejects_empty_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_done_marker(tmp_path / "x.done", "")


# ---------------------------------------------------------------------------
# Torch-free: coverage_assert
# ---------------------------------------------------------------------------


def test_coverage_assert_exact_match_passes() -> None:
    coverage_assert({"a", "b"}, {"a", "b"})


def test_coverage_assert_empty_matches_empty() -> None:
    coverage_assert(set(), set())


def test_coverage_assert_missing_raises() -> None:
    with pytest.raises(ValueError) as exc:
        coverage_assert({"a", "b"}, {"a"})
    assert "missing" in str(exc.value)


def test_coverage_assert_extra_raises() -> None:
    with pytest.raises(ValueError) as exc:
        coverage_assert({"a"}, {"a", "z"})
    assert "extra" in str(exc.value)


def test_coverage_assert_both_sides_raise() -> None:
    with pytest.raises(ValueError) as exc:
        coverage_assert({"a", "b"}, {"a", "z"})
    msg = str(exc.value)
    assert "missing" in msg and "extra" in msg


# ---------------------------------------------------------------------------
# Torch-dependent: save_h_file / load_h_file (skipped on torch-free hosts)
# ---------------------------------------------------------------------------


def _make_record(torch_mod: object) -> dict[str, object]:
    import torch

    h = torch.zeros(4, 4, dtype=torch.float32)
    h[0, 0] = 3.5
    return {
        "H": h,
        "count": 5,
        "num_total": 10,
        "inf_nan": (1, 2),
        "first_key": "model.layers.12.mlp.experts.0.up_proj",
        "device": 0,
        "finalized": True,
    }


def test_h_file_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("safetensors.torch")
    import torch

    rec = _make_record(torch)
    path = tmp_path / "h" / "qmap.safetensors"
    save_h_file(path, rec)
    back = load_h_file(path)

    assert set(back) == {
        "H",
        "count",
        "num_total",
        "inf_nan",
        "first_key",
        "device",
        "finalized",
    }
    assert torch.equal(back["H"], rec["H"])  # type: ignore[arg-type]
    assert back["count"] == 5
    assert back["num_total"] == 10
    assert back["inf_nan"] == (1, 2)
    assert isinstance(back["inf_nan"], tuple)
    assert back["first_key"] == "model.layers.12.mlp.experts.0.up_proj"
    assert back["device"] == 0
    assert isinstance(back["device"], int)
    assert back["finalized"] is True


def test_h_file_strips_h_swap_device(tmp_path: Path) -> None:
    pytest.importorskip("safetensors.torch")
    import torch

    rec = _make_record(torch)
    rec["H_swap_device"] = "cuda:1"  # type: ignore[index]
    path = tmp_path / "h" / "qmap.safetensors"
    save_h_file(path, rec)
    back = load_h_file(path)

    assert "H_swap_device" not in back
    assert set(back) == {
        "H",
        "count",
        "num_total",
        "inf_nan",
        "first_key",
        "device",
        "finalized",
    }


def test_h_file_round_trip_int_inf_nan_and_str_device(tmp_path: Path) -> None:
    pytest.importorskip("safetensors.torch")
    import torch

    rec = _make_record(torch)
    rec["inf_nan"] = 7
    rec["device"] = "cpu:0"
    path = tmp_path / "h" / "qmap.safetensors"
    save_h_file(path, rec)
    back = load_h_file(path)

    assert back["inf_nan"] == 7
    assert isinstance(back["inf_nan"], int)
    assert back["device"] == "cpu:0"
    assert isinstance(back["device"], str)


def test_h_file_no_tmp_left(tmp_path: Path) -> None:
    pytest.importorskip("safetensors.torch")
    import torch

    rec = _make_record(torch)
    path = tmp_path / "h" / "qmap.safetensors"
    save_h_file(path, rec)
    leftovers = [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []


def test_save_h_file_requires_h_tensor(tmp_path: Path) -> None:
    pytest.importorskip("safetensors.torch")

    with pytest.raises(ValueError):
        save_h_file(tmp_path / "x.safetensors", {"count": 1})
