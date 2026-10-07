"""Host tests for the torch-free shard planner."""

from __future__ import annotations

import json

import pytest

from exl3pack.shard import Shard, discover_moe_layers, module_args, plan_shards


def test_plan_flash_51_modules_four_shards() -> None:
    shards = plan_shards(51, 4, num_hidden_layers=48)
    assert [(s.module_start, s.module_end) for s in shards] == [
        (0, 12),
        (13, 25),
        (26, 38),
        (39, 50),
    ]


def test_plan_pro_73_modules_four_shards() -> None:
    shards = plan_shards(73, 4, num_hidden_layers=70)
    assert [(s.module_start, s.module_end) for s in shards] == [
        (0, 18),
        (19, 36),
        (37, 54),
        (55, 72),
    ]


@pytest.mark.parametrize(("modules", "layers"), [(51, 48), (73, 70)])
def test_plan_covers_all_layers_disjointly(modules: int, layers: int) -> None:
    shards = plan_shards(modules, 4, num_hidden_layers=layers)
    seen: list[int] = []
    for s in shards:
        assert s.layers, f"shard {s.index} empty"
        seen.extend(s.layers)
    assert sorted(seen) == list(range(layers))
    assert len(seen) == len(set(seen))


def test_plan_keeps_shards_roughly_even() -> None:
    shards = plan_shards(73, 4, num_hidden_layers=70)
    sizes = [s.module_end - s.module_start + 1 for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_module_args_uses_absolute_inclusive_end() -> None:
    shard = Shard(0, 0, 12, 48)
    assert module_args(shard) == ["--module-start", "0", "--max_module", "12"]
    assert module_args(13, 25) == ["--module-start", "13", "--max_module", "25"]


def test_plan_rejects_bad_counts() -> None:
    with pytest.raises(ValueError):
        plan_shards(0, 1)
    with pytest.raises(ValueError):
        plan_shards(10, 0)
    with pytest.raises(ValueError):
        plan_shards(10, 11)


def test_discover_moe_layers_reads_index(tmp_path) -> None:
    index = {
        "weight_map": {
            "model.embed_tokens.weight": "a.safetensors",
            "model.layers.0.mlp.experts.0.gate_proj.trellis": "b.safetensors",
            "model.layers.1.self_attn.o_proj.weight": "c.safetensors",
            "model.layers.2.mlp.experts.3.down_proj.trellis": "d.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert discover_moe_layers(tmp_path) == [0, 2]
