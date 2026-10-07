"""End-to-end test of `assemble.run()` on a synthetic source + pack.

`assemble` is fully runnable with plain torch + safetensors (no b12x/exllamav3),
so on a host with torch installed this exercises the whole dense-assembly path:
strip routed experts, dequant FP8-block pairs, re-shard, copy the exl3 container,
patch `quantization_config`, copy aux files, and the hidden_size guard.

Skips on a torch-free host (pytest.importorskip).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from exl3pack import assemble  # noqa: E402


def _write_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiMoV2ForCausalLM"],
                "text_config": {
                    "hidden_size": 8,
                    "moe_intermediate_size": 32,
                    "num_key_value_heads": 1,
                    "n_routed_experts": 2,
                    "num_hidden_layers": 2,
                },
                "quantization_config": {"weight_block_size": [128, 128], "fmt": "e4m3"},
            }
        )
    )
    bf16 = torch.bfloat16
    f8 = torch.float8_e4m3fn
    tensors = {
        "model.embed_tokens.weight": torch.ones(4, 8, dtype=bf16),
        "model.layers.0.self_attn.o_proj.weight": torch.ones(8, 8, dtype=bf16),
        # Fused QKV with an FP8 pair — the real MiMo fused-attention name.
        "model.layers.0.self_attn.qkv_proj.weight": torch.ones(12, 8, dtype=f8),
        "model.layers.0.self_attn.qkv_proj.weight_scale_inv": torch.ones(1, 1, dtype=torch.float32),
        "model.layers.0.self_attn.k_proj.weight": torch.ones(2, 8, dtype=bf16),
        "model.layers.0.self_attn.v_proj.weight": torch.ones(2, 8, dtype=bf16),
        "model.layers.0.mlp.gate_proj.weight": torch.ones(4, 8, dtype=f8),
        "model.layers.0.mlp.gate_proj.weight_scale_inv": torch.ones(1, 1, dtype=torch.float32),
        "model.layers.0.mlp.up_proj.weight": torch.ones(4, 8, dtype=f8),
        "model.layers.0.mlp.up_proj.weight_scale_inv": torch.ones(1, 1, dtype=torch.float32),
        "model.layers.0.mlp.down_proj.weight": torch.ones(8, 4, dtype=f8),
        "model.layers.0.mlp.down_proj.weight_scale_inv": torch.ones(1, 1, dtype=torch.float32),
        "model.layers.0.mlp.eh_proj.weight": torch.ones(8, 8, dtype=bf16),
        "model.layers.1.mlp.experts.0.gate_proj.weight": torch.ones(2, 2, dtype=bf16),
        "lm_head.weight": torch.ones(4, 8, dtype=bf16),
    }
    shard = "model-00001-of-00001.safetensors"
    safetensors_torch.save_file(tensors, str(root / shard))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard for k in tensors}})
    )
    # An aux file that must be copied to the output.
    (root / "tokenizer_config.json").write_text('{"model_max_length": 1}')


def _write_pack(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "exl3-manifest.json").write_text(
        json.dumps(
            {
                "kind": "exl3-manifest",
                "schema": "exl3-v1",
                "codebook": "mcg",
                "geometry": {"hidden_size": 8, "num_experts": 2, "intermediate_size": 32},
                "rates": {"structure": "uniform", "bits": 3},
                "layers": {"1": {"file": "exl3-layer-00001.safetensors", "sha256": "deadbeef"}},
            }
        )
    )
    safetensors_torch.save_file(
        {"codes": torch.zeros(2, 2, dtype=torch.int16)}, str(root / "exl3-layer-00001.safetensors")
    )


def test_assemble_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "src"
    pack = tmp_path / "pack"
    out = tmp_path / "serve"
    _write_source(src)
    _write_pack(pack)

    assemble.run(src, pack, out, progress=lambda _s: None)

    # Experts stripped from the assembled index.
    idx = json.loads((out / "model.safetensors.index.json").read_text())
    assert all(".mlp.experts." not in k for k in idx["weight_map"])
    assert "model.embed_tokens.weight" in idx["weight_map"]
    assert "lm_head.weight" in idx["weight_map"]
    # Output shards are renamed to the -of-<NNNNN> form.
    assert any(name.startswith("model-") and "-of-" in name for name in idx["weight_map"].values())

    # FP8 pairs dequantized: every raw *_scale_inv is gone, the weight remains.
    assert "model.layers.0.self_attn.qkv_proj.weight" in idx["weight_map"]
    assert "model.layers.0.mlp.gate_proj.weight" in idx["weight_map"]
    for k in idx["weight_map"]:
        assert not k.endswith("weight_scale_inv"), f"scale_inv leaked: {k}"
    # The dequantized weight is bf16, not the source fp8.
    with safetensors_torch.safe_open(
        str(out / idx["weight_map"]["model.layers.0.self_attn.qkv_proj.weight"]), framework="pt"
    ) as h:
        assert h.get_tensor("model.layers.0.self_attn.qkv_proj.weight").dtype == torch.bfloat16

    # exl3 container + manifest copied.
    assert (out / "exl3-manifest.json").is_file()
    assert (out / "exl3-layer-00001.safetensors").is_file()

    # Aux file copied.
    assert (out / "tokenizer_config.json").is_file()

    # quantization_config patched to the exl3 shape.
    cfg = json.loads((out / "config.json").read_text())
    qc = cfg["quantization_config"]
    assert qc["quant_method"] == "exl3"
    assert qc["codebook"] == "mcg"
    assert qc["bits"] == 3
    assert qc["dense_format"] == "bf16"  # fp8_bases present -> bf16
    # ignored_layers derives module basenames from the keys actually present.
    # This is the real MiMo shape: fused qkv + every _proj basename + gate/lm_head.
    ignored = set(qc["ignored_layers"])
    for name in (
        "qkv_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "eh_proj",
        "gate",
        "lm_head",
    ):
        assert name in ignored, f"{name} missing from ignored_layers"
    assert qc["original_quantization_config"]["weight_block_size"] == [128, 128]


def test_assemble_rejects_wrong_pack(tmp_path: Path) -> None:
    src = tmp_path / "src"
    pack = tmp_path / "pack"
    out = tmp_path / "serve"
    _write_source(src)
    _write_pack(pack)
    # Corrupt the manifest geometry so it mismatches the source hidden_size.
    m = json.loads((pack / "exl3-manifest.json").read_text())
    m["geometry"]["hidden_size"] = 4096
    (pack / "exl3-manifest.json").write_text(json.dumps(m))

    with pytest.raises(SystemExit):
        assemble.run(src, pack, out, progress=lambda _s: None)


def test_assemble_refuses_nonempty_output_without_force(tmp_path: Path) -> None:
    src = tmp_path / "src"
    pack = tmp_path / "pack"
    out = tmp_path / "serve"
    _write_source(src)
    _write_pack(pack)
    out.mkdir()
    (out / "stale.txt").write_text("x")
    with pytest.raises(SystemExit):
        assemble.run(src, pack, out, progress=lambda _s: None)
