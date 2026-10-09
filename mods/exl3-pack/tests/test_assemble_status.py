"""Assemble observability: `.pack-status.json` writes + JSONL event log.

Mirrors the fake source/pack fixtures from ``test_assemble_e2e.py`` (torch-free
via importorskip) and asserts the status-file phase transitions and JSONL
``assemble.*`` events emitted by ``assemble.run``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

import torch  # noqa: E402
from safetensors.torch import save_file as safetensors_save  # noqa: E402

from exl3pack import assemble  # noqa: E402

bf16 = torch.bfloat16
f8 = torch.float8_e4m3fn


def _make_source(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiMoV2ForCausalLM"],
                "text_config": {
                    "hidden_size": 8,
                    "num_key_value_heads": 1,
                    "num_hidden_layers": 2,
                },
                "quantization_config": {
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    tensors = {
        "model.embed_tokens.weight": torch.ones(4, 8, dtype=bf16),
        "model.layers.0.self_attn.o_proj.weight": torch.ones(8, 8, dtype=bf16),
        # FP8 QKV (MiMo fused-attention) with scale_inv
        "model.layers.0.self_attn.qkv_proj.weight": torch.ones(12, 8, dtype=f8),
        "model.layers.0.self_attn.qkv_proj.weight_scale_inv": torch.ones(1, 1, dtype=torch.float32),
        # FP8 MoE MLP
        "model.layers.0.mlp.gate_proj.weight": torch.ones(4, 8, dtype=f8),
        "model.layers.0.mlp.gate_proj.weight_scale_inv": torch.ones(1, 1, dtype=torch.float32),
        "model.layers.1.mlp.experts.0.gate_proj.weight": torch.ones(2, 2, dtype=bf16),
        "lm_head.weight": torch.ones(4, 8, dtype=bf16),
    }
    safetensors_save(tensors, str(root / "model-00001-of-00001.safetensors"))
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}
        )
    )
    (root / "tokenizer_config.json").write_text('{"model_max_length": 1}')
    return root


def _make_pack(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "exl3-manifest.json").write_text(
        json.dumps(
            {
                "format": "exl3-v1",
                "geometry": {"hidden_size": 8, "num_layers": 2, "num_slots": 32},
                "rates": {"bits": 3},
                "shards": {
                    "1": "exl3-layer-00001.safetensors",
                    "2": "exl3-layer-00002.safetensors",
                },
            }
        )
    )
    for name in ("exl3-layer-00001.safetensors", "exl3-layer-00002.safetensors"):
        safetensors_save(
            {"quant": torch.zeros(2, 2, dtype=torch.int16)}, str(root / name)
        )
    return root


def _events(log_dir: Path) -> list[dict]:
    lines = (log_dir / "assemble.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_status_done_and_events(tmp_path: Path) -> None:
    src = _make_source(tmp_path / "source")
    pack = _make_pack(tmp_path / "pack")
    out = tmp_path / "v1-out"
    status = tmp_path / "status.json"
    log_dir = tmp_path / "logs"

    assemble.run(
        src,
        pack,
        out,
        status_file=status,
        log_dir=log_dir,
        progress=lambda _msg: None,
    )

    st = json.loads(status.read_text())
    assert st["stage"] == "assemble"
    assert st["phase"] == "done"
    assert st["layers_completed"] == 2
    assert st["output_size"] > 0

    events = _events(log_dir)
    names = [e["event"] for e in events]
    assert "assemble.start" in names
    assert "assemble.layer_copied" in names
    assert "assemble.config_patched" in names
    assert names.count("assemble.layer_copied") == 2
    assert "assemble.done" in names
    assert "assemble.error" not in names


def test_status_error_written_before_reraise(tmp_path: Path) -> None:
    src = _make_source(tmp_path / "source")
    pack = _make_pack(tmp_path / "pack")
    out = tmp_path / "v1-out"
    out.mkdir()
    (out / "stale.txt").write_text("x")
    status = tmp_path / "status.json"
    log_dir = tmp_path / "logs"

    with pytest.raises(SystemExit):
        assemble.run(
            src,
            pack,
            out,
            status_file=status,
            log_dir=log_dir,
            progress=lambda _msg: None,
        )

    st = json.loads(status.read_text())
    assert st["stage"] == "assemble"
    assert st["phase"] == "error"
    assert st["errors"]
    assert "not empty" in st["errors"][0]

    names = [e["event"] for e in _events(log_dir)]
    assert "assemble.error" in names
