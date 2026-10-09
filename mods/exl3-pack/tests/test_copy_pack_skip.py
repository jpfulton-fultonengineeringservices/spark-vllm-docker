"""`_copy_pack` same-size skip: pre-existing layer files are not rewritten."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

import torch  # noqa: E402
from safetensors.torch import save_file as safetensors_save  # noqa: E402

from exl3pack import assemble  # noqa: E402


def test_same_size_layer_not_rewritten(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "exl3-manifest.json").write_text(
        json.dumps({"format": "exl3-v1", "rates": {"bits": 3}, "geometry": {}})
    )
    layer = pack / "exl3-layer-00001.safetensors"
    safetensors_save({"quant": torch.zeros(2, 2, dtype=torch.int16)}, str(layer))

    out = tmp_path / "out"
    out.mkdir()
    dest = out / layer.name
    safetensors_save({"quant": torch.zeros(2, 2, dtype=torch.int16)}, str(dest))
    before_stat = dest.stat()
    before = (before_stat.st_mtime_ns, before_stat.st_ino)

    copied = assemble._copy_pack(
        pack, out, progress=lambda _msg: None, cleanup=assemble.CleanupPolicy()
    )

    after_stat = dest.stat()
    after = (after_stat.st_mtime_ns, after_stat.st_ino)
    assert copied == 1
    assert before == after, "same-size layer must not be rewritten"
    # The source pack file must also survive (no cleanup requested).
    assert layer.is_file()
    assert os.path.exists(dest)
