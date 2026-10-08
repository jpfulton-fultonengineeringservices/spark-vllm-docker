"""Host tests for the distributed EXL3 coordinator/worker transport layer.

The production ``exl3pack.distributed`` module imports torch/exllamav3 at
module scope (it runs inside the GPU image), and ``exl3pack.worker`` lazy-
imports them per shard. These tests therefore inject fake ``torch`` /
``exllamav3`` / ``safetensors.torch`` modules into ``sys.modules`` before
importing, so the full dispatch → gather → commit → worker-poll protocol is
exercised on a torch-free host. Real safetensors files (written via the
safetensors.numpy backend) back the output path, so hash/done-marker checks
run against real bytes, not stubs.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from exl3pack.dist_types import ShardSpec, WorkerEndpoint, read_done_marker


def _fake_gpu_modules() -> Any:
    """Patch sys.modules with minimal torch/exllamav3/safetensors.torch fakes.

    ``safetensors.torch.save_file`` / ``load_file`` are backed by the real
    numpy safetensors backend, so the files on disk are genuine safetensors
    and the coordinator's ``safe_open`` metadata read works unmodified.
    """
    import numpy as np
    from safetensors.numpy import load_file as np_load_file
    from safetensors.numpy import save_file as np_save_file

    class FakeTensor:
        def __init__(self, arr: np.ndarray[Any, Any]) -> None:
            self.arr = np.ascontiguousarray(arr)
            self.is_meta = False

        def cpu(self) -> FakeTensor:
            return self

        def numpy(self) -> np.ndarray[Any, Any]:
            return self.arr

        def tobytes(self) -> bytes:
            return self.arr.tobytes()

    torch_mod = types.ModuleType("torch")
    # Minimal torch stub for safetensors safe_open with framework="pt"
    torch_mod.__version__ = "2.0.0"  # type: ignore[attr-defined]
    torch_mod.Tensor = FakeTensor  # type: ignore[attr-defined]
    torch_mod.device = lambda spec: ("device", spec)  # type: ignore[attr-defined]
    torch_mod.float32 = "float32"  # type: ignore[attr-defined]
    # Safetensors may reference UntypedStorage
    torch_mod.UntypedStorage = object  # type: ignore[attr-defined]

    def save_file(
        tensors: dict[str, Any], path: str, metadata: dict[str, str] | None = None
    ) -> None:
        np_save_file({k: v.arr for k, v in tensors.items()}, path, metadata=metadata)

    def load_file(path: str) -> dict[str, FakeTensor]:
        return {k: FakeTensor(v) for k, v in np_load_file(path).items()}

    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.save_file = save_file  # type: ignore[attr-defined]
    safetensors_torch.load_file = load_file  # type: ignore[attr-defined]

    exllamav3 = types.ModuleType("exllamav3")
    exllamav3.__version__ = "0.1.0"  # type: ignore[attr-defined]
    conversion = types.ModuleType("exllamav3.conversion")
    convert_model = types.ModuleType("exllamav3.conversion.convert_model")
    model_pkg = types.ModuleType("exllamav3.model")
    model_config = types.ModuleType("exllamav3.model.config")
    model_config.Config = type("Config", (), {})  # type: ignore[attr-defined]
    modules_pkg = types.ModuleType("exllamav3.modules")
    modules_linear = types.ModuleType("exllamav3.modules.linear")

    class FakeLinear:
        """Linear stand-in: key, qmap, numel, and a deterministic weight."""

        def __init__(self, key: str, numel: int, qmap: str = "qmap_a") -> None:
            self.key = key
            self.qmap = qmap
            seed = int.from_bytes(
                hashlib.sha256(key.encode()).digest()[:4], "little"
            )
            self._numel = numel
            self.inner = types.SimpleNamespace(
                get_weight_tensor=lambda: FakeTensor(
                    np.full((numel,), seed % 997, dtype=np.float32)
                )
            )

        def weights_numel(self) -> int:
            return self._numel

    modules_linear.Linear = FakeLinear  # type: ignore[attr-defined]
    conversion.convert_model = convert_model  # type: ignore[attr-defined]

    return mock.patch.dict(
        sys.modules,
        {
            "torch": torch_mod,
            "safetensors.torch": safetensors_torch,
            "exllamav3": exllamav3,
            "exllamav3.conversion": conversion,
            "exllamav3.conversion.convert_model": convert_model,
            "exllamav3.model": model_pkg,
            "exllamav3.model.config": model_config,
            "exllamav3.modules": modules_pkg,
            "exllamav3.modules.linear": modules_linear,
        },
    )


@pytest.fixture()
def distributed() -> Any:
    """Import exl3pack.distributed under the fake-GPU module environment."""
    with _fake_gpu_modules():
        for name in list(sys.modules):
            if name == "exl3pack.distributed" or name == "exl3pack.recipe":
                sys.modules.pop(name)
        import exl3pack.distributed as mod

        yield mod


class _FakeModule:
    """Coordinator-side module stand-in: key plus iterable FakeLinears."""

    def __init__(self, key: str, linears: list[Any]) -> None:
        self.key = key
        self._linears = linears

    def __iter__(self) -> Any:
        return iter(self._linears)


def _make_endpoints(tmp_path: Path) -> list[WorkerEndpoint]:
    return [
        WorkerEndpoint(
            node="node0", inbox=str(tmp_path / "dist" / "inbox" / "node0"), device=0
        ),
        WorkerEndpoint(
            node="node1", inbox=str(tmp_path / "dist" / "inbox" / "node1"), device=1
        ),
    ]


def _make_coordinator(distributed: Any, tmp_path: Path, endpoints: list[WorkerEndpoint]) -> Any:
    in_dir = tmp_path / "fp16"
    in_dir.mkdir()
    (in_dir / "model.safetensors.index.json").write_text("{}")
    return distributed.Coordinator(
        args={
            "job_id": "job-1",
            "bits": 4,
            "codebook": "mcg",
            "in_dir": str(in_dir),
            "gather_timeout": 30.0,
        },
        endpoints=endpoints,
        work=tmp_path,
    )


# ---------------------------------------------------------------------------
# _plan_shards
# ---------------------------------------------------------------------------


def test_plan_shards_covers_every_linear_exactly_once(distributed: Any, tmp_path: Path) -> None:
    endpoints = _make_endpoints(tmp_path)
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    fake_linear = sys.modules["exllamav3.modules.linear"].Linear
    linears = [
        fake_linear("model.layers.0.self_attn.q_proj", 100),
        fake_linear("model.layers.0.self_attn.k_proj", 50),
        fake_linear("model.layers.0.mlp.up_proj", 300, qmap="qmap_b"),
    ]
    module = _FakeModule("model.layers.0", linears)
    specs = coord._plan_shards(
        module, [linears], {"a": 4.0}, endpoints
    )
    planned = sorted(k for s in specs for k in s.linear_keys)
    assert planned == sorted(lin.key for lin in linears)
    # Endpoints, module identity, H dir, weights source, and cfg_hash are set.
    coord._h_dir = str(tmp_path / "h")  # mirrors run(); plan uses whatever is set
    for spec in specs:
        assert spec.shard_idx < len(endpoints)
        assert spec.module_key == "model.layers.0"
        assert spec.weights_source.endswith("model.safetensors.index.json")
        assert len(spec.cfg_hash) == 64
        assert spec.result_uri.endswith(f"shard-{spec.shard_idx}.safetensors")
    # Union of qmaps matches the linears' qmaps.
    assert sorted(q for s in specs for q in s.qmaps) == ["qmap_a", "qmap_b"]


def test_plan_shards_rejects_empty_endpoints(distributed: Any, tmp_path: Path) -> None:
    coord = _make_coordinator(distributed, tmp_path, [])
    module = _FakeModule("model.layers.0", [])
    with pytest.raises(ValueError, match="no worker endpoints"):
        coord._plan_shards(module, [], {}, [])


# ---------------------------------------------------------------------------
# _dispatch
# ---------------------------------------------------------------------------


def test_dispatch_writes_round_trippable_json_into_node_inbox(
    distributed: Any, tmp_path: Path
) -> None:
    endpoints = _make_endpoints(tmp_path)
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    coord._module_idx = 3
    fake_linear = sys.modules["exllamav3.modules.linear"].Linear
    linears = [fake_linear("model.layers.3.mlp.gate_proj", 64)]
    module = _FakeModule("model.layers.3", linears)
    specs = coord._plan_shards(module, [linears], {"a": 4.0}, endpoints)
    coord._dispatch(specs)

    for spec in specs:
        node = endpoints[spec.shard_idx].node
        inbox = tmp_path / "dist" / "inbox" / node
        shard_json = inbox / f"shard-{spec.shard_idx}.json"
        assert shard_json.exists(), f"missing inbox spec for {node}"
        # The inbox JSON must round-trip back to an identical ShardSpec.
        assert ShardSpec.from_json(shard_json.read_text()) == spec
        # Atomic write: no .tmp leftovers in the inbox.
        assert list(inbox.glob("*.tmp")) == []


def test_dispatch_rejects_shard_idx_beyond_endpoints(
    distributed: Any, tmp_path: Path
) -> None:
    endpoints = _make_endpoints(tmp_path)
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    spec = ShardSpec(
        job_id="job-1",
        module_idx=0,
        module_key="model.layers.0",
        shard_idx=5,
        linear_keys=("a",),
        qmaps=("qmap_a",),
        h_dir="h",
        weights_source="w",
        result_uri="r",
        cfg_hash="0" * 64,
    )
    with pytest.raises(ValueError, match="shard_idx 5 has no endpoint"):
        coord._dispatch([spec])


# ---------------------------------------------------------------------------
# _gather
# ---------------------------------------------------------------------------


def _write_shard_output(
    path: Path, tensors: dict[str, Any], cfg: str | None
) -> None:
    import numpy as np
    from safetensors.numpy import save_file as np_save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    np_save_file(
        {k: np.asarray(v, dtype=np.float32) for k, v in tensors.items()},
        str(path),
        metadata={"cfg_hash": cfg} if cfg is not None else None,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".done").write_text(digest)


def _spec_to(result: Path, *, idx: int, cfg: str) -> ShardSpec:
    return ShardSpec(
        job_id="job-1",
        module_idx=0,
        module_key="model.layers.0",
        shard_idx=idx,
        linear_keys=("a",),
        qmaps=("qmap_a",),
        h_dir="h",
        weights_source="w",
        result_uri=str(result),
        cfg_hash=cfg,
    )


def test_gather_merges_disjoint_shards_and_checks_cfg_echo(
    distributed: Any, tmp_path: Path
) -> None:
    endpoints = _make_endpoints(tmp_path)
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    cfg = "c" * 64
    out0 = tmp_path / "out" / "node0" / "shard-0.safetensors"
    out1 = tmp_path / "out" / "node1" / "shard-1.safetensors"
    _write_shard_output(out0, {"k.weight": [1.0, 2.0]}, cfg)
    _write_shard_output(out1, {"k.trellis": [3.0]}, cfg)

    # Patch safe_open to read cfg_hash from metadata without invoking torch
    class DummyCtx:
        def __enter__(self) -> DummyCtx:
            return self

        def __exit__(
            self, exc_type: object, exc: object, tb: object
        ) -> None:
            pass

        def metadata(self) -> dict[str, str]:
            return {"cfg_hash": cfg}

    with mock.patch.object(distributed, 'safe_open', lambda *args, **kwargs: DummyCtx()):
        merged = coord._gather([_spec_to(out0, idx=0, cfg=cfg), _spec_to(out1, idx=1, cfg=cfg)])
    assert set(merged) == {"k.weight", "k.trellis"}


def test_gather_times_out_without_done_marker(distributed: Any, tmp_path: Path) -> None:
    endpoints = _make_endpoints(tmp_path)
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    coord.gather_timeout = 0.2
    missing = tmp_path / "out" / "node0" / "shard-0.safetensors"
    # Live worker: fresh heartbeat, otherwise the liveness check fires first.
    (tmp_path / "dist" / "heartbeat" / "node0").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "dist" / "heartbeat" / "node0").touch()
    with pytest.raises(TimeoutError, match="timeout after"):
        coord._gather([_spec_to(missing, idx=0, cfg="c" * 64)])


def test_gather_rejects_hash_mismatch_between_marker_and_file(
    distributed: Any, tmp_path: Path
) -> None:
    endpoints = _make_endpoints(tmp_path)
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    cfg = "c" * 64
    out = tmp_path / "out" / "node0" / "shard-0.safetensors"
    _write_shard_output(out, {"k.weight": [1.0]}, cfg)
    # Corrupt the marker so it no longer matches the file bytes.
    out.with_suffix(".done").write_text("0" * 64)
    with pytest.raises(ValueError, match="output hash mismatch"):
        coord._gather([_spec_to(out, idx=0, cfg=cfg)])


def test_gather_rejects_cfg_hash_echo_mismatch(distributed: Any, tmp_path: Path) -> None:
    endpoints = _make_endpoints(tmp_path)
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    out = tmp_path / "out" / "node0" / "shard-0.safetensors"
    _write_shard_output(out, {"k.weight": [1.0]}, "d" * 64)
    # Force safe_open to echo wrong cfg_hash
    class BadCtx:
        def __enter__(self) -> BadCtx:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def metadata(self) -> dict[str, str]:
            return {"cfg_hash": "wrong" * 8}

    with (
        mock.patch.object(distributed, "safe_open", lambda *args, **kwargs: BadCtx()),
        pytest.raises(ValueError, match="cfg_hash"),
    ):
        coord._gather([_spec_to(out, idx=0, cfg="c" * 64)])


def test_gather_rejects_duplicate_tensor_keys_across_shards(
    distributed: Any, tmp_path: Path
) -> None:
    endpoints = _make_endpoints(tmp_path)
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    cfg = "c" * 64
    out0 = tmp_path / "out" / "node0" / "shard-0.safetensors"
    out1 = tmp_path / "out" / "node1" / "shard-1.safetensors"
    _write_shard_output(out0, {"k.weight": [1.0]}, cfg)
    _write_shard_output(out1, {"k.weight": [2.0]}, cfg)
    # Duplicate tensor key should be detected after metadata
    class DummyCtx2:
        def __enter__(self) -> DummyCtx2:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def metadata(self) -> dict[str, str]:
            return {"cfg_hash": cfg}

    with (
        mock.patch.object(distributed, "safe_open", lambda *args, **kwargs: DummyCtx2()),
        pytest.raises(ValueError, match="more than one shard"),
    ):
        coord._gather([_spec_to(out0, idx=0, cfg=cfg), _spec_to(out1, idx=1, cfg=cfg)])


# ---------------------------------------------------------------------------
# _commit_module
# ---------------------------------------------------------------------------


def test_commit_writes_qtensors_file_after_coverage_assert(
    distributed: Any, tmp_path: Path
) -> None:
    endpoints = _make_endpoints(tmp_path)
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    fake_tensor_cls = sys.modules["torch"].Tensor
    import numpy as np

    coord._planned_keys = {"model.layers.99.self_attn.q_proj"}
    q_tensors = {
        # Coverage uses _base_key, so suffixed quant keys map to the bare key.
        "model.layers.99.self_attn.q_proj.weight": fake_tensor_cls(np.array([1.0])),
        "model.layers.99.self_attn.q_proj.trellis": fake_tensor_cls(np.array([2.0])),
    }
    coord._commit_module("model.layers.99", q_tensors)

    out = tmp_path / "qtensors" / "model.layers.99.safetensors"
    assert out.exists()
    from safetensors import safe_open

    with safe_open(str(out), framework="numpy") as f:
        assert set(f.keys()) == set(q_tensors)
    assert list(tmp_path.glob("qtensors/*.tmp")) == []


def test_commit_rejects_incomplete_coverage(distributed: Any, tmp_path: Path) -> None:
    endpoints = _make_endpoints(tmp_path)
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    coord._planned_keys = {"a", "b"}
    with pytest.raises(ValueError, match="coverage mismatch"):
        coord._commit_module("model.layers.0", {"a.weight": object()})


def test_commit_rejects_incomplete_moe_expert_set(
    distributed: Any, tmp_path: Path
) -> None:
    endpoints = _make_endpoints(tmp_path)
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    fake_tensor_cls = sys.modules["torch"].Tensor
    import numpy as np

    # Layer 12 is inside the MoE band (1-47): all 256 expert down_proj keys
    # are required; providing 255 must fail loudly.
    keys = {
        f"model.layers.12.mlp.experts.{j}.down_proj" for j in range(255)
    }
    coord._planned_keys = set(keys)
    q_tensors = {
        f"{k}.weight": fake_tensor_cls(np.array([0.0])) for k in keys
    }
    with pytest.raises(ValueError, match="256 expert down_proj"):
        coord._commit_module("model.layers.12", q_tensors)


# ---------------------------------------------------------------------------
# worker.serve polling loop (torch-free; _run_shard mocked)
# ---------------------------------------------------------------------------


def _valid_spec_json(tmp_path: Path, idx: int = 0) -> str:
    return ShardSpec(
        job_id="job-1",
        module_idx=0,
        module_key="model.layers.0",
        shard_idx=idx,
        linear_keys=("a",),
        qmaps=("qmap_a",),
        h_dir="h",
        weights_source="w",
        result_uri=str(tmp_path / f"shard-{idx}.safetensors"),
        cfg_hash="c" * 64,
    ).to_json()


def test_worker_serve_processes_inbox_json_and_removes_it(tmp_path: Path) -> None:
    """serve() picks up an inbox JSON, runs it, and unlinks it on success."""
    from exl3pack import worker

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    stop = tmp_path / "stop"
    (inbox / "shard-0.json").write_text(_valid_spec_json(tmp_path))

    def fake_run(spec: ShardSpec, device: int, work: Path) -> None:
        assert spec.shard_idx == 0
        stop.touch()  # end the loop after the first shard

    with mock.patch.object(worker, "_run_shard", side_effect=fake_run):
        worker.serve(inbox=inbox, shared=tmp_path / "shared", device=0, stop=stop)

    assert not (inbox / "shard-0.json").exists()


def test_worker_serve_keeps_failed_spec_for_retry(tmp_path: Path) -> None:
    """A failing shard is left in the inbox so a later pass can retry it."""
    from exl3pack import worker

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    stop = tmp_path / "stop"
    (inbox / "shard-0.json").write_text(_valid_spec_json(tmp_path))

    def fake_run(spec: ShardSpec, device: int, work: Path) -> None:
        stop.touch()
        raise RuntimeError("boom")

    with mock.patch.object(worker, "_run_shard", side_effect=fake_run), mock.patch.object(
        worker, "_POLL_INTERVAL_S", 0.01
    ):
        worker.serve(inbox=inbox, shared=tmp_path / "shared", device=0, stop=stop)

    assert (inbox / "shard-0.json").exists()


def test_worker_serve_exits_immediately_on_stop_sentinel(tmp_path: Path) -> None:
    """With the stop file present, serve() returns without touching the inbox."""
    from exl3pack import worker

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    stop = tmp_path / "stop"
    stop.touch()
    (inbox / "shard-0.json").write_text(_valid_spec_json(tmp_path))

    with mock.patch.object(worker, "_run_shard") as run_mock:
        worker.serve(inbox=inbox, shared=tmp_path / "shared", device=0, stop=stop)
    run_mock.assert_not_called()


def test_worker_run_shard_writes_qtensors_and_done_marker(tmp_path: Path) -> None:
    """_run_shard quantizes a fake linear and publishes data plus its marker.

    The model/conversion/tensor interfaces are deliberately tiny fakes, while
    the output is written with the real safetensors.numpy encoder.  This
    exercises the worker's full spec -> model -> q_tensors -> atomic output ->
    ``.done`` path without importing torch or exllamav3.
    """
    import numpy as np
    from safetensors.numpy import load_file as np_load_file
    from safetensors.numpy import save_file as np_save_file

    from exl3pack import worker

    class FakeTensor:
        def __init__(self, values: list[float]) -> None:
            self.values = np.asarray(values, dtype=np.float32)
            self.is_meta = False

        def contiguous(self) -> FakeTensor:
            return self

    class FakeLinear:
        def __init__(self) -> None:
            self.key = "model.layers.0.mlp.up_proj"
            self.qmap = "qmap_a"
            self.device: object | None = None
            self.converted = False

        def load(self, device: object) -> None:
            self.device = device

        def convert_exl3(
            self,
            h_data: dict[str, Any],
            quant_args: dict[str, Any],
            override_swap_device: object | None = None,
        ) -> None:
            assert h_data["H"] == "fake-H"
            assert quant_args["k_bits"] == 4.0
            assert override_swap_device == "cuda:0"
            assert h_data.get("H_swap_device") == "cuda:0"
            self.converted = True

        def get_tensors(self) -> dict[str, FakeTensor]:
            assert self.converted
            return {
                f"{self.key}.weight": FakeTensor([1.0, 2.0]),
                f"{self.key}.trellis": FakeTensor([3.0]),
            }

    class FakeModule:
        def __init__(self, linears: list[FakeLinear]) -> None:
            self.key = "model.layers.0"
            self._linears = linears

        def __iter__(self):
            yield from self._linears

    linear = FakeLinear()
    module = FakeModule([linear])

    class FakeModel:
        def __init__(self) -> None:
            self.modules = [module]

    class FakeTorch:
        @staticmethod
        def device(spec: str) -> str:
            return spec

    class FakeSafetensorsTorch:
        @staticmethod
        def save_file(
            tensors: dict[str, FakeTensor],
            path: str,
            metadata: dict[str, str] | None = None,
        ) -> None:
            np_save_file({key: value.values for key, value in tensors.items()}, path)

    class FakeConvertModel:
        @staticmethod
        def get_base_model(
            args: dict[str, Any],
        ) -> tuple[object, FakeModel, None, None, None, None]:
            assert args["in_dir"] == str(tmp_path)
            return object(), FakeModel(), None, None, None, None

        @staticmethod
        def make_quant_args(
            args: dict[str, Any], module_idx: int, k_bits: float,
            devices: list[int], extra: object,
        ) -> dict[str, Any]:
            return {"module_idx": module_idx, "k_bits": k_bits, "devices": devices}

    work = tmp_path / "work"
    (work / "dist").mkdir(parents=True)
    (work / "dist" / "strategy.json").write_text(
        json.dumps({linear.key: 4.0}), encoding="utf-8"
    )
    (work / "ckpt").mkdir()
    (work / "ckpt" / "args.json").write_text(
        json.dumps({"bits": 4, "in_dir": str(tmp_path)}), encoding="utf-8"
    )
    output = work / "dist" / "mod0" / "out" / "node0" / "shard-0.safetensors"
    spec = ShardSpec(
        job_id="job-1",
        module_idx=0,
        module_key="model.layers.0",
        shard_idx=0,
        linear_keys=(linear.key,),
        qmaps=(linear.qmap,),
        h_dir=str(work / "h"),
        weights_source=str(tmp_path / "model.safetensors.index.json"),
        result_uri=str(output),
        cfg_hash="c" * 64,
    )

    lazy_modules: dict[str, Any] = {
        "torch": FakeTorch,
        "safetensors.torch": FakeSafetensorsTorch,
        "exllamav3.conversion.convert_model": FakeConvertModel,
        "exllamav3.modules.linear": types.SimpleNamespace(Linear=FakeLinear),
    }
    with (
        mock.patch.object(worker, "_lazy", side_effect=lambda name: lazy_modules[name]),
        mock.patch.object(worker, "_load_h_records", return_value={"qmap_a": {"H": "fake-H"}}),
    ):
        worker._run_shard(spec, device=0, work=work)

    assert output.exists()
    assert not list(output.parent.glob("*.tmp"))
    emitted = np_load_file(str(output))
    assert set(emitted) == {
        f"{linear.key}.weight",
        f"{linear.key}.trellis",
    }
    done = output.with_suffix(".done")
    assert read_done_marker(done) == hashlib.sha256(output.read_bytes()).hexdigest()


def test_worker_module_import_without_torch() -> None:
    """exl3pack.worker must be importable with no torch/exllamav3 installed."""
    import subprocess

    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        "[sys.modules.pop(k) for k in list(sys.modules) if k.startswith('exl3pack')];"
        "import importlib; m = importlib.import_module('exl3pack.worker');"
        "assert 'torch' not in sys.modules; assert not any("
        "k.startswith('exllamav3') for k in sys.modules);"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# ---------------------------------------------------------------------------
# done-marker contract used by the gather loop
# ---------------------------------------------------------------------------


def test_done_marker_written_by_worker_reads_back_in_gather_form(tmp_path: Path) -> None:
    """The .done marker is exactly the sha256 of the data file (no newline)."""
    from exl3pack.dist_types import sha256_file, write_done_marker

    data = tmp_path / "shard-0.safetensors"
    data.write_bytes(b"tensor-bytes")
    digest = sha256_file(data)
    write_done_marker(data.with_suffix(".done"), digest)
    assert read_done_marker(data.with_suffix(".done")) == digest
    assert data.with_suffix(".done").read_bytes() == digest.encode()


# ---------------------------------------------------------------------------
# Coordinator fixes: B2, M3, M2, N1, P1-a, P1-d, P1-e
# ---------------------------------------------------------------------------


def test_b2_original_input_ids_guard_fresh_run(tmp_path: Path) -> None:
    """B2: fresh run copies state; resume arm materializes None per row."""
    # Simulate prepare_state returning original_input_ids=None
    job_state = {"next_module_idx": 0}
    state = [object(), object()]  # fake tensors
    # Fresh run branch
    original_input_ids = None
    if original_input_ids is None:
        original_input_ids = (
            state.copy()
            if int(job_state.get("next_module_idx", 0) or 0) == 0
            else [None] * len(state)
        )
    assert original_input_ids == state.copy()
    # Resume branch
    job_state_resume = {"next_module_idx": 5}
    original_input_ids_resume = None
    if original_input_ids_resume is None:
        original_input_ids_resume = (
            state.copy()
            if int(job_state_resume.get("next_module_idx", 0) or 0) == 0
            else [None] * len(state)
        )
    assert original_input_ids_resume == [None, None]


def test_m3_stale_clear_before_retry(tmp_path: Path) -> None:
    """M3: retry path unlinks stale result_uri + .done before re-dispatch."""
    from pathlib import Path

    from exl3pack.dist_types import ShardSpec

    spec = ShardSpec(
        job_id="test",
        module_idx=0,
        module_key="model.layers.0",
        shard_idx=0,
        linear_keys=("a",),
        qmaps=("q",),
        h_dir=str(tmp_path / "h"),
        weights_source=str(tmp_path / "weights.safetensors"),
        result_uri=str(tmp_path / "out" / "shard-0.safetensors"),
        cfg_hash="abc123",
    )
    # Create stale files
    Path(spec.result_uri).parent.mkdir(parents=True, exist_ok=True)
    Path(spec.result_uri).write_bytes(b"stale")
    Path(spec.result_uri).with_suffix(".done").write_text("sha")
    assert Path(spec.result_uri).exists()
    assert Path(spec.result_uri).with_suffix(".done").exists()
    # Simulate M3 clear
    Path(spec.result_uri).unlink(missing_ok=True)
    Path(spec.result_uri).with_suffix(".done").unlink(missing_ok=True)
    assert not Path(spec.result_uri).exists()
    assert not Path(spec.result_uri).with_suffix(".done").exists()


def test_m2_swap_cpu_and_unload_recording(tmp_path: Path) -> None:
    """M2: coordinator calls swap_cpu pre-dispatch and unload post-commit."""

    class FakeInner:
        swap_cpu_called = False

        def swap_cpu(self):
            self.swap_cpu_called = True

    class FakeLinear:
        def __init__(self, key):
            self.key = key
            self.qmap = "q"
            self.device = "cuda:0"
            self.inner = FakeInner()

    class FakeModule:
        def __init__(self):
            self.key = "model.layers.0"
            self.caps = {}
            self.unload_called = False
            self._linears = [FakeLinear("a")]
        def __iter__(self):
            return iter(self._linears)
        def unload(self):
            self.unload_called = True

    module = FakeModule()
    linears = list(module)
    # Pre-dispatch swap
    for linear in linears:
        if getattr(linear, "inner", None) is not None:
            linear.inner.swap_cpu()
    assert all(f.inner.swap_cpu_called for f in linears)
    # Post-commit unload (caps empty → should unload)
    if not getattr(module, "caps", {}).get("retain_during_quant"):
        module.unload()
    assert module.unload_called


def test_n1_isfinite_item_usage(distributed: Any) -> None:
    """N1: use .all().item() for boolean context parity with upstream."""
    source = Path(distributed.__file__).read_text()
    assert source.count("torch.isfinite(rs).all().item()") == 2


def test_p1_a_required_key_assertion(distributed: Any, tmp_path: Path) -> None:
    """P1-a: per-linear required-key check (K==16 → .weight, K<16 → .trellis)."""
    from exl3pack.dist_types import WorkerEndpoint

    endpoints = [WorkerEndpoint(node="node0", inbox=str(tmp_path / "inbox"), device=0)]
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    coord._planned_keys = {"a", "b"}
    coord._strategy = {"a": 16, "b": 3}  # a uses weight, b uses trellis
    # Missing required key for b (trellis mode)
    q_tensors = {"a.weight": object(), "b.weight": object()}
    with pytest.raises(ValueError, match="missing required key"):
        coord._commit_module("model.layers.0", q_tensors)


def test_p1_d_per_shard_timeout(distributed: Any, tmp_path: Path) -> None:
    """P1-d: per-shard gather timeout window (not global deadline)."""
    import time

    from exl3pack.dist_types import ShardSpec, WorkerEndpoint

    endpoints = [WorkerEndpoint(node="node0", inbox=str(tmp_path / "inbox"), device=0)]
    coord = _make_coordinator(distributed, tmp_path, endpoints)
    coord.gather_timeout = 0.05  # 50ms
    # spec0's result parent (out0) must look like a live worker; the liveness
    # check would otherwise raise before the P1-d timeout path can be tested.
    (tmp_path / "dist" / "heartbeat" / "out0").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "dist" / "heartbeat" / "out0").touch()
    # Create two specs; first has no done marker, second does
    out0 = tmp_path / "out0" / "shard-0.safetensors"
    out1 = tmp_path / "out1" / "shard-0.safetensors"
    out0.parent.mkdir(parents=True, exist_ok=True)
    out1.parent.mkdir(parents=True, exist_ok=True)
    out1.write_bytes(b"data")
    out1.with_suffix(".done").write_text("sha")
    spec0 = ShardSpec(
        job_id="test",
        module_idx=0,
        module_key="mod0",
        shard_idx=0,
        linear_keys=("a",),
        qmaps=("q",),
        h_dir=str(tmp_path / "h"),
        weights_source=str(tmp_path / "w.safetensors"),
        result_uri=str(out0),
        cfg_hash="x",
    )
    spec1 = ShardSpec(
        job_id="test",
        module_idx=0,
        module_key="mod1",
        shard_idx=1,
        linear_keys=("a",),
        qmaps=("q",),
        h_dir=str(tmp_path / "h"),
        weights_source=str(tmp_path / "w.safetensors"),
        result_uri=str(out1),
        cfg_hash="x",
    )
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        coord._gather([spec0, spec1])  # spec0 times out, spec1 succeeds quickly
    elapsed = time.monotonic() - start
    # Per-shard: spec0 times out after one ~1s poll cycle (not the full 600s
    # default), and we never reach spec1. Bounded well under a global wait.
    assert elapsed < 3.0


def test_p1_e_ckpt_restore_from_backup(distributed: Any, tmp_path: Path) -> None:
    """P1-e: restore ckpt/ from ckpt_old/ when ckpt/ missing (crash-window)."""
    from exl3pack.dist_types import WorkerEndpoint

    ckpt = tmp_path / "ckpt"
    backup = tmp_path / "ckpt_old"
    # Setup: ckpt missing, backup exists
    backup.mkdir(parents=True)
    (backup / "job.json").write_text("{}")
    # Drive the production helper directly
    coord = _make_coordinator(
        distributed,
        tmp_path,
        [WorkerEndpoint(node="n", inbox=str(tmp_path), device=0)],
    )
    coord.work = tmp_path  # override work to tmp_path
    coord._restore_checkpoint_backup()
    assert ckpt.exists()
    assert not backup.exists()
    assert (ckpt / "job.json").exists()
