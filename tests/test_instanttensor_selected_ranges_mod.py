#!/usr/bin/env python3
"""Run the image's selection helper/iterator and 0.2.2 methods on CPU.

Only CUDA/native transport and loader construction are substituted. The native
seam validates physical ranges/mask and reads their bytes, not tensor names.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/instanttensor-selected-ranges"
MOD = ROOT / "mods/instanttensor-selected-ranges/run.sh"


class Reader:
    """Minimal CPU safetensors reader for real on-disk header/payload fixtures."""
    def __init__(self, filename, **kwargs):
        self.filename = filename
        with open(filename, "rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            self.metadata = json.loads(stream.read(size))
            self.start = 8 + size

    def offset_keys(self):
        return sorted(self.metadata, key=lambda name: self.metadata[name]["data_offsets"][0])

    def get_tensor(self, name):
        start, end = self.metadata[name]["data_offsets"]
        with open(self.filename, "rb") as stream:
            stream.seek(self.start + start)
            return torch.frombuffer(bytearray(stream.read(end - start)), dtype=torch.uint8).clone()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def write_checkpoint(path, entries):
    metadata, data = {}, bytearray()
    for name, values, dtype in entries:
        if isinstance(values, torch.Tensor):
            values = values.view(torch.uint8).tolist()
        start = len(data)
        data.extend(values)
        shape = [len(values) // (2 if dtype == "BF16" else 1)]
        metadata[name] = {"dtype": dtype, "shape": shape,
                          "data_offsets": [start, len(data)]}
    header = json.dumps(metadata).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    return str(path)


class NativeTransport:
    def __init__(self):
        self.opens = []

    def open(self, filenames, device, comm, buffer, chunk, concurrency, depth,
             backend, offsets, keep):
        if len(keep) != len(offsets) - len(filenames):
            raise ValueError("tensor_keep must have one entry per tensor")
        payloads = []
        for index, filename in enumerate(filenames):
            positions = [offset for file_index, offset in offsets if file_index == index]
            if len(positions) < 2:
                raise ValueError("missing dense logical range")
            with open(filename, "rb") as stream:
                for start, end in zip(positions, positions[1:]):
                    stream.seek(start)
                    payloads.append(bytearray(stream.read(end - start)))
        if not all(keep):
            raise ValueError("compacted ranges must all be selected")
        self.opens.append((list(filenames), list(offsets), list(keep), buffer, comm))
        return payloads

    def get_dl_tensor(self, handle, index, size):
        payload = handle[index]
        if len(payload) != size:
            raise ValueError("tensor byte interval mismatch")
        return torch.frombuffer(payload, dtype=torch.uint8)


class Progress:
    @staticmethod
    def _get_free_pos():
        return 0

    def __init__(self, **kwargs):
        pass

    def update(self, size):
        pass

    def close(self):
        pass


class SelectedRangesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "vllm/model_executor/model_loader/weight_utils.py"
        self.target.parent.mkdir(parents=True)
        self.original = (FIXTURE / "weight_utils.py").read_text()
        self.target.write_text(self.original)
        self.native = NativeTransport()
        self.finalize_args = []
        self.instances = []
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cuda = patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(synchronize=lambda: None))
        self.cuda.start()
        self.addCleanup(self.cuda.stop)

    def run_mod(self):
        return subprocess.run(["bash", str(MOD)], text=True, capture_output=True,
                              env={**os.environ, "PATH": os.defpath,
                                   "VLLM_SITE_PACKAGES": str(self.root)}, check=False)

    def load_iterator(self, patched=True):
        if patched:
            result = self.run_mod()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        impl = ModuleType("instanttensor._impl")
        impl.__dict__.update(torch=torch, os=os, _C=self.native,
                             recommended_buffer_size_for_tensors=lambda sizes: 2 * max(sizes, default=0),
                             required_buffer_size_for_io=lambda *args: 4,
                             safetensors_to_torch_dtype={"U8": torch.uint8, "BF16": torch.bfloat16},
                             get_tensor_size=lambda shape, dtype: shape[0] * (2 if dtype is torch.bfloat16 else 1))
        exec(compile((FIXTURE / "instanttensor_impl.py").read_text(), "<image instanttensor>", "exec"), impl.__dict__)
        base = impl.safe_open
        owner = self

        class Loader(base):
            def __init__(self, filenames, **kwargs):
                self.filename = sorted(filenames)
                self.ordered_tensor_metadatas, self.tensor_offsets = [], []
                for index, filename in enumerate(self.filename):
                    reader = Reader(filename)
                    items = [(name, reader.metadata[name]) for name in reader.offset_keys()]
                    self.ordered_tensor_metadatas.extend(items)
                    self.tensor_offsets.extend([(index, item["data_offsets"][0] + reader.start)
                                                for _, item in items])
                    self.tensor_offsets.append((index, items[-1][1]["data_offsets"][1] + reader.start))
                self.tensor_keep = [True] * len(self.ordered_tensor_metadatas)
                self.tensor_name_to_index = {name: i for i, (name, _) in enumerate(self.ordered_tensor_metadatas)}
                self.tensor_sizes = [item["data_offsets"][1] - item["data_offsets"][0]
                                     for _, item in self.ordered_tensor_metadatas]
                self.total_tensor_size = sum(self.tensor_sizes)
                self.loader_handle = None
                self.buffer_size = 9999  # Initial all-shard sizing must not be reused.
                self._device_memory_budget = 100000
                self.chunk_size = self.concurrency = self.io_depth = 1
                self.world_size = 4
                self.device_idx, self.device = 0, "cuda:0"
                self.process_group = kwargs["process_group"]
                self.backend = SimpleNamespace(value=0)
                self.copy, self.iterated, self._invalidated = kwargs["copy"], False, False
                owner.instances.append(self)
                if kwargs.get("load_now", True):
                    self._open()

            def _finalize_buffer_size(self, value):
                owner.finalize_args.append(value)
                super()._finalize_buffer_size(value)

            def __exit__(self, *args):
                self._invalidated = True

        package = ModuleType("instanttensor")
        package.safe_open = Loader
        package._impl = impl
        modules = patch.dict(sys.modules, {"instanttensor": package, "instanttensor._impl": impl})
        modules.start()
        self.addCleanup(modules.stop)
        group = SimpleNamespace(_get_backend=lambda device: SimpleNamespace(_comm_ptr=lambda: 42))
        namespace = {"torch": torch, "safe_open": Reader, "ExitStack": ExitStack,
                     "defaultdict": defaultdict, "tqdm": Progress, "_BAR_FORMAT": "",
                     "enable_tqdm": lambda value: False,
                     "logger": SimpleNamespace(info_once=lambda *args: None),
                     "current_platform": SimpleNamespace(is_cuda=lambda: True, current_device=lambda: "cuda:0"),
                     "get_world_group": lambda: SimpleNamespace(world_size=4, device_group=group)}
        exec(compile(self.target.read_text(), str(self.target), "exec"), namespace)
        return namespace["instanttensor_weights_iterator"]

    def checkpoints(self):
        base = write_checkpoint(self.root / "a.safetensors", [
            ("model.a", [1, 2], "U8"), ("model.overlay", [99, 99], "U8"),
            ("other.skip", [77], "U8"), ("model.b", [3, 4], "U8")])
        overlay = write_checkpoint(self.root / "z.safetensors", [
            ("model.overlay", [5, 6], "U8"), ("other.large", list(range(40)), "U8")])
        index = {"model.a": base, "model.b": base, "model.overlay": overlay,
                 "other.skip": base, "other.large": overlay}
        return [base, overlay], index

    def test_actual_api_canonical_overlay_ranges_and_automatic_sizing(self):
        iterator = self.load_iterator()
        files, index = self.checkpoints()
        output = [(name, tensor.tolist()) for name, tensor in iterator(
            files, False, ["model."], indexed_tensor_files=index)]
        self.assertEqual(output, [("model.a", [1, 2]), ("model.b", [3, 4]), ("model.overlay", [5, 6])])
        self.assertEqual(self.finalize_args, [None])
        self.assertEqual(self.native.opens[0][2], [True] * 3)
        self.assertEqual(self.native.opens[0][3:], (4, 42))
        self.assertEqual(self.native.opens[0][0], [files[0], files[0], files[1]])

    def test_bf16_mtp_head_uses_selected_physical_ranges_and_dtype(self):
        iterator = self.load_iterator()
        files, index = self.checkpoints()
        head = torch.tensor([0.5, -1.25, 3.75, 2.0], dtype=torch.bfloat16)
        mtp = write_checkpoint(self.root / "m.safetensors", [
            ("target.unused", [13, 14], "U8"),
            ("mtp.layers.0.weight", head, "BF16"),
            ("mtp.norm.weight", torch.tensor([1.0, 0.25], dtype=torch.bfloat16), "BF16"),
        ])
        files.append(mtp)
        index.update({"target.unused": mtp, "mtp.layers.0.weight": mtp,
                      "mtp.norm.weight": mtp})
        output = list(iterator(files, False, ["mtp."], indexed_tensor_files=index))
        self.assertEqual([name for name, _ in output],
                         ["mtp.layers.0.weight", "mtp.norm.weight"])
        self.assertEqual([tensor.dtype for _, tensor in output],
                         [torch.bfloat16, torch.bfloat16])
        self.assertEqual(output[0][1].float().tolist(), head.float().tolist())
        self.assertEqual(output[1][1].float().tolist(), [1.0, 0.25])
        self.assertEqual(self.native.opens[0][0], [mtp])
        self.assertEqual(self.native.opens[0][2], [True, True])

    def test_explicit_budget_cpu_fallback_and_priority_exclusion(self):
        os.environ["INSTANTTENSOR_BUFFER_SIZE"] = "2"
        iterator = self.load_iterator()
        files, index = self.checkpoints()
        # Oversized selected tensor uses CPU; an even larger unselected tensor
        # must not leave the constructor's adjusted buffer size in effect.
        big = write_checkpoint(self.root / "b.safetensors", [("model.big", [8, 9, 10], "U8")])
        files.append(big)
        index["model.big"] = big
        output = [(name, tensor.tolist()) for name, tensor in iterator(
            files, False, ["model."], indexed_tensor_files=index,
            priority_weight_name_prefixes=["model.a"])]
        self.assertEqual(output, [("model.a", [1, 2]), ("model.b", [3, 4]),
                                  ("model.big", [8, 9, 10]), ("model.overlay", [5, 6])])
        self.assertEqual(self.finalize_args, [2])
        self.assertEqual(self.native.opens[0][2], [True, True])
        self.assertEqual(self.native.opens[0][3], 2)

    def test_no_selected_gpu_tensors_never_open_transport(self):
        os.environ["INSTANTTENSOR_BUFFER_SIZE"] = "1"
        iterator = self.load_iterator()
        files, index = self.checkpoints()
        result = [(name, tensor.tolist()) for name, tensor in iterator(
            files, False, ["model."], indexed_tensor_files=index)]
        self.assertEqual(result, [("model.a", [1, 2]), ("model.b", [3, 4]), ("model.overlay", [5, 6])])
        self.assertEqual(self.native.opens, [])
        self.assertEqual(self.finalize_args, [])
        # The priority-only path has neither GPU tensors nor CPU fallbacks.
        result = list(iterator(files, False, ["model.a"], indexed_tensor_files=index,
                               priority_weight_name_prefixes=["model.a"]))
        self.assertEqual([name for name, _ in result], ["model.a"])
        self.assertEqual(self.native.opens, [])

    def test_unpatched_image_reproduces_missing_api_failure(self):
        iterator = self.load_iterator(patched=False)
        files, index = self.checkpoints()
        with self.assertRaisesRegex(RuntimeError, "_determine_buffer_size"):
            list(iterator(files, False, ["model."], indexed_tensor_files=index))

    def test_method_rename_alone_reproduces_native_mask_failure(self):
        self.target.write_text(self.original.replace("_determine_buffer_size", "_finalize_buffer_size"))
        iterator = self.load_iterator(patched=False)
        files, index = self.checkpoints()
        with self.assertRaisesRegex(ValueError, "tensor_keep must have one entry"):
            list(iterator(files, False, ["model."], indexed_tensor_files=index))

    def test_repeat_application_and_drift_fail_closed(self):
        first = self.run_mod()
        self.assertEqual(first.returncode, 0, first.stderr)
        patched = self.target.read_bytes()
        self.assertEqual(self.run_mod().returncode, 0)
        self.assertEqual(self.target.read_bytes(), patched)
        for source in (self.original.replace('"loader_handle",', '"changed_layout",'),
                       self.original.replace("instant_open._determine_buffer_size(None)", "instant_open._determine_buffer_size(10)"),
                       self.original.replace("instant_open.tensor_sizes = selected_sizes", "instant_open.tensor_sizes = []")):
            self.target.write_text(source)
            result = self.run_mod()
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(self.target.read_text(), source)


if __name__ == "__main__":
    unittest.main()
