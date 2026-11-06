#!/usr/bin/env python3
"""Exercise the actual image rotary classes on CPU after running the mod."""

import ast
import __future__
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/mrope-yarn"
MOD = ROOT / "mods/qwen3.8-coder390-mrope-yarn/run.sh"
TARGET = Path("vllm/model_executor/layers/rotary_embedding/mrope.py")


def load_class(source, name, namespace):
    """Compile the image's unmodified class, leaving unrelated imports out."""
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, f"<image {name}>", "exec", flags=__future__.annotations.compiler_flag), namespace)  # noqa: S102
    return namespace[name]


class CustomOp(torch.nn.Module):
    @staticmethod
    def enabled():
        return False


class ApplyRotaryEmb:
    def __init__(self, **kwargs):
        pass


def load_image_classes(mrope_source):
    namespace = {"torch": torch, "CustomOp": CustomOp, "ApplyRotaryEmb": ApplyRotaryEmb}
    exec(compile((FIXTURES / "yarn_helpers.py").read_text(), "<image helpers>", "exec"), namespace)  # noqa: S102
    base = (FIXTURES / "base.py").read_text()
    load_class(base, "RotaryEmbeddingBase", namespace)
    load_class(base, "RotaryEmbedding", namespace)
    load_class((FIXTURES / "yarn_scaling_rope.py").read_text(), "YaRNScalingRotaryEmbedding", namespace)
    load_class(mrope_source, "MRotaryEmbedding", namespace)
    return namespace["MRotaryEmbedding"], namespace["YaRNScalingRotaryEmbedding"]


class MropeYarnTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / TARGET
        self.path.parent.mkdir(parents=True)
        self.path.write_text((FIXTURES / "mrope.py").read_text())

    def run_mod(self):
        return subprocess.run(
            ["bash", str(MOD)], capture_output=True, text=True,
            env={**os.environ, "VLLM_SITE_PACKAGES": str(self.root)}, check=False,
        )

    def test_yarn_matches_native_and_unscaled_cache_is_unchanged(self):
        first = self.run_mod()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        patched = self.path.read_text()
        second = self.run_mod()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(self.path.read_text(), patched)
        mrope, yarn = load_image_classes(patched)

        # Compare the real image YaRN class with the native context, not a
        # reimplementation of the formula. Free its large cache before MRoPE.
        native = yarn(32, 32, 262144, 10000000.0, True, 4.0, torch.float32)
        native_freq = native._compute_inv_freq(4.0)
        del native
        scaled = mrope(32, 32, 262144, 10000000.0, True, torch.float32,
                       mrope_section=[6, 5, 5], scaling_factor=4.0)
        self.assertIsInstance(scaled, mrope)
        self.assertEqual(scaled.max_position_embeddings, 262144)
        self.assertEqual(tuple(scaled.cos_sin_cache.shape), (1048576, 32))
        torch.testing.assert_close(scaled._compute_inv_freq(4.0), native_freq, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(scaled.cos_sin_cache[[262143, 262144, 1048575]]).all().item())
        del scaled

        unscaled = mrope(32, 32, 2048, 10000000.0, True, torch.float32,
                         mrope_section=[6, 5, 5])
        self.assertEqual(unscaled.max_position_embeddings, 8192)
        self.assertEqual(tuple(unscaled.cos_sin_cache.shape), (8192, 32))

    def test_anchor_drift_fails_closed(self):
        original = self.path.read_text()
        self.path.write_text(original.replace("self.cache_max_position_num = max_position_embeddings * 4",
                                              "self.cache_max_position_num = max_position_embeddings * 8"))
        before = self.path.read_text()
        result = self.run_mod()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incompatible", result.stderr)
        self.assertEqual(self.path.read_text(), before)


if __name__ == "__main__":
    unittest.main()
