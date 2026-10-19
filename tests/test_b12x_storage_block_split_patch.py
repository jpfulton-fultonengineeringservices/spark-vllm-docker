#!/usr/bin/env python3
"""CPU-only checks for the B12X splittable-storage-block patch."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCHER_PATH = PROJECT_DIR / "docker/patch_vllm_b12x_storage_block_split.py"
SPEC = importlib.util.spec_from_file_location("b12x_storage_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


# The two rejected sites, excerpted verbatim from the shipped image's
# vllm/v1/attention/backends/b12x.py. MultipleOf/backend plumbing is absent
# because the patcher only rewrites these text blocks.
UPSTREAM = '''
_B12X_SUPPORTED_PAGE_SIZES = (64, 128)


class B12xPagedAttentionBackend(AttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return list(_B12X_SUPPORTED_PAGE_SIZES)

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or int(block_size) in _B12X_SUPPORTED_PAGE_SIZES


class B12xPagedAttentionImpl(AttentionImpl[B12xPagedMetadata]):
    def __init__(self, num_heads: int) -> None:
        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        spec_config = vllm_config.speculative_config
        default_block_size = int(cache_config.block_size)
        if default_block_size not in _B12X_SUPPORTED_PAGE_SIZES:
            raise ValueError(
                "b12x requires --block-size in "
                f"{_B12X_SUPPORTED_PAGE_SIZES}, got "
                f"{cache_config.block_size}."
            )
'''


def run_patched_source(source):
    """Execute the patched module and return its supports_block_size result."""
    import ast

    tree = ast.parse(source)
    namespace = {"AttentionBackend": object, "MultipleOf": object, "int": int}
    # Drop the class bases we cannot construct; the methods under test are static.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            node.bases = []
            node.keywords = []
    exec(compile(tree, "<patched>", "exec"), namespace)  # noqa: S102 - test fixture
    return namespace["B12xPagedAttentionBackend"].supports_block_size


class B12xStorageBlockPatchTests(unittest.TestCase):
    def test_patch_applies_and_is_idempotent(self):
        patched, message = PATCHER.patch_source(UPSTREAM)
        self.assertNotEqual(patched, UPSTREAM)
        self.assertIn(PATCHER.MARKER, patched)
        self.assertIn("Accepted", message)

        second, message2 = PATCHER.patch_source(patched)
        self.assertEqual(second, patched)
        self.assertIn("already present", message2)

    def test_patch_keeps_kernel_page_set_exact(self):
        patched, _ = PATCHER.patch_source(UPSTREAM)
        self.assertIn("return list(_B12X_SUPPORTED_PAGE_SIZES)", patched)
        self.assertNotIn("MultipleOf(64)", patched)

    def test_supports_block_size_accepts_splittable_storage(self):
        patched, _ = PATCHER.patch_source(UPSTREAM)
        supports = run_patched_source(patched)
        # 1664 = 13 * 128: the hybrid-aligned Qwen3.8-Coder390 TP=2 block.
        self.assertTrue(supports(1664))
        self.assertTrue(supports(2688))
        # Native kernel pages still accepted, and None means "no preference".
        self.assertTrue(supports(64))
        self.assertTrue(supports(128))
        self.assertTrue(supports(None))
        # FlashInfer's quantum-16 alignment lands on 1648 = 16*103 (prime):
        # divisible by neither 64 nor 128, so it must keep failing closed.
        self.assertFalse(supports(1648))
        self.assertFalse(supports(1234))

    def test_unpatched_source_rejects_aligned_block(self):
        supports = run_patched_source(UPSTREAM)
        self.assertFalse(supports(1664))

    def test_impl_guard_message_updated(self):
        patched, _ = PATCHER.patch_source(UPSTREAM)
        self.assertIn("or is a multiple of", patched)
        self.assertIn("default_block_size % page == 0", patched)

    def test_absent_backend_is_skipped(self):
        source = "def other():\n    return 1\n"
        patched, message = PATCHER.patch_source(source)
        self.assertEqual(patched, source)
        self.assertIn("absent", message)

    def test_partial_patch_fails_closed(self):
        # Apply only the first layout by hand: must be rejected, not "fixed".
        only_supports = UPSTREAM.replace(
            PATCHER.SUPPORTS_ANCHOR, PATCHER.SUPPORTS_REPLACEMENT
        )
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.patch_source(only_supports)

    def test_unexpected_layout_fails_closed(self):
        broken = UPSTREAM.replace(
            "if default_block_size not in _B12X_SUPPORTED_PAGE_SIZES:",
            "if default_block_size not in (64, 128, 256):",
        )
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.patch_source(broken)

    def test_cli_patches_file_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "vllm" / PATCHER.TARGET_REL
            target.parent.mkdir(parents=True)
            target.write_text(UPSTREAM)
            import contextlib
            import io
            import sys

            argv = sys.argv
            sys.argv = ["patch", str(root)]
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    PATCHER.main()
            finally:
                sys.argv = argv
            self.assertIn("Accepted", out.getvalue())
            self.assertIn(PATCHER.MARKER, target.read_text())


if __name__ == "__main__":
    unittest.main()
