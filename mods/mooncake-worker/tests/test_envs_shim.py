"""Offline tests for the vllm.envs shim (torch-free)."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from mooncake_worker import envs_shim

SAMPLE = (
    "import os\n"
    "\n"
    "environment_variables: dict[str, object] = {\n"
    '    "VLLM_NIC_SELECTION_VARS": lambda: os.getenv("VLLM_NIC_SELECTION_VARS", ""),\n'
    "}\n"
    "\n"
    "\n"
    "def __getattr__(name: str):\n"
    "    if name in environment_variables:\n"
    "        return environment_variables[name]()\n"
    "    raise AttributeError(name)\n"
)


def _write_sample() -> Path:
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as fh:
        fh.write(SAMPLE)
        return Path(fh.name)


class EnvsShimTests(unittest.TestCase):
    def test_needs_shim_true_on_clean(self):
        path = _write_sample()
        try:
            self.assertTrue(envs_shim.needs_shim(path))
        finally:
            path.unlink(missing_ok=True)

    def test_inject_adds_dict_entry_and_is_idempotent(self):
        path = _write_sample()
        try:
            self.assertTrue(envs_shim.inject(path))
            self.assertFalse(envs_shim.needs_shim(path))
            self.assertFalse(envs_shim.inject(path))
            self.assertIn(envs_shim.TARGET_ENV, path.read_text())
        finally:
            path.unlink(missing_ok=True)

    def test_injected_env_resolves_to_none_via_getattr(self):
        path = _write_sample()
        try:
            envs_shim.inject(path)
            spec = importlib.util.spec_from_file_location("envs_under_test", path)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertIsNone(module.VLLM_PREFIX_CACHE_RETENTION_INTERVAL)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_anchor_fails_closed(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as fh:
            fh.write("import os\nenvironment_variables = {}\n")
            path = Path(fh.name)
        try:
            with self.assertRaises(envs_shim.EnvShimError):
                envs_shim.inject(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
