#!/usr/bin/env python3
"""Exercise speculative-method selection and MTP safety without GPUs or weights."""

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


MOD = Path(__file__).resolve().parents[1] / "mods/fix-explicit-spec-method/run.sh"

# The method-selection section from the installed vLLM artifact. The surrounding
# stub supplies only the configuration objects needed by this section.
SOURCE = '''\
from typing import Literal, get_args

MTPModelTypes = Literal["qwen3_5_mtp", "deepseek_mtp", "step3p5_mtp", "inkling_mtp"]
_QWEN3_OMNI_DSPARK_ARCHITECTURE = "Qwen3OmniDSparkModel"

class Logger:
    def __init__(self):
        self.warnings = []
    def warning(self, message):
        self.warnings.append(message)

logger = Logger()

class SpeculativeConfig:
    def select(self):
        if True:
            if True:
                # Automatically detect the method
                if self.method in ("eagle", "eagle3", "dflash", "dspark"):
                    pass
                # examples:
                # yuhuili/EAGLE-LLaMA3-Instruct-8B
                # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
                # AngelSlim/Qwen3-8B_eagle3
                # deepseek-ai/dspark_qwen3_8b_block7
                elif "eagle-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif "eagle3" in self.draft_model_config.model.lower():
                    self.method = "eagle3"
                elif (
                    "dflash" in self.draft_model_config.model.lower()
                    or "MuseGlimmerAssistantModel"
                    in self.draft_model_config.architectures
                ):
                    self.method = "dflash"
                elif (
                    "dspark" in self.draft_model_config.model.lower()
                    or "Qwen3DSparkModel" in self.draft_model_config.architectures
                    or _QWEN3_OMNI_DSPARK_ARCHITECTURE
                    in self.draft_model_config.architectures
                    or "Gemma4DSparkModel" in self.draft_model_config.architectures
                    or (
                        "DSparkDraftModel" in self.draft_model_config.architectures
                        and self.draft_model_config.hf_config.model_type == "qwen3"
                    )
                ):
                    self.method = "dspark"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                    self.method = "mlp_speculator"
                elif self.draft_model_config.hf_config.model_type in get_args(
                    MTPModelTypes
                ):
                    self.method = "mtp"
                    if (
                        self.target_model_config is not None
                        and self.target_model_config.hf_config.model_type
                        == "deepseek_v41"
                    ):
                        raise ValueError(
                            "DeepSeek V4.1 has no classic-MTP draft: its "
                            "checkpoints ship DSpark stages under mtp.* "
                            "(main_proj/markov_head/confidence_head) and carry "
                            "no e_proj/h_proj/enorm/hnorm/hc_head weights. Use "
                            "speculative method 'dspark' instead of 'mtp'."
                        )
                    if (
                        self.num_speculative_tokens > 1
                        and self.draft_model_config.hf_config.model_type
                        not in ("step3p5_mtp", "inkling_mtp")
                    ):
                        logger.warning(
                            "Enabling num_speculative_tokens > 1 will run "
                            "multiple times of forward on same MTP layer"
                            ",which may result in lower acceptance rate"
                        )
                elif self.method == "draft_model":
                    pass
                else:
                    raise NotImplementedError(
                        f"Unsupported speculative method: '{self.method}'"
                    )

        return self.method
'''


class ExplicitSpecMethodTests(unittest.TestCase):
    def patch_source(self, revision):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        package = root / "vllm/config"
        package.mkdir(parents=True)
        (root / "vllm/__init__.py").touch()
        (package / "__init__.py").touch()
        path = package / "speculative.py"
        original = 'if self.method in ("eagle", "eagle3", "dflash", "dspark"):'
        variants = {
            "pristine": original,
            "none-guard": 'if self.method is None or self.method in ("eagle", "eagle3", "dflash", "dspark"):',
            "tuple-only": 'if self.method in ("eagle", "eagle3", "dflash", "dspark", "mtp"):',
        }
        path.write_text(SOURCE.replace(original, variants[revision]))
        for _ in range(2):
            result = subprocess.run(
                ["bash", str(MOD)], capture_output=True, text=True,
                env={**os.environ, "FIX_EXPLICIT_SPEC_METHOD": "1", "PYTHONPATH": str(root)},
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        spec = importlib.util.spec_from_file_location("speculative_fixture", path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def select(self, module, method, model, model_type="qwen3_5_mtp", target_type="qwen3_5", tokens=3):
        config = module.SpeculativeConfig()
        config.method = method
        config.draft_model_config = SimpleNamespace(
            model=model, architectures=[], hf_config=SimpleNamespace(model_type=model_type)
        )
        config.target_model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type=target_type))
        config.num_speculative_tokens = tokens
        module.logger.warnings.clear()
        result = config.select()
        return result, module.logger.warnings[:]

    def test_method_selection_and_native_mtp_safety(self):
        for revision in ("pristine", "none-guard", "tuple-only"):
            with self.subTest(revision=revision):
                module = self.patch_source(revision)
                for name in ("Qwen-DFlash2", "EAGLE-example", "eagle3-example", "dspark-example"):
                    method, warnings = self.select(module, "mtp", name)
                    self.assertEqual(method, "mtp")
                    self.assertEqual(len(warnings), 1)
                    self.assertIn("lower acceptance rate", warnings[0])
                method, warnings = self.select(module, None, "neutral")
                self.assertEqual(method, "mtp")
                self.assertEqual(len(warnings), 1)
                self.assertEqual(self.select(module, None, "DFlash2")[0], "dflash")
                self.assertEqual(self.select(module, "dflash", "neutral")[0], "dflash")
                self.assertEqual(self.select(module, "mtp", "DFlash2", model_type="unknown")[0], "dflash")
                with self.assertRaisesRegex(NotImplementedError, "Unsupported speculative method"):
                    self.select(module, "mtp", "neutral", model_type="unknown")
                for explicit, name in (("mtp", "DFlash2"), (None, "neutral")):
                    with self.assertRaisesRegex(ValueError, "DeepSeek V4.1"):
                        self.select(module, explicit, name, target_type="deepseek_v41")
                self.assertEqual(self.select(module, "mtp", "DFlash2", tokens=1)[1], [])
                self.assertEqual(self.select(module, "mtp", "DFlash2", model_type="step3p5_mtp")[1], [])


if __name__ == "__main__":
    unittest.main()
