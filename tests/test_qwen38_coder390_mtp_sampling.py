#!/usr/bin/env python3
"""MTP sampling patch behavior; opt in to real vLLM only in a disposable image."""

import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mods/qwen3.8-coder390-mtp/run.sh"
MODULES = (
    "vllm.entrypoints.openai.chat_completion.protocol",
    "vllm.entrypoints.openai.completion.protocol",
)

# The resolution and forwarding seam is copied from image
# sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02.
# The small stand-in executes the patched method without requiring vLLM on the
# test host. The opt-in test below also exercises the real runtime classes.
PROTOCOL = '''from types import SimpleNamespace

class SamplingParams:
    @staticmethod
    def from_optional(**kwargs):
        return SimpleNamespace(**kwargs)

class Request:
    _DEFAULT_SAMPLING_PARAMS = {
        "repetition_penalty": 1.0, "temperature": 1.0, "top_p": 1.0,
        "top_k": 0, "min_p": 0.0,
    }
    def __init__(self, **values):
        self.n = 1
        self.presence_penalty = 0.0
        self.frequency_penalty = 0.0
        self.repetition_penalty = None
        self.temperature = None
        self.watermarking = True
        self.top_p = None
        self.top_k = None
        self.min_p = None
        self.thinking_token_budget = None
        self.max_tokens = 16
        self.__dict__.update(values)

    def to_sampling_params(
        self,
        max_tokens: int,
        default_sampling_params: dict | None = None,
    ):
        if default_sampling_params is None:
            default_sampling_params = {}
        if (repetition_penalty := self.repetition_penalty) is None:
            repetition_penalty = default_sampling_params.get(
                "repetition_penalty",
                self._DEFAULT_SAMPLING_PARAMS["repetition_penalty"],
            )
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
            )
        if (top_p := self.top_p) is None:
            top_p = default_sampling_params.get(
                "top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"]
            )
        if (top_k := self.top_k) is None:
            top_k = default_sampling_params.get(
                "top_k", self._DEFAULT_SAMPLING_PARAMS["top_k"]
            )
        if (min_p := self.min_p) is None:
            min_p = default_sampling_params.get(
                "min_p", self._DEFAULT_SAMPLING_PARAMS["min_p"]
            )

        # Merge server-default stop_token_ids (irrelevant to this fixture)
        return SamplingParams.from_optional(
            n=self.n,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            watermarking=self.watermarking,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            thinking_token_budget=self.thinking_token_budget,
        )
'''


def checkpoint(root):
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "text_config": {"mtp_num_hidden_layers": 1},
        "quantization_config": {"ignore": ["mtp*"]},
    }))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"mtp.weight": "model.safetensors"},
    }))
    header = json.dumps({"mtp.weight": {"dtype": "BF16"}}).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)


def run_mod(model, pythonpath=None):
    env = os.environ.copy()
    env["FES_WEIGHTS_DIR"] = str(model)
    env.pop("FES_HUB_MODEL", None)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(["bash", str(MOD)], env=env, text=True, capture_output=True)


def assert_policy(test, request, max_tokens, budget, defaults):
    params = request.to_sampling_params(max_tokens, defaults)
    test.assertEqual(params.temperature, 1.0)
    test.assertEqual(params.top_p, 0.95)
    test.assertEqual(params.top_k, 20)
    test.assertEqual(params.repetition_penalty, 1.10)
    test.assertEqual(params.thinking_token_budget, budget)
    return params


class SamplingPatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        checkpoint(self.model)
        self.paths = []
        for module in MODULES:
            path = self.root.joinpath(*module.split(".")).with_suffix(".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(PROTOCOL)
            parent = path.parent
            while parent != self.root:
                (parent / "__init__.py").touch()
                parent = parent.parent
            self.paths.append(path)

    def run_patch(self):
        return run_mod(self.model, self.root)

    def load_request(self, path):
        spec = importlib.util.spec_from_file_location("protocol_fixture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Request

    def test_policy_flows_to_returned_parameters_and_is_idempotent(self):
        result = self.run_patch()
        self.assertEqual(result.returncode, 0, result.stderr)
        first = [path.read_text() for path in self.paths]
        for path in self.paths:
            request_type = self.load_request(path)
            defaults = {"temperature": 0.4, "top_p": 0.6, "top_k": 6,
                        "repetition_penalty": 1.4, "min_p": 0.12}
            for token_cap, expected in ((None, 32768), (340, 0), (341, 255),
                                        (512, 384), (1024, 768), (1025, 769),
                                        (2000, 1744), (100000, 32768)):
                with self.subTest(protocol=path, max_tokens=token_cap):
                    req = request_type(max_tokens=17, thinking_token_budget=9)
                    params = assert_policy(self, req, token_cap, expected, defaults)
                    self.assertEqual(params.max_tokens, token_cap)
                    self.assertEqual(params.min_p, 0.12)
            req = request_type(temperature=0, top_p=0.8, top_k=7,
                               repetition_penalty=1.3, min_p=0.24,
                               thinking_token_budget=12)
            self.assertEqual(assert_policy(self, req, 512, 384, defaults).min_p, 0.24)
        result = self.run_patch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([path.read_text() for path in self.paths], first)

    def test_drift_fails_before_modifying_either_protocol(self):
        for before, after in (
            ('if (min_p := self.min_p) is None:', 'if self.min_p is None:'),
            ('temperature=temperature,', 'temperature=self.temperature,'),
            ('thinking_token_budget=self.thinking_token_budget,', 'thinking_token_budget=None,'),
        ):
            with self.subTest(changed=before):
                self.paths[1].write_text(PROTOCOL.replace(before, after))
                original = [path.read_text() for path in self.paths]
                result = self.run_patch()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("FAIL: sampling-envelope", result.stderr)
                self.assertEqual([path.read_text() for path in self.paths], original)

    def test_missing_protocol_is_an_error(self):
        self.paths[1].unlink()
        original = self.paths[0].read_text()
        result = self.run_patch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot locate", result.stderr)
        self.assertEqual(self.paths[0].read_text(), original)

    def test_invalid_or_legacy_protocol_does_not_partially_patch(self):
        for suffix in (
            "\ncontinue\n",
            "\n# [qwen3.8-coder390-dflash] Force publisher sampling envelope\n",
        ):
            with self.subTest(suffix=suffix):
                self.paths[1].write_text(PROTOCOL + suffix)
                original = [path.read_text() for path in self.paths]
                result = self.run_patch()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("FAIL: sampling-envelope", result.stderr)
                self.assertEqual([path.read_text() for path in self.paths], original)


@unittest.skipUnless(os.environ.get("QWEN_MTP_REAL_VLLM") == "1",
                     "requires a disposable writable vLLM image")
class RealRuntimeSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        model = Path(cls.temp.name) / "model"
        checkpoint(model)
        for _ in range(2):
            result = run_mod(model)
            if result.returncode:
                raise AssertionError(result.stderr)
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest
        cls.requests = (
            (ChatCompletionRequest, {"messages": [{"role": "user", "content": "Hi"}]}),
            (CompletionRequest, {"prompt": "Hi"}),
        )

    def test_real_request_methods_forward_policy(self):
        defaults = {"temperature": 0.4, "top_p": 0.6, "top_k": 6,
                    "repetition_penalty": 1.4, "min_p": 0.12}
        for request_type, required in self.requests:
            for cap, budget in ((None, 32768), (340, 0), (341, 255),
                                (512, 384), (1024, 768), (1025, 769),
                                (2000, 1744), (100000, 32768)):
                with self.subTest(request=request_type.__name__, cap=cap):
                    req = request_type(**required, max_tokens=17, thinking_token_budget=9)
                    params = assert_policy(self, req, cap, budget, defaults)
                    self.assertEqual(params.max_tokens, cap)
                    self.assertEqual(params.min_p, 0.12)
            req = request_type(**required, temperature=0, top_p=0.8, top_k=7,
                               repetition_penalty=1.3, min_p=0.24,
                               thinking_token_budget=12)
            self.assertEqual(assert_policy(self, req, 512, 384, defaults).min_p, 0.24)


if __name__ == "__main__":
    unittest.main()
