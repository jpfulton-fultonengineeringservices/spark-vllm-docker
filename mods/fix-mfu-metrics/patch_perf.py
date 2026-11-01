"""Anchor-count-gated patcher: FES vllm MFU corrections -> installed b12x vLLM.

Applies the FES fork perf.py corrections to the installed vllm tree
(vllm/v1/metrics/perf.py). Idempotent (sentinel line), drift-safe (every
anchor must occur exactly `expected` times or the patch aborts with the
anchor name), and atomic (py_compile on a temp copy; the installed file is
os.replace'd only after the patch compiles; any failure restores the
original bytes).
"""

from __future__ import annotations

import importlib.util
import os
import py_compile
import sys
import tempfile

SENTINEL = "# fulton-mfu-fix: v1"


def build_edits() -> list[tuple[str, str, int, str]]:
    edits: list[tuple[str, str, int, str]] = []

    def e(name: str, anchor: str, expected: int, replacement: str) -> None:
        edits.append((name, anchor, expected, replacement))

    # (a) _QUANT_WEIGHT_BYTE_SIZE additions (fork dbf11e81f + e1f55c271)
    e(
        "quant-dict-tail",
        '    "inc": 0.5,\n    "experts_int8": 1,\n}',
        1,
        '    "inc": 0.5,\n    "experts_int8": 1,\n'
        '    # Additional methods (verified via get_name())\n'
        '    "auto_awq": 0.5,\n'
        '    "auto_gptq": 0.5,\n'
        '    "deepseek_v4_fp8": 1,\n'
        '    "gpt_oss_mxfp4": 0.5,\n'
        '    "modelopt_mixed": 0.5,  # most weights are NVFP4 (0.5 bytes)\n'
        "}",
    )

    # (b) modelopt_mixed per-component helpers (fork e1f55c271, verbatim)
    e(
        "modelopt-helpers",
        "\n\n\n#### Basic Data Types ####",
        1,
        "\n\n\n"
        + load("modelopt_helpers.py.tmpl")
        + "\n\n#### Basic Data Types ####",
    )

    # (d) BaseConfigParser hybrid layer counting
    e(
        "base-config-hybrid-layers",
        "        args.enable_ep = vllm_config.parallel_config.enable_expert_parallel\n\n"
        "        return args",
        1,
        "        args.enable_ep = vllm_config.parallel_config.enable_expert_parallel\n"
        "        # Hybrid models (e.g. Qwen3.5): count full/linear attention layers\n"
        "        # from layer_types (mirrors vLLM core config/model.py handling).\n"
        '        layer_types = getattr(model_config.hf_text_config, "layer_types", None)\n'
        "        if layer_types is not None:\n"
        '            args.num_full_attention_layers = sum(\n'
        '                1 for t in layer_types if t == "full_attention"\n'
        "            )\n"
        '            args.num_linear_attention_layers = sum(\n'
        '                1 for t in layer_types if t == "linear_attention"\n'
        "            )\n\n"
        "        return args",
    )

    # (e) AttentionMetrics hybrid override: field + _layer_counts branch
    e(
        "attention-hybrid-field",
        "    # From SlidingWindowAttentionParser\n"
        "    sliding_window: int | None = Field(None)\n"
        "    num_swa_layers: int = Field(0, ge=0)\n",
        1,
        "    # From SlidingWindowAttentionParser\n"
        "    sliding_window: int | None = Field(None)\n"
        "    num_swa_layers: int = Field(0, ge=0)\n"
        "\n"
        "    # From BaseConfigParser (hybrid models): counts of layer_types entries\n"
        "    num_full_attention_layers: int | None = Field(None)\n",
    )
    e(
        "layer-counts-hybrid",
        "        L = float(self.num_hidden_layers)\n"
        "        L_swa = float(self.num_swa_layers)\n"
        "        L_full = L - L_swa\n",
        1,
        "        if self.num_full_attention_layers is not None:\n"
        "            # Hybrid model (e.g. Qwen3.5): only full-attention layers bill attention.\n"
        "            L = L_full = float(self.num_full_attention_layers)\n"
        "            L_swa = float(self.num_swa_layers)\n"
        "        else:\n"
        "            L = float(self.num_hidden_layers)\n"
        "            L_swa = float(self.num_swa_layers)\n"
        "            L_full = L - L_swa\n",
    )

    # (f) Attention/Ffn quant parsers accept modelopt_mixed
    e(
        "attn-quant-modelopt",
        "        if quant_method in _QUANT_WEIGHT_BYTE_SIZE:\n"
        "            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]\n"
        "        else:\n"
        '            raise InvalidComponent(\n'
        '                f"Unsupported quantization method for attention metrics: {quant_method}"\n'
        "            )\n",
        1,
        '        if quant_method == "modelopt_mixed":\n'
        "            component_sizes = _compute_modelopt_mixed_component_byte_sizes(\n"
        "                cfg, args.weight_byte_size\n"
        "            )\n"
        '            args.weight_byte_size = component_sizes.get(\n'
        '                "attn", args.weight_byte_size\n'
        "            )\n"
        "        elif quant_method in _QUANT_WEIGHT_BYTE_SIZE:\n"
        "            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]\n"
        "        else:\n"
        '            raise InvalidComponent(\n'
        '                f"Unsupported quantization method for attention metrics: {quant_method}"\n'
        "            )\n",
    )
    e(
        "ffn-quant-modelopt",
        "        if quant_method in _QUANT_WEIGHT_BYTE_SIZE:\n"
        "            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]\n"
        "        else:\n"
        '            raise InvalidComponent(\n'
        '                f"Unsupported quantization method for FFN metrics: {quant_method}"\n'
        "            )\n",
        1,
        '        if quant_method == "modelopt_mixed":\n'
        "            component_sizes = _compute_modelopt_mixed_component_byte_sizes(\n"
        "                cfg, args.weight_byte_size\n"
        "            )\n"
        '            args.weight_byte_size = component_sizes.get(\n'
        '                "ffn", args.weight_byte_size\n'
        "            )\n"
        "        elif quant_method in _QUANT_WEIGHT_BYTE_SIZE:\n"
        "            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]\n"
        "        else:\n"
        '            raise InvalidComponent(\n'
        '                f"Unsupported quantization method for FFN metrics: {quant_method}"\n'
        "            )\n",
    )

    # (g) Unembed: add quant parser class + hook into UnembedMetrics chain +
    #     allow float weight_byte_size (fork 48f9cf884).
    e(
        "unembed-quant-parser",
        "class AttentionDetectionParser(Parser):",
        1,
        load("unembed_parser.py.tmpl") + "\n\nclass AttentionDetectionParser(Parser):",
    )
    e(
        "unembed-parser-chain",
        "    @classmethod\n"
        "    def get_parser(cls) -> ParserChain:\n"
        '        """Return the parser chain for UnembedMetrics (base model config only)."""\n'
        "        return ParserChain(\n"
        "            BaseConfigParser(),\n"
        "        )\n",
        1,
        "    @classmethod\n"
        "    def get_parser(cls) -> ParserChain:\n"
        '        """Return the parser chain for UnembedMetrics (base config + quant)."""\n'
        "        return ParserChain(\n"
        "            BaseConfigParser(),\n"
        "            UnembedQuantizationConfigParser(),\n"
        "        )\n",
    )
    e(
        "unembed-float-field",
        "    vocab_size: int = Field(..., gt=0)\n"
        "    weight_byte_size: int = Field(..., gt=0)\n"
        "    activation_byte_size: int = Field(..., gt=0)\n",
        1,
        "    vocab_size: int = Field(..., gt=0)\n"
        "    weight_byte_size: int | float = Field(..., gt=0)\n"
        "    activation_byte_size: int = Field(..., gt=0)\n",
    )

    # (h) FfnMetrics relu2 parity (fork e1f55c271). No-op for SwiGLU models.
    e(
        "ffn-relu2-parser",
        "        args.num_shared_experts = getattr_from_list(\n"
        '            cfg, ["n_shared_experts", "num_shared_experts"], 0\n'
        "        )\n\n"
        "        is_moe = args.num_experts != 0\n",
        1,
        "        args.num_shared_experts = getattr_from_list(\n"
        '            cfg, ["n_shared_experts", "num_shared_experts"], 0\n'
        "        )\n"
        "\n"
        '        # NemotronH non-gated MoE signal: mlp_hidden_act == "relu2"\n'
        '        args.mlp_hidden_act = getattr(cfg, "mlp_hidden_act", None)\n'
        "\n"
        "        is_moe = args.num_experts != 0\n",
    )
    e(
        "ffn-relu2-field",
        "    # From BaseConfigParser, can be overridden InterleaveMoeLayerStep or MoeLayerFreq\n"
        "    num_moe_layers: int = Field(..., ge=0)\n",
        1,
        "    # From BaseConfigParser, can be overridden InterleaveMoeLayerStep or MoeLayerFreq\n"
        "    num_moe_layers: int = Field(..., ge=0)\n"
        "\n"
        "    # From BaseFfnConfigParser\n"
        "    mlp_hidden_act: str | None = Field(None)\n",
    )
    e(
        "ffn-relu2-flops",
        "        flops = {}\n"
        "\n"
        "        # Dense FFN layers (SwiGLU: 3 linear layers: up, gate, down)\n"
        "        if Ld:\n"
        '            flops["dense_ffn"] = 2 * D * 3 * DI * T * Ld\n',
        1,
        "        flops = {}\n"
        "\n"
        "        # Dense FFN layers (SwiGLU: up, gate, down projections)\n"
        '        # For non-gated MoE (mlp_hidden_act == "relu2"), only 2 GEMMs: up + down\n'
        '        num_gemms = 2 if self.mlp_hidden_act == "relu2" else 3\n'
        "        if Ld:\n"
        '            flops["dense_ffn"] = 2 * num_gemms * D * DI * T * Ld\n',
    )
    e(
        "ffn-relu2-flops-routed",
        '            flops["routed_ffn"] = 2 * D * 3 * MI * num_activated_tokens * Lm\n',
        1,
        '            flops["routed_ffn"] = 2 * num_gemms * D * MI * num_activated_tokens * Lm\n',
    )
    e(
        "ffn-relu2-flops-shared",
        '            flops["shared_ffn"] = 2 * D * 3 * MI * S * T * Lm\n',
        1,
        '            flops["shared_ffn"] = 2 * num_gemms * D * MI * S * T * Lm\n',
    )
    e(
        "ffn-relu2-read-weights",
        '            read_bytes["dense_up_gate_weights"] = int(\n'
        "                2 * D * DI * self.weight_byte_size * Ld\n"
        "            )\n",
        1,
        '            read_bytes["dense_up_gate_weights"] = int(\n'
        "                num_gemms * D * DI * self.weight_byte_size * Ld\n"
        "            )\n",
    )
    e(
        "ffn-relu2-read-down-weights",
        '            read_bytes["dense_down_weights"] = int(D * DI * self.weight_byte_size * Ld)\n',
        1,
        '            read_bytes["dense_down_weights"] = int(\n'
        "                num_gemms * D * DI * self.weight_byte_size * Ld\n"
        "            )\n",
    )
    e(
        "ffn-relu2-read-routed",
        '                read_bytes["routed_up_gate_weights"] = int(\n'
        "                    2 * D * MI * num_activated_experts * self.weight_byte_size * Lm\n"
        "                )\n",
        1,
        '                read_bytes["routed_up_gate_weights"] = int(\n'
        "                    num_gemms * D * MI * num_activated_experts * self.weight_byte_size * Lm\n"
        "                )\n",
    )
    e(
        "ffn-relu2-read-routed-down",
        '                read_bytes["routed_down_weights"] = int(\n'
        "                    D * MI * num_activated_experts * self.weight_byte_size * Lm\n"
        "                )\n",
        1,
        '                read_bytes["routed_down_weights"] = int(\n'
        "                    num_gemms * D * MI * num_activated_experts * self.weight_byte_size * Lm\n"
        "                )\n",
    )
    e(
        "ffn-relu2-read-shared",
        '                read_bytes["shared_up_gate_weights"] = int(\n'
        "                    2 * D * MI * S * self.weight_byte_size * Lm\n"
        "                )\n",
        1,
        '                read_bytes["shared_up_gate_weights"] = int(\n'
        "                    num_gemms * D * MI * S * self.weight_byte_size * Lm\n"
        "                )\n",
    )
    e(
        "ffn-relu2-read-shared-down",
        '                read_bytes["shared_down_weights"] = int(\n'
        "                    D * MI * S * self.weight_byte_size * Lm\n"
        "                )\n",
        1,
        '                read_bytes["shared_down_weights"] = int(\n'
        "                    num_gemms * D * MI * S * self.weight_byte_size * Lm\n"
        "                )\n",
    )
    e(
        "ffn-relu2-write",
        "        write_bytes = {}\n"
        "\n"
        "        # Dense FFN layers\n"
        "        if Ld:\n",
        1,
        "        write_bytes = {}\n"
        "\n"
        "        # Dense FFN layers\n"
        '        num_gemms = 2 if self.mlp_hidden_act == "relu2" else 3\n'
        "        if Ld:\n",
    )
    e(
        "ffn-relu2-write-down",
        '            write_bytes["dense_down_output"] = int(\n'
        "                T * D * self.activation_byte_size * Ld\n"
        "            )\n",
        1,
        '            write_bytes["dense_down_output"] = int(\n'
        "                num_gemms * T * D * self.activation_byte_size * Ld\n"
        "            )\n",
    )
    e(
        "ffn-relu2-write-routed",
        '                write_bytes["routed_down_output"] = int(\n'
        "                    num_activated_tokens * D * self.activation_byte_size * Lm\n"
        "                )\n",
        1,
        '                write_bytes["routed_down_output"] = int(\n'
        "                    num_gemms * num_activated_tokens * D * self.activation_byte_size * Lm\n"
        "                )\n",
    )
    e(
        "ffn-relu2-write-shared",
        '                write_bytes["shared_down_output"] = int(\n'
        "                    T * S * D * self.activation_byte_size * Lm\n"
        "                )\n",
        1,
        '                write_bytes["shared_down_output"] = int(\n'
        "                    num_gemms * T * S * D * self.activation_byte_size * Lm\n"
        "                )\n",
    )

    # (i) GdnMetrics component (Qwen3.5 GatedDeltaNet). The b12x registry is
    # auto-populated via ComponentMetrics.__init_subclass__, so defining the
    # class registers the "mamba" key; ModelMetrics iterates the registry.
    e(
        "gdn-component",
        "\n\n#### ModelMetrics ####",
        1,
        "\n\n" + load("gdn_component.py.tmpl") + "\n\n#### ModelMetrics ####",
    )

    return edits


def load(name: str) -> str:
    """Load a sibling .py.tmpl payload verbatim."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, name), "r", encoding="utf-8") as f:
        return f.read()


def main() -> int:
    spec = importlib.util.find_spec("vllm.v1.metrics.perf")
    if spec is None or not spec.origin:
        print("vllm.v1.metrics.perf not importable", file=sys.stderr)
        return 2

    path = spec.origin
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    src = original

    if SENTINEL in src:
        print(f"already applied: {path}")
        return 0

    for name, anchor, expected, replacement in build_edits():
        n = src.count(anchor)
        if n != expected:
            print(
                f"anchor drift: {name} (found {n}, expected {expected})",
                file=sys.stderr,
            )
            return 3
        src = src.replace(anchor, replacement)

    src = SENTINEL + "\n" + src

    # py_compile on a temp copy, then atomic swap.
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".perf-fix-", suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(src)
        try:
            py_compile.compile(tmp, doraise=True)
        except py_compile.PyCompileError as exc:
            print(f"py_compile failed on patched file:\n{exc}", file=sys.stderr)
            return 4
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)

    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
