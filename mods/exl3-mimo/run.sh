#!/bin/bash
set -euo pipefail

# EXL3-for-MiMo mod.
#
# The b12x runtime image (vllm-node-b12x) ships an EXL3 routed-expert path
# (vllm/model_executor/layers/quantization/exl3.py + B12X kernels +
# b12x.moe.checkpoints.exl3) built for the **exl3-v1** container
# (`exl3-manifest.json`). It is specialised for DeepSeek-V4.1 / Kimi-K3. This
# mod relaxes the two fork-side gates that reject MiMo-V2.6:
#
# 1. Exl3Config.from_config required a DeepSeek-shaped non-routed arrangement:
#      - dense_format == "mxfp8"
#      - ignored_layers a superset of {g_proj, f_a_proj, f_b_proj, b_proj,
#        kv_b_proj} and disjoint from {q_proj, k_proj, v_proj}
#    MiMo keeps attention / dense-MLP in its source format; only routed experts
#    are EXL3. Keep the exl3-manifest requirement + fail-closed list check,
#    accept bf16/fp8 (as well as mxfp8) dense formats. "bf16" is what the
#    pack-build "assemble" stage emits: it dequants the source's FP8-block dense
#    weights (the fork's ModelOptMxFp8Config dense path rejects them) to BF16
#    and lists every non-routed linear module in ignored_layers.
#
# 2. Exl3MoEMethod required SiTU experts with beta 4/25. MiMo-V2.6 uses SiLU.
#    The B12X trellis MoE kernels accept nonlinearity="silu"; the SiTU check is
#    a fork-side restriction. The patch allows SiLU or the SiTU(4/25) contract,
#    keeping BF16 + bias-free, and builds the fused-MoE weight plan with
#    nonlinearity="silu".
#
# Container: the runtime image's own b12x ships the exl3-v1 writer
# (b12x.moe._shared.kernels.w4a16.exl3_synth: Exl3LayerPayloads /
# assemble_code_rows / write_exl3_checkpoint) and reader
# (b12x.moe.checkpoints.exl3); producing that container from MiMo weights is
# the pack step (see EXL3_MIMO_PACK.md). NOTE: the newer upstream
# `btx-atoms-v1` generation is NOT present in this image — do not repoint at it.
#
# Each edit is skipped when already applied and aborts if its anchor is
# missing, so a vLLM revision drift fails loudly rather than silently.

PREFIX="[exl3-mimo]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
EXL3="$PYTHON_ROOT/vllm/model_executor/layers/quantization/exl3.py"
MIMO_V2="$PYTHON_ROOT/vllm/model_executor/models/mimo_v2.py"

echo "=== EXL3-for-MiMo mod ==="

if [ ! -f "$EXL3" ]; then
  echo "$PREFIX Missing $EXL3; a vLLM runtime with the EXL3 quant method is required." >&2
  exit 1
fi

if [ ! -f "$MIMO_V2" ]; then
  echo "$PREFIX Missing $MIMO_V2; a vLLM runtime with the MiMo-V2 model is required." >&2
  exit 1
fi

python3 - "$EXL3" "$MIMO_V2" <<'PY'
from pathlib import Path
import sys

exl3_path = Path(sys.argv[1])
mimo_path = Path(sys.argv[2])


def patch(path: Path, edits: list[tuple[str, str]], label: str) -> None:
    text = path.read_text()
    changed = False
    for old, new in edits:
        if new in text:
            continue
        if old not in text:
            raise SystemExit(
                f"[exl3-mimo] {label}: expected anchor not found; "
                "the installed vLLM differs from the EXL3 layout this mod knows."
            )
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text)
        print(f"[exl3-mimo] Patched {label}.")
    else:
        print(f"[exl3-mimo] {label} already patched; skipping.")


patch(
    exl3_path,
    [
        # 1) from_config: accept a model-shaped (non-DeepSeek) dense arrangement.
        (
            "        exl3 = config.get(\"exl3\")\n"
            "        if (\n"
            "            not isinstance(exl3, dict)\n"
            "            or exl3.get(\"manifest\") != EXL3_MANIFEST_FILENAME\n"
            "            or config.get(\"dense_format\") != \"mxfp8\"\n"
            "        ):\n"
            "            raise ValueError(\n"
            "                f\"EXL3 requires an {EXL3_MANIFEST_FILENAME} expert container \"\n"
            "                \"and MXFP8 dense weights\"\n"
            "            )\n"
            "        ignored = config.get(\"ignored_layers\")\n"
            "        if (\n"
            "            not isinstance(ignored, list)\n"
            "            or not all(isinstance(name, str) for name in ignored)\n"
            "            or not {\"g_proj\", \"f_a_proj\", \"f_b_proj\", \"b_proj\", \"kv_b_proj\"}.issubset(\n"
            "                ignored\n"
            "            )\n"
            "            or {\"q_proj\", \"k_proj\", \"v_proj\"}.intersection(ignored)\n"
            "        ):\n"
            "            raise ValueError(\n"
            "                \"EXL3 requires BF16 KDA gates/factors/beta and MLA KV-B, \"\n"
            "                \"with MXFP8 Q/K/V\"\n"
            "            )\n"
            "        return cls(ignored)\n",
            "        exl3 = config.get(\"exl3\")\n"
            "        if (\n"
            "            not isinstance(exl3, dict)\n"
            "            or exl3.get(\"manifest\") != EXL3_MANIFEST_FILENAME\n"
            "            or config.get(\"dense_format\") not in (\"mxfp8\", \"fp8\", \"bf16\")\n"
            "        ):\n"
            "            raise ValueError(\n"
            "                f\"EXL3 requires an {EXL3_MANIFEST_FILENAME} expert container \"\n"
            "                \"and MXFP8/FP8/BF16 dense weights\"\n"
            "            )\n"
            "        # spark-vllm-docker/mods/exl3-mimo: accept a model-shaped\n"
            "        # non-routed arrangement. MiMo keeps attention / dense-MLP in\n"
            "        # its source format; only the routed experts are EXL3, so the\n"
            "        # DeepSeek KDA/MLA ignore-list checks do not apply. Keep the\n"
            "        # manifest requirement and a fail-closed list validation.\n"
            "        ignored = config.get(\"ignored_layers\")\n"
            "        if not isinstance(ignored, list) or not all(\n"
            "            isinstance(name, str) for name in ignored\n"
            "        ):\n"
            "            raise ValueError(\n"
            "                \"EXL3 requires ignored_layers (list of module names) \"\n"
            "                \"for the non-routed dense arrangement\"\n"
            "            )\n"
            "        return cls(ignored)\n",
        ),
        # 2) Exl3MoEMethod: allow SiLU (MiMo) alongside DeepSeek's SiTU(4/25).
        (
            "        if (\n"
            "            moe.activation != MoEActivation.SITU\n"
            "            or moe.activation_situ_beta != 4.0\n"
            "            or moe.activation_situ_linear_beta != 25.0\n"
            "            or moe.in_dtype != torch.bfloat16\n"
            "            or moe.has_bias\n"
            "        ):\n"
            "            raise ValueError(\n"
            "                \"EXL3 experts require bias-free BF16 SiTU experts with beta 4/25\"\n"
            "            )\n",
            "        # spark-vllm-docker/mods/exl3-mimo: MiMo routed experts use SiLU,\n"
            "        # not DeepSeek's SiTU(4/25). B12X trellis MoE kernels accept\n"
            "        # nonlinearity=\"silu\"; the SiTU gate is fork-side only.\n"
            "        _exl3_activation_ok = (\n"
            "            moe.activation == MoEActivation.SILU\n"
            "            or (\n"
            "                moe.activation == MoEActivation.SITU\n"
            "                and moe.activation_situ_beta == 4.0\n"
            "                and moe.activation_situ_linear_beta == 25.0\n"
            "            )\n"
            "        )\n"
            "        if (\n"
            "            not _exl3_activation_ok\n"
            "            or moe.in_dtype != torch.bfloat16\n"
            "            or moe.has_bias\n"
            "        ):\n"
            "            raise ValueError(\n"
            "                \"EXL3 experts require bias-free BF16 SiLU or SiTU(4/25) experts\"\n"
            "            )\n",
        ),
        # 3) fused-MoE weight plan: build the trellis kernel for SiLU.
        (
            "            activation=fused_moe.ActivationSpec(\n"
            "                mode=\"a16\",\n"
            "                nonlinearity=\"situ\",\n"
            "                io_dtype=torch.bfloat16,\n"
            "                rotation_dtype=torch.float16,\n"
            "            ),\n",
            "            activation=fused_moe.ActivationSpec(\n"
            "                mode=\"a16\",\n"
            "                # spark-vllm-docker/mods/exl3-mimo: MiMo routed experts\n"
            "                # are SiLU (DeepSeek-V4.1 uses situ).\n"
            "                nonlinearity=\"silu\",\n"
            "                io_dtype=torch.bfloat16,\n"
            "                rotation_dtype=torch.float16,\n"
            "            ),\n",
        ),
    ],
    "vllm/model_executor/layers/quantization/exl3.py",
)

# spark-vllm-docker/mods/exl3-mimo: the EXL3 serving checkpoint stores the
# fused attention qkv_proj in BF16 (dequantized from the source FP8-block
# dense weights) while preserving the source's interleaved [Q_i|K_i|V_i]
# checkpoint layout. The fork's FP8 loader shards such a tensor through
# `_shard_fp8_qkv_proj` (interleaved-aware), but a BF16 tensor falls through
# to the generic fused-qkv loader, which assumes a SIMPLE [Q|K|V] layout and
# selects the wrong rows at TP>1 -> corrupted attention -> garbage serving.
# Restore the interleaved-aware sharding for the BF16 fused qkv_proj.
patch(
    mimo_path,
    [
        # 1) Intercept BF16 fused qkv_proj before the generic fused loader.
        (
            "            ):\n"
            "                continue\n"
            "            stacked_matched = False\n",
            "            ):\n"
            "                continue\n"
            "            if self._try_load_bf16_qkv_proj(\n"
            "                name,\n"
            "                loaded_weight,\n"
            "                params_dict,\n"
            "                loaded_params,\n"
            "                tp_rank,\n"
            "                tp_size,\n"
            "            ):\n"
            "                continue\n"
            "            stacked_matched = False\n",
        ),
        # 2) Add the de-interleave method before _try_load_fp8_qkv_proj.
        (
            "    def _try_load_fp8_qkv_proj(\n",
            "    def _try_load_bf16_qkv_proj(\n"
            "        self,\n"
            "        name: str,\n"
            "        tensor: torch.Tensor,\n"
            "        params_dict: dict[str, torch.nn.Parameter],\n"
            "        loaded_params: set[str],\n"
            "        tp_rank: int,\n"
            "        tp_size: int,\n"
            "    ) -> bool:\n"
            "        \"\"\"De-interleave a BF16 fused qkv_proj for TP sharding.\n"
            "\n"
            "        The EXL3 serving checkpoint stores qkv_proj in BF16 while\n"
            "        preserving the source's interleaved [Q_i|K_i|V_i] layout;\n"
            "        the generic fused loader assumes a simple [Q|K|V] and would\n"
            "        select wrong rows at TP>1. Mirror _shard_fp8_qkv_proj's\n"
            "        de-interleave without the FP8 dequant/requant steps.\n"
            "        \"\"\"\n"
            "        if not (\n"
            "            name.endswith(\"qkv_proj.weight\")\n"
            "            and tensor.dtype == torch.bfloat16\n"
            "        ):\n"
            "            return False\n"
            "        if is_pp_missing_parameter(name, self):\n"
            "            return True\n"
            "        prefix = name.rsplit(\".\", 1)[0]\n"
            "        attn = self.get_submodule(prefix.rsplit(\".\", 1)[0])\n"
            "        ckpt_tp = self.config.num_key_value_heads\n"
            "        gpr = ckpt_tp // tp_size\n"
            "        if gpr == 1:\n"
            "            w_rank = tensor.chunk(tp_size, dim=0)[tp_rank]\n"
            "        else:\n"
            "            qg = (attn.total_num_heads // ckpt_tp) * attn.head_dim\n"
            "            kg = (attn.total_num_kv_heads // ckpt_tp) * attn.head_dim\n"
            "            vg = (\n"
            "                attn.total_num_kv_heads // ckpt_tp\n"
            "            ) * attn.v_head_dim\n"
            "            rpg = qg + kg + vg\n"
            "            qs, ks, vs = [], [], []\n"
            "            for g in range(tp_rank * gpr, (tp_rank + 1) * gpr):\n"
            "                off = g * rpg\n"
            "                qs.append(tensor[off : off + qg])\n"
            "                ks.append(tensor[off + qg : off + qg + kg])\n"
            "                vs.append(tensor[off + qg + kg : off + rpg])\n"
            "            w_rank = torch.cat(\n"
            "                [torch.cat(qs), torch.cat(ks), torch.cat(vs)],\n"
            "                dim=0,\n"
            "            ).contiguous()\n"
            "        param = params_dict[name]\n"
            "        if w_rank.shape[0] > param.shape[0]:\n"
            "            w_rank = w_rank[: param.shape[0]]\n"
            "        default_weight_loader(param, w_rank)\n"
            "        loaded_params.add(name)\n"
            "        return True\n"
            "\n"
            "    def _try_load_fp8_qkv_proj(\n",
        ),
    ],
    "vllm/model_executor/models/mimo_v2.py",
)
PY

echo "=====> Exl3Config.from_config accepts a MiMo-shaped non-routed arrangement"
echo "=====> Exl3MoEMethod admits SiLU experts"
echo "=====> EXL3 fused-MoE weight plan built with nonlinearity=silu"
