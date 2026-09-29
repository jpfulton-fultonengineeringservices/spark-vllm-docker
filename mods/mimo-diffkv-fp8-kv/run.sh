#!/bin/bash
set -euo pipefail

# MiMo-V2 fp8 KV cache enablement mod.
#
# Two upstream gaps (verified against vllm main 2026-09-22):
# 1. mimo_v2.py never passes cache_config to Attention(), so Attention()
#    resolves kv_cache_dtype to "auto" and --kv-cache-dtype fp8 is silently
#    ignored on all 48 target layers. Passing it has a trap: Attention() then
#    falls back to cache_config.sliding_window (128) for layers without a
#    per-layer window, so the full-attention layers get a copy with the window
#    cleared. Without that, long generations loop once the context passes the
#    window (reproduced 2026-09-23 at TP=2 and PP=3).
# 2. triton_attn_diffkv.py — the backend sm_121 selects for the 192/128 K/V
#    head dims — rejects quantized KV outright and does not view the cache as
#    fp8 on read.
# 3. attention.py — with --kv-cache-dtype-skip-layers the platform estimates
#    skip_page_size_padded from the model-level KV head count (4), but the SWA
#    layers have 8, so the shared page comes out smaller than even the
#    smallest B12X kernel block page (64) and the engine asserts in
#    resolve_kv_cache_layout. The patch floors the SWA spec's padded page at
#    its chosen kernel block's natural page (needed only with skip-layers on
#    coarse kernel block sizes such as B12X's 64/128).
# 4. qwen3_dflash.py — DFlashAttention.get_kv_cache_spec consumes
#    skip_page_size_padded directly (bypassing the attention.py chooser), so
#    the same underestimate leaves the drafter's padded page below its
#    natural 128 KiB page. Floored the same way.
#
# Same fix as tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark patches 01+03, verified
# on GB10 (fp8 KV pool measured at 12.6 GiB, needle tests at 250K). Unlike his
# patch, non-e4m3 quantized caches stay rejected: the fp8 view would corrupt
# them. Each file is patched independently and skipped when already fixed
# (assumptions.md).

PREFIX="[mimo-diffkv-fp8-kv]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
MIMO="$PYTHON_ROOT/vllm/model_executor/models/mimo_v2.py"
DIFFKV="$PYTHON_ROOT/vllm/v1/attention/backends/triton_attn_diffkv.py"
ATTENTION="$PYTHON_ROOT/vllm/model_executor/layers/attention/attention.py"
DFLASH="$PYTHON_ROOT/vllm/model_executor/models/qwen3_dflash.py"

echo "=== MiMo fp8 KV cache mod ==="

for f in "$MIMO" "$DIFFKV" "$ATTENTION" "$DFLASH"; do
  if [ ! -f "$f" ]; then
    echo "$PREFIX Missing $f; a newer image is required." >&2
    exit 1
  fi
done

python3 - "$MIMO" "$DIFFKV" "$ATTENTION" "$DFLASH" <<'PY'
from pathlib import Path
import sys

mimo_path, diffkv_path, attention_path, dflash_path = (Path(p) for p in sys.argv[1:5])


def patch(path: Path, edits: list[tuple[str, str]], label: str) -> None:
    text = path.read_text()
    changed = False
    for old, new in edits:
        if new in text:
            continue
        if old not in text:
            raise SystemExit(
                f"[mimo-diffkv-fp8-kv] {label}: expected anchor not found; "
                "the installed vLLM differs from the layout this mod knows."
            )
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text)
        print(f"[mimo-diffkv-fp8-kv] Patched {label}.")
    else:
        print(f"[mimo-diffkv-fp8-kv] {label} already patched; skipping.")


patch(
    mimo_path,
    [
        (
            "        self.attn = Attention(\n",
            "        if cache_config is None:\n"
            "            # spark-vllm-docker/mods/mimo-diffkv-fp8-kv: no caller passes\n"
            "            # cache_config, and Attention() then resolves kv_cache_dtype\n"
            "            # to \"auto\", silently ignoring --kv-cache-dtype.\n"
            "            cache_config = get_current_vllm_config().cache_config\n"
            "        if sliding_window is None and cache_config.sliding_window is not None:\n"
            "            # mimo-diffkv-fp8-kv: Attention() falls back to\n"
            "            # cache_config.sliding_window (the model's 128) when\n"
            "            # per_layer_sliding_window is None, which would cap the\n"
            "            # full-attention layers at a 128-token window. Give them a\n"
            "            # copy without one (tonyd2wild patch 01).\n"
            "            import copy\n"
            "            cache_config = copy.copy(cache_config)\n"
            "            cache_config.sliding_window = None\n"
            "        self.attn = Attention(\n",
        ),
    ],
    "mimo_v2.py",
)

patch(
    diffkv_path,
    [
        (
            "    # No FP8 / int8 KV cache for the DiffKV path yet; require fp16/bf16/fp32.\n"
            "    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n"
            '        "auto",\n'
            '        "bfloat16",\n'
            "    ]\n",
            "    # fp8 (e4m3) KV: the reshape kernel quantizes on write; the attention\n"
            "    # kernel upcasts K/V to bf16 on load. KV scales are not applied on read,\n"
            "    # which is exact for unit-scale checkpoints (MiMo-V2).\n"
            "    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n"
            '        "auto",\n'
            '        "bfloat16",\n'
            '        "fp8",\n'
            '        "fp8_e4m3",\n'
            "    ]\n",
        ),
        (
            "        if is_quantized_kv_cache(self.kv_cache_dtype):\n"
            "            raise NotImplementedError(\n"
            '                "TritonAttentionDiffKVBackend does not yet support quantized "\n'
            '                f"KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."\n'
            "            )\n",
            "        if is_quantized_kv_cache(self.kv_cache_dtype) and (\n"
            '            self.kv_cache_dtype not in ("fp8", "fp8_e4m3")\n'
            "        ):\n"
            "            raise NotImplementedError(\n"
            '                "TritonAttentionDiffKVBackend only supports fp8_e4m3 KV "\n'
            '                f"among quantized caches (got {self.kv_cache_dtype!r})."\n'
            "            )\n",
        ),
        (
            "        # Triton DiffKV kernels consume (B, N, H, D) cache views.\n"
            "        kv_cache = kv_cache.transpose(1, 2)\n"
            "        key_cache = kv_cache[..., :head_size_qk]\n",
            "        # Triton DiffKV kernels consume (B, N, H, D) cache views.\n"
            "        kv_cache = kv_cache.transpose(1, 2)\n"
            "        if is_quantized_kv_cache(self.kv_cache_dtype) and (\n"
            "            kv_cache.dtype != self.fp8_dtype\n"
            "        ):\n"
            "            kv_cache = kv_cache.view(self.fp8_dtype)\n"
            "        key_cache = kv_cache[..., :head_size_qk]\n",
        ),
    ],
    "triton_attn_diffkv.py",
)

patch(
    attention_path,
    [
        (
            "                page_size_padded=shared_page,\n",
            "                # mimo-diffkv-fp8-kv: the platform's skip_page_size_padded is\n"
            "                # estimated from the model-level KV head count (4 for MiMo),\n"
            "                # but the SWA layers have 8 and DiffKV is not slot-packed, so\n"
            "                # the estimate lands below even the smallest kernel block page\n"
            "                # (B12X supports only 64/128). The spec then violates\n"
            "                # page_size_padded >= unpadded_page_size_bytes and the engine\n"
            "                # dies in resolve_kv_cache_layout. Floor the padded page at the\n"
            "                # chosen kernel block's natural page; page unification scales\n"
            "                # the full-attention blocks up to match.\n"
            "                page_size_padded=(\n"
            "                    max(shared_page, sw_block_size * sw_per_token)\n"
            "                    if shared_page is not None\n"
            "                    else None\n"
            "                ),\n",
        ),
    ],
    "attention.py",
)

patch(
    dflash_path,
    [
        (
            "            return SlidingWindowSpec(\n"
            "                block_size=vllm_config.cache_config.block_size,\n"
            "                num_kv_heads=self.num_kv_heads,\n"
            "                head_size=self.head_size,\n"
            "                head_size_v=self.head_size_v,\n"
            "                dtype=self.kv_cache_torch_dtype,\n"
            "                sliding_window=self.sliding_window,\n"
            "                page_size_padded=getattr(\n"
            "                    vllm_config.cache_config, \"skip_page_size_padded\", None\n"
            "                ),\n",
            "            # mimo-diffkv-fp8-kv: the DFlash drafter consumes\n"
            "            # skip_page_size_padded directly (bypassing the attention.py\n"
            "            # block-size chooser patched above), so the same platform\n"
            "            # underestimate leaves its padded page below its natural one\n"
            "            # (block 128 x 1024 B/token = 128 KiB vs 48 KiB shared page)\n"
            "            # and the engine asserts in resolve_kv_cache_layout. Floor the\n"
            "            # padded page at the spec's natural page; unification pads the\n"
            "            # remaining specs up.\n"
            "            skip_page = getattr(\n"
            "                vllm_config.cache_config, \"skip_page_size_padded\", None\n"
            "            )\n"
            "            spec = SlidingWindowSpec(\n"
            "                block_size=vllm_config.cache_config.block_size,\n"
            "                num_kv_heads=self.num_kv_heads,\n"
            "                head_size=self.head_size,\n"
            "                head_size_v=self.head_size_v,\n"
            "                dtype=self.kv_cache_torch_dtype,\n"
            "                sliding_window=self.sliding_window,\n"
            "                page_size_padded=None,\n",
        ),
        (
            "                kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),\n"
            "                dcp_replicated=dcp_replicated,\n"
            "            )\n"
            "        spec = super().get_kv_cache_spec(vllm_config)\n",
            "                kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),\n"
            "                dcp_replicated=dcp_replicated,\n"
            "            )\n"
            "            if skip_page is not None:\n"
            "                spec = dataclasses.replace(\n"
            "                    spec,\n"
            "                    page_size_padded=max(\n"
            "                        skip_page, spec.unpadded_page_size_bytes\n"
            "                    ),\n"
            "                )\n"
            "            return spec\n"
            "        spec = super().get_kv_cache_spec(vllm_config)\n",
        ),
    ],
    "qwen3_dflash.py",
)
PY

echo "=====> MiMo attention layers honor --kv-cache-dtype fp8 on the DiffKV backend"