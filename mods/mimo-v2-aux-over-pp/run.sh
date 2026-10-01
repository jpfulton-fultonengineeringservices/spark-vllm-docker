#!/bin/bash
set -euo pipefail

# MiMo-V2 aux-hidden-states-over-pipeline mod.
#
# Enables DFlash (and eagle3-family) speculative decoding for MiMo-V2 when
# vLLM runs with pipeline parallelism (PP > 1). Without it, engine startup
# aborts with:
#
#   ValueError: MiMoV2FlashForCausalLM does not support dflash with pipeline
#   parallelism
#
# The DFlash drafter consumes target aux hidden states at the checkpoint's
# dflash/config.json target_layer_ids [0, 11, 23, 35, 47]. With PP=3 those
# layers land on all three pipeline stages (0-15 / 16-31 / 32-47), so the
# generic cross-stage aux relay must carry them to the last rank where the
# drafter lives. The relay machinery is generic
# (EagleModelMixin + pp_handler.configure_aux_hidden_state_relay) and is
# already proven for LlamaModel / Qwen2Model / DeepseekV4Model / KimiK3Model /
# MiniMaxM3Model; it is gated per model class by
# supports_aux_hidden_states_over_pp, which MiMoV2Model inherits as False.
# This mod flips the opt-in on MiMoV2Model, exactly like the classes above.
#
# Risk is bounded: draft tokens are self-verified against target logits, so a
# misbehaving drafter degrades acceptance rate and speed, not output
# correctness. Verify acceptance and temp=0 output equivalence on first
# launch (see mods/mimo-v2-aux-over-pp/README.md).
#
# Each file is patched independently and skipped when already fixed.
# Idempotent within the same fresh container.

PREFIX="[mimo-v2-aux-over-pp]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
MIMO="$PYTHON_ROOT/vllm/model_executor/models/mimo_v2.py"

echo "=== MiMo aux-over-PP mod ==="

if [ ! -f "$MIMO" ]; then
  echo "$PREFIX Missing $MIMO; a newer image is required." >&2
  exit 1
fi

python3 - "$MIMO" <<'PY'
from pathlib import Path
import sys

mimo_path = Path(sys.argv[1])


def patch(path: Path, edits: list[tuple[str, str]], label: str) -> None:
    text = path.read_text()
    changed = False
    for old, new in edits:
        if new in text:
            continue
        if old not in text:
            raise SystemExit(
                f"[mimo-v2-aux-over-pp] {label}: expected anchor not found; "
                "the installed vLLM differs from the layout this mod knows."
            )
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text)
        print(f"[mimo-v2-aux-over-pp] Patched {label}.")
    else:
        print(f"[mimo-v2-aux-over-pp] {label} already patched; skipping.")


patch(
    mimo_path,
    [
        (
            "class MiMoV2Model(nn.Module, EagleModelMixin):\n",
            "class MiMoV2Model(nn.Module, EagleModelMixin):\n"
            "    # spark-vllm-docker/mods/mimo-v2-aux-over-pp: the DFlash drafter\n"
            "    # consumes target aux hidden states at dflash/config.json\n"
            "    # target_layer_ids [0, 11, 23, 35, 47], which span all pipeline\n"
            "    # stages at PP=3. Opt in to the generic cross-stage aux relay\n"
            "    # (same one-line opt-in as LlamaModel and Qwen2Model) so the\n"
            "    # drafter on the last PP rank receives them.\n"
            "    supports_aux_hidden_states_over_pp = True\n",
        ),
    ],
    "mimo_v2.py",
)
PY

echo "=====> MiMoV2Model carries DFlash aux hidden states across pipeline stages"
