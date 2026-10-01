#!/bin/bash
set -euo pipefail

# MiMo-V2 aux-hidden-states-over-pipeline mod.
#
# Enables DFlash (and eagle3-family) speculative decoding for MiMo-V2 when
# vLLM runs with pipeline parallelism (PP > 1). Three gaps block it; all are
# fixed here:
#
# 1. Startup aborts with:
#
#      ValueError: MiMoV2FlashForCausalLM does not support dflash with pipeline
#      parallelism
#
#    The DFlash drafter consumes target aux hidden states at the checkpoint's
#    dflash/config.json target_layer_ids [0, 11, 23, 35, 47]. With PP=3 those
#    layers land on all three pipeline stages (0-15 / 16-31 / 32-47), so the
#    generic cross-stage aux relay must carry them to the last rank where the
#    drafter lives. The relay machinery is generic
#    (EagleModelMixin + pp_handler.configure_aux_hidden_state_relay) and is
#    already proven for LlamaModel / Qwen2Model / DeepseekV4Model /
#    KimiK3Model / MiniMaxM3Model; it is gated per model class by
#    supports_aux_hidden_states_over_pp, which MiMoV2Model inherits as False.
#    The mod flips the opt-in on MiMoV2Model, exactly like the classes above.
#
# 2. Drafter load aborts with:
#
#      RuntimeError: DFlashQwen3ForCausalLM needs the target input embedding,
#      but it is unavailable on this PP stage
#
#    The DFlash drafter ships no embed_tokens weights (its config declares no
#    own embedding), so load_dflash_model borrows the target's embed_tokens
#    (maybe_share_target_embed). Under PP that module is PPMissingLayer on
#    every stage but the first - and the drafter lives on the LAST stage. The
#    mod builds embed_tokens on every pipeline stage instead, so the borrow
#    finds a real module. Every stage loads the same checkpoint weights, so
#    the copies are identical; the cost is ~1.25 GiB per extra stage
#    (vocab 152576 x hidden 4096, bf16).
#
# 3. KV cache allocation aborts with:
#
#      setStorage: sizes [2, 130, 2, 128, 2048], strides [532480, 655360,
#      262144, 2048, 1] ... out of bounds for storage of size 85196800
#
#    _partition_parallel_draft_specs (vllm/v1/core/kv_cache_utils.py) keeps
#    the DFlash drafter in the target's cache group under PP. The split
#    early-returns at PP > 1, and its layer-index test compares against the
#    PER-RANK layer count (get_num_layers(parallel_config) = 16 at PP=3)
#    while the drafter's layers extend the GLOBAL index space (48..52), so
#    even without the guard the last stage's own target layers would be
#    misclassified as draft. One shared group forces one shared stride
#    across incompatible KV geometries (target fp8 320 B rows vs drafter
#    bf16 2048 B rows), and the layout math cannot absorb it. The patch
#    classifies draft layers against the global target layer count and
#    allows the split at PP > 1, giving the drafter its own cache group -
#    the proven PP=1 shape.
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
        (
            "        if get_pp_group().is_first_rank or (\n"
            "            config.tie_word_embeddings and get_pp_group().is_last_rank\n"
            "        ):\n"
            "            self.embed_tokens = VocabParallelEmbedding(\n"
            "                config.vocab_size,\n"
            "                config.hidden_size,\n"
            "                quant_config=quant_config,\n"
            "                prefix=f\"{prefix}.embed_tokens\",\n"
            "            )\n"
            "        else:\n"
            "            self.embed_tokens = PPMissingLayer()\n",
            "        # spark-vllm-docker/mods/mimo-v2-aux-over-pp: build the input\n"
            "        # embedding on every pipeline stage, not just the first (and the\n"
            "        # last when tied). The DFlash drafter lives on the last PP stage\n"
            "        # and borrows the target's embed_tokens\n"
            "        # (maybe_share_target_embed); PPMissingLayer here aborts startup\n"
            "        # with \"needs the target input embedding, but it is unavailable\n"
            "        # on this PP stage\". Every stage loads the same checkpoint\n"
            "        # weights, so the copies are identical; cost ~1.25 GiB per extra\n"
            "        # stage.\n"
            "        self.embed_tokens = VocabParallelEmbedding(\n"
            "            config.vocab_size,\n"
            "            config.hidden_size,\n"
            "            quant_config=quant_config,\n"
            "            prefix=f\"{prefix}.embed_tokens\",\n"
            "        )\n",
        ),
        (
            "        aux_hidden_states = self._maybe_add_hidden_state(\n"
            "            [], self.start_layer, hidden_states, residual\n"
            "        )\n"
            "        for idx, layer in enumerate(\n"
            "            islice(self.layers, self.start_layer, self.end_layer)\n"
            "        ):\n"
            "            hidden_states, residual = layer(positions, hidden_states, residual)\n"
            "            self._maybe_add_hidden_state(\n"
            "                aux_hidden_states, idx + 1, hidden_states, residual\n"
            "            )\n"
            "\n"
            "        if not get_pp_group().is_last_rank:\n"
            "            return IntermediateTensors(\n"
            "                {\"hidden_states\": hidden_states, \"residual\": residual}\n"
            "            )\n"
            "\n"
            "        hidden_states, _ = self.norm(hidden_states, residual)\n"
            "\n"
            "        if len(aux_hidden_states) > 0:\n"
            "            return hidden_states, aux_hidden_states\n"
            "        return hidden_states\n",
            "        # spark-vllm-docker/mods/mimo-v2-aux-over-pp: the aux-over-PP glue,\n"
            "        # matching the LlamaModel/Qwen2Model pattern. Three fixes over the\n"
            "        # TP-only version: collect the upstream aux states relayed through\n"
            "        # the PP intermediate tensors; capture local ones at GLOBAL layer\n"
            "        # indices (enumerate needs start=self.start_layer - without it the\n"
            "        # per-stage slice renumbers layers and the wrong aux states are\n"
            "        # captured, so the drafter sees fewer/different features than its\n"
            "        # target_layer_ids promise); and pack local aux into the outgoing\n"
            "        # intermediate tensors so middle stages can relay them, returning\n"
            "        # upstream + local in layer order on the last stage.\n"
            "        remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors)\n"
            "\n"
            "        aux_hidden_states: list[torch.Tensor] = []\n"
            "        if get_pp_group().is_first_rank:\n"
            "            self._maybe_add_hidden_state(\n"
            "                aux_hidden_states, self.start_layer, hidden_states, residual\n"
            "            )\n"
            "        for idx, layer in enumerate(\n"
            "            islice(self.layers, self.start_layer, self.end_layer),\n"
            "            start=self.start_layer,\n"
            "        ):\n"
            "            hidden_states, residual = layer(positions, hidden_states, residual)\n"
            "            self._maybe_add_hidden_state(\n"
            "                aux_hidden_states, idx + 1, hidden_states, residual\n"
            "            )\n"
            "\n"
            "        if not get_pp_group().is_last_rank:\n"
            "            return IntermediateTensors(\n"
            "                {\n"
            "                    \"hidden_states\": hidden_states,\n"
            "                    \"residual\": residual,\n"
            "                    **self.pack_local_aux_hidden_states(aux_hidden_states),\n"
            "                }\n"
            "            )\n"
            "\n"
            "        hidden_states, _ = self.norm(hidden_states, residual)\n"
            "\n"
            "        aux_hidden_states = remote_aux + aux_hidden_states\n"
            "        if len(aux_hidden_states) > 0:\n"
            "            return hidden_states, aux_hidden_states\n"
            "        return hidden_states\n",
        ),
    ],
    "mimo_v2.py",
)
PY

echo "=====> MiMoV2Model carries DFlash aux hidden states across pipeline stages"
echo "=====> MiMoV2Model replicates embed_tokens on every pipeline stage for the drafter"
echo "=====> MiMoV2Model.forward relays aux hidden states across pipeline stages"

KVUTILS="$PYTHON_ROOT/vllm/v1/core/kv_cache_utils.py"

if [ ! -f "$KVUTILS" ]; then
  echo "$PREFIX Missing $KVUTILS; a newer image is required." >&2
  exit 1
fi

python3 - "$KVUTILS" <<'PY'
from pathlib import Path
import sys

kvutils_path = Path(sys.argv[1])


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
    kvutils_path,
    [
        (
            "        or vllm_config.parallel_config.pipeline_parallel_size > 1\n"
            "        or vllm_config.scheduler_config.disable_hybrid_kv_cache_manager\n"
            "    ):\n"
            "        return kv_cache_spec, {}\n"
            "\n"
            "    target_num_layers = vllm_config.model_config.get_num_layers(\n"
            "        vllm_config.parallel_config\n"
            "    )\n",
            "        or vllm_config.scheduler_config.disable_hybrid_kv_cache_manager\n"
            "    ):\n"
            "        return kv_cache_spec, {}\n"
            "\n"
            "    # spark-vllm-docker/mods/mimo-v2-aux-over-pp: classify draft\n"
            "    # layers against the GLOBAL target layer count, and allow the\n"
            "    # split at PP > 1. The drafter's layers extend the global index\n"
            "    # space (48..52 for MiMo-V2.6 + 5 DFlash layers) and register\n"
            "    # only on the last PP stage, while\n"
            "    # get_num_layers(parallel_config) is per-rank (16 at PP=3).\n"
            "    # The old per-rank count plus the PP > 1 guard kept the drafter\n"
            "    # inside the target's cache group under PP, and one group shares\n"
            "    # strides across incompatible KV geometries (setStorage\n"
            "    # out-of-bounds on the 3x PP=3 launch, 2026-10-01).\n"
            "    # Independent draft groups are the proven PP=1 shape; max()\n"
            "    # keeps PP=1 classification bit-identical.\n"
            "    target_num_layers = max(\n"
            "        vllm_config.model_config.hf_config.num_hidden_layers,\n"
            "        vllm_config.model_config.get_num_layers(\n"
            "            vllm_config.parallel_config\n"
            "        ),\n"
            "    )\n",
        ),
    ],
    "kv_cache_utils.py",
)
PY

echo "=====> _partition_parallel_draft_specs gives the drafter its own cache group at PP>1"
