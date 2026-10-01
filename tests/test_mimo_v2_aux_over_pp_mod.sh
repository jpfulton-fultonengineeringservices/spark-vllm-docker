#!/bin/bash
set -euo pipefail

# Focused check for mods/mimo-v2-aux-over-pp/run.sh: flips
# supports_aux_hidden_states_over_pp on MiMoV2Model, replicates embed_tokens on
# every PP stage, rewires the forward-pass aux glue to the canonical
# LlamaModel/Qwen2Model pattern (global layer ids + pack/collect over PP), is
# idempotent on re-run, and fails fast when an anchor or the file is missing.

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/mimo-v2-aux-over-pp/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

ROOT="$TMP_DIR/site"
mkdir -p "$ROOT/vllm/model_executor/models"
MIMO="$ROOT/vllm/model_executor/models/mimo_v2.py"
mkdir -p "$ROOT/vllm/v1/core"
KVUTILS="$ROOT/vllm/v1/core/kv_cache_utils.py"

cat > "$KVUTILS" <<'EOF'
"""Fragment of vllm/v1/core/kv_cache_utils.py for the mod test."""


class VllmConfig:
    pass


class KVCacheSpec:
    pass


def _partition_parallel_draft_specs(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> tuple[dict[str, KVCacheSpec], dict[str, KVCacheSpec]]:
    """Split appended DFlash or GLM DSpark layers for PP1 cache grouping."""
    from vllm.model_executor.models.utils import extract_layer_index

    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is None
        or not (
            speculative_config.method == "dflash"
            or (
                speculative_config.method == "dspark"
                and speculative_config.draft_model_config.hf_config.model_type
                == "glm53_dspark"
            )
        )
        or vllm_config.parallel_config.pipeline_parallel_size > 1
        or vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
    ):
        return kv_cache_spec, {}

    target_num_layers = vllm_config.model_config.get_num_layers(
        vllm_config.parallel_config
    )
    target_specs: dict[str, KVCacheSpec] = {}
    draft_specs: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_spec.items():
        try:
            layer_index = extract_layer_index(layer_name)
        except (AssertionError, IndexError, ValueError):
            target_specs[layer_name] = spec
            continue
        if layer_index >= target_num_layers:
            draft_specs[layer_name] = spec
        else:
            target_specs[layer_name] = spec
    return target_specs, draft_specs
EOF

cat > "$MIMO" <<'EOF'
import types
from bisect import bisect_right
from itertools import islice


class _StubModule:
    pass


nn = types.SimpleNamespace(Module=_StubModule)


class VocabParallelEmbedding:
    def __init__(self, vocab_size, hidden_size, quant_config=None, prefix=""):
        self.vocab_size = vocab_size


class PPMissingLayer:
    pass


class IntermediateTensors:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]


class EagleModelMixin:
    supports_aux_hidden_states_over_pp = False
    AUX_HIDDEN_STATE_KEY = "aux_hidden_states_"
    _aux_slot_base_cached = 0
    _aux_upstream_total_cached = 0

    def _set_aux_hidden_state_layers(self, layers):
        self.aux_hidden_state_layers = tuple(sorted(set(layers)))
        self._aux_slot_base_cached = 0
        self._aux_upstream_total_cached = 0
        self._cache_aux_pp_layout()

    def _cache_aux_pp_layout(self):
        pp = get_pp_group()
        if pp.world_size < 2:
            return
        if not pp.is_first_rank:
            self._aux_slot_base_cached = bisect_right(
                self.aux_hidden_state_layers, self.start_layer
            )
        if pp.is_last_rank:
            self._aux_upstream_total_cached = self._aux_slot_base_cached

    def _maybe_add_hidden_state(self, aux_hidden_states, layer_idx, hidden_states, residual):
        if layer_idx in self.aux_hidden_state_layers:
            value = hidden_states + residual if residual is not None else hidden_states
            aux_hidden_states.append(value)
        return aux_hidden_states

    def pack_local_aux_hidden_states(self, aux_hidden_states):
        if not aux_hidden_states:
            return {}
        base = self._aux_slot_base_cached
        return {
            f"{self.AUX_HIDDEN_STATE_KEY}{base + i}": t
            for i, t in enumerate(aux_hidden_states)
        }

    def collect_remote_aux_hidden_states(self, intermediate_tensors):
        total = self._aux_upstream_total_cached
        if total == 0:
            return []
        assert intermediate_tensors is not None
        out = []
        for i in range(total):
            key = f"{self.AUX_HIDDEN_STATE_KEY}{i}"
            if key not in intermediate_tensors.tensors:
                raise RuntimeError(f"Missing {key} from PP intermediate tensors")
            out.append(intermediate_tensors[key])
        return out


def get_pp_group():
    return types.SimpleNamespace(
        world_size=1, is_first_rank=True, is_last_rank=True
    )


class MiMoV2Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config=None, prefix=""):
        super().__init__()
        config = vllm_config
        quant_config = None
        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if get_pp_group().is_first_rank:
            hidden_states = "H"
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def norm(self, hidden_states, residual):
        return hidden_states, residual
EOF

fail() { echo "FAIL: $1" >&2; exit 1; }

# 1. First run applies all three patches and the changes take effect.
out1=$(PYTHON_ROOT="$ROOT" bash "$MOD") || fail "mod exited non-zero"
grep -qF "supports_aux_hidden_states_over_pp = True" "$MIMO" \
    || fail "opt-in attribute missing"
if grep -qF "self.embed_tokens = PPMissingLayer()" "$MIMO"; then
    fail "embed_tokens still PPMissingLayer on non-first PP stages"
fi
grep -qF "start=self.start_layer," "$MIMO" \
    || fail "aux capture still renumbers layers per PP stage"
grep -qF "collect_remote_aux_hidden_states(intermediate_tensors)" "$MIMO" \
    || fail "upstream aux collection missing"
grep -qF "pack_local_aux_hidden_states(aux_hidden_states)" "$MIMO" \
    || fail "local aux packing into intermediate tensors missing"
grep -qF "Patched mimo_v2.py." <<< "$out1" || fail "missing patch report"
python3 - "$MIMO" <<'PY' || fail "patched mimo_v2.py does not behave"
import importlib.util, sys, types
spec = importlib.util.spec_from_file_location("mimo_v2_fixture", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.MiMoV2Model.supports_aux_hidden_states_over_pp is True, "class attr not set"
assert mod.EagleModelMixin.supports_aux_hidden_states_over_pp is False, "mixin was mutated"

cfg = types.SimpleNamespace(tie_word_embeddings=False, vocab_size=152576, hidden_size=4096)
mod.get_pp_group = lambda: types.SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True)
assert getattr(mod.MiMoV2Model(vllm_config=cfg), "supports_aux_hidden_states_over_pp", False) is True

# Embed replication on non-first / tied-last stages.
mod.get_pp_group = lambda: types.SimpleNamespace(world_size=3, is_first_rank=False, is_last_rank=False)
mid = mod.MiMoV2Model(vllm_config=cfg)
assert isinstance(mid.embed_tokens, mod.VocabParallelEmbedding), mid.embed_tokens
mod.get_pp_group = lambda: types.SimpleNamespace(world_size=3, is_first_rank=False, is_last_rank=True)
last_tied = mod.MiMoV2Model(vllm_config=types.SimpleNamespace(tie_word_embeddings=True, vocab_size=1, hidden_size=1))
assert isinstance(last_tied.embed_tokens, mod.VocabParallelEmbedding), last_tied.embed_tokens

# Aux-over-PP forward glue, 48 layers / 3 stages, aux ids (1, 12, 24, 36, 48).
AUX_IDS = (1, 12, 24, 36, 48)

def make_model(start, end, world, first, last):
    mod.get_pp_group = lambda: types.SimpleNamespace(world_size=world, is_first_rank=first, is_last_rank=last)
    m = mod.MiMoV2Model(vllm_config=cfg)
    m.start_layer, m.end_layer = start, end
    m.layers = [lambda pos, h, r, i=i: (f"H{i+1}", f"R{i+1}") for i in range(48)]
    m._set_aux_hidden_state_layers(AUX_IDS)
    return m

# Stage 0 (layers 0-15): captures ids 1 and 12 into output slots 0 and 1.
m0 = make_model(0, 16, 3, True, False)
out = m0.forward(None, None)
assert isinstance(out, mod.IntermediateTensors), out
assert set(k for k in out.tensors if k.startswith("aux_")) == {"aux_hidden_states_0", "aux_hidden_states_1"}, out.tensors.keys()

# Stage 1 (layers 16-31): relays nothing itself (pp_handler does), captures id
# 24 into slot 2 -- which REQUIRES global layer numbering.
m1 = make_model(16, 32, 3, False, False)
out = m1.forward(None, None, mod.IntermediateTensors({"hidden_states": "H", "residual": "R", "aux_hidden_states_0": "A1", "aux_hidden_states_1": "A12"}))
assert set(k for k in out.tensors if k.startswith("aux_")) == {"aux_hidden_states_2"}, out.tensors.keys()
assert out.tensors["aux_hidden_states_2"] == "H24R24", out.tensors["aux_hidden_states_2"]

# Stage 2 (layers 32-47, the drafter's stage): collects 3 upstream aux and
# captures ids 36 and 48 locally -> 5 features in LAYER order (the drafter
# concatenates them as its 5 x 4096 target features).
m2 = make_model(32, 48, 3, False, True)
hidden, aux = m2.forward(None, None, mod.IntermediateTensors({
    "hidden_states": "H", "residual": "R",
    "aux_hidden_states_0": "A1", "aux_hidden_states_1": "A12", "aux_hidden_states_2": "A24",
}))
assert aux == ["A1", "A12", "A24", "H36R36", "H48R48"], aux
PY

grep -qF "Patched kv_cache_utils.py." <<< "$out1" || fail "missing kv_cache_utils patch report"
grep -qF "target_num_layers = max(" "$KVUTILS" \
    || fail "global draft classification missing"
grep -qF "hf_config.num_hidden_layers" "$KVUTILS" \
    || fail "global target layer count missing"
! grep -qF "or vllm_config.parallel_config.pipeline_parallel_size > 1" "$KVUTILS" \
    || fail "PP>1 guard still blocks the draft-group split"

python3 - "$KVUTILS" <<'PY' || fail "patched _partition_parallel_draft_specs does not behave"
import importlib.util
import re
import sys
import types

stub = types.ModuleType("vllm.model_executor.models.utils")


def extract_layer_index(name):
    match = re.search(r"(\d+)", name)
    if not match:
        raise ValueError(name)
    return int(match.group(1))


stub.extract_layer_index = extract_layer_index
sys.modules["vllm.model_executor.models.utils"] = stub

spec = importlib.util.spec_from_file_location("kv_utils", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_cfg(pp):
    return NS(
        speculative_config=NS(
            method="dflash",
            draft_model_config=NS(hf_config=NS(model_type="dflash_qwen3")),
        ),
        parallel_config=NS(pipeline_parallel_size=pp),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
        model_config=NS(
            hf_config=NS(num_hidden_layers=48),
            get_num_layers=lambda parallel_config: (
                48 // parallel_config.pipeline_parallel_size
            ),
        ),
    )


def indices(names):
    return sorted(int(re.search(r"\d+", name).group()) for name in names)


# PP=3, last stage: global target layers 32..47 plus drafter 48..52. The old
# code early-returned (one shared group -> shared strides -> setStorage).
kv = {"model.layers.%d.self_attn" % i: object() for i in range(32, 53)}
target, draft = m._partition_parallel_draft_specs(make_cfg(3), kv)
assert indices(target) == list(range(32, 48)), target
assert indices(draft) == [48, 49, 50, 51, 52], draft

# PP=1: target 0..47 plus drafter 48..52 (bit-identical to the old behavior).
kv = {"model.layers.%d.self_attn" % i: object() for i in range(0, 53)}
target, draft = m._partition_parallel_draft_specs(make_cfg(1), kv)
assert len(target) == 48 and len(draft) == 5, (len(target), len(draft))

# No speculative config: unsplit.
cfg = make_cfg(3)
cfg.speculative_config = None
kv = {"model.layers.0.self_attn": object()}
assert m._partition_parallel_draft_specs(cfg, kv) == (kv, {})
PY

# 2. Second run is a no-op (idempotent within a fresh container).
cp "$MIMO" "$TMP_DIR/mimo.after1"
cp "$KVUTILS" "$TMP_DIR/kvutils.after1"
out2=$(PYTHON_ROOT="$ROOT" bash "$MOD") || fail "re-run exited non-zero"
cmp -s "$TMP_DIR/mimo.after1" "$MIMO" || fail "re-run modified an already-patched file"
cmp -s "$TMP_DIR/kvutils.after1" "$KVUTILS" \
    || fail "re-run modified an already-patched kv_cache_utils.py"
grep -qF "already patched; skipping" <<< "$out2" || fail "missing skip report"

# 3. Missing kv_cache_utils.py fails fast.
mv "$KVUTILS" "$TMP_DIR/kvutils.saved"
if PYTHON_ROOT="$ROOT" bash "$MOD" 2>/dev/null; then
    fail "mod should fail when kv_cache_utils.py is missing"
fi
mv "$TMP_DIR/kvutils.saved" "$KVUTILS"

# 4. Missing anchor fails fast instead of writing garbage.
cat > "$MIMO" <<'EOF'
def something_else():
    pass
EOF
if PYTHON_ROOT="$ROOT" bash "$MOD" 2>/dev/null; then
    fail "mod should fail when the anchor is missing"
fi

# 5. Missing mimo_v2.py fails fast.
rm -f "$MIMO"
if PYTHON_ROOT="$ROOT" bash "$MOD" 2>/dev/null; then
    fail "mod should fail when mimo_v2.py is missing"
fi

echo "PASS: mimo-v2-aux-over-pp mod"
