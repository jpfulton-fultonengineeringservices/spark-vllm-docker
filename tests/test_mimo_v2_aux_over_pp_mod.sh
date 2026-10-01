#!/bin/bash
set -euo pipefail

# Focused check for mods/mimo-v2-aux-over-pp/run.sh: flips
# supports_aux_hidden_states_over_pp on MiMoV2Model, replicates embed_tokens on
# every PP stage, is idempotent on re-run, and fails fast when the anchor or
# the file is missing.

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/mimo-v2-aux-over-pp/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

ROOT="$TMP_DIR/site"
mkdir -p "$ROOT/vllm/model_executor/models"
MIMO="$ROOT/vllm/model_executor/models/mimo_v2.py"

cat > "$MIMO" <<'EOF'
import types


class _StubModule:
    pass


nn = types.SimpleNamespace(Module=_StubModule)


class VocabParallelEmbedding:
    def __init__(self, vocab_size, hidden_size, quant_config=None, prefix=""):
        self.vocab_size = vocab_size


class PPMissingLayer:
    pass


class EagleModelMixin:
    supports_aux_hidden_states_over_pp = False


def get_pp_group():
    return types.SimpleNamespace(is_first_rank=True, is_last_rank=False)


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
EOF

fail() { echo "FAIL: $1" >&2; exit 1; }

# 1. First run applies both patches and the changes take effect.
out1=$(PYTHON_ROOT="$ROOT" bash "$MOD") || fail "mod exited non-zero"
grep -qF "supports_aux_hidden_states_over_pp = True" "$MIMO" \
    || fail "opt-in attribute missing"
if grep -qF "self.embed_tokens = PPMissingLayer()" "$MIMO"; then
    fail "embed_tokens still PPMissingLayer on non-first PP stages"
fi
grep -qF "Patched mimo_v2.py." <<< "$out1" || fail "missing patch report"
python3 - "$MIMO" <<'PY' || fail "patched mimo_v2.py does not behave"
import importlib.util, sys, types
spec = importlib.util.spec_from_file_location("mimo_v2_fixture", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.MiMoV2Model.supports_aux_hidden_states_over_pp is True, "class attr not set"
assert mod.EagleModelMixin.supports_aux_hidden_states_over_pp is False, "mixin was mutated"
cfg = types.SimpleNamespace(tie_word_embeddings=False, vocab_size=152576, hidden_size=4096)
assert getattr(mod.MiMoV2Model(vllm_config=cfg), "supports_aux_hidden_states_over_pp", False) is True
# Non-first PP stage (the drafter's stage): embed must be a real module so
# maybe_share_target_embed can borrow it.
mod.get_pp_group = lambda: types.SimpleNamespace(is_first_rank=False, is_last_rank=False)
mid = mod.MiMoV2Model(vllm_config=cfg)
assert isinstance(mid.embed_tokens, mod.VocabParallelEmbedding), mid.embed_tokens
# First stage unchanged; tied embeddings still reach the last stage.
mod.get_pp_group = lambda: types.SimpleNamespace(is_first_rank=True, is_last_rank=False)
first = mod.MiMoV2Model(vllm_config=cfg)
assert isinstance(first.embed_tokens, mod.VocabParallelEmbedding), first.embed_tokens
mod.get_pp_group = lambda: types.SimpleNamespace(is_first_rank=False, is_last_rank=True)
tied_cfg = types.SimpleNamespace(tie_word_embeddings=True, vocab_size=152576, hidden_size=4096)
last = mod.MiMoV2Model(vllm_config=tied_cfg)
assert isinstance(last.embed_tokens, mod.VocabParallelEmbedding), last.embed_tokens
PY

# 2. Second run is a no-op (idempotent within a fresh container).
cp "$MIMO" "$TMP_DIR/mimo.after1"
out2=$(PYTHON_ROOT="$ROOT" bash "$MOD") || fail "re-run exited non-zero"
cmp -s "$TMP_DIR/mimo.after1" "$MIMO" || fail "re-run modified an already-patched file"
grep -qF "already patched; skipping" <<< "$out2" || fail "missing skip report"

# 3. Missing anchor fails fast instead of writing garbage.
cat > "$MIMO" <<'EOF'
def something_else():
    pass
EOF
if PYTHON_ROOT="$ROOT" bash "$MOD" 2>/dev/null; then
    fail "mod should fail when the anchor is missing"
fi

# 4. Missing mimo_v2.py fails fast.
rm -f "$MIMO"
if PYTHON_ROOT="$ROOT" bash "$MOD" 2>/dev/null; then
    fail "mod should fail when mimo_v2.py is missing"
fi

echo "PASS: mimo-v2-aux-over-pp mod"
