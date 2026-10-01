#!/bin/bash
set -euo pipefail

# Focused check for mods/mimo-v2-aux-over-pp/run.sh: flips
# supports_aux_hidden_states_over_pp on MiMoV2Model only, is idempotent on
# re-run, and fails fast when the anchor or the file is missing.

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


class EagleModelMixin:
    supports_aux_hidden_states_over_pp = False


class MiMoV2Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config=None, prefix=""):
        super().__init__()
EOF

fail() { echo "FAIL: $1" >&2; exit 1; }

# 1. First run applies the patch and the class attribute takes effect.
out1=$(PYTHON_ROOT="$ROOT" bash "$MOD") || fail "mod exited non-zero"
grep -qF "supports_aux_hidden_states_over_pp = True" "$MIMO" \
    || fail "opt-in attribute missing"
grep -qF "Patched mimo_v2.py." <<< "$out1" || fail "missing patch report"
python3 - "$MIMO" <<'PY' || fail "patched mimo_v2.py does not behave"
import importlib.util, sys
spec = importlib.util.spec_from_file_location("mimo_v2_fixture", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.MiMoV2Model.supports_aux_hidden_states_over_pp is True, "class attr not set"
assert mod.EagleModelMixin.supports_aux_hidden_states_over_pp is False, "mixin was mutated"
assert getattr(mod.MiMoV2Model(), "supports_aux_hidden_states_over_pp", False) is True
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
