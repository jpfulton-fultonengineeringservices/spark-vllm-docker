#!/bin/bash
# qad-nvidia-runtime parked-mod test: bash 3.2 + 5.x portable.
#
# The mod is deliberately referenced by nothing in the build path, so it can
# only fail loudly: this test drives it against fixture trees and asserts the
# gate is fail-closed, and it asserts the parked invariant itself (no recipe,
# preset or map may name the mod while it is parked).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

MOD="$ROOT/mods/qad-nvidia-runtime/run.sh"
[ -f "$MOD" ] || fail "mod run.sh missing"
bash -n "$MOD" || fail "mod run.sh does not parse"
[ -f "$ROOT/mods/qad-nvidia-runtime/README.md" ] || fail "mod README.md missing (the faithful command must live in-repo)"

# --- fixtures -------------------------------------------------------------
WEIGHTS="$WORK/model"
SERVING="$WEIGHTS/serving"
PKG="$WORK/vllm_pkg"
mkdir -p "$SERVING" "$PKG/model_executor/layers/quantization"

echo "def worker(): ..." > "$PKG/model_executor/layers/quantization/modelopt.py"
ACTUAL="$( { sha256sum "$PKG/model_executor/layers/quantization/modelopt.py" 2>/dev/null \
            || shasum -a 256 "$PKG/model_executor/layers/quantization/modelopt.py"; } | awk '{print $1}')"
[ -n "$ACTUAL" ] || fail "could not hash the fixture modelopt.py"

write_manifest() {  # $1 = pin value to record
  cat > "$SERVING/runtime-manifest.json" <<JSON
{"pinned_modelopt_runtime_sha256": "$1"}
JSON
}

run_mod() {  # runs the mod, capturing rc/stdout+stderr
  OUT="$(FES_WEIGHTS_DIR="$WEIGHTS" VLLM_PKG_DIR="$PKG" "$@" bash "$MOD" 2>&1)" && RC=0 || RC=$?
}

echo "=== qad-nvidia-runtime parked mod ==="

# 1. Parked invariant: nothing in the build path may apply the mod.
if grep -rn "qad-nvidia-runtime" "$ROOT/recipes" "$ROOT/tests/expected_commands.sh" 2>/dev/null; then
  fail "the parked mod is referenced from the build path"
fi
# The faithful-flavour recipe YAMLs must never materialise; the mod is the
# parked artifact instead.
for ghost in "$ROOT/recipes"/*/qwen3.8-flash-next-nvfp4-qad-nvidia-2x-workercls.yaml \
             "$ROOT/recipes"/*/qwen3.8-flash-next-nvfp4-qad-nvidia-4x-workercls.yaml \
             "$ROOT/recipes"/*/*/qwen3.8-flash-next-nvfp4-qad-nvidia-2x-workercls.yaml \
             "$ROOT/recipes"/*/*/qwen3.8-flash-next-nvfp4-qad-nvidia-4x-workercls.yaml; do
  [ -e "$ghost" ] && fail "parked workercls recipe materialised: $ghost"
done

# 2. Inert without activation, even when the runtime would fail the gate.
printf 'x\n' > "$SERVING/canonical_worker.py"
write_manifest "0000000000000000000000000000000000000000000000000000000000000000"
run_mod env -u QAD_NVIDIA_RUNTIME
[ "$RC" -eq 0 ] || fail "inert run must exit 0 (rc=$RC): $OUT"
echo "$OUT" | grep -q "inert" || fail "inert run did not say so: $OUT"

# 3. Fail closed on a runtime that does not match the manifest pin.
run_mod env QAD_NVIDIA_RUNTIME=1
[ "$RC" -ne 0 ] || fail "gate accepted a mismatched modelopt.py: $OUT"
echo "$OUT" | grep -q "gate_modelopt_runtime" || fail "gate did not name its op: $OUT"
echo "$OUT" | grep -q "0000000000000000000000000000000000000000000000000000000000000000" \
  || fail "gate did not report the expected pin: $OUT"
echo "$OUT" | grep -q "$ACTUAL" || fail "gate did not report the installed hash: $OUT"
echo "$OUT" | grep -q "v0.30.0" || fail "gate did not name the remedy image build: $OUT"

# 4. Active and matching: ok, with the worker-class env exported.
write_manifest "$ACTUAL"
run_mod env QAD_NVIDIA_RUNTIME=1
[ "$RC" -eq 0 ] || fail "gate rejected a matching modelopt.py (rc=$RC): $OUT"
echo "$OUT" | grep -q "ok" || fail "matching run did not report ok: $OUT"
ENV_OUT="$(FES_WEIGHTS_DIR="$WEIGHTS" VLLM_PKG_DIR="$PKG" QAD_NVIDIA_RUNTIME=1 \
  bash -c "source '$MOD' >/dev/null; echo \"P=\${PYTHONPATH:-unset} M=\${VLLM_WORKER_MULTIPROC_METHOD:-unset}\"")"
[ "$ENV_OUT" = "P=$SERVING M=spawn" ] \
  || fail "mod did not export PYTHONPATH/VLLM_WORKER_MULTIPROC_METHOD: $ENV_OUT"

# PYTHONPATH must append, never clobber an existing search path.
ENV_OUT="$(FES_WEIGHTS_DIR="$WEIGHTS" VLLM_PKG_DIR="$PKG" QAD_NVIDIA_RUNTIME=1 PYTHONPATH=/keep/me \
  bash -c "source '$MOD' >/dev/null; echo \"\${PYTHONPATH:-unset}\"")"
[ "$ENV_OUT" = "$SERVING:/keep/me" ] || fail "mod clobbered PYTHONPATH: $ENV_OUT"

# 5. Missing vendor runtime fails before the hash gate.
mv "$SERVING/canonical_worker.py" "$WORK/canonical_worker.py.bak"
run_mod env QAD_NVIDIA_RUNTIME=1
[ "$RC" -ne 0 ] || fail "missing canonical_worker.py was accepted: $OUT"
echo "$OUT" | grep -q "require_canonical_worker" || fail "wrong op for missing worker: $OUT"
mv "$WORK/canonical_worker.py.bak" "$SERVING/canonical_worker.py"

# 6. A manifest without the pin is not a pass.
write_manifest ""
run_mod env QAD_NVIDIA_RUNTIME=1
[ "$RC" -ne 0 ] || fail "manifest without a pin was accepted: $OUT"
echo "$OUT" | grep -q "read_pin" || fail "wrong op for missing pin: $OUT"
write_manifest "$ACTUAL"

# 7. An unmounted weights dir fails rather than serving without the vendor code.
run_mod env QAD_NVIDIA_RUNTIME=1 FES_WEIGHTS_DIR="$WORK/nope"
[ "$RC" -ne 0 ] || fail "unmounted FES_WEIGHTS_DIR was accepted: $OUT"
echo "$OUT" | grep -q "require_canonical_worker" || fail "wrong op for unmounted weights: $OUT"

echo "qad-nvidia-runtime mod: PASS"
