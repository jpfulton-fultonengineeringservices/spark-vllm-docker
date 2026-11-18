#!/usr/bin/env bash
# qad-nvidia-runtime (v1): PARKED — the faithful NVIDIA QAD serving path for
# the qwen3.8-flash-next-nvfp4-qad-nvidia checkpoint, i.e. NVIDIA's own launch
# line with --worker-cls canonical_worker.CanonicalWorker against the
# checkpoint's bundled serving/ runtime.
#
# PARKED BY DESIGN. Nothing in the build path applies this mod: no recipe lists
# it in `mods:`, no `defaults`, no infra/cluster-config map and no
# cluster-control-infra preset names it. It exists so the faithful command
# lives in the repository instead of only in a design doc.
#
# WHY IT WAS A RECIPE IMPOSSIBILITY, AND WHAT CHANGED ON 2026-11-18. The
# checkpoint's serving/runtime-manifest.json AS DELIVERED pinned
#   pinned_modelopt_runtime_sha256 =
#     41e00493d2770af19db03f585e9edb0004093c98b851d0f6726c2870cba11e22
# which is vLLM v0.30.0's vllm/model_executor/layers/quantization/modelopt.py
# (2656 lines, verified against the tag). Every image we build ships
# modelopt.py hashing 5dfedfbd… instead — vLLM 0.1.dev21830+g8244d1c44
# (2026-11-02) — measured in four configs: upstream vllm-node-b12x:latest
# (sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02),
# the fork build ghcr.io/fulton-engineering-services/spark-vllm:
# vllm-node-b12x-20261106-0/1 (sha256:b76585f9020097bc9984b914bbcdebd5cdce5bbf66911ada28a9d7fa3649f096),
# and the newest vllm-node-b12x-20261118 (sha256:319205b2ff31…, also tagged
# 10.100.170.3:5000/spark-vllm:vllm-node-b12x-20261118). Building against
# v0.30.0 to satisfy that pin is NOT available: no upstream tag ships the B12X
# loader at all (model_loader/b12x_loader.py, model_executor/weight_transfer.py
# and model_loader/csf_scales.py are absent at v0.30.0, v0.32.0, v0.33.0,
# v0.34.0 and main), and our docker/patches/qwen3.8/vllm series cannot add it
# there — on a full clone with the build's own `git apply --3way --index`, 12
# of 13 patches fail and only 13-draft-vocab applies. A hash-faithful graft of
# v0.30.0's file onto the fork is out too: the two differ by +100/-5 lines with
# an identical class set, and those 100 lines ARE the b12x surface
# (Nvfp4CsfState/Nvfp4CsfMoEMethod, modelopt_iq2_xs block codecs,
# prepares_b12x MXFP8 prep, W4A16 zero-init calibrated scales, workspace
# plumbing).
# So, with operator approval, the delivered manifest's pin was RESTAMPED on
# 2026-11-18 to the fork revision actually installed,
#   5dfedfbdeda804aa8ff45d84f9d823bd8d45e13d4134a11d97012e05635d10d6
# leaving the four vendor .py files byte-identical (their manifest digests
# still match) and preserving the vendor value in a `restamp` block plus a
# backup file kept OUTSIDE the delivery tree, in .restamp-backups/ beside the
# checkpoint; see README.md for the hashes and the review trail.
# Because canonical_worker.py:33 asserts modelopt_hash against
# manifest["pinned_modelopt_runtime_sha256"] — the same JSON field this gate
# reads, with no separate literal in the worker — restamping moves BOTH gates
# together, and a vendor manifest bump or a re-download reverts both.
#
# STILL PARKED, for a different reason than before. The mod stays unapplied
# until an operator first-boot validates the GPU-side asserts the worker makes
# after loading (fused per-rank counts MXFP8 144 / NVFP4_shared 96, kernel
# identity, weight dtypes). Until then this mod fails closed below, before
# launch, so a half-enabled arm can never serve.
#
# What it does when applied:
#   1. resolve the fes-weights mount (FES_WEIGHTS_DIR, default /model) and
#      require serving/canonical_worker.py + serving/runtime-manifest.json;
#   2. export PYTHONPATH=<FES_WEIGHTS_DIR>/serving (so vLLM can import
#      canonical_worker) and VLLM_WORKER_MULTIPROC_METHOD=spawn;
#   3. fail closed unless sha256 of the INSTALLED
#      vllm/model_executor/layers/quantization/modelopt.py equals the manifest's
#      pinned_modelopt_runtime_sha256, read from the JSON — never hard-coded
#      here, so a future manifest bump needs no mod edit.
#
# The launch line itself lives in README.md: it is a command, not a container
# patch, so this mod only prepares and gates the environment for it.
#
# Opt-in, mirroring mods/fix-mfu-metrics: inert unless QAD_NVIDIA_RUNTIME=1 is
# in the mod's calling env or the recipe's exec-script references it.
# apply_mod_to_container runs with WORKDIR=/workspace/vllm, NOT the mod dir, so
# grep the canonical /workspace/exec-script.sh path directly; run-recipe.py
# renders the recipe env block into that script before mods are applied.
set -euo pipefail

PREFIX="[qad-nvidia-runtime]"
fail() { echo "$PREFIX FAIL $*" >&2; return 1; }
info() { echo "$PREFIX $*"; }

EXEC_SCRIPT="/workspace/exec-script.sh"
if [ "${QAD_NVIDIA_RUNTIME:-0}" != "1" ] \
   && { [ ! -f "$EXEC_SCRIPT" ] || ! grep -q 'QAD_NVIDIA_RUNTIME' "$EXEC_SCRIPT" 2>/dev/null; }; then
    echo "$PREFIX inert (parked: set QAD_NVIDIA_RUNTIME=1 and launch with the README command)"
    return 0 2>/dev/null || exit 0
fi

FES_WEIGHTS_DIR="${FES_WEIGHTS_DIR:-/model}"
SERVING_DIR="$FES_WEIGHTS_DIR/serving"
MANIFEST="$SERVING_DIR/runtime-manifest.json"
CANONICAL_WORKER="$SERVING_DIR/canonical_worker.py"

[ -f "$CANONICAL_WORKER" ] \
    || fail "op=require_canonical_worker missing $CANONICAL_WORKER (is mods/fes-weights mounted at $FES_WEIGHTS_DIR?)"
[ -f "$MANIFEST" ] || fail "op=require_manifest missing $MANIFEST"

export PYTHONPATH="${SERVING_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Locate the installed vLLM package directory (find_spec returns the package dir
# itself). VLLM_PKG_DIR overrides it, mirroring VLLM_PACKAGE_ROOT in
# mods/mooncake-worker/run.sh, so the gate is testable off-container.
if [ -z "${VLLM_PKG_DIR:-}" ]; then
    VLLM_PKG_DIR="$(python3 <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    sys.exit("vLLM not importable (is this the vllm-node container?)")
print(next(iter(spec.submodule_search_locations)))
PY
)"
fi
MODELLOPT="$VLLM_PKG_DIR/model_executor/layers/quantization/modelopt.py"
[ -f "$MODELLOPT" ] || fail "op=locate_modelopt missing $MODELLOPT"

PIN="$(python3 - "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
pin = manifest.get("pinned_modelopt_runtime_sha256", "")
if not isinstance(pin, str):
    sys.exit("pinned_modelopt_runtime_sha256 is not a string")
print(pin.removeprefix("sha256:").strip().lower())
PY
)"
[ -n "$PIN" ] || fail "op=read_pin $MANIFEST has no pinned_modelopt_runtime_sha256"

ACTUAL="$( { sha256sum "$MODELLOPT" 2>/dev/null || shasum -a 256 "$MODELLOPT"; } | awk '{print $1}')"
[ -n "$ACTUAL" ] || fail "op=hash_modelopt could not hash $MODELLOPT"

if [ "$ACTUAL" != "$PIN" ]; then
    fail "op=gate_modelopt_runtime installed=$ACTUAL pinned=$PIN — $CANONICAL_WORKER asserts the same value at load and would abort. This gate detects DRIFT: a re-synced delivery restoring the vendor pin, an image re-pinned off dev/karmic-kraken, or a swapped modelopt.py. Remedy: re-stamp pinned_modelopt_runtime_sha256 in $MANIFEST to the installed fork revision — the sanctioned restamp of 2026-11-18, whose hashes and backup path are in mods/qad-nvidia-runtime/README.md. Do NOT chase the vendor pin by building the b12x image against vLLM v0.30.0: no upstream ref ships the B12X loader there, and 12 of 13 docker/patches/qwen3.8/vllm patches fail git apply --3way on that tag. NEVER edit serving/*.py to widen the gate: runtime-manifest.json pins their hashes and the worker verifies them."
fi

info "ok (modelopt.py matches manifest pin ${PIN:0:12}…; PYTHONPATH=${PYTHONPATH:-unset}; VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-unset})"
