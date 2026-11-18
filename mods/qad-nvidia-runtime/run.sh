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
# WHY IT CANNOT BE A RECIPE TODAY. The checkpoint's
# serving/runtime-manifest.json pins
#   pinned_modelopt_runtime_sha256 =
#     41e00493d2770af19db03f585e9edb0004093c98b851d0f6726c2870cba11e22
# which is vLLM v0.30.0's vllm/model_executor/layers/quantization/modelopt.py
# (2656 lines, fetched from GitHub). Every image we build today ships
# modelopt.py hashing 5dfedfbd… instead — vLLM 0.1.dev21830+g8244d1c44
# (2026-11-02) — in both upstream vllm-node-b12x:latest (image ID
# sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02) and
# the fork build ghcr.io/fulton-engineering-services/spark-vllm:
# vllm-node-b12x-20261106-0/1 (sha256:b76585f9020097bc9984b914bbcdebd5cdce5bbf66911ada28a9d7fa3649f096).
# A fourth sample, measured on gx10-node1 on 2026-11-18: the newest build,
# vllm-node-b12x-20261118 (config ID sha256:319205b2ff31…, also tagged
# 10.100.170.3:5000/spark-vllm:vllm-node-b12x-20261118), ships the byte-identical
# 5dfedfbd… modelopt.py. It is a rebuild of the same vLLM commit 8244d1c44 whose
# only vllm/ delta over c34501eb is the tracked
# docker/patch_vllm_b12x_storage_block_split.py layer plus a refreshed dependency
# set (CUDA 13.4→13.5, nccl_extensions 0.1.0→0.2.0, instanttensor 0.2.2→0.2.3).
# Re-pinning a recipe to it therefore does not enable this mod — running it
# there reaches gate_modelopt_runtime and fails closed, which is the behaviour
# verified for both bash and sourced invocation.
# canonical_worker.py verifies that hash itself, so --worker-cls aborts on
# launch: NVIDIA's command cannot simply be dropped into a recipe.
#
# ENABLEMENT CONDITION (the only thing standing between this and a recipe):
# build the b12x image against the vLLM tag whose modelopt.py hashes to the
# manifest pin — vLLM v0.30.0, i.e. --build-arg VLLM_REF=v0.30.0 — then point a
# recipe at this mod and use the launch line in README.md. Until then this mod
# fails closed below, before launch, so a half-enabled arm can never serve.
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
    fail "op=gate_modelopt_runtime installed=$ACTUAL pinned=$PIN — $CANONICAL_WORKER would abort on its own hash gate. Remedy: build the b12x image against vLLM v0.30.0 (--build-arg VLLM_REF=v0.30.0) so $MODELLOPT hashes to the manifest pin, then re-enable this mod."
fi

info "ok (modelopt.py matches manifest pin ${PIN:0:12}…; PYTHONPATH=${PYTHONPATH:-unset}; VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-unset})"
