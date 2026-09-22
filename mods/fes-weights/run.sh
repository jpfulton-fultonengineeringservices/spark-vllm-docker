#!/bin/bash
# [fes-weights] Verify FES-staged model weights mounted into this container and,
# when FES_HUB_MODEL is set, present them in HF hub-cache layout so that
# `vllm serve <org/model>` resolves offline without touching the hub.
#
# The mod runs on the head and every worker before the launch script executes.
# Env (set via run-recipe.sh -e):
#   FES_WEIGHTS_DIR       read-only weights mount inside the container (default /model)
#   FES_DRAFT_DIR         optional draft-model weights mount (default /drafter)
#   FES_HUB_MODEL         org/model id served by the recipe; creates the hub shim
#   FES_DRAFT_HUB_MODEL   optional draft id; creates a second shim for FES_DRAFT_DIR
# Pair with -e HF_HUB_OFFLINE=1 so serve never reaches the hub.

set -euo pipefail

PREFIX="[fes-weights]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Opt-in mod: recipes may list fes-weights unconditionally. Without the FES
# env (no fes-eval.sh launch), stay inert so hub-cache mode is unaffected.
if [ -z "${FES_WEIGHTS_ENABLED:-}" ]; then
    echo "$PREFIX FES_WEIGHTS_SKIP op=verify reason=FES_WEIGHTS_ENABLED not set (hub-cache mode)"
    exit 0
fi

weights_dir="${FES_WEIGHTS_DIR:-/model}"
draft_dir="${FES_DRAFT_DIR:-/drafter}"
hub_root="${HF_HOME:-/root/.cache/huggingface}/hub"

fail() {
    echo "$PREFIX FES_WEIGHTS_FAIL op=verify dir=$1 reason=$2" >&2
    exit 1
}

verify() {
    local dir="$1" label="$2" out
    [ -d "$dir" ] || fail "$dir" "$label: mount missing"
    if ! out="$(python3 "$MOD_DIR/verify.py" "$dir" 2>&1)"; then
        fail "$dir" "$label: ${out##*$'\n'}"
    fi
    echo "$PREFIX $label verified: ${out##*$'\n'}"
}

make_shim() {
    local model="$1" src="$2"
    local org="${model%%/*}" name="${model#*/}"
    [ "$org" != "$model" ] || fail "$src" "hub model must be org/model, got '$model'"
    local repo_dir="$hub_root/models--${org}--${name}"
    local snap="$repo_dir/snapshots/fes-staged"
    local fp_file="$repo_dir/.fes-shim-fp"
    local fp base f
    fp="$(cd "$src" && find . -maxdepth 1 -type f -printf '%f %s\n' | sort | sha256sum | cut -d' ' -f1)"
    if [ -f "$fp_file" ] && [ "$(cat "$fp_file")" = "$fp" ] && [ -d "$snap" ]; then
        echo "$PREFIX $model: hub shim up to date ($snap -> $src)"
        return
    fi
    rm -rf "$snap"
    mkdir -p "$snap" "$repo_dir/refs"
    for f in "$src"/*; do
        base="$(basename "$f")"
        case "$base" in .*) continue ;; esac
        ln -sfn "$f" "$snap/$base"
    done
    printf '%s' "fes-staged" > "$repo_dir/refs/main"
    printf '%s\n' "$fp" > "$fp_file"
    echo "$PREFIX $model: hub shim created ($snap -> $src)"
}

verify "$weights_dir" "main weights"
if [ -n "${FES_DRAFT_DIR:-}" ]; then
    verify "$draft_dir" "draft weights"
fi
if [ -n "${FES_HUB_MODEL:-}" ]; then
    make_shim "$FES_HUB_MODEL" "$weights_dir"
fi
if [ -n "${FES_DRAFT_HUB_MODEL:-}" ]; then
    make_shim "$FES_DRAFT_HUB_MODEL" "$draft_dir"
fi

echo "$PREFIX FES_WEIGHTS_OK op=verify dir=$weights_dir slug=${FES_HUB_MODEL:-none}"