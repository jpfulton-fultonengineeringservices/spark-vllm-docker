#!/bin/bash
# Host-side companion for mods/fes-weights: pre-verify a FES-staged weight copy
# on this node, then launch a recipe with the read-only weights mount, the
# verify mod, and the offline hub shim wired up.
#
# Usage:
#   scripts/fes-eval.sh [--weights-root DIR] [--hub-model ID] <recipe> <fes-slug> [run-recipe.sh args...]
#
# Examples:
#   scripts/fes-eval.sh glm-5.3-flash glm-5.3-flash-nvfp4 --solo
#   scripts/fes-eval.sh qwen3.8-27b-nvfp4-w4a4 qwen3.8-27b-nvfp4-w4a4 -n 10.10.10.165
#
# --hub-model defaults to the recipe's `model:` field (the org/model id the
# command serves). Any remaining args are passed to run-recipe.sh verbatim.
#
# Multi-node recipes: every listed node needs the slug staged under the weights
# root (cluster-config scripts/model-weights.sh stage <slug> fans it out), since
# launch-cluster.sh applies the mount on every node's container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WEIGHTS_ROOT="/opt/llm/models"
HUB_MODEL=""

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights-root) WEIGHTS_ROOT="$2"; shift 2 ;;
        --hub-model) HUB_MODEL="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) break ;;
    esac
done

RECIPE="${1:-}"
SLUG="${2:-}"
[ -n "$RECIPE" ] && [ -n "$SLUG" ] || usage 2
shift 2

recipe_file=""
for ext in yaml yml; do
    if [ -f "$REPO_DIR/recipes/$RECIPE.$ext" ]; then
        recipe_file="$REPO_DIR/recipes/$RECIPE.$ext"
        break
    fi
done
[ -n "$recipe_file" ] || { echo "error: recipe not found under recipes/: $RECIPE" >&2; exit 2; }

if [ -z "$HUB_MODEL" ]; then
    HUB_MODEL="$(python3 - "$recipe_file" <<'PY'
import sys
try:
    import yaml
    print((yaml.safe_load(open(sys.argv[1])).get("model") or "").strip())
except ImportError:
    for line in open(sys.argv[1]):
        if line.startswith("model:"):
            print(line.split(":", 1)[1].strip().strip('"').strip("'"))
            break
PY
)"
    [ -n "$HUB_MODEL" ] || { echo "error: recipe has no model: field; pass --hub-model" >&2; exit 2; }
fi

weights_dir="$WEIGHTS_ROOT/$SLUG"
if ! python3 "$REPO_DIR/mods/fes-weights/verify.py" "$weights_dir" >/dev/null; then
    echo "error: staged weights failed verification at $weights_dir" >&2
    echo "       stage them first: cluster-config scripts/model-weights.sh stage $SLUG" >&2
    exit 1
fi

echo "[fes-eval] recipe=$RECIPE slug=$SLUG hub-model=$HUB_MODEL weights=$weights_dir"

exec "$REPO_DIR/run-recipe.sh" "$RECIPE" \
    --apply-mod "$REPO_DIR/mods/fes-weights" \
    -v "$weights_dir:/model:ro" \
    -e "FES_WEIGHTS_ENABLED=1" \
    -e "FES_WEIGHTS_DIR=/model" \
    -e "FES_HUB_MODEL=$HUB_MODEL" \
    -e "HF_HUB_OFFLINE=1" \
    "$@"