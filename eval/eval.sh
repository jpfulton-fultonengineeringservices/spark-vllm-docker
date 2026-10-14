#!/bin/bash
# eval/eval.sh - thin launcher for the eval rig (requires uv).
#
# Usage:
#   eval/eval.sh run <config.toml> [--dry-run] [extra lm-eval args...]
#   eval/eval.sh viz <results-dir> [--local] [--html] [--zeno]
#   eval/eval.sh setup
#   eval/eval.sh <config.toml> [...]        # shorthand for `run`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

command -v uv >/dev/null 2>&1 || {
    echo "error: uv not found; install it first (curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
    exit 1
}

[ "$#" -ge 1 ] || {
    echo "usage: eval/eval.sh run|viz|setup|<config.toml> [args...]" >&2
    exit 2
}

case "$1" in
    run|viz|setup)
        exec uv run --python 3.12 --no-project "$SCRIPT_DIR/cli.py" "$@"
        ;;
    *)
        exec uv run --python 3.12 --no-project "$SCRIPT_DIR/cli.py" run "$@"
        ;;
esac
