#!/bin/bash
set -euo pipefail

PREFIX="[fix-explicit-spec-method]"
PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
info() {
  echo "$PREFIX $*"
}

python3 - "$PYTHON_ROOT" <<'PY'
import ast
import importlib.util
from pathlib import Path
import sys

python_root = sys.argv[1]
prefix = "[fix-explicit-spec-method]"
if not Path(python_root, "vllm").is_dir():
    raise SystemExit(f"{prefix} FAIL: vLLM package not found at {python_root}/vllm")
spec = importlib.util.find_spec("vllm.config.speculative")
if spec is None or spec.origin is None:
    raise SystemExit(f"{prefix} FAIL: cannot locate vllm.config.speculative")
with open(spec.origin) as fh:
    source = fh.read()
lines = source.splitlines()
pristine = '                if self.method in ("eagle", "eagle3", "dflash", "dspark"):'
fixed = '                if self.method in ("eagle", "eagle3", "dflash", "dspark", "mtp"):'
broken = '                if self.method is None or self.method in ("eagle", "eagle3", "dflash", "dspark"):'
if source.count(fixed) == 1:
    print(f"{prefix} Patch is already applied; skipping.")
else:
    if source.count(broken) == 1:
        print(f"{prefix} Migrating ineffective 2026-11-04 None-guard revision.")
    elif source.count(pristine) != 1:
        raise SystemExit(
            f"{prefix} FAIL: expected exactly one pristine, fixed, or old "
            f"None-guard anchor at {spec.origin}; re-derive the patch"
        )
    lines = [
        fixed if line in (pristine, broken) else line
        for line in lines
    ]
    replacement = "\n".join(lines) + "\n"
    ast.parse(replacement)
    with open(spec.origin, "w") as fh:
        fh.write(replacement)
    print(f"{prefix} Applied explicit-spec-method trusted-tuple fix.")

with open(spec.origin) as fh:
    ast.parse(fh.read())
print(f"{prefix} speculative.py parses successfully at {spec.origin}.")
PY

info "Explicit MTP is trusted; inferred draft_model auto-detection is unchanged."
