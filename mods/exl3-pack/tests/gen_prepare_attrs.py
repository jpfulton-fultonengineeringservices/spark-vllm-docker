"""Regenerate ``prepare_attrs.json`` from the pinned exllamav3 wheel.

Single source of truth: the ``EXLLAMAV3_WHEEL_URL`` / ``EXLLAMAV3_WHEEL_SHA256``
pair in ``Dockerfile.exl3-pack``. Downloads the wheel (no install, no GPU),
verifies the SHA, AST-parses ``exllamav3/conversion/convert_model.py``, and
writes every ``args.<name>`` attribute read inside ``prepare`` (including
helpers called from it: ``override``, ``prepare_env``) plus the attributes
read via ``vars(args)`` inside ``override``.

Run on the host:
    uv run --extra hosttest python tests/gen_prepare_attrs.py

Drift is caught by ``test_required_attrs_match_pinned_wheel`` in
``test_dist_args_contract.py``, which re-derives the set without regenerating.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

TESTS = Path(__file__).parent
REPO_ROOT = TESTS.parent.parent.parent  # spark-vllm-docker/
DOCKERFILE = REPO_ROOT / "Dockerfile.exl3-pack"
CONVERT_MODEL = "exllamav3/conversion/convert_model.py"


def pinned_wheel() -> tuple[str, str]:
    """Extract wheel URL + sha256 from the Dockerfile ARGs."""
    text = DOCKERFILE.read_text()
    url = re.search(r'ARG EXLLAMAV3_WHEEL_URL=(\S+)', text)
    sha = re.search(r'ARG EXLLAMAV3_WHEEL_SHA256=(\S+)', text)
    if not url or not sha:
        sys.exit(f"cannot find EXLLAMAV3_WHEEL_URL/SHA256 in {DOCKERFILE}")
    return url.group(1), sha.group(1)


def load_convert_model_source() -> str:
    url, sha = pinned_wheel()
    with urllib.request.urlopen(url) as r:  # noqa: S310
        data = r.read()
    import hashlib
    if hashlib.sha256(data).hexdigest() != sha:
        sys.exit(f"sha256 mismatch for {url}")
    with zipfile.ZipFile(BytesIO(data)) as z:
        return z.read(CONVERT_MODEL).decode()


def attribute_reads(tree: ast.Module, function_name: str, seen: set[str] = ()) -> set[str]:
    """``args.X`` attribute reads inside one function, recursing into calls."""
    seen = set(seen)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Attribute)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == "args"
                ):
                    seen.add(inner.attr)
    return seen


def override_table_args(tree: ast.Module) -> set[str]:
    """Names in the ``for arg_, can_override, default in [ ... ]`` table in prepare()."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "prepare":
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.For)
                    and isinstance(inner.target, ast.Tuple)
                    and len(inner.target.elts) >= 1
                    and isinstance(inner.target.elts[0], ast.Name)
                    and isinstance(inner.iter, ast.List)
                ):
                    for entry in inner.iter.elts:
                        if (
                            isinstance(entry, ast.Tuple)
                            and entry.elts
                            and isinstance(entry.elts[0], ast.Constant)
                        ):
                            names.add(str(entry.elts[0].value))
    return names


def main() -> None:
    src = load_convert_model_source()
    tree = ast.parse(src)
    # prepare() plus helpers it calls that read args
    names = set()
    names |= attribute_reads(tree, "prepare")
    names |= attribute_reads(tree, "override")
    names |= attribute_reads(tree, "prepare_env")
    names |= override_table_args(tree)
    out = TESTS / "prepare_attrs.json"
    out.write_text(json.dumps(sorted(names), indent=2) + "\n")
    print(f"wrote {out.name}: {len(names)} attributes")


if __name__ == "__main__":
    main()
