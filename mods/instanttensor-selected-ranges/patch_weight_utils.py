#!/usr/bin/env python3
"""Adapt the image's index-aware vLLM loader to InstantTensor 0.2.2."""

import argparse
import ast
from pathlib import Path
import stat
import sys

PREFIX = "[instanttensor-selected-ranges]"
HELPER = "_restrict_instanttensor_to_selected_ranges"
OLD_ATTRS = '''        "tensor_name_to_index",
        "loader_handle",
        "_determine_buffer_size",
'''
NEW_ATTRS = '''        "tensor_name_to_index",
        "tensor_keep",
        "loader_handle",
        "_finalize_buffer_size",
'''
OLD_LAYOUT = '''        instant_open.tensor_offsets = selected_offsets
        instant_open.tensor_sizes = selected_sizes
        instant_open.total_tensor_size = sum(selected_sizes)
        instant_open._determine_buffer_size(None)
'''
NEW_LAYOUT = '''        instant_open.tensor_offsets = selected_offsets
        # spark-vllm mod: instanttensor-selected-ranges v1
        # Native open expects one keep bit per compacted physical tensor.
        instant_open.tensor_keep = [True] * len(selected_metadata)
        instant_open.tensor_sizes = selected_sizes
        instant_open.total_tensor_size = sum(selected_sizes)
        # The old determine method resolved this env override internally.
        from instanttensor._impl import env_buffer_size

        instant_open._finalize_buffer_size(env_buffer_size())
'''


def patched_text(source: str) -> str:
    tree = ast.parse(source)
    helpers = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == HELPER]
    if len(helpers) != 1:
        raise ValueError(f"expected one {HELPER}; found {len(helpers)}")
    helper = helpers[0]
    lines = source.splitlines(keepends=True)
    start, end = helper.lineno - 1, helper.end_lineno
    body = "".join(lines[start:end])
    old_counts = (body.count(OLD_ATTRS), body.count(OLD_LAYOUT))
    new_counts = (body.count(NEW_ATTRS), body.count(NEW_LAYOUT))
    if old_counts == (1, 1) and new_counts == (0, 0):
        body = body.replace(OLD_ATTRS, NEW_ATTRS, 1).replace(OLD_LAYOUT, NEW_LAYOUT, 1)
    elif old_counts != (0, 0) or new_counts != (1, 1):
        raise ValueError(f"incompatible or partially patched helper: old={old_counts}, new={new_counts}")
    result = "".join(lines[:start]) + body + "".join(lines[end:])
    compile(result, "<InstantTensor weight_utils.py>", "exec")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        original = args.target.read_text()
        result = patched_text(original)
        if args.check:
            print(f"{PREFIX} compatible: {args.target}")
        elif result == original:
            print(f"{PREFIX} already patched: {args.target}")
        else:
            mode = stat.S_IMODE(args.target.stat().st_mode)
            temporary = args.target.with_suffix(".selected-ranges.tmp")
            temporary.write_text(result)
            temporary.chmod(mode)
            temporary.replace(args.target)
            print(f"{PREFIX} Patched {args.target}")
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"{PREFIX} refusing to patch {args.target}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
