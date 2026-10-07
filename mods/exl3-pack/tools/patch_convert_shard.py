#!/usr/bin/env python3
"""Teach ``exllamav3.conversion.convert_model`` a lower module bound.

Parallel EXL3 packing splits one ``convert_model`` run into contiguous module
ranges, one per node.  ``--max_module`` already stops the quantization loop
after a module index; this patch adds the missing lower bound, ``--module-start``
(inclusive), exposed to the job args as ``min_module``.  With the two together a
node quantizes ``[min_module, max_module]`` and the shards are merged afterwards.

The patch is idempotent and fails loud: if the installed ``convert_model.py`` does
not carry the exact source shapes the three edits anchor on, it raises instead of
building an unpatched (or mismatched) image.
"""

from __future__ import annotations

import argparse
import sys
import sysconfig
from pathlib import Path

TARGET_REL = Path("exllamav3/conversion/convert_model.py")

# 1. CLI: the parser gains the lower bound, mirroring --max_module's shape.
PARSER_ANCHOR = (
    'parser.add_argument("--max_module", type = int, help = "End quantization '
    'after this many modules, includes embedding and norm layers (for debug '
    'purposes)", default = None)\n'
)
PARSER_INSERT = (
    'parser.add_argument("--module-start", type = int, default = None, '
    'help = "Start quantization at this module index, inclusive (for split/'
    'sharded packing, combined with --max_module)")\n'
)

# 2. prepare(): carry the CLI value into the persisted job args.
ARGS_ANCHOR = '    in_args["max_module"] = args.max_module\n'
ARGS_INSERT = '    in_args["min_module"] = args.module_start\n'

# 3. main(): skip modules below the lower bound, right after the resume skip.
LOOP_SKIP = (
    '        # If resuming, skip along to checkpoint index\n'
    '        if idx < job_state["next_module_idx"]:\n'
    '            continue\n'
)
LOOP_MAX = '        if args["max_module"] is not None and idx > args["max_module"]:\n'
LOOP_ANCHOR = LOOP_SKIP + '\n' + LOOP_MAX
LOOP_INSERT = (
    '\n'
    '        if args.get("min_module") is not None and idx < args["min_module"]:\n'
    '            continue\n'
)

# Presence of these strings means the corresponding edit is already applied.
PARSER_MARKER = '"--module-start"'
ARGS_MARKER = 'in_args["min_module"] = args.module_start'
LOOP_MARKER = 'args.get("min_module") is not None and idx < args["min_module"]'


class PatchError(RuntimeError):
    pass


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(
            f"expected one {description} source anchor, found {count}; "
            "the exllamav3 convert_model.py shape has changed"
        )
    return text.replace(old, new, 1)


def patch_parser(text: str) -> str:
    if PARSER_MARKER in text:
        return text
    return replace_once(
        text,
        PARSER_ANCHOR,
        PARSER_ANCHOR + PARSER_INSERT,
        "--max_module parser argument",
    )


def patch_args(text: str) -> str:
    if ARGS_MARKER in text:
        return text
    return replace_once(
        text,
        ARGS_ANCHOR,
        ARGS_ANCHOR + ARGS_INSERT,
        'in_args["max_module"] assignment',
    )


def patch_loop(text: str) -> str:
    if LOOP_MARKER in text:
        return text
    return replace_once(
        text,
        LOOP_ANCHOR,
        LOOP_SKIP + LOOP_INSERT + '\n' + LOOP_MAX,
        "resume-skip / --max_module loop guard",
    )


def apply_patch(text: str) -> str:
    text = patch_parser(text)
    text = patch_args(text)
    text = patch_loop(text)
    compile(text, TARGET_REL.as_posix(), "exec")
    return text


def resolve_target(root: Path) -> Path:
    target = root / TARGET_REL
    if not target.is_file():
        raise PatchError(f"{TARGET_REL} is missing under {root}")
    return target


def default_root() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv[1:])

    target = resolve_target(args.root)
    original = target.read_text()

    updated = apply_patch(original)
    if original == updated:
        print(f"{TARGET_REL} already carries --module-start")
        return 0

    if args.check:
        # Not yet patched, but every anchor is present and applicable.
        print(f"{TARGET_REL} is unpatched and --module-start is applicable")
        return 0

    target.write_text(updated)
    print(f"Applied --module-start/--max_module range patch to {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except PatchError as exc:
        raise SystemExit(f"convert_model shard patch failed: {exc}") from exc
