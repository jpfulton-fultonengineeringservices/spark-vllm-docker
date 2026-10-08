"""Generate ``prepare_attrs.json`` by probing the live convert_model.prepare().

Instead of AST parsing the source, this invokes ``prepare()`` with a minimal
Namespace and records every missing-argument error raised. This catches both
direct ``args.<name>`` reads and keys only reachable via the ``override()``
table.

Run on the host:
    uv run --extra hosttest python tests/gen_prepare_attrs.py

Or in-image (with the live wheel installed):
    PYTHONPATH=/opt/exl3-pack/src python tests/gen_prepare_attrs.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TESTS = Path(__file__).parent


def probe_prepare_contract() -> set[str]:
    """Invoke convert_model.prepare() with a minimal Namespace and collect
    every missing-argument name it raises. This catches both direct
    ``args.<name>`` reads and keys only reachable via the ``override()``
    table. Returns the union of all required attribute names."""
    from exllamav3.conversion import convert_model as cm

    # Seed with known override-table keys to avoid infinite loops on
    # interdependent checks; the probe will still catch anything prepare()
    # actually requires that we missed.
    seed_keys = {
        "in_dir", "out_dir", "work_dir", "recipe", "bits", "codebook",
        "devices", "device_ratios", "hessians", "hessians_reg",
        "cal_data", "cal_rows", "cal_cols", "shard_size", "vision_bits",
        "ngram_bits", "ngram_file", "last_checkpoint_index",
        "checkpoint_interval", "gather_timeout",
    }

    # Iteratively probe: run prepare() with current keys set to None (falsy),
    # catch ValueError("X is required"), add X, repeat until stable.
    required: set[str] = set()
    changed = True
    iterations = 0
    max_iterations = 50  # Safety limit

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        # Build namespace with all discovered keys set to None (falsy)
        ns = argparse.Namespace(**{k: None for k in required | seed_keys})

        try:
            cm.prepare(vars(ns))  # type: ignore[arg-type]
            # If prepare() succeeds with all None, we've found the full set
            break
        except ValueError as e:
            msg = str(e)
            # Match patterns like "X is required" or "must provide X"
            m = re.search(r"([a-z_]+)\s+is\s+required", msg, re.IGNORECASE)
            if m:
                name = m.group(1)
                if name not in required:
                    required.add(name)
                    changed = True
                    continue
            # Also catch override() table style: "X must ..."
            m = re.search(r'(?:must|--)[^"\']*["\']?([a-z_]+)["\']?', msg, re.IGNORECASE)
            if m:
                name = m.group(1)
                if name not in required and name not in seed_keys:
                    required.add(name)
                    changed = True
                    continue

    return required | seed_keys


def main() -> None:
    names = probe_prepare_contract()
    out = TESTS / "prepare_attrs.json"
    out.write_text(json.dumps(sorted(names), indent=2) + "\n")
    print(f"wrote {out.name}: {len(names)} attributes (runtime probe)")


if __name__ == "__main__":
    main()
