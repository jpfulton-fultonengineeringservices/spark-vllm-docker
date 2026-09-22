#!/usr/bin/env python3
"""Build a shadow tree of exactly the git-tracked files of a target subtree.

deepwiki-rs's directory walk and per-directory dossier reads do NOT honor
.gitignore, so pointing it at the working tree pollutes the analysis with
gitignored build output (bin/, build/, node_modules/, ...). Git itself
resolves the full .gitignore cascade, so this script materializes exactly
`git ls-files` (as hardlinks, falling back to copies) into a scratch
directory and scripts/generate-deepwiki.sh points deepwiki-rs at that.

Environment variables:
    REPO_ROOT  repository root, used as cwd for `git ls-files`
    TARGET     subtree relative to REPO_ROOT ("." or "" = whole repo)
    SHADOW     destination directory to populate (rebuilt every run)

Prints the number of entries linked to stdout.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys


def main() -> None:
    repo = os.environ["REPO_ROOT"]
    target = os.environ.get("TARGET", ".")
    shadow = os.environ["SHADOW"]

    cmd = ["git", "ls-files", "-z"]
    if target not in (".", ""):
        cmd += ["--", target]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write("git ls-files failed\n")
        sys.exit(1)

    count = 0
    for rel in proc.stdout.decode("utf-8", "surrogateescape").split("\0"):
        if not rel:
            continue
        src = os.path.join(repo, rel)
        dst_rel = os.path.relpath(rel, target) if target not in (".", "") else rel
        if dst_rel.startswith(".."):
            continue
        dst = os.path.join(shadow, dst_rel)
        parent = os.path.dirname(dst)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            os.link(src, dst)
            count += 1
        except OSError:
            try:
                shutil.copy2(src, dst)
                count += 1
            except OSError:
                # submodule gitlinks and other non-files - skip
                pass

    sys.stdout.write(str(count))


if __name__ == "__main__":
    main()