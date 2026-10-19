#!/usr/bin/env python3
"""Let B12X accept a hybrid-aligned KV *storage* block it can split.

On a hybrid model (e.g. Qwen3_5's Gated DeltaNet + full attention),
``Platform._align_hybrid_block_size`` enlarges ``cache_config.block_size`` so one
attention page covers the (larger) mamba page. B12X's alignment quantum is 64,
so the Qwen3.8-27B-Coder390 TP=2 shape lands on 1664 tokens — which is
13 x 128, i.e. perfectly splittable into B12X's supported kernel pages.

vLLM already implements that split: ``prepare_kernel_block_sizes`` asks
``select_common_block_size`` for a kernel page dividing the manager block, and
threads the result through ``AttentionGroup.kernel_block_size`` with a remapped
block table. Measured against the shipped image:

    select_common_block_size(1664, [B12xPagedAttentionBackend]) -> 128

So the kernel layer never sees 1664 and needs no change. Two sites do compare a
*storage* size against the *kernel page* set, though, and both reject 1664:

1. ``B12xPagedAttentionBackend.supports_block_size`` — reached from
   ``Platform._check_aligned_block_size``, which raises
   "Aligned KV cache block size 1664 is not supported by attention backend(s)".
2. ``B12xPagedAttentionImpl.__init__`` — reads ``cache_config.block_size`` and
   raises "b12x requires --block-size in (64, 128), got 1664". Confirmed
   reachable by constructing the impl with a 1664 config.

Both should accept a storage size that a supported kernel page divides.

This deliberately leaves ``get_supported_kernel_block_sizes`` returning the
exact ``[64, 128]``. Declaring ``MultipleOf(64)`` there would REGRESS the arm:
``_block_size_is_supported`` then reports the storage size itself as supported,
``select_common_block_size`` short-circuits and returns 1664, splitting is
disabled and the kernels get a 1664-token page — the exact failure this patch
is meant to remove.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path


MARKER = "spark-vllm-docker: accept splittable storage block"

TARGET_REL = Path("v1/attention/backends/b12x.py")

SUPPORTS_ANCHOR = """    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or int(block_size) in _B12X_SUPPORTED_PAGE_SIZES
"""

SUPPORTS_REPLACEMENT = """    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        # spark-vllm-docker: accept splittable storage block.
        # A hybrid-aligned storage block may exceed every kernel page size: the
        # KV manager splits it into kernel blocks (select_common_block_size ->
        # prepare_kernel_block_sizes). Accept exactly those sizes, so
        # _check_aligned_block_size stops rejecting a splittable layout while a
        # size no supported page divides still fails closed.
        if block_size is None:
            return True
        size = int(block_size)
        return size in _B12X_SUPPORTED_PAGE_SIZES or any(
            size % page == 0 for page in _B12X_SUPPORTED_PAGE_SIZES
        )
"""

IMPL_ANCHOR = """        default_block_size = int(cache_config.block_size)
        if default_block_size not in _B12X_SUPPORTED_PAGE_SIZES:
            raise ValueError(
                "b12x requires --block-size in "
                f"{_B12X_SUPPORTED_PAGE_SIZES}, got "
                f"{cache_config.block_size}."
            )
"""

IMPL_REPLACEMENT = """        # spark-vllm-docker: accept splittable storage block.
        # cache_config.block_size is the KV *storage* block, which hybrid models
        # align up to cover the mamba page. It is not the kernel page size: the
        # manager splits it into kernel blocks and the paged kernels only ever
        # see those. Accept a storage size that a supported kernel page divides.
        default_block_size = int(cache_config.block_size)
        if default_block_size not in _B12X_SUPPORTED_PAGE_SIZES and not any(
            default_block_size % page == 0 for page in _B12X_SUPPORTED_PAGE_SIZES
        ):
            raise ValueError(
                "b12x requires a --block-size that is, or is a multiple of, a "
                f"supported kernel page in {_B12X_SUPPORTED_PAGE_SIZES}; got "
                f"{cache_config.block_size}."
            )
"""

LAYOUTS = (
    ("supports_block_size", SUPPORTS_ANCHOR, SUPPORTS_REPLACEMENT),
    ("impl block-size guard", IMPL_ANCHOR, IMPL_REPLACEMENT),
)


class PatchError(RuntimeError):
    """The source has an unexpected or partially patched layout."""


def patch_source(source: str) -> tuple[str, str]:
    if MARKER in source:
        present = sum(source.count(r) for _, _, r in LAYOUTS)
        anchors = sum(source.count(a) for _, a, _ in LAYOUTS)
        if present != len(LAYOUTS) or anchors:
            raise PatchError("B12X splittable-storage patch is incomplete")
        return source, "B12X splittable-storage patch already present; skipping"

    if "class B12xPagedAttentionBackend" not in source:
        return source, "B12X attention backend is absent; skipping"

    patched = source
    for name, anchor, replacement in LAYOUTS:
        count = patched.count(anchor)
        if count != 1:
            raise PatchError(
                f"expected exactly one {name} block in b12x.py, found {count}; "
                "review the upstream implementation before updating this patch"
            )
        patched = patched.replace(anchor, replacement, 1)

    ast.parse(patched)
    return patched, "Accepted splittable hybrid storage blocks in B12X"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path)
    parser.add_argument("--installed", action="store_true", help="Patch installed vLLM")
    args = parser.parse_args()
    if args.installed:
        if args.source_root is not None:
            parser.error("Use either source_root or --installed")
        spec = importlib.util.find_spec("vllm")
        locations = list(spec.submodule_search_locations or []) if spec else []
        if len(locations) != 1:
            raise SystemExit("Unable to locate the installed vLLM package")
        package_root = Path(locations[0])
    else:
        package_root = (args.source_root or Path.cwd()) / "vllm"

    target = package_root / TARGET_REL
    if not target.exists():
        print(f"{TARGET_REL} is absent; B12X storage-block patch not applicable")
        return
    source = target.read_text()
    try:
        patched, message = patch_source(source)
    except (PatchError, SyntaxError) as exc:
        raise SystemExit(f"B12X storage-block patch failed: {exc}") from exc
    if patched != source:
        target.write_text(patched)
    print(message)


if __name__ == "__main__":
    main()
