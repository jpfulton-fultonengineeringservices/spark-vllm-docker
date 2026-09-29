# Vendored FlashAttention Kernel Library

*module · agent map*

CuTe-DSL kernel implementations covering forward and backward passes across SM80/SM90/SM100/SM120 architectures, MLA variants, split-K combine, and TMA-accelerated variants.

Each kernel class is implemented using CuTe-DSL (CUTLASS Python DSL), leveraging cpasync, tcgen05, TMA, and warp-specialized pipelines to achieve high-performance attention on NVIDIA GPUs. The forward kernels include FlashAttentionForwardBase, FlashAttentionForwardSm90,…

**Location:** [Home](../../index.md) › **Vendored FlashAttention Kernel Library**

## Diagrams

- [Flowchart](flowchart.md)
- [Sequence](sequence.md)

## Source

- [`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd.py`](../../../../.litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd.py)
- [`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm90.py`](../../../../.litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm90.py)
- [`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm100.py`](../../../../.litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm100.py)
- [`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm120.py`](../../../../.litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm120.py)
- [`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm120_tma.py`](../../../../.litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm120_tma.py)
- [`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_mla_sm100.py`](../../../../.litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_mla_sm100.py)
- [`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_combine.py`](../../../../.litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_combine.py)
- [`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd.py`](../../../../.litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd.py)
- [`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm90.py`](../../../../.litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm90.py)
- [`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm100.py`](../../../../.litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm100.py)
- [`mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm120.py`](../../../../.litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm120.py)

## Interaction

- The module provides kernel classes that are instantiated and invoked by the PyTorch-facing autograd interface (FlashAttnFunc/FlashAttnVarlenFunc) defined in interface.py. Additionally, the vLLM dispatch integration (patch_inkling.py, adapter.py) routes SM120 devices to use these…
