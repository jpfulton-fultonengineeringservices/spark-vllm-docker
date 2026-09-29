# Memory Profiling & Capacity Domain

*module · agent map*

A comprehensive solution for instrumenting vLLM worker startup to capture per-phase memory usage (native heap, CUDA memory, model/KV inventory) via AST-based patching, consolidating the collected JSONL events into consolidated YAML profile cards, and performing capacity analysis…

The implementation uses Python with AST-based source rewriting (patch.py) to inject a profiler marker string and optional import into vLLM modules. The profiler (probe.py) reads /proc/meminfo and uses ctypes for CUDA driver queries, records mallinfo and CUDA memory at each…

**Location:** [Home](../../index.md) › **Memory Profiling & Capacity Domain**

## Diagrams

- [Flowchart](flowchart.md)
- [Sequence](sequence.md)

## Source

- [`mods/memory-profile/probe.py`](../../../../.litho/tree/repo/mods/memory-profile/probe.py)
- [`mods/memory-profile/patch.py`](../../../../.litho/tree/repo/mods/memory-profile/patch.py)
- [`mods/memory-profile/run.sh`](../../../../.litho/tree/repo/mods/memory-profile/run.sh)
- [`mods/memory-profile/collect.py`](../../../../.litho/tree/repo/mods/memory-profile/collect.py)
- [`mods/memory-profile/profile_card.py`](../../../../.litho/tree/repo/mods/memory-profile/profile_card.py)
- [`mods/memory-profile/capacity.py`](../../../../.litho/tree/repo/mods/memory-profile/capacity.py)
- [`mods/memory-profile/report.py`](../../../../.litho/tree/repo/mods/memory-profile/report.py)
- [`mods/memory-profile/host_probe.py`](../../../../.litho/tree/repo/mods/memory-profile/host_probe.py)

## Interaction

- The module interacts with external systems through: 1) AST-based source rewriting to inject the profiler into the vLLM package without importing vLLM or initializing CUDA (patch.py and run.sh). 2) A standalone CPU-only host sampler and observer context within the vLLM worker…
