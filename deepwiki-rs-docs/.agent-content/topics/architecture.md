# Architecture

*topic · agent map*

Structural view: how the system is decomposed into domains and layers.

**Location:** [Home](../index.md) › **Architecture**

## Diagrams

- [Domain modules](diagrams/architecture-domains.md)
- [Domain dependencies](../diagrams/domain-dependencies.md)

## Summary

- spark-vllm-docker is a patch-and-orchestration layer around the open-source vLLM serving engine, purpose-built for Spark clusters with NVIDIA Blackwell GPUs. Architecturally it is organized as: (1) a Core Attention Kernel domain providing a vendored CuTe-DSL FlashAttention…
