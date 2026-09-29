# Image Composition & Build-Time Toolchain

*module · agent map*

Multi-stage vLLM image build with MXFP4 variant, including JIT provider wheel building, source patching via idempotent Python scripts, and validation executed during Docker build. The module encompasses Dockerfiles, build scripts, and over 20 targeted patch files that modify…

The build process starts from a base CUDA/PyTorch image, installs vLLM and dependencies, then applies a series of patches via COPY+run steps. Patches include: fixing EAGLE embedding, disabling MiniMax fusion, pinning CUTLASS DSL, enabling Tensor causal masks, restoring SWA block…

**Location:** [Home](../../index.md) › **Image Composition & Build-Time Toolchain**

## Diagrams

- [Flowchart](flowchart.md)
- [Sequence](sequence.md)

## Source

- [`Dockerfile`](../../../../.litho/tree/repo/Dockerfile)
- [`Dockerfile.mxfp4`](../../../../.litho/tree/repo/Dockerfile.mxfp4)
- [`build-and-copy.sh`](../../../../.litho/tree/repo/build-and-copy.sh)
- [`docker/build_flashinfer_jit_providers.sh`](../../../../.litho/tree/repo/docker/build_flashinfer_jit_providers.sh)
- [`docker/patch_vllm_flashinfer_b12x_swigluoai.py`](../../../../.litho/tree/repo/docker/patch_vllm_flashinfer_b12x_swigluoai.py)
- [`docker/patch_b12x_cache_integrity.py`](../../../../.litho/tree/repo/docker/patch_b12x_cache_integrity.py)
- [`docker/validate_flashinfer_wheels.py`](../../../../.litho/tree/repo/docker/validate_flashinfer_wheels.py)
- [`fastsafetensors.patch`](../../../../.litho/tree/repo/fastsafetensors.patch)
- [`fastsafetensors_mxfp4.patch`](../../../../.litho/tree/repo/fastsafetensors_mxfp4.patch)
- [`docker/patch_instanttensor_vllm_memory.py`](../../../../.litho/tree/repo/docker/patch_instanttensor_vllm_memory.py)
- [`docker/patch_torch_schema_enumeration.py`](../../../../.litho/tree/repo/docker/patch_torch_schema_enumeration.py)
- [`docker/patch_vllm_autogptq_symmetric_moe_qzeros.py`](../../../../.litho/tree/repo/docker/patch_vllm_autogptq_symmetric_moe_qzeros.py)
- [`docker/patch_vllm_b12x_c128a_topk_alignment.py`](../../../../.litho/tree/repo/docker/patch_vllm_b12x_c128a_topk_alignment.py)
- [`docker/patch_vllm_b12x_moe_tuning_memory.py`](../../../../.litho/tree/repo/docker/patch_vllm_b12x_moe_tuning_memory.py)
- [`docker/patch_vllm_diffusion_tensor_causal.py`](../../../../.litho/tree/repo/docker/patch_vllm_diffusion_tensor_causal.py)
- [`docker/patch_vllm_disable_minimax_qk_rmsnorm_ipc.py`](../../../../.litho/tree/repo/docker/patch_vllm_disable_minimax_qk_rmsnorm_ipc.py)
- [`docker/patch_vllm_gemma4_mtp_embedding_share.py`](../../../../.litho/tree/repo/docker/patch_vllm_gemma4_mtp_embedding_share.py)
- [`docker/patch_vllm_mrv2_speculator_cudagraph_pool.py`](../../../../.litho/tree/repo/docker/patch_vllm_mrv2_speculator_cudagraph_pool.py)
- [`docker/patch_vllm_preserve_sm12x_target.py`](../../../../.litho/tree/repo/docker/patch_vllm_preserve_sm12x_target.py)
- [`docker/patch_vllm_routed_experts_weight_shape.py`](../../../../.litho/tree/repo/docker/patch_vllm_routed_experts_weight_shape.py)
- [`docker/patch_vllm_sm120_cooperative_topk.py`](../../../../.litho/tree/repo/docker/patch_vllm_sm120_cooperative_topk.py)
- [`docker/patch_vllm_spark_kv_cache_cleanup.py`](../../../../.litho/tree/repo/docker/patch_vllm_spark_kv_cache_cleanup.py)
- [`docker/patch_vllm_startup_heap_trim.py`](../../../../.litho/tree/repo/docker/patch_vllm_startup_heap_trim.py)
- [`docker/patch_vllm_swa_block_size.py`](../../../../.litho/tree/repo/docker/patch_vllm_swa_block_size.py)
- [`docker/patch_vllm_topk_softplus_sqrt_control_flow.py`](../../../../.litho/tree/repo/docker/patch_vllm_topk_softplus_sqrt_control_flow.py)
- [`docker/patch_vllm_wsl_cuda_uma.py`](../../../../.litho/tree/repo/docker/patch_vllm_wsl_cuda_uma.py)
- [`docker/pin_cutlass_dsl.py`](../../../../.litho/tree/repo/docker/pin_cutlass_dsl.py)
- [`docker/b12x-cache-integrity.patch`](../../../../.litho/tree/repo/docker/b12x-cache-integrity.patch)

## Interaction

- The Dockerfiles (Dockerfile, Dockerfile.mxfp4) define multi-stage builds; during build, shell scripts (build-and-copy.sh, build_flashinfer_jit_providers.sh) and Python patch scripts (e.g., patch_vllm_flashinfer_b12x_swigluoai.py) are invoked as RUN commands. Patch scripts modify…
