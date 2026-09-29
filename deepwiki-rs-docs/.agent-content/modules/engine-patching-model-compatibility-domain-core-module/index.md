# Engine Patching & Model Compatibility Domain Core Module

*module · agent map*

The core module of the Engine Patching & Model Compatibility Domain encompasses three submodules: Model-Specific Fix Mods, Feature Enablement Mods, and vLLM Core Patch Set. It applies targeted source patches, AST-based rewriters, and chat templates to vLLM (and related…

Implementation details: The patching infrastructure uses idempotent application of unified diffs via patch or git apply, Python AST-based source modification for precise changes, and Jinja chat template injection. Each model fix is in mods/fix-* with its own run.sh. Core patches…

**Location:** [Home](../../index.md) › **Engine Patching & Model Compatibility Domain Core Module**

## Diagrams

- [Flowchart](flowchart.md)
- [Sequence](sequence.md)

## Source

- [`mods/fix-qwen3-next-autoround/patch_qwen3_next.py`](../../../../.litho/tree/repo/mods/fix-qwen3-next-autoround/patch_qwen3_next.py)
- [`mods/fix-qwen3-next-autoround/run.sh`](../../../../.litho/tree/repo/mods/fix-qwen3-next-autoround/run.sh)
- [`mods/fix-qwen3.5-chat-template/chat_template.jinja`](../../../../.litho/tree/repo/mods/fix-qwen3.5-chat-template/chat_template.jinja)
- [`mods/fix-qwen3.6-chat-template/chat_template.jinja`](../../../../.litho/tree/repo/mods/fix-qwen3.6-chat-template/chat_template.jinja)
- [`mods/fix-glm-4.7-flash-AWQ`](../../../../.litho/tree/repo/mods/fix-glm-4.7-flash-AWQ)
- [`mods/fix-qwen3-coder-next`](../../../../.litho/tree/repo/mods/fix-qwen3-coder-next)
- [`mods/fix-qwen35-tp4-marlin`](../../../../.litho/tree/repo/mods/fix-qwen35-tp4-marlin)
- [`mods/fix-Salyut1-GLM-4.7-NVFP4`](../../../../.litho/tree/repo/mods/fix-Salyut1-GLM-4.7-NVFP4)
- [`mods/fix-eagle-fine-prefix`](../../../../.litho/tree/repo/mods/fix-eagle-fine-prefix)
- [`mods/fix-gemma4-tool-parser`](../../../../.litho/tree/repo/mods/fix-gemma4-tool-parser)
- [`mods/fix-qwen3.5-autoround`](../../../../.litho/tree/repo/mods/fix-qwen3.5-autoround)
- [`mods/diffusiongemma/run.sh`](../../../../.litho/tree/repo/mods/diffusiongemma/run.sh)
- [`mods/diffusiongemma/diffusiongemma-support.patch`](../../../../.litho/tree/repo/mods/diffusiongemma/diffusiongemma-support.patch)
- [`mods/diffusiongemma/diffusiongemma-attention-main.patch`](../../../../.litho/tree/repo/mods/diffusiongemma/diffusiongemma-attention-main.patch)
- [`mods/diffusiongemma/diffusiongemma-attention-legacy.patch`](../../../../.litho/tree/repo/mods/diffusiongemma/diffusiongemma-attention-legacy.patch)
- [`mods/diffusiongemma/gemma4-content-channel-sanitizer.patch`](../../../../.litho/tree/repo/mods/diffusiongemma/gemma4-content-channel-sanitizer.patch)
- [`mods/diffusiongemma/gemma4-streaming-reasoning.patch`](../../../../.litho/tree/repo/mods/diffusiongemma/gemma4-streaming-reasoning.patch)
- [`mods/diffusiongemma/chat_template_no_think.jinja`](../../../../.litho/tree/repo/mods/diffusiongemma/chat_template_no_think.jinja)
- [`mods/instanttensor-zero-copy/patch_weight_utils.py`](../../../../.litho/tree/repo/mods/instanttensor-zero-copy/patch_weight_utils.py)
- [`mods/instanttensor-hybrid-draft-loader/patch_model_loader.py`](../../../../.litho/tree/repo/mods/instanttensor-hybrid-draft-loader/patch_model_loader.py)
- [`mods/kv-cache-prealloc-cleanup/run.sh`](../../../../.litho/tree/repo/mods/kv-cache-prealloc-cleanup/run.sh)
- [`mods/gpu-mem-util-gb/gpu_mem.patch`](../../../../.litho/tree/repo/mods/gpu-mem-util-gb/gpu_mem.patch)
- [`mods/mimo-diffkv-fp8-kv/run.sh`](../../../../.litho/tree/repo/mods/mimo-diffkv-fp8-kv/run.sh)
- [`mods/exp-b12x/run.sh`](../../../../.litho/tree/repo/mods/exp-b12x/run.sh)
- [`mods/exp-w4a16/run.sh`](../../../../.litho/tree/repo/mods/exp-w4a16/run.sh)
- [`mods/drop-caches/run.sh`](../../../../.litho/tree/repo/mods/drop-caches/run.sh)
- [`docker/patch_vllm_flashinfer_b12x_swigluoai.py`](../../../../.litho/tree/repo/docker/patch_vllm_flashinfer_b12x_swigluoai.py)
- [`docker/patch_b12x_cache_integrity.py`](../../../../.litho/tree/repo/docker/patch_b12x_cache_integrity.py)
- [`docker/patch_vllm_spark_kv_cache_cleanup.py`](../../../../.litho/tree/repo/docker/patch_vllm_spark_kv_cache_cleanup.py)
- [`docker/patch_vllm_startup_heap_trim.py`](../../../../.litho/tree/repo/docker/patch_vllm_startup_heap_trim.py)
- [`docker/patch_vllm_sm120_cooperative_topk.py`](../../../../.litho/tree/repo/docker/patch_vllm_sm120_cooperative_topk.py)
- [`docker/patch_vllm_swa_block_size.py`](../../../../.litho/tree/repo/docker/patch_vllm_swa_block_size.py)
- [`docker/patch_vllm_routed_experts_weight_shape.py`](../../../../.litho/tree/repo/docker/patch_vllm_routed_experts_weight_shape.py)
- [`docker/patch_instanttensor_vllm_memory.py`](../../../../.litho/tree/repo/docker/patch_instanttensor_vllm_memory.py)
- [`docker/pin_cutlass_dsl.py`](../../../../.litho/tree/repo/docker/pin_cutlass_dsl.py)
- [`docker/validate_flashinfer_wheels.py`](../../../../.litho/tree/repo/docker/validate_flashinfer_wheels.py)
- [`docker/b12x-cache-integrity.patch`](../../../../.litho/tree/repo/docker/b12x-cache-integrity.patch)
- [`docker/build_flashinfer_jit_providers.sh`](../../../../.litho/tree/repo/docker/build_flashinfer_jit_providers.sh)
- [`docker/patch_torch_schema_enumeration.py`](../../../../.litho/tree/repo/docker/patch_torch_schema_enumeration.py)
- [`docker/patch_vllm_autogptq_symmetric_moe_qzeros.py`](../../../../.litho/tree/repo/docker/patch_vllm_autogptq_symmetric_moe_qzeros.py)
- [`docker/patch_vllm_b12x_c128a_topk_alignment.py`](../../../../.litho/tree/repo/docker/patch_vllm_b12x_c128a_topk_alignment.py)
- [`docker/patch_vllm_b12x_moe_tuning_memory.py`](../../../../.litho/tree/repo/docker/patch_vllm_b12x_moe_tuning_memory.py)
- [`docker/patch_vllm_diffusion_tensor_causal.py`](../../../../.litho/tree/repo/docker/patch_vllm_diffusion_tensor_causal.py)
- [`docker/patch_vllm_disable_minimax_qk_rmsnorm_ipc.py`](../../../../.litho/tree/repo/docker/patch_vllm_disable_minimax_qk_rmsnorm_ipc.py)
- [`docker/patch_vllm_gemma4_mtp_embedding_share.py`](../../../../.litho/tree/repo/docker/patch_vllm_gemma4_mtp_embedding_share.py)
- [`docker/patch_vllm_mrv2_speculator_cudagraph_pool.py`](../../../../.litho/tree/repo/docker/patch_vllm_mrv2_speculator_cudagraph_pool.py)
- [`docker/patch_vllm_preserve_sm12x_target.py`](../../../../.litho/tree/repo/docker/patch_vllm_preserve_sm12x_target.py)
- [`docker/patch_vllm_topk_softplus_sqrt_control_flow.py`](../../../../.litho/tree/repo/docker/patch_vllm_topk_softplus_sqrt_control_flow.py)
- [`docker/patch_vllm_wsl_cuda_uma.py`](../../../../.litho/tree/repo/docker/patch_vllm_wsl_cuda_uma.py)

## Interaction

- Interfaces are provided through Bash orchestration scripts (run.sh) that apply patches to installed Python packages, Python patching scripts (patch_*.py) using AST rewriting or git apply, and patch/diff files. Environment variables control conditional patching. The system…
