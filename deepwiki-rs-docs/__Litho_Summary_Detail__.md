# Project Analysis Summary Report (Full Version)

Generation Time: 2026-09-29 15:15:41 UTC

## Execution Timing Statistics

- **Total Execution Time**: 3583.16 seconds
- **Preprocessing Phase**: 0.48 seconds (0.0%)
- **Research Phase**: 635.02 seconds (17.7%)
- **Document Generation Phase**: 2947.66 seconds (82.3%)
- **Output Phase**: 0.00 seconds (0.0%)
- **Summary Generation Time**: 0.001 seconds

## Cache Performance Statistics and Savings

### Performance Metrics
- **Cache Hit Rate**: 68.1%
- **Total Operations**: 69
- **Cache Hits**: 47 times
- **Cache Misses**: 22 times
- **Cache Writes**: 23 times

### Savings
- **Inference Time Saved**: 385.8 seconds
- **Tokens Saved**: 96332 input + 79750 output = 176082 total
- **Estimated Cost Savings**: $0.1620
- **Performance Improvement**: 68.1%
- **Efficiency Improvement Ratio**: 0.1x (saved time / actual execution time)

## Core Research Data Summary

Complete content of four types of research materials according to Prompt template data integration rules:

### System Context Research Report
Provides core objectives, user roles, and system boundary information for the project.

```json
{
  "business_value": "Enables efficient deployment of large language models on Spark clusters with NVIDIA Blackwell GPUs by providing hardware-specific optimizations, memory profiling, and targeted fixes for model architectures.",
  "confidence_score": 7.0,
  "external_systems": [
    {
      "description": "Open-source LLM serving engine that is patched and extended by this project.",
      "interaction_type": "patches and extends source code",
      "name": "vLLM"
    },
    {
      "description": "Library for high-performance attention kernels, used as a dependency in patching.",
      "interaction_type": "provides kernel source and JIT compilation",
      "name": "FlashInfer"
    },
    {
      "description": "NVIDIA's CUDA architecture for Blackwell GPUs (SM12), target hardware for kernel customization.",
      "interaction_type": "target hardware platform",
      "name": "NVIDIA Blackwell GPU (SM120)"
    },
    {
      "description": "Model repository used for offline weight access in cluster deployments.",
      "interaction_type": "model weight source",
      "name": "Hugging Face Hub"
    }
  ],
  "project_description": "A collection of patches, custom kernel implementations, memory profiling tools, and deployment recipes to optimize vLLM for Spark clusters equipped with NVIDIA Blackwell GPUs. Includes vendor kernel packages for FlashAttention on SM120, memory capacity analysis, and model-specific fixes for architectures like Qwen3, Gemma, and DiffusionGemma.",
  "project_name": "spark-vllm-docker",
  "project_type": "Other",
  "system_boundary": {
    "excluded_components": [
      "Core vLLM development",
      "Model training pipelines",
      "Hardware design and drivers",
      "Spark cluster orchestration and management"
    ],
    "included_components": [
      "Docker configuration for building custom vLLM images",
      "Kernel patches for FlashAttention on SM120/SM100 GPUs",
      "Patching scripts for specific model architectures (Qwen3, Gemma4, etc.)",
      "Memory profiling and capacity analysis modules",
      "Weight verification and offline model setup scripts",
      "Chat template fixes for Qwen3.5 and DiffusionGemma",
      "Example scripts and cluster deployment recipes (3x, 4x, 8x Spark clusters)"
    ],
    "scope": "Optimizing vLLM serving engine for Spark clusters with Blackwell GPUs through source-level patches, custom attention kernels, and deployment tooling."
  },
  "target_users": [
    {
      "description": "Engineers responsible for deploying and optimizing LLM inference on Spark clusters with NVIDIA Blackwell hardware.",
      "name": "ML Infrastructure Engineer",
      "needs": [
        "Apply hardware-specific optimizations to vLLM for Blackwell GPUs",
        "Fix model compatibility issues with vLLM for Qwen3, Gemma, etc.",
        "Profile and estimate memory capacity for cluster nodes",
        "Verify weight integrity for offline model serving",
        "Deploy vLLM in Spark cluster environments with custom chat templates"
      ]
    },
    {
      "description": "Researchers or engineers working on novel model architectures that require patching vLLM for support.",
      "name": "AI Model Developer",
      "needs": [
        "Integrate custom quantization support (e.g., NVFP4) for MoE routers",
        "Apply patches to support emerging model variants (e.g., DiffusionGemma, Qwen3.5)",
        "Leverage advanced attention kernels (MLA) for Blackwell hardware"
      ]
    }
  ]
}
```

### Domain Modules Research Report
Provides high-level domain division, module relationships, and core business process information.

```json
{
  "architecture_summary": "spark-vllm-docker is a patch-and-orchestration layer around the open-source vLLM serving engine, purpose-built for Spark clusters with NVIDIA Blackwell GPUs. Architecturally it is organized as: (1) a Core Attention Kernel domain providing a vendored CuTe-DSL FlashAttention package (SM90/SM100/SM120 forward, backward, MLA, split-K, paged-KV) integrated into vLLM through an AST-based dispatch patch; (2) a Core Engine Patching domain comprising a shared build-time patch library under docker/ plus per-model fix and feature mods under mods/, which restore quantization, chat template, and architecture support for Qwen, Gemma, DiffusionGemma, GLM, Nemotron and others; (3) a Core Deployment domain of declarative YAML recipes and shell/python runners (run-recipe, launch-cluster) that sequence mods and launch 3x/4x/8x cluster deployments; (4) a Supporting Memory Profiling domain that instruments vLLM startup, consolidates profile cards, and performs topology-aware capacity validation; (5) a Supporting Model Weights domain providing offline weight verification and HF hub-cache setup; (6) an Infrastructure Container Build domain (Dockerfile, Dockerfile.mxfp4, build scripts) that composes the patched image; and (7) minimal Developer Tooling (fes-eval, DeepWiki docs). The dominant interaction style is non-invasive source patching (idempotent AST/text rewrites with mod markers) of an external vLLM dependency, with recipes acting as the composition root that binds mods, weights, kernels, and cluster launch together. Code structure aligns well with the documented system boundary; the main observed misalignment is that engine-level fixes are split across two locations (build-time patches in docker/ and runtime mods in mods/fix-*), which are conceptually one 'Engine Compatibility' concern but physically separated by when they execute.",
  "business_flows": [
    {
      "description": "End-to-end path from selecting a model recipe to serving on a Spark cluster: parse recipe, verify offline weights, apply required fix/feature mods to the installed vLLM package, then launch head/worker nodes with the correct chat template and quantization configuration.",
      "entry_point": "run-recipe.sh",
      "importance": 9.5,
      "involved_domains_count": 4,
      "name": "Recipe-Driven Cluster Model Deployment Flow",
      "steps": [
        {
          "code_entry_point": "run-recipe.py",
          "domain_module": "Deployment Recipes & Cluster Orchestration Domain",
          "operation": "Select and parse a model/cluster recipe YAML (e.g., 4x-spark-cluster/qwen3.5-397b-int4-autoround.yaml) to determine mods, parallelism, and quantization.",
          "step": 1,
          "sub_module": "Recipe Catalog"
        },
        {
          "code_entry_point": "mods/fes-weights/run.sh",
          "domain_module": "Model Weights & Offline Serving Domain",
          "operation": "Verify staged model weights and construct the HF hub-cache layout for offline serving.",
          "step": 2,
          "sub_module": "Offline Hub Cache Setup"
        },
        {
          "code_entry_point": "mods/fes-weights/verify.py",
          "domain_module": "Model Weights & Offline Serving Domain",
          "operation": "Validate shards against safetensors index, size parity, and quantization config; fail fast on mismatch.",
          "step": 3,
          "sub_module": "Weight Verification"
        },
        {
          "code_entry_point": "mods/fix-qwen3-next-autoround/patch_qwen3_next.py",
          "domain_module": "Engine Patching & Model Compatibility Domain",
          "operation": "Apply model-specific patches and chat templates (AST rewrites, .patch/.diff files) to the installed vLLM/transformers sources.",
          "step": 4,
          "sub_module": "Model-Specific Fix Mods"
        },
        {
          "code_entry_point": "mods/diffusiongemma/run.sh",
          "domain_module": "Engine Patching & Model Compatibility Domain",
          "operation": "Run feature mods (DiffusionGemma support, zero-copy weights, KV-cache cleanup) that patch and then optionally launch the engine.",
          "step": 5,
          "sub_module": "Feature Enablement Mods"
        },
        {
          "code_entry_point": "launch-cluster.sh",
          "domain_module": "Deployment Recipes & Cluster Orchestration Domain",
          "operation": "Launch head and worker nodes across the Spark cluster with the patched container.",
          "step": 6,
          "sub_module": "Recipe Runners & Cluster Launchers"
        },
        {
          "code_entry_point": "mods/inkling-sm12-paged-kv/patch_inkling.py",
          "domain_module": "Attention Kernel Domain",
          "operation": "At engine start, the FA4 dispatch patch routes SM12.x devices to the vendored paged-KV attention kernel.",
          "step": 7,
          "sub_module": "vLLM Dispatch Integration"
        }
      ]
    },
    {
      "description": "Builds the custom vLLM Docker image by installing vLLM and FlashInfer, applying the shared engine patch library, building FlashInfer JIT-cache provider wheels, and validating wheels before producing the final (base or MXFP4) image.",
      "entry_point": "Dockerfile",
      "importance": 8.5,
      "involved_domains_count": 3,
      "name": "Custom Container Image Build Flow",
      "steps": [
        {
          "code_entry_point": "build-and-copy.sh",
          "domain_module": "Container Build & Image Composition Domain",
          "operation": "Invoke the Dockerfile build (base or MXFP4 variant) and copy resulting image artifacts.",
          "step": 1,
          "sub_module": "Image Definitions"
        },
        {
          "code_entry_point": "docker/build_flashinfer_jit_providers.sh",
          "domain_module": "Container Build & Image Composition Domain",
          "operation": "Build FlashInfer JIT-cache provider wheels for the target CUDA architectures.",
          "step": 2,
          "sub_module": "Build-Time Patch & JIT Toolchain"
        },
        {
          "code_entry_point": "docker/patch_vllm_flashinfer_b12x_swigluoai.py",
          "domain_module": "Engine Patching & Model Compatibility Domain",
          "operation": "Apply idempotent patches to installed vLLM/FlashInfer/B12X sources (SwiGLU-OAI plumbing, cache integrity, memory trims, top-k controls).",
          "step": 3,
          "sub_module": "vLLM Core Patch Set"
        },
        {
          "code_entry_point": "docker/validate_flashinfer_wheels.py",
          "domain_module": "Container Build & Image Composition Domain",
          "operation": "Validate built FlashInfer wheels and pin CUTLASS DSL versions for reproducibility.",
          "step": 4,
          "sub_module": "Build-Time Patch & JIT Toolchain"
        },
        {
          "code_entry_point": "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4",
          "domain_module": "Attention Kernel Domain",
          "operation": "Vendor the SM120 FlashAttention kernel package into the image for later dispatch integration.",
          "step": 5,
          "sub_module": "Vendored FlashAttention Kernel Library"
        }
      ]
    },
    {
      "description": "Profiles a model's startup memory on the target cluster, consolidates per-rank events into a profile card, and validates that the deployment fits available host memory to guide topology-aware host selection.",
      "entry_point": "mods/memory-profile/run.sh",
      "importance": 8.5,
      "involved_domains_count": 2,
      "name": "Memory Profiling & Capacity Validation Flow",
      "steps": [
        {
          "code_entry_point": "mods/memory-profile/probe.py",
          "domain_module": "Memory Profiling & Capacity Domain",
          "operation": "Install and activate the startup instrumentation plugin inside vLLM to record per-phase memory during worker startup.",
          "step": 1,
          "sub_module": "Startup Memory Probe"
        },
        {
          "code_entry_point": "mods/memory-profile/profile_card.py",
          "domain_module": "Memory Profiling & Capacity Domain",
          "operation": "Collect per-rank event JSONL from run directories and merge into a model/recipe YAML profile card.",
          "step": 2,
          "sub_module": "Profile Collection & Card Generation"
        },
        {
          "code_entry_point": "mods/memory-profile/capacity.py",
          "domain_module": "Memory Profiling & Capacity Domain",
          "operation": "Estimate startup RAM and KV-cache budget, validate against available host memory per topology, and emit capacity metrics.",
          "step": 3,
          "sub_module": "Capacity Analysis & Reporting"
        },
        {
          "code_entry_point": "mods/memory-profile/report.py",
          "domain_module": "Memory Profiling & Capacity Domain",
          "operation": "Probe host hardware and generate the human-readable capacity report.",
          "step": 4,
          "sub_module": "Capacity Analysis & Reporting"
        },
        {
          "code_entry_point": "run-recipe.sh",
          "domain_module": "Deployment Recipes & Cluster Orchestration Domain",
          "operation": "Use capacity findings to choose topology-appropriate hosts when launching the cluster deployment.",
          "step": 5,
          "sub_module": "Recipe Runners & Cluster Launchers"
        }
      ]
    },
    {
      "description": "At inference startup, vLLM's FlashAttention 4 dispatch is patched so that compute-capability 12.x devices route attention work to the vendored paged-KV kernel, which selects architecture-specific forward/MLA/split-K kernels and exposes them through the PyTorch autograd interface.",
      "entry_point": "mods/inkling-sm12-paged-kv/run.sh",
      "importance": 9.0,
      "involved_domains_count": 1,
      "name": "Attention Kernel Dispatch on Blackwell Flow",
      "steps": [
        {
          "code_entry_point": "mods/inkling-sm12-paged-kv/patch_inkling.py",
          "domain_module": "Attention Kernel Domain",
          "operation": "Patch vLLM's NVIDIA FA4 dispatch source to inject a cached _use_sm12_paged_kv capability check.",
          "step": 1,
          "sub_module": "vLLM Dispatch Integration"
        },
        {
          "code_entry_point": "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/interface.py",
          "domain_module": "Attention Kernel Domain",
          "operation": "Validate inputs, compute tile sizes and split-KV heuristics, and wrap kernels in autograd Functions.",
          "step": 2,
          "sub_module": "Kernel Runtime Support"
        },
        {
          "code_entry_point": "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/tile_scheduler.py",
          "domain_module": "Attention Kernel Domain",
          "operation": "Schedule persistent work tiles and prepare paged-KV cache layouts for the launch.",
          "step": 3,
          "sub_module": "Kernel Runtime Support"
        },
        {
          "code_entry_point": "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm100.py",
          "domain_module": "Attention Kernel Domain",
          "operation": "Execute architecture-selected forward (SM90/SM100/SM120/MLA) or backward kernels, including split-K combine when work is split across KV chunks.",
          "step": 4,
          "sub_module": "Vendored FlashAttention Kernel Library"
        }
      ]
    },
    {
      "description": "Onboarding a new or emerging model architecture: author a fix or feature mod (patch, diff, or chat template), register it in a recipe, and have the recipe runner apply it before the vLLM server starts with the corrected template/configuration.",
      "entry_point": "mods/<model-fix>/run.sh",
      "importance": 8.0,
      "involved_domains_count": 2,
      "name": "Model Support Enablement Flow",
      "steps": [
        {
          "code_entry_point": "mods/diffusiongemma/run.sh",
          "domain_module": "Engine Patching & Model Compatibility Domain",
          "operation": "Author model-support patches (e.g., diffusiongemma-support.patch, attention backend patches) and a run.sh orchestrator that applies them with legacy fallbacks.",
          "step": 1,
          "sub_module": "Feature Enablement Mods"
        },
        {
          "code_entry_point": "mods/fix-qwen3.5-chat-template/chat_template.jinja",
          "domain_module": "Engine Patching & Model Compatibility Domain",
          "operation": "Provide corrected chat templates or AST-based rewrites for model-specific serialization and quantization behavior.",
          "step": 2,
          "sub_module": "Model-Specific Fix Mods"
        },
        {
          "code_entry_point": "recipes/diffusion-gemma-nvfp4.yaml",
          "domain_module": "Deployment Recipes & Cluster Orchestration Domain",
          "operation": "Reference the mod from a new recipe YAML describing model, quantization, and cluster size.",
          "step": 3,
          "sub_module": "Recipe Catalog"
        },
        {
          "code_entry_point": "run-recipe.sh",
          "domain_module": "Deployment Recipes & Cluster Orchestration Domain",
          "operation": "Run the recipe so mods are applied in order and the server starts with the custom template.",
          "step": 4,
          "sub_module": "Recipe Runners & Cluster Launchers"
        }
      ]
    }
  ],
  "confidence_score": 7.5,
  "domain_modules": [
    {
      "code_paths": [
        "mods/inkling-sm12-paged-kv"
      ],
      "complexity": 9.5,
      "description": "Provides custom FlashAttention kernel implementations vendored for NVIDIA Blackwell (SM90/SM100/SM120) hardware, including forward/backward/MLA kernels, paged-KV support, tile scheduling, and a PyTorch-facing autograd interface. This is the computational core that extends vLLM's attention capability on Spark cluster GPUs.",
      "domain_type": "Core Business Domain",
      "importance": 9.5,
      "name": "Attention Kernel Domain",
      "sub_modules": [
        {
          "code_paths": [
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm90.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm100.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm120.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm120_tma.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_mla_sm100.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_combine.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm90.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm100.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm120.py"
          ],
          "description": "CuTe-DSL kernel implementations covering forward and backward passes across SM80/SM90/SM100/SM120 architectures, MLA variants, split-K combine, and TMA-accelerated variants.",
          "importance": 9.5,
          "key_functions": [
            "FlashAttentionForwardBase",
            "FlashAttentionForwardSm90",
            "FlashAttentionForwardSm100",
            "FlashAttentionMLAForwardSm100",
            "FlashAttentionBackwardSm80",
            "FlashAttentionBackwardSm90",
            "FlashAttentionBackwardSm100",
            "FlashAttentionForwardCombine"
          ],
          "name": "Vendored FlashAttention Kernel Library"
        },
        {
          "code_paths": [
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/interface.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/paged_kv.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/tile_scheduler.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/mask.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/softmax.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/blackwell_helpers.py",
            "mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/pipeline.py"
          ],
          "description": "Shared infrastructure for kernels: PyTorch autograd interface with tile heuristics and validation, paged-KV cache handling, persistent tile scheduling, masking, softmax, and hardware helper utilities.",
          "importance": 9.0,
          "key_functions": [
            "FwdConfig/BwdConfig parsing and validation",
            "FlashAttnFunc autograd Function",
            "FlashAttnVarlenFunc autograd Function",
            "CLC-based tile scheduling"
          ],
          "name": "Kernel Runtime Support"
        },
        {
          "code_paths": [
            "mods/inkling-sm12-paged-kv/patch_inkling.py",
            "mods/inkling-sm12-paged-kv/adapter.py",
            "mods/inkling-sm12-paged-kv/run.sh"
          ],
          "description": "Patch and adapter layer that routes compute-capability 12.x devices in vLLM's FA4 dispatch path to the vendored paged-KV kernel.",
          "importance": 8.5,
          "key_functions": [
            "AST-based FA4 dispatch patching",
            "_use_sm12_paged_kv capability check injection",
            "mod run entry"
          ],
          "name": "vLLM Dispatch Integration"
        }
      ]
    },
    {
      "code_paths": [
        "mods",
        "docker"
      ],
      "complexity": 7.5,
      "description": "Applies targeted source patches, AST-based rewriters, and chat templates to vLLM (and related libraries) so that specific model families (Qwen3/Qwen3.5/Qwen3.6, Gemma4, DiffusionGemma, GLM-4.7, Nemotron, MiniMax, Step-3.7, Mimo, RadixArk) and experimental features work correctly on the target hardware and quantization schemes.",
      "domain_type": "Core Business Domain",
      "importance": 8.5,
      "name": "Engine Patching & Model Compatibility Domain",
      "sub_modules": [
        {
          "code_paths": [
            "mods/fix-qwen3-next-autoround/patch_qwen3_next.py",
            "mods/fix-qwen3-next-autoround/run.sh",
            "mods/fix-qwen3.5-chat-template/chat_template.jinja",
            "mods/fix-qwen3.6-chat-template/chat_template.jinja",
            "mods/fix-glm-4.7-flash-AWQ",
            "mods/fix-qwen3-coder-next",
            "mods/fix-qwen35-tp4-marlin",
            "mods/fix-Salyut1-GLM-4.7-NVFP4",
            "mods/fix-eagle-fine-prefix",
            "mods/fix-gemma4-tool-parser",
            "mods/fix-qwen3.5-autoround"
          ],
          "description": "Per-model fix modules under mods/fix-*, each with a run.sh entry and patches/diffs restoring correct behavior (quantization, chat templates, crash/slowness fixes).",
          "importance": 8.5,
          "key_functions": [
            "gate_call/patched_text AST rewrite for MoE router quantization",
            "chat template rendering with multimodal validation",
            "model patch application via run.sh"
          ],
          "name": "Model-Specific Fix Mods"
        },
        {
          "code_paths": [
            "mods/diffusiongemma/run.sh",
            "mods/diffusiongemma/*.patch",
            "mods/instanttensor-zero-copy/patch_weight_utils.py",
            "mods/instanttensor-hybrid-draft-loader/patch_model_loader.py",
            "mods/kv-cache-prealloc-cleanup/run.sh",
            "mods/gpu-mem-util-gb/gpu_mem.patch",
            "mods/mimo-diffkv-fp8-kv/run.sh",
            "mods/exp-b12x/run.sh",
            "mods/exp-w4a16/run.sh",
            "mods/drop-caches/run.sh"
          ],
          "description": "Mods that enable additional model architectures or runtime behaviors: DiffusionGemma/Gemma4 support, speculative decoding loaders, KV-cache and GPU memory tuning, drop-caches, and experimental W4A16/B12X flags.",
          "importance": 8.0,
          "key_functions": [
            "Patch-and-launch orchestration (run.sh)",
            "zero-copy tensor view rewrite in weight iterator",
            "hybrid draft model loading"
          ],
          "name": "Feature Enablement Mods"
        },
        {
          "code_paths": [
            "docker/patch_vllm_flashinfer_b12x_swigluoai.py",
            "docker/patch_b12x_cache_integrity.py",
            "docker/patch_vllm_spark_kv_cache_cleanup.py",
            "docker/patch_vllm_startup_heap_trim.py",
            "docker/patch_vllm_sm120_cooperative_topk.py",
            "docker/patch_vllm_swa_block_size.py",
            "docker/patch_vllm_routed_experts_weight_shape.py",
            "docker/patch_instanttensor_vllm_memory.py",
            "docker/pin_cutlass_dsl.py",
            "docker/validate_flashinfer_wheels.py"
          ],
          "description": "Build-time patch scripts applied to installed vLLM/FlashInfer/B12X source trees inside the Docker image: attention backend fixes, memory trims, quantization adjustments, and SWiGLU-OAI parameter plumbing.",
          "importance": 8.5,
          "key_functions": [
            "Idempotent AST/text patching of vLLM sources",
            "FlashInfer B12x SwiGLU-OAI parameter plumbing",
            "cache-integrity patch application"
          ],
          "name": "vLLM Core Patch Set"
        }
      ]
    },
    {
      "code_paths": [
        "recipes",
        "run-recipe.sh",
        "run-recipe.py",
        "launch-cluster.sh",
        "examples"
      ],
      "complexity": 6.0,
      "description": "YAML recipe catalog plus orchestration scripts that describe how to deploy a given model on 3x/4x/8x Spark clusters, selecting which mods to apply and how to launch the vLLM service. Recipes are the primary user-facing deployment artifact and drive mod data flows.",
      "domain_type": "Core Business Domain",
      "importance": 9.0,
      "name": "Deployment Recipes & Cluster Orchestration Domain",
      "sub_modules": [
        {
          "code_paths": [
            "recipes",
            "recipes/3x-spark-cluster",
            "recipes/4x-spark-cluster",
            "recipes/8x-spark-cluster"
          ],
          "description": "Model/cluster recipe YAML files (deepseek, qwen, glm, nemotron, minimax, diffusion-gemma, step, mimo, etc.) for solo and clustered Spark deployments.",
          "importance": 9.0,
          "key_functions": [
            "Declarative model + mod + cluster configuration",
            "quantization and parallelism selection"
          ],
          "name": "Recipe Catalog"
        },
        {
          "code_paths": [
            "run-recipe.sh",
            "run-recipe.py",
            "launch-cluster.sh",
            "autodiscover.sh",
            "examples"
          ],
          "description": "Entry scripts that parse recipes, orchestrate mod execution, and launch head/worker nodes on Spark clusters.",
          "importance": 8.5,
          "key_functions": [
            "recipe parsing and mod sequencing",
            "cluster node bootstrap and launch"
          ],
          "name": "Recipe Runners & Cluster Launchers"
        },
        {
          "code_paths": [
            "mods/use-ngc-vllm/run.sh",
            "mods/use-official-vllm/run.sh"
          ],
          "description": "Mods choosing the base vLLM build (NGC vs official) at launch time.",
          "importance": 5.0,
          "key_functions": [
            "container flavor switching"
          ],
          "name": "vLLM Flavor Selection Mods"
        }
      ]
    },
    {
      "code_paths": [
        "mods/memory-profile"
      ],
      "complexity": 7.0,
      "description": "Instruments vLLM worker startup to collect per-phase memory data, consolidates it into profile cards, and performs capacity analysis to validate that a profiled deployment fits within available host memory — enabling topology-aware host selection for Spark clusters.",
      "domain_type": "Supporting Domain",
      "importance": 8.5,
      "name": "Memory Profiling & Capacity Domain",
      "sub_modules": [
        {
          "code_paths": [
            "mods/memory-profile/probe.py",
            "mods/memory-profile/patch.py",
            "mods/memory-profile/run.sh"
          ],
          "description": "Plugin imported into vLLM as vllm._spark_memory_profile that records phase-by-phase native heap, CUDA memory, and model/KV inventory during worker startup; also runnable as standalone sampler.",
          "importance": 8.5,
          "key_functions": [
            "phase memory instrumentation",
            "CUDA and heap sampling",
            "model/KV inventory capture"
          ],
          "name": "Startup Memory Probe"
        },
        {
          "code_paths": [
            "mods/memory-profile/collect.py",
            "mods/memory-profile/profile_card.py"
          ],
          "description": "Collects per-rank event JSONL from run directories and merges them into a consolidated model/recipe YAML profile card.",
          "importance": 8.0,
          "key_functions": [
            "per-rank JSONL summarization",
            "phase peak computation",
            "YAML profile card emission"
          ],
          "name": "Profile Collection & Card Generation"
        },
        {
          "code_paths": [
            "mods/memory-profile/capacity.py",
            "mods/memory-profile/report.py",
            "mods/memory-profile/host_probe.py"
          ],
          "description": "Estimates startup RAM requirements, budgets KV cache, validates topology-aware host selection against available memory, and generates the human-readable report.",
          "importance": 8.5,
          "key_functions": [
            "KV-cache budgeting",
            "peak host memory estimation",
            "capacity utilization metrics",
            "host topology validation"
          ],
          "name": "Capacity Analysis & Reporting"
        }
      ]
    },
    {
      "code_paths": [
        "mods/fes-weights",
        "hf-download.sh"
      ],
      "complexity": 4.0,
      "description": "Prepares and verifies staged model weights so vLLM can serve fully offline from local storage rather than querying the Hugging Face hub — essential for air-gapped or network-constrained Spark clusters.",
      "domain_type": "Supporting Domain",
      "importance": 7.5,
      "name": "Model Weights & Offline Serving Domain",
      "sub_modules": [
        {
          "code_paths": [
            "mods/fes-weights/verify.py"
          ],
          "description": "Validates staged shard files against safetensors index, checks total size parity, and enforces quantization config presence for NVFP4 checkpoints.",
          "importance": 8.0,
          "key_functions": [
            "shard/index consistency check",
            "size parity tolerance check",
            "hf_quant_config.json validation"
          ],
          "name": "Weight Verification"
        },
        {
          "code_paths": [
            "mods/fes-weights/run.sh",
            "hf-download.sh"
          ],
          "description": "Shell entrypoint that verifies mounted weights and lays out the HF hub-cache structure for offline vLLM resolution.",
          "importance": 7.5,
          "key_functions": [
            "FES_WEIGHTS_DIR defaulting",
            "HF hub-cache layout construction",
            "weight download helper"
          ],
          "name": "Offline Hub Cache Setup"
        }
      ]
    },
    {
      "code_paths": [
        "Dockerfile",
        "Dockerfile.mxfp4",
        "docker",
        "build-and-copy.sh"
      ],
      "complexity": 6.0,
      "description": "Dockerfiles, build scripts, and build-time patch application that compose the custom vLLM container image — bundling FlashInfer JIT providers, vendored kernels, and all source patches into deployable images (including an MXFP4 variant).",
      "domain_type": "Infrastructure Domain",
      "importance": 7.5,
      "name": "Container Build & Image Composition Domain",
      "sub_modules": [
        {
          "code_paths": [
            "Dockerfile",
            "Dockerfile.mxfp4",
            "build-and-copy.sh"
          ],
          "description": "Base and MXFP4 image definitions plus build-and-copy automation.",
          "importance": 7.5,
          "key_functions": [
            "multi-stage vLLM image build",
            "MXFP4 image variant"
          ],
          "name": "Image Definitions"
        },
        {
          "code_paths": [
            "docker/build_flashinfer_jit_providers.sh",
            "docker/patch_vllm_flashinfer_b12x_swigluoai.py",
            "docker/patch_b12x_cache_integrity.py",
            "docker/validate_flashinfer_wheels.py",
            "fastsafetensors.patch",
            "fastsafetensors_mxfp4.patch"
          ],
          "description": "Patch scripts and shell tools executed during image build to modify installed vLLM/FlashInfer/B12X sources and produce FlashInfer JIT-cache provider wheels.",
          "importance": 7.5,
          "key_functions": [
            "JIT provider wheel build",
            "idempotent source patching",
            "wheel validation"
          ],
          "name": "Build-Time Patch & JIT Toolchain"
        }
      ]
    },
    {
      "code_paths": [
        "scripts"
      ],
      "complexity": 2.0,
      "description": "Auxiliary scripts supporting developer workflows: evaluation runners and automated repository documentation generation via DeepWiki.",
      "domain_type": "Tool Support Domain",
      "importance": 3.5,
      "name": "Developer Tooling Domain",
      "sub_modules": [
        {
          "code_paths": [
            "scripts/fes-eval.sh"
          ],
          "description": "Shell entrypoint that invokes FES evaluation weight tooling.",
          "importance": 4.0,
          "key_functions": [
            "fes-weights evaluation invocation"
          ],
          "name": "Evaluation Runner"
        },
        {
          "code_paths": [
            "scripts/deepwiki/generate-deepwiki.sh",
            "scripts/deepwiki/shadow-tree.py"
          ],
          "description": "Generates a shadow documentation tree for the repository source using DeepWiki.",
          "importance": 3.5,
          "key_functions": [
            "shadow tree generation",
            "DeepWiki page publishing"
          ],
          "name": "Documentation Generator"
        }
      ]
    }
  ],
  "domain_relations": [
    {
      "description": "Recipes declare which fix/feature mods must be applied; the recipe runner sequences mods/fix-*/run.sh and mods/<feature>/run.sh before launching vLLM.",
      "from_domain": "Deployment Recipes & Cluster Orchestration Domain",
      "relation_type": "Configuration Dependency",
      "strength": 9.0,
      "to_domain": "Engine Patching & Model Compatibility Domain"
    },
    {
      "description": "fes-weights mod runs before launch to verify mounted weights and build the offline HF hub-cache consumed by the serving process.",
      "from_domain": "Deployment Recipes & Cluster Orchestration Domain",
      "relation_type": "Data Dependency",
      "strength": 7.0,
      "to_domain": "Model Weights & Offline Serving Domain"
    },
    {
      "description": "docker/ patch scripts are executed during image build to mutate installed vLLM/FlashInfer/B12X sources; runtime mods patch the same installed tree at container start.",
      "from_domain": "Engine Patching & Model Compatibility Domain",
      "relation_type": "Build-Time Composition",
      "strength": 8.0,
      "to_domain": "Container Build & Image Composition Domain"
    },
    {
      "description": "The inkling-sm12-paged-kv mod patches vLLM's FA4 dispatch to route SM12.x devices to the vendored kernel package, composing the custom kernels into the engine.",
      "from_domain": "Attention Kernel Domain",
      "relation_type": "Module Composition",
      "strength": 8.5,
      "to_domain": "Engine Patching & Model Compatibility Domain"
    },
    {
      "description": "The vendored kernel sources are copied into the image; related B12X cache-integrity and CUTLASS DSL pinning patches are applied at build time.",
      "from_domain": "Attention Kernel Domain",
      "relation_type": "Build-Time Composition",
      "strength": 7.0,
      "to_domain": "Container Build & Image Composition Domain"
    },
    {
      "description": "memory-profile probe.py installs itself into vLLM as vllm._spark_memory_profile via its patch mechanism, observing worker startup inside the engine.",
      "from_domain": "Memory Profiling & Capacity Domain",
      "relation_type": "Instrumentation Dependency",
      "strength": 7.5,
      "to_domain": "Engine Patching & Model Compatibility Domain"
    },
    {
      "description": "Probe output is collected per run directory, merged into a profile card keyed by model/recipe, and used for topology-aware host selection for cluster deployments.",
      "from_domain": "Memory Profiling & Capacity Domain",
      "relation_type": "Data Dependency",
      "strength": 7.0,
      "to_domain": "Deployment Recipes & Cluster Orchestration Domain"
    },
    {
      "description": "The docker/ patch library is shared build infrastructure used to apply both engine-level and kernel-level fixes into the installed package.",
      "from_domain": "Container Build & Image Composition Domain",
      "relation_type": "Tool Support",
      "strength": 8.0,
      "to_domain": "Engine Patching & Model Compatibility Domain"
    },
    {
      "description": "Recipes for SM12-class hardware (e.g., inkling-small-nvfp4) implicitly enable the vendored paged-KV attention kernel mod.",
      "from_domain": "Deployment Recipes & Cluster Orchestration Domain",
      "relation_type": "Configuration Dependency",
      "strength": 6.0,
      "to_domain": "Attention Kernel Domain"
    },
    {
      "description": "scripts/fes-eval.sh invokes the fes-weights evaluation entrypoint to score staged weight directories.",
      "from_domain": "Developer Tooling Domain",
      "relation_type": "Function Call",
      "strength": 4.0,
      "to_domain": "Model Weights & Offline Serving Domain"
    },
    {
      "description": "DeepWiki documentation generation walks the repository source tree, including patch and mod sources, to produce reference documentation.",
      "from_domain": "Developer Tooling Domain",
      "relation_type": "Documentation Dependency",
      "strength": 3.0,
      "to_domain": "Engine Patching & Model Compatibility Domain"
    },
    {
      "description": "Verified weights and hub-cache layout are the runtime model input resolved by the recipe-launched vLLM server.",
      "from_domain": "Model Weights & Offline Serving Domain",
      "relation_type": "Runtime Input",
      "strength": 6.0,
      "to_domain": "Deployment Recipes & Cluster Orchestration Domain"
    }
  ]
}
```

### Workflow Research Report
Contains static analysis results of the codebase and business process analysis.

```json
"# System Workflow Analysis\n\n**Project**: `spark-vllm-docker`\n**Scope**: Patch-and-orchestration layer that optimizes the vLLM serving engine for Spark clusters with NVIDIA Blackwell GPUs (SM90/SM100/SM120).\n\n---\n\n## 1. Main Workflow\n\n- **Workflow Name**: Recipe-Driven Cluster Model Deployment Flow\n- **Description**: This is the system's primary end-to-end workflow. It begins when an operator selects a declarative YAML recipe (defining model, quantization, mods, and cluster size) and ends with a running vLLM service on a Spark cluster. The flow validates staged model weights for offline serving, applies the required model-specific fix/feature mods to the installed vLLM package via idempotent AST/text patches, launches head and worker nodes, and finally routes attention compute on SM12.x devices to the vendored paged-KV FlashAttention kernel.\n\n- **Flow Diagram**:\n```mermaid\ngraph TD\n    Start[Select model recipe YAML] --> Parse[Parse recipe: mods, parallelism, quantization]\n    Parse --> WeightSetup[Setup offline HF hub-cache layout]\n    WeightSetup --> Verify{Weight verification}\n    Verify -->|Pass| ApplyMods[Apply model-specific fix mods]\n    Verify -->|Fail| Abort[Fail fast with reason]\n    ApplyMods --> FeatureMods[Apply feature enablement mods]\n    FeatureMods --> Launch[Launch head and worker nodes]\n    Launch --> DispatchPatch[Patch FA4 dispatch for SM12.x devices]\n    DispatchPatch --> Kernels[Route to vendored paged-KV kernels]\n    Kernels --> Serve[Serve inference requests]\n```\n\n- **Key Steps**:\n  1. **Recipe selection and parsing** (`run-recipe.sh` / `run-recipe.py`) — Reads a recipe YAML from `recipes/3x|4x|8x-spark-cluster/` to determine which mods to apply, parallelism layout, and quantization scheme.\n  2. **Offline hub-cache setup** (`mods/fes-weights/run.sh`) — Verifies mounted weight directories and constructs the Hugging Face hub-cache layout so vLLM resolves models locally in air-gapped clusters.\n  3. **Weight verification** (`mods/fes-weights/validate`) — Compares shard files against `model.safetensors.index.json`, checks size parity (1.02 tolerance), and enforces `hf_quant_config.json` for NVFP4 checkpoints; fails fast on mismatch.\n  4. **Model-specific fix mods** (e.g., `mods/fix-qwen3-next-autoround/`) — Applies AST-based rewrites and patches that restore correct quantization, chat templates, and architecture support.\n  5. **Feature enablement mods** (e.g., `mods/diffusiongemma/`) — Applies feature patches with legacy/main fallbacks, then optionally launches the server with a custom Jinja chat template.\n  6. **Cluster launch** (`launch-cluster.sh`) — Starts head and worker nodes across the Spark cluster using the patched container image.\n  7. **Attention kernel dispatch** (`mods/inkling-sm12-paged-kv/patch_inkling.py`) — Injects a cached `_use_sm12_paged_kv` capability check into vLLM's NVIDIA FA4 dispatch so compute-capability 12.x devices route to the vendored kernel package.\n\n---\n\n## 2. Other Important Workflows\n\n### 2.1 Custom Container Image Build Flow\n\n- **Description**: Builds the custom vLLM Docker image (base or MXFP4 variant) by installing vLLM and FlashInfer, applying the shared build-time patch library, building FlashInfer JIT-cache provider wheels, vendoring the SM120 FlashAttention kernel package, and validating wheels for reproducibility.\n\n- **Flow Diagram**:\n```mermaid\ngraph TD\n    BuildStart[Invoke Dockerfile build] --> JIT[Build FlashInfer JIT provider wheels]\n    JIT --> CorePatches[Apply vLLM core patch set]\n    CorePatches --> Vendor[Vendor SM120 FlashAttention package]\n    Vendor --> Validate[Validate wheels and pin CUTLASS DSL]\n    Validate --> Image[Produce base or MXFP4 image]\n```\n\n- **Key Steps**:\n  1. `build-and-copy.sh` triggers the Dockerfile build (base or MXFP4 variant).\n  2. `docker/build_flashinfer_jit_providers.sh` builds FlashInfer JIT-cache provider wheels for target CUDA architectures.\n  3. `docker/patch_vllm_flashinfer_b12x_swigluoai.py` applies idempotent patches (SwiGLU-OAI plumbing, cache integrity, memory trims, top-k controls) — fails on unknown source shapes to avoid corrupting builds.\n  4. The vendored `inkling_sm120_fa4` kernel package is copied into the image.\n  5. `docker/validate_flashinfer_wheels.py` validates built wheels and pins CUTLASS DSL versions.\n\n---\n\n### 2.2 Memory Profiling & Capacity Validation Flow\n\n- **Description**: Profiles a model's startup memory on the target cluster, consolidates per-rank events into a profile card, and validates that the deployment fits available host memory — enabling topology-aware host selection before cluster launch.\n\n- **Flow Diagram**:\n```mermaid\ngraph TD\n    Probe[Install startup probe in vLLM] --> Record[Record per-phase memory events]\n    Record --> Collect[Collect per-rank JSONL]\n    Collect --> Card[Generate YAML profile card]\n    Card --> Capacity[Estimate RAM and KV-cache budget]\n    Capacity --> Report[Generate capacity report]\n    Report --> HostSelect[Select topology-appropriate hosts]\n    HostSelect --> Deploy[Proceed with cluster deployment]\n```\n\n- **Key Steps**:\n  1. **Startup probe** (`mods/memory-profile/probe.py`) — Installed as `vllm._spark_memory_profile`; records phase-by-phase native heap, CUDA memory, and model/KV inventory during worker startup.\n  2. **Profile collection** (`profile_card.py`) — Merges per-rank event JSONL files into a consolidated model/recipe YAML profile card.\n  3. **Capacity analysis** (`capacity.py`) — Estimates startup RAM, budgets KV cache, and validates deployment against available host memory per topology.\n  4. **Reporting** (`report.py`) — Generates the human-readable capacity report.\n  5. **Host selection** — Capacity findings inform topology-appropriate host choices when launching cluster deployments via `run-recipe.sh`.\n\n---\n\n### 2.3 Attention Kernel Dispatch on Blackwell Flow\n\n- **Description**: At inference startup, vLLM's FlashAttention 4 dispatch is patched so compute-capability 12.x devices route attention work to the vendored paged-KV kernel. The kernel interface validates inputs, computes tile heuristics, schedules persistent work tiles, and executes architecture-specific forward/MLA/backward kernels.\n\n- **Flow Diagram**:\n```mermaid\ngraph TD\n    Patch[Patch FA4 dispatch source] --> Check[Inject _use_sm12_paged_kv check]\n    Check --> Interface[Validate inputs and compute tile heuristics]\n    Interface --> Schedule[Schedule persistent work tiles]\n    Schedule --> Select[Select arch-specific kernel]\n    Select --> Fwd[SM90/SM100/SM120 forward kernel]\n    Select --> MLA[MLA forward kernel]\n    Select --> Bwd[Backward gradient kernel]\n    Fwd --> Split{Split-KV needed?}\n    Split -->|Yes| Combine[Split-K combine kernel]\n    Split -->|No| Output[Produce attention output]\n    Combine --> Output\n```\n\n- **Key Steps**:\n  1. **Dispatch patching** (`patch_inkling.py`) — AST-based validation and idempotent single-occurrence text replacement injects the capability check, guarded by a mod marker.\n  2. **Kernel interface** (`interface.py`) — Parses device architecture, validates head dims, computes tile sizes and split-KV heuristics, resolves causal/local window parameters, and wraps kernels in `FlashAttnFunc` / `FlashAttnVarlenFunc` autograd Functions.\n  3. **Tile scheduling** (`tile_scheduler.py`) — Distributes work across persistent kernel tiles using CLC-based scheduling modes.\n  4. **Kernel execution** — Architecture-selected forward (SM90/SM100/SM120/MLA), backward, or split-K combine kernels execute the attention computation.\n\n---\n\n### 2.4 Model Support Enablement Flow\n\n- **Description**: The workflow for onboarding a new or emerging model architecture. A developer authors a fix or feature mod (patch, diff, or chat template), registers it in a recipe, and the recipe runner applies it before the vLLM server starts.\n\n- **Flow Diagram**:\n```mermaid\ngraph TD\n    Author[Author patch, diff, or chat template] --> Orchestrator[Write run.sh orchestrator with fallbacks]\n    Orchestrator --> Register[Register mod in recipe YAML]\n    Register --> Run[Run recipe]\n    Run --> Apply[Apply mods in order]\n    Apply --> Server[Start vLLM server with corrected config]\n```\n\n- **Key Steps**:\n  1. Author model-support patches (e.g., `diffusiongemma-support.patch`) and a `run.sh` orchestrator with legacy/main fallbacks.\n  2. Provide corrected chat templates (e.g., `chat_template.jinja` for Qwen3.5 with multimodal validation) or AST-based rewrites.\n  3. Reference the mod from a recipe YAML describing model, quantization, and cluster size.\n  4. Run the recipe so mods apply in order and the server starts with the custom template.\n\n---\n\n## 3. Workflow Insights\n\n### Operational Patterns\n- **Non-invasive source patching as the dominant interaction style**: Nearly all customization is achieved through idempotent AST/text rewrites of the installed vLLM package, each guarded by a mod marker for traceability and double-patch prevention. This keeps the upstream dependency unmodified in version control while allowing deep behavioral changes.\n- **Recipes as the composition root**: YAML recipes are the single user-facing artifact that binds mods, weights, kernels, and cluster launch together. All supporting workflows (weight verification, memory profiling, kernel dispatch) are orchestrated through recipe execution.\n- **Two-phase patching**: Engine fixes are split by execution time — build-time patches under `docker/` are baked into the image, while runtime mods under `mods/fix-*` are applied at container start. This is a conceptual \"Engine Compatibility\" concern physically separated by when they execute.\n- **Fail-fast validation gates**: Both weight verification and build-time patch scripts fail fast on unexpected input (unknown source shapes, shard mismatches) to prevent corrupted builds and unsafe serving states.\n\n### Workflow Dependencies\n- **Deployment depends on Image Build**: The container image must be built with all build-time patches and vendored kernels before any recipe can run.\n- **Deployment depends on Weight Verification**: Recipe execution halts if staged weights fail integrity checks — a hard data dependency.\n- **Deployment informs and is informed by Capacity Profiling**: Capacity findings guide host topology selection; profiling itself requires running a recipe-modified vLLM instance.\n- **Kernel Dispatch is a runtime sub-flow of Deployment**: The FA4 dispatch patch is applied during engine startup as part of the deployment sequence.\n\n### Potential Optimization Opportunities\n- **Unify the split Engine Compatibility concern**: Build-time (`docker/`) and runtime (`mods/fix-*`) patches target the same installed vLLM tree. Consolidating them into a single patch manifest with execution-phase annotations would reduce drift and simplify auditing.\n- **Profile-card reuse across recipes**: Capacity profile cards are keyed by model/recipe; caching and auto-retrieving existing cards before re-profiling could save significant cluster time for repeated deployments.\n- **Patch validation standardization**: Patch scripts use varying validation strategies (AST vs. text replacement vs. git apply). Standardizing on AST-based validation with mod markers across all patchers would improve robustness against upstream vLLM changes.\n\n### Gaps and Observations\n- The system context documentation lists \"Spark cluster orchestration and management\" as out of scope, yet `launch-cluster.sh` and `autodiscover.sh` handle node bootstrap — this is a thin orchestration layer consistent with the boundary but worth noting for clarity.\n- The vendored kernel package spans SM80/SM90/SM100/SM120 architectures, while the project scope emphasizes Blackwell (SM100/SM120) — the SM80/SM90 kernels serve as fallbacks and training-support paths, extending the documented hardware target.\n- No explicit rollback workflow is documented for failed mod application at runtime; idempotent patching mitigates this, but a formal revert path would strengthen operational safety."
```

### Code Insights Data
Code analysis results from preprocessing phase, including definitions of functions, classes, and modules.

```json
{
  "directory_insights": [
    {
      "file_count": 23,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [],
          "detailed_description": "This is a patch file (not directly executable) that modifies B12X source code to add cache integrity checks and durable publication of CuTe cache artifacts. It introduces a new cache_integrity.py module and reuses existing locked compilation paths for repair, and adds fsync for temporary files.",
          "file_path": ".litho/tree/repo/docker/b12x-cache-integrity.patch",
          "importance_score": 0.7,
          "interfaces": [],
          "name": "b12x-cache-integrity.patch",
          "responsibilities": [
            "Validate cache artifacts integrity",
            "Prevent incomplete cache artifacts from being used",
            "Ensure durable publication via fsync",
            "Integrate repair through locked compilation"
          ],
          "source_summary": "The patch adds a new file b12x/_lib/cache_integrity.py (content shown truncated) that implements validation and repair logic for CuTe cache artifacts, and modifies existing compilation paths to use locked mode for repair and add durability guarantees via fsync.",
          "summary": "A patch file that validates and durably publishes CuTe cache artifacts in B12X, ensuring incomplete artifacts are repaired and properly synced."
        },
        {
          "code_purpose": "config",
          "dependencies": [],
          "detailed_description": "This shell script builds the FlashInfer JIT-cache provider wheels for the specified CUDA architectures. It is designed to run from a FlashInfer checkout, checks for the legacy monolithic wheel path, and uses the provided Python interpreter and output directory to build the providers.",
          "file_path": ".litho/tree/repo/docker/build_flashinfer_jit_providers.sh",
          "importance_score": 0.9,
          "interfaces": [],
          "name": "build_flashinfer_jit_providers.sh",
          "responsibilities": [
            "Build FlashInfer JIT cache provider wheels",
            "Handle legacy monolithic wheel case",
            "Support architecture-specific building"
          ],
          "source_summary": "The script checks for the presence of flashinfer-jit-cache-provider directory, requires FLASHINFER_JIT_CACHE_PROVIDER_ARCHS environment variable and takes prepared python and wheel dir as arguments, then builds in that directory.",
          "summary": "Builds FlashInfer JIT-cache provider wheels for specified CUDA architectures."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "importlib.util",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "subprocess",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This Python script applies the temporary upstream CuTe cache-integrity fix to installed B12X. It uses git apply to apply the patch file, checks for existing application to allow idempotent reapply, and raises errors if the patch cannot be applied safely.",
          "file_path": ".litho/tree/repo/docker/patch_b12x_cache_integrity.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "apply_patch",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "root",
                  "param_type": "Path"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "patch_b12x_cache_integrity.py",
          "responsibilities": [
            "Apply cache-integrity patch",
            "Check patch applicability",
            "Support idempotent reapplication"
          ],
          "source_summary": "The script defines apply_patch(root) that checks for the existence of 'b12x/_lib/compile_plan.py' to detect PyPI version, then executes git apply with the patch file. It handles idempotency and returns boolean success, and main() parses arguments and calls apply_patch.",
          "summary": "Applies the b12x-cache-integrity.patch to an installed B12X source tree, with idempotent apply and reapply support."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "importlib.util",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script modifies InstantTensor source code to replace direct torch.cuda.mem_get_info() calls with vLLM's MemorySnapshot utility, which accounts for UMA memory and CUDA-on-WSL policies. It performs an AST-based exact match replacement on the target source.",
          "file_path": ".litho/tree/repo/docker/patch_instanttensor_vllm_memory.py",
          "importance_score": 0.75,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "patch_memory_query",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "patch_instanttensor_vllm_memory.py",
          "responsibilities": [
            "Replace memory query logic",
            "Integrate vLLM memory utilities",
            "Apply patch to source file"
          ],
          "source_summary": "The script defines patch_memory_query(source) that replaces the original memory query block with a new block using MemorySnapshot from vllm.utils.mem_utils, and main() applies this to a target file.",
          "summary": "Patches InstantTensor to use vLLM's platform-aware memory accounting for memory budget calculation."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "importlib.util",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This patch script optimizes a PyTorch source code loop in fill_defaults to avoid repeated lookups of schema.arguments, improving performance. It uses AST matching to find the original pattern and replace it with an optimized version using enumerate.",
          "file_path": ".litho/tree/repo/docker/patch_torch_schema_enumeration.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "patch_fill_defaults",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "patch_torch_schema_enumeration.py",
          "responsibilities": [
            "Optimize PyTorch source",
            "Replace loop construct",
            "Apply patch"
          ],
          "source_summary": "The script defines patch_fill_defaults that replaces a for loop over schema.arguments with a for loop over enumerate(schema.arguments), and main() applies it to the target file.",
          "summary": "Optimizes PyTorch's fill_defaults by caching schema.arguments enumeration."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script fixes a bug in vLLM where AutoGPTQ MoE layers incorrectly pass zero-points even for symmetric quantization. It replaces the bad code block with a conditional that only passes zero-points when not in symmetric mode.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_autogptq_symmetric_moe_qzeros.py",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "patch_vllm_autogptq_symmetric_moe_qzeros.py",
          "responsibilities": [
            "Fix AutoGPTQ quantization",
            "Conditional zero-point passing",
            "Apply patch to vLLM source"
          ],
          "source_summary": "The script reads the target file vllm/model_executor/layers/quantization/auto_gptq.py, replaces the hardcoded w1_zp/w2_zp assignments with a conditional based on self.quant_config.",
          "summary": "Patches vLLM's AutoGPTQ quantization to omit zero-points for symmetric MoE quantization."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "os",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script patches vLLM to import the _C128A_TOPK_ALIGNMENT constant from compressor_utils in the B12X-specific path, ensuring correct top-k width for DeepSeek V4. It is opt-in via environment variable and strictly validates the source to avoid touching regular builds.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_b12x_c128a_topk_alignment.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "module_binds_name",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "tree",
                  "param_type": "ast.Module"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "module_constant_is_128",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "tree",
                  "param_type": "ast.Module"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            }
          ],
          "name": "patch_vllm_b12x_c128a_topk_alignment.py",
          "responsibilities": [
            "Add missing constant import",
            "Support C128A alignment",
            "Opt-in via environment flag",
            "Idempotent and strict source validation"
          ],
          "source_summary": "Uses AST to find module-level binds and checks for constant value 128, then inserts an import statement into the module.__init__ or similar. It has logic to detect if the import already exists.",
          "summary": "Adds missing import for C128A top-k alignment constant in vLLM B12X path."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script modifies vLLM's fused_moe/b12x.py to ensure that MoE trial tensors are released after being used, preventing them from being retained in serving plans. It uses AST to find the _PreparedMoECall class and adds a marker to keep owners out of plans.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_b12x_moe_tuning_memory.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "patch_source",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "patch_vllm_b12x_moe_tuning_memory.py",
          "responsibilities": [
            "Reduce MoE tuning memory",
            "Release trial tensors",
            "Apply AST-based patch"
          ],
          "source_summary": "The patch_source function parses the target file, finds _PreparedMoECall class, and likely adds cleanup code or modifies ownership to release tensors after call.",
          "summary": "Patches B12X MoE tuning to keep trial tensors alive only for the call lifetime, reducing memory usage."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script modifies vLLM's attention utilities to accept torch.Tensor as a valid causal mask type, and adjusts the group_causal determination accordingly. It fixes a bug introduced by a vLLM PR.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_diffusion_tensor_causal.py",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "patch_vllm_diffusion_tensor_causal.py",
          "responsibilities": [
            "Allow Tensor causal masks",
            "Update type signatures",
            "Apply patch"
          ],
          "source_summary": "Replaces function signature and group_causal condition to include torch.Tensor, allowing Tensor causal masks.",
          "summary": "Patches vLLM to allow Tensor-type causal masks for DiffusionGemma."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script sets the _MINIMAX_FUSED_AR_RMS_QK to None to disable the fused operation, likely due to incompatibility with multi-node setups.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_disable_minimax_qk_rmsnorm_ipc.py",
          "importance_score": 0.5,
          "interfaces": [],
          "name": "patch_vllm_disable_minimax_qk_rmsnorm_ipc.py",
          "responsibilities": [
            "Disable CUDA IPC fusion",
            "Modify runtime setting"
          ],
          "source_summary": "Replaces the assignment of _MINIMAX_FUSED_AR_RMS_QK with None, adding a comment about disabling for DGX Spark.",
          "summary": "Disables MiniMax QK RMSNorm CUDA IPC fusion for multi-node DGX Spark."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This is the most complex and critical patch script in the directory. It modifies multiple vLLM files to pass SwiGLU-OAI parameters to FlashInfer and allows the NVFP4 oracle to select it. It is idempotent and fails on unknown source shapes to avoid corrupting regular builds.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_flashinfer_b12x_swigluoai.py",
          "importance_score": 1.0,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "PatchError",
              "parameters": [],
              "return_type": "class",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "replace_once",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "old",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "new",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "description",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "patch_flashinfer_util",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "patch_expert",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "patch_oracle",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "int",
              "visibility": ""
            }
          ],
          "name": "patch_vllm_flashinfer_b12x_swigluoai.py",
          "responsibilities": [
            "Apply FlashInfer compatibility patch",
            "Pass SwiGLU-OAI parameters",
            "Enable NVFP4 oracle selection",
            "Maintain idempotency and strictness"
          ],
          "source_summary": "Defines PatchError class, replace_once helper, patch_flashinfer_util, patch_expert, patch_oracle, and main. Uses exact string replacement with validation to apply the patch.",
          "summary": "Applies the runtime subset of vLLM PR #47392 to teach FlashInfer B12x to pass SwiGLU-OAI parameters."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script modifies vLLM's llm_base_proposer.py to make the embedding-sharing guard more flexible, preventing type-check failures when the embedding tensors are not standard tensors (e.g., in Gemma4 MTP).",
          "file_path": ".litho/tree/repo/docker/patch_vllm_gemma4_mtp_embedding_share.py",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "patch_vllm_gemma4_mtp_embedding_share.py",
          "responsibilities": [
            "Fix embedding sharing guard",
            "Support non-Tensor embeddings",
            "Apply patch to spec decode"
          ],
          "source_summary": "Replaces the code that assumes embed_tokens is a torch.Tensor with a more permissive guard.",
          "summary": "Fixes EAGLE embedding-width guard that breaks Gemma4 MTP."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script patches vLLM's cudagraph_utils.py to isolate speculator graphs by capturing and restoring the original memory pools during profiling. It adds a marker comment and modifies the profiling context.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_mrv2_speculator_cudagraph_pool.py",
          "importance_score": 0.75,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "PatchError",
              "parameters": [],
              "return_type": "class",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "is_fixed",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "is_affected_profiler",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "replace_once",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "anchor",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "replacement",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "label",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "patch_source",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                }
              ],
              "return_type": "tuple[str, str]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "patch_vllm_mrv2_speculator_cudagraph_pool.py",
          "responsibilities": [
            "Isolate speculator CUDA graphs",
            "Preserve memory pools",
            "Apply patch to cudagraph utils"
          ],
          "source_summary": "Defines is_fixed, is_affected_profiler, replace_once, patch_source, and main. The patch inserts code to preserve original pools and restore them after profiling.",
          "summary": "Isolates MRV2 speculator graphs during CUDA-graph memory profiling to prevent interference."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "os",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script conditionally patches CMakeLists.txt to preserve user-selected SM12x targets (like sm_121) when building with CUDA 13, as default CMake logic may strip them. It is activated by an environment variable.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_preserve_sm12x_target.py",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "patch_vllm_preserve_sm12x_target.py",
          "responsibilities": [
            "Preserve SM12x architecture targets",
            "Conditional patch based on env var",
            "Modify CMake configuration"
          ],
          "source_summary": "If VLLM_PRESERVE_SM12X_TARGET is set, it modifies CMakeLists to adjust the default architecture list. Reads and replaces a specific CUDA 13 default set.",
          "summary": "Prevents vLLM's CMake from overriding explicit CUDA 13 Blackwell subarchitecture targets."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This patch fixes a regression in RoutedExperts parameter loading that dropped weight_shape metadata when loading per-expert parameters, important for certain models.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_routed_experts_weight_shape.py",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "patch_vllm_routed_experts_weight_shape.py",
          "responsibilities": [
            "Preserve weight shape metadata",
            "Fix expert loading",
            "Apply patch"
          ],
          "source_summary": "Replaces the _load_single_value method in routed_experts.py to preserve weight_shape when loading weights.",
          "summary": "Preserves vector weight_shape metadata in vLLM's RoutedExperts loading."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script adds a check to the top-k selection logic to only use cooperative top-k on SM90 GPUs, preventing potential errors on newer Blackwell architectures like SM120.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_sm120_cooperative_topk.py",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "patch_vllm_sm120_cooperative_topk.py",
          "responsibilities": [
            "Restrict cooperative top-k to SM90",
            "Improve compatibility",
            "Apply patch to sparse attention"
          ],
          "source_summary": "Modifies sparse_attn_indexer.py to add a condition that only enables cooperative top-k when the platform is not SM120.",
          "summary": "Restricts vLLM's cooperative sparse-attention top-k path to SM90 only, avoiding SM120 issues."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "re",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This patch adds cleanup of profiling allocations in vLLM's GPU worker before cache creation, improving memory efficiency on DGX Spark. It inserts code to free profiling memory after warmup.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_spark_kv_cache_cleanup.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "find_line",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "pattern",
                  "param_type": "str"
                }
              ],
              "return_type": "tuple[int, re.Match[str]]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "find_first_line",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "patterns",
                  "param_type": "tuple[str, ...]: Any"
                }
              ],
              "return_type": "tuple[int, re.Match[str]]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "insert_after_docstring",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "func_index",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "func_indent",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "block",
                  "param_type": "list[str]"
                }
              ],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "patch_vllm_spark_kv_cache_cleanup.py",
          "responsibilities": [
            "Clean profiling allocations",
            "Insert cleanup before KV cache creation",
            "Patch GPU worker"
          ],
          "source_summary": "Uses regex and line manipulation to find the appropriate location and insert a cleanup block. Defines find_line, find_first_line, insert_after_docstring, and main.",
          "summary": "Cleans profiling allocations before vLLM sizes and creates the KV cache to reduce memory pressure."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "textwrap",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script adds a call to malloc_trim after vLLM's startup GC to return unused heap pages to the OS, reducing memory footprint. It uses AST to insert the trim block into gc_utils.py.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_startup_heap_trim.py",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "is_gc_call",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "node",
                  "param_type": "ast.stmt"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "method",
                  "param_type": "str"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "patch_source",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "patch_vllm_startup_heap_trim.py",
          "responsibilities": [
            "Trim CPU heap after GC",
            "Reduce memory footprint",
            "Apply AST-based patch"
          ],
          "source_summary": "Defines is_gc_call, patch_source, and main. The patch inserts a try block with malloc_trim after GC calls in the startup sequence.",
          "summary": "Trims unused CPU heap pages after vLLM startup garbage collection."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script patches vLLM's attention layer to preserve the fallback for unsupported primary SWA block sizes, ensuring compatibility with certain models. It uses AST to modify the attention backend selection.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_swa_block_size.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "PatchError",
              "parameters": [],
              "return_type": "class",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "patch_source",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                }
              ],
              "return_type": "tuple[str, str]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "patch_vllm_swa_block_size.py",
          "responsibilities": [
            "Preserve SWA fallback",
            "Maintain attention compatibility",
            "Apply AST-based patch"
          ],
          "source_summary": "Defines PatchError class, patch_source, and main. The patch adds a fallback block size calculation when the primary block size is unsupported.",
          "summary": "Restores the unsupported-primary SWA block size fallback removed by vLLM PR #53007."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script corrects control flow in vLLM's _custom_ops.py to ensure that the XPU-specific branch returns correctly without bypassing fallback logic. It moves a return statement.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_topk_softplus_sqrt_control_flow.py",
          "importance_score": 0.5,
          "interfaces": [],
          "name": "patch_vllm_topk_softplus_sqrt_control_flow.py",
          "responsibilities": [
            "Fix control flow",
            "Apply patch to custom ops"
          ],
          "source_summary": "Finds the function marker and replaces a misplaced return with a correct one that returns from the XPU block.",
          "summary": "Fixes a misplaced XPU-only return in vLLM's topk_softplus_sqrt function."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "importlib.util",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script modifies vLLM's memory utilities to exclude WSL environments from UMA memory detection, because CUDA on WSL should use its own memory budget rather than the guest OS's RAM. It patches the condition that enables integrated GPU memory sharing.",
          "file_path": ".litho/tree/repo/docker/patch_vllm_wsl_cuda_uma.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "patch_mem_utils",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "patch_vllm_wsl_cuda_uma.py",
          "responsibilities": [
            "Adjust memory detection for WSL",
            "Patch UMA conditions",
            "Apply to vLLM memory utils"
          ],
          "source_summary": "Defines patch_mem_utils that adds a WSL check to the UMA_CONDITION, and main() applies it to the target file.",
          "summary": "Keeps CUDA's memory budget on WSL instead of using guest RAM availability."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "re",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script updates pinned versions of nvidia-cutlass-dsl packages in vLLM's metadata files (setup.py, pyproject.toml, etc.) to match the version installed in the Docker image, ensuring consistency.",
          "file_path": ".litho/tree/repo/docker/pin_cutlass_dsl.py",
          "importance_score": 0.65,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "pin_text",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "version",
                  "param_type": "str"
                }
              ],
              "return_type": "tuple[str, int]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "replace",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "match",
                  "param_type": "re.Match[str]"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "replace",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "match",
                  "param_type": "re.Match[str]"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "int",
              "visibility": ""
            }
          ],
          "name": "pin_cutlass_dsl.py",
          "responsibilities": [
            "Align CUTLASS DSL versions",
            "Regex-based replacement",
            "Pin wheel metadata"
          ],
          "source_summary": "Uses regex to find and replace version pins for cutlass DSL packages in requirement strings, with support for optional extras.",
          "summary": "Pins CUTLASS DSL requirements in vLLM source metadata to match the installed image version."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "email.parser.BytesParser",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "re",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "zipfile",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script validates that built FlashInfer JIT-cache provider wheels have correct metadata and architecture tags. It uses only the standard library to avoid heavy dependencies during Docker build.",
          "file_path": ".litho/tree/repo/docker/validate_flashinfer_wheels.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "normalize_name",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "read_metadata",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "wheel",
                  "param_type": "Path"
                }
              ],
              "return_type": "object",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "provider_tag",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "architecture",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "validate_providers",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "jit_wheel",
                  "param_type": "Path"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "architectures",
                  "param_type": "str"
                }
              ],
              "return_type": "list[Path]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "int",
              "visibility": ""
            }
          ],
          "name": "validate_flashinfer_wheels.py",
          "responsibilities": [
            "Validate wheel metadata",
            "Check provider tags",
            "Support build verification"
          ],
          "source_summary": "Defines normalize_name, read_metadata, provider_tag, validate_providers, and main. It reads wheel zip metadata and checks provider requirements.",
          "summary": "Validates FlashInfer JIT-cache provider wheels against expected metadata."
        }
      ],
      "importance_score": 0.9,
      "key_files": [
        "patch_vllm_flashinfer_b12x_swigluoai.py",
        "build_flashinfer_jit_providers.sh",
        "patch_vllm_b12x_c128a_topk_alignment.py",
        "patch_vllm_spark_kv_cache_cleanup.py",
        "patch_b12x_cache_integrity.py"
      ],
      "name": "docker",
      "path": ".litho/tree/repo/docker",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a comprehensive collection of Python patch scripts for modifying vLLM, B12X, FlashInfer, and PyTorch source code, primarily for Docker container builds on Blackwell GPU (SM12x) and DGX Spark systems. This patches address cache integrity, MoE tuning, CUDA memory management, FlashInfer JIT cache providers, and various vLLM compatibility fixes required for specific model architectures and NVIDIA hardware. The patch scripts work together with the included .patch file and utility scripts to automate source-tree modifications during the Docker build process."
    },
    {
      "file_count": 3,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "launch-cluster.sh",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file is an example deployment profile that launches a vLLM server for the QuantTrio/MiniMax-M2-AWQ model. It configures tensor parallelism across 2 GPUs, 128K context length, fastsafetensors loading, and MiniMax-specific tool-call and reasoning parsers with auto tool choice enabled.",
          "file_path": ".litho/tree/repo/examples/example-vllm-minimax.sh",
          "importance_score": 0.45,
          "interfaces": [
            {
              "description": null,
              "interface_type": "shell_command",
              "name": "vllm serve (script invocation)",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "model",
                  "param_type": "string (QuantTrio/MiniMax-M2-AWQ)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--port",
                  "param_type": "int (8000)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--host",
                  "param_type": "string (0.0.0.0)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--gpu-memory-utilization",
                  "param_type": "float (0.8)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "-tp",
                  "param_type": "int (2)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--max-model-len",
                  "param_type": "int (128000)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--load-format",
                  "param_type": "string (fastsafetensors)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--tool-call-parser",
                  "param_type": "string (minimax_m2)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--reasoning-parser",
                  "param_type": "string (minimax_m2)"
                }
              ],
              "return_type": "running vLLM server process",
              "visibility": ""
            }
          ],
          "name": "example-vllm-minimax.sh",
          "responsibilities": [
            "Define vLLM serving parameters for MiniMax-M2-AWQ",
            "Configure tensor parallelism and GPU memory utilization",
            "Enable MiniMax-specific tool-call and reasoning parsing",
            "Serve as an example profile for launch-cluster.sh"
          ],
          "source_summary": "The script invokes 'vllm serve QuantTrio/MiniMax-M2-AWQ' with options: port 8000, host 0.0.0.0, gpu-memory-utilization 0.8, tensor parallel size 2 (-tp 2), max-model-len 128000, load-format fastsafetensors, enable-auto-tool-choice, and minimax_m2 tool-call and reasoning parsers. Header comments label it as a PROFILE with a DESCRIPTION for launch-cluster.sh integration.",
          "summary": "Shell script profile for serving the MiniMax-M2-AWQ quantized model via vLLM, intended for use with launch-cluster.sh."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fix-Salyut1-GLM-4.7-NVFP4",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "launch-cluster.sh",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file is an example deployment profile that launches a vLLM server for the GLM-4.7-NVFP4 quantized model using the FlashInfer attention backend. It documents a required patch (--apply-mod mods/fix-Salyut1-GLM-4.7-NVFP4) to fix k/v scales incompatibility, with a reference link to the Hugging Face discussion, and enables GLM-specific tool-call and reasoning parsers.",
          "file_path": ".litho/tree/repo/examples/vllm-glm-4.7-nvfp4.sh",
          "importance_score": 0.45,
          "interfaces": [
            {
              "description": null,
              "interface_type": "shell_command",
              "name": "vllm serve (script invocation)",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "model",
                  "param_type": "string (Salyut1/GLM-4.7-NVFP4)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--attention-config.backend",
                  "param_type": "string (flashinfer)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--tool-call-parser",
                  "param_type": "string (glm47)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--reasoning-parser",
                  "param_type": "string (glm45)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--enable-auto-tool-choice",
                  "param_type": "flag"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "-tp",
                  "param_type": "int (2)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--gpu-memory-utilization",
                  "param_type": "float"
                }
              ],
              "return_type": "running vLLM server process",
              "visibility": ""
            }
          ],
          "name": "vllm-glm-4.7-nvfp4.sh",
          "responsibilities": [
            "Define vLLM serving parameters for GLM-4.7-NVFP4",
            "Select the FlashInfer attention backend",
            "Document the required k/v scales fix mod and reference link",
            "Enable GLM-specific tool-call and reasoning parsing"
          ],
          "source_summary": "The script invokes 'vllm serve Salyut1/GLM-4.7-NVFP4' with options: --attention-config.backend flashinfer, --tool-call-parser glm47, --reasoning-parser glm45, --enable-auto-tool-choice, -tp 2, and --gpu-memory-utilization (truncated in source). Header comments label it as a PROFILE with a NOTE about the required mod and an external reference URL.",
          "summary": "Shell script profile for serving the Salyut1/GLM-4.7-NVFP4 model via vLLM, requiring a companion mod to fix k/v scale incompatibility."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "launch-cluster.sh",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file is an example deployment profile that launches a vLLM server for the openai/gpt-oss-120b model, enabling FlashInfer Mixture-of-Experts optimization via the VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8 environment variable. It configures tensor parallelism across 2 GPUs, FP8 KV cache dtype, 128K context length, and OpenAI-style tool-call parsing with auto tool choice.",
          "file_path": ".litho/tree/repo/examples/vllm-openai-gpt-oss-120b.sh",
          "importance_score": 0.5,
          "interfaces": [
            {
              "description": null,
              "interface_type": "shell_command",
              "name": "vllm serve (script invocation)",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "model",
                  "param_type": "string (openai/gpt-oss-120b)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--tool-call-parser",
                  "param_type": "string (openai)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--enable-auto-tool-choice",
                  "param_type": "flag"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--tensor-parallel-size",
                  "param_type": "int (2)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--kv-cache-dtype",
                  "param_type": "string (fp8)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--gpu-memory-utilization",
                  "param_type": "float (0.70)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--max-model-len",
                  "param_type": "int (128000)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--max-num-batched-tokens",
                  "param_type": "int (4096)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--max-num-seqs",
                  "param_type": "int"
                }
              ],
              "return_type": "running vLLM server process",
              "visibility": ""
            }
          ],
          "name": "vllm-openai-gpt-oss-120b.sh",
          "responsibilities": [
            "Define vLLM serving parameters for GPT-OSS 120B",
            "Enable FlashInfer MOE with MXFP4/MXFP8 quantization via environment variable",
            "Configure FP8 KV cache and batching limits",
            "Enable OpenAI-style tool-call parsing with auto tool choice"
          ],
          "source_summary": "The script exports VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 and then invokes 'vllm serve openai/gpt-oss-120b' with options: --tool-call-parser openai, --enable-auto-tool-choice, --tensor-parallel-size 2, --kv-cache-dtype fp8, --gpu-memory-utilization 0.70, --max-model-len 128000, --max-num-batched-tokens 4096, and --max-num-seqs (truncated in source). Header comments label it as a PROFILE with a DESCRIPTION mentioning FlashInfer MOE optimization.",
          "summary": "Shell script profile for serving OpenAI's GPT-OSS 120B model via vLLM with FlashInfer MOE MXFP4/MXFP8 optimization and FP8 KV cache."
        }
      ],
      "importance_score": 0.45,
      "key_files": [
        "example-vllm-minimax.sh",
        "vllm-glm-4.7-nvfp4.sh",
        "vllm-openai-gpt-oss-120b.sh"
      ],
      "name": "examples",
      "path": ".litho/tree/repo/examples",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The examples directory contains shell script profiles for serving large language models with vLLM, covering MiniMax-M2-AWQ, GLM-4.7-NVFP4, and OpenAI GPT-OSS-120B. Each script defines serving configurations including tensor parallelism, quantization formats, tool/reasoning parsers, and memory settings, and appears to be consumed by a launch-cluster.sh orchestration script. This is a configuration/example layer rather than core business logic, providing ready-to-use deployment profiles for different model checkpoints."
    },
    {
      "file_count": 9,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vLLM chat serving (chat_template.jinja consumer)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Copied from a training checkpoint's chat template with one intentional change: when add_generation_prompt is true and thinking is not enabled, it does not prefill an empty '<|channel>thought\\n<channel|>' block after the model turn marker. It defines Jinja macros for formatting tool parameters and required keys, enabling OpenAI-compatible chat serving of the model.",
          "file_path": ".litho/tree/repo/mods/diffusiongemma/chat_template_no_think.jinja",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "jinja_macro",
              "name": "format_parameters",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "properties",
                  "param_type": "dict"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "required",
                  "param_type": "list"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "filter_keys",
                  "param_type": "bool"
                }
              ],
              "return_type": "string",
              "visibility": ""
            }
          ],
          "name": "chat_template_no_think.jinja",
          "responsibilities": [
            "Render conversation history into DiffusionGemma's turn/channel markup",
            "Format tool/function parameter schemas for prompts",
            "Conditionally prefill or omit the reasoning thought channel",
            "Support both thinking and no-think generation modes"
          ],
          "source_summary": "A Jinja template defining macros such as format_parameters to render JSON-schema-like tool parameter definitions, handling of system/user/model turns via <|turn> markers, tool call serialization, and conditional injection of the reasoning ('thought') channel block depending on the add_generation_prompt and thinking flags.",
          "summary": "Custom vLLM chat template for DiffusionGemma that formats conversation turns, tool calls, and parameters, with a variant that omits prefilling an empty thought channel when thinking is disabled."
        },
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm/v1/attention/backends/flash_attn.py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Modifies vllm/v1/attention/backends/flash_attn.py, changing FlashAttentionMetadata.causal from a plain bool to 'bool | torch.Tensor' and threading the tensor through the attention kernel invocation. This is the legacy-branch variant of the attention change, kept alongside the main-branch version for compatibility with different vLLM checkouts.",
          "file_path": ".litho/tree/repo/mods/diffusiongemma/diffusiongemma-attention-legacy.patch",
          "importance_score": 0.65,
          "interfaces": [],
          "name": "diffusiongemma-attention-legacy.patch",
          "responsibilities": [
            "Enable tensor-valued causal masks in FlashAttention metadata",
            "Support block-diffusion attention patterns on the legacy vLLM branch",
            "Serve as an alternative to diffusiongemma-attention-main.patch"
          ],
          "source_summary": "A unified diff that relaxes the causal type annotation in FlashAttentionMetadata and updates downstream call sites so a per-sequence/per-token causal mask tensor can be passed to flash_attn, enabling mixed causal/bidirectional attention required by block diffusion decoding.",
          "summary": "Patch against a legacy vLLM branch making FlashAttention's causal flag accept a per-token tensor so DiffusionGemma's block-diffusion attention (bidirectional within blocks, causal across) can be expressed."
        },
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm/v1/attention/backends/flash_attn.py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Targets the current vLLM main branch's flash_attn backend, changing FlashAttentionMetadata.causal to 'bool | torch.Tensor' and adapting the kernel call path. It coexists with PrefixLM bidirectional range metadata used for multimodal tokens, so the patch must preserve those semantics while adding block-diffusion support.",
          "file_path": ".litho/tree/repo/mods/diffusiongemma/diffusiongemma-attention-main.patch",
          "importance_score": 0.65,
          "interfaces": [],
          "name": "diffusiongemma-attention-main.patch",
          "responsibilities": [
            "Enable per-token causal/bidirectional attention on vLLM main",
            "Integrate with existing PrefixLM bidirectional range logic",
            "Provide the up-to-date counterpart of the legacy attention patch"
          ],
          "source_summary": "A unified diff against vllm/v1/attention/backends/flash_attn.py that widens the causal field type and propagates tensor causal masks through metadata construction and the flash attention invocation, allowing sequences where some token blocks attend bidirectionally.",
          "summary": "Main-branch variant of the attention patch that lets FlashAttention accept a per-token causal tensor for DiffusionGemma's block-diffusion attention, including PrefixLM bidirectional range handling for multimodal tokens."
        },
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm model registry and configs",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "benchmarks/kernels/benchmark_moe.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "The foundational patch that teaches vLLM about the DiffusionGemma architecture. It updates benchmarks/kernels/benchmark_moe.py to read num_experts, topk, intermediate and hidden sizes from the model's text_config for 'DiffusionGemmaForBlockDiffusion', and (per its full diff) adds model registration, scheduler/decoder support for block diffusion sampling, and related plumbing.",
          "file_path": ".litho/tree/repo/mods/diffusiongemma/diffusiongemma-support.patch",
          "importance_score": 0.8,
          "interfaces": [],
          "name": "diffusiongemma-support.patch",
          "responsibilities": [
            "Register the DiffusionGemmaForBlockDiffusion architecture with vLLM",
            "Wire MoE benchmark parameters from the model's text_config",
            "Provide block-diffusion decoding/sampling support in the engine",
            "Act as the primary patch applied by run.sh"
          ],
          "source_summary": "A multi-file unified diff; the visible hunk extends get_model_params in benchmarks/kernels/benchmark_moe.py with an elif branch for DiffusionGemmaForBlockDiffusion that extracts MoE configuration (experts, topk, intermediate_size, hidden_size) from the text_config, with additional hunks across vLLM model/config/serving modules.",
          "summary": "Core model-support patch that registers the DiffusionGemmaForBlockDiffusion architecture in vLLM, including config parsing (text_config MoE parameters), model implementation, and benchmark parameter wiring."
        },
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm/entrypoints/openai/chat_completion/serving.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Modifies vllm/entrypoints/openai/chat_completion/serving.py to define Gemma4 reasoning channel constants and a _strip_gemma4_content_channels helper that removes internal 'thought' channel text from assistant content. This is the legacy-branch counterpart of the sanitizer patch, applied when the main-branch patch does not apply cleanly.",
          "file_path": ".litho/tree/repo/mods/diffusiongemma/gemma4-content-channel-sanitizer-legacy.patch",
          "importance_score": 0.55,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_strip_gemma4_content_channels",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "content",
                  "param_type": "str | None"
                }
              ],
              "return_type": "str | None",
              "visibility": ""
            }
          ],
          "name": "gemma4-content-channel-sanitizer-legacy.patch",
          "responsibilities": [
            "Strip reasoning/thought channel markup from response content",
            "Hide internal chain-of-thought from API consumers on the legacy branch",
            "Provide fallback variant of the sanitizer patch"
          ],
          "source_summary": "A unified diff inserting module-level constants (_GEMMA4_REASONING_START, _GEMMA4_REASONING_END, _GEMMA4_THOUGHT_PREFIX) and the _strip_gemma4_content_channels function near the logger setup, plus call sites that sanitize content in non-streaming chat completion responses.",
          "summary": "Legacy-branch patch adding a sanitizer that strips Gemma4 reasoning channel markers (<|channel>thought ... <channel|>) from OpenAI chat completion content before returning responses."
        },
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm/entrypoints/openai/chat_completion/serving.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Adds the same _GEMMA4_* constants and _strip_gemma4_content_channels helper to the current vLLM chat completion serving module, hooking it into response assembly. Works together with the streaming-reasoning patch to give complete reasoning suppression across both streaming and non-streaming paths.",
          "file_path": ".litho/tree/repo/mods/diffusiongemma/gemma4-content-channel-sanitizer.patch",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_strip_gemma4_content_channels",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "content",
                  "param_type": "str | None"
                }
              ],
              "return_type": "str | None",
              "visibility": ""
            }
          ],
          "name": "gemma4-content-channel-sanitizer.patch",
          "responsibilities": [
            "Sanitize non-streaming chat completion content",
            "Remove <|channel>thought ... <channel|> blocks from outputs",
            "Keep reasoning hidden unless explicitly requested"
          ],
          "source_summary": "A unified diff against vllm/entrypoints/openai/chat_completion/serving.py that inserts reasoning-channel constants and a sanitizer function, then applies it to message content before building OpenAI ChatCompletion responses.",
          "summary": "Main-branch patch that strips Gemma4 reasoning channel markers from chat completion content so internal thought text is not leaked to OpenAI API clients."
        },
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm/entrypoints/openai/chat_completion/serving.py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "OpenAI DeltaMessage schema",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Modifies OpenAIServingChat's streaming delta handling so that when request.include_reasoning is false, deltas belonging to the reasoning channel are filtered out, and when true they can be emitted as reasoning content. Complements the content-channel sanitizer for the streaming code path.",
          "file_path": ".litho/tree/repo/mods/diffusiongemma/gemma4-streaming-reasoning.patch",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "gemma4-streaming-reasoning.patch",
          "responsibilities": [
            "Filter reasoning-channel deltas during streaming",
            "Honor the include_reasoning request flag",
            "Emit reasoning content separately when enabled"
          ],
          "source_summary": "A unified diff inside the streaming loop of vllm/entrypoints/openai/chat_completion/serving.py that inspects generated deltas for channel markers and either skips reasoning-channel deltas (DeltaMessage) or converts them into reasoning_content fields for the client.",
          "summary": "Patch adding streaming support for Gemma4 reasoning channels in the OpenAI chat serving layer, suppressing or routing reasoning deltas based on the request's include_reasoning flag."
        },
        {
          "code_purpose": "doc",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "docs/design/attention_backends.md",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "A docs-only patch that revises the attention backends design document, correcting/extending the backend priority and capability matrix (dtypes, kv-cache formats, head-size support) so the documentation matches the patched attention behavior. Lowest-risk file in the kit, applied for documentation completeness.",
          "file_path": ".litho/tree/repo/mods/diffusiongemma/mr5-attention-backends-docs.patch",
          "importance_score": 0.35,
          "interfaces": [],
          "name": "mr5-attention-backends-docs.patch",
          "responsibilities": [
            "Keep attention backend design docs in sync with patches",
            "Document backend dtype/kv-format/head-size support matrix",
            "Support FlashInfer and CPU backend documentation"
          ],
          "source_summary": "A unified diff against docs/design/attention_backends.md modifying a table row area around line 172, updating entries such as CPU_ATTN and FLASHINFER with supported dtypes (fp16/bf16/fp32), kv cache formats (auto, fp8 variants), and head-size constraints.",
          "summary": "Documentation patch updating docs/design/attention_backends.md with attention backend capability tables, including FlashInfer and CPU backend dtype/format coverage relevant to the DiffusionGemma attention changes."
        },
        {
          "code_purpose": "entry",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "diffusiongemma-support.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "diffusiongemma-attention-main.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "diffusiongemma-attention-legacy.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "gemma4-streaming-reasoning.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "gemma4-content-channel-sanitizer.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "chat_template_no_think.jinja",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "patch / bash coreutils",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "The executable entry point of the directory: it resolves the Python site-packages root, applies diffusiongemma-support.patch first, then tries the main-branch attention and sanitizer patches with legacy fallbacks, plus the streaming-reasoning and docs patches, and finally starts the server using chat_template_no_think.jinja. It uses 'set -euo pipefail' for fail-fast behavior.",
          "file_path": ".litho/tree/repo/mods/diffusiongemma/run.sh",
          "importance_score": 0.85,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "main (script body)",
              "parameters": [],
              "return_type": "exit code",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Locate the installed vLLM package directory",
            "Apply model-support, attention, sanitizer, streaming, and docs patches in order",
            "Choose between legacy and main patch variants automatically",
            "Launch the vLLM server with the DiffusionGemma chat template"
          ],
          "source_summary": "A bash script defining variables for the Python root, module directory, and each patch file path (support, attention legacy/main, streaming reasoning, content channel sanitizer, docs), then applying them via patch/dry-run logic with fallback selection between legacy and main variants before invoking the vLLM entrypoint.",
          "summary": "Bash orchestration script that applies all DiffusionGemma/Gemma4 patches to the installed vLLM package (with legacy/main fallbacks) and then launches the vLLM server with the custom chat template."
        }
      ],
      "importance_score": 0.72,
      "key_files": [
        "run.sh",
        "diffusiongemma-support.patch",
        "diffusiongemma-attention-main.patch",
        "gemma4-content-channel-sanitizer.patch",
        "chat_template_no_think.jinja"
      ],
      "name": "diffusiongemma",
      "path": ".litho/tree/repo/mods/diffusiongemma",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The 'diffusiongemma' directory is a self-contained integration kit for running the DiffusionGemma (Gemma4 block-diffusion) model on a patched vLLM stack. It bundles a custom Jinja chat template, five patch files that add model support, per-token causal attention flexibility, reasoning-channel sanitization, and streaming reasoning handling, plus a run.sh orchestrator that applies the patches to the installed vLLM package before launching the server."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "command",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "nohup",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "sync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "/proc/sys/vm/drop_caches",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "sleep",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script is an operational utility that starts a detached background process (via nohup) running an infinite loop that executes 'sync; echo 3 > /proc/sys/vm/drop_caches' every 60 seconds. It records the background PID to /tmp/drop_caches.pid and redirects output to /tmp/drop_caches.log, enabling later inspection or termination of the loop.",
          "file_path": ".litho/tree/repo/mods/drop-caches/run.sh",
          "importance_score": 0.3,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "run.sh (script execution)",
              "parameters": [],
              "return_type": "void (prints startup message with PID and log path)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Periodically sync filesystem buffers and drop page cache, dentries, and inodes via /proc/sys/vm/drop_caches",
            "Run the cache-drop loop as a detached background process using nohup",
            "Persist the background process PID to /tmp/drop_caches.pid for later management",
            "Log command output to /tmp/drop_caches.log for diagnostics",
            "Report startup status including PID and log location"
          ],
          "source_summary": "The script defines a CMD variable containing the cache-drop command, a LOG path (/tmp/drop_caches.log), and a PIDFILE path (/tmp/drop_caches.pid). It spawns a nohup'd bash subshell with a while-true loop that appends the command output to the log and sleeps 60 seconds between iterations, then writes the background process PID to the pidfile and prints a confirmation message with the PID and log location.",
          "summary": "A bash script that launches a background loop dropping filesystem caches every minute to help load very large models."
        }
      ],
      "importance_score": 0.3,
      "key_files": [
        "run.sh"
      ],
      "name": "drop-caches",
      "path": ".litho/tree/repo/mods/drop-caches",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The drop-caches directory contains a single operational shell script that periodically flushes Linux filesystem caches (page cache, dentries, inodes) every 60 seconds via /proc/sys/vm/drop_caches. It is a DevOps/infrastructure utility used to unstuck large model loading (e.g., Qwen3.5-397B) by preventing memory pressure from cached filesystem data. It has no business logic and serves purely as a system maintenance tool."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "entry",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm (flashinfer_b12x_moe.py module)",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "build-and-copy.sh",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This shell script acts as the execution entry point for the experimental b12x-patches mod. It begins with strict error handling (set -e), defines the Python site-packages path, and verifies that the vLLM installation includes the flashinfer_b12x_moe.py module required by vLLM PR 40082, exiting with an actionable rebuild instruction if it is missing. The script then proceeds to check required environment variables (content truncated in the provided source).",
          "file_path": ".litho/tree/repo/mods/exp-b12x/run.sh",
          "importance_score": 0.5,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "run.sh (main execution flow)",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "environment variables",
                  "param_type": "env"
                }
              ],
              "return_type": "exit code (0 on success, 1 on missing b12x support)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Validate that the installed vLLM build includes b12x MoE support (flashinfer_b12x_moe.py)",
            "Provide actionable rebuild instructions when prerequisites are missing",
            "Configure the Python site-packages path for the runtime environment",
            "Check required environment variables before running the experiment",
            "Serve as the execution entry point for the experimental b12x-patches setup"
          ],
          "source_summary": "The script starts with '#!/bin/bash' and 'set -e' for fail-fast behavior, sets SITE_PACKAGES to /usr/local/lib/python3.12/dist-packages, and prints an 'EXPERIMENTAL b12x-patches mod' banner. Its first functional step (0a) checks for the existence of vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py and aborts with guidance to rebuild via './build-and-copy.sh -t vllm-node-40082 --apply-vllm-pr 40082' if absent. Step 0b begins checking environment variables, though the remainder of the script is truncated.",
          "summary": "Bash entry script that sets up and validates an experimental vLLM environment with b12x-patches modifications."
        }
      ],
      "importance_score": 0.45,
      "key_files": [
        "run.sh"
      ],
      "name": "exp-b12x",
      "path": ".litho/tree/repo/mods/exp-b12x",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The exp-b12x directory contains a single experimental setup script for enabling the 'b12x-patches' modification of vLLM. The run.sh script validates that the environment has the required b12x MoE support (flashinfer_b12x_moe.py from vLLM PR 40082) before proceeding with further environment checks and setup steps. It serves as an entry point for running an experimental vLLM configuration rather than containing core business logic."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "curl",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "mktemp",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script automates the setup of a W4A16 quantization experiment environment by applying selected vLLM pull requests to the system-installed vLLM package. It uses strict error handling (set -euo pipefail), validates the vLLM installation path, and manages a temporary directory with a trap-based cleanup. Each PR is fetched via curl and applied as a patch.",
          "file_path": ".litho/tree/repo/mods/exp-w4a16/run.sh",
          "importance_score": 0.5,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "apply_pr",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "pr",
                  "param_type": "string (PR number)"
                }
              ],
              "return_type": "void",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "command",
              "name": "main (script execution)",
              "parameters": [],
              "return_type": "int (exit code)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Validate that the vLLM package is installed at the expected path",
            "Create and clean up a temporary working directory via trap",
            "Download PR diffs (42124, 42566, 42546) from a remote source via curl",
            "Apply the PR patches to the installed vLLM package with per-PR check logs",
            "Enforce strict shell error handling with set -euo pipefail"
          ],
          "source_summary": "The script defines PYTHON_ROOT and VLLM_ROOT paths pointing to /usr/local/lib/python3.12/dist-packages/vllm and a PR list (42124 42566 42546). It checks that the vLLM directory exists and exits with an error message otherwise, creates a temp directory with mktemp and an EXIT trap to remove it, then defines an apply_pr function that downloads each PR diff via curl into the temp dir and applies it (with a check log per PR) after cd-ing into the Python packages root.",
          "summary": "Bash entry script that patches an installed vLLM package with specific W4A16-related PRs."
        }
      ],
      "importance_score": 0.45,
      "key_files": [
        "run.sh"
      ],
      "name": "exp-w4a16",
      "path": ".litho/tree/repo/mods/exp-w4a16",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The exp-w4a16 directory contains a single experiment automation script for applying W4A16 quantization-related pull requests to a locally installed vLLM package. run.sh verifies the vLLM installation, downloads diffs for a set of PRs (42124, 42566, 42546), and applies them in a temporary workspace with cleanup on exit. It serves as infrastructure/tooling for a quantization experiment rather than core business logic."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "entry",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "verify.py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This bash script runs on every head and worker node before the launch script. It defaults FES_WEIGHTS_DIR to /model, handles optional FES_DRAFT_DIR weights, and invokes verification so vLLM can resolve a model locally rather than querying the Hugging Face hub.",
          "file_path": ".litho/tree/repo/mods/fes-weights/run.sh",
          "importance_score": 0.84,
          "interfaces": [],
          "name": "run.sh",
          "responsibilities": [
            "Verify FES-staged model weights are mounted inside the container",
            "Create HF hub-cache layout so vllm serve resolves offline",
            "Support optional draft-model weights via FES_DRAFT_DIR",
            "Run on head and worker nodes before the launch script executes"
          ],
          "source_summary": "The script header documents its purpose and environment variables, then verifies the staged FES weights mount and, when FES_HUB_MODEL is set, reorganizes or links weights into the HF hub-cache layout. The full body is truncated, but the available content confirms it is the mod's node-level activation script.",
          "summary": "Shell entrypoint for the fes-weights mod; verifies mounted weights and sets up HF hub-cache layout for offline vLLM serving when FES_HUB_MODEL is set."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "json",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "os",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This module ports the FES verify_location checks into Python: it compares actual shard files against model.safetensors.index.json, verifies total size parity within a 1.02 tolerance, and requires hf_quant_config.json for nvfp4 checkpoints. It is intended to be called before model serving and is the core validation logic used by run.sh.",
          "file_path": ".litho/tree/repo/mods/fes-weights/verify.py",
          "importance_score": 0.81,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "fail",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "reason",
                  "param_type": "Any"
                }
              ],
              "return_type": "NoReturn",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "verify",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "path",
                  "param_type": "Any"
                }
              ],
              "return_type": "dict",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "verify.py",
          "responsibilities": [
            "Validate shard count against model.safetensors.index.json",
            "Check total on-disk size against index metadata within tolerance",
            "Ensure hf_quant_config.json exists for nvfp4 checkpoints",
            "Print JSON summary on success or a single failure reason on stderr"
          ],
          "source_summary": "The file defines fail(), verify(), and main(). fail() writes a reason to stderr and exits with status 1; verify() inspects shard counts, sizes, and quantization config; main() emits a JSON summary on success. Metrics show 77 lines, 3 functions, and complexity 16.0, reflecting multiple validation branches.",
          "summary": "Standalone Python verifier for a staged model weights directory, reporting success as JSON or failing with a single reason line."
        }
      ],
      "importance_score": 0.76,
      "key_files": [
        "run.sh",
        "verify.py"
      ],
      "name": "fes-weights",
      "path": ".litho/tree/repo/mods/fes-weights",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The `fes-weights` directory is a Litho mod that validates staged model weight directories and prepares them for offline Hugging Face hub resolution. `run.sh` acts as the node-level entrypoint that orchestrates weight verification and hub-cache layout setup, while `verify.py` performs shard, size, and quantization-config checks. These files work together to make `vllm serve <org/model>` work on head and worker nodes without contacting the HF hub."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm.model_executor.models.glm4_moe",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "patch (unified diff format)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This patch modifies /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe.py at runtime, inserting a guard clause inside the parameter-loading loop. When a parameter name contains 'k_scale' or 'v_scale' and is not present in params_dict, loading is skipped instead of raising a KeyError. This enables NVFP4-quantized GLM-4 MoE checkpoints (such as Salyut1) to load successfully in vLLM.",
          "file_path": ".litho/tree/repo/mods/fix-Salyut1-GLM-4.7-NVFP4/glm4_moe.patch",
          "importance_score": 0.7,
          "interfaces": [],
          "name": "glm4_moe.patch",
          "responsibilities": [
            "Patch vLLM's glm4_moe.py weight-loading logic at deployment time",
            "Skip missing k_scale/v_scale quantization parameters during checkpoint loading",
            "Enable compatibility of NVFP4-quantized GLM-4 MoE models with vLLM",
            "Prevent KeyError crashes when checkpoint scales are absent from params_dict"
          ],
          "source_summary": "The diff targets the weight-loading section of the Glm4Moe model class (around line 537), where is_pp_missing_parameter checks occur. It adds a single-line condition: if the parameter name includes 'k_scale' or 'v_scale' and the name is not in params_dict, the loop continues without attempting to access the parameter. This prevents crashes when quantization scale tensors are present in the checkpoint but not registered as model parameters.",
          "summary": "A unified diff patch that fixes the vLLM GLM-4 MoE model's weight loading to tolerate missing k_scale/v_scale quantization scale parameters."
        },
        {
          "code_purpose": "entry",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "glm4_moe.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "run.sh is the executable entry point for this fix package. It uses 'set -e' to abort on any error and invokes 'patch -p1 -d /' to apply glm4_moe.patch against the root filesystem, rewriting the installed vLLM glm4_moe.py in place. It is typically run inside a container or environment where the vLLM package lives at /usr/local/lib/python3.12/dist-packages.",
          "file_path": ".litho/tree/repo/mods/fix-Salyut1-GLM-4.7-NVFP4/run.sh",
          "importance_score": 0.5,
          "interfaces": [
            {
              "description": null,
              "interface_type": "shell_script",
              "name": "run.sh",
              "parameters": [],
              "return_type": "exit_code",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Serve as the deployment entry point for applying the fix",
            "Apply glm4_moe.patch to the root filesystem via the patch utility",
            "Fail fast (set -e) if the patch application encounters errors"
          ],
          "source_summary": "The script consists of a bash shebang, 'set -e' for fail-fast behavior, and a single patch command: 'patch -p1 -d / < glm4_moe.patch'. The -p1 flag strips the leading path component (a/) so the diff applies to the absolute path /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe.py. No other setup, validation, or cleanup logic is present.",
          "summary": "A shell entry script that applies the glm4_moe.patch to the filesystem root using the patch utility, with error-on-failure semantics."
        }
      ],
      "importance_score": 0.55,
      "key_files": [
        "glm4_moe.patch",
        "run.sh"
      ],
      "name": "fix-Salyut1-GLM-4.7-NVFP4",
      "path": ".litho/tree/repo/mods/fix-Salyut1-GLM-4.7-NVFP4",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a targeted hotfix for the vLLM inference framework's GLM-4 MoE model implementation, packaged as a unified diff patch plus a shell script that applies it. The patch modifies the weight-loading logic in glm4_moe.py to skip k_scale/v_scale quantization parameters that are absent from the model's parameter dictionary, resolving load failures for NVFP4-quantized checkpoints (e.g., Salyut1). run.sh applies the patch directly against the filesystem root, making this a deployment-time fix utility rather than core application code."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm.v1.core.sched.scheduler",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "git apply",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This unified diff contains the actual code fix for the EAGLE fine-grained prefix caching issue. It inserts a conditional block in the Scheduler class that, when use_eagle is set and tail_boundary > 0, caches the predecessor FullAttention hash so that changes to the prompt suffix do not invalidate the only partially reusable prefix cache entries.",
          "file_path": ".litho/tree/repo/mods/fix-eagle-fine-prefix/fix-eagle-fine-prefix.patch",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "patch",
              "name": "apply patch to vllm/v1/core/sched/scheduler.py",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "target_file",
                  "param_type": "file path"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "hunk_context",
                  "param_type": "Scheduler.schedule context around line 393"
                }
              ],
              "return_type": "modified scheduler.py",
              "visibility": ""
            }
          ],
          "name": "fix-eagle-fine-prefix.patch",
          "responsibilities": [
            "Define the code change fixing EAGLE prefix cache invalidation",
            "Patch the vLLM v1 Scheduler scheduling logic",
            "Preserve predecessor FullAttention hash across prompt suffix changes",
            "Integrate with existing mamba partial cache hit logic"
          ],
          "source_summary": "The patch targets vllm/v1/core/sched/scheduler.py (hunk near line 393 in the Scheduler class). It adds roughly five lines that check self.use_eagle and tail_boundary > 0, then cache the predecessor FullAttention hash with a comment explaining that this prevents changed prompt suffixes from invalidating the prefix cache. The diff context shows interaction with mamba_partial_cache_hit logic.",
          "summary": "A git patch that modifies vLLM's v1 scheduler to cache the predecessor FullAttention hash when EAGLE is enabled, fixing prefix cache invalidation for changed prompt suffixes."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "git",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "fix-eagle-fine-prefix.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm (installed package)",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/use-official-vllm",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script is the entry point for installing the mod. It validates that git is available and that the vLLM package exists at the configured PYTHON_ROOT, then applies the accompanying patch file. It uses strict shell mode (set -euo pipefail) and a '[fix-eagle-fine-prefix]' log prefix, and hints that mods/use-official-vllm may need to be applied first.",
          "file_path": ".litho/tree/repo/mods/fix-eagle-fine-prefix/run.sh",
          "importance_score": 0.55,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "main (script execution)",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "PYTHON_ROOT",
                  "param_type": "environment variable (path to python site-packages)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "MOD_DIR",
                  "param_type": "derived from script location"
                }
              ],
              "return_type": "exit code (0 on success, 1 on missing prerequisites)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Validate prerequisites (git availability, vLLM package presence)",
            "Locate the patch file relative to the script directory",
            "Apply the patch to the installed vLLM source tree",
            "Provide prefixed error/logging output and fail fast with set -euo pipefail"
          ],
          "source_summary": "The script sets PYTHON_ROOT (default /usr/local/lib/python3.12/dist-packages), resolves its own directory to locate the patch file, and defines a log prefix. It checks for the git command and exits with an error if missing, verifies the vLLM package directory exists at $PYTHON_ROOT/vllm, and then uses git to apply fix-eagle-fine-prefix.patch to the installed package.",
          "summary": "Bash installer script that applies the fix-eagle-fine-prefix.patch to the installed vLLM package using git, with environment checks and prefixed logging."
        }
      ],
      "importance_score": 0.55,
      "key_files": [
        "run.sh",
        "fix-eagle-fine-prefix.patch"
      ],
      "name": "fix-eagle-fine-prefix",
      "path": ".litho/tree/repo/mods/fix-eagle-fine-prefix",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory is a self-contained 'mod' package that patches the vLLM v1 scheduler to fix prefix caching behavior when EAGLE speculative decoding is enabled. The patch file modifies vllm/v1/core/sched/scheduler.py so that the predecessor FullAttention hash is cached, preventing changed prompt suffixes from invalidating the prefix cache, while run.sh is the installer script that applies the patch via git to the installed vLLM package."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "curl",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "git",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm (installed in /usr/local/lib/python3.12/dist-packages)",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script acts as a deployment-time hotfix utility for a containerized environment where vLLM is installed as a system Python package. It changes into the dist-packages directory, fetches the unified diff for vLLM pull request #38909 from GitHub's patch-diff endpoint, and applies it while excluding tests. Error handling with 'set -e' plus an if-guard ensures the script reports success or skips the patch without aborting the surrounding startup process.",
          "file_path": ".litho/tree/repo/mods/fix-gemma4-tool-parser/run.sh",
          "importance_score": 0.45,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "main (script execution)",
              "parameters": [],
              "return_type": "exit code (0 on success or skipped patch)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Apply upstream vLLM PR #38909 patch to installed package",
            "Download patch diff from GitHub via curl",
            "Exclude test files from the patch application",
            "Handle patch failure gracefully without aborting startup",
            "Log patch application status"
          ],
          "source_summary": "The script enables strict error mode (set -e), changes directory to /usr/local/lib/python3.12/dist-packages, and prints 'Applying PR #38909'. It pipes the PR #38909 diff from GitHub (via curl -fsL) into 'git apply --exclude=\"tests/*\"', echoing a success message if the patch applies or a skip message if it fails. No functions or variables are defined; it is a linear, single-purpose script.",
          "summary": "A bash script that applies the vLLM PR #38909 patch (Gemma 4 tool parser fix) to the installed vLLM package in /usr/local/lib/python3.12/dist-packages. It downloads the PR diff via curl and applies it with git apply, excluding test files, and skips gracefully on failure."
        }
      ],
      "importance_score": 0.45,
      "key_files": [
        "run.sh"
      ],
      "name": "fix-gemma4-tool-parser",
      "path": ".litho/tree/repo/mods/fix-gemma4-tool-parser",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a single operational shell script that patches an installed vLLM Python package at runtime. It applies PR #38909 (a fix for the Gemma 4 tool parser) by downloading the PR diff from GitHub and applying it to the vLLM source in the system's dist-packages directory, gracefully skipping if the patch cannot be applied. It serves as a deployment-time fix/hotfix utility rather than core application logic."
    },
    {
      "file_count": 3,
      "file_insights": [
        {
          "code_purpose": "entry",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "grep",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "glm47_flash.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "glm47_vllm_bug.patch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script is the main entry point for the directory's fix workflow. It applies glm47_flash.patch to the root filesystem, then checks whether the vLLM mla_attention.py file already contains the fix from PR 34695 before applying glm47_vllm_bug.patch, making the patching process idempotent. It uses 'set -e' to abort on any failure.",
          "file_path": ".litho/tree/repo/mods/fix-glm-4.7-flash-AWQ/run.sh",
          "importance_score": 0.75,
          "interfaces": [
            {
              "description": null,
              "interface_type": "shell_script",
              "name": "main",
              "parameters": [],
              "return_type": "exit_code",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Apply the GLM 4.7 Flash Triton MLA performance patch at startup",
            "Detect whether the vLLM crash fix (PR 34695) is already present",
            "Apply the vLLM MLA attention crash patch only when needed",
            "Fail fast on errors via 'set -e'"
          ],
          "source_summary": "A bash script with 'set -e' that first applies glm47_flash.patch via 'patch -p1 -d /' targeting /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/triton_mla.py. It then checks for the existence of vllm/model_executor/layers/attention/mla_attention.py and greps for the PR 34695 fix line ('and hasattr(self.kv_b_proj, \"weight\")') to decide whether the crash patch still needs to be applied.",
          "summary": "Entry-point shell script that applies the GLM 4.7 AWQ performance and crash-fix patches to the installed vLLM package, with checks to avoid double-applying."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm.v1.attention.backends.mla.triton_mla",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This patch targets triton_mla.py in vLLM's v1 attention backends. It replaces the batch-invariance-based fixed num_kv_splits logic (1 or 4 splits) with a dynamic split calculation tuned for the GLM 4.7 Flash AWQ model, improving attention kernel throughput while preserving deterministic reduction behavior where required.",
          "file_path": ".litho/tree/repo/mods/fix-glm-4.7-flash-AWQ/glm47_flash.patch",
          "importance_score": 0.7,
          "interfaces": [],
          "name": "glm47_flash.patch",
          "responsibilities": [
            "Patch vLLM's triton_mla.py attention backend",
            "Replace fixed KV split count with dynamic token-based split sizing",
            "Clamp split count to [32, 128] for kernel stability",
            "Optimize GLM 4.7 Flash AWQ inference performance"
          ],
          "source_summary": "The diff modifies the Triton MLA kernel setup around line 135, where log-sum-exp (lse) tensors are initialized. The original code chose num_kv_splits = 1 if vllm_is_batch_invariant() else 4; the patched code computes splits dynamically at roughly 1.5K tokens per split, clamped to the range [32, 128], to speed up long-context attention for the GLM 4.7 Flash model.",
          "summary": "Unified diff patch that modifies vLLM's Triton MLA attention backend to use dynamic KV split sizing (~1.5K tokens per split, clamped to [32, 128]) instead of a fixed split count."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm.model_executor.layers.attention.mla_attention",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This patch targets mla_attention.py in vLLM's model executor layers. It guards the check of kv_b_proj.weight.dtype with hasattr so that quantized projections without a standard weight attribute do not crash when determining whether AITER Triton FP4 BMM is enabled. This is required for running AWQ-quantized GLM 4.7 Flash models.",
          "file_path": ".litho/tree/repo/mods/fix-glm-4.7-flash-AWQ/glm47_vllm_bug.patch",
          "importance_score": 0.68,
          "interfaces": [],
          "name": "glm47_vllm_bug.patch",
          "responsibilities": [
            "Patch vLLM's mla_attention.py MLA attention layer",
            "Guard kv_b_proj weight access with hasattr to avoid AttributeError",
            "Enable AWQ-quantized GLM 4.7 Flash models to run without crashing",
            "Incorporate the upstream fix from vLLM PR 34695"
          ],
          "source_summary": "The diff modifies the is_aiter_triton_fp4_bmm_enabled assignment around line 403. Originally it compared self.kv_b_proj.weight.dtype == torch.bfloat16 directly; the fix wraps the dtype access in a conditional expression: (self.kv_b_proj.weight.dtype if hasattr(self.kv_b_proj, \"weight\") else torch.bfloat16) == torch.bfloat16, preventing AttributeError for quantized layers.",
          "summary": "Unified diff patch fixing a vLLM crash (issue/PR 34695) in the MLA attention layer where kv_b_proj may lack a weight attribute when quantized (e.g., AWQ)."
        }
      ],
      "importance_score": 0.72,
      "key_files": [
        "run.sh",
        "glm47_flash.patch",
        "glm47_vllm_bug.patch"
      ],
      "name": "fix-glm-4.7-flash-AWQ",
      "path": ".litho/tree/repo/mods/fix-glm-4.7-flash-AWQ",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a set of deployment-time patches and an orchestration script for running the GLM 4.7 Flash model with AWQ quantization on vLLM. The two .patch files fix a Triton MLA attention kernel (dynamic KV split tuning for performance) and a crash bug in vLLM's MLA attention layer (missing weight attribute on quantized kv_b_proj), while run.sh applies both patches to the installed vLLM package at container startup with idempotency checks."
    },
    {
      "file_count": 5,
      "file_insights": [
        {
          "code_purpose": "entry",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "fix_crash.diff",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "fix_slowness.diff",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "patch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Acts as the orchestrator of this patch kit: it uses the 'patch' utility with -p1 against /usr/local/lib/python3.12/dist-packages to apply fix_crash.diff and reverse-apply fix_slowness.diff (reverting PR #34279). Each patch application is fault-tolerant, printing a skip message if the patch is not applicable (e.g., already applied or reverted upstream).",
          "file_path": ".litho/tree/repo/mods/fix-qwen3-coder-next/run.sh",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "run.sh",
              "parameters": [],
              "return_type": "exit code",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Apply fix_crash.diff to installed vLLM package",
            "Reverse-apply fix_slowness.diff to revert PR #34279",
            "Gracefully skip patches that are not applicable",
            "Provide operational logging of patch status"
          ],
          "source_summary": "A bash script with 'set -e' that echoes status messages and runs two patch commands: one forward-applying fix_crash.diff to fix Qwen3-Coder-Next crashing on start, and one reverse-applying (-R) fix_slowness.diff to undo PR #34279 which causes slowness. Both commands tolerate failure with '|| echo' fallbacks, and a commented-out grep check hints at an additional int64-overflow cast patch.",
          "summary": "Bash entry script that applies the crash-fix and slowness-revert patches to the installed vLLM distribution."
        },
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "triton.runtime._allocation",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Patches triton.runtime._allocation.NullAllocator.__call__ with a static lambda that delegates allocations to torch.cuda.caching_allocator_alloc, passing size and stream through. The entire patch is wrapped in try/except so failures (e.g., missing Triton or torch) are silently ignored, making it safe to import unconditionally at interpreter startup.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3-coder-next/_triton_alloc_setup.py",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "module",
              "name": "_triton_alloc_setup (module import side-effect)",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "_triton_alloc_setup.py",
          "responsibilities": [
            "Redirect Triton allocations to PyTorch's CUDA caching allocator",
            "Preserve stream-awareness in allocation calls",
            "Fail silently when Triton/torch are unavailable"
          ],
          "source_summary": "A 9-line module that imports triton.runtime._allocation and torch, then overrides NullAllocator.__call__ as a staticmethod lambda accepting (size, alignment, stream) and returning torch.cuda.caching_allocator_alloc(size, stream=stream). The try/except pass guard ensures the module never breaks startup if dependencies are absent.",
          "summary": "Monkeypatch module that replaces Triton's NullAllocator with PyTorch's CUDA caching allocator."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "_triton_alloc_setup",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "A single-line .pth file placed in site-packages; Python executes lines starting with 'import' in .pth files at startup, so this guarantees the Triton allocator monkeypatch is installed before any vLLM/Triton code runs. It works in tandem with _triton_alloc_setup.py to activate the allocator fix without code changes.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3-coder-next/_triton_alloc_setup.pth",
          "importance_score": 0.5,
          "interfaces": [],
          "name": "_triton_alloc_setup.pth",
          "responsibilities": [
            "Auto-import the allocator monkeypatch at Python startup",
            "Ensure the patch activates before vLLM/Triton load"
          ],
          "source_summary": "Contains only the line 'import _triton_alloc_setup', which Python's site module executes when processing .pth files at interpreter startup, triggering the allocator monkeypatch defined in _triton_alloc_setup.py.",
          "summary": "Python .pth hook file that auto-imports the _triton_alloc_setup monkeypatch at interpreter startup."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm/v1/core/single_type_kv_cache_manager.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Targets vllm/v1/core/single_type_kv_cache_manager.py around line 1000 in the cache_blocks method, modifying the loop that iterates over request blocks in req_to_blocks. Applied by run.sh with 'patch -p1' to resolve a crash that occurs when Qwen3-Coder-Next starts on vLLM.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3-coder-next/fix_crash.diff",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "fix_crash.diff",
          "responsibilities": [
            "Fix startup crash in KV cache block caching",
            "Patch vLLM's single_type_kv_cache_manager.py",
            "Serve as input to run.sh patch application"
          ],
          "source_summary": "A git-style diff (index 0b6b7ed42ac1..b6e0305a312d) modifying the cache_blocks function in vllm/v1/core/single_type_kv_cache_manager.py, adjusting the iteration over self.req_to_blocks[request.request_id] block slices between num_cached_blocks_before and num_cached_blocks_after.",
          "summary": "Unified diff patch fixing a startup crash in vLLM's single_type_kv_cache_manager.py cache_blocks function."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm/model_executor/layers/fused_moe/fused_moe.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Targets vllm/model_executor/layers/fused_moe/fused_moe.py around line 95 in fused_moe_kernel_gptq_awq, restoring prior stride computation logic. run.sh applies it with 'patch -p1 -R' (reverse mode) to undo the upstream PR that introduced slowness, skipping gracefully if the PR was already reverted in recent commits.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3-coder-next/fix_slowness.diff",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "fix_slowness.diff",
          "responsibilities": [
            "Revert performance-regressing PR #34279",
            "Patch vLLM's fused_moe.py MoE kernel",
            "Restore original stride computation in fused_moe_kernel_gptq_awq"
          ],
          "source_summary": "A git-style diff (index 63aae43c3ddf..6ca321f312d) modifying fused_moe_kernel_gptq_awq in vllm/model_executor/layers/fused_moe/fused_moe.py, changing stride-related lines (stride_am etc.) that describe pointer arithmetic for row traversal in the MoE GPTQ/AWQ kernel.",
          "summary": "Unified diff patch that reverts PR #34279 in vLLM's fused_moe.py to restore MoE kernel performance."
        }
      ],
      "importance_score": 0.55,
      "key_files": [
        "run.sh",
        "_triton_alloc_setup.py",
        "fix_crash.diff",
        "fix_slowness.diff",
        "_triton_alloc_setup.pth"
      ],
      "name": "fix-qwen3-coder-next",
      "path": ".litho/tree/repo/mods/fix-qwen3-coder-next",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory is a deployment-time patch kit for running the Qwen3-Coder-Next model on vLLM, containing a startup shell script, two unified diff patches (one fixing a startup crash in the KV cache manager, one reverting a PR that caused MoE slowness), and a Python monkeypatch module that redirects Triton's NullAllocator to PyTorch's CUDA caching allocator. The run.sh entry point applies the patches to the installed vLLM package in site-packages, while the .pth file ensures the allocator monkeypatch is auto-imported at Python startup."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "Path",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script is the core of the fix, using Python's ast module to programmatically locate the Qwen3NextSparseMoeBlock class definition and replace the gate call to use checkpoint-configured quantization. It defines three main functions: gate_call to find the relevant call, patched_text to produce the modified source code, and main for CLI execution.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3-next-autoround/patch_qwen3_next.py",
          "importance_score": 0.9,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "gate_call",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                }
              ],
              "return_type": "ast.Call",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "patched_text",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "int",
              "visibility": ""
            }
          ],
          "name": "patch_qwen3_next.py",
          "responsibilities": [
            "Locate and identify the Qwen3NextSparseMoeBlock class in source code",
            "Detect and replace the gate call to use checkpoint-configured quantization",
            "Provide command-line interface for applying the patch",
            "Output patched source code for further use"
          ],
          "source_summary": "The script parses the source code to find the Qwen3NextSparseMoeBlock class, locates the gate call, and replaces it with a version that respects the checkpoint's quantization settings. It provides a CLI interface for applying the patch to a vLLM installation.",
          "summary": "AST-based patch script that modifies the Qwen3NextSparseMoeBlock class to restore checkpoint-configured quantization for the MoE router."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "python3",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "patch_qwen3_next.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script automates the patching process by finding the active Python interpreter's vLLM package location without importing vLLM (to avoid CUDA initialization). It supports an environment variable override for offline fixture tests and executes the patch script to modify the vLLM source files.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3-next-autoround/run.sh",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "run.sh",
          "responsibilities": [
            "Locate vLLM package path without importing vLLM",
            "Support environment variable override for test fixtures",
            "Invoke the patch script to modify vLLM source",
            "Ensure robust error handling via bash strict mode"
          ],
          "source_summary": "The script determines the vLLM package root, validates its existence, and then invokes patch_qwen3_next.py to apply the patch. It uses set -euo pipefail for robust error handling and provides a fallback for test environments.",
          "summary": "Shell script that locates the vLLM package and applies the patch_qwen3_next.py script to it, with support for offline testing."
        }
      ],
      "importance_score": 0.7,
      "key_files": [
        "patch_qwen3_next.py",
        "run.sh"
      ],
      "name": "fix-qwen3-next-autoround",
      "path": ".litho/tree/repo/mods/fix-qwen3-next-autoround",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a patch script and a runner script that restore checkpoint-configured quantization for the Qwen3-Next MoE router in vLLM. The patch script uses AST analysis to locate and modify the Qwen3NextSparseMoeBlock class, while the runner script sets up the environment and applies the patch to the vLLM package."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "transformers.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script is the executable entry point of the fix. It uses patch -p1 targeting the dist-packages directory so the diff paths (a/transformers/...) resolve correctly, and relies on 'set -e' with an '|| echo' fallback so a non-applicable patch does not fail the surrounding pipeline. It is intended to run once before AutoRound quantization jobs for Qwen3.5.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3.5-autoround/run.sh",
          "importance_score": 0.5,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "main (script execution)",
              "parameters": [],
              "return_type": "exit code (0 on success or skipped patch)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Apply the transformers rope-utils patch to the installed transformers package",
            "Ensure failures in patching do not abort the enclosing workflow (graceful skip)",
            "Target the correct Python 3.12 dist-packages location for patch -p1 path resolution"
          ],
          "source_summary": "A bash script with 'set -e' strict error handling that runs a single patch command against /usr/local/lib/python3.12/dist-packages using transformers.patch. If the patch is not applicable (e.g., already applied), it prints 'Patch is not applicable, skipping...' and continues.",
          "summary": "Shell entry script that applies the transformers.patch to the system-installed transformers package under /usr/local/lib/python3.12/dist-packages. It tolerates failure by echoing a skip message instead of aborting."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "transformers.modeling_rope_utils",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "The patch modifies the RoPE validation logic inside transformers' rotary embedding utilities, which is exercised by Qwen3.5-style configs that use partial_rotary_factor. Without the fix, the expression ignore_keys_at_rope_validation | {...} raises a TypeError when the value is None, breaking model loading/quantization. Wrapping the value in set() makes the union safe.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3.5-autoround/transformers.patch",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "transformers.patch",
          "responsibilities": [
            "Fix TypeError in RoPE key validation when ignore_keys_at_rope_validation is None",
            "Ensure partial_rotary_factor is always excluded from rope validation checks",
            "Enable Qwen3.5 AutoRound quantization workflows to run against patched transformers"
          ],
          "source_summary": "A single-hunk unified diff against a/transformers/modeling_rope_utils.py around line 648. It changes 'ignore_keys_at_rope_validation = ignore_keys_at_rope_validation | {\"partial_rotary_factor\"}' to 'ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation) | {\"partial_rotary_factor\"}', guarding against None/non-set inputs during rope validation.",
          "summary": "Unified diff patch that fixes a bug in transformers/modeling_rope_utils.py by coercing ignore_keys_at_rope_validation to a set before unioning with {'partial_rotary_factor'}. This prevents crashes when the config value is None or another non-set iterable."
        }
      ],
      "importance_score": 0.45,
      "key_files": [
        "run.sh",
        "transformers.patch"
      ],
      "name": "fix-qwen3.5-autoround",
      "path": ".litho/tree/repo/mods/fix-qwen3.5-autoround",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a small environment-fix utility for running AutoRound quantization on Qwen3.5 models. It consists of a shell script that applies a unified diff patch to the installed transformers package in site-packages, fixing a TypeError in modeling_rope_utils.py where a set union operation fails when ignore_keys_at_rope_validation is None or a non-set type. The patch is idempotent: if it cannot be applied (already applied or context mismatch), the script skips gracefully."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "jinja2 template engine (raise_exception, namespace)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file is the core artifact of the directory: a corrected chat template for Qwen3.5 that defines how chat messages are serialized into model prompts. It uses namespace counters to track image and video occurrences and a render_content macro to handle string, iterable, and multimodal message content, including validation that rejects images in system messages.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3.5-chat-template/chat_template.jinja",
          "importance_score": 0.85,
          "interfaces": [
            {
              "description": null,
              "interface_type": "macro",
              "name": "render_content",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "content",
                  "param_type": "string|list|dict"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "do_vision_count",
                  "param_type": "bool"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "is_system_content",
                  "param_type": "bool"
                }
              ],
              "return_type": "string (rendered template output)",
              "visibility": ""
            }
          ],
          "name": "chat_template.jinja",
          "responsibilities": [
            "Render chat conversation messages into the model's expected prompt format",
            "Track and process image and video content occurrences via namespace counters",
            "Handle string, iterable, and mapping message content types",
            "Validate content placement (e.g., reject images in system messages via raise_exception)"
          ],
          "source_summary": "The template initializes image_count and video_count namespace counters, defines a render_content macro that renders string content directly, iterates over list-style content items, detects image entries via 'image'/'image_url' keys or type=='image', and calls raise_exception when images appear in system content. It is a standard Hugging Face-style chat template with multimodal support fixes for Qwen3.5.",
          "summary": "A Jinja2 chat template for the Qwen3.5 model that renders conversation messages, including multimodal content with images and videos, and raises exceptions for invalid content placement."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "chat_template.jinja",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "WORKSPACE_DIR (environment variable)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This shell script automates installation of the fixed chat template by copying chat_template.jinja to $WORKSPACE_DIR/unsloth.jinja. It enables error propagation with 'set -e' and echoes a usage hint telling users to pass --chat-template unsloth.jinja when running the model.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3.5-chat-template/run.sh",
          "importance_score": 0.4,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "run.sh (main script execution)",
              "parameters": [],
              "return_type": "void (exit code)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Copy the fixed chat template into the target workspace directory",
            "Rename the template to unsloth.jinja for consumption by the unsloth tooling",
            "Print usage instructions for applying the template via --chat-template",
            "Fail fast on errors using 'set -e'"
          ],
          "source_summary": "The script starts with '#!/bin/bash' and 'set -e' for fail-fast behavior, copies chat_template.jinja to the path defined by the $WORKSPACE_DIR environment variable as unsloth.jinja, and echoes an informational message about applying the template via the --chat-template unsloth.jinja option.",
          "summary": "A small bash deployment script that copies the fixed chat template into the workspace as unsloth.jinja and prints instructions for applying it with the --chat-template flag."
        }
      ],
      "importance_score": 0.45,
      "key_files": [
        "chat_template.jinja",
        "run.sh"
      ],
      "name": "fix-qwen3.5-chat-template",
      "path": ".litho/tree/repo/mods/fix-qwen3.5-chat-template",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a fix/patch for the Qwen3.5 chat template, providing a custom Jinja chat template (chat_template.jinja) that handles multimodal content (images/videos) and a deployment script (run.sh) that copies the template into a workspace for use with unsloth via the --chat-template flag. The two files work together: the Jinja template is the actual fix artifact, and run.sh is the deployment helper that installs it."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "jinja2 template engine (raise_exception, namespace)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file is the core artifact of the directory: a patched chat template to be loaded by an inference server (e.g., vLLM/SGLang) via --chat-template. It tracks image and video counts using Jinja namespaces and defines a render_content macro that handles string content, iterable content lists, and multimodal items, raising exceptions for invalid placements such as images in system messages.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3.6-chat-template/chat_template.jinja",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "macro",
              "name": "render_content",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "content",
                  "param_type": "string|list|dict"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "do_vision_count",
                  "param_type": "bool"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "is_system_content",
                  "param_type": "bool"
                }
              ],
              "return_type": "string (rendered text)",
              "visibility": ""
            }
          ],
          "name": "chat_template.jinja",
          "responsibilities": [
            "Render conversation messages into the model's expected prompt format",
            "Handle multimodal content (images and videos) with counting logic",
            "Validate content placement (e.g., reject images in system messages)",
            "Serve as a drop-in replacement chat template for inference servers"
          ],
          "source_summary": "The template initializes image_count and video_count namespaces, then defines a render_content macro that renders string content directly, iterates over list-style content items, detects image/image_url items or type=='image' entries, and raises an exception when image content appears in system messages. It is a declarative template with no executable code beyond Jinja control flow.",
          "summary": "Jinja2 chat template for the Qwen3.6 model that converts structured conversation messages into the final prompt string, with special handling for multimodal (image/video) content."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "chat_template.jinja",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "bash / coreutils (cp, echo)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "A minimal shell script with 'set -e' error handling that copies chat_template.jinja to $WORKSPACE_DIR/fixed_chat_template.jinja and echoes a hint telling users to pass --chat-template fixed_chat_template.jinja when launching the inference server. It acts as the operational entry point for applying the template fix.",
          "file_path": ".litho/tree/repo/mods/fix-qwen3.6-chat-template/run.sh",
          "importance_score": 0.3,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "main (script execution)",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "WORKSPACE_DIR",
                  "param_type": "environment variable"
                }
              ],
              "return_type": "void",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Copy the fixed chat template into the target workspace",
            "Emit usage instructions for applying the template",
            "Fail fast on errors during deployment"
          ],
          "source_summary": "The script enables fail-fast behavior with 'set -e', performs a single cp command to place the fixed template in the workspace directory, and prints an informational message describing how to apply the template via the --chat-template command-line flag.",
          "summary": "Bash deployment helper that copies the fixed chat template into the workspace and prints usage instructions for applying it."
        }
      ],
      "importance_score": 0.45,
      "key_files": [
        "chat_template.jinja",
        "run.sh"
      ],
      "name": "fix-qwen3.6-chat-template",
      "path": ".litho/tree/repo/mods/fix-qwen3.6-chat-template",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a fix/patch for the Qwen3.6 chat template used in LLM inference serving. The chat_template.jinja file defines a Jinja2 template that renders conversation messages (including multimodal image/video content) into model-ready prompt text, while run.sh is a small deployment helper that copies the fixed template into a workspace and instructs users to apply it via the --chat-template flag. Together they form a self-contained template-fix utility package rather than core application logic."
    },
    {
      "file_count": 4,
      "file_insights": [
        {
          "code_purpose": "command",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "qwen3_5.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "qwen3_next.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "patch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Serves as the main entry point of the fix bundle, locating the installed vLLM model directory and applying the unified diff patches with 'patch --forward' so they are skipped if already applied. It documents the root cause (in_proj_ba output_size=128 / TP=4 = 32 falling below Marlin's min_thread_n=64) and the solution (ReplicatedLinear for B/A projections).",
          "file_path": ".litho/tree/repo/mods/fix-qwen35-tp4-marlin/run.sh",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "run.sh",
          "responsibilities": [
            "Orchestrate application of model patches to installed vLLM",
            "Make patching idempotent via patch --forward",
            "Document the Marlin TP=4 constraint and chosen fix",
            "Fail fast on errors with set -e"
          ],
          "source_summary": "A bash script with 'set -e' that resolves MOD_DIR relative to itself, points at /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models, echoes a status message, and applies the qwen3_5.patch and qwen3_next.patch files idempotently using the patch utility.",
          "summary": "Bash entry script that orchestrates applying the vLLM patches for the Qwen3.5 TP=4 Marlin fix."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "qwen3_5.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Targets the Qwen3.5 MoE model implementation in vLLM, altering the linear projection setup inside the attention/SSM block. By removing the tensor-parallel sharding of the in_proj_ba layer, it ensures the B/A projection outputs remain large enough for Marlin kernel execution under TP=4.",
          "file_path": ".litho/tree/repo/mods/fix-qwen35-tp4-marlin/qwen3_5.patch",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "qwen3_5.patch",
          "responsibilities": [
            "Patch vLLM qwen3_5.py model code",
            "Replace MergedColumnParallelLinear B/A path with replicated projections",
            "Preserve full B/A output dimensions under tensor parallelism"
          ],
          "source_summary": "A unified diff against qwen3_5.py.orig that modifies the split of mixed_qkvz into qkv and z components and replaces the sharded in_proj_ba computation (chunked b/a tensors) with replicated B/A projections producing full outputs, with comments explaining the compatibility rationale.",
          "summary": "Unified diff patch modifying vLLM's qwen3_5.py to replace sharded B/A projections with replicated full-output projections."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "qwen3_next.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Applies the same TP=4/Marlin fix to the Qwen3-Next model implementation in vLLM. It changes the in_proj_ba layer definition from a MergedColumnParallelLinear (kept for Qwen3_5 compatibility) to ReplicatedLinear, avoiding blockwise FP8 quantization issues and the Marlin min-thread constraint.",
          "file_path": ".litho/tree/repo/mods/fix-qwen35-tp4-marlin/qwen3_next.patch",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "qwen3_next.patch",
          "responsibilities": [
            "Patch vLLM qwen3_next.py model code",
            "Convert in_proj_ba to ReplicatedLinear",
            "Avoid blockwise FP8 quantization incompatibility for ba_proj"
          ],
          "source_summary": "A unified diff against qwen3_next.py.orig that modifies the constructor around line 411 where in_proj_qkvz is created, replacing the MergedColumnParallelLinear definition of in_proj_ba with ReplicatedLinear and updating associated comments about ba_proj quantization support.",
          "summary": "Unified diff patch modifying vLLM's qwen3_next.py to switch in_proj_ba from MergedColumnParallelLinear to ReplicatedLinear."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "re",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm.transformers_utils.configs.qwen3_5_moe",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "A one-off fixer script that edits the installed vLLM transformers_utils config for Qwen3.5 MoE in place. It resolves a transformers incompatibility where the config uses set union (|) on ignore_keys_at_rope_validation, which fails when the value is a list.",
          "file_path": ".litho/tree/repo/mods/fix-qwen35-tp4-marlin/fix_rope.py",
          "importance_score": 0.4,
          "interfaces": [],
          "name": "fix_rope.py",
          "responsibilities": [
            "Patch vLLM qwen3_5_moe config file in place",
            "Convert ignore_keys_at_rope_validation from list to set",
            "Fix RoPE validation TypeError under newer transformers"
          ],
          "source_summary": "Opens /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/qwen3_5_moe.py, reads its content, and performs a regex/text replacement converting kwargs[\"ignore_keys_at_rope_validation\"] from a list literal ([\"mrope_section\", \"mrope_interleaved\"]) to a set literal ({...}), then writes the file back.",
          "summary": "Standalone Python script that rewrites vLLM's qwen3_5_moe config file, changing ignore_keys_at_rope_validation from a list to a set."
        }
      ],
      "importance_score": 0.5,
      "key_files": [
        "run.sh",
        "qwen3_5.patch",
        "qwen3_next.patch",
        "fix_rope.py"
      ],
      "name": "fix-qwen35-tp4-marlin",
      "path": ".litho/tree/repo/mods/fix-qwen35-tp4-marlin",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory is a hotfix bundle for running the Qwen3.5-397B MoE model on vLLM with tensor parallelism TP=4 and Marlin quantization. It contains a shell driver (run.sh) that applies two unified diff patches to vLLM's model implementation files, replacing MergedColumnParallelLinear B/A projections with ReplicatedLinear to satisfy Marlin's minimum-thread constraint, plus a Python script that patches a config file to fix a list-vs-set type mismatch in RoPE validation ignore keys. The files work together as a portable, idempotent delivery mechanism for monkey-patching an installed vLLM distribution."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "python3",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "gpu_mem.patch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script serves as the executable entry point for the mod. It verifies that the vLLM package exists at a configurable PYTHON_ROOT and that python3 is available, then invokes an inline Python program (via heredoc) that uses ast and re to apply the patch to the installed vLLM source files.",
          "file_path": ".litho/tree/repo/mods/gpu-mem-util-gb/run.sh",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "main (script execution)",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "PYTHON_ROOT",
                  "param_type": "environment variable (string, default /usr/local/lib/python3.12/dist-packages)"
                }
              ],
              "return_type": "exit code (0 on success, non-zero on failure)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Validate that the vLLM package is installed at PYTHON_ROOT",
            "Check that python3 is available on the system",
            "Execute the embedded Python patcher to modify vLLM's CacheConfig",
            "Fail fast with clear error messages via set -euo pipefail"
          ],
          "source_summary": "The script sets strict shell options (set -euo pipefail), defaults PYTHON_ROOT to /usr/local/lib/python3.12/dist-packages, and performs existence checks for the vllm directory and python3 binary, exiting with error messages if missing. It then runs an embedded Python script that imports ast, re, sys, and pathlib to locate and modify vLLM's cache configuration source, applying the gpu_memory_utilization_gb change.",
          "summary": "Bash entry script that applies the gpu-mem-util-gb mod to an installed vLLM package by running an embedded Python patcher."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm.config.cache.CacheConfig",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "pydantic Field",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This patch file contains the actual code change that the mod applies to vLLM. It targets vllm/config/cache.py and inserts a new optional float field gpu_memory_utilization_gb (default None, must be > 0) alongside the existing gpu_memory_utilization fraction field, with documentation explaining its purpose.",
          "file_path": ".litho/tree/repo/mods/gpu-mem-util-gb/gpu_mem.patch",
          "importance_score": 0.55,
          "interfaces": [
            {
              "description": null,
              "interface_type": "config_field",
              "name": "gpu_memory_utilization_gb",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "default",
                  "param_type": "float | None (None)"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "gt",
                  "param_type": "int (0)"
                }
              ],
              "return_type": "float | None",
              "visibility": ""
            }
          ],
          "name": "gpu_mem.patch",
          "responsibilities": [
            "Define the new gpu_memory_utilization_gb configuration field for CacheConfig",
            "Specify validation constraints (optional, must be positive)",
            "Document the semantics of absolute GB-based GPU memory utilization",
            "Serve as the canonical change applied by run.sh to the installed vLLM package"
          ],
          "source_summary": "The diff modifies the CacheConfig class in vllm/config/cache.py, adding a gpu_memory_utilization_gb: float | None pydantic Field with default None and a greater-than-zero constraint. The accompanying docstring describes it as the amount of GPU memory to use in GB, complementing the existing fractional gpu_memory_utilization setting used when sharing a GPU between multiple vLLM instances.",
          "summary": "Unified diff patch that adds a gpu_memory_utilization_gb field to vLLM's CacheConfig, allowing GPU memory allocation to be specified in absolute gigabytes."
        }
      ],
      "importance_score": 0.55,
      "key_files": [
        "run.sh",
        "gpu_mem.patch"
      ],
      "name": "gpu-mem-util-gb",
      "path": ".litho/tree/repo/mods/gpu-mem-util-gb",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The gpu-mem-util-gb directory contains a self-contained 'mod' package that patches vLLM to support specifying GPU memory utilization as an absolute value in gigabytes (gpu_memory_utilization_gb) instead of only a fraction. run.sh is the entry point that validates the environment and applies gpu_mem.patch to the installed vLLM package, modifying vllm/config/cache.py's CacheConfig."
    },
    {
      "file_count": 3,
      "file_insights": [
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "collections.abc.Callable",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing.Any",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "vendor/inkling_sm120_fa4 (vendored FA4 bundle)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This module bridges the vendored SM12 FA4 kernel bundle and vLLM's Inkling inference path. The downstream bundle exposes preallocated output support only in its internal forward entry point, not in the public flash_attn_varlen_func wrapper, so this adapter re-exports/wraps the entry point to preserve the out= contract used during both warmup and inference without modifying bundled kernel sources.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/adapter.py",
          "importance_score": 0.75,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "adapter classes (2)",
              "parameters": [],
              "return_type": "n/a",
              "visibility": ""
            }
          ],
          "name": "adapter.py",
          "responsibilities": [
            "Wrap the vendored SM12 FA4 internal forward entry point",
            "Preserve the out= preallocated-output contract for vLLM Inkling",
            "Avoid modifying the bundled kernel sources",
            "Provide adapter classes bridging public and internal kernel APIs"
          ],
          "source_summary": "The file defines two classes and one function across 91 lines. It imports torch, Callable from collections.abc, and Any from typing, then adapts the vendored kernel's internal forward function so callers can pass out= preallocated tensors, maintaining compatibility with vLLM Inkling's calling convention.",
          "summary": "Runtime adapter that wraps the vendored SM12 FlashAttention 4 bundle to support preallocated output buffers for vLLM Inkling."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib.Path",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm (current_platform, FA4 dispatch source)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "A standalone Python script that patches vLLM's FlashAttention 4 dispatch code, targeting only Inkling's NVIDIA FA4 path. It uses AST-based validation and idempotent single-occurrence text replacement to inject a cached capability check (_use_sm12_paged_kv) that routes compute-capability 12.x devices to the vendored SM12 paged-KV kernel, guarded by a mod marker for traceability.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/patch_inkling.py",
          "importance_score": 0.85,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_use_sm12_paged_kv",
              "parameters": [],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "replace_once",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "old",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "new",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "label",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "validate_shape",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "patched_text",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "int",
              "visibility": ""
            }
          ],
          "name": "patch_inkling.py",
          "responsibilities": [
            "Detect SM12 (compute capability major 12) devices at runtime via a cached capability check",
            "Inject the dispatch patch into vLLM's FA4 helper code anchored at _get_score_mod",
            "Validate patched source with AST parsing before writing",
            "Ensure idempotent, single-occurrence replacements with labeled errors",
            "Provide a CLI main entry point with exit-code semantics"
          ],
          "source_summary": "The script defines MARKER, HELPER_ANCHOR, and HELPER_REPLACEMENT constants, plus helper functions: _use_sm12_paged_kv (checks current_platform.get_device_capability().major == 12), replace_once (idempotent text substitution with a label), validate_shape (AST-based shape validation of patched text), patched_text (applies the patch), and main (CLI entry returning an int exit code). It uses argparse, ast, sys, and pathlib.",
          "summary": "CLI patcher that modifies vLLM's NVIDIA FA4 dispatch source so SM12 devices use the vendored paged-KV kernel."
        },
        {
          "code_purpose": "command",
          "dependencies": [
            {
              "dependency_type": "include",
              "is_external": false,
              "line_number": null,
              "name": "vendor/inkling_sm120_fa4",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "adapter.py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "patch_inkling.py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm site-packages",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "A bash entry script with strict error handling (set -euo pipefail) that resolves the vLLM installation root from VLLM_SITE_PACKAGES or PYTHON_ROOT (defaulting to /usr/local/lib/python3.12/dist-packages). It copies vendor/inkling_sm120_fa4 into vllm/third_party/, installs adapter.py as vllm/third_party/inkling_sm120_fa4_adapter.py, and invokes patch_inkling.py to patch the FA4 dispatch, logging progress with a [inkling-sm12-paged-kv] prefix.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/run.sh",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "run.sh (bash script)",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "VLLM_SITE_PACKAGES",
                  "param_type": "env"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "PYTHON_ROOT",
                  "param_type": "env"
                }
              ],
              "return_type": "exit code",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Resolve the vLLM installation root from environment variables with a sensible default",
            "Vendor the SM12 FA4 kernel bundle into vllm/third_party/",
            "Install adapter.py into the vLLM tree as the FA4 adapter module",
            "Execute patch_inkling.py to patch Inkling's FA4 dispatch",
            "Fail fast on any error via set -euo pipefail"
          ],
          "source_summary": "The script defines PREFIX, MOD_DIR, PYTHON_ROOT, VLLM_ROOT, VENDOR_SOURCE/TARGET, ADAPTER_SOURCE/TARGET, and PATCHER variables, then performs directory copy, adapter installation, and patcher execution steps with echoed status messages.",
          "summary": "Installation shell script that copies the vendored SM12 FA4 bundle and adapter into the vLLM site-packages tree and runs the Inkling patcher."
        }
      ],
      "importance_score": 0.72,
      "key_files": [
        "patch_inkling.py",
        "adapter.py",
        "run.sh"
      ],
      "name": "inkling-sm12-paged-kv",
      "path": ".litho/tree/repo/mods/inkling-sm12-paged-kv",
      "purpose": "other",
      "subdirectory_count": 1,
      "summary": "This directory is a vLLM modification module ('inkling-sm12-paged-kv') that adapts a vendored SM12 (Blackwell) FlashAttention 4 kernel bundle to support preallocated output buffers ('out=') required by vLLM Inkling passes. run.sh orchestrates installation by copying the vendored kernel and adapter into the vLLM site-packages tree, while patch_inkling.py performs an AST-based source patch of vLLM's NVIDIA FA4 dispatch logic to route SM12 (compute capability major 12) devices to the paged-KV vendored kernel. adapter.py provides the runtime shim that exposes the internal forward entry point with preallocated-output support through the public flash_attn_varlen_func contract."
    },
    {
      "file_count": 47,
      "file_insights": [
        {
          "code_purpose": "doc",
          "dependencies": [],
          "detailed_description": "A simple text file listing the authors and their contact information.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/AUTHORS",
          "importance_score": 0.1,
          "interfaces": [],
          "name": "AUTHORS",
          "responsibilities": [
            "List project authors"
          ],
          "source_summary": "Contains the name and email of Tri Dao.",
          "summary": "Lists the authors of the project."
        },
        {
          "code_purpose": "doc",
          "dependencies": [],
          "detailed_description": "The license file for the project, specifying the BSD 3-Clause terms.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/LICENSE",
          "importance_score": 0.1,
          "interfaces": [],
          "name": "LICENSE",
          "responsibilities": [
            "Define licensing terms"
          ],
          "source_summary": "Contains the standard BSD 3-Clause License text with copyright notice.",
          "summary": "BSD 3-Clause License text."
        },
        {
          "code_purpose": "doc",
          "dependencies": [],
          "detailed_description": "A text file documenting the upstream repository, commit hash, and mechanical substitutions made during migration.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/UPSTREAM_COMMIT",
          "importance_score": 0.1,
          "interfaces": [],
          "name": "UPSTREAM_COMMIT",
          "responsibilities": [
            "Document upstream source and migration"
          ],
          "source_summary": "Lists the upstream repo, commit hash, and notes about CUTLASS DSL API migration.",
          "summary": "Records the upstream commit and migration details."
        },
        {
          "code_purpose": "module",
          "dependencies": [],
          "detailed_description": "Initializes the package and provides a docstring describing the bundled SM120 improvements. May import key submodules.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/__init__.py",
          "importance_score": 0.8,
          "interfaces": [],
          "name": "__init__.py",
          "responsibilities": [
            "Package initialization",
            "Expose public API"
          ],
          "source_summary": "Contains a docstring listing five upstream PRs targeting SM120 and likely imports core modules.",
          "summary": "Package initialization for the inkling_sm120_fa4 module."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides utility functions for computing shared memory layout atoms for different data types and dimensions, used in attention kernels.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/ampere_helpers.py",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "get_smem_layout_atom",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dtype",
                  "param_type": "Type[cutlass.Numeric]"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "k_dim",
                  "param_type": "int"
                }
              ],
              "return_type": "cute.ComposedLayout",
              "visibility": ""
            }
          ],
          "name": "ampere_helpers.py",
          "responsibilities": [
            "Compute shared memory layouts for Ampere"
          ],
          "source_summary": "Defines get_smem_layout_atom which returns a ComposedLayout based on dtype and k_dim.",
          "summary": "Helper functions for Ampere architecture shared memory layouts."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass._mlir.dialects.llvm",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides low-level barrier functions like ld_acquire and wait_eq using inline PTX assembly for GPU synchronization.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/barrier.py",
          "importance_score": 0.5,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "ld_acquire",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "lock_ptr",
                  "param_type": "cute.Pointer"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "loc",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ip",
                  "param_type": "Any"
                }
              ],
              "return_type": "cutlass.Int32",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "wait_eq",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "lock_ptr",
                  "param_type": "cute.Pointer"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "thread_idx",
                  "param_type": "int | Int32"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "flag_offset",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "val",
                  "param_type": "Int32"
                }
              ],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "barrier.py",
          "responsibilities": [
            "Provide synchronization primitives"
          ],
          "source_summary": "Defines ld_acquire and wait_eq functions using LLVM inline asm for acquire loads and waiting on flags.",
          "summary": "Implements memory barriers for synchronization."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cudnn",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides functions for computing reference attention, setting up cuDNN graphs for forward and backward passes, and calculating FLOPS.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/bench_utils.py",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "attention_ref",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "q",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "k",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "v",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "causal",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "cudnn_fwd_setup",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "q",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "k",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "v",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "causal",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "window_size_left",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "cudnn_bwd_setup",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "q",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "k",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "v",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "o",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "g",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "lse",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "causal",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "window_size_left",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "bench_utils.py",
          "responsibilities": [
            "Provide reference implementations",
            "Set up cuDNN benchmarks",
            "Calculate FLOPS"
          ],
          "source_summary": "Contains attention_ref, cudnn_fwd_setup, cudnn_bwd_setup, and flops calculation.",
          "summary": "Shared benchmark utilities including reference attention and cuDNN helpers."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides utilities to benchmark PyTorch functions, including memory usage and timing, with support for AMP.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/benchmark.py",
          "importance_score": 0.5,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "benchmark_forward",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "fn",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "inputs",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "repeats",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "desc",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "verbose",
                  "param_type": "bool"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "amp",
                  "param_type": "bool"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "amp_dtype",
                  "param_type": "torch.dtype"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kwinputs",
                  "param_type": "dict"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "benchmark.py",
          "responsibilities": [
            "Benchmark forward/backward passes",
            "Measure memory usage"
          ],
          "source_summary": "Defines benchmark_forward, benchmark_backward, benchmark_memory, etc.",
          "summary": "Benchmarking functions for forward and backward passes."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "A standalone benchmark for FP8 attention using CuTe DSL, comparing against PyTorch baseline and cuDNN.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/benchmark_flash_attention_fp8.py",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "argv",
                  "param_type": "Iterable[str] | None"
                }
              ],
              "return_type": "int",
              "visibility": ""
            }
          ],
          "name": "benchmark_flash_attention_fp8.py",
          "responsibilities": [
            "Benchmark FP8 attention",
            "Validate correctness"
          ],
          "source_summary": "Contains functions for parsing arguments, running benchmarks, and comparing outputs.",
          "summary": "Benchmark script for FP8 attention on SM100."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.tcgen05",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides extensive utilities for Blackwell-specific features like tcgen05 MMA, PTX descriptors, and memory operations.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/blackwell_helpers.py",
          "importance_score": 0.9,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_tcgen05_mma_kind",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "op",
                  "param_type": "cute.nvgpu.tcgen05.mma.MmaOp"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "i64_to_i32x2",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "i",
                  "param_type": "int"
                }
              ],
              "return_type": "Tuple[int, int]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "declare_ptx_idesc",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "op",
                  "param_type": "cute.nvgpu.tcgen05.mma.MmaOp"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "var_name",
                  "param_type": "str"
                }
              ],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "blackwell_helpers.py",
          "responsibilities": [
            "Provide Blackwell-specific helpers",
            "Support tcgen05 MMA",
            "Generate PTX descriptors"
          ],
          "source_summary": "Defines functions like _tcgen05_mma_kind, i64_to_i32x2, declare_ptx_idesc, and many others for Blackwell kernels.",
          "summary": "Helper functions for Blackwell (SM100/SM120) architecture."
        },
        {
          "code_purpose": "model",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides a BlockInfo dataclass for attention tiling and functions to compute min/max m-block indices for causal/local attention.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/block_info.py",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "get_m_block_min_max",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "seqlen_info",
                  "param_type": "SeqlenInfoQK"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n_block",
                  "param_type": "Int32"
                }
              ],
              "return_type": "Tuple[Int32, Int32]",
              "visibility": ""
            }
          ],
          "name": "block_info.py",
          "responsibilities": [
            "Define block tiling information",
            "Compute block ranges"
          ],
          "source_summary": "Contains BlockInfo class and get_m_block_min_max function.",
          "summary": "Defines BlockInfo dataclass and block range calculations."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Contains functions for producing and consuming block-sparse loads, used by CUTE DSL kernels for block-sparse attention.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/block_sparse_utils.py",
          "importance_score": 0.8,
          "interfaces": [],
          "name": "block_sparse_utils.py",
          "responsibilities": [
            "Provide block-sparse runtime utilities",
            "Support block-sparse loads"
          ],
          "source_summary": "Large module with 22 functions for block-sparse operations, including mask handling and load/store utilities.",
          "summary": "Runtime utilities for block-sparse attention kernels."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines data structures like BlockSparseTensors and functions for converting to CuTe tensors and checking block sparsity.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/block_sparsity.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "is_block_sparsity_enabled",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "tensors",
                  "param_type": "BlockSparseTensorsTorch"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "fast_sampling",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mask_mod",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "block_sparsity.py",
          "responsibilities": [
            "Define block-sparse data structures",
            "Convert to CuTe tensors"
          ],
          "source_summary": "Contains BlockSparseTensors, BlockSparseTensorsTorch, is_block_sparsity_enabled, and fast_sampling.",
          "summary": "Block-sparsity utilities for FlexAttention."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "tvm_ffi",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides a JITCache class and file locking for caching compiled kernels, with fingerprinting to invalidate on source changes.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/cache_utils.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "get_cache_path",
              "parameters": [],
              "return_type": "Path",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "JITCache",
              "parameters": [],
              "return_type": "JITCache",
              "visibility": ""
            }
          ],
          "name": "cache_utils.py",
          "responsibilities": [
            "Cache compiled kernels",
            "Manage file locks",
            "Compute source fingerprints"
          ],
          "source_summary": "Contains FileLock, JITCache, and functions for cache path and fingerprinting.",
          "summary": "Manages AOT compiled kernel cache."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements a kernel to compute block sparsity masks for FlexAttention, using CuTe DSL.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/compute_block_sparsity.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "BlockSparsityKernel",
              "parameters": [],
              "return_type": "BlockSparsityKernel",
              "visibility": ""
            }
          ],
          "name": "compute_block_sparsity.py",
          "responsibilities": [
            "Compute block sparsity masks"
          ],
          "source_summary": "Defines BlockSparsityKernel class and SharedStorage, with methods to compute sparsity.",
          "summary": "Kernel for computing block sparsity."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides functions for efficient data movement using TMA and bulk copy operations, including pipeline integration.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/copy_utils.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "load_s2r",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "src",
                  "param_type": "cute.Tensor"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "loc",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ip",
                  "param_type": "Any"
                }
              ],
              "return_type": "cute.Tensor",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "copy_bulk",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "src_idx",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dst_idx",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "new_kwargs",
                  "param_type": "dict"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "copy_utils.py",
          "responsibilities": [
            "Implement TMA copies",
            "Provide bulk copy utilities"
          ],
          "source_summary": "Contains load_s2r, copy_bulk, copy_tma, and tma_producer_copy_fn.",
          "summary": "Utilities for data copying (TMA, bulk copies)."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Patches CUTLASS DSL to use a system ptxas compiler, with environment variable configuration.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/cute_dsl_ptxas.py",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "patch",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "cute_dsl_ptxas.py",
          "responsibilities": [
            "Replace ptxas with system version",
            "Compile PTX"
          ],
          "source_summary": "Contains functions to compile PTX and patch the CUDA library loading.",
          "summary": "System ptxas replacement for CUTLASS DSL."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides functions for converting PyTorch tensors to CuTe tensors, getting device capacity, and dumping kernel attributes.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/cute_dsl_utils.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "to_cute_tensor",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "t",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "assumed_align",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "leading_dim",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "fully_dynamic",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "enable_tvm_ffi",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "cute_dsl_utils.py",
          "responsibilities": [
            "Convert tensors to CuTe",
            "Provide device info"
          ],
          "source_summary": "Contains to_cute_tensor, get_device_capacity, assume_strides_aligned, etc.",
          "summary": "Utilities for CuTe DSL tensor handling."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements FA2-style dropout using Philox PRNG, matching forward/backward tiles for consistent masks.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/dropout.py",
          "importance_score": 0.7,
          "interfaces": [],
          "name": "dropout.py",
          "responsibilities": [
            "Implement dropout for attention"
          ],
          "source_summary": "Contains functions for dropout mask generation and application.",
          "summary": "Dropout implementation for Flash Attention CuTe DSL kernels."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "logging",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides logging controlled by FA_LOG_LEVEL environment variable, with host and device logging.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/fa_logging.py",
          "importance_score": 0.4,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "fa_log",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "level",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "msg",
                  "param_type": "str"
                }
              ],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "fa_logging.py",
          "responsibilities": [
            "Provide logging utilities"
          ],
          "source_summary": "Contains fa_log, fa_printf, and log level configuration.",
          "summary": "Unified logging for FlashAttention."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides a fast count-leading-zeros (clz) implementation using CuTe.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/fast_math.py",
          "importance_score": 0.4,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "clz",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "x",
                  "param_type": "Int32"
                }
              ],
              "return_type": "Int32",
              "visibility": ""
            }
          ],
          "name": "fast_math.py",
          "responsibilities": [
            "Provide fast math primitives"
          ],
          "source_summary": "Defines clz function using a loop.",
          "summary": "Fast math functions for GPU kernels."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cuda.bindings.driver",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.cpasync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.warp",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "flash_bwd (self-module imports)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements the FlashAttentionBackwardSm80 kernel class adapted for SM120 Blackwell hardware, handling the main backward mainloop that computes dQ, dK, and dV gradients. It manages shared memory layouts (with separate or shared Q/V storage), software pipelining of Q and dO loads, and hooks for the dQ MMA operation. This is the computational heart of the backward pass in this vendored kernel package.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionBackwardSm80",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_setup_attributes",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get_tiled_mma",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get_shared_storage_cls",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "load_Q_next",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "load_dO_next",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "dQ_mma",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "hook_fn",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "advance_pipeline",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "pipeline_index",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "num_stages",
                  "param_type": "cutlass.Constexpr"
                }
              ],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "flash_bwd.py",
          "responsibilities": [
            "Compute attention backward gradients (dQ, dK, dV) via tiled MMA operations",
            "Manage shared memory storage layouts for Q, K, V, dO tiles",
            "Software-pipeline Q and dO loads across multiple stages",
            "Provide kernel capability checks and configuration via _setup_attributes and can_implement"
          ],
          "source_summary": "Defines the FlashAttentionBackwardSm80 class with 27 functions across 8 classes, including shared storage variants (SharedStorageSeparateQV, SharedStorageSharedQV), tiled MMA setup (_get_tiled_mma), next-tile loaders (load_Q_next, load_dO_next), the dQ MMA step with hook support, and a multi-stage pipeline advance routine. Imports cutlass, cute, cpasync/warp primitives, and CUDA driver bindings.",
          "summary": "Core FlashAttention backward-pass kernel for SM120, reimplemented from CUTLASS C++ (mainloop_bwd_sm80.hpp) into the Cute-DSL. It is the largest and most complex file in the directory, orchestrating the tiled MMA pipeline for computing attention gradients."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cuda.bindings.driver",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.utils.hopper_helpers",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.utils.blackwell_helpers",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "flash_bwd_postprocess (self-module imports)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements the FlashAttentionBackwardPostprocess kernel class, which performs the final processing stage of the backward pass on SM120 hardware. It validates inputs via can_implement (checking dtype, head_dim, tile_m, and thread count) and sets up its own tiled MMA configuration. It complements the main backward kernel by converting partial gradient accumulators into final results.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_postprocess.py",
          "importance_score": 0.75,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionBackwardPostprocess",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "can_implement",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dtype",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "head_dim",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "tile_m",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "num_threads",
                  "param_type": "Any"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get_tiled_mma",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_setup_attributes",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "flash_bwd_postprocess.py",
          "responsibilities": [
            "Finalize backward-pass gradient outputs after the mainloop",
            "Validate kernel applicability via dtype/head_dim/tile/thread checks",
            "Configure tiled MMA and shared memory attributes for the postprocess stage"
          ],
          "source_summary": "Defines FlashAttentionBackwardPostprocess with can_implement (dtype, head_dim, tile_m, num_threads -> bool), _get_tiled_mma, and _setup_attributes. Imports cutlass, cute, and Blackwell/Hopper helper utility modules (sm90_utils_basic, sm100_utils_ba...) to construct the postprocess kernel.",
          "summary": "Postprocess kernel for the FlashAttention backward pass, ported from flash_bwd_postprocess_kernel.h to Cute-DSL. It finalizes gradient outputs (e.g., dQ accumulation/normalization) after the main backward mainloop."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cuda.bindings.driver",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "flash_bwd_preprocess (self-module imports)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Implements the FlashAttentionBackwardPreprocess kernel class, computing the D term (D_i = (dO_i * O_i).sum(dim=-1)) used in the backward recurrence dS_ij = P_ij * (dP_ij - D_i). It also supports the differentiable-LSE variant D'_i = D_i - dLSE_i. Like its siblings, it exposes can_implement validation and tiled MMA setup for SM120.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_preprocess.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionBackwardPreprocess",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "can_implement",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dtype",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "head_dim",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "tile_m",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "num_threads",
                  "param_type": "Any"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get_tiled_mma",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_setup_attributes",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "flash_bwd_preprocess.py",
          "responsibilities": [
            "Compute the D statistic (rowsum of dO * O) needed by the backward mainloop",
            "Support LSE-differentiable adjustment (D' = D - dLSE)",
            "Validate kernel applicability via can_implement checks",
            "Configure tiled MMA and kernel attributes for SM120"
          ],
          "source_summary": "Defines FlashAttentionBackwardPreprocess with can_implement (dtype, head_dim, tile_m, num_threads -> bool), _get_tiled_mma, and _setup_attributes. The header comments explain the mathematical basis of the D statistic and its LSE-gradient adjustment; imports cutlass, cute, and CUDA driver bindings.",
          "summary": "Preprocess kernel for the FlashAttention backward pass, ported from flash_bwd_preprocess_kernel.h to Cute-DSL. It computes D_i = rowsum(dO_i * O_i), optionally adjusted for LSE gradients, which is required by the main backward kernel."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cuda.bindings.driver",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.cpasync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.tcgen05",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.utils.blackwell_helpers",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.pipeline",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file provides the core backward-pass kernel for FlashAttention on SM100 Blackwell hardware, defining the FlashAttentionBackwardSm100 class along with supporting SharedStorage classes for shared memory management. It handles the full backward attention computation including dQ, dK, dV gradient accumulation with pipelined async copies and Blackwell tcgen05 tensor core operations. Its internal helper functions (_setup_attributes, _get_tiled_mma, _setup_smem_layout) configure kernel tile shapes, MMA atoms, and shared memory layouts at compile time.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm100.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionBackwardSm100",
              "parameters": [],
              "return_type": "void",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_setup_attributes",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_get_tiled_mma",
              "parameters": [],
              "return_type": "cute.TiledMma",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_setup_smem_layout",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "SharedStorage",
              "parameters": [],
              "return_type": "void",
              "visibility": ""
            }
          ],
          "name": "flash_bwd_sm100.py",
          "responsibilities": [
            "Compute the FlashAttention backward pass (dQ/dK/dV gradients) on SM100 Blackwell GPUs",
            "Configure tiled MMA atoms and shared memory layouts for tcgen05 tensor cores",
            "Manage shared memory storage and pipelining via SharedStorage classes",
            "Handle paged KV cache indexing with fast divmod arithmetic"
          ],
          "source_summary": "The source imports CUDA driver bindings, CUTLASS/CuTe primitives (Float32, Int32, FastDivmodDivisor, LayoutEnum), cpasync and tcgen05 NVIDIA GPU modules, and blackwell_helpers utilities. It defines FlashAttentionBackwardSm100 with 27 functions across 3 classes, including SharedStorage classes for smem allocation and setup methods for tiled MMA and smem layouts, implementing a 4000+ line high-complexity (337) attention backward kernel.",
          "summary": "Implements the FlashAttention backward pass kernel for SM100 (Blackwell datacenter) GPUs using the CUTLASS CuTe DSL with tcgen05 MMA instructions. It is the largest and most complex file in the directory, orchestrating tiled MMA operations, shared memory pipelines, and gradient accumulation."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": false,
              "line_number": null,
              "name": ".flash_bwd.FlashAttentionBackwardSm80",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This module provides FlashAttentionBackwardSm120, which adapts the attention backward kernel to SM120 Blackwell consumer GPUs. Since SM120 uses the same SM80-era mma.sync.aligned.m16n8k16 instructions but has smaller shared memory (99 KB vs 163 KB), the class subclasses FlashAttentionBackwardSm80 and only overrides the SMEM capacity validation logic. This keeps the SM120 implementation minimal while inheriting the full backward-pass computation.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm120.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionBackwardSm120",
              "parameters": [],
              "return_type": "void",
              "visibility": ""
            }
          ],
          "name": "flash_bwd_sm120.py",
          "responsibilities": [
            "Provide the FlashAttention backward kernel for SM120 Blackwell GeForce hardware",
            "Override SMEM capacity checks for the reduced 99 KB shared memory budget",
            "Reuse SM80-era mma.sync MMA instructions via inheritance from FlashAttentionBackwardSm80"
          ],
          "source_summary": "The file imports cutlass and cutlass.utils, then imports FlashAttentionBackwardSm80 from the sibling flash_bwd module. It defines a single FlashAttentionBackwardSm120 class that overrides the shared memory capacity check to account for SM120's 99 KB smem limit, delegating all kernel computation to the SM80 parent implementation.",
          "summary": "Defines the SM120 (Blackwell GeForce / DGX Spark) backward pass by subclassing the SM80 backward implementation and overriding the shared memory capacity check for the reduced 99 KB smem budget. It is a small adapter module that reuses existing SM80-era mma.sync kernels on SM120 hardware."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.copy_utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.layout_utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.sm90_utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cuda.bindings.driver",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.utils.hopper_helpers",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file is the forward-pass core of the attention library, providing FlashAttentionForwardBase with shared setup logic (attributes, MMA tiling, shared storage layouts, pipeline advancement) and FlashAttentionForwardSm80 as a concrete architecture specialization. It handles tiled MMA construction, shared memory layout atoms, and shared storage classes (SharedStorageQKV, SharedStorageSharedQV) needed for high-throughput attention computation. It is one of the largest and most complex files in the directory (1791 lines, complexity 208).",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionForwardBase",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionForwardSm80",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_setup_attributes",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_get_smem_layout_atom",
              "parameters": [],
              "return_type": "cute.Tensor layout",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_get_tiled_mma",
              "parameters": [],
              "return_type": "tiled MMA object",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_get_shared_storage_cls",
              "parameters": [],
              "return_type": "Type",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "advance_pipeline",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "pipeline_index",
                  "param_type": "Any"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "SharedStorageQKV",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "SharedStorageSharedQV",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "flash_fwd.py",
          "responsibilities": [
            "Implement FlashAttention forward kernel logic for SM80/SM90 architectures",
            "Manage shared memory layouts (SharedStorageQKV, SharedStorageSharedQV) and smem layout atoms",
            "Configure tiled MMA operations for tensor-core attention computation",
            "Advance the software pipeline across k-block iterations",
            "Provide a reusable base class for architecture-specific forward kernel variants"
          ],
          "source_summary": "The file re-implements flash-attention's hopper flash_fwd_kernel_sm80.h and flash_fwd_kernel_sm90.h from CUTLASS C++ into the CuTe-DSL, building on the NVIDIA ampere flash_attention_v2 Python example. It defines FlashAttentionForwardBase and FlashAttentionForwardSm80 classes with methods for attribute setup, shared-memory layout atoms, tiled MMA selection, shared storage classes, and software pipeline advancement. It imports heavily from cutlass, cutlass.cute, quack copy/layout utilities, and cuda.bindings.driver.",
          "summary": "Implements FlashAttention forward kernels in CuTe-DSL, including a base class plus SM80 and SM90 architecture-specific variants, reinterpreted from the official flash-attention Hopper C++ kernels."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.cpasync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.warpgroup",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.utils.hopper_helpers",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.copy_utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.layout_utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.sm90_utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cuda.bindings.driver",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file provides FlashAttentionBackwardSm90, the Hopper-architecture backward kernel for flash attention, handling the complex gradient computation across Q, K, V tensors. It includes utilities for QKV transposition, per-row statistics extraction from the S-matrix with warp shuffles, tiled MMA configuration, and shared storage management. At 1962 lines with complexity 183, it is the largest file in the directory and critical for training workloads.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_bwd_sm90.py",
          "importance_score": 0.93,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionBackwardSm90",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_setup_attributes",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_get_tiled_mma",
              "parameters": [],
              "return_type": "tiled MMA object",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_get_shared_storage_cls",
              "parameters": [],
              "return_type": "Type",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "SharedStorageQKV",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_qkv_transpose",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "t",
                  "param_type": "Any"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get_stat",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "tSrS",
                  "param_type": "cute.Tensor"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "row",
                  "param_type": "Int32"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "lane",
                  "param_type": "Int32"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "shuffle",
                  "param_type": "bool"
                }
              ],
              "return_type": "Float32",
              "visibility": ""
            }
          ],
          "name": "flash_bwd_sm90.py",
          "responsibilities": [
            "Implement the SM90 backward attention kernel computing dQ/dK/dV gradients",
            "Recompute softmax statistics from saved LSE for gradient scaling",
            "Transpose QKV tensors into layouts suitable for gradient MMA",
            "Manage shared storage and tiled MMA for the backward pass",
            "Perform warp-level shuffle-based statistic reduction (_get_stat)"
          ],
          "source_summary": "The file defines FlashAttentionBackwardSm90 with 31 functions across 5 classes, including _setup_attributes for kernel configuration, _get_tiled_mma for tensor-core MMA setup, _get_shared_storage_cls and SharedStorageQKV for shared memory definitions, _qkv_transpose for transposing Q/K/V tensors, and _get_stat which extracts a Float32 statistic from a CuTe tensor at a given row/lane with optional warp shuffle. It imports from cutlass, cutlass.cute.nvgpu (cpasync, warpgroup), quack utilities, and cuda.bindings.driver.",
          "summary": "Implements the FlashAttention backward (gradient) kernel for SM90 Hopper GPUs in CuTe-DSL, computing dQ, dK, and dV gradients with statistics-based softmax recomputation."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.cpasync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cuda.bindings.driver",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file defines FlashAttentionForwardCombine, a post-processing kernel that combines partial outputs from the split-K (varlen/split) forward attention pass using log-sum-exp weighted merging. It is a focused, self-contained kernel (698 lines, 2 classes) with its own attribute setup and shared storage definition, using cpasync for data movement. It complements flash_fwd.py by handling the reduction step required when the forward pass splits work across KV chunks.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_combine.py",
          "importance_score": 0.82,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionForwardCombine",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_setup_attributes",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "SharedStorage",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "flash_fwd_combine.py",
          "responsibilities": [
            "Merge split-K partial attention outputs into final results",
            "Apply log-sum-exp-based numerical correction when combining partials",
            "Manage shared memory staging (SharedStorage) for the combine kernel",
            "Configure kernel attributes and async copy (cpasync) data movement"
          ],
          "source_summary": "The file re-implements flash-attention's hopper flash_fwd_combine_kernel.h from CUTLASS C++ into CuTe-DSL. It defines FlashAttentionForwardCombine with _setup_attributes for kernel configuration and a SharedStorage class for shared memory, importing Float32/Int32/Boolean types and cpasync primitives from cutlass.cute. The kernel reads partial outputs and LSE statistics, applies numerically-stable exponential reweighting, and writes the combined final attention output.",
          "summary": "Implements the split-K combine kernel for FlashAttention forward, merging numerically-corrected partial attention outputs produced by the split forward kernel into final results."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch.utils.benchmark",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cuda.bindings.driver",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.pipeline",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.cpasync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.tcgen05",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.runtime.from_dlpack",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.utils.blackwell_helpers",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file is the centerpiece of the vendor package, defining the FlashAttentionMLAForwardSm100 class that orchestrates the entire MLA forward pass on Blackwell hardware. It handles shared-memory layout construction (SharedStorage), mbarrier-based multi-stage pipelines, TMA descriptor creation, and hdim splitting for the latent attention decomposition. It also includes benchmarking utilities (timeit) for kernel performance evaluation.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_mla_sm100.py",
          "importance_score": 0.92,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionMLAForwardSm100",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get_shared_storage_cls",
              "parameters": [],
              "return_type": "type",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "smem_struct_align",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dtype",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "staged_layout",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "mbar_struct",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "num_stages",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "split_hdimv",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "m",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dim",
                  "param_type": "int"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "make_tma",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "make_fn",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mX",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "smem_layout",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mma_tiler",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "tiled_mma",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "make_pipeline",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mbar_ptr",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "num_stages",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "producer",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "consumer",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "tx_count",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "timeit",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "fn",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "args",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kwargs",
                  "param_type": "dict"
                }
              ],
              "return_type": "float",
              "visibility": ""
            }
          ],
          "name": "flash_fwd_mla_sm100.py",
          "responsibilities": [
            "Implement the MLA forward attention kernel for SM100/SM120 Blackwell GPUs using tcgen05 MMA instructions",
            "Manage shared memory layouts, mbarriers, and multi-stage producer/consumer pipelines",
            "Create and configure TMA (Tensor Memory Accelerator) descriptors for efficient global-to-shared memory transfers",
            "Split the value head dimension (hdimv) to fit register/smem constraints of the latent attention decomposition",
            "Provide kernel benchmarking utilities for performance measurement"
          ],
          "source_summary": "The source imports torch, cuda.bindings.driver, and the CUTLASS/CuTe DSL (cutlass.cute, cutlass.pipeline, cpasync, tcgen05, blackwell_helpers) to build a warp-specialized MLA attention kernel. Key components include FlashAttentionMLAForwardSm100 (main kernel class), helpers for shared storage classes, mbarrier structs, TMA creation, pipeline construction, hdimv splitting, and a timeit benchmarking function using torch.utils.benchmark.",
          "summary": "Implements the FlashAttention MLA forward kernel for NVIDIA Blackwell SM100 GPUs using the CUTLASS CuTe Python DSL, with warp-specialized producer/consumer pipelines and tcgen05 tensor-core MMA. It serves as the core compute kernel for Multi-head Latent Attention inference with paged KV caches."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass (CuTe DSL)",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "math",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "flash_fwd_sm100 (self-referential kernel imports)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This is the largest and most complex file in the directory (3149 lines, complexity 397), providing the SM100-specific FlashAttention kernel. It is based on the CUTLASS Blackwell FMHA example and handles the full pipeline of Q/K/V loading, tiled MMA computation, softmax rescaling, and output epilogue. It defines the core kernel classes, shared storage layouts, and KV head indexing helpers used by the paged-KV attention runtime.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm100.py",
          "importance_score": 0.92,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionForwardSm100",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "DescaleTensors",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "SharedStorage",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_kv_head_idx",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "head_idx",
                  "param_type": "Int32"
                }
              ],
              "return_type": "Int32",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "offset_kv_smem",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "sX",
                  "param_type": "cute.Tensor"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "stage",
                  "param_type": "Int32"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "phase",
                  "param_type": "Int32"
                }
              ],
              "return_type": "void",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "__new_from_mlir_values__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "values",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "flash_fwd_sm100.py",
          "responsibilities": [
            "Implement the SM100 FlashAttention forward kernel with TMA and warp specialization",
            "Support MHA, GQA, and MQA attention variants with causal/noncausal masking",
            "Handle variable-length sequences, sliding window attention, and split-KV scheduling",
            "Manage shared memory layouts and multi-stage KV pipelines via mbarrier synchronization",
            "Provide KV head indexing and SMEM offset computation for paged KV caches"
          ],
          "source_summary": "The file declares supported features (BF16/FP16 dtypes, causal and noncausal attention, MHA/GQA/MQA, head dims 64/96/128/(192,128), varlen, sliding window, split-KV) and notes pending work such as non-128 page sizes and additional head dims. It defines FlashAttentionForwardSm100 as the main kernel class, a SharedStorage class for SMEM layout, DescaleTensors for scaling factors, and helpers like _kv_head_idx and offset_kv_smem for KV paging and pipeline stage addressing.",
          "summary": "Implements the FlashAttention forward pass kernel for SM100 (Blackwell datacenter) GPUs using CuTe DSL, supporting BF16/FP16, causal/noncausal, MHA/GQA/MQA, varlen, sliding window, and split-KV configurations."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.base_dsl.arch.Arch",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This small adapter file (76 lines) provides the CpAsync-based SM120 forward pass. Because SM120 uses the same SM80-era mma.sync.aligned.m16n8k16 MMA instructions, it reuses FlashAttentionForwardSm80 and only overrides the SMEM capacity validation to reflect the 99 KB shared memory limit. It acts as the base class that flash_fwd_sm120_tma.py builds upon for the TMA-enabled variant.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm120.py",
          "importance_score": 0.78,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionForwardSm120",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "__init__",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "args",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kwargs",
                  "param_type": "dict"
                }
              ],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "flash_fwd_sm120.py",
          "responsibilities": [
            "Provide the SM120 forward pass by subclassing the SM80 kernel implementation",
            "Override shared memory capacity checks for SM120's 99 KB SMEM limit",
            "Serve as the CpAsync-based base class for the SM120 TMA variant"
          ],
          "source_summary": "The file imports cutlass, cutlass.utils, and Arch from the CuTe DSL base, then defines FlashAttentionForwardSm120 as a subclass of the SM80 forward kernel with a custom __init__ accepting generic args/kwargs. Its primary logic is the adjusted SMEM capacity check distinguishing SM120's 99 KB from SM80's 163 KB shared memory.",
          "summary": "Defines the SM120 (Blackwell GeForce/DGX Spark) FlashAttention forward pass by subclassing the SM80 kernel and overriding the shared memory capacity check for the smaller 99 KB SMEM."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass (CuTe DSL)",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "flash_fwd_sm120 (SM120 CpAsync base variant)",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file (1019 lines, 12 classes) provides the higher-performance SM120 variant using Tensor Memory Accelerator (TMA) loads and warp specialization with one DMA warp and N MMA warps. It defines TMA-specific shared memory layout atoms, tiled MMA construction, pipeline setup with PipelineTmaAsync, and a dedicated SharedStorage class. It complements flash_fwd_sm120.py by offering SM80-compatible kernel semantics with Blackwell-era data movement efficiency.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm120_tma.py",
          "importance_score": 0.85,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionForwardSm120Tma",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "get_smem_layout_atom_tma",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dtype",
                  "param_type": "Type[cutlass.Numeric]"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "k_dim",
                  "param_type": "int"
                }
              ],
              "return_type": "cute.ComposedLayout",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_get_smem_layout_atom",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_get_tiled_mma",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_get_shared_storage_cls",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "_setup_attributes_tma",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "SharedStorage",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "flash_fwd_sm120_tma.py",
          "responsibilities": [
            "Implement the TMA-based SM120 FlashAttention forward kernel",
            "Set up warp specialization with a DMA warp and MMA compute warps",
            "Build TMA shared memory layout atoms and tiled MMA abstractions",
            "Manage KV double-buffering via PipelineTmaAsync and mbarrier synchronization",
            "Define SM120-specific shared storage layout classes"
          ],
          "source_summary": "The module documents its key differences from the CpAsync SM120 kernel: TMA (cp.async.bulk) for Q/K/V global-to-shared transfers, warp specialization with a DMA warp plus MMA warps, and PipelineTmaAsync with mbarrier synchronization for KV double-buffering. Main components include get_smem_layout_atom_tma for composing SMEM layouts from dtype and head dimension, FlashAttentionForwardSm120Tma as the kernel class with _get_smem_layout_atom, _get_tiled_mma, _get_shared_storage_cls, and _setup_attributes_tma hooks, plus a SharedStorage class.",
          "summary": "Implements a TMA-based, warp-specialized FlashAttention forward pass for SM120 GPUs, adding cp.async.bulk loads and mbarrier-synchronized KV double buffering on top of the CpAsync SM120 variant."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cuda.bindings.driver",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.cpasync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.warpgroup",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file is the SM90 (Hopper) forward kernel extracted from flash_fwd.py. It defines the FlashAttentionForwardSm90 class along with helper functions for building shared-memory layout atoms, tiled MMA operations, shared storage classes, and warp-scheduler barrier synchronization primitives. It is the compute heart of the forward pass on Hopper hardware.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/flash_fwd_sm90.py",
          "importance_score": 0.9,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttentionForwardSm90",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get_smem_layout_atom",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get_tiled_mma",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_get_shared_storage_cls",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "warp_scheduler_barrier_sync",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "flash_fwd_sm90.py",
          "responsibilities": [
            "Implement the SM90 FlashAttention forward kernel",
            "Construct shared-memory layouts and tiled MMA atoms",
            "Manage warp-scheduler barriers and synchronization",
            "Define shared storage classes for QKV and shared Q/V variants"
          ],
          "source_summary": "The source imports CUDA driver bindings, CUTLASS, and Cute-DSL primitives (cpasync, warpgroup) and defines FlashAttentionForwardSm90 with 40 functions across 9 classes at high complexity (177). Key helpers include _get_smem_layout_atom, _get_tiled_mma, _get_shared_storage_cls (with SharedStorageQKV and SharedStorageSharedQV variants), mma_init, and warp_scheduler_barrier_sync/arrive for producer-consumer synchronization.",
          "summary": "Implements the FlashAttention forward pass kernel for Hopper (SM90) GPUs using the CUTLASS Cute-DSL, including shared-memory layout construction, tiled MMA setup, and warp-scheduler synchronization."
        },
        {
          "code_purpose": "api",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "torch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cuda.bindings.driver",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.compile_utils",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This is the largest and most central file (2417 lines, complexity 328), consolidating the user-facing interface for both forward and backward attention on Hopper and Blackwell. It parses device architecture, validates head dims and tensors, computes tile sizes and split-KV heuristics, resolves causal/local window parameters, and wraps kernels in torch.autograd.Function classes for dense and varlen inputs.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/interface.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_parse_arch_str",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "arch_str",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_validate_head_dims",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "head_dim",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "head_dim_v",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "compute_capability",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "alignment",
                  "param_type": "int"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_tile_size_fwd_sm90",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "head_dim",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "head_dim_v",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "is_causal",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "is_local",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "sparse_block_size_q",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "num_splits_heuristic",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "total_mblocks",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "num_SMs",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "num_n_blocks",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "max_splits",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttnFunc",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "FlashAttnVarlenFunc",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "interface.py",
          "responsibilities": [
            "Expose the public FlashAttention API to PyTorch users",
            "Validate inputs (shapes, dtypes, devices, head dims)",
            "Compute tile sizes and split-KV heuristics per architecture",
            "Provide autograd Function wrappers with backward support for dense and varlen inputs"
          ],
          "source_summary": "The source imports torch, CUDA driver bindings, CUTLASS/Cute-DSL, and quack compile utilities. It defines _parse_arch_str/_get_device_arch for capability detection, _validate_head_dims and _validate_tensor for input checks, _tile_size_fwd_sm90/_tile_size_bwd_sm90 heuristics, num_splits_heuristic for split-KV scheduling, make_fake_bwd_tensors, and the FlashAttnFunc/FlashAttnVarlenFunc autograd classes each with a backward method.",
          "summary": "Provides the top-level PyTorch-facing API for FlashAttention, including FwdConfig/BwdConfig dataclasses, tile-size heuristics, input validation, and autograd Function classes (FlashAttnFunc, FlashAttnVarlenFunc) with backward passes."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.layout_utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": ".utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": false,
              "line_number": null,
              "name": ".seqlen_info.SeqlenInfoQK",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file centralizes masking logic for the attention kernels. It provides functions to build 32-bit R2P bitmasks for column limits (below/above) and maps rows to R2P indices, then wraps them in an AttentionMask dataclass that exposes seqlen_q/seqlen_k accessors and a mask_gen_fn factory producing per-chunk masks for various mask types.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/mask.py",
          "importance_score": 0.75,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "r2p_bitmask_below",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "limit",
                  "param_type": "Int32"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "s",
                  "param_type": "int"
                }
              ],
              "return_type": "Uint32",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "r2p_bitmask_above",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "limit",
                  "param_type": "Int32"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "s",
                  "param_type": "int"
                }
              ],
              "return_type": "Uint32",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "sm90_col_to_r2p_idx",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "col_limit",
                  "param_type": "Int32"
                }
              ],
              "return_type": "Int32",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "AttentionMask",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "mask_gen_fn",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "s",
                  "param_type": "int"
                }
              ],
              "return_type": "Uint32",
              "visibility": ""
            }
          ],
          "name": "mask.py",
          "responsibilities": [
            "Generate R2P bitmasks for causal/local masking",
            "Map row/column indices to R2P chunk indices",
            "Provide AttentionMask abstraction with per-variant mask generation",
            "Consolidate seqlen accessors used by kernels"
          ],
          "source_summary": "The source defines MaskGenFn as a type alias for mask-generation callables and MASK_R2P_CHUNK_SIZE=32. Functions r2p_bitmask_below/above produce 32-bit bitmasks honoring a column limit, sm90_col_to_r2p_idx and row_to_r2p_idx map indices, and the AttentionMask class (723 lines, complexity 146) provides multiple mask_gen_fn implementations for different masking semantics.",
          "summary": "Implements attention mask generation for the kernels, including R2P (register-to-permuted) bitmask helpers and an AttentionMask class supporting causal, local/sliding-window, and arbitrary mask_mod functions."
        },
        {
          "code_purpose": "types",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file translates the CUTLASS C++ headers mma_sm100_desc.hpp and mma_traits_sm100.hpp into Python for Blackwell (SM100) tensor-core usage. It defines IntEnum classes whose values must exactly match hardware encodings, plus conversion functions from CUTLASS types to UMMA/C formats and descriptor construction from layouts, swizzles, and pointers.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/mma_sm100_desc.py",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "to_UMMA_format",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cutlass_type",
                  "param_type": "Any"
                }
              ],
              "return_type": "int",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "to_C_format",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cutlass_type",
                  "param_type": "Any"
                }
              ],
              "return_type": "int",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "mma_op_to_idesc",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "op",
                  "param_type": "cute.nvgpu.tcgen05.mma.MmaOp"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "make_smem_desc_base",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "layout",
                  "param_type": "cute.Layout"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "swizzle",
                  "param_type": "cute.Swizzle"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "major",
                  "param_type": "Major"
                }
              ],
              "return_type": "int",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "smem_desc_base_from_tensor",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "sA",
                  "param_type": "cute.Tensor"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "major",
                  "param_type": "Major"
                }
              ],
              "return_type": "int",
              "visibility": ""
            }
          ],
          "name": "mma_sm100_desc.py",
          "responsibilities": [
            "Encode SM100 MMA instruction descriptors in Python",
            "Map CUTLASS dtypes to hardware UMMA/C formats",
            "Derive layout types from swizzle patterns",
            "Build shared-memory descriptor base addresses from tensors"
          ],
          "source_summary": "The source defines enums Major, ScaleIn, Saturate, CFormat, F16F32Format, S8Format, MXF8F6F4Format, MaxShift, and LayoutType, with conversion helpers to_UMMA_format and to_C_format. It also provides mma_op_to_idesc to derive instruction descriptors from MmaOp objects, and make_smem_desc_base/_start_addr/smem_desc_base_from_tensor to build shared-memory descriptors from layouts and tensors.",
          "summary": "A Python port of CUTLASS C++ SM100 MMA descriptor encodings, providing hardware-matching enumerations (formats, major modes, swizzle layout types) and functions to build shared-memory MMA descriptors from tensors."
        },
        {
          "code_purpose": "types",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "enum",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "A small (56-line) enum-only module that assigns unique named-barrier IDs used by kernel producer/consumer synchronization. Separate enums exist for forward (Epilogue, WarpScheduler WGs, PFull/PEmpty), forward SM100, backward, backward SM100, and a forward SM100 MLA 2-CTA configuration.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/named_barrier.py",
          "importance_score": 0.45,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "NamedBarrierFwd",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "NamedBarrierFwdSm100",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "NamedBarrierBwd",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "NamedBarrierBwdSm100",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "NamedBarrierFwdSm100_MLA2CTA",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "named_barrier.py",
          "responsibilities": [
            "Assign unique named-barrier IDs for kernel synchronization",
            "Provide per-architecture (SM90/SM100) and per-direction (fwd/bwd) barrier sets",
            "Support MLA 2-CTA barrier configuration"
          ],
          "source_summary": "The source imports enum and defines five IntEnum classes: NamedBarrierFwd (Epilogue, WarpSchedulerWG1-3, PFull, PEmpty), NamedBarrierFwdSm100, NamedBarrierBwd, NamedBarrierBwdSm100, and NamedBarrierFwdSm100_MLA2CTA, all starting from auto() since barrier 0 is reserved for __syncthreads().",
          "summary": "Defines named hardware barrier ID enumerations for forward and backward kernels on SM90 and SM100, plus a special MLA 2-CTA variant."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.cpasync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.layout_utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": ".utils",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This module enables efficient handling of grouped-query attention by reshaping tensors so that the extra query heads per KV head are folded into the seqlen mode, avoiding separate head-dim iteration in kernels. pack_gqa_layout and unpack_gqa_layout perform the reshaping for a tensor with the head dimension at a given mode, and the PackGQA dataclass encapsulates the runtime logic.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/pack_gqa.py",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "pack_gqa_layout",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "T",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "qhead_per_kvhead",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "nheads_kv",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "head_idx",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "unpack_gqa_layout",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "T",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "qhead_per_kvhead",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "head_idx",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "PackGQA",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "pack_gqa.py",
          "responsibilities": [
            "Fold GQA query heads into the seqlen dimension for kernel efficiency",
            "Provide inverse unpacking of packed layouts",
            "Encapsulate GQA packing logic in the PackGQA class"
          ],
          "source_summary": "The source defines pack_gqa_layout(T, qhead_per_kvhead, nheads_kv, head_idx) which folds qhead_per_kvhead into mode 0 while preserving preceding modes, and unpack_gqa_layout for the inverse. The PackGQA class (263 lines) uses CUTLASS/Cute-DSL and cpasync primitives, leveraging quack layout_utils and local utils.",
          "summary": "Implements GQA (grouped-query attention) packing utilities that fold qhead_per_kvhead into the sequence-length dimension of tensors, with pack/unpack layout functions and a PackGQA class."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.cpasync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": true,
              "line_number": null,
              "name": "quack.cute_dsl_utils.ParamsBase",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": ".utils",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file is central to the paged-KV feature of the project (the parent mod is named inkling-sm12-paged-kv). PagedKVManager holds the page table and paged K/V tensors plus a fast divmod divisor for page-size arithmetic, and provides load_page_table, compute_X_ptr (for K or V), and load_KV methods used by the attention kernels to fetch KV blocks from non-contiguous paged memory.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/paged_kv.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "PagedKVManager",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "load_page_table",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n_block",
                  "param_type": "Int32"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "compute_X_ptr",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "K_or_V",
                  "param_type": "str"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "load_KV",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n_block",
                  "param_type": "Int32"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "sX",
                  "param_type": "cute.Tensor"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "K_or_V",
                  "param_type": "str"
                }
              ],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "paged_kv.py",
          "responsibilities": [
            "Manage paged KV-cache metadata and page tables",
            "Compute K/V pointers for arbitrary pages",
            "Load KV tiles from paged memory for a given n_block",
            "Provide fast page/offset divmod arithmetic"
          ],
          "source_summary": "The source defines a @dataclass PagedKVManager(ParamsBase) with fields mPageTable, mK_paged, mV_paged, thread_idx, page_size_divmod (FastDivmodDivisor), and seqlen info. Its methods load_page_table(n_block), compute_X_ptr(K_or_V), and load_KV(n_block, sX, K_or_V) coordinate gathering KV tiles from paged (block-table-indexed) global memory using cpasync.",
          "summary": "Implements the PagedKVManager dataclass that manages paged KV-cache loading, including page-table loads, pointer computation for K/V, and block-wise KV tile loading."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.pipeline.PipelineState",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.pipeline.PipelineAsync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.pipeline.PipelineCpAsync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.pipeline.NamedBarrier",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file implements the software pipelining infrastructure used by the attention kernels to overlap global-memory loads with compute. It wraps CUTLASS's original pipeline classes (PipelineAsync, PipelineCpAsync, etc.) with an _override_create/create factory pattern allowing subclass behavior injection, and defines PipelineStateSimple with stage index/phase tracking, advance, clone, and MLIR value extraction hooks for the DSL.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/pipeline.py",
          "importance_score": 0.75,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "create",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "args",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kwargs",
                  "param_type": "dict"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_make_state",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "index",
                  "param_type": "Int32"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "phase",
                  "param_type": "Int32"
                }
              ],
              "return_type": "PipelineState",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "PipelineStateSimple",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "advance",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "clone",
              "parameters": [],
              "return_type": "PipelineStateSimple",
              "visibility": ""
            }
          ],
          "name": "pipeline.py",
          "responsibilities": [
            "Abstract multi-stage producer/consumer synchronization for kernels",
            "Track pipeline state (stage index and phase) with advance/clone",
            "Adapt CUTLASS pipeline classes via a create/override factory",
            "Integrate pipeline state with the Cute-DSL MLIR value system"
          ],
          "source_summary": "The source imports from cutlass.pipeline (PipelineState, PipelineUserType, NamedBarrier, PipelineAsync, PipelineCpAsync) and defines _override_create and create classmethods for subclassing. PipelineStateSimple (13 classes, 26 functions total) tracks stages, index, and phase with _make_state, advance, clone, and __extract_mlir_values__/__new_from_mlir_values__ DSL integration methods.",
          "summary": "Provides multi-stage producer-consumer pipeline abstractions (PipelineState, async and cpasync pipelines) built on CUTLASS pipeline primitives, with a factory mechanism for subclass customization."
        },
        {
          "code_purpose": "model",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": true,
              "line_number": null,
              "name": "quack.copy_utils",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This module defines SeqlenInfo, SeqlenInfoQK, and SeqlenInfoQKNewK dataclasses that bundle all sequence-length-related data (offsets, padded offsets, seqlen_q, seqlen_k) so kernels compute n_block_min/max and other bounds from a single gather at tile start. SeqlenInfoQKNewK additionally supports the case where the KV length changes (e.g., KV-cache append during decode).",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/seqlen_info.py",
          "importance_score": 0.65,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "SeqlenInfo",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "SeqlenInfoQK",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "SeqlenInfoQKNewK",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "seqlen_info.py",
          "responsibilities": [
            "Bundle sequence-length metadata for Q and K/V into one object",
            "Support varlen and padded-offset scenarios",
            "Support new-K (KV append) scenarios via SeqlenInfoQKNewK",
            "Minimize per-tile global-memory reads of length info"
          ],
          "source_summary": "The source defines @dataclass(frozen=True) SeqlenInfo with offset and offset_padded fields, plus derived classes SeqlenInfoQK and SeqlenInfoQKNewK (287 lines, 3 classes). It uses quack copy_utils and CUTLASS Int32/const_expr, and its docstring explains the design goal of doing all gmem reads once at the beginning of each tile.",
          "summary": "Consolidates sequence-length information (offsets, lengths for Q/K/V, varlen handling) into frozen dataclasses read once per tile to avoid repeated global-memory reads in kernels."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "math",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "A configuration exploration script that enumerates tile sizes, swap modes, atom layouts, and staging options for SM90 (H100) attention kernels. It validates GMMA divisibility, register budgets, and shared memory budgets, and prints feasible forward/backward configs.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/sm90_config_search.py",
          "importance_score": 0.45,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_divisors",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "n",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_acc_regs",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "M",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "N",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "num_wg",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_check_mma",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "M",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "N",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "num_wg",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "atom_layout_m",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "swap_AB",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_mma_traffic",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "M_eff",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "N_eff",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "K_red",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "num_wg",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "wg_n",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "is_rs",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "print_fwd_configs",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "configs",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "max_results",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "print_bwd_configs",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "configs",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "max_results",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "sm90_config_search.py",
          "responsibilities": [
            "Enumerate candidate tile/atom layouts for attention kernels",
            "Validate GMMA divisibility and register/shared-memory budgets",
            "Model MMA traffic for swap-AB and register-SRAM staging variants",
            "Print feasible forward and backward configurations"
          ],
          "source_summary": "Defines helper functions _divisors, _acc_regs, _check_mma, _mma_traffic, _wg_n, and _check_fwd_config to model hardware constraints, plus print_bwd_configs and print_fwd_configs to report results. Invoked via CLI with --headdim, --mode, and --tile-n arguments.",
          "summary": "Standalone search tool that enumerates feasible SM90 forward/backward attention kernel configurations for given head dimensions."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.layout_utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.cute_dsl_utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": false,
              "line_number": null,
              "name": "seqlen_info",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Core numerical component of the attention kernel implementing the online (streaming) softmax algorithm used in FlashAttention. Provides dataclass-based parameter objects and device-side methods for computing row maxima, updating running statistics, and rescaling the output accumulator.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/softmax.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "Softmax",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "SoftmaxSm100",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "reset",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "rescale_O",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "acc_O",
                  "param_type": "cute.Tensor"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "row_scale",
                  "param_type": "cute.Tensor"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "compute_row_max_local",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "acc_S_row",
                  "param_type": "cute.TensorSSA"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "is_first",
                  "param_type": "Boolean"
                }
              ],
              "return_type": "Float32",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "update_row_max",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "acc_S_row",
                  "param_type": "cute.TensorSSA"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "is_first",
                  "param_type": "int"
                }
              ],
              "return_type": "Tuple[Float32, Float32]",
              "visibility": ""
            }
          ],
          "name": "softmax.py",
          "responsibilities": [
            "Implement online softmax with running row max and row sum",
            "Rescale output accumulators when row max updates",
            "Support SM100-specific softmax variants",
            "Provide scale_log2-based exponentiation for numerical stability"
          ],
          "source_summary": "Defines Softmax and SoftmaxSm100 dataclass classes (ParamsBase subclasses) with fields like scale_log2, num_rows, row_max, and row_sum. Methods include reset, rescale_O for rescaling the accumulator tensor, compute_row_max_local, and update_row_max operating on cute.TensorSSA fragments with Float32/Boolean types.",
          "summary": "Implements online softmax classes (Softmax, SoftmaxSm100) for attention kernels, handling row max tracking, exponentiation, and output rescaling."
        },
        {
          "code_purpose": "lib",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.pipeline",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass._mlir.ir",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "typing_extensions",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "The most complex file in this batch (79 functions, 16 classes), providing the work distribution layer for persistent GPU kernels. It defines SchedulingMode, WorkTileInfo, ClcState, and a TileSchedulerProtocol, integrating with MLIR values and cutlass pipeline primitives for cluster launch control (CLC) based scheduling.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/tile_scheduler.py",
          "importance_score": 0.85,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "SchedulingMode",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "WorkTileInfo",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "ClcState",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "class",
              "name": "TileSchedulerProtocol",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "initial_work_tile_info",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "get_current_work",
              "parameters": [],
              "return_type": "WorkTileInfo",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "prefetch_next_work",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "state",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "loc",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ip",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "consumer_wait",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "state",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "loc",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ip",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "consumer_release",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "state",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "loc",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ip",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "producer_tail",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "state",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "loc",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ip",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "tile_scheduler.py",
          "responsibilities": [
            "Distribute work tiles across persistent CTAs",
            "Support CLC (cluster launch control) based dynamic scheduling",
            "Manage producer/consumer pipeline synchronization for tile fetch",
            "Convert scheduler state to/from MLIR values for the DSL"
          ],
          "source_summary": "Defines an IntEnum SchedulingMode, ClcState, and WorkTileInfo dataclass with __new_from_mlir_values__ for MLIR value reconstruction. Provides initial_work_tile_info, get_current_work, prefetch_next_work, consumer_wait/release, and producer_tail methods that wrap pipeline operations, plus a runtime-checkable TileSchedulerProtocol interface.",
          "summary": "Implements persistent kernel tile scheduling for attention kernels, including CLC-based scheduling modes, work tile info, and pipeline synchronization primitives."
        },
        {
          "code_purpose": "specificfeature",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.pipeline",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.nvgpu.cpasync",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.cute_dsl_utils",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "utils",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Provides the KV-gathering stage for sparse/top-K attention on paged KV caches. The CpasyncGatherKVManager consumes top-K indices and a bitmask tensor and asynchronously copies the selected KV pages into shared memory, coordinating CTA rank within a cluster.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/topk_gather_kv.py",
          "importance_score": 0.75,
          "interfaces": [
            {
              "description": null,
              "interface_type": "class",
              "name": "CpasyncGatherKVManager",
              "parameters": [],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "topk_gather_kv.py",
          "responsibilities": [
            "Gather top-K selected KV blocks from paged KV cache",
            "Issue cp.async copies into shared memory",
            "Track bitmask of selected KV blocks",
            "Coordinate CTA rank within thread block clusters"
          ],
          "source_summary": "Defines the CpasyncGatherKVManager dataclass (ParamsBase) holding mIndexTopk and sBitmask cute.Tensors plus cta_rank_in_cluster state. Uses cutlass.pipeline and cutlass.cute.nvgpu.cpasync to issue asynchronous copies of gathered KV blocks, with warp_reduce imported from utils.",
          "summary": "Implements CpasyncGatherKVManager, a manager that gathers top-K selected KV blocks from paged KV cache using cp.async operations."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cutlass_dsl",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass._mlir.dialects.nvvm",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "cutlass.cute.runtime",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "quack.activation",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "A broad utility hub (46 functions) used by nearly every other file in the package. It mixes pure-Python helpers (hashing for kernel caching, softmax scale computation, fastdiv setup) with device-side DSL ops (smid, atomic_add_fp32, warp prefix sum, bit shifts, fp16 conversion) implemented via NVVM/LLVM intrinsics.",
          "file_path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4/utils.py",
          "importance_score": 0.78,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "create_softcap_scoremod",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "softcap_val",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "create_softcap_scoremod_bwd",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "softcap_val",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "compute_softmax_scale_log2",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "softmax_scale",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "score_mod",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "compute_fastdiv_mods",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mQ",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mK",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "qhead_per_kvhead",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "pack_gqa",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "aux_tensors",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "mPageTable",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "convert_from_dlpack",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "x",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "leading_dim",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "alignment",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "divisibility",
                  "param_type": "Any"
                }
              ],
              "return_type": "cute.Tensor",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "smid",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "loc",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ip",
                  "param_type": "Any"
                }
              ],
              "return_type": "Int32",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "atomic_add_fp32",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "a",
                  "param_type": "float | Float32"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "gmem_ptr",
                  "param_type": "cute.Pointer"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "loc",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ip",
                  "param_type": "Any"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "warp_prefix_sum",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "val",
                  "param_type": "cutlass.Int32"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "lane",
                  "param_type": "Optional[cutlass.Int32]"
                }
              ],
              "return_type": "cutlass.Int32",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "cvt_f16",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "src",
                  "param_type": "cute.Tensor"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "dst_or_dtype",
                  "param_type": "Any"
                }
              ],
              "return_type": null,
              "visibility": ""
            }
          ],
          "name": "utils.py",
          "responsibilities": [
            "Provide device-side primitives (atomics, bit ops, warp reductions, prefix sums)",
            "Compute softmax scaling and softcap score modifiers",
            "Build cute.Tensors from dlpack with alignment/divisibility hints",
            "Manage kernel-cache hashing and environment-based defaults",
            "Support fast division/modulo setup for head-dim indexing"
          ],
          "source_summary": "Provides create_softcap_scoremod and create_softcap_scoremod_bwd for attention score capping, compute_softmax_scale_log2, compute_fastdiv_mods, convert_from_dlpack for tensor construction, and device primitives such as smid, atomic_add_fp32, elem_pointer, predicate_k, canonical_warp_group_idx, shl_u32/shr_u32, warp_prefix_sum, and cvt_f16. Also reads environment variables for CLC scheduler and 2-SM-CTA defaults and computes base hashes for compile caching.",
          "summary": "Shared utility module providing low-level GPU primitives, score-modulation functions, tensor conversion helpers, and environment-based defaults for the attention kernels."
        }
      ],
      "importance_score": 0.88375,
      "key_files": [
        "blackwell_helpers.py",
        "block_sparse_utils.py",
        "copy_utils.py",
        "__init__.py",
        "cache_utils.py"
      ],
      "name": "inkling_sm120_fa4",
      "path": ".litho/tree/repo/mods/inkling-sm12-paged-kv/vendor/inkling_sm120_fa4",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains multiple file groups. Key aspects: This directory contains the core implementation of Flash Attention 4 using CUTLASS CuTe DSL, specifically optimized for SM120 (consumer Blackwell) GPUs. It includes kernel helpers, block-sparse utilities, benchmarking tools, and infrastructure for JIT compilation and caching. The files work together to provide high-performance attention kernels with features like FP8 support, block sparsity, and dropout. This directory contains a vendored Cute-DSL (NVIDIA CUTLASS Python) implementation of FlashAttention backward-pass kernels targeting the SM120 (Blackwell) GPU architecture, ported from the FlashAttention Hopper C++/CUTLASS codebase. The files work together as a kernel suite: a main backward mainloop kernel, plus preprocess (D = rowsum(dO*O)) and postprocess (dQ accumulation/normalization) companion kernels, all structured as pipeline-based tiled MMA kernel classes. This directory is a vendored FlashAttention-4 kernel implementation targeting NVIDIA Blackwell GPUs (SM100/SM120) built on the CUTLASS/CuTe Python DSL. It contains highly optimized CUDA kernel code for the forward and backward passes of scaled dot-product attention, with the SM120 variant adapting the SM80-era MMA instructions to Blackwell GeForce's smaller shared memory capacity. These kernels form the performance-critical compute layer of the parent inkling-sm12-paged-kv project. The inkling_sm120_fa4 directory is a vendored, high-performance FlashAttention kernel library written in NVIDIA's CuTe-DSL (Python-based CUTLASS), targeting Blackwell (SM120) and Hopper (SM90) GPUs with paged KV-cache support. The files work together as a complete attention pipeline: flash_fwd.py implements forward attention kernels (SM80/SM90 variants), flash_bwd_sm90.py implements the backward/dgradient kernel, and flash_fwd_combine.py implements the split-K combine kernel that merges partial outputs from the split forward pass. This is core GPU compute infrastructure for the inkling-sm12-paged-kv project. The inkling_sm120_fa4 directory is a vendor package containing high-performance FlashAttention MLA (Multi-head Latent Attention) forward-pass GPU kernels implemented with the CUTLASS/CuTe Python DSL targeting NVIDIA Blackwell (SM100/SM120) architectures. The files work together as specialized CUDA kernel implementations for paged-KV attention inference, with the sm100 MLA kernel being the flagship implementation featuring TMA-based memory pipelines, tcgen05 MMA operations, and shared-memory management for warp-specialized execution. This directory contains vendored FlashAttention forward-pass kernel implementations written in NVIDIA CuTe DSL for Blackwell GPUs (SM100 datacenter and SM120 GeForce/DGX Spark architectures). The files provide architecture-specific specializations of the attention forward kernel: SM100 with TMA and warp specialization, SM120 with CpAsync-based SM80-era MMA instructions, and an SM120 TMA variant, all sharing a common structure of tiled MMA, shared memory pipelines, and paged KV-cache support. This is high-performance GPU library code vendored into the inkling-sm12-paged-kv project to accelerate transformer attention inference. This directory is a vendored, Python/CUTLASS-DSL implementation of FlashAttention forward/backward kernels targeting NVIDIA Hopper (SM90), Blackwell (SM100), and SM120 GPUs, with special support for paged KV caches. The files work together as a layered kernel library: interface.py exposes the PyTorch-facing API and configuration, flash_fwd_sm90.py implements the Hopper forward kernel, and the remaining files provide kernel infrastructure (pipelines, barriers, masking, paged KV management, GQA packing, sequence-length info, and MMA descriptor encodings). It is core computational infrastructure for the inkling-sm12-paged-kv project. This directory contains the SM120 (Blackwell) FlashAttention-4 kernel implementation written in CUTLASS Python DSL, part of a vendored flash-attention package for a paged-KV attention project. The files work together as a GPU kernel library: softmax.py implements online softmax rescaling, tile_scheduler.py manages persistent kernel work distribution, topk_gather_kv.py gathers top-K KV blocks via cp.async, utils.py provides low-level GPU primitives and helpers, and sm90_config_search.py is a standalone tool for enumerating feasible tile configurations."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "from_import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib.Path",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm.model_executor.model_loader.__init__",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script performs automated source patching of the installed vLLM package's model_executor/model_loader/__init__.py. It uses AST parsing and anchor-based text matching to locate the get_model function and insert a hybrid loader policy guarded by a marker comment for idempotency. It includes validation logic to verify patched shapes and a CLI main entry point.",
          "file_path": ".litho/tree/repo/mods/instanttensor-hybrid-draft-loader/patch_model_loader.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "validate_shape",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "patched",
                  "param_type": "bool"
                }
              ],
              "return_type": "None",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "patched_text",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "int",
              "visibility": ""
            }
          ],
          "name": "patch_model_loader.py",
          "responsibilities": [
            "Locate and patch vLLM's model loader source via anchor matching",
            "Validate patched code structure using AST parsing",
            "Ensure idempotent patching via a marker comment",
            "Provide CLI entry point with argument parsing and exit codes"
          ],
          "source_summary": "Defines a MARKER constant for patch idempotency and a GET_MODEL_ANCHOR template matching vLLM's get_model function signature. Key functions include validate_shape (verifies patched text structure), patched_text (applies the hybrid loader transformation to source text), and main (CLI orchestration with argparse, returning an exit code). It relies on ast for structural validation and pathlib for file handling.",
          "summary": "Python patcher that modifies vLLM's model loader to add a hybrid loader policy for speculative draft models."
        },
        {
          "code_purpose": "entry",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "patch_model_loader.py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm.model_executor.model_loader.__init__.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This shell script orchestrates execution of the patch module. It resolves the mod directory, determines the Python site-packages root from environment variables (VLLM_SITE_PACKAGES or PYTHON_ROOT, defaulting to /usr/local/lib/python3.12/dist-packages), locates vLLM's model_loader/__init__.py target, and runs patch_model_loader.py with a mode read from INSTANTTENSOR_DRAFT_LOADER (normalized, defaulting to 'auto'). It uses strict error handling via set -euo pipefail.",
          "file_path": ".litho/tree/repo/mods/instanttensor-hybrid-draft-loader/run.sh",
          "importance_score": 0.55,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "run.sh",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "VLLM_SITE_PACKAGES",
                  "param_type": "env"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "PYTHON_ROOT",
                  "param_type": "env"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "INSTANTTENSOR_DRAFT_LOADER",
                  "param_type": "env"
                }
              ],
              "return_type": "exit_code",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Resolve vLLM site-packages location via environment variables with defaults",
            "Locate the patch target file (model_loader/__init__.py)",
            "Read and normalize the loader mode configuration",
            "Execute the Python patcher with strict error handling"
          ],
          "source_summary": "The script begins with strict mode (set -euo pipefail), defines a logging PREFIX, computes MOD_DIR from the script location, and resolves PYTHON_ROOT with layered environment variable fallbacks. It constructs the VLLM_ROOT and TARGET paths pointing at vllm/model_executor/model_loader/__init__.py, references the patcher script, and normalizes the INSTANTTENSOR_DRAFT_LOADER mode variable (default 'auto') before invoking the patcher.",
          "summary": "Bash entry script that resolves the vLLM installation path and executes the Python patcher with configurable mode."
        }
      ],
      "importance_score": 0.62,
      "key_files": [
        "patch_model_loader.py",
        "run.sh"
      ],
      "name": "instanttensor-hybrid-draft-loader",
      "path": ".litho/tree/repo/mods/instanttensor-hybrid-draft-loader",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a self-contained patching module ('mod') that injects an opt-in hybrid loader policy for speculative draft models into an installed vLLM package. run.sh serves as the entry point, resolving the vLLM site-packages location and mode configuration, then invoking patch_model_loader.py, which performs AST-based anchored code patching of vLLM's model_loader/__init__.py with validation and idempotency checks."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "stat",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "sys",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib.Path",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This is the core logic of the mod: it parses vLLM source files with the ast module, finds tensor copy calls (copy=True) in the InstantTensor weights iterator, and rewrites them to zero-copy views. It inserts ownership-explanation comments and a mod marker, then writes the patched text back, guarding against unsafe open() calls and double-patching.",
          "file_path": ".litho/tree/repo/mods/instanttensor-zero-copy/patch_weight_utils.py",
          "importance_score": 0.85,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "_is_safe_open_call",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "node",
                  "param_type": "ast.Call"
                }
              ],
              "return_type": "bool",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_copy_keyword",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                }
              ],
              "return_type": "tuple[ast.keyword, bool]",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "_source_offset",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "lineno",
                  "param_type": "int"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "column",
                  "param_type": "int"
                }
              ],
              "return_type": "int",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "patched_text",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "text",
                  "param_type": "str"
                }
              ],
              "return_type": "str",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "int",
              "visibility": ""
            }
          ],
          "name": "patch_weight_utils.py",
          "responsibilities": [
            "Parse and analyze vLLM source code using the ast module",
            "Rewrite copy=True tensor creation into zero-copy views",
            "Insert ownership comments and mod marker into patched code",
            "Validate safety of file open calls before patching",
            "Provide a CLI entry point (main) for running the patcher"
          ],
          "source_summary": "The script defines helper functions _is_safe_open_call, _copy_keyword, and _source_offset for safe AST manipulation and source offset computation, a patched_text function that performs the actual code transformation, and a main() CLI entry point with argparse that locates and patches the target vLLM file. It uses constants PREFIX, MARKER, and ownership/zero-copy comment strings to tag patched code.",
          "summary": "AST-based source patcher that modifies vLLM's weight utility code to enable zero-copy tensor views in the InstantTensor weights iterator."
        },
        {
          "code_purpose": "entry",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "python3",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "patch_weight_utils.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This shell script is the operational entry point of the mod. It resolves its own directory, verifies python3 availability and the presence of the patcher script, and supports a VLLM_PACKAGE_ROOT environment variable for tests or non-standard vLLM locations before executing patch_weight_utils.py.",
          "file_path": ".litho/tree/repo/mods/instanttensor-zero-copy/run.sh",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "command",
              "name": "main (script execution)",
              "parameters": [],
              "return_type": "int (exit code)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Serve as the mod's execution entry point",
            "Validate python3 availability before patching",
            "Verify the patcher script exists",
            "Support VLLM_PACKAGE_ROOT override for tests and unusual installs",
            "Provide prefixed logging output"
          ],
          "source_summary": "The script uses strict bash mode (set -euo pipefail), defines a PREFIX for log messages, resolves MOD_DIR from BASH_SOURCE, and checks that python3 and the patcher script exist before running the patcher. It echoes a header identifying the InstantTensor zero-copy weight loader mod.",
          "summary": "Bash entry script that validates prerequisites and invokes the Python patcher to apply the zero-copy mod to a vLLM installation."
        }
      ],
      "importance_score": 0.55,
      "key_files": [
        "patch_weight_utils.py",
        "run.sh"
      ],
      "name": "instanttensor-zero-copy",
      "path": ".litho/tree/repo/mods/instanttensor-zero-copy",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory is a self-contained 'mod' (patch package) that enables zero-copy tensor views in vLLM's InstantTensor weight loading path. run.sh serves as the entry point that locates the vLLM installation and invokes the patcher, while patch_weight_utils.py performs AST-based source patching of vLLM's weight utility code, replacing copy=True tensor creation with zero-copy views plus ownership comments and a mod marker."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm/v1/worker/gpu_worker.py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm/config/cache.py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "python3",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script serves as the sole entry point for the kv-cache-prealloc-cleanup operation. It resolves the Python packages root (defaulting to /usr/local/lib/python3.12/dist-packages), verifies that the vLLM target files exist, and checks for a python3 interpreter before proceeding with patching. It uses strict error handling (set -euo pipefail) to fail fast on any missing prerequisite.",
          "file_path": ".litho/tree/repo/mods/kv-cache-prealloc-cleanup/run.sh",
          "importance_score": 0.5,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "main (script execution)",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "PYTHON_ROOT",
                  "param_type": "environment variable (string path)"
                }
              ],
              "return_type": "exit code (0 on success, 1 on missing prerequisites)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Resolve the Python installation root via PYTHON_ROOT environment variable with a sensible default",
            "Validate that vLLM's gpu_worker.py and cache.py exist before patching",
            "Check that python3 is available on the system for patch execution",
            "Apply KV cache pre-allocation cleanup modifications to the installed vLLM package",
            "Fail fast with clear error messages using strict shell error handling"
          ],
          "source_summary": "The script begins with strict shell mode settings, defines PYTHON_ROOT and TARGET/CACHE_CONFIG paths pointing into the vLLM v1 worker and config modules, and performs existence checks on both files, exiting with descriptive error messages if absent. It also verifies python3 availability via command -v; the remainder (truncated) presumably applies in-place edits or a Python-based patch to remove/adjust KV cache pre-allocation code.",
          "summary": "A Bash setup/patch script that locates and modifies vLLM's gpu_worker.py and cache.py to clean up KV cache pre-allocation logic."
        }
      ],
      "importance_score": 0.45,
      "key_files": [
        "run.sh"
      ],
      "name": "kv-cache-prealloc-cleanup",
      "path": ".litho/tree/repo/mods/kv-cache-prealloc-cleanup",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a single Bash utility script that patches an installed vLLM installation, specifically targeting gpu_worker.py and cache.py to clean up or modify KV cache pre-allocation behavior. It acts as a deployment/maintenance tool rather than core application logic, validating the presence of target files and a Python interpreter before applying modifications."
    },
    {
      "file_count": 8,
      "file_insights": [
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "yaml",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "report",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "json",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This is the analytical heart of the memory-profile subsystem. It provides KV-cache budgeting, peak-host-memory estimation, and topology-aware host selection checks, and emits capacity utilization metrics.",
          "file_path": ".litho/tree/repo/mods/memory-profile/capacity.py",
          "importance_score": 0.95,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "positive",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "config",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rank",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "kv_breakdown",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "model",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "context",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "sequences",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "shared_kv_budget",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "models",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "demands",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "peak_requirement",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "host",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "old_kv",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "new_kv",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "analyze",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "card",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "context",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "reserve",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "select_hosts",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "count",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "config_path",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "explicit",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "probe_host",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "host",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "timeout",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "check_cluster",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "card",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "analysis",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "inventories",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "aggregate",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "key",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "aggregate",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "key",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "capacity.py",
          "responsibilities": [
            "Estimate startup RAM and KV-cache requirements",
            "Validate Spark deployment host memory capacity",
            "Compute shared KV budgets and host peak requirements",
            "Provide CLI-driven host selection and cluster probing"
          ],
          "source_summary": "Defines ~23 functions for positive validation, config loading, KV cache breakdown, shared-budget computation, utilization estimation, and cluster analysis. It also includes a TopologyError class and an aggregate helper with a method variant.",
          "summary": "Core capacity-analysis module that estimates startup RAM requirements and validates a profiled Spark deployment against available host memory."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "profile_card",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "tarfile",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "shutil",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This utility extracts Docker-copied tar streams from remote/local containers, filters profile files, and invokes profile_card.write_card to consolidate a run.",
          "file_path": ".litho/tree/repo/mods/memory-profile/collect.py",
          "importance_score": 0.55,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "extract",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "archive",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "destination",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "collect.py",
          "responsibilities": [
            "Extract memory-profile archives from Docker/SSH",
            "Filter to directories and regular profile files",
            "Write consolidated profile card via profile_card.write_card"
          ],
          "source_summary": "Implements extract(archive, destination) that unpacks tar streams while ignoring links/devices/owners, and a main() CLI that drives collection and card writing.",
          "summary": "Collects one memory-profile run from local/SSH containers and writes its profile card."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ctypes",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "json",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "platform",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "socket",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Collects host memory, GPU inventory, and system identifiers without initializing CUDA or Torch, enabling lightweight remote capacity checks.",
          "file_path": ".litho/tree/repo/mods/memory-profile/host_probe.py",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "inventory",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "call",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "args",
                  "param_type": "tuple"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "call",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "name",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "args",
                  "param_type": "tuple"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "host_probe.py",
          "responsibilities": [
            "Gather memory and host identifiers",
            "Collect CUDA device/driver metadata",
            "Provide JSON inventory for remote SSH usage"
          ],
          "source_summary": "Reads /proc/meminfo, gathers CUDA driver/device info via ctypes, and returns a JSON inventory; includes a call wrapper for SSH invocation.",
          "summary": "Read-only Linux/CUDA inventory probe executable over SSH stdin."
        },
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ast",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "hashlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "json",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script patches the vLLM source tree to insert memory-profile instrumentation markers and worker hooks, enabling runtime memory sampling.",
          "file_path": ".litho/tree/repo/mods/memory-profile/patch.py",
          "importance_score": 0.7,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "identifier",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "patched",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "source",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kind",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "install",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "root",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "patch.py",
          "responsibilities": [
            "Patch vLLM source tree with instrumentation",
            "Avoid importing vLLM or CUDA during install",
            "Apply idempotent marker-based modifications"
          ],
          "source_summary": "Uses AST-based source rewriting and marker strings to inject the profiler, with functions identifier(), patched(), and install() that handle idempotent installation.",
          "summary": "Installs the opt-in memory profiler into an existing vLLM package without importing vLLM or initializing CUDA."
        },
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "ctypes",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "functools",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "hashlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "json",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "weakref",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Imported inside vLLM as vllm._spark_memory_profile, it records phase-by-phase memory, native heap, CUDA memory, and model/KV inventory during worker startup. It can also run as a standalone sampler.",
          "file_path": ".litho/tree/repo/mods/memory-profile/probe.py",
          "importance_score": 0.9,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "scalar",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "host_memory",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "process_memory",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "pid",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "native_heap",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "observe",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "function",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "args",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kwargs",
                  "param_type": "dict"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "cuda_memory",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "device",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "metadata",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "worker",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "install_worker",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "namespace",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "probe.py",
          "responsibilities": [
            "Instrument vLLM worker startup memory phases",
            "Sample process, native heap, and CUDA memory",
            "Record model and KV-cache inventory",
            "Expose install_worker hook for vLLM namespace"
          ],
          "source_summary": "Defines many functions for memory/proc counters, CUDA memory queries, manifest generation, observer context, and worker installation. Includes a Mallinfo2 class and visitor methods for inventorying model/storage objects.",
          "summary": "Startup instrumentation and standalone CPU-only host sampler that observes vLLM worker memory and CUDA usage."
        },
        {
          "code_purpose": "service",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "yaml",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "json",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "Counter",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "tempfile",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This module reads per-rank event JSONL, summarizes CUDA and KV memory, computes phase peaks, and writes a consolidated YAML card consumed by capacity analysis and report generation.",
          "file_path": ".litho/tree/repo/mods/memory-profile/profile_card.py",
          "importance_score": 0.85,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "jsonl",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "path",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "host_directories",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "inputs",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "last",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rows",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "phase",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "cuda_summary",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "cuda",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "checkpoint",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "row",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "kv_summary",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "kv",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "rank_card",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "events",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "build_card",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "inputs",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "session",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "local",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "phase_peak",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "before",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "method",
              "name": "phase_peak",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "before",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "write_card",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "inputs",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "output",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "session",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "local",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "profile_card.py",
          "responsibilities": [
            "Parse per-rank JSONL event rows",
            "Compute CUDA and KV-cache summaries",
            "Build consolidated YAML profile cards",
            "Provide CLI entry point for card creation"
          ],
          "source_summary": "Contains JSONL readers, host-directory discovery, summary builders, rank-card assembly, phase-peak computation, and write_card/main CLI functions for card generation.",
          "summary": "Merges memory-profile run directories into a model/recipe YAML profile card."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "yaml",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": false,
              "line_number": null,
              "name": "profile_card",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "math",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "pathlib",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "urllib",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This reporting utility converts profile-card data into human-readable Markdown tables and Matplotlib-based charts of CPU and CUDA memory over startup phases.",
          "file_path": ".litho/tree/repo/mods/memory-profile/report.py",
          "importance_score": 0.8,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "number",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "amount",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "divisor",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "unit",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "difference",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "baseline",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "cell",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "value",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "table",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "headers",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rows",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "load_card",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "path",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "process_label",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "process",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "api",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "startup_points",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "process",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "host",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "series",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "points",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "field",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "session",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "relative",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "baseline",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "host_points",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "card",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "host",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "cpu_stack_series",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "processes",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "host",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "plot_cpu_stack",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "ax",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "processes",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "host",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "rank_count",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "colors",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "markdown_report",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "card",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "graph",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "session",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "relative",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "plot_card",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "card",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "output",
                  "param_type": "Any"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "session",
                  "param_type": "tuple"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "relative",
                  "param_type": "Any"
                }
              ],
              "return_type": "Any",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "Any",
              "visibility": ""
            }
          ],
          "name": "report.py",
          "responsibilities": [
            "Render Markdown memory reports",
            "Plot CPU/CUDA startup memory charts",
            "Load profile cards and extract series",
            "Support CLI report generation"
          ],
          "source_summary": "Provides number/amount/cell/table helpers, card loading, process/host point extraction, CPU stack series, plot_cpu_stack, markdown_report, and plot_card; main runs report generation.",
          "summary": "Renders a memory profile card as Markdown and a CPU/CUDA startup chart."
        },
        {
          "code_purpose": "entry",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "python3",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "patch.py",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Bash entry point that sets the vLLM package root, calls patch.py to install instrumentation, and initiates profile collection.",
          "file_path": ".litho/tree/repo/mods/memory-profile/run.sh",
          "importance_score": 0.65,
          "interfaces": [],
          "name": "run.sh",
          "responsibilities": [
            "Locate installed vLLM package root",
            "Invoke patch.py to install profiler",
            "Run memory-profile collection pipeline"
          ],
          "source_summary": "Uses set -euo pipefail, resolves the module directory and vLLM root via Python importlib, then executes patch.py and the profiler.",
          "summary": "Shell orchestrator that locates the vLLM package, patches it, and runs the memory profiler."
        }
      ],
      "importance_score": 0.75,
      "key_files": [
        "capacity.py",
        "probe.py",
        "profile_card.py",
        "report.py",
        "patch.py"
      ],
      "name": "memory-profile",
      "path": ".litho/tree/repo/mods/memory-profile",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a memory profiling and capacity-planning subsystem for vLLM-on-Spark deployments. It instruments vLLM worker processes to capture CPU/CUDA memory traces, merges collected runs into YAML profile cards, and analyzes those cards to estimate startup RAM requirements and validate cluster capacity. The files form a pipeline: patch/install instrumentation, probe/collect data, build cards, analyze capacity, and render reports."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This bash script is the only file in the directory and acts as a one-step enabler for fp8 KV cache in MiMo-V2. It documents two upstream vLLM bugs and likely contains a diff or sed/patch commands to modify mimo_v2.py accordingly.",
          "file_path": ".litho/tree/repo/mods/mimo-diffkv-fp8-kv/run.sh",
          "importance_score": 0.72,
          "interfaces": [],
          "name": "run.sh",
          "responsibilities": [
            "Patch vLLM MiMo-V2 to support fp8 KV cache",
            "Propagate cache_config to Attention()",
            "Prevent sliding_window fallback on full-attention layers",
            "Provide runnable mod script for deployment/integration"
          ],
          "source_summary": "The script sets strict shell options (set -euo pipefail) and begins with a detailed comment explaining the two verified upstream gaps: (1) mimo_v2.py never passes cache_config to Attention(), so fp8 is ignored; (2) passing cache_config causes Attention() to fall back to sliding_window=128 for full-attention layers, which must be avoided. The remainder of the script presumably applies the required source modifications.",
          "summary": "Applies a patch/configuration mod to vLLM's MiMo-V2 code to make --kv-cache-dtype fp8 effective by propagating cache_config to Attention() and preventing an unwanted sliding_window fallback."
        }
      ],
      "importance_score": 0.62,
      "key_files": [
        "run.sh"
      ],
      "name": "mimo-diffkv-fp8-kv",
      "path": ".litho/tree/repo/mods/mimo-diffkv-fp8-kv",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains run.sh, a targeted shell script that enables fp8 KV cache support for MiMo-V2 in vLLM by patching how cache_config is passed to Attention(). It addresses two specific upstream gaps and serves as a configuration/patch utility for the inference engine."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vLLM",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "transformers",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This is the only file in the directory and serves as an operational enablement script. It addresses compatibility issues by symlinking weight files into a flat staged directory and generating a corrected config.json.",
          "file_path": ".litho/tree/repo/mods/mimo-v2.6-flash/run.sh",
          "importance_score": 0.5,
          "interfaces": [],
          "name": "run.sh",
          "responsibilities": [
            "Create a loadable staging directory for the DFlash drafter",
            "Symlink original model weight files into the staged directory",
            "Generate a sanitized config.json without trailing commas",
            "Enable vLLM to use the drafter as a draft model"
          ],
          "source_summary": "The script begins with strict shell options and a comment explaining the need to stage a DFlash drafter copy. It creates /workspace/MiMo-V2.6-Flash-RL-dflash, symlinks original weight files, and writes a sanitized config to avoid the trailing comma that transformers rejects.",
          "summary": "Bash script that stages a loadable copy of the MiMo-V2.6-Flash-RL DFlash drafter at /workspace/MiMo-V2.6-Flash-RL-dflash for use with vLLM."
        }
      ],
      "importance_score": 0.45,
      "key_files": [
        "run.sh"
      ],
      "name": "mimo-v2.6-flash",
      "path": ".litho/tree/repo/mods/mimo-v2.6-flash",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains a single deployment helper script that prepares the MiMo-V2.6-Flash-RL DFlash drafter checkpoint for vLLM inference. It stages a loadable copy by symlinking original weight files and writing a sanitized config, working around vLLM's inability to use Hub subdirectories and transformers' strict JSON parsing."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "wget",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "huggingface.co (NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 repository)",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "WORKSPACE_DIR",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script acts as a setup/bootstrap utility for the nemotron-nano workspace. It changes into the workspace directory and fetches a reasoning parser module associated with the NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 model. It enables error propagation via 'set -e' so failures abort subsequent steps.",
          "file_path": ".litho/tree/repo/mods/nemotron-nano/run.sh",
          "importance_score": 0.3,
          "interfaces": [
            {
              "description": null,
              "interface_type": "shell_script",
              "name": "run.sh (script execution)",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "WORKSPACE_DIR",
                  "param_type": "environment_variable"
                }
              ],
              "return_type": "exit_code",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Set fail-fast shell behavior with 'set -e'",
            "Navigate to the workspace directory via $WORKSPACE_DIR",
            "Download the nano_v3_reasoning_parser.py asset from Hugging Face",
            "Bootstrap model-related parsing resources for the Nemotron-Nano setup"
          ],
          "source_summary": "The script begins with '#!/bin/bash' and 'set -e' to enable fail-fast behavior. It changes the working directory to $WORKSPACE_DIR, then uses wget to download nano_v3_reasoning_parser.py from the Hugging Face resolve URL for the nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 model repository.",
          "summary": "A shell script that downloads the nano_v3_reasoning_parser.py file from NVIDIA's Hugging Face model repository."
        }
      ],
      "importance_score": 0.25,
      "key_files": [
        "run.sh"
      ],
      "name": "nemotron-nano",
      "path": ".litho/tree/repo/mods/nemotron-nano",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The 'nemotron-nano' directory is a minimal setup directory containing a single shell script that downloads the Nemotron-3-Nano reasoning parser Python file from NVIDIA's Hugging Face repository. It serves as a bootstrap/utility step for fetching model-related parsing assets rather than containing core business logic."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "wget",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "WORKSPACE_DIR",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script acts as a setup/bootstrap utility for the nemotron-super workspace. It changes into the workspace directory (defined by the WORKSPACE_DIR environment variable) and fetches a reasoning parser Python module associated with the NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 model. The 'set -e' flag ensures the script aborts immediately if any command fails.",
          "file_path": ".litho/tree/repo/mods/nemotron-super/run.sh",
          "importance_score": 0.3,
          "interfaces": [],
          "name": "run.sh",
          "responsibilities": [
            "Navigate to the workspace directory via the WORKSPACE_DIR environment variable",
            "Download the super_v3_reasoning_parser.py reasoning parser from Hugging Face",
            "Fail fast on errors using 'set -e' to ensure reliable setup"
          ],
          "source_summary": "The script begins with '#!/bin/bash' and 'set -e' for strict error handling. It changes the working directory to $WORKSPACE_DIR, then uses wget to download super_v3_reasoning_parser.py from the Hugging Face repository 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4'. No other logic, functions, or error handling beyond fail-fast behavior is present.",
          "summary": "A shell script that downloads the super_v3_reasoning_parser.py file from the NVIDIA Nemotron 3 Super model repository on Hugging Face into the workspace directory."
        }
      ],
      "importance_score": 0.25,
      "key_files": [
        "run.sh"
      ],
      "name": "nemotron-super",
      "path": ".litho/tree/repo/mods/nemotron-super",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The 'nemotron-super' directory is a minimal setup directory containing a single shell script that downloads a reasoning parser Python file for the NVIDIA Nemotron 3 Super 120B model from Hugging Face. It serves as an environment preparation step rather than containing core business logic, acting as a utility bootstrap for model-related tooling."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "plugin",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm/config/speculative.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This patch file contains the actual code changes that enable DSpark draft model support in vLLM. It extends the architecture/model-name detection logic in vllm/config/speculative.py so that DSpark-family draft models are accepted by the speculative decoding configuration.",
          "file_path": ".litho/tree/repo/mods/radixark-dspark/radixark-dspark.patch",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "patch",
              "name": "apply_patch",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "target",
                  "param_type": "vllm/config/speculative.py"
                }
              ],
              "return_type": "modified source file",
              "visibility": ""
            }
          ],
          "name": "radixark-dspark.patch",
          "responsibilities": [
            "Extend vLLM speculative decoding config with DSpark draft model detection",
            "Add architecture-based matching for DSparkDraftModel",
            "Serve as the payload applied by run.sh during installation"
          ],
          "source_summary": "The patch targets vllm/config/speculative.py around line 1077, adding an 'or' clause to a boolean condition that checks the draft model config. The new clause matches 'DSparkDraftModel' in the draft model's architectures list, complementing existing checks for 'dspark' in the model name, 'Qwen3DSparkModel', and 'Gemma4DSparkModel'.",
          "summary": "A unified diff patch that modifies vLLM's speculative decoding config to recognize DSpark draft model architectures."
        },
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "python3",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "radixark-dspark.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script is the entry point for installing the radixark-dspark module. It validates prerequisites (python3, patch file presence), locates the vLLM package root (with an optional VLLM_PACKAGE_ROOT override for tests and unusual image layouts), and applies radixark-dspark.patch to the discovered vLLM installation.",
          "file_path": ".litho/tree/repo/mods/radixark-dspark/run.sh",
          "importance_score": 0.65,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "main",
              "parameters": [],
              "return_type": "exit code (0 on success, non-zero on failure)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Validate runtime prerequisites (python3, patch file existence)",
            "Discover the installed vLLM package root, honoring VLLM_PACKAGE_ROOT override",
            "Apply radixark-dspark.patch to the vLLM installation",
            "Provide prefixed logging and fail-fast error handling"
          ],
          "source_summary": "The script uses strict bash mode (set -euo pipefail), defines a PREFIX for log messages, resolves its own directory to find the patch file, and checks that python3 is available. It then discovers the vLLM package root via python3 (supporting a VLLM_PACKAGE_ROOT environment variable override) and applies the patch file to the located vLLM source.",
          "summary": "A bash installation script that discovers the installed vLLM package and applies the bundled DSpark patch to it."
        }
      ],
      "importance_score": 0.55,
      "key_files": [
        "run.sh",
        "radixark-dspark.patch"
      ],
      "name": "radixark-dspark",
      "path": ".litho/tree/repo/mods/radixark-dspark",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The radixark-dspark directory is a self-contained patch module that extends vLLM's speculative decoding configuration to support DSpark-family draft models (e.g., Qwen3DSparkModel, Gemma4DSparkModel, DSparkDraftModel). The run.sh script locates and validates the installed vLLM package and applies the bundled radixark-dspark.patch to its speculative config, making the two files an installer-plus-payload pair."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "step-3.7-support.patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "patch",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "python3",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script is the executable entry point of the directory. It resolves the Python dist-packages root and its own directory, locates the patch file, and checks whether upstream vLLM already contains Step-3.7 model support before applying the patch. It uses strict error handling (set -euo pipefail) and supports environment-variable overrides for the Python root, binary, and workspace.",
          "file_path": ".litho/tree/repo/mods/step-3.7-flash/run.sh",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "has_upstream_step37_support",
              "parameters": [],
              "return_type": "boolean (exit status)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Resolve environment configuration (PYTHON_ROOT, PYTHON_BIN, WORKSPACE) with defaults",
            "Detect whether upstream vLLM already ships Step-3.7 support",
            "Apply the step-3.7-support.patch to the vLLM source tree with path exclusions",
            "Enforce strict error handling via set -euo pipefail"
          ],
          "source_summary": "The script sets up strict bash mode, defines configurable variables (PYTHON_ROOT, PYTHON_BIN, MOD_DIR, WORKSPACE) with sensible defaults, and builds a PATCH_EXCLUDES array excluding docs/*, examples/*, and tests/*. It defines a helper function has_upstream_step37_support that inspects vllm/model_executor/models/step3p7.py and the model registry to determine whether the patch needs to be applied.",
          "summary": "Bash entry script that applies the Step-3.7 support patch to a vLLM installation if upstream support is absent."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "docs/models/supported_models.md",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This patch file contains the actual code changes needed to enable Step-3.7 model inference in vLLM. It modifies the supported-models documentation (docs/models/supported_models.md) to register the new Step-3.7 model class, and presumably adds or updates the model implementation and registry entries referenced by run.sh's detection logic. It is consumed by run.sh and applied with docs/examples/tests exclusions.",
          "file_path": ".litho/tree/repo/mods/step-3.7-flash/step-3.7-support.patch",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "step-3.7-support.patch",
          "responsibilities": [
            "Define code changes that add Step-3.7 model support to vLLM",
            "Update supported-models documentation to list the new model",
            "Serve as the payload applied by run.sh during installation",
            "Carry diff metadata (index lines, hunks) for idempotent patch application"
          ],
          "source_summary": "The visible portion of the diff updates docs/models/supported_models.md, adding a new row for the Step-3.7 model class alongside existing Step3-VL entries (e.g., Step3VLForConditionalGeneration / stepfun-ai/step3). The full patch presumably also touches vllm/model_executor/models/step3p7.py and the model registry to wire in the new architecture.",
          "summary": "Unified diff patch that adds Step-3.7 (StepFun) model support to the vLLM codebase, including documentation updates."
        }
      ],
      "importance_score": 0.55,
      "key_files": [
        "run.sh",
        "step-3.7-support.patch"
      ],
      "name": "step-3.7-flash",
      "path": ".litho/tree/repo/mods/step-3.7-flash",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory provides a self-contained patch mechanism to add Step-3.7 (StepFun) model support to a vLLM installation. The run.sh script orchestrates the process: it detects whether upstream vLLM already includes Step-3.7 support, and if not, applies the step-3.7-support.patch file to the installed vLLM source tree (excluding docs, examples, and tests). It acts as an installation/patching utility layer rather than core application logic."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "launch-cluster.sh",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "require",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This shell script is part of a deployment or setup pipeline for an NGC vLLM environment. It enables 'set -e' for fail-fast behavior and echoes a message indicating the NGC vLLM mod has been applied. Its actual initialization work has been delegated to launch-cluster.sh, so it now functions primarily as a marker or hook in the mod application sequence.",
          "file_path": ".litho/tree/repo/mods/use-ngc-vllm/run.sh",
          "importance_score": 0.15,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "run.sh (script execution)",
              "parameters": [],
              "return_type": "exit code (0 on success)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Mark the application of the NGC vLLM mod in the deployment pipeline",
            "Fail fast on errors via 'set -e'",
            "Serve as a hook point for mod-related setup steps"
          ],
          "source_summary": "The script starts with '#!/bin/bash' and 'set -e' to abort on any command failure. It contains a single echo statement: 'NGC vLLM mod applied.' with a comment noting that container initialization is now handled by launch-cluster.sh. There is no other executable logic.",
          "summary": "A minimal bash script that applies an NGC vLLM modification by printing a status message. Container initialization logic has been relocated to launch-cluster.sh, leaving this script as a placeholder."
        }
      ],
      "importance_score": 0.15,
      "key_files": [
        "run.sh"
      ],
      "name": "use-ngc-vllm",
      "path": ".litho/tree/repo/mods/use-ngc-vllm",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The 'use-ngc-vllm' directory is a minimal infrastructure/mod directory containing a single shell script that applies an NGC vLLM modification. The script itself is now largely a no-op placeholder, as container initialization has been moved to launch-cluster.sh. It serves as a marker or hook point in a larger deployment pipeline rather than containing business logic."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "apt/package manager",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "pip",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "git",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "earlyoom",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "InstantTensor",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "SciPy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "nvidia-nccl-cu13 / libnccl2",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "launch-cluster.sh",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script acts as a bootstrap/compatibility shim executed inside official vLLM container images. It bridges gaps between the minimal official images and the expectations of launch-cluster.sh (e.g., earlyoom support) and other mods that require git. It also handles hardware-specific workarounds such as redirecting the pip-installed nvidia-nccl-cu13 libnccl.so.2 to the system libnccl2 soname on DGX Spark.",
          "file_path": ".litho/tree/repo/mods/use-official-vllm/run.sh",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "script",
              "name": "run.sh (script entry)",
              "parameters": [],
              "return_type": "int (exit code)",
              "visibility": ""
            }
          ],
          "name": "run.sh",
          "responsibilities": [
            "Install git into official vLLM containers that omit it",
            "Install earlyoom to support launch-cluster.sh --earlyoom",
            "Install InstantTensor and SciPy while preserving the existing Torch build",
            "Redirect pip-installed nvidia-nccl-cu13 libnccl.so.2 to the system libnccl2 soname on DGX Spark",
            "Fail fast on errors via strict shell mode (set -euo pipefail)"
          ],
          "source_summary": "The script begins with a strict bash preamble (set -euo pipefail) and a comment block explaining its compatibility goals for official vLLM containers. Its main actions include installing git and earlyoom via the system package manager, installing InstantTensor and SciPy via pip without disturbing the existing Torch build, and creating a symlink/redirect so the pip-installed NCCL library satisfies the system libnccl2 soname when both are present, specifically addressing DGX Spark failures.",
          "summary": "A bash setup script that prepares official vLLM Docker containers for compatibility with the project's cluster launch tooling. It installs required packages and patches NCCL library linkage issues."
        }
      ],
      "importance_score": 0.55,
      "key_files": [
        "run.sh"
      ],
      "name": "use-official-vllm",
      "path": ".litho/tree/repo/mods/use-official-vllm",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The 'use-official-vllm' directory contains a single environment-setup script that adapts official vLLM Docker containers for use with the surrounding cluster infrastructure. run.sh installs missing system packages (git, earlyoom, InstantTensor, SciPy) while preserving the pre-built Torch installation, and fixes NCCL library soname mismatches on DGX Spark hardware. It serves as a compatibility/bootstrap layer rather than containing core business logic."
    },
    {
      "file_count": 36,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node-b12x",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "--exp-b12x",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This YAML recipe specifies the DeepSeek-V4-Flash-0731 model, the vllm-node-b12x container, and the --exp-b12x build argument for cluster-only deployment. It includes default serving parameters such as port, host, and tensor parallelism, and leaves mods empty with commented-out alternatives.",
          "file_path": ".litho/tree/repo/recipes/deepseek-v4-flash-0731.yaml",
          "importance_score": 0.61,
          "interfaces": [],
          "name": "deepseek-v4-flash-0731.yaml",
          "responsibilities": [
            "Declare serving configuration for DeepSeek-V4-Flash-0731",
            "Enable B12X-optimized build",
            "Restrict to cluster execution",
            "Provide default server and parallelism settings"
          ],
          "source_summary": "Contains recipe_version, model, container, build_args, cluster_only flag, mods list, and defaults for port, host, and tensor settings.",
          "summary": "Defines a recipe for serving DeepSeek-V4-Flash-0731 on a dual Spark cluster using the B12X-optimized vLLM container."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node-b12x",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "--exp-b12x",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the vision-capable variant of DeepSeek V4 Flash for cluster-only deployment with the B12X container and build args. It mirrors the standard flash recipe with defaults for port, host, and parallelization.",
          "file_path": ".litho/tree/repo/recipes/deepseek-v4-flash-vision-exp.yaml",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "deepseek-v4-flash-vision-exp.yaml",
          "responsibilities": [
            "Declare serving configuration for the vision variant",
            "Enable B12X-optimized build",
            "Restrict to cluster execution",
            "Provide default server settings"
          ],
          "source_summary": "Specifies recipe metadata, model ID, container, build args, cluster_only flag, empty mods, and default serving options.",
          "summary": "Defines a recipe for serving DeepSeek-V4-Flash-Vision-Exp on a dual Spark cluster using the B12X-optimized vLLM container."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe is the base configuration for the DeepSeek-V4-Flash model on a DGX Spark cluster using the standard vllm-node container. It represents the flagship DeepSeek V4 deployment and includes defaults for port, host, and tensor parallelism.",
          "file_path": ".litho/tree/repo/recipes/deepseek-v4-flash.yaml",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "deepseek-v4-flash.yaml",
          "responsibilities": [
            "Declare DeepSeek-V4-Flash model serving",
            "Use standard vllm-node container",
            "Restrict to cluster execution",
            "Provide default serving parameters"
          ],
          "source_summary": "Includes recipe_version, name, description, model ID, container, cluster_only flag, empty mods list, and defaults section.",
          "summary": "Defines a recipe for serving DeepSeek-V4-Flash with sparse MLA/DeepGEMM experimental SM120 support."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/diffusiongemma",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the DiffusionGemma BF16 thinking variant using the diffusiongemma mod and solo-only execution. It enables thinking behavior for the model on a single node.",
          "file_path": ".litho/tree/repo/recipes/diffusion-gemma-bf16-thinking.yaml",
          "importance_score": 0.5,
          "interfaces": [],
          "name": "diffusion-gemma-bf16-thinking.yaml",
          "responsibilities": [
            "Configure DiffusionGemma BF16 serving",
            "Enable Gemma4 thinking",
            "Restrict to solo node execution",
            "Apply diffusiongemma mod"
          ],
          "source_summary": "Specifies model, container, solo_only flag, and a mods list containing mods/diffusiongemma (truncated in provided content).",
          "summary": "Defines a recipe for serving DiffusionGemma-4-26B-A4B-IT in BF16 with Gemma4 thinking enabled on a single node."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/diffusiongemma",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe is the non-thinking BF16 variant for DiffusionGemma, applying the diffusiongemma mod for reasoning parsing. It is solo-only and uses the standard vllm-node container.",
          "file_path": ".litho/tree/repo/recipes/diffusion-gemma-bf16.yaml",
          "importance_score": 0.5,
          "interfaces": [],
          "name": "diffusion-gemma-bf16.yaml",
          "responsibilities": [
            "Configure DiffusionGemma BF16 serving",
            "Apply Gemma4 reasoning parser mod",
            "Restrict to solo node execution",
            "Provide default settings"
          ],
          "source_summary": "Includes model, container, solo_only flag, mods list with mods/diffusiongemma, and default settings.",
          "summary": "Defines a recipe for serving DiffusionGemma-4-26B-A4B-IT in BF16 on a single node with a Gemma4 reasoning parser."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/diffusiongemma",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the NVFP4 quantized DiffusionGemma model with thinking enabled, using solo-only execution and the diffusiongemma mod. It targets the nvidia-hosted model repository.",
          "file_path": ".litho/tree/repo/recipes/diffusion-gemma-nvfp4-thinking.yaml",
          "importance_score": 0.5,
          "interfaces": [],
          "name": "diffusion-gemma-nvfp4-thinking.yaml",
          "responsibilities": [
            "Configure NVFP4 DiffusionGemma serving",
            "Enable thinking mode",
            "Restrict to solo execution",
            "Apply diffusiongemma mod"
          ],
          "source_summary": "Specifies nvidia model ID, vllm-node container, solo_only flag, and mods for diffusiongemma.",
          "summary": "Defines a recipe for serving DiffusionGemma-26B-A4B-IT-NVFP4 with thinking enabled on a single node."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/diffusiongemma",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe is the non-thinking NVFP4 variant of DiffusionGemma, applying the diffusiongemma mod for reasoning parsing. It is solo-only and uses the standard vllm-node container.",
          "file_path": ".litho/tree/repo/recipes/diffusion-gemma-nvfp4.yaml",
          "importance_score": 0.5,
          "interfaces": [],
          "name": "diffusion-gemma-nvfp4.yaml",
          "responsibilities": [
            "Configure NVFP4 DiffusionGemma serving",
            "Apply reasoning parser mod",
            "Restrict to solo execution",
            "Provide default settings"
          ],
          "source_summary": "Includes nvidia model ID, container, solo_only flag, mods list, and default settings for the serving process.",
          "summary": "Defines a recipe for serving DiffusionGemma-26B-A4B-IT-NVFP4 with a Gemma4 reasoning parser on a single node."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe provides the NVFP4 quantized version of Gemma4-26B-A4B, supporting solo-only execution. It notes that vLLM uses Transformers v5 by default, eliminating legacy TF5 build args, and comments out an optional tool parser fix mod.",
          "file_path": ".litho/tree/repo/recipes/gemma4-26b-a4b-nvfp4.yaml",
          "importance_score": 0.58,
          "interfaces": [],
          "name": "gemma4-26b-a4b-nvfp4.yaml",
          "responsibilities": [
            "Configure NVFP4 Gemma4 serving",
            "Restrict to solo execution",
            "Document Transformers v5 compatibility",
            "Provide defaults for serving"
          ],
          "source_summary": "Specifies nvidia model ID, vllm-node container, solo_only true, cluster_only false, and commented-out mods.",
          "summary": "Defines a recipe for serving Gemma4-26B-A4B-NVFP4 in NVFP4 quantization."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the base Gemma4-26B-A4B model using Google's repository, with no execution mode restrictions. It includes default settings for port, host, and tensor parallelism.",
          "file_path": ".litho/tree/repo/recipes/gemma4-26b-a4b.yaml",
          "importance_score": 0.57,
          "interfaces": [],
          "name": "gemma4-26b-a4b.yaml",
          "responsibilities": [
            "Configure FP8 Gemma4 serving",
            "Support both solo and cluster modes",
            "Provide default serving parameters"
          ],
          "source_summary": "Specifies google model ID, vllm-node container, cluster_only false, solo_only false, and defaults section.",
          "summary": "Defines a recipe for serving Gemma4-26B-A4B in online FP8 quantization."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe serves the GLM-4.7-Flash-AWQ-4bit model and requires an inference speed patch. It documents suboptimal vLLM implementation, noting ~40 t/s generation speed in both single node and cluster modes, with cluster only improving prompt processing.",
          "file_path": ".litho/tree/repo/recipes/glm-4.7-flash-awq.yaml",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "glm-4.7-flash-awq.yaml",
          "responsibilities": [
            "Configure AWQ GLM-4.7 serving",
            "Document performance expectations",
            "Require inference speed patch",
            "Provide defaults"
          ],
          "source_summary": "Contains model ID, recipe metadata, and extensive comments about performance limitations and the required patch.",
          "summary": "Defines a recipe for serving cyankiwi's AWQ quantized GLM-4.7-Flash model with performance caveats."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node-b12x",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "--exp-b12x",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the GLM 5.3 Flash model for cluster-only deployment with the B12X-optimized container. It sets tensor_parallel to 2 and includes pipeline parallelism and decoding defaults.",
          "file_path": ".litho/tree/repo/recipes/glm-5.3-flash.yaml",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "glm-5.3-flash.yaml",
          "responsibilities": [
            "Declare GLM-5.3 Flash serving",
            "Enable B12X-optimized stack",
            "Restrict to cluster execution",
            "Set TP/PP and decoding defaults"
          ],
          "source_summary": "Specifies local model ID, vllm-node-b12x container, --exp-b12x build arg, cluster_only flag, empty mods, and defaults.",
          "summary": "Defines a recipe for serving GLM-5.3-Flash-NVFP4 on a dual DGX Spark cluster using the B12X stack."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the thinkingmachines Inkling-Small model with NVFP4 quantization, requiring cluster mode with tensor parallelism across two nodes. It includes patches and memory-pressure handling via mods.",
          "file_path": ".litho/tree/repo/recipes/inkling-small-nvfp4.yaml",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "inkling-small-nvfp4.yaml",
          "responsibilities": [
            "Configure NVFP4 Inkling serving",
            "Enable MTP speculative decoding",
            "Require two-node cluster",
            "Apply patches and memory settings"
          ],
          "source_summary": "Specifies model, vllm-node container, cluster_only flag, and mods for required patches and memory pressure.",
          "summary": "Defines a recipe for serving Inkling-Small-NVFP4 on a two-node DGX Spark cluster with MTP speculative decoding."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node-b12x",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "--exp-b12x",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fes-weights",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe scales the MiMo-V2.6-Flash model to four nodes, leveraging B12X attention (DiffKV + fp8 KV), b12x MoE (MXFP4), and async checkpoint loading. It is intended for FES evaluation and supports both FES-staged weights and plain launches.",
          "file_path": ".litho/tree/repo/recipes/mimo-v2.6-flash-4x.yaml",
          "importance_score": 0.58,
          "interfaces": [],
          "name": "mimo-v2.6-flash-4x.yaml",
          "responsibilities": [
            "Configure 4-node MiMo deployment",
            "Enable b12x optimized kernels",
            "Support FES evaluation weights",
            "Set KV cache dtype strategies"
          ],
          "source_summary": "Specifies TP=4, b12x build flags, KV cache dtype skip layers for fp8 KV, and references fes-eval scripts and mods.",
          "summary": "Defines a recipe for MiMo-V2.6-Flash-RL with TP=4 across four DGX Spark nodes using the b12x stack."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the 309B parameter MoE model (15B active) with MXFP4 experts and fp8 attention, using TP=2 across two nodes. It enables DFlash block-diffusion speculative decoding.",
          "file_path": ".litho/tree/repo/recipes/mimo-v2.6-flash.yaml",
          "importance_score": 0.57,
          "interfaces": [],
          "name": "mimo-v2.6-flash.yaml",
          "responsibilities": [
            "Configure MiMo-V2.6 Flash serving",
            "Enable MXFP4/fp8 quantization path",
            "Enable DFlash speculative decoding",
            "Set cluster tensor parallelism"
          ],
          "source_summary": "Specifies model, container, cluster configuration, and DFlash speculative decoding features.",
          "summary": "Defines a recipe for serving Xiaomi MiMo-V2.6-Flash-RL on a dual DGX Spark cluster with DFlash speculative decoding."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the AWQ quantized MiniMax M2 model for cluster-only execution using the standard vllm-node container, with no mods required.",
          "file_path": ".litho/tree/repo/recipes/minimax-m2-awq.yaml",
          "importance_score": 0.56,
          "interfaces": [],
          "name": "minimax-m2-awq.yaml",
          "responsibilities": [
            "Configure MiniMax-M2 AWQ serving",
            "Restrict to cluster execution",
            "Provide default server settings",
            "Declare no mods required"
          ],
          "source_summary": "Specifies QuantTrio model ID, container, cluster_only flag, empty mods, and defaults for port, host, and tensor parallelism.",
          "summary": "Defines a recipe for serving MiniMax-M2-AWQ on a multi-node cluster."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures cyankiwi's AWQ quantized MiniMax M2.5 model for cluster-only execution with the standard vllm-node container and no mods.",
          "file_path": ".litho/tree/repo/recipes/minimax-m2.5-awq.yaml",
          "importance_score": 0.56,
          "interfaces": [],
          "name": "minimax-m2.5-awq.yaml",
          "responsibilities": [
            "Configure MiniMax-M2.5 AWQ serving",
            "Restrict to cluster execution",
            "Provide default serving settings"
          ],
          "source_summary": "Specifies model ID, container, cluster_only flag, empty mods, and defaults for port, host, and parallelism.",
          "summary": "Defines a recipe for serving MiniMax-M2.5-AWQ on a multi-node cluster."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures cyankiwi's AWQ quantized MiniMax M2.7 model for cluster-only execution with the standard vllm-node container and no mods.",
          "file_path": ".litho/tree/repo/recipes/minimax-m2.7-awq.yaml",
          "importance_score": 0.56,
          "interfaces": [],
          "name": "minimax-m2.7-awq.yaml",
          "responsibilities": [
            "Configure MiniMax-M2.7 AWQ serving",
            "Restrict to cluster execution",
            "Provide default serving settings"
          ],
          "source_summary": "Specifies model ID, container, cluster_only flag, empty mods, and defaults for port, host, and parallelism.",
          "summary": "Defines a recipe for serving MiniMax-M2.7-AWQ on a multi-node cluster."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the NVFP4 Nemotron-3 Nano model with solo-only execution, noting that cluster mode currently fails with an error.",
          "file_path": ".litho/tree/repo/recipes/nemotron-3-nano-nvfp4.yaml",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "nemotron-3-nano-nvfp4.yaml",
          "responsibilities": [
            "Configure Nemotron-3 Nano NVFP4 serving",
            "Restrict to solo execution",
            "Document cluster mode limitation"
          ],
          "source_summary": "Specifies nvidia model ID, container, solo_only flag, and description warning about single-node restriction.",
          "summary": "Defines a recipe for serving Nemotron-3-Nano-30B-A3B-NVFP4 on a single node only."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "VLLM_FLASHINFER_ALLREDUCE_BACKEND",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe enables VLLM_CUTLASS for NVFP4 on the Nemotron-3-Super-120B model and sets FlashInfer allreduce backend environment variables. It supports both solo and cluster modes with defaults including tensor parallelism.",
          "file_path": ".litho/tree/repo/recipes/nemotron-3-super-nvfp4.yaml",
          "importance_score": 0.57,
          "interfaces": [],
          "name": "nemotron-3-super-nvfp4.yaml",
          "responsibilities": [
            "Configure Nemotron-3 Super NVFP4 serving",
            "Enable CUTLASS kernels",
            "Set FlashInfer allreduce backend",
            "Allow long max model length"
          ],
          "source_summary": "Specifies model, container, env vars (VLLM_FLASHINFER_ALLREDUCE_BACKEND, VLLM_ALLOW_LONG_MAX_MODEL_LEN), and defaults.",
          "summary": "Defines a recipe for serving Nemotron-3-Super-120B-A12B-NVFP4 using CUTLASS kernels with environment optimizations."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the Nemotron 3.5 Lightning model to run with TP=1 on solo DGX Spark and TP=2 in cluster mode, using NVFP4 quantization and the DSpark draft model for speculative decoding.",
          "file_path": ".litho/tree/repo/recipes/nemotron-3.5-lightning.yaml",
          "importance_score": 0.58,
          "interfaces": [],
          "name": "nemotron-3.5-lightning.yaml",
          "responsibilities": [
            "Configure Nemotron 3.5 Lightning serving",
            "Enable DSpark speculative decoding",
            "Support both solo and cluster modes",
            "Set TP and memory defaults"
          ],
          "source_summary": "Specifies model, container, and defaults with tensor_parallel and gpu_memory_utilization settings.",
          "summary": "Defines a recipe for serving NVIDIA Nemotron 3.5 Lightning 30B-A3B-NVFP4 with DSpark speculative draft model."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node-mxfp4",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "--exp-mxfp4",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures OpenAI's open source 120B MoE model on a solo node using the vllm-node-mxfp4 container with the --exp-mxfp4 build argument. It represents a notable open-weight model deployment.",
          "file_path": ".litho/tree/repo/recipes/openai-gpt-oss-120b.yaml",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "openai-gpt-oss-120b.yaml",
          "responsibilities": [
            "Configure GPT-OSS 120B serving",
            "Enable MXFP4 quantization build",
            "Restrict to solo execution",
            "Use FlashInfer backend"
          ],
          "source_summary": "Specifies openai model ID, mxfp4 container, solo_only flag, and build args for MXFP4 support.",
          "summary": "Defines a recipe for serving OpenAI GPT-OSS 120B with MXFP4 quantization and FlashInfer."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fix-qwen3-coder-next",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the Qwen3-Coder-Next-FP8 model with a commented-out fix mod referencing a vLLM GitHub issue for cluster performance problems. It includes default serving settings.",
          "file_path": ".litho/tree/repo/recipes/qwen3-coder-next-fp8.yaml",
          "importance_score": 0.57,
          "interfaces": [],
          "name": "qwen3-coder-next-fp8.yaml",
          "responsibilities": [
            "Configure Qwen3-Coder-Next FP8 serving",
            "Document cluster fix mod requirement",
            "Provide default settings"
          ],
          "source_summary": "Specifies model ID, container, commented solo_only flag, commented mods, and defaults.",
          "summary": "Defines a recipe for serving Qwen3-Coder-Next-FP8 with a mod to fix cluster slowness and crashes."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fix-qwen3-next-autoround",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures Intel's int4-AutoRound quantized Qwen3-Coder-Next model for solo-only execution, applying the fix-qwen3-next-autoround mod to address weight loading issues.",
          "file_path": ".litho/tree/repo/recipes/qwen3-coder-next-int4-autoround.yaml",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "qwen3-coder-next-int4-autoround.yaml",
          "responsibilities": [
            "Configure int4 AutoRound serving",
            "Restrict to solo execution",
            "Apply autoround weight loading fix"
          ],
          "source_summary": "Specifies Intel model ID, vllm-node container, solo_only flag, and mods/fix-qwen3-next-autoround.",
          "summary": "Defines a recipe for serving Qwen3-Coder-Next-int4-Autoround with a weight loading fix mod."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fix-qwen3.5-chat-template",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures Qwen3.5-122B in native FP8 quantization for cluster-only execution and applies the fix-qwen3.5-chat-template mod.",
          "file_path": ".litho/tree/repo/recipes/qwen3.5-122b-fp8.yaml",
          "importance_score": 0.57,
          "interfaces": [],
          "name": "qwen3.5-122b-fp8.yaml",
          "responsibilities": [
            "Configure Qwen3.5-122B FP8 serving",
            "Restrict to cluster execution",
            "Apply chat template fix mod",
            "Provide default settings"
          ],
          "source_summary": "Specifies model ID, container, cluster_only flag, mods list, and defaults for serving parameters.",
          "summary": "Defines a recipe for serving Qwen3.5-122B-A10B-FP8 with a chat template fix mod."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fix-qwen3.5-chat-template",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures Intel's INT4-AutoRound quantized Qwen3.5-122B model, applying the fix-qwen3.5-chat-template mod to address ROPE syntax errors. It represents a key quantization variant for the Qwen3.5 line.",
          "file_path": ".litho/tree/repo/recipes/qwen3.5-122b-int4-autoround.yaml",
          "importance_score": 0.58,
          "interfaces": [],
          "name": "qwen3.5-122b-int4-autoround.yaml",
          "responsibilities": [
            "Configure INT4 AutoRound serving",
            "Apply ROPE/chat template fixes",
            "Support cluster deployment",
            "Provide default settings"
          ],
          "source_summary": "Specifies model ID, container, commented solo_only flag, and mods including chat template fix.",
          "summary": "Defines a recipe for serving Qwen3.5-122B-A10B-INT4-Autoround with chat template and ROPE fixes."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fix-qwen3-coder-next",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures Qwen3.5-35B-A3B in native FP8 format, applying the fix-qwen3-coder-next mod to prevent cluster slowness and crashes tracked in a vLLM GitHub issue. It is a commonly used small MoE deployment.",
          "file_path": ".litho/tree/repo/recipes/qwen3.5-35b-a3b-fp8.yaml",
          "importance_score": 0.58,
          "interfaces": [],
          "name": "qwen3.5-35b-a3b-fp8.yaml",
          "responsibilities": [
            "Configure Qwen3.5-35B FP8 serving",
            "Apply cluster crash fix mod",
            "Provide default settings"
          ],
          "source_summary": "Specifies model ID, container, commented solo_only flag, mods list with fix-qwen3-coder-next, and defaults.",
          "summary": "Defines a recipe for serving Qwen3.5-35B-A3B-FP8 with cluster performance fix mods."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the large 397B INT4-AutoRound quantized model with important operational notes, including requiring memory utilization in GB, --no-ray for full context, and GPU clock limiting to avoid node shutdown.",
          "file_path": ".litho/tree/repo/recipes/qwen3.5-397b-int4-autoround.yaml",
          "importance_score": 0.52,
          "interfaces": [],
          "name": "qwen3.5-397b-int4-autoround.yaml",
          "responsibilities": [
            "Configure experimental 397B serving",
            "Document memory utilization requirements",
            "Recommend --no-ray mode",
            "Provide GPU clock guidance"
          ],
          "source_summary": "Includes model ID, description, and extensive operational comments about memory and GPU clock settings.",
          "summary": "Defines an experimental recipe for serving Qwen3.5-397B-INT4-Autoround with memory and GPU clock guidance."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fix-qwen3.6-chat-template",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the Qwen3.6-35B-A3B model in FP8 with the fix-qwen3.6-chat-template mod and DFlash decoding features. It is the speculative-decoding variant of the standard Qwen3.6 recipe.",
          "file_path": ".litho/tree/repo/recipes/qwen3.6-35b-a3b-fp8-dflash.yaml",
          "importance_score": 0.58,
          "interfaces": [],
          "name": "qwen3.6-35b-a3b-fp8-dflash.yaml",
          "responsibilities": [
            "Configure Qwen3.6-35B FP8 serving",
            "Apply chat template fix",
            "Enable DFlash speculative decoding",
            "Provide default settings"
          ],
          "source_summary": "Specifies model ID, container, mods list, and DFlash-related defaults.",
          "summary": "Defines a recipe for serving Qwen3.6-35B-A3B-FP8 with DFlash speculative decoding."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fix-qwen3.6-chat-template",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe is a commonly used deployment for the Qwen3.6-35B MoE model in FP8 format, applying the fix-qwen3.6-chat-template mod and setting TP=2 defaults. It represents a frequently referenced small MoE serving configuration.",
          "file_path": ".litho/tree/repo/recipes/qwen3.6-35b-a3b-fp8.yaml",
          "importance_score": 0.59,
          "interfaces": [],
          "name": "qwen3.6-35b-a3b-fp8.yaml",
          "responsibilities": [
            "Configure Qwen3.6-35B FP8 serving",
            "Apply chat template fix",
            "Set TP=2 and memory defaults",
            "Support cluster deployment"
          ],
          "source_summary": "Specifies model ID, container, mods list, and defaults including tensor_parallel and gpu_memory_utilization.",
          "summary": "Defines a recipe for serving Qwen3.6-35B-A3B-FP8 with a chat template fix mod."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the NVFP4 quantized Qwen3.6-35B model with the Marlin MoE backend, explicitly disabling MTP speculative decoding in favor of the Marlin path.",
          "file_path": ".litho/tree/repo/recipes/qwen3.6-35b-a3b-nvfp4-no-mtp.yaml",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "qwen3.6-35b-a3b-nvfp4-no-mtp.yaml",
          "responsibilities": [
            "Configure NVFP4 Qwen3.6 serving",
            "Enable Marlin MoE backend",
            "Disable MTP speculative decoding",
            "Provide default settings"
          ],
          "source_summary": "Specifies nvidia model ID, container, and defaults for the Marlin-based serving configuration.",
          "summary": "Defines a recipe for serving Qwen3.6-35B-A3B-NVFP4 with Marlin MoE backend and no MTP speculative decoding."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the NVFP4 quantized Qwen3.6-35B model using the Marlin MoE backend with tensor parallelism defaults.",
          "file_path": ".litho/tree/repo/recipes/qwen3.6-35b-a3b-nvfp4.yaml",
          "importance_score": 0.56,
          "interfaces": [],
          "name": "qwen3.6-35b-a3b-nvfp4.yaml",
          "responsibilities": [
            "Configure NVFP4 Qwen3.6 serving",
            "Enable Marlin MoE backend",
            "Provide TP and other defaults"
          ],
          "source_summary": "Specifies nvidia model ID, container, and defaults including tensor_parallel settings.",
          "summary": "Defines a recipe for serving Qwen3.6-35B-A3B-NVFP4 with Marlin MoE backend."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the Qwen3.8-27B NVFP4 model with the z-lab DFlash2 draft model, supporting TP=1 on solo and TP=2 on cluster with long context defaults.",
          "file_path": ".litho/tree/repo/recipes/qwen3.8-27b-nvfp4-dflash2.yaml",
          "importance_score": 0.57,
          "interfaces": [],
          "name": "qwen3.8-27b-nvfp4-dflash2.yaml",
          "responsibilities": [
            "Configure Qwen3.8-27B NVFP4 serving",
            "Enable DFlash2 speculative decoding",
            "Support both solo and cluster modes",
            "Set long context and memory defaults"
          ],
          "source_summary": "Specifies model ID, container, cluster_only/solo_only false, and defaults for TP, memory, max_model_len, and max_num_seqs.",
          "summary": "Defines a recipe for serving Qwen3.8-27B-NVFP4 with DFlash2 speculative decoding."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node-b12x",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "--exp-b12x",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the Qwen3.8-Flash-Next model for cluster-only deployment using the vllm-node-b12x container with --exp-b12x build args. It sets TP=2 with pipeline parallelism defaults.",
          "file_path": ".litho/tree/repo/recipes/qwen3.8-flash-next-nvfp4-cluster.yaml",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "qwen3.8-flash-next-nvfp4-cluster.yaml",
          "responsibilities": [
            "Configure Qwen3.8-Flash-Next cluster serving",
            "Enable B12X-optimized stack",
            "Set TP=2 and PP defaults",
            "Restrict to cluster execution"
          ],
          "source_summary": "Specifies local model ID, container, build args, cluster_only flag, empty mods, and defaults.",
          "summary": "Defines a recipe for serving Qwen3.8-Flash-Next-NVFP4 on a dual-node cluster with the B12X stack."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node-b12x",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "--exp-b12x",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the Qwen3.8-Flash-Next model for solo-only deployment using the B12X stack, offloading PLE (positional lookup embedding) tables to disk to fit the model on one node.",
          "file_path": ".litho/tree/repo/recipes/qwen3.8-flash-next-nvfp4-solo.yaml",
          "importance_score": 0.56,
          "interfaces": [],
          "name": "qwen3.8-flash-next-nvfp4-solo.yaml",
          "responsibilities": [
            "Configure Qwen3.8-Flash-Next solo serving",
            "Enable PLE disk offload",
            "Enable B12X-optimized stack",
            "Restrict to solo execution"
          ],
          "source_summary": "Specifies model ID, container, build args, solo_only flag, empty mods, and tensor_parallel defaults.",
          "summary": "Defines a recipe for serving Qwen3.8-Flash-Next-NVFP4 on a single node with PLE disk offload."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/step-3.7-flash",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/gpu-mem-util-gb",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the Step 3.7 Flash FP8 model for cluster-only deployment, applying the step-3.7-flash and gpu-mem-util-gb mods, with --no-ray mode recommended.",
          "file_path": ".litho/tree/repo/recipes/step-3.7-flash-fp8.yaml",
          "importance_score": 0.57,
          "interfaces": [],
          "name": "step-3.7-flash-fp8.yaml",
          "responsibilities": [
            "Configure Step-3.7 Flash FP8 serving",
            "Apply step-3.7 and memory mods",
            "Restrict to cluster execution",
            "Recommend --no-ray mode"
          ],
          "source_summary": "Specifies stepfun-ai model ID, container, cluster_only flag, mods list, and defaults.",
          "summary": "Defines a recipe for serving Step-3.7-Flash-FP8 on two or more Sparks with mods for memory management."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/step-3.7-flash",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This recipe configures the Step 3.7 Flash NVFP4 model for cluster-only deployment, applying only the step-3.7-flash mod without the memory mod.",
          "file_path": ".litho/tree/repo/recipes/step-3.7-flash-nvfp4.yaml",
          "importance_score": 0.56,
          "interfaces": [],
          "name": "step-3.7-flash-nvfp4.yaml",
          "responsibilities": [
            "Configure Step-3.7 Flash NVFP4 serving",
            "Apply step-3.7 mod",
            "Restrict to cluster execution",
            "Provide default settings"
          ],
          "source_summary": "Specifies stepfun-ai model ID, container, cluster_only flag, mods list, and defaults.",
          "summary": "Defines a recipe for serving Step-3.7-Flash-NVFP4 on two or more Sparks."
        }
      ],
      "importance_score": 0.8,
      "key_files": [
        "deepseek-v4-flash.yaml",
        "qwen3.5-122b-int4-autoround.yaml",
        "qwen3.6-35b-a3b-fp8.yaml",
        "gemma4-26b-a4b.yaml",
        "glm-5.3-flash.yaml"
      ],
      "name": "recipes",
      "path": ".litho/tree/repo/recipes",
      "purpose": "other",
      "subdirectory_count": 3,
      "summary": "This directory contains YAML recipe configuration files for serving various large language models (LLMs) on DGX Spark clusters using vLLM. Each recipe defines model selection, container images, build arguments, execution mode constraints (solo vs. cluster), mods/patches, and default serving parameters such as tensor parallelism, port, and memory utilization. The recipes work together as a declarative launch catalog, allowing different models and inference configurations to be consistently deployed via a common runner tool."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "nvidia-smi",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "Spark cluster runtime",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file is a configuration recipe that specifies how to deploy the Qwen3.5-397B-INT4-Autoround model on a 3-node GPU mesh using pipeline parallelism (PP=3). It includes operational notes such as setting memory utilization in GB rather than percentage, using --no-ray to fit full context on two Sparks, and instructions to limit GPU clocks via nvidia-smi if nodes shut down. It serves as the primary deployment artifact in this directory.",
          "file_path": ".litho/tree/repo/recipes/3x-spark-cluster/qwen3.5-397b-int4-autoround.yaml",
          "importance_score": 0.75,
          "interfaces": [],
          "name": "qwen3.5-397b-int4-autoround.yaml",
          "responsibilities": [
            "Define deployment recipe metadata (name, version, description) for the Qwen3.5-397B INT4 model",
            "Specify pipeline-parallel (PP=3) serving configuration across a 3-node Spark mesh",
            "Document memory utilization requirements (GB units, --no-ray flag) for fitting full context",
            "Provide operational guidance for GPU clock limiting to prevent node shutdowns"
          ],
          "source_summary": "The file starts with recipe metadata (recipe_version, name, description) identifying it as a pipeline-parallel recipe for a 3-node mesh. Comments document the model variant (Qwen3.5-122B/397B with Intel INT4-Autoround quantization) and operational caveats: memory utilization must be specified in GB, the --no-ray flag is required to fit full context on two Sparks, and GPU clocks may need to be limited (e.g., sudo nvidia-smi -lgc 200,2150) to prevent node shutdowns. The remainder of the recipe body is truncated in the provided content.",
          "summary": "Deployment recipe YAML for serving the Qwen3.5-397B model in INT4-Autoround quantization across a 3-node Spark cluster in pipeline-parallel mode."
        }
      ],
      "importance_score": 0.6,
      "key_files": [
        "qwen3.5-397b-int4-autoround.yaml"
      ],
      "name": "3x-spark-cluster",
      "path": ".litho/tree/repo/recipes/3x-spark-cluster",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains deployment recipes for running large quantized language models on a 3-node Spark GPU cluster. The single YAML file defines a pipeline-parallel (PP=3) serving configuration for the Qwen3.5-397B model in Intel INT4-Autoround quantization, including memory utilization settings and GPU clock-limiting guidance for node stability. It functions as a configuration layer for cluster-based LLM inference deployment."
    },
    {
      "file_count": 4,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "MiniMaxAI/MiniMax-M2.5",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines a cluster-only recipe (recipe_version 1) for deploying MiniMaxAI/MiniMax-M2.5 using the vllm-node container image. It specifies default serving parameters such as port 8000, host 0.0.0.0, tensor_parallel 4, and GPU memory utilization, all overridable via CLI. No mods are required for this model.",
          "file_path": ".litho/tree/repo/recipes/4x-spark-cluster/minimax-m2.5.yaml",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "config-field",
              "name": "recipe_version",
              "parameters": [],
              "return_type": "string",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "config-field",
              "name": "model",
              "parameters": [],
              "return_type": "string",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "config-field",
              "name": "defaults",
              "parameters": [],
              "return_type": "map",
              "visibility": ""
            }
          ],
          "name": "minimax-m2.5.yaml",
          "responsibilities": [
            "Declare the MiniMax-M2.5 model deployment recipe",
            "Specify container image and cluster-only constraint",
            "Define default serving parameters (port, host, TP=4, GPU memory utilization)",
            "Indicate no mods are required"
          ],
          "source_summary": "The YAML declares the recipe name, description, HuggingFace model ID (MiniMaxAI/MiniMax-M2.5), container image (vllm-node), and cluster_only: true flag. The defaults block sets port, host, tensor_parallel: 4, and gpu_memory_utilization settings for the vLLM server launch.",
          "summary": "Recipe configuration for serving MiniMax-M2.5 via vLLM on a 4-node cluster with tensor parallelism of 4."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/nemotron-ultra",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Configures vLLM serving of the 550B-parameter Nemotron-3-Ultra model in NVFP4 precision, distributed across four DGX Spark nodes with tensor parallelism of 4. It is intended for a no-ray multi-node launch and requires the nemotron-ultra mod to be applied. Uses the shared vllm-node container image.",
          "file_path": ".litho/tree/repo/recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml",
          "importance_score": 0.65,
          "interfaces": [
            {
              "description": null,
              "interface_type": "config-field",
              "name": "recipe_version",
              "parameters": [],
              "return_type": "string",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "config-field",
              "name": "model",
              "parameters": [],
              "return_type": "string",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "config-field",
              "name": "container",
              "parameters": [],
              "return_type": "string",
              "visibility": ""
            }
          ],
          "name": "nemotron-3-ultra-nvfp4.yaml",
          "responsibilities": [
            "Define deployment recipe for Nemotron-3-Ultra-550B NVFP4",
            "Configure TP=4 across 4 DGX Spark nodes",
            "Specify no-ray multi-node launch mode",
            "Require the nemotron-ultra mod"
          ],
          "source_summary": "The recipe header notes the NVFP4 quantized 550B model and TP=4 across 4 DGX Spark nodes with a no-ray multi-node launch strategy. It sets the HuggingFace model to nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4, container to vllm-node, and marks the recipe as cluster-only.",
          "summary": "Recipe for serving NVIDIA Nemotron-3-Ultra-550B-A55B in NVFP4 quantization with TP=4 across 4 DGX Spark nodes using a no-ray multi-node launch."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "Qwen/Qwen3.5-397B-A17B-FP8",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fix-qwen3.5-autoround",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Defines a vLLM serving recipe for Qwen/Qwen3.5-397B-A17B-FP8, a multi-modal input model in FP8 precision. It requires the mods/fix-qwen3.5-autoround mod to fix a ROPE syntax error, and uses the vllm-node container. A commented-out solo_only flag suggests the recipe may also be usable in single-node mode.",
          "file_path": ".litho/tree/repo/recipes/4x-spark-cluster/qwen3.5-397b-a17B-fp8.yaml",
          "importance_score": 0.6,
          "interfaces": [
            {
              "description": null,
              "interface_type": "config-field",
              "name": "recipe_version",
              "parameters": [],
              "return_type": "string",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "config-field",
              "name": "model",
              "parameters": [],
              "return_type": "string",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "config-field",
              "name": "mods",
              "parameters": [],
              "return_type": "list",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "config-field",
              "name": "defaults",
              "parameters": [],
              "return_type": "map",
              "visibility": ""
            }
          ],
          "name": "qwen3.5-397b-a17B-fp8.yaml",
          "responsibilities": [
            "Declare the Qwen3.5-397B FP8 deployment recipe",
            "Specify required mod for ROPE syntax fix",
            "Support multi-modal input model serving",
            "Define default serving parameters"
          ],
          "source_summary": "The YAML declares the recipe name, multi-modal FP8 model ID, and container image, with a mods list containing mods/fix-qwen3.5-autoround. The defaults block begins defining port and other overridable serving settings for the vLLM server.",
          "summary": "Recipe for serving the multi-modal Qwen3.5-397B-A17B model in FP8 precision via vLLM, requiring a mod to fix a ROPE syntax error."
        },
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "vllm-node",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "Qwen/Qwen3.5-397B (INT4-Autoround)",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fix-qwen3.5-autoround",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Configures vLLM serving of the Qwen3.5-397B model quantized with Intel INT4-Autoround, distributed with TP=4 across four DGX Spark nodes with a Marlin fix applied. The recipe documents benchmark results (37 tok/s single-user, 103 tok/s aggregate with 4 concurrent users) and warns that NVIDIA driver 580.x is required because 590.x has a CUDAGraph deadlock bug on GB10 hardware.",
          "file_path": ".litho/tree/repo/recipes/4x-spark-cluster/qwen3.5-397b-int4-autoround.yaml",
          "importance_score": 0.65,
          "interfaces": [
            {
              "description": null,
              "interface_type": "config-field",
              "name": "recipe_version",
              "parameters": [],
              "return_type": "string",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "config-field",
              "name": "model",
              "parameters": [],
              "return_type": "string",
              "visibility": ""
            },
            {
              "description": null,
              "interface_type": "config-field",
              "name": "defaults",
              "parameters": [],
              "return_type": "map",
              "visibility": ""
            }
          ],
          "name": "qwen3.5-397b-int4-autoround.yaml",
          "responsibilities": [
            "Declare the Qwen3.5-397B INT4-Autoround deployment recipe",
            "Document benchmark performance figures",
            "Specify NVIDIA driver version constraint (580.x)",
            "Apply Marlin fix and TP=4 across 4 DGX Spark nodes"
          ],
          "source_summary": "The header comments provide performance benchmarks and a critical driver compatibility note (580.x required, 590.x causes CUDAGraph deadlock on GB10). The recipe sets the name, description mentioning the Marlin fix, HuggingFace model reference, and default serving settings overridable via CLI.",
          "summary": "Recipe for serving Qwen3.5-397B in Intel INT4-Autoround quantization with TP=4 across 4 DGX Spark nodes, including benchmark data and a driver version constraint."
        }
      ],
      "importance_score": 0.6,
      "key_files": [
        "minimax-m2.5.yaml",
        "nemotron-3-ultra-nvfp4.yaml",
        "qwen3.5-397b-a17B-fp8.yaml",
        "qwen3.5-397b-int4-autoround.yaml"
      ],
      "name": "4x-spark-cluster",
      "path": ".litho/tree/repo/recipes/4x-spark-cluster",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "This directory contains vLLM serving recipe configurations for running large language models on a 4-node DGX Spark cluster. Each YAML file defines a model deployment recipe (MiniMax-M2.5, Nemotron-3-Ultra-NVFP4, and two Qwen3.5-397B variants) with settings for tensor parallelism, container images, mods, and default serving parameters. The files work together as a recipe catalog consumed by a cluster launch tool, enabling multi-node inference deployments across different model architectures and quantization formats."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "config",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "nvidia/GLM-5.2-NVFP4",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vllm-node-b12x",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "vLLM",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This file is a recipe configuration (recipe_version 1) that describes how to deploy the GLM-5.2-NVFP4 quantized model in a multi-node inference cluster. It specifies the model identifier, the container image (vllm-node-b12x), cluster-only mode, and a set of environment variables that tune vLLM runtime behavior for the DGX Spark (sm_121a) hardware. It acts as the declarative deployment definition consumed by an orchestration or recipe-runner tool.",
          "file_path": ".litho/tree/repo/recipes/8x-spark-cluster/glm-5.2-nvfp4.yaml",
          "importance_score": 0.6,
          "interfaces": [],
          "name": "glm-5.2-nvfp4.yaml",
          "responsibilities": [
            "Declare the model (nvidia/GLM-5.2-NVFP4) and serving recipe metadata for cluster deployment",
            "Specify the container image (vllm-node-b12x) and cluster-only execution mode for TP=8 across 8 DGX Spark nodes",
            "Configure vLLM runtime environment variables for long context, sparse indexer, MLA gather, and AOT compilation",
            "Target the sm_121a GPU architecture via CUTE_DSL_ARCH for DGX Spark hardware"
          ],
          "source_summary": "The file defines a recipe named 'GLM-5.2-NVFP4 (TP=8)' with metadata fields (recipe_version, name, description), deployment settings (model: nvidia/GLM-5.2-NVFP4, container: vllm-node-b12x, cluster_only: true, empty mods list), and an env block containing vLLM tuning flags such as VLLM_ALLOW_LONG_MAX_MODEL_LEN, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB, VLLM_B12X_MLA_CKV_GATHER, CUTE_DSL_ARCH=sm_121a, and VLLM_USE_AOT_COMPILE. The content is truncated mid-way through the environment variable list.",
          "summary": "Deployment recipe YAML configuring vLLM serving of the nvidia/GLM-5.2-NVFP4 model with TP=8 across 8 DGX Spark nodes."
        }
      ],
      "importance_score": 0.55,
      "key_files": [
        "glm-5.2-nvfp4.yaml"
      ],
      "name": "8x-spark-cluster",
      "path": ".litho/tree/repo/recipes/8x-spark-cluster",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The '8x-spark-cluster' directory contains deployment recipe configuration for serving large language models on a multi-node DGX Spark cluster. Its single file, glm-5.2-nvfp4.yaml, defines a vLLM serving recipe that runs the nvidia/GLM-5.2-NVFP4 model with tensor parallelism (TP=8) distributed across 8 DGX Spark nodes, including environment variable tuning for model length, sparse indexing, MLA attention, and AOT compilation targeting the sm_121a GPU architecture."
    },
    {
      "file_count": 1,
      "file_insights": [
        {
          "code_purpose": "command",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "bash",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "run-recipe.sh",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "mods/fes-weights",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "offline hub shim",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "fes-eval.sh is a Bash CLI entrypoint for FES evaluation runs. It accepts optional --weights-root and --hub-model flags, validates the staged weights, and delegates the recipe execution to run-recipe.sh with the correct FES-specific wiring. This keeps weight verification and mod/hub setup out of individual recipe scripts and centralizes them in one repeatable command.",
          "file_path": ".litho/tree/repo/scripts/fes-eval.sh",
          "importance_score": 0.74,
          "interfaces": [
            {
              "description": null,
              "interface_type": "command",
              "name": "fes-eval.sh",
              "parameters": [
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--weights-root",
                  "param_type": "string"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "--hub-model",
                  "param_type": "string"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "recipe",
                  "param_type": "string"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "fes-slug",
                  "param_type": "string"
                },
                {
                  "description": null,
                  "is_optional": false,
                  "name": "run-recipe args",
                  "param_type": "string[]"
                }
              ],
              "return_type": "exit code",
              "visibility": ""
            }
          ],
          "name": "fes-eval.sh",
          "responsibilities": [
            "Parse CLI arguments such as --weights-root, --hub-model, recipe, FES slug, and pass-through args",
            "Pre-verify the FES-staged weight copy before launching evaluation",
            "Wire the read-only weights mount, verification mod, and offline hub shim into the recipe run",
            "Delegate final execution to run-recipe.sh with the remaining arguments"
          ],
          "source_summary": "The script contains a shebang, detailed usage/example comments, argument parsing for the weights root, hub model, recipe, FES slug, and pass-through run-recipe arguments, and logic to pre-verify the staged FES weights. It then assembles the read-only mount, verification mod, and offline hub shim before executing run-recipe.sh with any additional arguments.",
          "summary": "Host-side companion script that pre-verifies FES-staged weight copies and launches recipes with the read-only weights mount, verify mod, and offline hub shim."
        }
      ],
      "importance_score": 0.66,
      "key_files": [
        "fes-eval.sh"
      ],
      "name": "scripts",
      "path": ".litho/tree/repo/scripts",
      "purpose": "tool",
      "subdirectory_count": 1,
      "summary": "The scripts directory contains host-side orchestration tooling for FES weight evaluation, with fes-eval.sh as the primary launcher. The script pre-verifies staged FES weights and then delegates the actual evaluation run to run-recipe.sh while wiring read-only weights mounts, verification mods, and the offline hub shim. This is a thin operational layer that supports evaluation workflows rather than core business logic, and is complemented by a subdirectory holding additional script tooling."
    },
    {
      "file_count": 2,
      "file_insights": [
        {
          "code_purpose": "tool",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "deepwiki-rs",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "LiteLLM proxy",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "git",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "use",
              "is_external": false,
              "line_number": null,
              "name": "shadow-tree.py",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "This script is the main entry point for generating deepwiki architecture docs. It accepts a target path and options, invokes deepwiki-rs, and writes the resulting markdown under deepwiki-rs-docs/ while keeping the .litho cache local-only and gitignored.",
          "file_path": ".litho/tree/repo/scripts/deepwiki/generate-deepwiki.sh",
          "importance_score": 0.55,
          "interfaces": [],
          "name": "generate-deepwiki.sh",
          "responsibilities": [
            "Parse path and options arguments",
            "Invoke deepwiki-rs (Litho) with the LiteLLM proxy backend",
            "Write generated docs into deepwiki-rs-docs/",
            "Keep .litho cache local-only and gitignored"
          ],
          "source_summary": "The script sets up and executes deepwiki-rs against a target subtree (defaulting to the whole repository), backed by a local LiteLLM proxy. It manages the output documentation directory and ensures the local cache does not get committed.",
          "summary": "Bash entry point that runs deepwiki-rs (Litho) to generate C4 architecture documentation, using a local LiteLLM proxy as the model backend."
        },
        {
          "code_purpose": "util",
          "dependencies": [
            {
              "dependency_type": "use",
              "is_external": true,
              "line_number": null,
              "name": "git",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "os",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "shutil",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "subprocess",
              "path": null,
              "version": null
            },
            {
              "dependency_type": "import",
              "is_external": true,
              "line_number": null,
              "name": "argparse",
              "path": null,
              "version": null
            }
          ],
          "detailed_description": "Because deepwiki-rs does not honor .gitignore, this script pre-filters the repository via git ls-files and creates hardlinks (or copies) of the tracked files into a temporary directory. The resulting shadow tree is a clean input for deepwiki-rs, ensuring documentation reflects only committed source code.",
          "file_path": ".litho/tree/repo/scripts/deepwiki/shadow-tree.py",
          "importance_score": 0.5,
          "interfaces": [
            {
              "description": null,
              "interface_type": "function",
              "name": "main",
              "parameters": [],
              "return_type": "None",
              "visibility": ""
            }
          ],
          "name": "shadow-tree.py",
          "responsibilities": [
            "Identify git-tracked files via git ls-files",
            "Create a scratch shadow-tree directory",
            "Materialize files as hardlinks with fallback to copies",
            "Provide a clean analysis input for deepwiki-rs"
          ],
          "source_summary": "main() computes the set of git-tracked files for a given subtree, creates a scratch directory, and populates it with hardlinks or copies of those files. It provides deepwiki-rs with a sanitized directory tree free of node_modules, coverage, and other gitignored output.",
          "summary": "Python utility that materializes exactly the git-tracked files into a scratch shadow tree using git ls-files, so deepwiki-rs excludes .gitignore'd build artifacts from its analysis."
        }
      ],
      "importance_score": 0.35,
      "key_files": [
        "generate-deepwiki.sh",
        "shadow-tree.py"
      ],
      "name": "deepwiki",
      "path": ".litho/tree/repo/scripts/deepwiki",
      "purpose": "other",
      "subdirectory_count": 0,
      "summary": "The deepwiki directory contains tooling scripts that generate AI-assisted C4 architecture documentation using deepwiki-rs (Litho) and a local LiteLLM proxy. The generate-deepwiki.sh script orchestrates the documentation run, while shadow-tree.py prepares a git-tracked-only shadow tree so that gitignored build artifacts do not pollute the analysis. Together they form a small but focused infrastructure pipeline for producing and maintaining architecture docs from the repository source."
    }
  ],
  "file_insights": []
}
```

## Memory Storage Statistics

**Total Storage Size**: 928899 bytes

- **studies_research**: 124549 bytes (13.4%)
- **timing**: 38 bytes (0.0%)
- **preprocess**: 441856 bytes (47.6%)
- **documentation**: 362456 bytes (39.0%)

## Generated Documents Statistics

Number of Generated Documents: 11

- Architecture Description
- Boundary Interfaces
- Core Workflows
- Key Modules and Components Research Report_Attention Kernel Domain
- Key Modules and Components Research Report_Container Build & Image Composition Domain
- Key Modules and Components Research Report_Deployment Recipes & Cluster Orchestration Domain
- Key Modules and Components Research Report_Developer Tooling Domain
- Key Modules and Components Research Report_Engine Patching & Model Compatibility Domain
- Key Modules and Components Research Report_Memory Profiling & Capacity Domain
- Key Modules and Components Research Report_Model Weights & Offline Serving Domain
- Project Overview
