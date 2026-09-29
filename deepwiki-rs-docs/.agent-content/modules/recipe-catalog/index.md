# Recipe Catalog

*module · agent map*

A declarative YAML recipe catalog that defines model, cluster, quantization, and parallelism configuration for serving large language models on Spark clusters. It serves as the primary user-facing deployment artifact, driving mod selection and cluster orchestration flows.

Recipes are YAML files with fields: recipe_version, name, description, model (HuggingFace or nvidia ID), container (image tag), build_args (optional), cluster_only/solo_only flags (boolean), mods (list of paths to mod run.sh scripts), and defaults (dictionary of vLLM server…

**Location:** [Home](../../index.md) › **Recipe Catalog**

## Diagrams

- [Flowchart](flowchart.md)
- [Sequence](sequence.md)

## Source

- [`.litho/tree/repo/recipes/deepseek-v4-flash-0731.yaml`](../../../../.litho/tree/repo/.litho/tree/repo/recipes/deepseek-v4-flash-0731.yaml)
- [`.litho/tree/repo/recipes/minimax-m2-awq.yaml`](../../../../.litho/tree/repo/.litho/tree/repo/recipes/minimax-m2-awq.yaml)
- [`.litho/tree/repo/recipes/qwen3.5-122b-fp8.yaml`](../../../../.litho/tree/repo/.litho/tree/repo/recipes/qwen3.5-122b-fp8.yaml)
- [`.litho/tree/repo/recipes/4x-spark-cluster/minimax-m2.5.yaml`](../../../../.litho/tree/repo/.litho/tree/repo/recipes/4x-spark-cluster/minimax-m2.5.yaml)
- [`.litho/tree/repo/recipes/8x-spark-cluster/glm-5.2-nvfp4.yaml`](../../../../.litho/tree/repo/.litho/tree/repo/recipes/8x-spark-cluster/glm-5.2-nvfp4.yaml)
- [`.litho/tree/repo/recipes/3x-spark-cluster/qwen3.5-397b-int4-autoround.yaml`](../../../../.litho/tree/repo/.litho/tree/repo/recipes/3x-spark-cluster/qwen3.5-397b-int4-autoround.yaml)
- [`.litho/tree/repo/mods/use-ngc-vllm/run.sh`](../../../../.litho/tree/repo/.litho/tree/repo/mods/use-ngc-vllm/run.sh)
- [`.litho/tree/repo/mods/use-official-vllm/run.sh`](../../../../.litho/tree/repo/.litho/tree/repo/mods/use-official-vllm/run.sh)
- [`.litho/tree/repo/examples/example-vllm-minimax.sh`](../../../../.litho/tree/repo/.litho/tree/repo/examples/example-vllm-minimax.sh)
- [`.litho/tree/repo/examples/vllm-glm-4.7-nvfp4.sh`](../../../../.litho/tree/repo/.litho/tree/repo/examples/vllm-glm-4.7-nvfp4.sh)
- [`.litho/tree/repo/run-recipe.sh`](../../../../.litho/tree/repo/.litho/tree/repo/run-recipe.sh)

## Interaction

- The Recipe Catalog provides YAML files that are parsed by Recipe Runners (run-recipe.sh/run-recipe.py). The runners select and apply mods listed in the recipe, then launch cluster head/worker nodes using launch-cluster.sh. The catalog interacts with the vLLM Flavor Selection…
