# Workflow — Recipe-Driven Cluster Model Deployment Flow

End-to-end path from selecting a model recipe to serving on a Spark cluster: parse recipe, verify offline weights, apply required fix/feature mods to the installed vLLM package, then launch head/worker nodes with the correct chat template and quantization configuration.

```mermaid
flowchart LR
    step_0["1. Select and parse a model/cluster recipe YAML (e.g., 4x-spark-cluster/qwen3.5-397b-int4-autoround.yaml) to determine mods, parallelism, and quantization."]
    step_1["2. Verify staged model weights and construct the HF hub-cache layout for offline serving."]
    step_0 --> step_1
    step_2["3. Validate shards against safetensors index, size parity, and quantization config; fail fast on mismatch."]
    step_1 --> step_2
    step_3["4. Apply model-specific patches and chat templates (AST rewrites, .patch/.diff files) to the installed vLLM/transformers sources."]
    step_2 --> step_3
    step_4["5. Run feature mods (DiffusionGemma support, zero-copy weights, KV-cache cleanup) that patch and then optionally launch the engine."]
    step_3 --> step_4
    step_5["6. Launch head and worker nodes across the Spark cluster with the patched container."]
    step_4 --> step_5
    step_6["7. At engine start, the FA4 dispatch patch routes SM12.x devices to the vendored paged-KV attention kernel."]
    step_5 --> step_6
```

[← Back to Workflow](../../../workflow.md)
