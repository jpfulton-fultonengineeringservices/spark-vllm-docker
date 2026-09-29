# Workflow — Model Support Enablement Flow

Onboarding a new or emerging model architecture: author a fix or feature mod (patch, diff, or chat template), register it in a recipe, and have the recipe runner apply it before the vLLM server starts with the corrected template/configuration.

```mermaid
flowchart LR
    step_0["1. Author model-support patches (e.g., diffusiongemma-support.patch, attention backend patches) and a run.sh orchestrator that applies them with legacy fallbacks."]
    step_1["2. Provide corrected chat templates or AST-based rewrites for model-specific serialization and quantization behavior."]
    step_0 --> step_1
    step_2["3. Reference the mod from a new recipe YAML describing model, quantization, and cluster size."]
    step_1 --> step_2
    step_3["4. Run the recipe so mods are applied in order and the server starts with the custom template."]
    step_2 --> step_3
```

[← Back to Workflow](../../../workflow.md)
