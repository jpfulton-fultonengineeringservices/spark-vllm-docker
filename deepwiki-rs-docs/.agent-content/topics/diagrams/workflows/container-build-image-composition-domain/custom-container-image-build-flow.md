# Workflow — Custom Container Image Build Flow

Builds the custom vLLM Docker image by installing vLLM and FlashInfer, applying the shared engine patch library, building FlashInfer JIT-cache provider wheels, and validating wheels before producing the final (base or MXFP4) image.

```mermaid
flowchart LR
    step_0["1. Invoke the Dockerfile build (base or MXFP4 variant) and copy resulting image artifacts."]
    step_1["2. Build FlashInfer JIT-cache provider wheels for the target CUDA architectures."]
    step_0 --> step_1
    step_2["3. Apply idempotent patches to installed vLLM/FlashInfer/B12X sources (SwiGLU-OAI plumbing, cache integrity, memory trims, top-k controls)."]
    step_1 --> step_2
    step_3["4. Validate built FlashInfer wheels and pin CUTLASS DSL versions for reproducibility."]
    step_2 --> step_3
    step_4["5. Vendor the SM120 FlashAttention kernel package into the image for later dispatch integration."]
    step_3 --> step_4
```

[← Back to Workflow](../../../workflow.md)
