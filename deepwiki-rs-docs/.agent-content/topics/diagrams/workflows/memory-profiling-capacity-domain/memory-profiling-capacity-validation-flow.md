# Workflow — Memory Profiling & Capacity Validation Flow

Profiles a model's startup memory on the target cluster, consolidates per-rank events into a profile card, and validates that the deployment fits available host memory to guide topology-aware host selection.

```mermaid
flowchart LR
    step_0["1. Install and activate the startup instrumentation plugin inside vLLM to record per-phase memory during worker startup."]
    step_1["2. Collect per-rank event JSONL from run directories and merge into a model/recipe YAML profile card."]
    step_0 --> step_1
    step_2["3. Estimate startup RAM and KV-cache budget, validate against available host memory per topology, and emit capacity metrics."]
    step_1 --> step_2
    step_3["4. Probe host hardware and generate the human-readable capacity report."]
    step_2 --> step_3
    step_4["5. Use capacity findings to choose topology-appropriate hosts when launching the cluster deployment."]
    step_3 --> step_4
```

[← Back to Workflow](../../../workflow.md)
