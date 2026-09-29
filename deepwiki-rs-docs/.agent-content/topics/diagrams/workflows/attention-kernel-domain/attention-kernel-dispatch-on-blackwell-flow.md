# Workflow — Attention Kernel Dispatch on Blackwell Flow

At inference startup, vLLM's FlashAttention 4 dispatch is patched so that compute-capability 12.x devices route attention work to the vendored paged-KV kernel, which selects architecture-specific forward/MLA/split-K kernels and exposes them through the PyTorch autograd interface.

```mermaid
flowchart LR
    step_0["1. 修补vLLM的NVIDIA FA4调度源码以注入缓存的SM12分页KV能力检查"]
    step_1["2. 验证输入并计算分块大小与拆分KV启发式策略 将内核封装为自动求导函数"]
    step_0 --> step_1
    step_2["3. 调度持久化工作分块并准备分页KV缓存布局"]
    step_1 --> step_2
    step_3["4. 执行架构选择的前向或反向内核 包括跨KV分块时的拆分K合并"]
    step_2 --> step_3
```

[← Back to Workflow](../../../workflow.md)
