# mods/mimo-v2.6-flash-rl-nvfp4

Serves the **NVFP4** (ModelOpt MIXED_PRECISION) export of `XiaomiMiMo/MiMo-V2.6-Flash-RL` on b12x vLLM runtime.

## Checkpoint provenance

- **Source model:** `XiaomiMiMo/MiMo-V2.6-Flash-RL-NVFP4` (FES-staged archive under `/nas-1/models/mimo/mimo-v2.6-flash-rl-nvfp4`).
- **Quantization:** `modelopt_mixed` (`quant_method: modelopt`, `quant_algo: MIXED_PRECISION`). Routed experts are NVFP4 (W4A4, 16-elem scale grid, on-disk `weight_scale`/`weight_scale_2`); dense/attention linears are FP8_PB_WO (on-disk block scale `weight_scale_inv`, 2-D).
- **DFlash:** Shipped with an identical EXL3-source DFlash drafter under `dflash/` (staged unchanged by `mods/mimo-v2.6-flash`).

## Why this mod is needed

The b12x image's `modelopt.py` natively maps `MIXED_PRECISION` to `modelopt_mixed`, NVFP4 routed experts to `ModelOptNvFp4FusedMoE`, and the FP8_PB_WO linears to `KFp8Block128` — which registers the module scale **param** as `weight_scale` (4-D `[out_blk,1,in_blk,1]`). The gap is in `models/mimo_v2.py`: its fused-qkv loader keys on the `weight_scale_inv` on-disk **name** and the `weight_scale_inv` **param** name, so the ModelOpt block scale (and the fused-qkv weight it guards) is silently dropped. Without the bridge the fused qkv_proj weight never loads → garbage output.

`mods/mimo-v2.6-flash-rl-nvfp4/run.sh` ports the checkpoint's shipped ModelOpt load bridge (`vllm_patch/mimo_v2.diff`) into the image's `vllm/model_executor/models/mimo_v2.py` via an anchor-gated, fail-loud inline python3 patcher.

## Important notes

- **Additive only:** The existing FP8 `weight_scale_inv` path is untouched. Every ModelOpt-only branch is guarded on `_mopt_ws` or `param.dim() == 4`, so EXL3 and pro-FP8 checkpoints continue to load and serve exactly as before.
- **No `exl3-mimo`:** `mods/exl3-mimo` is **not** applied to NVFP4 recipes.
- **TP=1 infeasibility:** NVFP4 routed experts are ~4.5 bpw (~170 GB weights alone), which exceeds a single GB10 node's ~122 GiB UMA weight cap (GB10 has no weight offload; experts cannot be paged). TP=1 is documented infeasible; serve exclusively via TP=2 (2-node non-PD or 4-node PD-disaggregated TP=2/TP=2) or TP=4.
