# MiMo-V2 aux-hidden-states-over-pipeline mod

Runtime mod that enables DFlash (and eagle3-family) speculative decoding for
MiMo-V2 when vLLM runs with pipeline parallelism (PP > 1). Required by
`3x-spark-cluster/mimo-v2.6-flash-pp3`, the only local MiMo recipe that runs
DFlash under PP. Two PP gaps are fixed here: the aux-hidden-state relay
opt-in and the drafter's input-embedding borrow.

## What it fixes (verified against vllm main, 2026-10-01)

### 1. Aux hidden states over PP

DFlash builds its draft context from target-model aux hidden states
(`target_layer_ids` in the checkpoint's `dflash/config.json`; for
MiMo-V2.6-Flash-RL that is `[0, 11, 23, 35, 47]`). Under PP those layers land
on every pipeline stage, so the engine must relay non-local aux states to the
last rank, where the drafter lives in full.

vLLM implements that relay generically
(`EagleModelMixin` + `pp_handler.configure_aux_hidden_state_relay`), but gates
it per model class behind `supports_aux_hidden_states_over_pp`
(`vllm/model_executor/models/interfaces.py`, default `False`). Only
`LlamaModel`, `Qwen2Model`, `DeepseekV4Model`, `KimiK3Model`, and
`MiniMaxM3Model` opt in today. `MiMoV2Model` does not, so engine startup with
PP>1 and `method: "dflash"` aborts with:

```
ValueError: MiMoV2FlashForCausalLM does not support dflash with pipeline parallelism
```

(raised from `verify_supports_aux_hidden_states_over_pp` in
`vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py`, called from the V2 GPU
model runner during setup).

The mod patches `vllm/model_executor/models/mimo_v2.py` to set
`supports_aux_hidden_states_over_pp = True` on `MiMoV2Model` — the same
one-line opt-in the supported classes carry.

### 2. Target input embedding on the drafter's PP stage

The DFlash drafter ships no `embed_tokens` weights of its own (its
`dflash/config.json` declares none), so `load_dflash_model` borrows the
target's embedding (`maybe_share_target_embed` in
`vllm/v1/worker/gpu/spec_decode/eagle/utils.py`). Under PP the target's
`embed_tokens` is a `PPMissingLayer` on every stage except the first — and the
drafter lives on the LAST stage, so drafter load aborts with:

```
RuntimeError: DFlashQwen3ForCausalLM needs the target input embedding,
but it is unavailable on this PP stage
```

(Observed live on the 3x PP=3 launch, 2026-10-01: PP rank 2 dies during
`speculator.load_model`; rank 0 then reports a gloo "Connection closed by
peer" cascade.)

The mod also patches `MiMoV2Model.__init__` to build `embed_tokens` on every
pipeline stage instead of only the first (and the last when
`tie_word_embeddings`). Every stage loads the same checkpoint weights, so the
copies are identical and the borrow on the last stage shares real weights.
Cost: ~1.25 GiB per extra stage (vocab 152576 x hidden 4096, bf16); only the
first stage's copy is used by the target forward, the last stage's copy is
shared into the drafter.

## Behavior

- Idempotent: already-patched regions are skipped, so repeat application in
  the same container is safe.
- Fails fast with a clear message when the installed vLLM no longer contains
  the expected `class MiMoV2Model(nn.Module, EagleModelMixin):` or embed-block
  anchors (layout drift after an image rebuild or fork rebase).
- Env: `VLLM_SITE_PACKAGES` / `PYTHON_ROOT` (default
  `/usr/local/lib/python3.12/dist-packages`).
- Runs after `mods/mimo-diffkv-fp8-kv` in the recipe's mod order; the mods
  patch disjoint regions of `mimo_v2.py`.

## Why the risk is bounded

Draft tokens are proposals only: the target model verifies every drafted token
against its own logits before acceptance. A misbehaving drafter (for example,
wrong aux states) shows up as a collapsed acceptance rate and slower decode,
not as wrong output. The target's PP forward path itself is ordinary vLLM
pipeline parallelism and does not depend on this mod.

## First-launch verification

After the first real PP=3 launch, confirm:

1. Startup log shows `Patch ... mimo_v2.py` (or the skip line) and no
   `does not support dflash with pipeline parallelism` error.
2. Output equivalence at `temperature=0` against a known-good TP=2 baseline
   (the same six-prompt probe used by `mods/mimo-diffkv-fp8-kv`).
3. Spec-decode health is sane (the 2x/4x fleet runs ~50% acceptance and ~4.5
   tokens per step at 7 draft tokens):

```promql
sum(rate(vllm:spec_decode_num_accepted_tokens_total{job="vllm-serve"}[10m]))
  / sum(rate(vllm:spec_decode_num_draft_tokens_total{job="vllm-serve"}[10m]))
```

Fallback if drafting misbehaves: drop `--speculative-config` from the recipe
command (pure PP=3 serving) to isolate drafter issues from target issues.

## Not covered here

- `dflash/config.json` trailing comma and drafter staging:
  `mods/mimo-v2.6-flash`.
- fp8 KV cache enablement on the DiffKV path: `mods/mimo-diffkv-fp8-kv`.

Test: `./tests/test_mimo_v2_aux_over_pp_mod.sh`.
