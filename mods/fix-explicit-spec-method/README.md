# Explicit Speculative Method Guard

**Last updated:** 2026-11-04

Runtime-only mod that makes an explicitly configured `--speculative-config`
`method` win over the draft-model-name auto-detect in
`vllm/config/speculative.py` (`SpeculativeConfig.__post_init__`).

## Defect (observed live 2026-11-04)

Serving the Qwen3.8-27B-Coder390 checkpoint with
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
crashed inside `SpeculativeConfig` -> `EAGLEConfig.__init__` with
`AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'`.

Chain: explicit `method="mtp"` with no `model` resolves the draft from the
target id; the auto-detect gate only skips detection for
`("eagle", "eagle3", "dflash", "dspark")` -- `mtp` is NOT in that set, so
detection re-runs and the draft id containing `DFlash2` rewrites the method
to `dflash`; the dflash path then wraps the draft config in `EAGLEConfig`,
whose `__init__` reads `self.model.vocab_size` on a `Qwen3_5Config` that
only declares `text_config.vocab_size`. Explicit `method="mtp"` never
reached the MTP path at all.

This matches upstream vLLM's documented contract for the `method` parameter:
"A configured method is never changed based on the draft checkpoint" --
detection is for UNCONFIGURED methods.

## Fix

One-line guard: auto-detect runs only when `method is None` (or is one of
the already-handled explicit methods, which are no-ops there).

## Relation to the vocab_size recipe pin

`recipes/qwen3.8-27b-coder390-4x-store-l2-mtp.yaml` also pins
top-level `"vocab_size":248320` in `--hf-overrides`. That pin is still
correct and load-bearing: proven in-container, `EAGLEConfig` on this build
reads `self.model.vocab_size` before any method dispatch, so ANY method
wrap over a nested `text_config.vocab` config needs it. With this mod
applied, the MTP recipe stops reaching `EAGLEConfig` entirely -- but the
pin keeps the dflash sibling robust if it is ever relaunched on a build
whose eagle.py reads vocab unconditionally.

## Interaction with other mods

Unrelated to `mods/fix-eagle-fine-prefix` (scheduler/KV-cache prefix
granularity for EAGLE/MTP on hybrid models). Both may be applied together;
that mod remains recommended for MTP serving quality on Qwen3.5 hybrids
per its README.
