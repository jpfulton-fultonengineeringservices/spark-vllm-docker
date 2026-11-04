# Explicit Speculative Method Guard

**Last updated:** 2026-11-04

Runtime-only mod that preserves an explicitly configured
`--speculative-config method="mtp"` when the draft model name would otherwise
match the DFlash auto-detection branch.

## Defect (observed live 2026-11-04)

Serving Qwen3.8-27B-Coder390 with
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
crashed in `EAGLEConfig.__init__` with
`AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'`.

With no explicit draft model, MTP resolves the draft id from the target id.
The installed `SpeculativeConfig` auto-detect gate trusted only
`("eagle", "eagle3", "dflash", "dspark")`. The resulting draft id containing
`DFlash2` rewrote explicit `mtp` to `dflash`; `EAGLEConfig` then read
top-level `self.model.vocab_size`, while `Qwen3_5Config` declares only
`text_config.vocab_size`.

## Fix

Add only `"mtp"` to the trusted tuple:

```python
if self.method in ("eagle", "eagle3", "dflash", "dspark", "mtp"):
    pass
```

The inferred `None` -> `draft_model` default, `MTPModelTypes` inference,
DeepSeek V4.1 rejection, depth warning, and MTP validation branches are
unchanged. The earlier `self.method is None or ...` revision was a no-op for
explicit `mtp`; `run.sh` detects that exact old line and migrates it to the
fixed tuple. Any other anchor drift is a hard failure. Re-running on the
fixed tree is a no-op.

## Why the vocab_size pin was removed

The top-level `vocab_size:248320` in both MTP recipes was not a valid
workaround. On this image, `compose_draft_hf_overrides` returns
`SpeculativeConfig.hf_config_override` when target overrides are a dictionary:
dictionary overrides are target-only and are not applied to the draft config.
The unused `_apply_mtp_dict_hf_override` is not called by this path. MTP does
not wrap the draft in `EAGLEConfig`; the observed failure occurred only after
the method was incorrectly rewritten to `dflash`. The real vocabulary remains
at `text_config.vocab_size`.

## Verification (image and exact dependency)

Verified in a throwaway CPU configuration-only container, with no GPU worker
or serve process:

- image tag: `vllm-node-b12x:latest`
- image ID: `sha256:e9f6e32941ec5680a58edfbfefaad6997f0527e22344b5b505cf8267de4625f2`
- registry digest:
  `10.100.170.3:5000/spark-vllm@sha256:7bb070b92ab20cfbcfde07e7eb2e8913a0aaf4544db41701e726f7499e085377`
- installed vLLM: `0.32.0rc1.dev301+g1ac942555.d20261016`
- installed file: `/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py`
- codekeeper: `gilded-gnosis-vllm`, `dev/karmic-kraken`, revision
  `41d700e9-6b9a-4bce-ae22-db30d741576b`, commit
  `8244d1c44348ae8a5bbfc9f1b607021652283ed4`.

The live model mount and the same Hugging Face, vLLM, FlashInfer, B12X,
Triton, B12X-kcache, and TileLang cache mounts as the live container were
used. `run.sh` was applied, its import/AST checks passed, and a second apply
reported `Patch is already applied; skipping.` Observed `SpeculativeConfig`
outputs:

```text
EXPLICIT_NEUTRAL mtp 3 qwen3_5_mtp ['Qwen3_5MTP']
EXPLICIT_DFLASH2 mtp 3 qwen3_5_mtp ['Qwen3_5MTP'] top_vocab False text_vocab 248320
INFERRED None mtp 3 qwen3_5_mtp ['Qwen3_5MTP']
INFERRED 'draft_model' mtp 3 qwen3_5_mtp ['Qwen3_5MTP']
IMPORT_OK
```

The explicit DFlash2 case used a local path ending in
`local-inference-lab/Qwen3.8-27B-Coder390-DFlash2`, with the live checkpoint
config but no weight loading. Both inferred cases used `/tmp/NeutralDraft`,
avoiding the unrelated DFlash2 name inference. All cases used
`num_speculative_tokens=3` and the target's `mtp_num_hidden_layers=1`
override.

The checked-in `git apply` patch also applied in the installed
`dist-packages` root, passed `PATCH_AST_IMPORT_OK` and `PATCH_IMPORT_OK`, and
reverse-applied. A subsequent migration test from the exact old `None`-guard
line reported `Migrating ineffective 2026-11-04 None-guard revision.` and
then passed the same idempotency check.
