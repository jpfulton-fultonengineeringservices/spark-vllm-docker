# Explicit Speculative Method Guard

Runtime-only opt-in mod for an explicit MTP draft whose inherited model name
contains `DFlash2` (or another speculative-method name). Without the guard,
name inference can replace `method="mtp"` with the wrong method before the
MTP model-type branch is reached. For the Qwen3.8-27B-Coder390 checkpoint,
DFlash selection wraps the composite `Qwen3_5Config` in `EAGLEConfig`, which
reads a nonexistent top-level `vocab_size`; the vocabulary is in `text_config`.

The fix preserves explicit MTP when the draft's model type is an entry in
`MTPModelTypes`. It leaves name-based inference and unsupported draft types on
their original paths. The native DeepSeek V4.1 rejection and multi-token MTP
warning execute for both explicit and inferred MTP. No model weights or config
values are changed. In particular, this does not inject a top-level vocabulary.

`run.sh` requires `FIX_EXPLICIT_SPEC_METHOD=1` in its environment, or the flag
in `/workspace/exec-script.sh`. Mods run before native `launch-cluster.sh`
copies the exec script, so initial native launches must pass the flag as a
container environment argument. CCI passes recipe environment as container
`-e` arguments before mod application. The unified diff is for review/manual
application; the runner performs exact anchor replacements, fails closed on
incompatible or partial source, compiles the result before writing, and is
idempotent. It also migrates the exact earlier None-guard and tuple-only
revisions when their remaining anchors match; the tuple-only revision bypassed
the native MTP safety checks for explicitly specified MTP.

## Runtime evidence and limits

Read-only inspection on 2026-11-06 of `gx10-node1`'s `vllm_node` found image
content ID `sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02`,
registry digest `10.100.170.3:5000/spark-vllm@sha256:8e2b633962c089b2e79fd9077f4f4b48d7a39dd9b6b5a2b57a090c8498446f7c`,
and installed vLLM `0.1.dev21830+g8244d1c44.d20261102`. Its installed
`vllm/config/speculative.py` has the pristine method-selection anchors targeted
here. It contains `_apply_mtp_dict_hf_override` and uses it for explicit MTP
with dictionary overrides, preserving target YaRN overrides on the draft; the
older claim that this helper was absent does not apply to this artifact.

The live launch requested MTP with three speculative tokens and one bundled
MTP layer. Safetensors header inspection found 15 `mtp.*` tensors, all BF16,
and the checkpoint's ModelOpt exclusion lists include the MTP head. These are
read-only observations, not a serve or weight-loading test. The earlier
configuration-only proof on a different image applied an older tuple-only
patch; it does not validate this revised patch or its safety checks. A new
configuration-only proof must use an isolated copy of this image and read-only
model mount; it cannot establish GPU loading, generation, or Mooncake transfer.
