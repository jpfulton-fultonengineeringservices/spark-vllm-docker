#!/bin/bash
set -euo pipefail

PREFIX="[fix-explicit-spec-method]"
info() {
  echo "$PREFIX $*"
}

EXEC_SCRIPT="/workspace/exec-script.sh"
if [ "${FIX_EXPLICIT_SPEC_METHOD:-0}" != "1" ] \
   && { [ ! -f "$EXEC_SCRIPT" ] || ! grep -q 'FIX_EXPLICIT_SPEC_METHOD' "$EXEC_SCRIPT" 2>/dev/null; }; then
  echo "$PREFIX inert (set FIX_EXPLICIT_SPEC_METHOD=1 to activate)"
  return 0 2>/dev/null || exit 0
fi

python3 - <<'PY'
import importlib.util

prefix = "[fix-explicit-spec-method]"
spec = importlib.util.find_spec("vllm.config.speculative")
if spec is None or spec.origin is None:
    raise SystemExit(f"{prefix} FAIL: cannot locate vllm.config.speculative")
with open(spec.origin) as fh:
    source = fh.read()

pristine = '''                if self.method in ("eagle", "eagle3", "dflash", "dspark"):
                    pass
'''
old_none_guard = '''                if self.method is None or self.method in ("eagle", "eagle3", "dflash", "dspark"):
                    pass
'''
old_tuple_guard = '''                if self.method in ("eagle", "eagle3", "dflash", "dspark", "mtp"):
                    pass
'''
guard = pristine + '''                elif self.method == "mtp" and (
                    self.draft_model_config.hf_config.model_type in get_args(MTPModelTypes)
                ):
                    pass
'''
safety = '''                    if (
                        self.target_model_config is not None
                        and self.target_model_config.hf_config.model_type
                        == "deepseek_v41"
                    ):
                        raise ValueError(
                            "DeepSeek V4.1 has no classic-MTP draft: its "
                            "checkpoints ship DSpark stages under mtp.* "
                            "(main_proj/markov_head/confidence_head) and carry "
                            "no e_proj/h_proj/enorm/hnorm/hc_head weights. Use "
                            "speculative method 'dspark' instead of 'mtp'."
                        )
                    if (
                        self.num_speculative_tokens > 1
                        and self.draft_model_config.hf_config.model_type
                        not in ("step3p5_mtp", "inkling_mtp")
                    ):
                        logger.warning(
                            "Enabling num_speculative_tokens > 1 will run "
                            "multiple times of forward on same MTP layer"
                            ",which may result in lower acceptance rate"
                        )
'''
mtp_branch = '''                elif self.draft_model_config.hf_config.model_type in get_args(
                    MTPModelTypes
                ):
                    self.method = "mtp"
'''
end_detection = '''                else:
                    raise NotImplementedError(
                        f"Unsupported speculative method: '{self.method}'"
                    )

'''
relocated_safety = '                if self.method == "mtp":\n' + safety + '\n'

if (source.count(guard) == 1 and source.count(end_detection + relocated_safety) == 1
        and source.count(mtp_branch + safety) == 0
        and source.count(relocated_safety) == 1
        and source.count(old_none_guard) == 0 and source.count(old_tuple_guard) == 0):
    compile(source, spec.origin, "exec")
    print(f"{prefix} Patch is already applied; skipping.")
else:
    candidates = (pristine, old_none_guard, old_tuple_guard)
    matches = [anchor for anchor in candidates if source.count(anchor) == 1]
    if (len(matches) != 1 or sum(source.count(a) for a in candidates) != 1
            or source.count(mtp_branch + safety) != 1
            or source.count(end_detection) != 1
            or source.count(guard) != 0
            or source.count(relocated_safety) != 0):
        raise SystemExit(
            f"{prefix} FAIL: incompatible or partially patched method-selection anchors "
            f"at {spec.origin}; re-derive the patch"
        )
    replacement = source.replace(matches[0], guard, 1)
    replacement = replacement.replace(mtp_branch + safety, mtp_branch, 1)
    replacement = replacement.replace(end_detection, end_detection + relocated_safety, 1)
    compile(replacement, spec.origin, "exec")
    with open(spec.origin, "w") as fh:
        fh.write(replacement)
    print(f"{prefix} Applied explicit MTP guard and retained MTP safety checks.")

print(f"{prefix} speculative.py compiles successfully at {spec.origin}.")
PY

info "Supported explicit MTP bypasses name inference; native MTP safety checks remain active."
