#!/bin/bash
# [qwen3.8-coder390-mtp] Boot-time guard for the bundled BF16 MTP head used by
# the MTP speculative-decode variant
# (recipes/qwen3.8-27b-coder390-4x-store-l2-mtp.yaml).
#
# Unlike mods/qwen3.8-coder390-dflash there is NOTHING TO STAGE: method "mtp"
# resolves the draft head from the main checkpoint itself -- the 15 mtp.*
# BF16 tensors live in model-00003-of-00003.safetensors of the very tree
# mods/fes-weights mounts at /model, and the recipe's --speculative-config
# carries no model path.
#
# This mod does three things:
#
# 1. MTP-presence assertion: the weight index must carry mtp.* tensors and
#    those tensors must be BF16 (a quantized-head load would corrupt drafts),
#    and config.json must declare mtp_num_hidden_layers (top-level or under
#    text_config -- the checkpoint ships text_config.mtp_num_hidden_layers=1;
#    the recipe additionally pins it top-level via --hf-overrides so depth
#    resolution does not depend on the engine's speculative.py vintage).
#    Any miss is a hard FAIL: serving without the head is worse than not
#    serving.
#
# 2. modelopt_mixed quantization exemption for mtp.*: the checkpoint already
#    declares quantization.exclude_modules ["mtp*","mtp.layers.0*"] in
#    hf_quant_config.json (vLLM's ModelOpt config file, per
#    get_config_filenames) and "ignore" with the same entries in config.json's
#    quantization_config; no mtp.* prefix appears in quantized_layers. vLLM
#    honors these natively -- ModelOpt exclude lists are wildcards matched
#    with fnmatch in ModelOptQuantConfigBase.is_layer_excluded, so unlike the
#    DFlash2 drafter's SGLang-shaped modules_to_not_convert there is no
#    naming-space rewrite to perform. This mod VERIFIES the exemption in the
#    SERVED copies (the fes-weights hub shim's snapshot first, then /model)
#    and, should a future re-quantization drop it, injects the wildcard
#    entries into the shim's hf_quant_config.json (replacing the symlink with
#    a real patched file -- /model itself is read-only).
#
# 3. The same publisher sampling envelope + thinking_token_budget gate as the
#    dflash mod (publisher-confirmed sampling defaults; identical values,
#    idempotent against the dflash mod's mark).

set -euo pipefail

PREFIX="[qwen3.8-coder390-mtp]"
MODEL_DIR="${FES_WEIGHTS_DIR:-/model}"
HUB_MODEL="${FES_HUB_MODEL:-}"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "$PREFIX FAIL: no checkpoint config at $MODEL_DIR/config.json" >&2
    exit 1
fi

echo "=== qwen3.8-coder390-mtp mod ==="

python3 - "$MODEL_DIR" "$HUB_MODEL" <<'PY'
import glob
import json
import os
import struct
import sys

model_dir, hub_model = sys.argv[1], sys.argv[2]
PREFIX = "[qwen3.8-coder390-mtp]"


def fail(msg):
    print(f"{PREFIX} FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# --- 1a. Weight index carries mtp.* tensors -------------------------------
index_path = os.path.join(model_dir, "model.safetensors.index.json")
if not os.path.isfile(index_path):
    fail(f"no weight index at {index_path}")
with open(index_path) as fh:
    weight_map = json.load(fh)["weight_map"]
mtp_entries = {k: v for k, v in weight_map.items() if k.startswith("mtp.")}
if not mtp_entries:
    fail(
        "weight index carries no mtp.* tensors -- the bundled MTP head is "
        "absent from this checkpoint; the MTP recipe cannot serve"
    )

# --- 1b. The mtp.* tensors are BF16 ----------------------------------------
# Parse the safetensors header (8-byte little-endian length + JSON) of every
# shard that holds an mtp tensor.
dtypes = {}
for shard in sorted(set(mtp_entries.values())):
    shard_path = os.path.join(model_dir, shard)
    with open(shard_path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
    for name in mtp_entries:
        if mtp_entries[name] == shard:
            dtypes[name] = header[name]["dtype"]
bad = {k: v for k, v in dtypes.items() if v != "BF16"}
if bad:
    fail(f"mtp.* tensors are not BF16: {bad}")
print(f"{PREFIX} MTP head present: {len(mtp_entries)} mtp.* tensors, all BF16")

# --- 1c. mtp_num_hidden_layers declared ------------------------------------
with open(os.path.join(model_dir, "config.json")) as fh:
    cfg = json.load(fh)
tc = cfg.get("text_config") or {}
n_mtp = cfg.get("mtp_num_hidden_layers", tc.get("mtp_num_hidden_layers"))
if not isinstance(n_mtp, int) or n_mtp < 1:
    fail(
        "config.json declares no mtp_num_hidden_layers (top-level or "
        "text_config); SpeculativeConfig cannot size the MTP draft"
    )
print(f"{PREFIX} mtp_num_hidden_layers={n_mtp} declared in config.json")


# --- 2. Quantization exemption ----------------------------------------------
def covers_mtp(entries):
    """True when an exclude/ignore list exempts every mtp.* module prefix.

    vLLM matches ModelOpt exclude entries with fnmatch (wildcards cross
    dots), and expands a trailing 'path*' into 'path' + 'path.*' in
    apply_vllm_mapper. Accept the checkpoint's own spellings plus the
    expanded forms.
    """
    accepted = {"mtp", "mtp.*", "mtp*", "mtp.layers.0*"}
    return any(e in accepted for e in entries)


def patch_file(path, section_get, key):
    with open(path) as fh:
        doc = json.load(fh)
    section = section_get(doc)
    entries = section.get(key) or []
    if covers_mtp(entries):
        return False
    entries = list(entries) + ["mtp*", "mtp.layers.0*"]
    section[key] = entries
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    return True


targets = []

# Prefer the fes-weights hub shim's snapshot copies: those are the files
# `vllm serve <id>` actually resolves, and the shim dir is writable (the mod
# runs as root on node1 before perms are handed to the serving user -- see
# mods/fes-weights/run.sh).
if hub_model:
    org, _, name = hub_model.partition("/")
    hub_root = os.path.join(os.environ.get("HF_HOME", "/root/.cache/huggingface"), "hub")
    repo_dir = os.path.join(hub_root, f"models--{org}--{name}")
    for snap in sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*"))):
        for fname in ("hf_quant_config.json", "config.json"):
            p = os.path.join(snap, fname)
            if os.path.exists(p):
                targets.append(p)

if not targets:
    # Fall back to the mount itself (writable only in non-shim layouts).
    for fname in ("hf_quant_config.json", "config.json"):
        p = os.path.join(model_dir, fname)
        if os.path.isfile(p):
            targets.append(p)

if not targets:
    fail("no served hf_quant_config.json/config.json found to verify")

verified = []
patched = []
for path in targets:
    base = os.path.basename(path)
    if base == "hf_quant_config.json":
        section_get = lambda d: d.setdefault("quantization", {})
        key = "exclude_modules"
    else:
        section_get = lambda d: d.setdefault("quantization_config", {})
        key = "ignore"
    with open(path) as fh:
        doc = json.load(fh)
    entries = section_get(doc).get(key) or []
    if covers_mtp(entries):
        verified.append(path)
        continue
    # Replace a shim symlink with a real file before writing (the link target
    # lives in the read-only /model mount).
    if os.path.islink(path):
        os.unlink(path)
        with open(os.path.join(model_dir, base)) as fh:
            doc = json.load(fh)
        section = section_get(doc)
        section[key] = list(section.get(key) or []) + ["mtp*", "mtp.layers.0*"]
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        patched.append(path)
    elif os.access(path, os.W_OK):
        if patch_file(path, section_get, key):
            patched.append(path)
    else:
        fail(f"mtp.* not exempted in {path} and the file is not writable")

for p in verified:
    print(f"{PREFIX} mtp.* quant-exemption verified: {p}")
for p in patched:
    print(f"{PREFIX} mtp.* quant-exemption INJECTED into: {p}")
PY

# --- Force publisher sampling envelope at the sampling-param layer ---
# This checkpoint ships generation_config.json with do_sample=true,
# temperature=1.0, top_p=0.95, top_k=20 and NO repetition penalty.
# Agent/tool-calling clients send temperature=0 (greedy), which is out of the
# validated envelope and triggers the "fails to stop" reasoning degeneracy
# documented in docs/qwen38-coder390-degenerate-repetition.md.
#
# Force the publisher's envelope at the sampling-param layer
# (to_sampling_params) so it applies to ALL callers - LiteLLM, direct :8000,
# internal tool calls - always, regardless of client inputs. The sampling-param
# layer is the single point where client request fields meet server defaults
# before the Sampler sees them; clamping at the business/tool-call layer is
# partial and leaves other entry paths unprotected.
#
# Idempotent: re-running on an already-patched wheel is a no-op (MARK check).
python3 - <<'PY'
import importlib.util
import sys

MARK = "[qwen3.8-coder390-mtp] Force publisher sampling envelope"
LEGACY_MARK = "[qwen3.8-coder390-dflash] Force publisher sampling envelope"
ANCHOR = (
    "            for name, default in self._DEFAULT_SAMPLING_PARAMS.items()\n"
    "        }\n"
    "\n"
    "        # Merge server-default stop_token_ids"
)
INJECT = (
    "            for name, default in self._DEFAULT_SAMPLING_PARAMS.items()\n"
    "        }\n"
    "\n"
    "        # [qwen3.8-coder390-mtp] Force publisher sampling envelope:\n"
    "        # temperature=1.0, top_p=0.95, top_k=20 always, regardless of\n"
    "        # client inputs. The checkpoint ships do_sample=true with no\n"
    "        # repetition penalty; greedy (temperature=0) from agent/tool-calling\n"
    "        # clients is outside the validated envelope and triggers the\n"
    "        # 'fails to stop' reasoning degeneracy\n"
    "        # (docs/qwen38-coder390-degenerate-repetition.md).\n"
    '        sampling_params["temperature"] = 1.0\n'
    '        sampling_params["top_p"] = 0.95\n'
    '        sampling_params["top_k"] = 20\n'
    "        # [qwen3.8-coder390-mtp] Anti-repetition: 1.10 is the floor\n"
    "        # proven (commit 87c40e8fbd) to break the reasoning-channel loop\n"
    "        # deterministically while leaving prose length intact (1.15\n"
    "        # measurably shortens it). Set in the dict: repetition_penalty\n"
    "        # reaches from_optional ONLY via **sampling_params (no explicit\n"
    "        # kwarg), and the comprehension has already captured the request\n"
    "        # value (None -> 1.0), so a self-assignment here would never flow\n"
    "        # through. A dict set is safe: the key exists in\n"
    "        # _DEFAULT_SAMPLING_PARAMS, so no duplicate-kwarg collision.\n"
    '        sampling_params["repetition_penalty"] = 1.10\n'
    "        # [qwen3.8-coder390-mtp] Reasoning-budget gate: this checkpoint's\n"
    "        # reasoning channel degenerates into unbounded repetition on\n"
    "        # near-duplicate context (repeated file reads, license headers,\n"
    "        # similar tool results) -- content=0 with the entire generation\n"
    "        # budget billed as reasoning_tokens. thinking_token_budget forces\n"
    "        # the </think> close when the budget exhausts, converting every\n"
    "        # loop into a completed answer (verified live 2026-11-04: baseline\n"
    "        # looped 512/512 reasoning with content=0; budget=64 produced\n"
    "        # finish=stop with 1041 chars of content).\n"
    "        # Ratio: reasoning-heavy -- 75% of the generation budget goes to\n"
    "        # reasoning, 25% reserved for output; a default budget of 32768\n"
    "        # applies when the caller sends no generation cap. Never below\n"
    "        # max_gen - 256 so output always retains at least 256 tokens.\n"
    "        # NOTE: assigned on `self`, NOT into sampling_params -- the return\n"
    "        # statement passes thinking_token_budget=self.thinking_token_budget\n"
    "        # explicitly, so a dict entry would raise 'multiple values for\n"
    "        # keyword argument'. `max_tokens` is the method parameter (the\n"
    "        # effective generation budget), not a sampling_params key.\n"
    "        _gen_budget = max_tokens\n"
    "        if _gen_budget is None:\n"
    "            self.thinking_token_budget = 32768\n"
    "        elif _gen_budget <= 340:\n"
    "            self.thinking_token_budget = 0\n"
    "        else:\n"
    "            self.thinking_token_budget = min(32768, max(int(_gen_budget) * 75 // 100, int(_gen_budget) - 256))\n"
    "\n"
    "        # Merge server-default stop_token_ids"
)

targets = []
for modname in (
    "vllm.entrypoints.openai.chat_completion.protocol",
    "vllm.entrypoints.openai.completion.protocol",
):
    try:
        spec = importlib.util.find_spec(modname)
    except ModuleNotFoundError:
        # vLLM itself is absent (e.g. running on a host, not in the serving
        # container) -- nothing to patch here.
        spec = None
    if spec is not None and spec.origin is not None:
        targets.append(spec.origin)

if not targets:
    print(
        "[qwen3.8-coder390-mtp] sampling-envelope: cannot locate protocol files",
        file=sys.stderr,
    )
    sys.exit(0)

for path in targets:
    with open(path) as f:
        src = f.read()
    if MARK in src or LEGACY_MARK in src:
        print(f"[qwen3.8-coder390-mtp] sampling envelope already forced: {path}")
        continue
    count = src.count(ANCHOR)
    if count != 1:
        print(
            f"[qwen3.8-coder390-mtp] sampling-envelope: anchor count={count} != 1 in {path}; skipping",
            file=sys.stderr,
        )
        continue
    src = src.replace(ANCHOR, INJECT, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"[qwen3.8-coder390-mtp] sampling envelope forced: {path}")
PY

