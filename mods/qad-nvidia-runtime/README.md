# qad-nvidia-runtime (PARKED)

Faithful NVIDIA serving path for the ModelOpt NVFP4 **QAD** checkpoint
`nvidia/Qwen3.8-Flash-Next-NVFP4-QAD`, staged on the NAS as
`/nas-1/models/qwen3.8/qwen3.8-flash-next-nvfp4-qad-nvidia`.

Nothing applies this mod: no recipe lists it in `mods:`, no `defaults`, no
`infra/cluster-config` map entry and no `cluster-control-infra` preset names it.
It is parked in the repository on purpose so NVIDIA's own serve command survives
as executable gate plus this document instead of living only in a design note.
`tests/test_qad_nvidia_runtime_mod.sh` enforces both halves: the gate must be
fail-closed, and **nothing in the build path may reference this mod** while it is
parked.

## The two QAD flavours

| flavour | artifact | `--worker-cls` | status |
| --- | --- | --- | --- |
| bootable | `recipes/fulton/qwen3.8-flash-next-nvfp4/qwen3.8-flash-next-nvfp4-qad-nvidia-2x.yaml`, `-4x.yaml` | no | registered; first-boot validation pending |
| faithful | this mod | `canonical_worker.CanonicalWorker` | parked, inert |

## Why the faithful path is parked (cite both hashes)

The checkpoint's `serving/runtime-manifest.json` pins

```
pinned_modelopt_runtime_sha256 = 41e00493d2770af19db03f585e9edb0004093c98b851d0f6726c2870cba11e22
```

and that hash **is** vLLM **v0.30.0**'s
`vllm/model_executor/layers/quantization/modelopt.py` (2656 lines, fetched from
GitHub, hashed locally). Every image we build today ships `modelopt.py` hashing
**`5dfedfbd…`** — recorded prefix, vLLM `0.1.dev21830+g8244d1c44` (2026-11-02) —
in both the upstream `vllm-node-b12x:latest` (image ID
`sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02`) and
the fork build
`ghcr.io/fulton-engineering-services/spark-vllm:vllm-node-b12x-20261106-0/1`
(`sha256:b76585f9020097bc9984b914bbcdebd5cdce5bbf66911ada28a9d7fa3649f096`).

`serving/canonical_worker.py` verifies that hash itself, so
`--worker-cls canonical_worker.CanonicalWorker` fails its own hash gate on every
image we build today. `run.sh` re-checks it from the outside and fails closed
**before launch**, so a half-enabled arm can never serve.

There is no vendor-provided runtime to install: `serving/` holds only
`canonical_worker.py`, `baseline_canonical_worker.py`, `indexer_order.py`,
`mxfp8_padding.py`, `README.md`, `runtime-manifest.json` and `LICENSE`. The only
supported route to a matching `modelopt.py` is a vLLM version change.

## Prerequisites for enablement (all three, from NVIDIA's own text)

1. **The hash gate.** Build the b12x image against the vLLM release whose
   `modelopt.py` hashes to the manifest pin — **vLLM `v0.30.0`**, i.e.
   `--build-arg VLLM_REF=v0.30.0`. Risk to re-check when that build exists: the
   b12x patch stack targets vLLM `main` and may not apply cleanly to the tag.
2. **The model implementation.** The helper imports require
   `vllm.models.qwen4_exp.nvidia` plus native MXFP8/NVFP4 ModelOpt linear
   kernels. `vllm.models.qwen4_exp.nvidia` *is* present in the current b12x
   image; the kernel counts the worker asserts are the fused per-rank counts in
   `runtime-manifest.json` (`MXFP8: 144`, `NVFP4_shared: 96`).
3. **PLE support.** The base architecture requires the PLE support of vLLM
   commit `d4d703caf908786416585ceb1f369e2e0363358b`; NVIDIA states that commit
   alone does not establish compatibility with this more aggressive recipe.

**Not gated by `run.sh` — verify these yourself before the first boot.** The
mod compares exactly one file: the installed `modelopt.py` against
`pinned_modelopt_runtime_sha256`. Everything else that must hold is visible in
the manifest but invisible to the gate, so `ok` is necessary, not sufficient:

- helper integrity — `runtime-manifest.json` also pins
  `files["canonical_worker.py"] =
  01c907d271aacf14322d0119aac9145050179c8dafe7c7790148a91c715baa9d` (equal to
  the staged file today, checked on gx10-node1 2026-11-18); a swapped or
  truncated helper passes the modelopt gate and then fails inside the worker;
- prerequisite 2 — `vllm.models.qwen4_exp.nvidia` must be importable in the
  image (`python3 -c 'import importlib.util,sys;
  sys.exit(0 if importlib.util.find_spec("vllm.models.qwen4_exp.nvidia") else 1)'`);
- prerequisite 3 — the PLE support of vLLM `d4d703caf…` must be in the build;
- the fused-module counts the helper asserts (`MXFP8: 144`, `NVFP4_shared: 96`
  per rank) can only fail at runtime, after launch.

NVIDIA's evaluated configuration was vLLM 0.30.0 on **GB300, TP=4** with expert
parallelism — not GB10/Spark. Enabling this on our nodes is a fresh validation,
not a port.

## The faithful command, verbatim from the checkpoint

Copied character-for-character from
`/nas-1/models/qwen3.8/qwen3.8-flash-next-nvfp4-qad-nvidia/serving/README.md`
(sha256 `417351a0a3b13996be9dabe7446ba803acf230ef20aed5cc3302fd687b6f6721`,
2681 bytes, read off gx10-node1 on 2026-11-18). Vendor file hashes for
reference: `runtime-manifest.json`
`f1331ad9c8a269b8bc40b2e6a250e46074aca49baae8c8c51d276b4a866fe06e`,
`canonical_worker.py`
`01c907d271aacf14322d0119aac9145050179c8dafe7c7790148a91c715baa9d` (the latter
equals the manifest's own entry for that file, so the staged tree is intact).

```bash
hf download nvidia/Qwen3.8-Flash-Next-NVFP4-QAD --local-dir ./Qwen3.8-Flash-Next-NVFP4-QAD
export PYTHONPATH="$PWD/Qwen3.8-Flash-Next-NVFP4-QAD/serving${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve ./Qwen3.8-Flash-Next-NVFP4-QAD \
--served-model-name qwen38-nvfp4-qad \
--tensor-parallel-size 4 \
--enable-expert-parallel \
--quantization modelopt \
--worker-cls canonical_worker.CanonicalWorker \
--dtype bfloat16 \
--kv-cache-dtype auto \
--max-model-len 262144 \
--max-num-seqs 256 \
--max-num-batched-tokens 8192 \
--gpu-memory-utilization 0.9 \
--enable-prefix-caching \
--enable-chunked-prefill \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--reasoning-parser qwen3 \
--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
--override-generation-config '{"max_new_tokens":null}' \
--trust-remote-code
```

The two `export` lines are exactly what `run.sh` exports, with the download dir
replaced by the `mods/fes-weights` read-only mount:
`PYTHONPATH=<FES_WEIGHTS_DIR>/serving` (default `/model/serving`) and
`VLLM_WORKER_MULTIPROC_METHOD=spawn`. Because the weights and the runtime come
from the same mounted tree, the worker class, the manifest and the weights can
never disagree.

Deliberate divergences from the bootable recipes, all inherited from the vendor
line rather than chosen: `--quantization modelopt` + `--dtype bfloat16` +
`--kv-cache-dtype auto` instead of the b12x stack's `modelopt_mixed` / `fp8`;
`--max-num-seqs 256` / `--gpu-memory-utilization 0.9` instead of the
family-validated `16` / `0.7`; no `--load-format b12x`, no `--block-size`, no
`--mamba-cache-mode`, no `--additional-config '{"ple_table_memory":"device"}'`.
**No `--speculative-config` in either flavour:** "MTP speculative decoding was
disabled… this QAD release does not claim to validate MTP with these helpers or
recover its original acceptance rate."

NVIDIA also notes the benchmark requests used `temperature=1.0`, `top_p=0.95`,
`top_k=20`, `reasoning_effort=xhigh`, thinking enabled, and that "this server
command alone does not reproduce the benchmarks" — so first-boot acceptance
numbers on our nodes are not comparable to NVIDIA's without those sampling
parameters.

## Gating behaviour

`run.sh` is inert unless `QAD_NVIDIA_RUNTIME=1` (calling env, or the rendered
`/workspace/exec-script.sh` references it — same activation pattern as
`mods/fix-mfu-metrics`). Once active it fails closed, in order, on:

1. `op=require_canonical_worker` — `serving/canonical_worker.py` absent;
2. `op=require_manifest` — `serving/runtime-manifest.json` absent;
3. `op=locate_modelopt` — installed vLLM package not found (`VLLM_PKG_DIR`
   overrides discovery, mirroring `VLLM_PACKAGE_ROOT` in
   `mods/mooncake-worker/run.sh`);
4. `op=read_pin` — manifest carries no `pinned_modelopt_runtime_sha256` (the
   value is read from the JSON, never hard-coded, so a vendor manifest bump
   needs no mod edit);
5. `op=gate_modelopt_runtime` — installed `modelopt.py` hash ≠ the manifest pin,
   naming the v0.30.0 image build as the remedy.

On success the `ok` line interpolates the live `PYTHONPATH` and
`VLLM_WORKER_MULTIPROC_METHOD`, so the log itself is evidence the exports
landed.
