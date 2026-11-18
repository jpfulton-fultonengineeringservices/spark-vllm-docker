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
| faithful | this mod | `canonical_worker.CanonicalWorker` | parked; hash gate cleared by the 2026-11-18 restamp, first boot pending |

## Why the faithful path was hash-blocked, and the 2026-11-18 restamp

The checkpoint's `serving/runtime-manifest.json` **as delivered** pinned

```
pinned_modelopt_runtime_sha256 = 41e00493d2770af19db03f585e9edb0004093c98b851d0f6726c2870cba11e22
```

and that hash **is** vLLM **v0.30.0**'s
`vllm/model_executor/layers/quantization/modelopt.py` (2656 lines, fetched from
GitHub, hashed locally). Every image we build ships `modelopt.py` hashing
**`5dfedfbdeda804aa8ff45d84f9d823bd8d45e13d4134a11d97012e05635d10d6`** — vLLM
`0.1.dev21830+g8244d1c44` (2026-11-02) — in the upstream
`vllm-node-b12x:latest` (image ID
`sha256:c34501ebfa793ce49408a72e559e70b765fd1e28e1f6012370a744635340cc02`), the
fork build
`ghcr.io/fulton-engineering-services/spark-vllm:vllm-node-b12x-20261106-0/1`
(`sha256:b76585f9020097bc9984b914bbcdebd5cdce5bbf66911ada28a9d7fa3649f096`) and
the newest `vllm-node-b12x-20261118` (`sha256:319205b2ff31…`).

`serving/canonical_worker.py:32-33` hashes the **installed**
`ModelOptLinearMethod` module and asserts it equals
`manifest["pinned_modelopt_runtime_sha256"]` — the same JSON field `run.sh`
reads. There is no second copy of the digest anywhere in the worker, so the
manifest is the single source of truth for both gates; `run.sh` only checks it
from the outside and fails closed **before launch**, so a half-enabled arm can
never serve.

There is no vendor-provided runtime to install: `serving/` holds only
`canonical_worker.py`, `baseline_canonical_worker.py`, `indexer_order.py`,
`mxfp8_padding.py`, `README.md`, `runtime-manifest.json` and `LICENSE`.

### Why "build against v0.30.0" was never a remedy (measured 2026-11-18)

The original enablement note named a vLLM version change as the only route. It
is not one: that ref satisfies the pin and has no loader to run.

| ref | `modelopt.py` | vs vendor pin | `model_loader/b12x_loader.py` | `model_executor/weight_transfer.py` | `model_loader/csf_scales.py` |
| --- | --- | --- | --- | --- | --- |
| `v0.30.0` | `41e00493…` (2656 L) | **exact** | absent | absent | absent |
| `v0.32.0` | `4c47880b…` (2750) | no | absent | absent | absent |
| `v0.33.0` | `e76a666b…` (2790) | no | absent | absent | absent |
| `v0.34.0` (newest tag) | `2e46ff1b…` (2911) | no | absent | absent | absent |
| `main` | `7bff3020…` (3283) | no | absent | absent | absent |
| `dev/karmic-kraken` | `5dfedfbd…` (2760) | no | **present** | present | absent |

No upstream tag ships the B12X loader at all, so an upstream-tagged image has
nothing to serve `--load-format b12x` with. Our own patch series cannot supply
it at `v0.30.0` either: on a full clone using the build's own
`git apply --3way --index --binary`, **12 of the 13** `docker/patches/qwen3.8/vllm`
patches fail and only `13-draft-vocab` applies. And a hash-faithful graft is
impossible by construction — the fork's file differs from v0.30.0's by
**+100/−5 lines with an identical class set**, and those 100 lines *are* the
b12x quant surface (`Nvfp4CsfState`/`Nvfp4CsfMoEMethod` for the `csf` and
`csf_32x2_v1` scale encodings, the `modelopt_iq2_xs` `IQ2_XS`/`IQ2_XXS`/`Q8_0`
block codecs, `select_mxfp8_moe_backend(…, prepares_b12x=True)`, the
`W4A16_NVFP4` zero-init calibrated scales, `workspace=` plumbing). The pin is
an exact hash, so matching it and keeping those paths are mutually exclusive.

### The restamp (operator-approved, 2026-11-18)

| | sha256 |
| --- | --- |
| vendor-delivered manifest | `f1331ad9c8a269b8bc40b2e6a250e46074aca49baae8c8c51d276b4a866fe06e` |
| restamped manifest | `bc8cb1bd7b40256924570b8ab95435ec27098fc8472db65154e4f17394045301` |
| backup, kept beside it | `runtime-manifest.json.orig-41e00493d2770af19db03f585e9edb0004093c98b851d0f6726c2870cba11e22` (hashes to `f1331ad9…`) |

The edit touches **exactly one key**, `pinned_modelopt_runtime_sha256`, plus an
additive `restamp` block recording the vendor value, the date and the reason.
That is safe rather than merely tidy: `canonical_worker.py:29-31` derives
`snapshot_sha256` and `snapshot` from `manifest["files"]` **alone** — the four
vendor `.py` files, which were not touched and still hash to their pinned
digests (all four re-verified on gx10-node1 2026-11-18, `canonical_worker.py`
= `01c907d2…`). The integrity chain over the vendor code is unbroken and a
pre-load replay of all four asserts in `load_model()` passes inside
`319205b2ff31` with the serving dir mounted.

What did change is whose claim the file makes. It no longer asserts "NVIDIA
evaluated this against vLLM v0.30.0"; it asserts "this deployment runs the
`dev/karmic-kraken` revision", and that attestation is **ours**, not the
vendor's. Re-downloading the checkpoint restores `f1331ad9…` and
`op=gate_modelopt_runtime` fails closed again — intended, not a regression:
re-apply the restamp deliberately rather than editing `serving/*.py`, which the
manifest pins and the worker verifies.

## Prerequisites for enablement (all three, from NVIDIA's own text)

1. **The hash gate.** Cleared on 2026-11-18 by the restamp above, not by an
   image change: `pinned_modelopt_runtime_sha256` now equals the installed
   `modelopt.py` hash in both `c34501ebfa79` and `319205b2ff31`, so
   `op=gate_modelopt_runtime` evaluates `ok`. The v0.30.0 route it used to name
   is unavailable — see the table above.
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

- helper integrity — `runtime-manifest.json` also pins the four vendor helpers,
  `files["canonical_worker.py"] =
  01c907d271aacf14322d0119aac9145050179c8dafe7c7790148a91c715baa9d` among them
  (all four still match the staged bytes after the restamp, checked on
  gx10-node1 2026-11-18); a swapped or truncated helper passes the modelopt
  gate and then fails inside the worker;
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
reference: `runtime-manifest.json` was delivered as
`f1331ad9c8a269b8bc40b2e6a250e46074aca49baae8c8c51d276b4a866fe06e` and is
`bc8cb1bd7b40256924570b8ab95435ec27098fc8472db65154e4f17394045301` since the
2026-11-18 restamp (the vendor bytes survive in the `.orig-41e00493…` backup and
in its own `restamp` block); `canonical_worker.py`
`01c907d271aacf14322d0119aac9145050179c8dafe7c7790148a91c715baa9d` — unmodified,
and equal to the manifest's own entry for that file, so the staged vendor tree
is intact.

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
5. `op=gate_modelopt_runtime` — installed `modelopt.py` hash ≠ the manifest
   pin. The failure names both remedies (re-stamp the manifest to the installed
   fork revision, or build against vLLM v0.30.0) and warns never to edit
   `serving/*.py`, whose hashes the manifest pins and the worker verifies.

On success the `ok` line interpolates the live `PYTHONPATH` and
`VLLM_WORKER_MULTIPROC_METHOD`, so the log itself is evidence the exports
landed.
