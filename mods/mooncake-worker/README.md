# mods/mooncake-worker

Ships the **host-staged** Mooncake store connector worker for the
`vllm-node-b12x` image, from the Fulton-Engineering-Services vLLM fork
branch `cuda13.3-aarch64-gb10-glm5next-modular` (tip `4e79d6be6f`,
"merge: rebase GB10/SM121 patches onto vLLM v0.28.0").

## Why

GB10 has no GPUDirect RDMA for NIXL-style GPU-VA registration: the store
connector's direct `store.register_buffer(gpu_addr)` fails with
`ibv_reg_mr` EFAULT ("Bad address [14]"). The Fulton fork's
`_StagingSlotPool` (ring-buffer staging slots, `a7d24d6392`) instead
D2H-copies GPU KV into pinned host slots and RDMA-writes from there —
the design proven in the qwen38 systemd deployments.

The installed image's connector (`e93769fd9b`, 2026-08-21) predates the
final staging fixes:
- `4fa2b5be07` allocate get pool on all host_staging roles
- `66621d73f1` forward *args in submit(), per-request slot sync
- `dc9ae4b8ac` reference GPU blocks for in-flight store jobs / store ledger by store_job_id (#52372)
- `4e79d6be6f` rebase onto vLLM v0.28.0 (the branch tip, this mod's source)

## What it ships

- `worker.py` — the `store/worker.py` from the branch tip (2574 lines,
  md5 `b11bd659baf724298b2f347dad503b4e`). Installed over the image's
  `/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/
  kv_connector/v1/mooncake/store/worker.py`.
- `run.sh` — fail-closed SHA-gated install: verify the installed file's
  md5 matches the known image version (`e93769fd9b` →
  `13aa4784d38f8c1ff0d9284d0033a335`) before replacing; skip if already
  ported; refuse on unknown base (image drift).

## Runtime requirements (shipped elsewhere)

- `mooncake-transfer-engine-cuda13==0.3.12.post1` python module —
  installed at boot by `mods/pd-disagg/run.sh` (connector imports
  `mooncake.store` in-process).
- `cupy-cuda13x<14.1.0` — required by `_StagingSlotPool`
  (`cupy.cuda` pinned allocs); installed by `mods/pd-disagg/run.sh`
  alongside the wheel. NOT shipped here to keep this mod single-purpose.

## Config (kv_connector_extra_config)

`mods/pd-disagg/dispatch.sh` sets:

```json
{"host_staging": true, "staging_num_slots": 2, "staging_slot_size_mb": 2048}
```

Budget-neutral vs the default single 4 GiB buffer (2×2048 MiB);
see nixl-gb10-optimization kilo plan `1787331814960-mooncake-store-staging-ring-buffer.md`.
