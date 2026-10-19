# Testing the B12X attention arm (Qwen3.8-Coder390)

Runbook for verifying the defect-5 fix in
[`b12x-coder390-attention-gap.md`](./b12x-coder390-attention-gap.md): B12X paged
attention on the hybrid Qwen3_5 model, where mamba-page alignment forces a KV
**storage** block (1664) above B12X's **kernel page** sizes (64/128).

Tests run cheapest-first. Each layer answers a question the next one can't.

All on-node access goes through the `fulton-remote-shell` MCP
(`shell_open` then `shell_exec`); never raw `ssh`.

---

## 0. Preflight

```bash
# Expect: no vllm containers, and ~115 GiB available (the serve holds the GPUs).
docker ps --format '{{.Names}} {{.Status}}' | grep vllm || echo 'no vllm'
free -g | head -2
```

If the production serve is running, see §6 — you must tear it down for §4.

---

## 1. Unit tests (CPU only, no GPU, seconds)

```bash
cd /cluster-shared/projects/spark-vllm-docker
python3 tests/test_b12x_storage_block_split_patch.py     # 9 cases
```

Expect `OK`. Also run the sibling patchers, since they share the
`docker/patch_vllm_*.py` import-by-path convention:

```bash
python3 tests/test_vllm_swa_block_size_patch.py
python3 tests/test_vllm_startup_heap_trim_patch.py
python3 tests/test_vllm_wsl_cuda_uma_patch.py
python3 tests/test_vllm_flashinfer_b12x_patch.py
```

Mutation check (confirms the suite isn't vacuous): remove the divisibility
clause from `supports_block_size` in the patcher and re-run — the suite must
fail. Restore afterwards.

**Answers:** patcher is idempotent, keeps `get_supported_kernel_block_sizes`
at exact `[64, 128]`, accepts 1664/2688, still rejects 1648.

---

## 2. Patcher against the real shipped `b12x.py`

Unit fixtures can drift from the image. Anchor against the actual file:

```bash
mkdir -p /tmp/ptest
docker run --rm --entrypoint bash vllm-node-b12x:latest \
  -c 'cat /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/b12x.py' \
  > /tmp/ptest/b12x_orig.py

cp docker/patch_vllm_b12x_storage_block_split.py /tmp/ptest/
cd /tmp/ptest && mkdir -p root/vllm/v1/attention/backends
cp b12x_orig.py root/vllm/v1/attention/backends/b12x.py

docker run --rm -v /tmp/ptest:/ptest --entrypoint bash vllm-node-b12x:latest -c '
  python3 /ptest/patch_vllm_b12x_storage_block_split.py /ptest/root   # applies
  python3 /ptest/patch_vllm_b12x_storage_block_split.py /ptest/root   # skips
  diff -u /ptest/b12x_orig.py /ptest/root/vllm/v1/attention/backends/b12x.py'
```

**Accept:** first run prints `Accepted splittable hybrid storage blocks in B12X`,
second prints `already present; skipping`, and the diff touches only the two
blocks (`supports_block_size`, impl `--block-size` guard).

If `diff` shows anything else, or the patcher raises `expected exactly one ...
block`, the upstream layout changed — stop and re-read `b12x.py`.

---

## 3. Both gates accept 1664, and the split reaches the kernel page

```bash
cat > /tmp/ptest/gates.py <<'PY'
from vllm.v1.attention.backends.b12x import B12xPagedAttentionBackend as B
from vllm.v1.worker.utils import select_common_block_size
print('supports_block_size(1664):', B.supports_block_size(1664))
print('supports_block_size(1648):', B.supports_block_size(1648))
print('supports_block_size(64)  :', B.supports_block_size(64))
print('kernel block for 1664:', select_common_block_size(1664, [B]))
print('kernel block for 2688:', select_common_block_size(2688, [B]))
PY

docker run --rm -v /tmp/ptest:/ptest --entrypoint bash vllm-node-b12x:latest -c '
  cp /ptest/root/vllm/v1/attention/backends/b12x.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/b12x.py
  python3 /ptest/gates.py'
```

**Accept:**
```
supports_block_size(1664): True
supports_block_size(1648): False     <- must stay False (1648 = 16*103, prime)
supports_block_size(64)  : True
kernel block for 1664: 128
kernel block for 2688: 128
```

Gate 2 (the impl guard) is exercised in §4's boot; to check it directly,
construct `B12xPagedAttentionImpl` with a config whose
`cache_config.block_size == 1664` and confirm no `ValueError`.

---

## 4. Live boot — the decisive test

This is the only layer that exercises `_find_non_ssm_backends`, the planner,
the remapped kernel block table, and CUDA-graph replay together.

### 4a. Stage the verification mod

> **Confirm before launching.** §4 occupies nodes 1–2 and the launch tears down
> whatever is already there, including a running production serve. Check
> `docker ps | grep vllm_node` and `curl -s localhost:8000/v1/models` first; if
> the production 1M-context serve is up, get explicit approval to stop it, or
> schedule the test for a maintenance window. §6 restores it afterwards.

Applies the patcher inside the container at boot, so no image rebuild is needed.

```bash
mkdir -p /tmp/b12x-split-mod
cp /cluster-shared/projects/spark-vllm-docker/docker/patch_vllm_b12x_storage_block_split.py \
   /tmp/b12x-split-mod/

cat > /tmp/b12x-split-mod/run.sh <<'SH'
#!/bin/bash
set -euo pipefail
D=$(dirname "$0")
echo "[b12x-storage-block-split] applying patcher to installed vLLM"
python3 "$D/patch_vllm_b12x_storage_block_split.py" --installed
python3 - <<'PY'
from vllm.v1.attention.backends.b12x import B12xPagedAttentionBackend as B
print('[b12x-storage-block-split] supports_block_size(1664)=', B.supports_block_size(1664))
PY
echo "=== OK ==="
SH
chmod +x /tmp/b12x-split-mod/run.sh
```

### 4b. Build the verification recipe

Derive from the real 2x recipe so the mamba-page geometry is genuine (TP=2, fp8
KV, prefix caching — *not* `max_model_len` — is what produces 1664). Shrink
context/seqs only to make the boot fast.

```bash
cd /tmp && python3 - <<'PY'
import pathlib
s = pathlib.Path('/cluster-shared/projects/spark-vllm-docker/recipes/qwen3.8-27b-coder390-2x.yaml').read_text()
assert '    --block-size 16 \\' in s
s = s.replace('    --block-size 16 \\', '    --attention-backend B12X \\\n    --block-size 64 \\')
s = s.replace('  max_model_len: 1048576', '  max_model_len: 262144')
s = s.replace('  max_num_seqs: 32', '  max_num_seqs: 8')
s = s.replace('  max_num_batched_tokens: 16384', '  max_num_batched_tokens: 4096')
s = s.replace('  max_cudagraph_capture_size: 256', '  max_cudagraph_capture_size: 32')
s = s.replace('  - mods/b12x-reclaim-gate', '  - mods/b12x-reclaim-gate\n  - /tmp/b12x-split-mod')
pathlib.Path('/tmp/b12x-verify-2x.yaml').write_text(s)
print('ok')
PY
```

### 4c. Render the launch script

`launch-cluster.sh --launch-script` needs a **file that already exists**; it
does not generate one. Produce it from the recipe:

```bash
cd /cluster-shared/projects/spark-vllm-docker
python3 run-recipe.py /tmp/b12x-verify-2x.yaml --dry-run -n 10.100.171.3,10.100.171.4 2>&1 \
  | awk '/^=== Generated Launch Script ===/{f=1;next} f&&/^=== /{exit} f' \
  > examples/b12x-verify.sh
grep -c 'attention-backend B12X' examples/b12x-verify.sh   # expect 1
```

### 4d. Launch (detached — the MCP transport caps at 240 s)

```bash
cd /cluster-shared/projects/spark-vllm-docker
rm -f /tmp/b12x-verify.log
setsid nohup bash -c 'LOCAL_IP=10.100.171.3 bash launch-cluster.sh \
  -t vllm-node-b12x \
  --image-id sha256:a1fc4f18d3cb1671986feb92b05243ecd31a21957dd64e28da5c9e6125733aa0 \
  --apply-mod mods/fes-weights --apply-mod mods/qwen3.8-coder390-dflash \
  --apply-mod mods/b12x-kernel-cache --apply-mod mods/b12x-reclaim-gate \
  --apply-mod /tmp/b12x-split-mod \
  -n 10.100.171.3,10.100.171.4 \
  -e FES_WEIGHTS_ENABLED=1 -e FES_WEIGHTS_DIR=/model \
  -e FES_HUB_MODEL=local-inference-lab/Qwen3.8-27B-Coder390-DFlash2 \
  -e HF_HUB_OFFLINE=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v /opt/llm/models/qwen3.8-27b-coder390-ethink-dflash2-nvfp4-w4a16:/model:ro \
  --eth-if enP2p1s0f1np1 --ib-if roceP2p1s0f1 \
  --launch-script examples/b12x-verify.sh' \
  </dev/null > /tmp/b12x-verify.log 2>&1 & disown
```

Expect both ranks to log `Accepted splittable hybrid storage blocks in B12X`
during mod application.

### 4e. Collect evidence

Do **not** use `docker logs vllm_node` — PID 1 is `sleep infinity` for this
launch path and the container log driver captures nothing. Read the host log.

Always bound the read: the log spans two ranks and keeps growing, so an
unbounded grep or `tail -f` can flood the terminal (and, through an agent tool,
the context window). Every command below is filtered *and* capped.

```bash
# wait for readiness
for i in $(seq 1 16); do
  c=$(curl -s -m 3 -o /dev/null -w '%{http_code}' localhost:8000/v1/models)
  [ "$c" = "200" ] && break; sleep 15
done; echo "http=$c"

# log size first — a runaway file means something looped
ls -l /tmp/b12x-verify.log

# each read is pattern-filtered AND capped
grep -iE 'supports_block_size|planned_page_sizes|Setting attention block size|kv lcm block sizes|GPU KV cache size' \
  /tmp/b12x-verify.log | head -10
grep -icE 'device-side|AcceleratorError|IllegalInstruction' /tmp/b12x-verify.log
grep -oE 'Capturing CUDA graphs \([A-Z]+\): *[0-9]+%' /tmp/b12x-verify.log | tail -2

# worst case, bound a raw tail by time as well as lines
timeout 10 tail -n 40 /tmp/b12x-verify.log
```

Never `tail -f` (or `docker logs -f`) against this serve: the EOF never comes
and the read runs until a timeout kills it.

### Acceptance criteria

All must hold, node1 rank 0:

| Check | Expected |
|---|---|
| patch applied on both ranks | `Accepted splittable hybrid storage blocks in B12X` ×2 |
| backend resolved from real layers | `Using b12x with ... planned_page_sizes=(64, 128)` |
| aligned storage block | `Setting attention block size to 1664 tokens` |
| manager agrees | `kv lcm block sizes 1664` |
| KV split served | `GPU KV cache size: <N> tokens` |
| graph replay | `Capturing CUDA graphs (FULL): 100%` |
| CUDA errors | `0` |
| endpoint | HTTP 200 on `/v1/models` and a chat completion |

Then confirm the arm is functional, not merely bootable:

```bash
curl -s -m 90 -X POST localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-inference-lab/Qwen3.8-27B-Coder390-DFlash2",
       "messages":[{"role":"user","content":"Reply with exactly: OK"}],
       "max_tokens":8,"temperature":0}' -w '\nhttp=%{http_code}\n' | tail -3
```

**Failure triage.** The three ways this historically failed:

- `Aligned KV cache block size 1664 is not supported` → gate 1, patcher not applied.
- `b12x requires --block-size in (64, 128), got 1664` → gate 2, same cause.
- `index out of bounds ... < 262144` (device assert) → *not* this defect; that's
  the drafter rope cache, see the YaRN notes in the recipe header.

---

## 5. Optional — decode throughput

`caaba0f`'s protocol, for comparison against 32.2 vs 26.4 tok/s per stream:

- C1 / 60k-token probe, 8 sequential prompts, `temperature=0`, `ignore_eos=true`
- one run with `--attention-backend B12X --block-size 64`, one without
- report per-stream decode tok/s and mean TPOT

Do not treat the recorded numbers as a current baseline: they were measured on
an older image that no longer boots this arm, so this is the first measurement
of the arm on the pinned one.

---

## 6. Teardown and restore production

**Never run a bare `launch-cluster.sh stop`.** It resolves its peer list from
`-n`, else `DOTENV_CLUSTER_NODES` in the fork `.env`. That happens to be
`10.100.171.3,10.100.171.4` today, but the file is autodiscovery-written and
machine-specific — if it ever lists all four nodes, an unranked stop reaches
every rank and kills co-tenants on nodes 3–4 (which run unrelated
`vllm_node` instances and other services). Always pass the node list explicitly.

```bash
cd /cluster-shared/projects/spark-vllm-docker

# confirm what the .env fallback would do, but do not rely on it
grep -E '^CLUSTER_NODES' .env

# scope the stop to exactly the two ranks this runbook launched
timeout 200 bash launch-cluster.sh -n 10.100.171.3,10.100.171.4 stop

# confirm both ranks are gone before moving on
docker ps --format '{{.Names}}' | grep -q vllm_node && echo 'node1 STILL UP' || echo 'node1 clean'
```

Then clean the scratch script and restore production:

```bash
cd /cluster-shared/projects/spark-vllm-docker
rm -f examples/b12x-verify.sh          # scratch, do not commit

# restore the 1M-context YaRN serve
cd ~/cluster-config
setsid nohup bash -c 'bash scripts/spark-vllm-docker-serve.sh launch \
  --recipe qwen3.8-27b-coder390-2x \
  --rank-order "gx10-cb11:0,gx10-f1d8:1" --daemon' \
  </dev/null > /tmp/restore.log 2>&1 & disown
```

Verify with `curl -s localhost:8000/v1/models` →
`max_model_len: 1048576`.

Also confirm nodes 3–4 were untouched — they host separate instances and
co-tenants:

```bash
# on gx10-node3 and gx10-node4: your containers should still show the same
# StartedAt they had before the test run (restarts=0).
docker inspect -f '{{.State.StartedAt}} {{.RestartCount}}' vllm_node
```

---

## 7. Gotchas that cost time

| Symptom | Cause / fix |
|---|---|
| `exec: b12x-kcache: not found` | that wrapper is a host path, not in the image. Use `vllm serve` directly, or launch via `launch-cluster.sh` as above. |
| `Launch script '/tmp/x.sh' not found` | `--launch-script` requires an existing file; render it with `run-recipe.py --dry-run` (§4c). |
| `docker logs vllm_node` returns 0 bytes | manual launch keeps PID 1 as `sleep infinity`; the engine is a detached `docker exec` writing to the host log. Read `/tmp/b12x-verify.log`. |
| MCP call killed at ~240 s | run the launch detached (`setsid nohup ... & disown`) and poll; never block a `shell_exec` on a boot. |
| Probe reports `64/64` and no error | config-only probes are false negatives: `_find_non_ssm_backends` reads *instantiated* layers, returns `[]`, and the aligner early-returns. Only a real boot is conclusive. |
| Probe says `1648` instead of `1664` | the alignment quantum follows the chosen backend (16 for FlashInfer, 64 for B12X). Test the backend you mean to ship. |
| `stop` without `--rank-order` | defaults to all four nodes. Scope it to `gx10-cb11:0,gx10-f1d8:1` so co-tenants on nodes 3–4 are untouched. |
| Files vanish from `/tmp` | session churn. Keep artifacts on a durable path (`~/...`) or re-stage. |
| Log read floods terminal/context | `/tmp/b12x-verify.log` spans two ranks and grows. Always `grep ... \| head -N`, or `timeout 10 tail -n 40`. Never `tail -f` / `docker logs -f`. |

---

## 8. What this runbook does and does not cover

**Covers:** the patcher, both gates, the split-to-kernel-page path, and a live
end-to-end boot of the B12X arm with the patch injected in-container.

**Does not cover:** a boot from an image built with the Dockerfile `RUN` stages.
§4 injects the patch at runtime, so the build wiring (`Dockerfile` source stage
and `--installed` stage) remains unexercised by a real build. To close that,
rebuild vLLM (`build-and-copy.sh --rebuild-vllm`), then run §4 without the
`--apply-mod /tmp/b12x-split-mod` argument and confirm the patch appears in the
image's `b12x.py` (`grep -c 'accept splittable storage block'`).
