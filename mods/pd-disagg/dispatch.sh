#!/bin/bash
#
# [pd-disagg] Role-aware PD-disaggregated MiMo-V2.6-Flash-RL-UNCENSORED EXL3 (4-node GB10)
#
# Sourced by recipes/pd-disagg-mimo-uncensored-exl3-4x.yaml's single-line recipe command,
# which launch-cluster.sh's outer engine invokes per node with:
#   dispatch.sh --nnodes N --node-rank R --master-addr HEAD --master-port P [--headless]
#
# Topology (NNODES must be 4; see tmp/spark-vllm-docker/pd-3x-completion-plan.md):
#   ranks 0..1  PREFILL  : EXL3 pack, tensor-parallel 2 (2-rank sub-group,
#                         mirrors recipes/mimo-v2.6-flash-rl-uncensored-exl3-2x.yaml),
#                         API port 8100, rank 0 also runs the toy_proxy_server router.
#   ranks 2..3  DECODE   : same EXL3 pack TP=2 sub-group + DFlash speculative
#                         decoding, API port 8200, kv_consumer. Both roles serve
#                         PD_EXL3_MODEL so prefill/decode quantization match.
#
# TP-TRIMMING FOOTGUN: launch-cluster.sh's outer engine may append its own
# '-tp N'/'--tensor-parallel-size N' style args for cluster orchestration;
# vLLM-level TP flags are NEVER inherited from the outer engine. Each role
# derives its TP size ONLY from PD_TP_PREFILL / PD_TP_DECODE and re-emits the
# correct vLLM flags itself (including its own sub-group rendezvous args for
# the prefill TP=2 pair).
#
# The env file published by mods/pd-disagg/run.sh (UCX_NET_DEVICES for RoCE
# HCAs, VLLM_NIXL_SIDE_CHANNEL_PORT, shared parity vars) is sourced first;
# run.sh must have run in-container before dispatch.sh unless PD_DISAGG_ENABLED=0.
#
# Parity by construction: every flag that MUST match across roles (speculative
# config incl. DFlash method/depth, block size, KV dtype + skip layers, max
# model len) is derived ONLY from the shared PD_* vars below, and
# PD_PARITY_SHA256 (sha256 of the canonical shared config) is computed and
# logged identically on every role at boot. Verify the hash matches across
# all three roles' logs before trusting a PD session.

set -euo pipefail

PREFIX="[pd-disagg]"

fail() { echo "$PREFIX PD_DISAGG_FAIL $*" >&2; exit 1; }

# Opt-in gate (run.sh already enforces this, keep a second guard here).
if [ "${PD_DISAGG_ENABLED:-0}" != "1" ]; then
  echo "$PREFIX disabled (PD_DISAGG_ENABLED != 1); PD not active."
  exit 0
fi

# Env file published by mods/pd-disagg/run.sh
ENV_FILE="${PD_DISAGG_ENV_FILE:-/tmp/pd-disagg.env}"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  . "$ENV_FILE"
else
  # B1 fallback: the exec-script env block already exported PD_DISAGG_ENABLED=1,
  # so executing the sibling run.sh in-process publishes $ENV_FILE AND
  # re-exports the UCX RoCE pinning + NIXL vars (M2) before the role logic runs.
  SIBLING_RUN_SH="$(dirname "$0")/run.sh"
  if [ ! -f "$SIBLING_RUN_SH" ]; then
    fail "reason=missing_env_file file=$ENV_FILE sibling=$SIBLING_RUN_SH (mods/pd-disagg/run.sh must exist to publish the env file)"
  fi
  echo "$PREFIX info: env file $ENV_FILE missing; sourcing sibling run.sh in-process ($SIBLING_RUN_SH)"
  # In-process (sourced) so run.sh's UCX RoCE pinning + NIXL re-exports land in
  # this process (M2), not a subprocess; its opt-in gate is already satisfied
  # (PD_DISAGG_ENABLED=1 from the exec-script env block) and it ends without an
  # explicit exit, so sourcing returns here after publishing $ENV_FILE.
  # shellcheck disable=SC1091
  . "$SIBLING_RUN_SH"
  if [ ! -f "$ENV_FILE" ]; then
    fail "reason=run_sh_did_not_publish file=$ENV_FILE sibling=$SIBLING_RUN_SH"
  fi
  # shellcheck disable=SC1090
  . "$ENV_FILE"
fi

# KV-transfer allocator constraint (GATE on NixlConnector): PyTorch's
# CUDA-VMM expandable_segments can remap KV virtual addresses to different
# physical pages, invalidating NIXL-registered IB memory regions — vLLM
# hard-rejects the pair at config validation. launch-cluster.sh exports
# expandable_segments:True as a GB10 platform default (good for non-PD
# recipes); unset it for BOTH PD roles only. (Alternative per the validator
# message: enable_cumem_allocator / sleep mode — not needed here.)
case "${PYTORCH_CUDA_ALLOC_CONF:-}" in
  *expandable_segments:True*|*expandable_segments=true*)
    echo "$PREFIX info: unsetting PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF (NixlConnector + expandable_segments invalidates registered KV memory)"
    unset PYTORCH_CUDA_ALLOC_CONF
    ;;
esac

# Outer-engine-appended args. Topology flags are parsed and consumed; anything
# after `--` is an engine flag (e.g. trace flags appended by cluster-config
# when SPARK_VLLM_DOCKER_TRACES=1) and is forwarded to BOTH roles' vLLM
# commands verbatim — the same semantics every non-PD recipe gets from
# run-recipe.py's post-`--` pass-through.
NNODES=""
NODE_RANK=""
MASTER_ADDR=""
MASTER_PORT=""
PD_ENGINE_ARGS=()
# shellcheck disable=SC2034  # MASTER_PORT parsed for symmetry only; the
# outer-engine's master port is never forwarded to per-role vLLM commands.
while [ $# -gt 0 ]; do
  case "$1" in
    --nnodes) NNODES="$2"; shift 2 ;;
    --node-rank) NODE_RANK="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --headless) shift ;; # per-role headless is re-derived below
    --) shift; PD_ENGINE_ARGS=("$@"); break ;;
    # Unknown args are engine flags for BOTH roles (run-recipe.py flattens the
    # recipe's post-`--` args into this line ahead of the topology flags —
    # e.g. the OTLP trace pair when SPARK_VLLM_DOCKER_TRACES=1). Forward them
    # verbatim rather than fail: vLLM parses its own flags, headless workers
    # included, so unknown-to-us-but-valid-to-vLLM args ride both role commands.
    *) PD_ENGINE_ARGS+=("$1"); shift ;;
  esac
done
: "${NNODES:?op=argparse reason=missing_--nnodes}"
: "${NODE_RANK:?op=argparse reason=missing_--node-rank}"
[ "$NNODES" -eq 4 ] || fail "op=topology reason=pd_disagg_4_node_only got=$NNODES (this mod targets the 4-node prefill-TP2/decode-TP2 shape; see mods/pd-disagg/README.md)"

# Role sizes: derived ONLY here (see TP-trimming footgun above).
TP_PREFILL="${PD_TP_PREFILL:-2}"
TP_DECODE="${PD_TP_DECODE:-2}"
[ "$TP_PREFILL" -eq 2 ] || fail "op=topology reason=prefill_tp_must_be_2 got=$TP_PREFILL (EXL3 pack rendezvous is a 2-rank sub-group; NNODES=4 prefill must be TP=2)"
[ "$TP_DECODE" -eq 2 ] || fail "op=topology reason=decode_tp_must_be_2 got=$TP_DECODE (EXL3 decode rendezvous is a 2-rank sub-group; NNODES=4 decode must be TP=2)"

# This node's IP: prefer VLLM_HOST_IP (set per node by launch-cluster.sh),
# else outer master addr, else hostname -I.
NODE_IP="${VLLM_HOST_IP:-${MASTER_ADDR:-}}"
if [ -z "$NODE_IP" ]; then
  NODE_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
: "${NODE_IP:?op=env reason=missing_node_ip set_VLLM_HOST_IP}"

# NIXL side channel: each node advertises its OWN IP (agents dial each other),
# so VLLM_NIXL_SIDE_CHANNEL_HOST must never be the head's IP on workers.
export VLLM_NIXL_SIDE_CHANNEL_HOST="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_PORT="${PD_NIXL_PORT:-5600}"

# ---------------------------------------------------------------------------
# Shared parity config: defaults first (so a bare boot still works), then a
# hard fail on any unset/empty value. Every role derives identical vLLM flags
# from these and ONLY these.
# ---------------------------------------------------------------------------
PD_NUM_SPECULATIVE_TOKENS="${PD_NUM_SPECULATIVE_TOKENS:-7}"
PD_BLOCK_SIZE="${PD_BLOCK_SIZE:-128}"
PD_MAX_MODEL_LEN="${PD_MAX_MODEL_LEN:-1048576}"
PD_MAX_NUM_SEQS="${PD_MAX_NUM_SEQS:-32}"
PD_MAX_NUM_BATCHED_TOKENS="${PD_MAX_NUM_BATCHED_TOKENS:-16384}"
: "${PD_NUM_SPECULATIVE_TOKENS:?op=env reason=PD_NUM_SPECULATIVE_TOKENS required for PD parity}"
: "${PD_BLOCK_SIZE:?op=env reason=PD_BLOCK_SIZE required for PD parity}"
: "${PD_MAX_MODEL_LEN:?op=env reason=PD_MAX_MODEL_LEN required for PD parity}"
: "${PD_MAX_NUM_SEQS:?op=env reason=PD_MAX_NUM_SEQS required for PD parity}"
: "${PD_MAX_NUM_BATCHED_TOKENS:?op=env reason=PD_MAX_NUM_BATCHED_TOKENS required for PD parity}"

# Canonical shared config hash: identical bytes -> identical hash on rank 0, 1, 2.
# Compare across roles' boot logs before trusting a PD session.
# n3 single-source: the canonical payload string is published by run.sh as
# PD_PARITY_PAYLOAD in $ENV_FILE; the printf format string lives ONLY in
# run.sh (never re-typed here). The payload var holds the canonical string
# minus its trailing newline ($() strips it); hashing re-appends the newline
# so hashed bytes match run.sh byte-for-byte. A stale env file that predates
# the var fails closed below — re-run run.sh (or let the B1 fallback above
# publish it) to refresh.
: "${PD_PARITY_PAYLOAD:?op=env reason=PD_PARITY_PAYLOAD missing from env file (stale publish — re-run mods/pd-disagg/run.sh)}"
PD_PARITY_SHA256="$(printf '%s\n' "$PD_PARITY_PAYLOAD" \
  | { sha256sum 2>/dev/null || shasum -a 256; } | awk '{print $1}')"
export PD_PARITY_SHA256

# b12x-kcache needs a persistent inductor cache (live-learned; per-boot /tmp
# cache re-compiles kernels).
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/.cache/torchinductor}"

# e.g. "b12x-kcache exec vllm serve" (mods/b12x-kernel-cache).
# shellcheck disable=SC2206
VLLM_SERVE=(${PD_VLLM_SERVE:-vllm serve})

# enforce_handshake_compat=false: NIXL's compatibility hash mismatches by
# design here -- decode runs DFlash speculative decoding (extra draft KV
# layers) while prefill does not, so the two roles hash different configs.
# Cross-role parity of the TRANSFERRED layout (block size, fp8 KV,
# sliding_window skip, max_model_len, seq/batch caps) is enforced at boot by
# PD_PARITY_SHA256 on every role; NIXL's redundant whole-config check is
# disabled rather than silenced per-request.
KV_PRODUCER='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"enforce_handshake_compat":false}}'
KV_CONSUMER='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"enforce_handshake_compat":false}}'

# Shared flag payload (identical on both roles): everything from the live
# recipes/mimo-v2.6-flash-rl-uncensored-exl3-2x.yaml EXCEPT port / TP / GMU,
# which are role-specific and appended separately.
shared_spec_cfg() {
  printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"attention_backend":"B12X"}' \
    "${PD_DFLASH_MODEL:-/workspace/MiMo-V2.6-Flash-RL-dflash}" \
    "${PD_NUM_SPECULATIVE_TOKENS}"
}

# ---------------------------------------------------------------------------
# Ranks 0..1: PREFILL (EXL3 pack, TP=2, 2-rank sub-group).
# The sub-group rendezvous is re-emitted here (rank 0 = sub-group master):
#   --nnodes 2 --node-rank $NODE_RANK (0 or 1) --master-addr $PREFILL_MASTER_ADDR
# PREFILL_MASTER_ADDR defaults to outer HEAD (rank 0's IP).
# ---------------------------------------------------------------------------
if [ "$NODE_RANK" -lt 2 ]; then
  PREFILL_MASTER_ADDR="${PD_PREFILL_MASTER_ADDR:-${MASTER_ADDR:-$NODE_IP}}"
  PREFILL_MASTER_PORT="${PD_PREFILL_MASTER_PORT:-29501}"

  echo "$PREFIX role=PREFILL rank=$NODE_RANK tp=$TP_PREFILL nixl_host=$NODE_IP:$VLLM_NIXL_SIDE_CHANNEL_PORT prefill_master=$PREFILL_MASTER_ADDR:$PREFILL_MASTER_PORT PD_PARITY_SHA256=$PD_PARITY_SHA256"

  # Router only on rank 0 (prefill needs to boot first; see PD_ROUTER_START_DELAY).
  if [ "$NODE_RANK" -eq 0 ] && [ "${PD_ROUTER_ENABLED:-0}" = "1" ]; then
    # B2: default to the mod-local vendored router (lands at
    # /workspace/mods/pd-disagg/toy_proxy_server.py via apply_mod_to_container);
    # /workspace/toy_proxy_server.py stays as an override.
    ROUTER_SCRIPT="${PD_ROUTER_SCRIPT:-$(dirname "$0")/toy_proxy_server.py}"
    if [ -f "$ROUTER_SCRIPT" ]; then
      PD_ROUTER_START_DELAY="${PD_ROUTER_START_DELAY:-30}"
      # The router must dial the decode sub-group MASTER only (first
      # PD_DECODE_NODE_IPS entry = rank 2): --headless sub-group members
      # (rank 3) never bind the API port, so round-robin over the full CSV
      # dies with Connection refused on every other request. Same for
      # prefill: --prefiller-hosts is this node's IP (rank 0 master); the
      # headless rank 1 never serves.
      router_decoder_hosts=("${PD_DECODE_NODE_IPS%%,*}")
      router_decoder_ports=("${PD_DECODE_PORT:-8200}")
      # M1: sleep-gated start (README + recipe comments promise the 30s gate);
      # subshell keeps set -e safe, exec replaces the subshell with python3.
      # --host 0.0.0.0: the vendored router defaults --host to 127.0.0.1,
      # which would bind loopback only and break http://<head>:8000/v1/models.
      ( sleep "$PD_ROUTER_START_DELAY"; exec python3 "$ROUTER_SCRIPT" \
        --host "${PD_ROUTER_HOST:-0.0.0.0}" \
        --port "${PD_ROUTER_PORT:-8000}" \
        --prefiller-hosts "$NODE_IP" \
        --prefiller-ports "${PD_PREFILL_PORT:-8100}" \
        --decoder-hosts ${router_decoder_hosts[@]+"${router_decoder_hosts[@]}"} \
        --decoder-ports ${router_decoder_ports[@]+"${router_decoder_ports[@]}"} \
      ) >/tmp/pd-router.log 2>&1 &
      echo "$PREFIX router scheduled (delay=${PD_ROUTER_START_DELAY}s, port=${PD_ROUTER_PORT:-8000}, log=/tmp/pd-router.log)"
    else
      echo "$PREFIX WARN router script not found (PD_ROUTER_SCRIPT=$ROUTER_SCRIPT); skipping router" >&2
    fi
  fi

  # shellcheck disable=SC2206
  PREFILL_CMD=("${VLLM_SERVE[@]}" "${PD_EXL3_MODEL:-XiaomiMiMo/MiMo-V2.6-Flash-RL-EXL3}"
    --host 0.0.0.0
    --port "${PD_PREFILL_PORT:-8100}"
    --tensor-parallel-size "$TP_PREFILL"
    --nnodes 2
    --node-rank "$NODE_RANK"
    --master-addr "$PREFILL_MASTER_ADDR"
    --master-port "$PREFILL_MASTER_PORT"
    --quantization exl3
    --attention-backend B12X
    --linear-backend b12x
    --block-size "$PD_BLOCK_SIZE"
    --max-model-len "$PD_MAX_MODEL_LEN"
    --kv-cache-dtype fp8
    --kv-cache-dtype-skip-layers sliding_window
    --max-num-seqs "$PD_MAX_NUM_SEQS"
    --max-num-batched-tokens "$PD_MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "${PD_PREFILL_GMU:-0.7}"
    --enable-chunked-prefill
    --async-scheduling
    --enable-prefix-caching
    --generation-config vllm
    # Recipe command: templates use {{...}} because run-recipe.py renders them
    # via str.format; dispatch.sh args are NEVER format-rendered (verbatim to
    # vllm serve), so single braces -- {{...}} would fail argparse json.loads.
    --override-generation-config '{"top_p":0.95}'
    --reasoning-parser mimo
    --tool-call-parser mimo
    --enable-auto-tool-choice
    --kv-cache-metrics
    --enable-mfu-metrics
    --kv-transfer-config "$KV_PRODUCER"
    --trust-remote-code
  )
  # Outer-engine trailing args (post-`--`, e.g. OTLP trace flags) forwarded
  # verbatim to the prefill engine.
  PREFILL_CMD+=("${PD_ENGINE_ARGS[@]+"${PD_ENGINE_ARGS[@]}"}")
  # NOTE: no --speculative-config on prefill; spec-decode (DFlash) is
  # decode-only. Prefill/decode parity on num_speculative_tokens is enforced
  # via the shared var + PD_PARITY_SHA256, not by duplicating the flag.
  if [ "$NODE_RANK" -gt 0 ]; then
    PREFILL_CMD+=(--headless)
  fi
  exec "${PREFILL_CMD[@]}"
fi

# ---------------------------------------------------------------------------
# Ranks 2..3: DECODE (same EXL3 pack TP=2 sub-group + DFlash).
# The decode pair forms its OWN sub-group rendezvous, mirroring prefill above:
# rank 2 = decode sub-group master, rank 3 = worker with --headless.
# PD_DECODE_MASTER_ADDR is REQUIRED at NNODES=4 — decode ranks cannot
# rendezvous without it (the outer master addr is the prefill head, wrong
# subnet group for the decode pair).
# ---------------------------------------------------------------------------
if [ "$NODE_RANK" -ge 2 ]; then
  : "${PD_DECODE_MASTER_ADDR:?op=argparse reason=PD_DECODE_MASTER_ADDR_required_for_4x}"

  echo "$PREFIX role=DECODE rank=$NODE_RANK tp=$TP_DECODE nixl_host=$NODE_IP:$VLLM_NIXL_SIDE_CHANNEL_PORT PD_PARITY_SHA256=$PD_PARITY_SHA256"

  # shellcheck disable=SC2206
  DECODE_CMD=("${VLLM_SERVE[@]}" "${PD_EXL3_MODEL:-XiaomiMiMo/MiMo-V2.6-Flash-RL-EXL3}"
    --host 0.0.0.0
    --port "${PD_DECODE_PORT:-8200}"
    --tensor-parallel-size "$TP_DECODE"
    --nnodes 2
    --node-rank "$((NODE_RANK-2))"
    --master-addr "$PD_DECODE_MASTER_ADDR"
    --master-port "${PD_DECODE_MASTER_PORT:-29502}"
    --quantization exl3
    --trust-remote-code
    --max-model-len "$PD_MAX_MODEL_LEN"
    --attention-backend B12X
    --linear-backend b12x
    --block-size "$PD_BLOCK_SIZE"
    --kv-cache-dtype fp8
    --kv-cache-dtype-skip-layers sliding_window
    --max-num-seqs "$PD_MAX_NUM_SEQS"
    --max-num-batched-tokens "$PD_MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "${PD_DECODE_GMU:-0.7}"
    --enable-chunked-prefill
    --async-scheduling
    --enable-prefix-caching
    --speculative-config "$(shared_spec_cfg)"
    --generation-config vllm
    --override-generation-config '{"top_p":0.95}'
    --reasoning-parser mimo
    --tool-call-parser mimo
    --enable-auto-tool-choice
    --kv-cache-metrics
    --enable-mfu-metrics
    --kv-transfer-config "$KV_CONSUMER"
  )
  # Outer-engine trailing args (post-`--`) forwarded verbatim to the decode
  # engine as well, so trace/metric flags apply symmetrically to both roles.
  DECODE_CMD+=("${PD_ENGINE_ARGS[@]+"${PD_ENGINE_ARGS[@]}"}")
  if [ "$NODE_RANK" -eq 3 ]; then
    DECODE_CMD+=(--headless)
  fi
  exec "${DECODE_CMD[@]}"
fi

fail "op=topology reason=no_role_for_rank rank=$NODE_RANK nnodes=$NNODES (expected 0-1 prefill, $((NNODES-1)) decode)"
