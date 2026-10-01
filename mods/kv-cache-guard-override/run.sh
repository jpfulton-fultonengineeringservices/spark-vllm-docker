#!/bin/bash
set -euo pipefail

# KV-cache memory-guard override mod.
#
# _check_enough_kv_cache_memory (vllm/v1/core/kv_cache_utils.py) is a
# fail-fast startup guard: it refuses to start when one request at the full
# max_model_len would not fit in the available KV cache. That assumption is
# far more conservative than real workloads (observed on the TP=4 fleet: max
# 22% KV utilization with 8 concurrent sequences), and on the 3x PP=3
# topology it blocks a perfectly serviceable engine (33.84 GiB needed for a
# single 1M-token request vs 27.17 GiB available at gmu 0.80).
#
# The patch keeps the guard as the default and adds an explicit override:
# with VLLM_SKIP_KV_CACHE_MEMORY_CHECK=1 the raise becomes a warning and the
# engine starts. A request that outgrows the pool queues instead of crashing.
#
# Each file is patched independently and skipped when already fixed.
# Idempotent within the same fresh container.

PREFIX="[kv-cache-guard-override]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
KVUTILS="$PYTHON_ROOT/vllm/v1/core/kv_cache_utils.py"

echo "=== KV-cache memory-guard override mod ==="

if [ ! -f "$KVUTILS" ]; then
  echo "$PREFIX Missing $KVUTILS; a newer image is required." >&2
  exit 1
fi

python3 - "$KVUTILS" <<'PY'
import sys
from pathlib import Path

target = Path(sys.argv[1])
text = target.read_text()

OLD = '''    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the model's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` (which also controls "
            f"CPU memory on the CPU backend) or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )'''

NEW = '''    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        message = (
            f"To serve at least one request with the model's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` (which also controls "
            f"CPU memory on the CPU backend) or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )
        # spark-vllm-docker/mods/kv-cache-guard-override:
        # VLLM_SKIP_KV_CACHE_MEMORY_CHECK=1 downgrades this startup guard to
        # a warning. The guard requires one request to fit max_model_len in
        # KV; observed fleet workloads stay far below that (TP=4: max 22% KV
        # util at 8 concurrent sequences), and a request that outgrows the
        # pool queues instead of crashing. Unset (or any other value) keeps
        # the fail-fast behavior.
        if os.environ.get("VLLM_SKIP_KV_CACHE_MEMORY_CHECK", "0") not in (
            "1",
            "true",
        ):
            raise ValueError(message)
        logger.warning(
            "VLLM_SKIP_KV_CACHE_MEMORY_CHECK=1: KV memory guard skipped. %s",
            message,
        )'''

if "spark-vllm-docker/mods/kv-cache-guard-override" in text:
    print("already patched; skipping")
    sys.exit(0)
if OLD not in text:
    print(
        "_check_enough_kv_cache_memory guard block not found; file shape changed",
        file=sys.stderr,
    )
    sys.exit(1)
target.write_text(text.replace(OLD, NEW, 1))
print("patched", target)
PY
