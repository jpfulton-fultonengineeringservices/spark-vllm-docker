#!/bin/bash
# b12x-kcache: archive/restore the b12x/CuTe autotuned-kernel cache around a
# vLLM serve.
#
# Every cold boot autotunes all b12x/CuTe kernels (JIT compile + tuning
# selection), which dominates launch time on the GX10 cluster. The tuned cache
# lives under the per-node b12x cache root (default ~/.cache/b12x, mounted at
# /root/.cache/b12x) and is NOT shared across nodes or preserved across image
# rebuilds. This tool snapshots it after a successful serve and restores the
# matching snapshot before later serves.
#
# Store: candidates in priority order
#   1. $B12X_KCACHE_STORE (operator-owned)
#   2. /root/.cache/huggingface/.spark-vllm/b12x-kcache
#      (on the /cluster-shared NFS filesystem via the HF-cache mount; head can
#      write, root-squashed peers cannot)
#   3. /root/.cache/b12x/_archives (per-node fallback)
# `save` writes to the first candidate that passes an actual write probe;
# `restore`/`verify` search all candidates and use the first EXISTING archive,
# so peers read the head's archive from the shared store.
#
# No equivalent archive/replay mechanism exists upstream; this is
# spark-vllm-docker-local (docker/B12X_CACHE_INTEGRITY.md covers atomic
# on-disk writes only, not capture/restore).
#
# bash 3.2 + 5.x portable (repo test convention).

PREFIX="[b12x-kcache]"
# No -e: save/restore must degrade to "skip" log lines, never kill the serve.
set -uo pipefail

PROGNAME="b12x-kcache"
UNIT=$'\x1f'
SELF_DIR="${BASH_SOURCE[0]%/*}"
[ "$SELF_DIR" = "${BASH_SOURCE[0]}" ] && SELF_DIR="."

KCACHE_LIB_DIR="$SELF_DIR/lib"
case "${BASH_SOURCE[0]}" in
	*/*) ;;
	*) KCACHE_LIB_DIR="lib" ;;
esac


for _f in common store key save restore verify exec main; do
	. "$KCACHE_LIB_DIR/$_f.sh"
done
unset _f KCACHE_LIB_DIR
