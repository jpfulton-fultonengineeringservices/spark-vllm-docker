# ---------------------------------------------------------------- store

# write_probe <dir>: mkdir -p, write .probe.$$ , read back, compare, unlink.
# 0 when the dir is writable, 1 otherwise.
write_probe() {
  local d="$1" tmp rd
  if ! mkdir -p "$d" 2>/dev/null; then
    return 1
  fi
  tmp="$d/.probe.$$"
  if ! printf 'b12x-kcache' > "$tmp" 2>/dev/null; then
    return 1
  fi
  if ! rd=$(cat "$tmp" 2>/dev/null); then
    rm -f "$tmp" 2>/dev/null
    return 1
  fi
  rm -f "$tmp" 2>/dev/null
  [ "$rd" = "b12x-kcache" ]
}

# store_candidates(): echo candidate dirs, one per line, priority order.
store_candidates() {
  if [ -n "${B12X_KCACHE_STORE:-}" ]; then
    echo "$B12X_KCACHE_STORE"
  fi
  echo "/root/.cache/huggingface/.spark-vllm/b12x-kcache"
  echo "/root/.cache/b12x/_archives"
}

# resolve_store <mode>: mode "write" (first writable candidate) or "read"
# (first candidate holding a <key>.tar.zst). Prints "<path><TAB><shared>".
resolve_store() {
  local mode="$1" key="${2:-}" cand first_writable="" existing="" any_writable=""
  for cand in $(store_candidates); do
    if write_probe "$cand"; then
      any_writable="yes"
      if [ -z "$first_writable" ]; then
        first_writable="$cand"
      fi
    fi
    if [ "$mode" = "read" ] && [ -n "$key" ]; then
      if [ -f "$cand/$key.tar.zst" ]; then
        existing="$cand"
        break
      fi
    fi
  done
  if [ "$mode" = "write" ]; then
    if [ -n "$first_writable" ]; then
      printf '%s\tyes\n' "$first_writable"
      return 0
    fi
    return 1
  fi
  if [ -n "$existing" ]; then
    # shared=yes only if the dir is also writable (head reading its own
    # shared store); a root-squashed peer read-only hit reports shared=no.
    if [ "$any_writable" = "yes" ] && write_probe "$existing"; then
      printf '%s\tyes\n' "$existing"
    else
      printf '%s\tno\n' "$existing"
    fi
    return 0
  fi
  return 1
}

