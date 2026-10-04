# ---------------------------------------------------------------- key

# argv_value <flag>: extract the value of --flag from argv (ARGV_STR).
argv_value() {
  local flag="$1" tok next="0"
  for tok in ${ARGV_STR:-}; do
    if [ "$next" = "1" ]; then
      echo "$tok"
      return 0
    fi
    case "$tok" in
      "$flag") next="1" ;;
      "$flag"=*) echo "${tok#*=}" ; return 0 ;;
    esac
  done
}

field_or_override() { # <ENV-SUFFIX> <resolved-value>
  local ov
  eval "ov=\"\${B12X_KCACHE_ID_$1:-}\""
  if [ -n "$ov" ]; then
    printf '%s' "$ov"
  else
    printf '%s' "$2"
  fi
}

# resolve_id: sets F_B12X_REV F_CUTLASS_DSL F_TORCH F_VLLM F_VLLM_OBJS F_ARCH
resolve_id() {
  local v

  v=$(field_or_override B12X_REV "$(cat /workspace/b12x-source-commit 2>/dev/null || true)")
  if [ -z "$v" ]; then
    v=$(field_or_override B12X_REV "$(pyver b12x)")
  fi
  F_B12X_REV=$(_require b12x_rev "$v")

  v=$(field_or_override CUTLASS_DSL "$(pyver nvidia-cutlass-dsl)")
  F_CUTLASS_DSL=$(_require cutlass_dsl "$v")

  v=$(field_or_override TORCH "$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)")
  F_TORCH=$(_require torch "$v")

  v=$(field_or_override VLLM "$(pyver vllm)")
  F_VLLM=$(_require vllm "$v")

  # Newline+sha256 of each vllm _C*.so (sorted); empty list is legitimate.
  if [ -n "${B12X_KCACHE_ID_VLLM_OBJS:-}" ]; then
    F_VLLM_OBJS="$B12X_KCACHE_ID_VLLM_OBJS"
  else
    F_VLLM_OBJS=""
    local so_list f h
    so_list=$(find /usr/local/lib/python3*/dist-packages/vllm \
      -maxdepth 1 -name '_C*.so' 2>/dev/null | LC_ALL=C sort)
    if [ -n "$so_list" ]; then
      while IFS= read -r f; do
        h=$(sha "$f")
        F_VLLM_OBJS="${F_VLLM_OBJS:+$F_VLLM_OBJS;}$h"
      done <<EOF
$so_list
EOF
    fi
  fi

  F_ARCH=$(field_or_override CUDA "${CUTE_DSL_ARCH:-}")
  if [ -z "$F_ARCH" ]; then
    die arch
  fi
}

key() {
  local model tp quant attn lin kv_dtype block maxlen
  local roots_pairs hash short

  resolve_id

  tp=$(argv_value --tensor-parallel-size); [ -n "$tp" ] || tp="1"
  quant=$(argv_value --quantization)
  attn=$(argv_value --attention-backend)
  lin=$(argv_value --linear-backend)
  kv_dtype=$(argv_value --kv-cache-dtype)
  block=$(argv_value --block-size)
  maxlen=$(argv_value --max-model-len)

  # model: first non-flag token after the `serve` literal in argv.
  model=$(printf '%s\n' ${ARGV_STR:-} | awk '
    {
      for (i = 1; i <= NF; i++) {
        if (seen) { if ($i !~ /^-/) { print $i; exit }; seen = 0; continue }
        if ($i == "serve") seen = 1
      }
    }
  ')
  if [ -z "$model" ]; then
    # no `serve` literal: first non-flag token
    model=$(printf '%s\n' ${ARGV_STR:-} | awk '
      { for (i = 1; i <= NF; i++) if ($i !~ /^-/) { print $i; exit } }
    ')
  fi
  _require model "$model" >/dev/null

  roots_pairs=$(roots | awk -F'\t' 'NF==2 {print $1 ":" $2}' | LC_ALL=C sort)
  _require roots "$roots_pairs" >/dev/null

  hash=$(printf '%s\n' \
    "b12x-kcache/v1" "$F_B12X_REV" "$F_CUTLASS_DSL" "$F_TORCH" "$F_VLLM" \
    "$F_VLLM_OBJS" "$F_ARCH" "$tp" "$quant" "$attn" "$lin" "$kv_dtype" \
    "$block" "$maxlen" "$model" "$roots_pairs" \
    | sed '$ { G; s/\n$//; }' \
    | tr '\n' "$UNIT" \
    | stdin_sha)

  short=$(printf '%s' "$model" | awk -F/ '{print $NF}' | cut -c1-24)
  printf 'b12x-kernels__%s__tp%s__%s\n' "$short" "$tp" "${hash:0:16}"
}

cmd_key() {
  key
}

cmd_store() {
  local res
  if ! res=$(resolve_store write); then
    log "store: no writable candidate"
    return 1
  fi
  printf '%s\n' "$res"
}

