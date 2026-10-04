# ---------------------------------------------------------------- exec

cmd_exec() {
  if [ $# -eq 0 ]; then
    cmd_help >&2
    exit 2
  fi
  local key port child=0 rc=0 healthy="0"
  key=$(key)
  local res store
  if res=$(resolve_store read "$key"); then
    store=$(printf '%s' "$res" | cut -f1)
    if [ -f "$store/$key.tar.zst" ]; then
      cmd_restore "$key"
    else
      log "no archive for ${key: -16}; cold path"
    fi
  else
    log "no archive for ${key: -16}; cold path"
  fi

  if [ "${B12X_KCACHE_FORCE_AUTOTUNE_OFF:-}" = "true" ]; then
    export B12X_AUTOTUNE=0
    log "B12X_AUTOTUNE=0 exported (forced autotune off)"
  fi

  "$@" &
  child=$!
  trap 'kill -TERM $child 2>/dev/null' TERM INT

  port=$(argv_value --port)
  [ -n "$port" ] || port=8000

  # Poll readiness every 5 s, max 120 attempts (10 min). Never kill the serve
  # on timeout: skip save and keep supervising.
  local i=0 code
  while [ "$i" -lt 120 ]; do
    if ! kill -0 "$child" 2>/dev/null; then
      break  # child exited early; skip to wait
    fi
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 \
      "http://127.0.0.1:$port/health" 2>/dev/null)
    if [ "$code" = "200" ]; then
      healthy="1"
      break
    fi
    i=$((i + 1))
    sleep 5
  done

  if [ "$healthy" = "1" ]; then
    if [ "${B12X_KCACHE_SAVE:-1}" != "0" ]; then
      cmd_save "$key"
    fi
  else
    log "readiness not observed within 10 min; skipping save (serve continues)"
  fi

  wait "$child"
  rc=$?
  trap - TERM INT
  return $rc
}

