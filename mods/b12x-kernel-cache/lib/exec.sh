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
  # Readiness budget: total seconds to poll /health before giving up on
  # saving (default 3600 = 60 min covers 30-45 min cold b12x/CuTe autotune;
  # a warm restore hits 200 in seconds). Poll every 5 s. Never kill serve on
  # timeout: skip save and keep supervising.
  local ready_timeout="${B12X_KCACHE_READY_TIMEOUT:-3600}"
  case "$ready_timeout" in
  ''|*[!0-9]*)
  log "invalid B12X_KCACHE_READY_TIMEOUT='$ready_timeout'; using 3600s"
  ready_timeout=3600
  ;;
  esac
  if [ "$ready_timeout" -eq 0 ]; then
  log "invalid B12X_KCACHE_READY_TIMEOUT='0'; using 3600s"
  ready_timeout=3600
  fi
  local attempts=$(( (ready_timeout + 4) / 5 ))
  local i=0 code exited="0"
  while [ "$i" -lt "$attempts" ]; do
    if ! kill -0 "$child" 2>/dev/null; then
      exited="1"
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
    if [ "$exited" = "1" ]; then
      log "readiness not observed (serve exited); skipping save"
    else
      log "readiness not observed within ${ready_timeout}s; skipping save (serve continues)"
    fi
  fi

  wait "$child"
  rc=$?
  trap - TERM INT
  return $rc
}

