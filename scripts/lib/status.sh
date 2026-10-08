# Progress polling and status/watch verbs.
# Sourced by scripts/exl3-pack-drive.sh; defines functions only.
# Target: /bin/bash 3.2 (macOS). No top-level side effects.

# --- progress streaming -------------------------------------------------------

poll() {
  local status_file="$1"
  local tmp
  tmp="$(mktemp "${TMPDIR:-/tmp}/pack-status.XXXXXX")"

  tput civis 2>/dev/null || true
  cleanup_poll() {
    tput cnorm 2>/dev/null || true
    # `tmp` is local to poll(); the EXIT trap can fire after poll() returns,
    # when that local is out of scope — guard it against `set -u`.
    rm -f "${tmp:-}"
  }
  trap cleanup_poll EXIT

  local prev_done=-1
  local first_poll=true

  while true; do
    if ssh "$(ssh_target)" "cat '${status_file}' 2>/dev/null" > "$tmp" 2>/dev/null \
        && [ -s "$tmp" ]; then
      local cur_done
      cur_done=$(python3 -c "
import json,sys
d=json.load(open('$tmp'))
print(d.get('layers_completed',0) or 0)
" 2>/dev/null || echo 0)
      local delta=0
      if [ "$prev_done" -ge 0 ] 2>/dev/null && [ "$cur_done" -ge "$prev_done" ] 2>/dev/null; then
        delta=$((cur_done - prev_done))
      fi
      prev_done=$cur_done

      if [ "$first_poll" = true ]; then
        printf '\033[2J\033[H'
        first_poll=false
      else
        printf '\033[H'
      fi
      python3 "$PACK_STATUS_BIN" --delta "$delta" --interval "$POLL_INTERVAL" "$tmp"
      # Terminal phase: stop even if the container lingers (inspect races).
      if grep -qE '"phase"[[:space:]]*:[[:space:]]*"(done|error)"' "$tmp"; then
        printf '\n'
        break
      fi
    else
      printf '\033[H'
      printf 'waiting for %s ...\n' "$status_file"
    fi

    local state
    state="$(container_state)"
    case "$state" in
      exited*) printf '\n'; break ;;
      gone*)   printf '\n'; break ;;
    esac

    sleep "$POLL_INTERVAL"
  done
  # Normal exit: run cleanup now (restores cursor, removes tmp) rather than
  # leaving it to the EXIT trap, which may fire later — and clear the trap so
  # it does not refire on shell exit when poll()'s locals are gone.
  cleanup_poll
  trap - EXIT
}

cmd_status() {
  require_model
  local status_file
  status_file="$(derive_status_file)"
  ssh "$(ssh_target)" "cat '${status_file}' 2>/dev/null" \
    | python3 "$PACK_STATUS_BIN" /dev/stdin 2>/dev/null \
    || err "no status file at ${status_file}"
}

cmd_watch() {
  require_model
  poll "$(derive_status_file)"
}

# paths.resolve_destination(<slug>, ...) mirror.
v1_out_dir() {
  if [ -n "$V1_OUT" ]; then
    printf '%s' "$V1_OUT"
  else
    printf '%s/%s-exl3-v1' "${OUT_ROOT%/}" "$MODEL"
  fi
}

derive_status_file() {
  if [ -n "$STATUS_FILE" ]; then
    printf '%s' "$STATUS_FILE"
  else
    printf '%s/.pack-status.json' "$(v1_out_dir)"
  fi
}
