#!/usr/bin/env bash
# pack-monitor.sh — progress tracking library for pack-build operations.
#
# Source this file to use its functions. It provides pre-flight checks,
# progress parsing, status-file writing, and resource sampling.
# Runs inside the exl3-pack container (Linux, bash ≥ 5, GNU coreutils).

set -euo pipefail

# --- configuration -----------------------------------------------------------

PACK_STATUS_FILE="${PACK_STATUS_FILE:-/work/.pack-status.json}"
PACK_STATUS_TMP="${PACK_STATUS_FILE}.tmp"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-10}"

# --- helpers -----------------------------------------------------------------

_timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

_now_epoch() { date +%s; }

_gpu_memory_mb() {
  # Print "used total" only when both are plain integers. nvidia-smi can report
  # [N/A] (MIG / driver quirk), which would otherwise corrupt the status JSON.
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | head -1 \
    | awk -F', *' 'NF>=2 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { print $1, $2 }'
}

_disk_free_gb() {
  local dir="${1:-/work}"
  df --output=avail "$dir" 2>/dev/null | tail -1 | awk '{printf "%.0f", $1/1024/1024}' || true
}

# --- status file -------------------------------------------------------------

# Write a JSON status snapshot atomically.
# Arguments after the first become key=value pairs merged into the JSON.
_monitor_write_status() {
  local started_at="$1"
  shift
  local now
  now=$(_timestamp)
  local elapsed
  elapsed=$(($(_now_epoch) - $(date -d "$started_at" +%s 2>/dev/null || echo 0)))

  local mem_info
  mem_info=$(_gpu_memory_mb)
  local gpu_used="" gpu_total=""
  if [ -n "$mem_info" ]; then
    gpu_used=$(echo "$mem_info" | awk '{print $1}')
    gpu_total=$(echo "$mem_info" | awk '{print $2}')
  fi

  local disk_free
  disk_free=$(_disk_free_gb "${PACK_WORK_DIR:-/work}")

  # Build JSON manually — no jq dependency.
  {
    printf '{\n'
    printf '  "started_at": "%s",\n' "$started_at"
    printf '  "last_update": "%s",\n' "$now"
    printf '  "elapsed_seconds": %d,\n' "$elapsed"
    if [[ "$gpu_used" =~ ^[0-9]+$ ]] && [[ "$gpu_total" =~ ^[0-9]+$ ]]; then
      printf '  "gpu_memory_used_mb": %s,\n' "$gpu_used"
      printf '  "gpu_memory_total_mb": %s,\n' "$gpu_total"
    fi
    if [ -n "$disk_free" ]; then
      printf '  "disk_free_gb": %s,\n' "$disk_free"
    fi
    local first=1
    for kv in "$@"; do
      local k="${kv%%=*}"
      local v="${kv#*=}"
      if [ "$first" -eq 1 ]; then
        first=0
      else
        printf ',\n'
      fi
      # Emit as string; integers pass through without quotes.
      if [[ "$v" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        printf '  "%s": %s' "$k" "$v"
      elif [ "$v" = "null" ] || [ "$v" = "true" ] || [ "$v" = "false" ]; then
        printf '  "%s": %s' "$k" "$v"
      else
        printf '  "%s": "%s"' "$k" "$(echo "$v" | sed 's/"/\\"/g')"
      fi
    done
    printf '\n}\n'
  } > "$PACK_STATUS_TMP"
  mv "$PACK_STATUS_TMP" "$PACK_STATUS_FILE"
}

# --- pre-flight checks -------------------------------------------------------

_monitor_check_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[monitor] WARNING: nvidia-smi not found; GPU monitoring disabled"
    return 0
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "[monitor] ERROR: nvidia-smi failed; no GPU available"
    return 1
  fi
  local mem
  mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "[monitor] GPU OK (free: ${mem} MB)"
}

_monitor_check_disk() {
  local dir="${1:-/work}"
  local avail
  avail=$(_disk_free_gb "$dir")
  if [ -z "$avail" ]; then
    echo "[monitor] WARNING: cannot check disk space on $dir"
    return 0
  fi
  # Require at least 100 GB free for safety.
  if [ "$avail" -lt 100 ]; then
    echo "[monitor] ERROR: only ${avail} GB free on $dir; need at least 100 GB"
    return 1
  fi
  echo "[monitor] disk OK ($dir: ${avail} GB free)"
}

_monitor_check_source() {
  local src="$1"
  if [ ! -d "$src" ]; then
    echo "[monitor] ERROR: source model directory not found: $src"
    return 1
  fi
  if [ ! -f "$src/config.json" ]; then
    echo "[monitor] ERROR: source model missing config.json: $src"
    return 1
  fi
  echo "[monitor] source OK: $src"
}

_monitor_check_mounts() {
  local src="$1"
  local work="$2"
  local out="$3"
  if [ ! -d "$src" ]; then
    echo "[monitor] ERROR: source not mounted: $src"
    return 1
  fi
  if [ ! -d "$work" ]; then
    echo "[monitor] ERROR: work dir not mounted: $work"
    return 1
  fi
  if [ ! -d "$out" ]; then
    echo "[monitor] ERROR: output dir not mounted: $out"
    return 1
  fi
}

# --- progress parsing --------------------------------------------------------

# Parse a line from convert_model stdout for layer/expert progress.
# Returns "layer=NN", "expert=NN", or empty string.
_parse_progress_line() {
  local line="$1"
  # exllamav3 convert_model patterns:
  #   "Layer 1/48" or "Quantizing layer 15" or "Layer 12 done"
  if echo "$line" | grep -qiE '(layer|quantiz|encoding|calibrat).*[0-9]'; then
    # Try to extract a layer number.
    local lnum
    lnum=$(echo "$line" | grep -oEi 'layer[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | head -1)
    if [ -n "$lnum" ]; then
      echo "layer=${lnum}"
    fi
    # Try to extract an expert number.
    local enum
    enum=$(echo "$line" | grep -oEi 'expert[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | head -1)
    if [ -n "$enum" ]; then
      echo "expert=${enum}"
    fi
    # Try to extract proxy_err or g_sc.
    local perr
    perr=$(echo "$line" | grep -oE 'proxy_err[[:space:]]*[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [ -n "$perr" ]; then
      echo "proxy_err=${perr}"
    fi
    local gsc
    gsc=$(echo "$line" | grep -oE 'g_sc[[:space:]]*[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if [ -n "$gsc" ]; then
      echo "g_sc=${gsc}"
    fi
  fi
}

# --- monitor background process ----------------------------------------------

# Background monitor loop: poll status file periodically.
# Run this in a background subshell before starting a stage.
# Usage: _monitor_start <stage_name> <started_at>
_monitor_bg_loop() {
  local stage="$1"
  local started_at="$2"
  while true; do
    sleep "$MONITOR_INTERVAL"
    _monitor_write_status "$started_at" \
      "stage=${stage}" \
      "model=${PACK_MODEL_NAME:-}" \
      "codebook=${PACK_CODEBOOK:-}" \
      "bits=${PACK_BITS:-}"
  done
}

# Start the monitor background loop. Returns the PID.
monitor_start() {
  local stage="$1"
  local started_at
  started_at=$(_timestamp)
  _monitor_bg_loop "$stage" "$started_at" &
  echo $!
}

# Stop the monitor background loop.
monitor_stop() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

# Write a final error status.
monitor_error() {
  local stage="$1"
  local started_at="$2"
  local msg="$3"
  _monitor_write_status "$started_at" \
    "stage=${stage}" \
    "phase=error" \
    "errors=[\"$(echo "$msg" | sed 's/"/\\"/g')\"]"
}

# Write a completion status.
monitor_done() {
  local stage="$1"
  local started_at="$2"
  _monitor_write_status "$started_at" \
    "stage=${stage}" \
    "phase=done"
}

# --- stdout capture with progress parsing ------------------------------------

# Run a command, tee its output to stderr (visible), and parse lines for
# progress. Updates the status file with layer/phase info.
# Usage: monitor_run <stage> <started_at> <total_layers> <cmd...>
monitor_run() {
  local stage="$1"
  local started_at="$2"
  local total_layers="${3:-0}"
  shift 3

  local last_layer=0
  local last_expert=0
  local perr=""
  local gsc=""
  local exit_code=0

  # Run the command, capturing stdout line by line.
  # Use a temp file to avoid subshell variable scoping issues.
  local tmp_out
  tmp_out=$(mktemp -p /tmp pack-monitor.XXXXXX)
  trap 'rm -f "$tmp_out"' RETURN

  set +e
  "$@" > "$tmp_out" 2>&1
  exit_code=$?
  set -e

  # Parse the output for progress.
  while IFS= read -r line; do
    echo "$line" >&2  # tee to stderr for Docker logs
    local parsed
    parsed=$(_parse_progress_line "$line")
    if [ -n "$parsed" ]; then
      for item in $parsed; do
        case "$item" in
          layer=*) last_layer="${item#layer=}" ;;
          expert=*) last_expert="${item#expert=}" ;;
          proxy_err=*) perr="${item#proxy_err=}" ;;
          g_sc=*) gsc="${item#g_sc=}" ;;
        esac
      done
      # Update status with latest progress.
      local extra="stage=${stage}"
      extra="${extra} layers_completed=${last_layer}"
      extra="${extra} layers_total=${total_layers}"
      extra="${extra} current_layer=${last_layer}"
      if [ "$last_expert" -gt 0 ]; then
        extra="${extra} layer_detail=experts ${last_expert}"
      fi
      if [ -n "$perr" ]; then
        extra="${extra} proxy_err=${perr}"
      fi
      if [ -n "$gsc" ]; then
        extra="${extra} g_sc=${gsc}"
      fi
      # Compute ETA: elapsed / done * remaining
      local elapsed
      elapsed=$(($(_now_epoch) - $(date -d "$started_at" +%s 2>/dev/null || echo 0)))
      if [ "$last_layer" -gt 0 ] && [ "$total_layers" -gt 0 ]; then
        local eta
        eta=$((elapsed * (total_layers - last_layer) / last_layer))
        extra="${extra} eta_seconds=${eta}"
      fi
      _monitor_write_status "$started_at" "$extra"
    fi
  done < "$tmp_out"

  rm -f "$tmp_out"
  trap - RETURN
  return "$exit_code"
}

# --- pre-flight (public) -----------------------------------------------------

monitor_preflight() {
  local src="$1"
  local work="${2:-/work}"
  echo "=== pack-build pre-flight ==="
  _monitor_check_gpu || return 1
  _monitor_check_disk "$work" || return 1
  _monitor_check_source "$src" || return 1
  echo "=== pre-flight OK ==="
}