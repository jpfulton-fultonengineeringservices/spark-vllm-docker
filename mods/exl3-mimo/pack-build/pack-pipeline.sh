#!/usr/bin/env bash
# pack-pipeline.sh — orchestrate convert + repack with progress tracking.
#
# Sourced by the pack-build entrypoint. All functions use _monitor_write_status
# from pack-monitor.sh. Progress is parsed from stdout line-by-line; no
# background monitor process is needed because convert_model and repack both
# produce per-layer output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pack-monitor.sh"

# --- helpers -----------------------------------------------------------------

_detect_geometry() {
  python3 "${SCRIPT_DIR}/pack-model-info.py" "$1"
}

_moe_layer_count() {
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('moe_layer_count',0))"
}

_guess_total_layers() {
  # Try num_hidden_layers from config; fall back to moe_layer_count from detect.
  local cfg
  if [ -f "$1/config.json" ]; then
    cfg=$(python3 -c "
import sys,json
cfg=json.load(open('$1/config.json'))
tc=cfg.get('text_config',cfg)
print(tc.get('num_hidden_layers',0))
" 2>/dev/null) || cfg=0
    if [ "$cfg" -gt 0 ]; then
      echo "$cfg"
      return
    fi
  fi
  local info
  info=$(_detect_geometry "$1")
  echo "$info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('moe_layer_count',0))"
}

_create_dir() {
  if [ ! -d "$1" ]; then
    mkdir -p "$1"
  fi
}

# --- pipeline stages ---------------------------------------------------------

pipeline_preflight() {
  local src="$1" work="$2"
  echo "=== pack-build pre-flight ==="
  monitor_preflight "$src" "$work"
  if ! python3 -c "import exllamav3" >/dev/null 2>&1; then
    echo "[pipeline] ERROR: exllamav3 not importable" >&2
    return 1
  fi
  if ! python3 -c "import b12x" >/dev/null 2>&1; then
    echo "[pipeline] ERROR: b12x not importable" >&2
    return 1
  fi
  echo "[pipeline] all checks passed"
}

pipeline_detect() {
  local src="$1"
  echo "[pipeline] detecting model geometry..."
  _detect_geometry "$src" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))"
}

pipeline_convert() {
  local src="$1" out="$2" work="$3" bits="$4" codebook="$5"
  shift 5

  _create_dir "$out"
  _create_dir "$work"

  # Resume an interrupted job: exllamav3 saves per-module state under
  # $work/qtensors and requires an explicit -r. Auto-enable it when quantized
  # state exists and the output is still empty, so a crash mid-run (e.g. a
  # missing runtime dep) does not discard hours of quantization.
  local resume_flag=""
  if [ -d "$work/qtensors" ] && [ -n "$(ls -A "$work/qtensors" 2>/dev/null)" ] \
     && [ -z "$(ls -A "$out" 2>/dev/null)" ]; then
    resume_flag="-r"
    echo "[pipeline] resuming: found quantized state in ${work}/qtensors"
  fi

  local total_layers
  total_layers=$(_guess_total_layers "$src")
  local model_name
  model_name=$(basename "$src")

  echo "[pipeline] convert: ${src} -> ${out} (${codebook} K${bits})"
  echo "[pipeline] detected ${total_layers} layers"

  local started_at
  started_at=$(_timestamp)

  _monitor_write_status "$started_at" \
    "stage=convert" "phase=starting" \
    "model=${model_name}" "codebook=${codebook}" "bits=${bits}" \
    "layers_total=${total_layers}"

  local exit_code=0
  set +e
  python3 -u -m exllamav3.conversion.convert_model \
    -i "$src" -o "$out" -w "$work" \
    -b "$bits" -cb "$codebook" ${resume_flag} "$@" 2>&1 | \
  while IFS= read -r line; do
    echo "$line"
    local layer_num=""
    local expert_num=""
    local perr=""
    local gsc=""

    # Extract layer number from exllamav3 output (various formats).
    layer_num=$(echo "$line" | grep -oPi '(?:layer|quantizing|encoding|calibrat)\D*\K\d+' | head -1 || true)
    expert_num=$(echo "$line" | grep -oPi 'expert\D*\K\d+' | head -1 || true)
    perr=$(echo "$line" | grep -oP 'proxy_err\D*\K[\d.]+' | head -1 || true)
    gsc=$(echo "$line" | grep -oP 'g_sc\D*\K[\d.]+' | head -1 || true)

    if [ -n "$layer_num" ]; then
      local elapsed
      elapsed=$(($(_now_epoch) - $(date -d "$started_at" +%s 2>/dev/null || echo 0)))
      local eta=0
      if [ "$layer_num" -gt 0 ] 2>/dev/null && [ "$total_layers" -gt 0 ]; then
        eta=$((elapsed * (total_layers - layer_num) / layer_num))
      fi
      local detail=""
      if [ -n "$expert_num" ]; then detail="experts_${expert_num}"; fi
      _monitor_write_status "$started_at" \
        "stage=convert" "phase=quantizing" \
        "model=${model_name}" "codebook=${codebook}" "bits=${bits}" \
        "layers_total=${total_layers}" \
        "current_layer=${layer_num}" "layers_completed=${layer_num}" \
        "layer_detail=${detail}" \
        "proxy_err=${perr}" "g_sc=${gsc}" \
        "eta_seconds=${eta}" "elapsed_seconds=${elapsed}"
    fi
  done
  exit_code=${PIPESTATUS[0]}
  set -e

  if [ "$exit_code" -ne 0 ]; then
    echo "[pipeline] convert FAILED (exit ${exit_code})" >&2
    _monitor_write_status "$started_at" \
      "stage=convert" "phase=error" \
      "errors=[\"convert_model exited with code ${exit_code}\"]"
    return "$exit_code"
  fi

  _monitor_write_status "$started_at" "stage=convert" "phase=done"
  echo "[pipeline] convert complete: ${out}"
  return 0
}

pipeline_repack() {
  local pack="$1" v1_out="$2" bits="$3"
  shift 3

  local total_layers=0
  if [ -f "$pack/model.safetensors.index.json" ]; then
    total_layers=$(python3 -c "
import json
wm = json.load(open('$pack/model.safetensors.index.json'))['weight_map']
layers = {int(k.split('.')[2]) for k in wm if '.mlp.experts.' in k}
print(len(layers))
" 2>/dev/null || echo 0)
  fi

  _create_dir "$v1_out"

  echo "[pipeline] repack: ${pack} -> ${v1_out} (K${bits})"

  local started_at
  started_at=$(_timestamp)

  _monitor_write_status "$started_at" \
    "stage=repack" "phase=starting" "codebook=mcg" "bits=${bits}" \
    "layers_total=${total_layers}"

  local exit_code=0
  set +e
  python3 -u "${SCRIPT_DIR}/repack_exl3_to_b12x.py" \
    --pack "$pack" --out "$v1_out" --bits "$bits" "$@" 2>&1 | \
  while IFS= read -r line; do
    echo "$line"
    local ln
    ln=$(echo "$line" | grep -oP 'layer \K\d+' | head -1 || true)
    if [ -n "$ln" ]; then
      _monitor_write_status "$started_at" \
        "stage=repack" "phase=assembling" "codebook=mcg" "bits=${bits}" \
        "layers_total=${total_layers}" \
        "current_layer=${ln}" "layers_completed=${ln}"
    fi
  done
  exit_code=${PIPESTATUS[0]}
  set -e

  if [ "$exit_code" -ne 0 ]; then
    echo "[pipeline] repack FAILED (exit ${exit_code})" >&2
    _monitor_write_status "$started_at" \
      "stage=repack" "phase=error" \
      "errors=[\"repack exited with code ${exit_code}\"]"
    return "$exit_code"
  fi

  _monitor_write_status "$started_at" "stage=repack" "phase=done" "codebook=mcg" "bits=${bits}" \
    "layers_total=${total_layers}"
  echo "[pipeline] repack complete: ${v1_out}"
  return 0
}

pipeline_run() {
  local src="$1" work="$2" exl3_out="$3" v1_out="$4" bits="$5" codebook="$6"
  shift 6

  local started_total
  started_total=$(_timestamp)

  echo "============================================================"
  echo "pack-build pipeline start: $(date)"
  echo "  source:   ${src}"
  echo "  work:     ${work}"
  echo "  exl3:     ${exl3_out}"
  echo "  v1:       ${v1_out}"
  echo "  codebook: ${codebook}"
  echo "  bits:     ${bits}"
  echo "============================================================"

  _monitor_write_status "$started_total" \
    "stage=pipeline" "phase=preflight" \
    "model=$(basename "$src")" "codebook=${codebook}" "bits=${bits}"

  pipeline_preflight "$src" "$work"

  pipeline_detect "$src"

  _monitor_write_status "$started_total" "stage=convert" "phase=starting"
  local stage_start
  stage_start=$(_now_epoch)
  pipeline_convert "$src" "$exl3_out" "$work" "$bits" "$codebook" "$@"
  local convert_dur
  convert_dur=$(($(_now_epoch) - stage_start))
  echo "[pipeline] convert duration: $((convert_dur / 60))m $((convert_dur % 60))s"

  _monitor_write_status "$started_total" "stage=repack" "phase=starting"
  stage_start=$(_now_epoch)
  pipeline_repack "$exl3_out" "$v1_out" "$bits" --self-check
  local repack_dur
  repack_dur=$(($(_now_epoch) - stage_start))
  echo "[pipeline] repack duration: $((repack_dur / 60))m $((repack_dur % 60))s"

  local total_dur
  total_dur=$(($(_now_epoch) - $(date -d "$started_total" +%s 2>/dev/null || echo 0)))

  local v1_size
  v1_size=$(du -sh "$v1_out" 2>/dev/null | awk '{print $1}' || echo "?")

  echo "============================================================"
  echo "pack-build pipeline complete: $(date)"
  echo "  convert:  $((convert_dur / 60))m $((convert_dur % 60))s"
  echo "  repack:   $((repack_dur / 60))m $((repack_dur % 60))s"
  echo "  total:    $((total_dur / 60))m $((total_dur % 60))s"
  echo "  output:   ${v1_out} (${v1_size})"
  echo "============================================================"

  _monitor_write_status "$started_total" \
    "stage=pipeline" "phase=done" \
    "elapsed_seconds=${total_dur}" \
    "output_size=${v1_size}" \
    "convert_duration=${convert_dur}" \
    "repack_duration=${repack_dur}"

  return 0
}

pipeline_convert_direct() {
  local src="$1" out="$2" work="$3" bits="$4" codebook="$5"
  shift 5
  pipeline_preflight "$src" "$work"
  pipeline_convert "$src" "$out" "$work" "$bits" "$codebook" "$@"
}

pipeline_repack_direct() {
  local pack="$1" v1_out="$2" bits="$3"
  shift 3
  pipeline_repack "$pack" "$v1_out" "$bits" "$@"
}

# Library guard — when sourced, nothing runs.
# When executed directly, print usage.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "pack-pipeline.sh is a library; source it from the pack-build entrypoint." >&2
  exit 1
fi