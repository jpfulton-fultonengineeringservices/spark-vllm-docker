#!/usr/bin/env bash
# exl3-pack-drive.sh — drive the EXL3 pack builder on a remote GX10 node.
#
# Runs on the local workstation (macOS). Builds the exl3-pack image on a GPU
# node over SSH and drives the pack-build container there, streaming progress
# (layer / ETA) back to the local terminal.
#
# The build runs ON the node because the base image (vllm-node-b12x:latest)
# and the exllamav3 aarch64 wheel live there. Only the small build context
# (Dockerfile + mods/exl3-mimo) is rsynced over.
#
# Subcommands:
#   pipeline  <src> <work> <exl3-out> <v1-out> <bits> <codebook> [convert-args...]
#   convert   <src> <out> <work> <bits> <codebook> [convert-args...]
#   repack    <pack> <v1-out> <bits> [repack-args...]
#   assemble  <src> <pack> <serve-out> [assemble-args...]
#   detect    <src>
#   build                                   build the image on the node
#   sync                                    rsync build context to the node
#   status    <output>                      print current progress once
#   watch     <output>                      live progress, refresh each poll
#   help
#
# Global options (before or after subcommand):
#   -H, --host <alias>      SSH host alias; REQUIRED (no default)
#   -u, --user <user>       remote user (default: ssh_config alias's user)
#   --tag <tag>             image tag (default: exl3-pack:cu13.0)
#   --model <slug>          per-model spec to select (e.g. mimo-v2.6-pro-rl-uncensored)
#   --local-root <path>     node-local checkpoint root to mount (default: /opt/llm)
#   --nofile <n>            container nofile ulimit (default: 1048576; env
#                           EXL3_PACK_NOFILE). Raise for highly sharded sources.
#   --status-file <path>    override derived status file path
#   --name <name>           container name (default: exl3-pack-job)
#   --no-sync               skip rsync of build context
#   --no-build              skip docker build (assume image exists)
#   --detach                don't stream progress; print container + status
#   --dry-run               print the docker run command + mounts; do not run
#   --interval <sec>        poll interval (default: 5)
#
# The progress status file defaults to <output>/.pack-status.json, where
# <output> is the exl3 pack dir (convert) or the exl3-v1 dir (repack/pipeline).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACK_STATUS_BIN="${REPO_ROOT}/mods/exl3-pack/src/exl3pack/status.py"

# --- defaults -----------------------------------------------------------------
HOST=""
SSH_USER=""
TAG="exl3-pack:cu13.0"
STATUS_FILE=""
NAME="exl3-pack-job"
NO_SYNC=false
NO_BUILD=false
DETACH=false
POLL_INTERVAL=5
CONTEXT_DIR="/tmp/spark-vllm-docker-context"
MODEL=""
LOCAL_ROOT="/opt/llm"
DRY_RUN=false
NAS_ROOT="/nas-1"
# Docker nofile ulimit for the pack container. Highly sharded checkpoints
# (e.g. pro-rl-uncensored, 128 EP shards) exhaust Docker's default 1024-soft
# limit during parallel safetensors loading ("Too many open files", errno=24).
# Matches launch-cluster.sh / README's recommended 1048576. Override via
# --nofile or EXL3_PACK_NOFILE.
NOFILE_LIMIT="${EXL3_PACK_NOFILE:-1048576}"

# --- helpers ------------------------------------------------------------------

ssh_target() {
  if [ -n "$SSH_USER" ]; then
    printf '%s@%s' "$SSH_USER" "$HOST"
  else
    printf '%s' "$HOST"
  fi
}

usage() {
  # Portable header printer (BSD/GNU awk; macOS sed rejects the sed form).
  awk 'NR==1{next} /^#/{ sub(/^# ?/,""); print; next } { exit }' "$0"
}

err() {
  echo "exl3-pack-drive: $*" >&2
}

require_host() {
  if [ -z "$HOST" ]; then
    err "--host is required (no default). Use an ssh_config alias, e.g. home-gx10-node1."
    echo "" >&2
    usage >&2
    exit 2
  fi
}

derive_status_file() {
  local output="$1"
  if [ -n "$STATUS_FILE" ]; then
    printf '%s' "$STATUS_FILE"
  else
    printf '%s/.pack-status.json' "${output%/}"
  fi
}

# Print a docker run command (no side effects) and return 0 when --dry-run.
# Args: <status_file> <stage-arg-vector...>
dry_run_echo() {
  local status_file="$1"; shift
  local mounts=(-v /nas-1:/nas-1 -v "${LOCAL_ROOT}:/opt/llm")
  local ulim="--ulimit nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT}"
  if [ "$status_file" = "(none)" ]; then
    echo "docker run --rm --gpus all ${ulim} --name ${NAME} ${mounts[*]} ${TAG} $*"
  else
    echo "docker run --rm --gpus all ${ulim} --name ${NAME} ${mounts[*]} -e PACK_STATUS_FILE=${status_file} ${TAG} $*"
  fi
}

model_arg() {
  if [ -n "$MODEL" ]; then printf -- '--model %s' "$MODEL"; fi
}

# --- stages -------------------------------------------------------------------

do_sync() {
  echo "syncing build context to $(ssh_target):${CONTEXT_DIR}/ ..."
  # The generalized image COPYs: Dockerfile.exl3-pack, the shared lib
  # (mods/exl3-pack/), and every per-model spec dir (mods/<model>/pack-build/).
  # Mirror that subset of mods/ so the image build finds exactly what it COPYs.
  ssh "$(ssh_target)" "mkdir -p '${CONTEXT_DIR}/mods/exl3-pack' '${CONTEXT_DIR}/mods/mimo-v2.6-flash-rl-uncensored' '${CONTEXT_DIR}/mods/mimo-v2.6-pro-rl-uncensored'"
  rsync -a \
    --exclude '*.pyc' --exclude '__pycache__' --exclude '.DS_Store' --exclude '.venv' \
    --exclude 'tests' --exclude 'fixtures' \
    "${REPO_ROOT}/Dockerfile.exl3-pack" \
    "$(ssh_target):${CONTEXT_DIR}/Dockerfile.exl3-pack"
  rsync -a --delete \
    --exclude '*.pyc' --exclude '__pycache__' --exclude '.DS_Store' --exclude '.venv' \
    --exclude '.pytest_cache' --exclude '.mypy_cache' --exclude '.ruff_cache' \
    "${REPO_ROOT}/mods/exl3-pack/" \
    "$(ssh_target):${CONTEXT_DIR}/mods/exl3-pack/"
  local m
  for m in mimo-v2.6-flash-rl-uncensored mimo-v2.6-pro-rl-uncensored; do
  rsync -a --delete \
    --exclude '*.pyc' --exclude '__pycache__' --exclude '.DS_Store' \
      --exclude '.pytest_cache' --exclude '.mypy_cache' --exclude '.ruff_cache' \
      "${REPO_ROOT}/mods/${m}/pack-build/" \
      "$(ssh_target):${CONTEXT_DIR}/mods/${m}/pack-build/"
  done
  echo "sync complete."
}

do_build() {
  echo "building ${TAG} on $(ssh_target) ..."
  ssh "$(ssh_target)" \
    "docker build -f '${CONTEXT_DIR}/Dockerfile.exl3-pack' -t '${TAG}' '${CONTEXT_DIR}'"
  echo "build complete: ${TAG}"
}

ensure_dirs() {
  local mkdir_cmd=""
  local d
  for d in "$@"; do
    [ -n "$d" ] || continue
    if [ -n "$mkdir_cmd" ]; then mkdir_cmd="${mkdir_cmd} "; fi
    mkdir_cmd="${mkdir_cmd}'${d}'"
  done
  [ -n "$mkdir_cmd" ] || return 0
  ssh "$(ssh_target)" "mkdir -p ${mkdir_cmd}"
}

# --- container state ----------------------------------------------------------

container_state() {
  ssh "$(ssh_target)" \
    "docker inspect -f '{{.State.Status}}|{{.State.ExitCode}}' '${NAME}' 2>/dev/null" \
    || echo "gone|"
}

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

# --- subcommands --------------------------------------------------------------

cmd_pipeline() {
  [ $# -ge 6 ] || { err "pipeline needs <src> <work> <exl3-out> <v1-out> <bits> <codebook>"; exit 2; }
  local src="$1" work="$2" exl3_out="$3" v1_out="$4" bits="$5" codebook="$6"
  shift 6
  local status_file
  status_file="$(derive_status_file "$v1_out")"

  if [ "$DRY_RUN" = true ]; then
    dry_run_echo "$status_file" "$(model_arg)" "$@" \
      pipeline --source "$src" --work "$work" --exl3-out "$exl3_out" --v1-out "$v1_out"
    return 0
  fi

  [ "$NO_SYNC" = true ] || do_sync
  [ "$NO_BUILD" = true ] || do_build
  ensure_dirs "$work" "$exl3_out" "$v1_out"

  # ENTRYPOINT is `python3 -m exl3pack.cli`; do NOT repeat it here.
  local model_arg=()
  [ -n "$MODEL" ] && model_arg=(--model "$MODEL")
  local mounts=(-v /nas-1:/nas-1 -v "${LOCAL_ROOT}:/opt/llm")
  ssh "$(ssh_target)" docker run -d --rm --gpus all \
    --ulimit nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT} \
    --name "$NAME" \
    "${mounts[@]}" \
    -e "PACK_STATUS_FILE=${status_file}" \
    "$TAG" \
    "${model_arg[@]}" "$@" \
      pipeline --source "$src" --work "$work" --exl3-out "$exl3_out" --v1-out "$v1_out"

  if [ "$DETACH" = true ]; then
    echo "started container ${NAME}; status: ${status_file}"
    echo "  watch:  $0 watch --host ${HOST} ${v1_out}"
    echo "  logs:   ssh $(ssh_target) docker logs -f ${NAME}"
    return 0
  fi

  poll "$status_file"
  echo "--- container logs (tail) ---"
  ssh "$(ssh_target)" "docker logs '${NAME}' 2>&1 | tail -30" || true
}

cmd_convert() {
  [ $# -ge 5 ] || { err "convert needs <src> <out> <work> <bits> <codebook>"; exit 2; }
  local src="$1" out="$2" work="$3" bits="$4" codebook="$5"
  shift 5
  local status_file
  status_file="$(derive_status_file "$out")"

  if [ "$DRY_RUN" = true ]; then
    dry_run_echo "$status_file" "$(model_arg)" "$@" \
      convert --source "$src" --exl3-out "$out" --work "$work"
    return 0
  fi
  [ "$NO_SYNC" = true ] || do_sync
  [ "$NO_BUILD" = true ] || do_build
  ensure_dirs "$out" "$work"

  local model_arg=()
  [ -n "$MODEL" ] && model_arg=(--model "$MODEL")
  ssh "$(ssh_target)" docker run -d --rm --gpus all \
    --ulimit nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT} \
    --name "$NAME" \
    -v /nas-1:/nas-1 -v "${LOCAL_ROOT}:/opt/llm" \
    -e "PACK_STATUS_FILE=${status_file}" \
    "$TAG" \
    "${model_arg[@]}" "$@" \
      convert --source "$src" --exl3-out "$out" --work "$work"

  if [ "$DETACH" = true ]; then
    echo "started container ${NAME}; status: ${status_file}"
    echo "  watch:  $0 watch --host ${HOST} ${out}"
    return 0
  fi

  poll "$status_file"
  echo "--- container logs (tail) ---"
  ssh "$(ssh_target)" "docker logs '${NAME}' 2>&1 | tail -30" || true
}

cmd_repack() {
  [ $# -ge 3 ] || { err "repack needs <pack> <v1-out> <bits>"; exit 2; }
  local pack="$1" v1_out="$2" bits="$3"
  shift 3
  local status_file
  status_file="$(derive_status_file "$v1_out")"

  if [ "$DRY_RUN" = true ]; then
    dry_run_echo "$status_file" "$(model_arg)" "$@" \
      repack --exl3-out "$pack" --v1-out "$v1_out"
    return 0
  fi
  [ "$NO_SYNC" = true ] || do_sync
  [ "$NO_BUILD" = true ] || do_build
  ensure_dirs "$v1_out"

  local model_arg=()
  [ -n "$MODEL" ] && model_arg=(--model "$MODEL")
  ssh "$(ssh_target)" docker run -d --rm --gpus all \
    --ulimit nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT} \
    --name "$NAME" \
    -v /nas-1:/nas-1 -v "${LOCAL_ROOT}:/opt/llm" \
    -e "PACK_STATUS_FILE=${status_file}" \
    "$TAG" \
    "${model_arg[@]}" "$@" \
      repack --exl3-out "$pack" --v1-out "$v1_out"

  if [ "$DETACH" = true ]; then
    echo "started container ${NAME}; status: ${status_file}"
    echo "  watch:  $0 watch --host ${HOST} ${v1_out}"
    return 0
  fi

  poll "$status_file"
  echo "--- container logs (tail) ---"
  ssh "$(ssh_target)" "docker logs '${NAME}' 2>&1 | tail -30" || true
}

cmd_assemble() {
  [ $# -ge 3 ] || { err "assemble needs <src> <pack> <serve-out>"; exit 2; }
  local src="$1" pack="$2" serve_out="$3"
  shift 3
  local status_file
  status_file="$(derive_status_file "$serve_out")"

  if [ "$DRY_RUN" = true ]; then
    dry_run_echo "$status_file" "$(model_arg)" "$@" \
      assemble --source "$src" --exl3-out "$pack" --v1-out "$serve_out"
    return 0
  fi
  [ "$NO_SYNC" = true ] || do_sync
  [ "$NO_BUILD" = true ] || do_build
  ensure_dirs "$serve_out"

  local model_arg=()
  [ -n "$MODEL" ] && model_arg=(--model "$MODEL")
  ssh "$(ssh_target)" docker run -d --rm --gpus all \
    --ulimit nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT} \
    --name "$NAME" \
    -v /nas-1:/nas-1 -v "${LOCAL_ROOT}:/opt/llm" \
    -e "PACK_STATUS_FILE=${status_file}" \
    "$TAG" \
    "${model_arg[@]}" "$@" \
      assemble --source "$src" --exl3-out "$pack" --v1-out "$serve_out"

  if [ "$DETACH" = true ]; then
    echo "started container ${NAME}; status: ${status_file}"
    echo "  watch:  $0 watch --host ${HOST} ${serve_out}"
    return 0
  fi

  poll "$status_file"
  echo "--- container logs (tail) ---"
  ssh "$(ssh_target)" "docker logs '${NAME}' 2>&1 | tail -30" || true
}

cmd_detect() {
  [ $# -ge 1 ] || { err "detect needs <src>"; exit 2; }
  local src="$1"
  shift
  if [ "$DRY_RUN" = true ]; then
    dry_run_echo "(none)" "$(model_arg)" "$@" detect "$src"
    return 0
  fi
  [ "$NO_SYNC" = true ] || do_sync
  [ "$NO_BUILD" = true ] || do_build
  local model_arg=()
  [ -n "$MODEL" ] && model_arg=(--model "$MODEL")
  ssh "$(ssh_target)" docker run --rm \
    --ulimit nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT} \
    -v /nas-1:/nas-1 -v "${LOCAL_ROOT}:/opt/llm" \
    "$TAG" \
    "${model_arg[@]}" "$@" detect "$src"
}

cmd_build() {
  do_sync
  do_build
}

cmd_sync() {
  do_sync
}

cmd_status() {
  [ $# -ge 1 ] || { err "status needs <output> (or --status-file)"; exit 2; }
  local output="$1"
  local status_file
  status_file="$(derive_status_file "$output")"
  ssh "$(ssh_target)" "cat '${status_file}' 2>/dev/null" \
    | python3 "$PACK_STATUS_BIN" /dev/stdin 2>/dev/null \
    || err "no status file at ${status_file}"
}

cmd_watch() {
  [ $# -ge 1 ] || { err "watch needs <output> (or --status-file)"; exit 2; }
  local output="$1"
  local status_file
  status_file="$(derive_status_file "$output")"
  poll "$status_file"
}

# --- dispatch ----------------------------------------------------------------

subcommand="${1:-help}"
shift || true

POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    -H|--host) HOST="${2:-}"; shift 2 ;;
    -u|--user) SSH_USER="${2:-}"; shift 2 ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    --status-file) STATUS_FILE="${2:-}"; shift 2 ;;
    --name) NAME="${2:-}"; shift 2 ;;
    --no-sync) NO_SYNC=true; shift ;;
    --no-build) NO_BUILD=true; shift ;;
    --detach) DETACH=true; shift ;;
    --interval) POLL_INTERVAL="${2:-5}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --local-root) LOCAL_ROOT="${2:-/opt/llm}"; shift 2 ;;
    --nofile) NOFILE_LIMIT="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

# help does not need a host
case "$subcommand" in
  help|-h|--help) usage; exit 0 ;;
esac

require_host

case "$subcommand" in
  pipeline) cmd_pipeline "${POSITIONAL[@]}" ;;
  convert)  cmd_convert "${POSITIONAL[@]}" ;;
  repack)   cmd_repack "${POSITIONAL[@]}" ;;
  assemble) cmd_assemble "${POSITIONAL[@]}" ;;
  detect)   cmd_detect "${POSITIONAL[@]}" ;;
  build)    cmd_build ;;
  sync)     cmd_sync ;;
  status)   cmd_status "${POSITIONAL[@]}" ;;
  watch)    cmd_watch "${POSITIONAL[@]}" ;;
  help|-h|--help) usage ;;
  *)
    err "unknown subcommand '${subcommand}'"
    usage >&2
    exit 2
    ;;
esac