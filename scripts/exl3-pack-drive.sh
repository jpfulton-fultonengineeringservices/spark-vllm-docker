#!/usr/bin/env bash
# exl3-pack-drive.sh — drive the EXL3 pack builder on a remote GX10 node.
#
# Runs on the local workstation (macOS). Builds the exl3-pack image on a GPU
# node over SSH and drives the pack-build container there, streaming progress
# (layer / ETA) back to the local terminal.
#
# The build runs ON the node because the base image (vllm-node-b12x:latest)
# and the exllamav3 aarch64 wheel live there. Only the small build context
# (Dockerfile + mods/) is rsynced over.
#
# ============================================================================
# DIST-RUN PLAYBOOK (multi-node EXL3 pack)
# ============================================================================
#
# One command starts a worker on every --nodes host, runs the coordinator on
# --host, then stops the workers. Teardown also runs on Ctrl-C or failure.
#
# STEP 0  DRY-RUN FIRST. Prints the commands; launches nothing. (VERIFIED)
#
#   scripts/exl3-pack-drive.sh dist-run --model mimo-v2.6-flash-rl-uncensored \
#     --host home-gx10-node1 --nodes gx10-cb11,gx10-f1d8 \
#     --no-build --no-sync --dry-run
#
#   Prints "launching worker on <host> (container <name>)" per node, then
#   the coordinator command. Nothing is synced, built, probed, or started.
#
# STEP 1  ARGUMENTS (VERIFIED)
#
#   --host    coordinator host. Runs the dist-coordinator container.
#   --nodes   comma-separated node ids from mods/exl3-pack/node-model-map.json.
#             Each id's "alias" is the ssh host for that node's worker.
#             One worker container per id, on GPU --device (default 0).
#             Containers: <name>-coord and <name>-worker-<node-id>.
#   --codebook  optional: mcg (default), lut_e4m3, or lut_fp16. Validated by
#             the driver and by the in-image CLI before any launch.
#   --debug   keep worker containers after they exit (drops --rm), so a failed
#             worker's logs survive for `docker logs <name>` on its host.
#             Off by default. Remove stopped containers by hand when done.
#
# STEP 2  WORK, the shared data directory
#
#   Defaults to /nas-1/fes-projects/exl3-mimo-build/<model>-work-k3.
#   It MUST be on /nas-1 (cluster-visible NFS). The coordinator writes every
#   node's inbox and reads every node's outputs under WORK. Override with
#   --work. --allow-node-local-work bypasses the check, single-node only.
#   (Default path VERIFIED; node-local override UNVERIFIED.)
#
# STEP 3  BEFORE ANY CONTAINER STARTS
#
#   Prep    Sync and build on each host; skip with --no-sync / --no-build.
#           The image was rebuilt on gx10-node1 with the codebook fix. (VERIFIED)
#   Probe   Write+remove inside the image, to catch NFS root-squash before
#           launch. Passes on gx10-node1. (VERIFIED)
#   Launch  Workers start detached; each is checked Running, else the run
#           aborts and prints its last 50 log lines. (failure path UNVERIFIED)
#   Coord   Coordinator runs in the foreground and streams progress.
#           (UNVERIFIED end-to-end: no real launch has completed yet.)
#
# STEP 4  STOP AND TEARDOWN
#
#   Workers poll for <WORK>/dist/stop-<node-id>. The driver writes those files
#   and docker-stops the worker containers on normal exit, coordinator failure,
#   and Ctrl-C/TERM (EXIT/INT/TERM trap). Exit code is the coordinator's.
#   VERIFIED once: a failed run wrote both stop-files and stopped both workers.
#   Ctrl-C and TERM paths are UNVERIFIED.
#
# KNOWN LIMITS (as of this revision)
#   * The codebook blocker is FIXED: cli.py now takes --codebook as a validated
#     string (default mcg). Parse-checked in the image; no real run yet.
#   * The in-image code is baked at build time. Rebuild without --no-build
#     after any mods/exl3-pack change.
#   * Default runs use --rm, so a worker that exits takes its logs with it.
#     Use --debug to keep them for inspection.
#   * Do NOT launch without explicit approval: a real run starts GPU work for
#     hours on both nodes.
#
# ============================================================================
# OTHER SUBCOMMANDS
# ============================================================================
#
# Single-node (the common case is one line):
#
#   exl3-pack-drive.sh pack --model mimo-v2.6-flash-rl-uncensored --host home-gx10-node1
#
# No terminal environment variables are required. EXL3_PACK_NOFILE is optional.
# Paths and bits are NOT required: they come from the model's PackSpec and the
# node map (mods/exl3-pack/src/exl3pack/paths.py). Codebook is NOT yet
# sourced from the spec for dist-run; see KNOWN BLOCKERS. The driver
# is a thin launcher — the in-image CLI resolves:
#
#   source    = node-local checkpoint, else /nas-1/models/mimo/<slug>
#   work      = <node-local>/fes-projects/exl3-mimo-build/<slug>
#   exl3-out  = <work>/exl3
#   v1-out    = /nas-1/models/mimo/<slug>-exl3-v1
#
# Subcommands:
#   pack        full pipeline: convert + repack (+ optional --assemble)
#   convert     quantization stage only
#   repack      EXL3 -> v1 repack stage only
#   assemble    build the servable v1 tree from source + pack
#   detect      print the detected architecture for a checkpoint
#   plan        print the resolved pipeline plan (paths, shards, recipe)
#   dist-coordinator distributed pack: coordinator role
#   dist-worker distributed pack: worker role
#   dist-run    distributed pack, one command: launches workers, runs the
#               coordinator, then stops workers (also on Ctrl-C or failure)
#   build       build the image on the node
#   sync        rsync the build context to the node
#   status      print current progress once
#   watch       live progress, refresh each poll
#   help
#
# Global options (before or after the subcommand):
#   -H, --host <alias>      SSH host alias; REQUIRED (no default)
#   -u, --user <user>       remote user (default: ssh_config alias's user)
#   --model <slug>          model to pack; selects its PackSpec (bits,
#                         codebook, geometry) and derives every path below.
#                         REQUIRED for pack/convert/repack/assemble/detect/plan.
#   --node <alias>          node alias for source/work derivation via the node
#                         map (default: derived from --host)
#   --src <path>            override the source checkpoint path
#   --work <path>           override the work dir
#   --exl3-out <path>       override the EXL3 pack output dir
#   --out <path>            override the final v1 output dir
#   --recipe <path>         pipeline recipe (default: derived from spec)
#   --cleanup <mode>        pipeline cleanup: none|work|all (default: work)
#   --assemble              pack: also run the assemble stage at the end
#   --bits <n>              override spec bits (rare; dist-coordinator only)
#   --codebook <n>          override spec codebook (rare; dist-coordinator only)
#   --nodes <n[,n...]>      dist-run/dist-coordinator: node ids (see above)
#   --node-map <path>       dist-run/dist-coordinator: node map override
#   --allow-node-local-work dist-run: permit a --work outside NAS_ROOT
#                         (single-node only; multi-node inboxes go invisible)
#   --gather-timeout <s>    dist-run/dist-coordinator: per-shard gather timeout
#   --checkpoint-interval <s>  dist-run/dist-coordinator: checkpoint cadence
#   --inbox <path>          dist-worker (standalone): shard inbox dir
#   --shared <path>         dist-worker (standalone): shared work dir
#   --stop <path>           dist-worker (standalone): stop-file path
#   --device <n>            dist-run/dist-worker: GPU index (default: 0)
#   --tag <tag>             image tag (default: exl3-pack:cu13.0)
#   --local-root <path>     node-local checkpoint/staging root (default: /opt/llm)
#   --nofile <n>            container nofile ulimit (default: 1048576; env
#                         EXL3_PACK_NOFILE). Raise for highly sharded sources.
#   --status-file <path>    override derived status file path
#   --name <name>           container name (default: exl3-pack-job)
#                         dist-run names: <name>-coord, <name>-worker-<node-id>
#   --no-sync               skip rsync of build context
#   --no-build              skip docker build (assume image exists)
#   --detach                don't stream progress; print container + status
#   --dry-run               print the docker run command; do not run
#   --interval <sec>        poll interval (default: 5)
#
# The progress status file defaults to <v1-out>/.pack-status.json.
#
# dist-run details:
#   WORK     defaults to /nas-1/fes-projects/exl3-mimo-build/<model>-work-k3.
#            It must be under /nas-1 (cluster-visible NFS). The coordinator writes
#            every node's inbox and reads every node's outputs under WORK, so a
#            node-local path stalls the run silently. Override with --work, or
#            pass --allow-node-local-work to bypass the check (single node only).
#   Prep    Syncs and builds the image on the coordinator host and every worker
#            host. Skip with --no-sync / --no-build when those are already staged.
#   Probe   Before launch, runs a real write+remove inside the image on every
#            host. This catches NFS root-squash before any container starts.
#   Stop    Workers poll for <WORK>/dist/stop-<node-id>. The driver writes those
#            files and docker-stops the worker containers on normal exit, on
#            coordinator failure, and on Ctrl-C/TERM (EXIT/INT/TERM trap).
#   Exit    The command returns the coordinator's exit code.

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
DRY_RUN=false
DEBUG=false
FORCE=false
NOFILE_LIMIT="${EXL3_PACK_NOFILE:-1048576}"
dist_run_teardown_done=false

# model / path derivation
MODEL=""
NODE=""
SRC=""
WORK=""
EXL3_OUT=""
V1_OUT=""
RECIPE=""
CLEANUP=""
ASSEMBLE=false
BITS=""
CODEBOOK=""
NODES=""
ALLOW_NODE_LOCAL_WORK=false
NODE_MAP=""
GATHER_TIMEOUT=""
CHECKPOINT_INTERVAL=""
INBOX=""
SHARED=""
STOP=""
DEVICE="0"
LOCAL_ROOT="/opt/llm"
NAS_ROOT="/nas-1"
OUT_ROOT="/nas-1/models/mimo"

# --- helpers ------------------------------------------------------------------

ssh_target() {
  local h="${1:-$HOST}"
  if [ -n "$SSH_USER" ]; then
    printf '%s@%s' "$SSH_USER" "$h"
  else
    printf '%s' "$h"
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

require_model() {
  if [ -z "$MODEL" ]; then
    err "--model is required for '$subcommand' (paths and bits/codebook derive from its PackSpec)."
    exit 2
  fi
}

# The node's ssh_config alias doubles as the node-map alias.
node_alias() {
  if [ -n "$NODE" ]; then
    printf '%s' "$NODE"
  else
    printf '%s' "$HOST"
  fi
}

# Resolve a --nodes slug to its ssh host alias via the node map.
# Falls back to the global --host when the slug is absent from the map.
node_ssh_host() {
  local slug="$1"
  local map="${NODE_MAP}"
  if [ -z "$map" ]; then
    map="${REPO_ROOT}/mods/exl3-pack/node-model-map.json"
  fi
  local alias=""
  if [ -n "$map" ] && [ -f "$map" ]; then
    alias="$(python3 - "$map" "$slug" <<'PY' 2>/dev/null
import sys, json
try:
    data = json.load(open(sys.argv[1]))
    print(data.get("nodes", {}).get(sys.argv[2], {}).get("alias", ""))
except Exception:
    pass
PY
    )"
  fi
  if [ -n "$alias" ]; then
    printf '%s' "$alias"
  else
    printf '%s' "$HOST"
  fi
}

# Unique ordered list of ssh hosts needed for a --nodes list plus the coordinator.
resolved_hosts() {
  local hosts="$HOST"
  local slug
  IFS=',' read -ra _nodes <<< "$NODES"
  for slug in "${_nodes[@]}"; do
    slug="${slug// /}"
    [ -n "$slug" ] || continue
    local h
    h="$(node_ssh_host "$slug")"
    case ",$hosts," in
      *",$h,"*) ;;
      *) hosts="$hosts,$h" ;;
    esac
  done
  printf '%s' "$hosts"
}

# Final v1 output dir, mirroring paths.resolve_destination(<slug>, out_root).
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

# --- remote argument assembly -------------------------------------------------
# The image ENTRYPOINT is `python3 -m exl3pack.cli`; its --model/--node/
# --node-map/--local-root/--out-root options are GLOBAL (argparse) and must
# precede the verb. Verb-specific options (--source/--work/...) follow it.
# Anything left unset is derived by the CLI from the PackSpec + node map.
#
# gargs  = global (pre-verb) argv; vargs = per-verb argv. build_cli_argv fills
# both from the driver's flags, emitting only options the caller set.
gargs=()
vargs=()

build_global_args() {
  gargs=()
  [ -n "$MODEL" ]      && gargs+=(--model "$MODEL")
  [ -n "$NODE" ]       && gargs+=(--node "$NODE")
  [ -n "$NODE_MAP" ]   && gargs+=(--node-map "$NODE_MAP")
  [ -n "$LOCAL_ROOT" ] && gargs+=(--local-root "$LOCAL_ROOT")
  [ -n "$OUT_ROOT" ]   && gargs+=(--out-root "$OUT_ROOT")
  return 0
}

# Verb path/param overrides. $1 selects which overrides apply:
#   paths   = --source/--work/--exl3-out/--v1-out/--recipe/--cleanup
#   source  = --source only (plan)
build_override_args() {
  vargs=()
  case "${1:-paths}" in
    paths)
      [ -n "$SRC" ]      && vargs+=(--source "$SRC")
      [ -n "$WORK" ]     && vargs+=(--work "$WORK")
      [ -n "$EXL3_OUT" ] && vargs+=(--exl3-out "$EXL3_OUT")
      [ -n "$V1_OUT" ]   && vargs+=(--v1-out "$V1_OUT")
      [ -n "$RECIPE" ]   && vargs+=(--recipe "$RECIPE")
      [ -n "$CLEANUP" ]  && vargs+=(--cleanup "$CLEANUP")
      ;;
    source)
      [ -n "$SRC" ] && vargs+=(--source "$SRC")
      [ -n "$V1_OUT" ] && vargs+=(--v1-out "$V1_OUT")
      ;;
  esac
  return 0
}

# --- build context ------------------------------------------------------------

do_sync() {
  local host="${1:-$HOST}"
  echo "syncing build context to $(ssh_target):${CONTEXT_DIR}/ ..."
  # The generalized image COPYs: Dockerfile.exl3-pack, the shared lib
  # (mods/exl3-pack/), and every per-model spec dir (mods/<model>/pack-build/).
  ssh "$(ssh_target "$host")" "mkdir -p '${CONTEXT_DIR}/mods/exl3-pack' '${CONTEXT_DIR}/mods/mimo-v2.6-flash-rl-uncensored' '${CONTEXT_DIR}/mods/mimo-v2.6-pro-rl-uncensored'"
  rsync -a \
    --exclude '*.pyc' --exclude '__pycache__' --exclude '.DS_Store' --exclude '.venv' \
    --exclude 'tests' --exclude 'fixtures' \
    "${REPO_ROOT}/Dockerfile.exl3-pack" \
    "$(ssh_target "$host"):${CONTEXT_DIR}/Dockerfile.exl3-pack"
  rsync -a --delete \
    --exclude '*.pyc' --exclude '__pycache__' --exclude '.DS_Store' --exclude '.venv' \
    --exclude '.pytest_cache' --exclude '.mypy_cache' --exclude '.ruff_cache' \
    "${REPO_ROOT}/mods/exl3-pack/" \
    "$(ssh_target "$host"):${CONTEXT_DIR}/mods/exl3-pack/"
  local m
  for m in mimo-v2.6-flash-rl-uncensored mimo-v2.6-pro-rl-uncensored; do
    rsync -a --delete \
      --exclude '*.pyc' --exclude '__pycache__' --exclude '.DS_Store' \
      --exclude '.pytest_cache' --exclude '.mypy_cache' --exclude '.ruff_cache' \
      "${REPO_ROOT}/mods/${m}/pack-build/" \
      "$(ssh_target "$host"):${CONTEXT_DIR}/mods/${m}/pack-build/"
  done
  echo "sync complete."
}

do_build() {
  local host="${1:-$HOST}"
  echo "building ${TAG} on $(ssh_target) ..."
  ssh "$(ssh_target "$host")" \
    "docker build -f '${CONTEXT_DIR}/Dockerfile.exl3-pack' -t '${TAG}' '${CONTEXT_DIR}'"
  echo "build complete: ${TAG}"
}

prep() {
  [ "$DRY_RUN" = true ] && return 0
  [ "$NO_SYNC" = true ] || do_sync
  [ "$NO_BUILD" = true ] || do_build
}

# Sync + build on every resolved host (coordinator host + each worker alias).
prep_all() {
  [ "$DRY_RUN" = true ] && return 0
  local hosts one
  hosts="$(resolved_hosts)"
  local IFS=','
  for one in $hosts; do
    one="${one// /}"
    [ -n "$one" ] || continue
    [ "$NO_SYNC" = true ] || do_sync "$one"
    [ "$NO_BUILD" = true ] || do_build "$one"
  done
}

# Verify the shared WORK root is writable from inside the container image
# (NFS root-squash would otherwise silently break the first container write).
check_shared_work_writable() {
  local w="${1:-$WORK}"
  [ -n "$w" ] || { err "check_shared_work_writable: no WORK set"; return 1; }
  [ "$DRY_RUN" = true ] && return 0
  # Real write probe: container uid 0 touches + removes a file under the shared
  # root. A bare `test -w` only checks permission bits and misses NFS root-squash,
  # so we attempt an actual write. Run on every resolved host (coordinator + each
  # worker) — workers write from their own hosts.
  local hosts one
  hosts="$(resolved_hosts)"
  local IFS=','
  for one in $hosts; do
    one="${one// /}"
    [ -n "$one" ] || continue
    ssh "$(ssh_target "$one")" "docker run --rm --entrypoint sh -v '${w}:/w' '${TAG}' -c 'touch /w/.exl3-write-probe && rm -f /w/.exl3-write-probe'" || {
        err "${w} is not writable inside the container on ${one} (NFS root-squash?). Aborting."
        return 1
      }
  done
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

# --- run plumbing -------------------------------------------------------------
# docker_run <mode> <status_file|"-"> <verb> [verb-args...]
#   mode = fg (attached, --rm) or bg (detached, -d --rm, status env).
# The full CLI argv is: <global args> <verb> <verb args>.
docker_run() {
  # Preserve the original (mode, status_env, verb) contract for run_stage/cmd_*.
  docker_run_host "$HOST" "$NAME" "$@"
}

# Like docker_run, but targets an arbitrary resolved ssh host and container name.
docker_run_host() {
  local host="$1"; shift
  local cname="$1"; shift
  local mode="$1" status_env="$2" verb="$3"
  shift 3
  build_global_args
  # --rm removes the container and its logs on exit. --debug keeps it so a
  # failed worker can be inspected with `docker logs` / `docker inspect`.
  local -a argv=(
    --gpus all
    --ulimit "nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT}"
    --name "$cname"
    -v "${NAS_ROOT}:${NAS_ROOT}"
    -v "${LOCAL_ROOT}:/opt/llm"
  )
  [ "$DEBUG" = true ] || argv=(--rm "${argv[@]}")
  if [ "$mode" = bg ]; then
    argv=(-d "${argv[@]}" -e "PACK_STATUS_FILE=${status_env}")
  fi
  if [ "$DRY_RUN" = true ]; then
    echo "docker run ${argv[*]} \\"
    echo "  \"${TAG}\" ${gargs[*]+"${gargs[*]}"} ${verb} $*"
    echo "(would ssh to $(ssh_target "$host"))"
    return 0
  fi
  ssh "$(ssh_target "$host")" docker run "${argv[@]}" \
    "$TAG" "${gargs[@]+"${gargs[@]}"}" "$verb" "$@"
}

worker_container_name() {
  printf '%s-worker-%s' "$NAME" "$1"
}

# Launch one detached dist-worker container per node in --nodes.
launch_workers() {
  [ -n "$NODES" ] || { err "launch_workers: --nodes required"; return 2; }
  local IFS=','
  local slug host cname
  for slug in $NODES; do
    slug="${slug// /}"
    [ -n "$slug" ] || continue
    host="$(node_ssh_host "$slug")"
    cname="$(worker_container_name "$slug")"
    echo "launching worker on ${host} (container ${cname})"
    docker_run_host "$host" "$cname" bg "" dist-worker \
      --inbox "${WORK}/dist/inbox/${slug}" \
      --shared "${WORK}" \
      --device "${DEVICE}" \
      --stop "${WORK}/dist/stop-${slug}"
    # `-d --rm` returns success even if the container dies at startup. Confirm
    # it is still running, or fail before the coordinator starts dispatching.
    if [ "$DRY_RUN" != true ]; then
      local running
      running="$(ssh "$(ssh_target "$host")" \
        "docker inspect -f '{{.State.Running}}' '${cname}' 2>/dev/null" || true)"
      if [ "$running" != "true" ]; then
        err "worker ${cname} on ${host} is not running after launch; its logs:"
        ssh "$(ssh_target "$host")" "docker logs --tail 50 '${cname}' 2>&1" >&2 || true
        return 1
      fi
    fi
  done
}

# Write the stop sentinel on every worker node (shared WORK => one touch per node).
signal_workers_done() {
  [ -n "$NODES" ] || return 0
  [ "$DRY_RUN" = true ] && return 0
  local IFS=','
  local slug host
  for slug in $NODES; do
    slug="${slug// /}"
    [ -n "$slug" ] || continue
    host="$(node_ssh_host "$slug")"
    echo "signaling worker stop on ${host} (${WORK}/dist/stop-${slug})"
    ssh "$(ssh_target "$host")" "touch '${WORK}/dist/stop-${slug}'" || true
  done
}

# Stop every worker container (teardown ownership, C2). Never blocks on one node.
stop_worker_containers() {
  [ -n "$NODES" ] || return 0
  [ "$DRY_RUN" = true ] && return 0
  local IFS=','
  local slug host cname
  for slug in $NODES; do
    slug="${slug// /}"
    [ -n "$slug" ] || continue
    host="$(node_ssh_host "$slug")"
    cname="$(worker_container_name "$slug")"
    echo "stopping worker container ${cname} on ${host}"
    ssh "$(ssh_target "$host")" "docker stop '${cname}'" || true
  done
}

# Run a long stage (convert/repack/assemble/pipeline) with progress streaming.
run_stage() {
  local verb="$1"; shift
  local status_file
  status_file="$(derive_status_file)"
  if [ "$DRY_RUN" = true ]; then
    docker_run bg "$status_file" "$verb" "$@"
    return 0
  fi
  prep
  docker_run bg "$status_file" "$verb" "$@"
  if [ "$DETACH" = true ]; then
    echo "started container ${NAME}; status: ${status_file}"
    echo "  watch:  $0 watch --host ${HOST} --model ${MODEL}"
    echo "  logs:   ssh $(ssh_target) docker logs -f ${NAME}"
    return 0
  fi
  poll "$status_file"
  echo "--- container logs (tail) ---"
  ssh "$(ssh_target)" "docker logs '${NAME}' 2>&1 | tail -30" || true
}

# --- subcommands --------------------------------------------------------------
#
# Every cmd_* builds two arrays:
#   gargs — global (pre-verb) options for exl3pack.cli
#   vargs — verb-specific options
# docker_run(mode, status_file|"-", verb, vargs...) handles the actual
# ssh/docker invocation, prepending gargs and the image tag.

# Run `plan` in the container and return the resolved JSON on stdout.
# plan is read-only, so this never syncs/builds — it uses the image already on
# the node. Runs even under --dry-run so path resolution stays accurate.
_run_plan() {
  build_global_args
  local -a argv=(
    --rm --gpus all
    --ulimit "nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT}"
    --name "${NAME}-plan"
    -v "${NAS_ROOT}:${NAS_ROOT}"
    -v "${LOCAL_ROOT}:/opt/llm"
  )
  ssh "$(ssh_target)" docker run "${argv[@]}" \
    "$TAG" "${gargs[@]+"${gargs[@]}"}" plan
}

# Extract a field from plan JSON on stdin.
_plan_field() {
  python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('$1',''))"
}

cmd_pack() {
  require_model
  build_global_args
  build_override_args paths
  if [ "$ASSEMBLE" = true ]; then
    run_stage pipeline "${vargs[@]+"${vargs[@]}"}"
    # assemble needs source + pack + serve-out; reuse same overrides
    run_stage assemble "${vargs[@]+"${vargs[@]}"}"
  else
    run_stage pipeline "${vargs[@]+"${vargs[@]}"}"
  fi
}

cmd_convert() {
  require_model
  build_global_args
  build_override_args paths
  run_stage convert "${vargs[@]+"${vargs[@]}"}"
}

cmd_repack() {
  require_model
  build_global_args
  build_override_args paths
  run_stage repack "${vargs[@]+"${vargs[@]}"}"
}

cmd_assemble() {
  require_model
  build_global_args
  build_override_args paths
  run_stage assemble "${vargs[@]+"${vargs[@]}"}"
}

cmd_detect() {
  require_model
  build_global_args
  # detect takes source as a positional argument, not --source.
  # If --src was given, use it; otherwise derive from the plan.
  local src="$SRC"
  if [ -z "$src" ]; then
    local plan_json
    plan_json="$(_run_plan)"
    src="$(echo "$plan_json" | _plan_field source)"
  fi
  [ -n "$src" ] || { err "detect: could not resolve source (pass --src)"; exit 2; }
  [ "$DRY_RUN" = true ] || prep
  docker_run fg - detect "$src"
}

cmd_plan() {
  require_model
  _run_plan
}

cmd_dist_coordinator() {
  require_model
  [ -n "$NODES" ] || { err "dist-coordinator: --nodes required"; exit 2; }
  build_global_args
  # dist-coordinator requires --source/--work/--exl3-out/--recipe (required=True).
  # Resolve them from the plan unless the caller overrode every one.
  local src="$SRC" work="$WORK" exl3_out="$EXL3_OUT" recipe="$RECIPE"
  local plan_json=""
  if [ -z "$src" ] || [ -z "$work" ] || [ -z "$exl3_out" ] || [ -z "$recipe" ]; then
    plan_json="$(_run_plan)"
    [ -n "$src" ]      || src="$(echo "$plan_json" | _plan_field source)"
    [ -n "$work" ]     || work="$(echo "$plan_json" | _plan_field work)"
    [ -n "$exl3_out" ] || exl3_out="$(echo "$plan_json" | _plan_field exl3_out)"
    [ -n "$recipe" ]   || recipe="${work}/recipe.yaml"
  fi
  [ -n "$src" ]      || { err "dist-coordinator: could not resolve --source"; exit 2; }
  [ -n "$work" ]     || { err "dist-coordinator: could not resolve --work"; exit 2; }
  [ -n "$exl3_out" ] || { err "dist-coordinator: could not resolve --exl3-out"; exit 2; }
  [ -n "$recipe" ]   || { err "dist-coordinator: could not resolve --recipe"; exit 2; }
  vargs=(--source "$src" --work "$work" --exl3-out "$exl3_out" --recipe "$recipe")
  [ -n "$BITS" ]     && vargs+=(--bits "$BITS")
  [ -n "$CODEBOOK" ] && vargs+=(--codebook "$CODEBOOK")
  vargs+=(--nodes "$NODES")
  [ -n "$GATHER_TIMEOUT" ]      && vargs+=(--gather-timeout "$GATHER_TIMEOUT")
  [ -n "$CHECKPOINT_INTERVAL" ] && vargs+=(--checkpoint-interval "$CHECKPOINT_INTERVAL")
  # dist-coordinator has its own --node-map (not the global one)
  [ -n "$NODE_MAP" ] && vargs+=(--node-map "$NODE_MAP")
  [ "$DRY_RUN" = true ] || prep
  docker_run fg - dist-coordinator "${vargs[@]}"
}

cmd_dist_worker() {
  [ -n "$INBOX" ]  || { err "dist-worker: --inbox required"; exit 2; }
  [ -n "$SHARED" ] || { err "dist-worker: --shared required"; exit 2; }
  [ -n "$STOP" ]   || { err "dist-worker: --stop required"; exit 2; }
  build_global_args
  [ "$DRY_RUN" = true ] || prep
  docker_run fg - dist-worker --inbox "$INBOX" --shared "$SHARED" --device "$DEVICE" --stop "$STOP"
}

# Reject a node-local WORK for the distributed path (coordinator writes every
# node's inbox/outputs under one --work; must be cluster-visible shared storage).
validate_shared_work() {
  [ -n "$WORK" ] || return 0
  case "$WORK" in
    "${NAS_ROOT%/}/"*|"${NAS_ROOT%/}") : ok ;;
    *)
      if [ "$ALLOW_NODE_LOCAL_WORK" = true ]; then
        echo "WARNING: --work '$WORK' is not under NAS_ROOT ($NAS_ROOT); multi-node inboxes may be invisible." >&2
      else
        err "--work '$WORK' must be under NAS_ROOT ($NAS_ROOT) for distributed runs."
        err "Pass --allow-node-local-work to override (single-node only)."
        exit 2
      fi
      ;;
  esac
}

# Pre-launch guard for dist-run. Fails closed if a worker or coordinator
# container with this run's names is already alive on its host: a second copy
# would race the same inbox. Also clears STALE stop-files for this run's nodes:
# a leftover stop-file makes a new worker exit at once with status 0 and no
# logs (the silent failure seen on gx10-node1). Dry-run reports, changes nothing.
preflight_dist_run() {
  local IFS=',' slug host cname running
  for slug in $NODES; do
    slug="${slug// /}"
    [ -n "$slug" ] || continue
    host="$(node_ssh_host "$slug")"
    cname="$(worker_container_name "$slug")"
    if [ "$DRY_RUN" = true ]; then
      echo "preflight: would check ${cname} on ${host} and clear ${WORK}/dist/stop-${slug}"
      continue
    fi
    running="$(ssh "$(ssh_target "$host")" \
      "docker inspect -f '{{.State.Running}}' '${cname}' 2>/dev/null" || true)"
    if [ "$running" = "true" ]; then
      if [ "$FORCE" != true ]; then
        err "preflight: ${cname} is already running on ${host}. Stop it first, or rerun with --force."
        return 1
      fi
      teardown_container "$host" "$cname" "$WORK" "$slug"
    fi
    ssh "$(ssh_target "$host")" "rm -f '${WORK}/dist/stop-${slug}'" || {
      err "preflight: could not clear stale ${WORK}/dist/stop-${slug} on ${host}."
      return 1
    }
  done
  local coord_running
  coord_running="$(ssh "$(ssh_target "$HOST")" \
    "docker inspect -f '{{.State.Running}}' '${NAME}-coord' 2>/dev/null" || true)"
  if [ "$coord_running" = "true" ]; then
    if [ "$FORCE" != true ]; then
      err "preflight: ${NAME}-coord is already running on ${HOST}. Stop it first, or rerun with --force."
      return 1
    fi
    teardown_container "$HOST" "${NAME}-coord" "" ""
  fi
}

# Tear down one live container for --force. Workers: write the stop-file first
# so the worker exits between shards, then docker stop/rm. Coordinator has no
# stop-file, so it is stopped directly.
teardown_container() {
  local host="$1" cname="$2" work="$3" slug="$4"
  echo "force: tearing down ${cname} on ${host}"
  if [ -n "$slug" ]; then
    ssh "$(ssh_target "$host")" "touch '${work}/dist/stop-${slug}'" || return 1
    sleep 5
  fi
  ssh "$(ssh_target "$host")" "docker stop -t 20 '${cname}' >/dev/null 2>&1; docker rm -f '${cname}' >/dev/null 2>&1; true" || return 1
}

cmd_dist_run() {
  case "$CODEBOOK" in
    ""|mcg|lut_e4m3|lut_fp16) : ok ;;
    *) err "dist-run: --codebook must be mcg, lut_e4m3, or lut_fp16 (got '$CODEBOOK')"; exit 2 ;;
  esac
  require_model
  [ -n "$NODES" ] || { err "dist-run: --nodes required"; exit 2; }
  # Default WORK to cluster-visible shared storage so coordinator-written
  # inboxes/outputs are visible to every worker (C1). Caller may override
  # with --work (validated below) or --allow-node-local-work.
  if [ -z "$WORK" ]; then
    WORK="${NAS_ROOT}/fes-projects/exl3-mimo-build/${MODEL}-work-k3"
  fi
  # Resolve coordinator-side paths from the plan unless overridden.
  local src="$SRC" work="$WORK" exl3_out="$EXL3_OUT" recipe="$RECIPE"
  if [ -z "$src" ] || [ -z "$work" ] || [ -z "$exl3_out" ] || [ -z "$recipe" ]; then
    local plan_json
    plan_json="$(_run_plan)"
    [ -n "$src" ]      || src="$(echo "$plan_json" | _plan_field source)"
    # Keep the NAS default for work even if the plan reports a node-local path.
    [ -n "$work" ] && case "$work" in
      "${NAS_ROOT%/}/"*|"${NAS_ROOT%/}") ;;
      *) work="$WORK" ;;
    esac
    [ -n "$exl3_out" ] || exl3_out="$(echo "$plan_json" | _plan_field exl3_out)"
    [ -n "$recipe" ]   || recipe="${work}/recipe.yaml"
  fi
  [ -n "$src" ]      || { err "dist-run: could not resolve --source"; exit 2; }
  [ -n "$work" ]     || { err "dist-run: could not resolve --work"; exit 2; }
  [ -n "$exl3_out" ] || { err "dist-run: could not resolve --exl3-out"; exit 2; }
  [ -n "$recipe" ]   || { err "dist-run: could not resolve --recipe"; exit 2; }
  WORK="$work"
  validate_shared_work
  [ "$DRY_RUN" = true ] || { prep_all; check_shared_work_writable "$WORK"; }
  # Teardown runs on every exit path (normal return, set -e failure, Ctrl-C,
  # TERM): workers poll for the stop sentinel and otherwise never exit.
  dist_run_teardown_done=false
  trap 'dist_run_teardown' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  launch_workers
  local coord_rc=0
  build_global_args
  vargs=(--source "$src" --work "$work" --exl3-out "$exl3_out" --recipe "$recipe")
  [ -n "$BITS" ]     && vargs+=(--bits "$BITS")
  [ -n "$CODEBOOK" ] && vargs+=(--codebook "$CODEBOOK")
  vargs+=(--nodes "$NODES")
  [ -n "$GATHER_TIMEOUT" ]      && vargs+=(--gather-timeout "$GATHER_TIMEOUT")
  [ -n "$CHECKPOINT_INTERVAL" ] && vargs+=(--checkpoint-interval "$CHECKPOINT_INTERVAL")
  [ -n "$NODE_MAP" ] && vargs+=(--node-map "$NODE_MAP")
  preflight_dist_run || return 1
  docker_run_host "$HOST" "$NAME-coord" fg - dist-coordinator "${vargs[@]}" || coord_rc=$?
  echo "dist-run: coordinator exited rc=${coord_rc}; tearing down workers."
  return "$coord_rc"
}

# Idempotent worker teardown for the dist-run EXIT trap: write stop sentinels,
# then docker-stop each worker container. Safe to run more than once.
dist_run_teardown() {
  [ "${dist_run_teardown_done}" = true ] && return 0
  dist_run_teardown_done=true
  trap - EXIT
  signal_workers_done
  stop_worker_containers
}

cmd_build() {
  do_sync
  do_build
}

cmd_sync() {
  do_sync
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

# --- dispatch -----------------------------------------------------------------

# Global options may appear before OR after the subcommand. The first token
# that is not a recognized option (or an option's value) is the subcommand;
# all arguments are named — there are no positional path arguments anymore.
subcommand=""

while [ $# -gt 0 ]; do
  case "$1" in
    -H|--host) HOST="${2:-}"; shift 2 ;;
    -u|--user) SSH_USER="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --node) NODE="${2:-}"; shift 2 ;;
    --src|--source) SRC="${2:-}"; shift 2 ;;
    --work) WORK="${2:-}"; shift 2 ;;
    --exl3-out) EXL3_OUT="${2:-}"; shift 2 ;;
    --out|--v1-out) V1_OUT="${2:-}"; shift 2 ;;
    --recipe) RECIPE="${2:-}"; shift 2 ;;
    --cleanup) CLEANUP="${2:-}"; shift 2 ;;
    --assemble) ASSEMBLE=true; shift ;;
    --bits) BITS="${2:-}"; shift 2 ;;
    --codebook) CODEBOOK="${2:-}"; shift 2 ;;
    --nodes) NODES="${2:-}"; shift 2 ;;
    --node-map) NODE_MAP="${2:-}"; shift 2 ;;
    --allow-node-local-work) ALLOW_NODE_LOCAL_WORK=true; shift ;;
    --gather-timeout) GATHER_TIMEOUT="${2:-}"; shift 2 ;;
    --checkpoint-interval) CHECKPOINT_INTERVAL="${2:-}"; shift 2 ;;
    --inbox) INBOX="${2:-}"; shift 2 ;;
    --shared) SHARED="${2:-}"; shift 2 ;;
    --stop) STOP="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-0}"; shift 2 ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    --status-file) STATUS_FILE="${2:-}"; shift 2 ;;
    --name) NAME="${2:-}"; shift 2 ;;
    --no-sync) NO_SYNC=true; shift ;;
    --no-build) NO_BUILD=true; shift ;;
    --detach) DETACH=true; shift ;;
    --interval) POLL_INTERVAL="${2:-5}"; shift 2 ;;
    --local-root) LOCAL_ROOT="${2:-/opt/llm}"; shift 2 ;;
    --nofile) NOFILE_LIMIT="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --debug) DEBUG=true; shift ;;
    --force) FORCE=true; shift ;;
    -h|--help)
      case "$subcommand" in
        "") usage; exit 0 ;;
        *)  usage; exit 0 ;;
      esac ;;
    -*)
      err "unknown option '$1'"
      usage >&2
      exit 2 ;;
    *)
      if [ -z "$subcommand" ]; then
        subcommand="$1"
      else
        err "unexpected positional argument '$1' (all arguments are named; see help)"
        exit 2
      fi
      shift ;;
  esac
done

# A lone invocation (no subcommand) prints usage.
subcommand="${subcommand:-help}"

# help does not need a host
case "$subcommand" in
  help|-h|--help) usage; exit 0 ;;
esac

# Validate the nofile value before it is embedded into the remote docker run.
case "$NOFILE_LIMIT" in
  ''|*[!0-9]*) err "--nofile/EXL3_PACK_NOFILE must be a positive integer (got '${NOFILE_LIMIT}')"; exit 2 ;;
  0) err "--nofile/EXL3_PACK_NOFILE must be > 0"; exit 2 ;;
esac

# Validate cleanup mode early (it is forwarded verbatim to the CLI).
if [ -n "$CLEANUP" ]; then
  case "$CLEANUP" in
    none|work|all) ;;
    *) err "--cleanup must be one of: none, work, all (got '${CLEANUP}')"; exit 2 ;;
  esac
fi

require_host

cmd_dist_preflight() {
  require_model
  [ -n "$NODES" ] || { err "dist-preflight: --nodes required"; exit 2; }
  [ -n "$WORK" ] || WORK="${NAS_ROOT}/fes-projects/exl3-mimo-build/${MODEL}-work-k3"
  preflight_dist_run
}

case "$subcommand" in
  pack)       cmd_pack ;;
  convert)    cmd_convert ;;
  repack)     cmd_repack ;;
  assemble)   cmd_assemble ;;
  detect)     cmd_detect ;;
  plan)       cmd_plan ;;
  dist-coordinator) cmd_dist_coordinator ;;
  dist-worker) cmd_dist_worker ;;
  dist-run) cmd_dist_run ;;
  dist-preflight) cmd_dist_preflight ;;
  build) cmd_build ;;
  sync)       cmd_sync ;;
  status)     cmd_status ;;
  watch)      cmd_watch ;;
  *)
    err "unknown subcommand '${subcommand}'"
    usage >&2
    exit 2
    ;;
esac

