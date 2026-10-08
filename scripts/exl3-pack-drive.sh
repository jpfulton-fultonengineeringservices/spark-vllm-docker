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
#   typecheck   in-image type safety: stubtest (stubs vs installed wheel)
#               + mypy (src vs stubs). Read-only; needs --host only.
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

# Layout: this file holds the header playbook, defaults, argument parsing,
# cmd_dist_preflight, and the subcommand dispatch. Function bodies live in
# scripts/lib/, sourced in dependency order:
#   common.sh      ssh target, usage, host/model checks, pack argv builders
#   nodes.sh       node-map resolution and per-node SSH hosts
#   image.sh       build-context sync, image build, prep
#   containers.sh  container state queries and docker run construction
#   guard.sh       shared-WORK checks, stale stop-file and teardown guards
#   dist.sh        dist-run worker launch, stop-file signalling, stop
#   stages.sh      pack pipeline verbs (run_stage, plan, pack, convert, ...)
#   status.sh      progress polling, status and watch verbs
#   dist-cmds.sh   dist-coordinator, dist-worker, dist-run, teardown

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

# Module library: functions only. Sourced in dependency order.
for _mod in common nodes image containers guard dist stages status dist-cmds; do
  . "${SCRIPT_DIR}/lib/${_mod}.sh"
done
unset _mod


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
  typecheck) cmd_typecheck ;;
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
