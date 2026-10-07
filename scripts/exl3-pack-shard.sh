#!/usr/bin/env bash
# exl3-pack-shard.sh — parallel EXL3 packing across the 4 GX10 nodes.
#
# Runs ON a cluster node (jpfulton) using its own ssh aliases; it is NOT a
# workstation driver. Node-local NVMe holds per-shard work; NAS (/nas-1) holds
# the source checkpoint and the final merged pack.
#
# Subcommands:
#   stage  --model <slug> --nodes <n1,n2,...> [--source <path>]
#       rsync the source checkpoint to each node's <local_root>/staging/<slug>/.
#   run    --model <slug> --nodes <n1,...> --shards <N> [--recipe <path>]
#          [--bits <n>] [--dry-run]
#       Launch one detached EXL3 convert job per node, each on that node's
#       inclusive absolute module range (--module-start/--max_module). All
#       shards share ONE model-global recipe.yaml (recipe is shard-invariant).
#   merge  --model <slug> --nodes <n1,...> --merge-node <n> [--bits <n>]
#          [--dry-run]
#       rsync per-node qtensors to the merge node, validate + merge them into
#       one pack with -m exl3pack.merge (requires >=150GB free NVMe).
#
# Node rail IPs (rail-a 10.100.170.0/24): node1=.3 node2=.4 node3=.2 node4=.1.
# MTP parity: every shard uses convert_model's default mtp_bits=4; cli.py has
# no per-shard MTP override, so non-default MTP rates require cli.py wiring.
#
#   watch|status --model <slug> --nodes <n1,n2,...> [--interval <sec>] [--bits <n>]
#       Read-only: cat each node's .pack-status.json over ssh and render it with
#       the shared status.py. Single pass then exit unless --interval is given
#       on a tty.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXL3_SRC="${REPO_ROOT}/mods/exl3-pack/src"
PACK_STATUS_BIN="${EXL3_SRC}/exl3pack/status.py"

TAG="exl3-pack:cu13.0"
NOFILE_LIMIT="${EXL3_PACK_NOFILE:-1048576}"
NAS_ROOT="/nas-1"
LOCAL_ROOT="/opt/llm"
STATUS_NAME=".pack-status.json"
BITS=""
RECIPE=""
DRY_RUN=false

err() {
    echo "exl3-pack-shard: $*" >&2
}

usage() {
    # Everything after the shebang up to the first non-comment line, with the
    # '# ' prefix stripped.
    awk 'NR==1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"
}

# Node -> rail-a IP (10.100.170.<x>).
node_rail_ip() {
    case "$1" in
        node1|gx10-cb11) echo "10.100.170.3" ;;
        node2|gx10-f1d8) echo "10.100.170.4" ;;
        node3|gx10-becc) echo "10.100.170.2" ;;
        node4|gx10-9273) echo "10.100.170.1" ;;
        *) err "unknown node '$1'"; return 1 ;;
    esac
}

# Node -> ssh alias for in-fabric node access.
node_alias() {
    case "$1" in
        node1|gx10-cb11) echo "home-gx10-node1" ;;
        node2|gx10-f1d8) echo "home-gx10-node2" ;;
        node3|gx10-becc) echo "home-gx10-node3" ;;
        node4|gx10-9273) echo "home-gx10-node4" ;;
        *) err "unknown node '$1'"; return 1 ;;
    esac
}

run_or_echo() {
    if "$DRY_RUN"; then
        echo "$*"
    else
        eval "$@"
    fi
}

# Shared model-global recipe path (ONE per model). Defaults to NAS so every
# node can read the identical in-container path (/nas-1 is mounted everywhere);
# a node-local /opt/llm recipe would be invisible to the other shards.
recipe_path() {
    local model="$1"
    if [ -n "$RECIPE" ]; then
        echo "$RECIPE"
    else
        echo "${NAS_ROOT}/models/mimo/${model}-exl3-recipe/recipe.yaml"
    fi
}

# Emit the ONE shared recipe.yaml in-image on one node before launching shards.
# Idempotent: skipped when --recipe was supplied or the file already exists.
# NOTE: cli.py's convert exposes no --mtp-bits, so every shard uses convert_model's
# default mtp_bits (4) uniformly — MTP parity holds by construction. A non-default
# MTP rate would need a cli.py flag added before it can be sharded.
ensure_recipe() {
    local model="$1" in_recipe="$2" first_alias="$3"
    if [ -n "$RECIPE" ]; then
        echo "recipe: using provided ${RECIPE}"
        return 0
    fi
    if ! "$DRY_RUN" && ssh "$first_alias" "test -f '${in_recipe}'" 2>/dev/null; then
        echo "recipe: reusing ${in_recipe}"
        return 0
    fi
    echo "recipe: emitting shared ${in_recipe} on ${first_alias}"
    local cmd="docker run --rm --gpus all"
    cmd="${cmd} --ulimit nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT}"
    cmd="${cmd} -v ${NAS_ROOT}:${NAS_ROOT} -v ${LOCAL_ROOT}:/opt/llm"
    cmd="${cmd} '${TAG}'"
    cmd="${cmd} --model '${model}' recipe"
    cmd="${cmd} --source /opt/llm/staging/${model}"
    cmd="${cmd} --out $(printf '%q' "$in_recipe")"
    run_or_echo "ssh '${first_alias}' $(printf '%q' "$cmd")"
}

# Emit one "index<TAB>module_start<TAB>module_end<TAB>layers_csv" line per shard
# via the torch-free planner (exl3pack.shard).
plan_shards_tsv() {
    local model="$1" shards="$2" alias="$3"
    local src_path="${NAS_ROOT}/models/mimo/${model}"
    # The planner reads config.json/index.json from the NAS source, which only
    # exists on the nodes (the workstation has no /nas-1). Run it in the image
    # on a node; stdout is the TSV the caller parses.
    ssh "$alias" "docker run --rm -v ${NAS_ROOT}:${NAS_ROOT} --entrypoint python3 '${TAG}' \
        -m exl3pack.shard --model '${src_path}' --shards ${shards} --tsv"
}

# Stage the source checkpoint onto each node's local NVMe staging dir.
cmd_stage() {
    local model="" nodes="" source=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --model) model="${2:-}"; shift 2 ;;
            --nodes) nodes="${2:-}"; shift 2 ;;
            --source) source="${2:-}"; shift 2 ;;
            *) err "stage: unknown arg '$1'"; return 2 ;;
        esac
    done
    [ -n "$model" ] || { err "stage: --model required"; return 2; }
    [ -n "$nodes" ] || { err "stage: --nodes required"; return 2; }
    local src="${source:-${NAS_ROOT}/models/mimo/${model}}"
    local node
    for node in ${nodes//,/ }; do
        local ip alias dest
        ip="$(node_rail_ip "$node")"
        alias="$(node_alias "$node")"
        dest="${LOCAL_ROOT}/staging/${model}/"
        echo "stage: ${src} -> ${alias}(${ip}):${dest}"
        run_or_echo "rsync -a --delete -e 'ssh -o HostName=${ip}' '${src}/' '${alias}:${dest}'"
    done
}

# Launch one detached convert container per node, each on its module range.
cmd_run() {
    local model="" nodes="" shards=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --model) model="${2:-}"; shift 2 ;;
            --nodes) nodes="${2:-}"; shift 2 ;;
            --shards) shards="${2:-}"; shift 2 ;;
            --recipe) RECIPE="${2:-}"; shift 2 ;;
            --bits) BITS="${2:-}"; shift 2 ;;
            --dry-run) DRY_RUN=true; shift ;;
            *) err "run: unknown arg '$1'"; return 2 ;;
        esac
    done
    [ -n "$model" ] || { err "run: --model required"; return 2; }
    [ -n "$nodes" ] || { err "run: --nodes required"; return 2; }
    [ -n "$shards" ] || { err "run: --shards required"; return 2; }

    local work_k="k${BITS:-3}"
    local node_list="${nodes//,/ }"
    local node_count
    node_count="$(echo "$node_list" | wc -w | tr -d ' ')"
    [ "$node_count" = "$shards" ] || {
        err "run: --nodes count (${node_count}) must equal --shards (${shards})"
        return 2
    }

    local in_recipe
    in_recipe="$(recipe_path "$model")"
    local first_node first_alias
    first_node="$(echo "$node_list" | cut -d' ' -f1)"
    first_alias="$(node_alias "$first_node")"
    ensure_recipe "$model" "$in_recipe" "$first_alias"
    local plan
    plan="$(plan_shards_tsv "$model" "$shards" "$first_alias")"

    local i=0
    local idx start end layers_csv
    # The plan is read from fd 3 so the loop body's stdin (which the remote
    # launch consumes) can't swallow plan lines and stop after the first shard.
    while IFS="$(printf '\t')" read -r idx start end layers_csv <&3; do
        [ -n "$idx" ] || continue
        local node alias ip name work_rel out_rel status_file
        node="$(echo "$node_list" | cut -d' ' -f$((i + 1)))"
        alias="$(node_alias "$node")"
        ip="$(node_rail_ip "$node")"
        name="exl3-pack-job-${model}-${i}"
        work_rel="fes-projects/exl3-mimo-build/${model}-work-${work_k}/node${i}"
        out_rel="fes-projects/exl3-mimo-build/${model}-mcg-${work_k}/node${i}"
        status_file="${LOCAL_ROOT}/${work_rel}/${STATUS_NAME}"

        local cmd="docker run -d --rm --gpus all"
        cmd="${cmd} --ulimit nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT}"
        cmd="${cmd} --name '${name}'"
        cmd="${cmd} -v ${NAS_ROOT}:${NAS_ROOT} -v ${LOCAL_ROOT}:/opt/llm"
        cmd="${cmd} -e PACK_STATUS_FILE=${status_file}"
        cmd="${cmd} '${TAG}'"
        cmd="${cmd} --model '${model}' convert"
        cmd="${cmd} --source /opt/llm/staging/${model}"
        cmd="${cmd} --work /opt/llm/${work_rel}"
        cmd="${cmd} --exl3-out /opt/llm/${out_rel}"
        cmd="${cmd} --module-start ${start} --max_module ${end}"
        cmd="${cmd} --recipe $(printf '%q' "$in_recipe")"
        echo "run: ${alias}(${ip}) ${name} modules ${start}-${end}"
        run_or_echo "ssh '${alias}' $(printf '%q' "$cmd")"
        i=$((i + 1))
    done 3<<EOF
$plan
EOF
}

# rsync per-node qtensors to the merge node and merge into one pack.
cmd_merge() {
    local model="" nodes="" merge_node=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --model) model="${2:-}"; shift 2 ;;
            --nodes) nodes="${2:-}"; shift 2 ;;
            --merge-node) merge_node="${2:-}"; shift 2 ;;
            --bits) BITS="${2:-}"; shift 2 ;;
            --dry-run) DRY_RUN=true; shift ;;
            *) err "merge: unknown arg '$1'"; return 2 ;;
        esac
    done
    [ -n "$model" ] || { err "merge: --model required"; return 2; }
    [ -n "$nodes" ] || { err "merge: --nodes required"; return 2; }
    [ -n "$merge_node" ] || { err "merge: --merge-node required"; return 2; }

    local work_k="k${BITS:-3}"
    local merge_alias
    merge_alias="$(node_alias "$merge_node")"
    local merge_root="${LOCAL_ROOT}/fes-projects/exl3-mimo-build/${model}-merge-${work_k}"

    # Fail early if the merge node lacks NVMe headroom (>=150GB free).
    if "$DRY_RUN"; then
        echo "merge: require >=150GB free under ${LOCAL_ROOT} on ${merge_alias}"
    else
        local free_gb
        free_gb="$(ssh "$merge_alias" "df -BG --output=avail '${LOCAL_ROOT}' | tail -1 | tr -dc '0-9'")"
        case "$free_gb" in
            ''|*[!0-9]*) err "could not determine free NVMe space on ${merge_alias}"; return 1 ;;
        esac
        [ "$free_gb" -ge 150 ] || {
            err "merge node ${merge_alias} has ${free_gb}GB free; need at least 150GB"
            return 1
        }
    fi

    run_or_echo "ssh '${merge_alias}' 'mkdir -p \"${merge_root}\"'"

    local work_args=""
    local i=0 node
    for node in ${nodes//,/ }; do
        local alias ip qsrc qdst
        alias="$(node_alias "$node")"
        ip="$(node_rail_ip "$node")"
        qsrc="${LOCAL_ROOT}/fes-projects/exl3-mimo-build/${model}-work-${work_k}/node${i}/qtensors/"
        qdst="${merge_root}/node${i}/qtensors/"
        echo "merge: rsync ${alias}:${qsrc} -> ${merge_alias}:${qdst}"
        run_or_echo "ssh '${merge_alias}' $(printf '%q' "rsync -a -e 'ssh -o HostName=${ip}' '${alias}:${qsrc}' '${qdst}'")"
        work_args="${work_args} ${merge_root}/node${i}"
        i=$((i + 1))
    done

    local merge_cmd="docker run --rm --ulimit nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT}"
    merge_cmd="${merge_cmd} -v ${NAS_ROOT}:${NAS_ROOT} -v ${LOCAL_ROOT}:/opt/llm"
    merge_cmd="${merge_cmd} --entrypoint python3 '${TAG}' -m exl3pack.merge"
    merge_cmd="${merge_cmd} --work${work_args}"
    merge_cmd="${merge_cmd} --out /opt/llm/fes-projects/exl3-mimo-build/${model}-merge-${work_k}/out"
    merge_cmd="${merge_cmd} --source ${NAS_ROOT}/models/mimo/${model}"
    merge_cmd="${merge_cmd} --required-shared model.embed_tokens.safetensors"
    echo "merge: running ${merge_alias}"
    run_or_echo "ssh '${merge_alias}' $(printf '%q' "$merge_cmd")"
}

# Read-only status watch. The .pack-status.json files live on the node-local
# NVMe (/opt/llm), invisible on the workstation, so each node's file is cat'd
# over ssh and rendered by the shared status.py (same renderer the sibling
# exl3-pack-drive.sh uses). Paths and shard indices mirror cmd_run exactly.
status_file_for() {
    local model="$1" work_k="$2" i="$3"
    echo "${LOCAL_ROOT}/fes-projects/exl3-mimo-build/${model}-work-${work_k}/node${i}/${STATUS_NAME}"
}

shard_container_state() {
    local alias="$1" model="$2" i="$3"
    ssh "$alias" "docker inspect -f '{{.State.Status}}' 'exl3-pack-job-${model}-${i}' 2>/dev/null" \
        2>/dev/null || echo "gone"
}

cmd_watch() {
    local model="" nodes="" interval=5 interval_set=false
    while [ $# -gt 0 ]; do
        case "$1" in
            --model) model="${2:-}"; shift 2 ;;
            --nodes) nodes="${2:-}"; shift 2 ;;
            --interval) interval="${2:-}"; interval_set=true; shift 2 ;;
            --bits) BITS="${2:-}"; shift 2 ;;
            -h|--help) usage; return 0 ;;
            *) err "watch: unknown arg '$1'"; return 2 ;;
        esac
    done
    [ -n "$model" ] || { err "watch: --model required"; return 2; }
    [ -n "$nodes" ] || { err "watch: --nodes required"; return 2; }
    case "$interval" in
        ''|*[!0-9]*) err "watch: --interval must be a positive integer"; return 2 ;;
        0) err "watch: --interval must be a positive integer"; return 2 ;;
    esac

    local work_k="k${BITS:-3}"
    local node_list="${nodes//,/ }"
    local node_count
    node_count="$(echo "$node_list" | wc -w | tr -d ' ')"

    # Loop only when explicitly asked (--interval) and stdout is a tty; a
    # non-tty (pipe/CI) always does one pass and exits.
    local loop=true
    if [ ! -t 1 ] && [ "$interval_set" != true ]; then
        loop=false
    fi

    local first=true saw_any=false
    while :; do
        if [ "$loop" = true ]; then
            if [ "$first" = true ]; then
                printf '\033[2J\033[H'
                first=false
            else
                printf '\033[H'
            fi
        fi
        local terminal=0 i=0 node
        for node in $node_list; do
            local alias path tmp st
            alias="$(node_alias "$node")" || { err "watch: unknown node '${node}'"; return 2; }
            path="$(status_file_for "$model" "$work_k" "$i")"
            tmp="$(mktemp "${TMPDIR:-/tmp}/pack-status.XXXXXX")"
            echo "--- ${node} (${alias}) node${i} ---"
            if ssh "$alias" "cat '${path}' 2>/dev/null" >"$tmp" 2>/dev/null && [ -s "$tmp" ]; then
                saw_any=true
                python3 "$PACK_STATUS_BIN" "$tmp" || true
                if grep -qE '"phase"[[:space:]]*:[[:space:]]*"(done|error)"' "$tmp"; then
                    terminal=$((terminal + 1))
                else
                    st="$(shard_container_state "$alias" "$model" "$i")"
                    case "$st" in
                        exited*|gone*) terminal=$((terminal + 1)) ;;
                    esac
                fi
            else
                echo "waiting for ${path} ..."
                st="$(shard_container_state "$alias" "$model" "$i")"
                case "$st" in
                    exited*|gone*) if [ "$saw_any" = true ]; then terminal=$((terminal + 1)); fi ;;
                esac
            fi
            rm -f "$tmp"
            i=$((i + 1))
        done

        if [ "$loop" = false ]; then
            break
        fi
        if [ "$saw_any" = true ] && [ "$terminal" -ge "$node_count" ]; then
            break
        fi
        sleep "$interval"
    done
}

# --- arg dispatch ---
sub="${1:-help}"
[ $# -gt 0 ] && shift
case "$sub" in
    stage) cmd_stage "$@" ;;
    run) cmd_run "$@" ;;
    merge) cmd_merge "$@" ;;
    watch|status) cmd_watch "$@" ;;
    help|-h|--help) usage ;;
    *) err "unknown subcommand '$sub'"; usage; exit 2 ;;
esac
