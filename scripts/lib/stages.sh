# Pack pipeline stage verbs: run_stage, plan, pack/convert/repack/assemble/detect/plan.
# Sourced by scripts/exl3-pack-drive.sh; defines functions only.
# Target: /bin/bash 3.2 (macOS). No top-level side effects.

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
