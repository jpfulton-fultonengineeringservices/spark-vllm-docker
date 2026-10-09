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
    echo "  logs:   ssh $(ssh_target) 'docker logs -f ${NAME} 2>&1 | tee -a $(v1_out_dir)/.pack-build.log'"
    return 0
  fi
  # Persist stdout/stderr on the node (--rm destroys container logs on exit).
  # Always under v1-out (never $WORK): v1-out is the durable NAS destination.
  local v1 log_pid
  v1="$(v1_out_dir)"
  ssh "$(ssh_target)" \
    "{ mkdir -p '${v1}' 2>/dev/null || true; } && docker logs -f '${NAME}' 2>&1 | tee -a '${v1}/.pack-build.log' >/dev/null" &
  log_pid=$!
  poll "$status_file"
  wait "$log_pid" 2>/dev/null || true
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

# Run the in-image type-safety verb: stubtest (stubs vs the installed pinned
# wheel) then mypy (src vs stubs, repo config). Read-only like plan — it
# validates what is already baked into the image, so it never syncs/builds;
# rebuild first (drop --no-build) after any mods/exl3-pack change.
cmd_typecheck() {
  local -a argv=(
    --rm --gpus all
    --ulimit "nofile=${NOFILE_LIMIT}:${NOFILE_LIMIT}"
    --name "${NAME}-typecheck"
    -v "${NAS_ROOT}:${NAS_ROOT}"
    -v "${LOCAL_ROOT}:/opt/llm"
  )
  if [ "$DRY_RUN" = true ]; then
    echo "ssh $(ssh_target) docker run ${argv[*]} $TAG typecheck"
    return 0
  fi
  ssh "$(ssh_target)" docker run "${argv[@]}" "$TAG" typecheck
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
    # Chained assemble: pack source is the repack output (v1-out), NOT
    # exl3-out — default cleanup=work may have deleted exl3-out, and the
    # in-image pack.is_dir() guard would reject it anyway. Repack pre-fills
    # v1-out, so pass --force; the in-image copy-skip makes re-copying the
    # pack layers already present there a no-op.
    local v1
    local -a avars
    v1="$(v1_out_dir)"
    avars=(--source "$v1" --force
      --status-file "$v1/.pack-status.json"
      --log-dir "$v1/.pack-build-logs")
    [ -n "$WORK" ]     && avars+=(--work "$WORK")
    [ -n "$V1_OUT" ]   && avars+=(--v1-out "$V1_OUT")
    [ -n "$RECIPE" ]   && avars+=(--recipe "$RECIPE")
    [ -n "$CLEANUP" ]  && avars+=(--cleanup "$CLEANUP")
    run_stage assemble "${avars[@]+"${avars[@]}"}"
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
  # Assemble is observable: status file + JSONL event logs live under v1-out.
  local v1
  v1="$(v1_out_dir)"
  vargs+=(--status-file "$(derive_status_file)")
  vargs+=(--log-dir "$v1/.pack-build-logs")
  [ "$FORCE" = true ] && vargs+=(--force)
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
