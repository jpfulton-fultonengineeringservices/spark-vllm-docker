# Distributed verbs: dist-coordinator, dist-worker, dist-run, teardown.
# Sourced by scripts/exl3-pack-drive.sh; defines functions only.
# Target: /bin/bash 3.2 (macOS). No top-level side effects.

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

# Emit <work>/recipe.yaml in-image before workers launch. The distributed
# coordinator passes --recipe to convert_model.prepare(), which requires the
# file to exist; unlike the single-node convert stage it is never emitted
# implicitly, so dist-run must do it explicitly.
_emit_dist_recipe() {
  local src="$1" work="$2" recipe="$3"
  [ -n "$src" ] && [ -n "$work" ] && [ -n "$recipe" ] \
    || { err "dist-run: recipe emission needs --source/--work/--recipe"; exit 2; }
  [ "$DRY_RUN" = true ] && { echo "[dry-run] dist-run: would emit ${recipe} via exl3-pack recipe"; return 0; }
  build_global_args
  docker_run_host "$HOST" "${NAME}-recipe" fg - recipe \
    --source "$src" --work "$work" --out "$recipe" \
    || { err "dist-run: recipe emission failed"; return 1; }
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
  # Guard BEFORE the EXIT trap is armed: a refusal must change nothing on any
  # node, and the trap's teardown writes stop-files on every node.
  preflight_dist_run || return 1
  _emit_dist_recipe "$src" "$work" "$recipe" || return 1
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
