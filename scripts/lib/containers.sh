# Container state queries and docker run construction.
# Sourced by scripts/exl3-pack-drive.sh; defines functions only.
# Target: /bin/bash 3.2 (macOS). No top-level side effects.

# --- container state ----------------------------------------------------------

container_state() {
  ssh "$(ssh_target)" \
    "docker inspect -f '{{.State.Status}}|{{.State.ExitCode}}' '${NAME}' 2>/dev/null" \
    || echo "gone|"
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
