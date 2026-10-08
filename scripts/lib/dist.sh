# dist-run worker lifecycle: launch, stop-file signalling, stop.
# Sourced by scripts/exl3-pack-drive.sh; defines functions only.
# Target: /bin/bash 3.2 (macOS). No top-level side effects.

# Launch one detached dist-worker container per node in --nodes.
launch_workers() {
  [ -n "$NODES" ] || { err "launch_workers: --nodes required"; return 2; }
  local IFS=','
  local slug host cname
  for slug in $NODES; do
    slug="${slug// /}"
    [ -n "$slug" ] || continue
    local log_dir="${WORK}/dist/logs/${slug}"
    # Log dir is created in-image by logconfig.get_logger (mkdir -p) where the
    # NFS mount is writable; the driver host has no /nas-1 so never mkdir here.
    if [ "$DRY_RUN" = true ]; then
      echo "[dist] DRY-RUN log dir: ${WORK}/dist/logs/${slug} (created in-image)"
    fi
    host="$(node_ssh_host "$slug")"
    cname="$(worker_container_name "$slug")"
    echo "launching worker on ${host} (container ${cname})"
    docker_run_host "$host" "$cname" bg "" dist-worker \
      --inbox "${WORK}/dist/inbox/${slug}" \
      --shared "${WORK}" \
      --device "${DEVICE}" \
      --stop "${WORK}/dist/stop-${slug}" \
      --log-dir "${log_dir}"
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
