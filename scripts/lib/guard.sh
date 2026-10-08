# Pre-launch guards: shared-WORK checks, stale stop-files, teardown.
# Sourced by scripts/exl3-pack-drive.sh; defines functions only.
# Target: /bin/bash 3.2 (macOS). No top-level side effects.

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

# Pre-launch guard for dist-run. Runs BEFORE any launch and before the EXIT
# trap is armed, so a refusal changes nothing on any node.
#
# A run owns exactly these names: <NAME>-worker-<slug> per node and
# <NAME>-coord on HOST, plus <WORK>/dist/stop-<slug> per node. The guard checks
# by name, not by running state: an EXITED container with a run name still
# makes `docker run --name` fail, so it counts as a conflict too.
#
# Without --force: any existing name is a refusal. Nothing is changed.
# With --force:    tear down exactly those named containers (stop-file first,
#                  then graceful stop, then rm), and clear this run's
#                  stop-files. Nothing outside this run's names is touched.
preflight_dist_run() {
  local IFS=',' slug host cname conflicts=""
  for slug in $NODES; do
    slug="${slug// /}"
    [ -n "$slug" ] || continue
    host="$(node_ssh_host "$slug")"
    cname="$(worker_container_name "$slug")"
    if [ "$DRY_RUN" = true ]; then
      echo "preflight: would check ${cname} on ${host}; clear ${WORK}/dist/stop-${slug} (with --force: tear down ${cname})"
      continue
    fi
    if container_exists "$host" "$cname"; then
      conflicts="${conflicts}${cname} on ${host}; "
    fi
  done
  if [ "$DRY_RUN" != true ] && container_exists "$HOST" "${NAME}-coord"; then
    conflicts="${conflicts}${NAME}-coord on ${HOST}; "
  fi
  if [ "$DRY_RUN" = true ]; then
    return 0
  fi
  if [ -n "$conflicts" ] && [ "$FORCE" != true ]; then
    err "preflight: existing run containers: ${conflicts}Nothing changed. Rerun with --force to tear down exactly these."
    return 1
  fi
  for slug in $NODES; do
    slug="${slug// /}"
    [ -n "$slug" ] || continue
    host="$(node_ssh_host "$slug")"
    cname="$(worker_container_name "$slug")"
    if container_exists "$host" "$cname"; then
      teardown_container "$host" "$cname" "$WORK" "$slug" || return 1
    fi
    clear_stop_file "$host" "$WORK" "$slug" || return 1
  done
  if container_exists "$HOST" "${NAME}-coord"; then
    teardown_container "$HOST" "${NAME}-coord" "" "" || return 1
  fi
}

# True if a container with this exact name exists on host, running or exited.
container_exists() {
  local host="$1" cname="$2" out
  out="$(ssh "$(ssh_target "$host")" \
    "docker inspect -f '{{.Name}}' '${cname}' 2>/dev/null" || true)"
  [ -n "$out" ]
}

# Remove a stale stop-file for this run's node. Only the named file is touched.
clear_stop_file() {
  local host="$1" work="$2" slug="$3"
  ssh "$(ssh_target "$host")" "rm -f '${work}/dist/stop-${slug}'" || {
    err "preflight: could not clear ${work}/dist/stop-${slug} on ${host}."
    return 1
  }
}

# Tear down one named container for --force. A RUNNING worker gets the
# stop-file first, so it exits between shards, then a graceful stop. A RUNNING
# coordinator has no stop-file and is stopped directly. An EXITED container
# cannot read a stop-file, so it is removed directly without the wait.
teardown_container() {
  local host="$1" cname="$2" work="$3" slug="$4"
  local running
  running="$(ssh "$(ssh_target "$host")" \
    "docker inspect -f '{{.State.Running}}' '${cname}' 2>/dev/null" || true)"
  echo "force: tearing down ${cname} on ${host}"
  if [ "$running" = "true" ] && [ -n "$slug" ]; then
    ssh "$(ssh_target "$host")" "touch '${work}/dist/stop-${slug}'" || return 1
    sleep 5
  fi
  if [ "$running" = "true" ]; then
    ssh "$(ssh_target "$host")" "docker stop -t 20 '${cname}' >/dev/null 2>&1; true" || return 1
  fi
  ssh "$(ssh_target "$host")" "docker rm -f '${cname}' >/dev/null 2>&1; true" || return 1
}
