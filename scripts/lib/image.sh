# Build-context sync, image build, and prep orchestration.
# Sourced by scripts/exl3-pack-drive.sh; defines functions only.
# Target: /bin/bash 3.2 (macOS). No top-level side effects.

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

cmd_build() {
  if [ "$DRY_RUN" = true ]; then
    echo "dry-run: would sync and build ${TAG} on ${HOST}"
    return 0
  fi
  do_sync
  do_build
}

cmd_sync() {
  if [ "$DRY_RUN" = true ]; then
    echo "dry-run: would sync build context to ${HOST}"
    return 0
  fi
  do_sync
}
