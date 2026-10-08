# Shared helpers: ssh target, usage, validation, path and argv builders.
# Sourced by scripts/exl3-pack-drive.sh; defines functions only.
# Target: /bin/bash 3.2 (macOS). No top-level side effects.

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

# Final v1 output dir, mirroring paths.resolve_destination(<slug>, out_root).

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
