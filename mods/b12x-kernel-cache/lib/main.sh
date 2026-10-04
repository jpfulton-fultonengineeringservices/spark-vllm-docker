# ---------------------------------------------------------------- help

cmd_help() {
  cat <<EOF
Usage: $PROGNAME <command> [args]

Commands:
  exec <command...>   restore matching archive, run <command>, save after ready
  key                 print the cache key for the current environment+argv
  store               print "<store><TAB>shared" for the resolved write store
  save [key]          snapshot the b12x cache roots into the store
  restore [key]       restore the matching archive into the cache roots
  verify [key]        validate a stored archive without extracting
  prune               enforce \$B12X_KCACHE_PRUNE (default 3) on the store
  help                this text

Env: B12X_KCACHE_STORE, B12X_KCACHE_PRUNE, B12X_KCACHE_SAVE,
     B12X_KCACHE_FORCE_AUTOTUNE_OFF, B12X_KCACHE_ROOTS,
     B12X_KCACHE_ID_* (key-field overrides, tests)
EOF
}

main() {
  local cmd="${1:-help}"
  [ $# -gt 0 ] && shift
  ARGV_STR="${ARGV_STR:-$*}"
  case "$cmd" in
    key)     cmd_key ;;
    exec)    cmd_exec "$@" ;;
    store)   cmd_store ;;
    save)    cmd_save "$@" ;;
    restore) cmd_restore "$@" ;;
    verify)  cmd_verify "$@" ;;
    prune)   cmd_prune ;;
    help|-h|--help) cmd_help ;;
    *)
      cmd_help >&2
      exit 2
      ;;
  esac
}

main "$@"
