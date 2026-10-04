# ---------------------------------------------------------------- helpers

sha() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

stdin_sha() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
  else
    shasum -a 256 | awk '{print $1}'
  fi
}

log() { echo "$PREFIX $*"; }

die() { # die <field>
  log "key: cannot resolve $1"
  exit 3
}

_require() { # _require <field> <value-or-empty>
  if [ -z "${2:-}" ]; then
    die "$1"
  fi
  printf '%s' "$2"
}

pyver() { # pyver <dist-name>
  python3 - "$1" <<'PY' 2>/dev/null
import importlib.metadata as m
import sys
print(m.version(sys.argv[1]))
PY
}

init_self_dir() {
  local self="${BASH_SOURCE[0]}"
  SELF_DIR="${self%/*}"
  if [ "$SELF_DIR" = "$self" ]; then
    SELF_DIR="."
  fi
}

# roots(): echo "<name><TAB><abs path>" lines from $B12X_KCACHE_ROOTS if set,
# else the roots.txt sitting next to kcache.sh.
roots() {
  if [ -n "${B12X_KCACHE_ROOTS:-}" ]; then
    cat "$B12X_KCACHE_ROOTS"
    return
  fi
  [ -n "$SELF_DIR" ] || init_self_dir
  if [ -f "$SELF_DIR/roots.txt" ]; then
    cat "$SELF_DIR/roots.txt"
  fi
}

