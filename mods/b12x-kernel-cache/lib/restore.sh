# ---------------------------------------------------------------- restore

# verify_manifest <stage-dir>: recompute per-root file hashes + tree hash
# against manifest.json. Prints nothing; exit 0 valid, 1 mismatch.
verify_manifest() {
  local stage="$1"
  python3 - "$stage" <<'PY'
import hashlib, json, os, sys

stage = sys.argv[1]
m = json.load(open(os.path.join(stage, "manifest.json")))
if m.get("schema") != 1:
    sys.exit(1)
for r in m.get("roots", []):
    name = r["name"]
    root_dir = os.path.join(stage, name)
    by_file = r["sha256_by_file"]
    relpaths = sorted(by_file)
    if r.get("file_count") != len(relpaths):
        sys.exit(1)
    actual = []
    for rel in relpaths:
        p = os.path.join(root_dir, rel)
        if not os.path.isfile(p):
            sys.exit(1)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != by_file[rel]:
            sys.exit(1)
        actual.append("%s:%s" % (rel, h))
    # tree_sha256: sha over "<rel>:<sha>" lines, each newline-terminated
    # (save-side builds "rel:sha\n" per line).
    tree = hashlib.sha256(
        "".join(line + "\n" for line in actual).encode()
    ).hexdigest()
    if tree != r["tree_sha256"]:
        sys.exit(1)
sys.exit(0)
PY
}

# extract_archive <arch> <stage> <manifest-compression>
extract_archive() {
  local arch="$1" stage="$2" comp="$3"
  case "$comp" in
    zstd)  [ -f "$arch" ] && zstd -d -q -c "$arch" 2>/dev/null | tar -xf - -C "$stage" ;;
    gzip)  tar -xzf "$arch" -C "$stage" 2>/dev/null ;;
    *)     return 1 ;;
  esac
}

cmd_restore() {
  local key="${1:-}"
  if [ -z "$key" ]; then
    key=$(key)
  fi
  local res store shared arch
  res=$(resolve_store read "$key")
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "no archive for ${key: -16}; cold path"
    return 0
  fi
  store=$(printf '%s' "$res" | cut -f1)
  shared=$(printf '%s' "$res" | cut -f2)
  arch="$store/$key.tar.zst"
  log "store=$store shared=$shared"
  # Stage under the store when it is writable; peers (read-only) fall back
  # to a local temp dir.
  local stage="$store/.restore.$key.$$"
  rm -rf "$stage"
  if ! mkdir -p "$stage" 2>/dev/null; then
    stage="${TMPDIR:-/tmp}/kcache-restore.$key.$$"
    rm -rf "$stage"; mkdir -p "$stage"
  fi
  local comp
  local magic
  magic=$(dd if="$arch" bs=4 count=1 2>/dev/null | od -An -tx1 | tr -d ' \n')
  case "$magic" in
    28b52ffd) comp="zstd" ;;
    1f8b*)    comp="gzip" ;;
    *)        comp="unknown" ;;
  esac
  if [ "$comp" = "unknown" ]; then
    log "archive corrupt for ${key: -16}; leaving cache untouched"
    rm -rf "$stage"
    return 0
  fi
  if ! extract_archive "$arch" "$stage" "$comp" \
     || [ ! -f "$stage/manifest.json" ]; then
    log "archive corrupt for ${key: -16}; leaving cache untouched"
    rm -rf "$stage"
    return 0
  fi
  if ! verify_manifest "$stage"; then
    log "archive corrupt for ${key: -16}; leaving cache untouched"
    rm -rf "$stage"
    return 0
  fi

  # mkdir lock: atomic via mkdir; stale locks removed after 10 min. The lock
  # lives in the (writable) stage dir, NOT the store: root-squashed peers
  # cannot mkdir in the read-only shared store.
  local lock="$stage/lock"
  mkdir -p "$(dirname "$lock")"
  if ! mkdir "$lock" 2>/dev/null; then
    # stale?
    local lock_ts now_ts
    lock_ts=$(stat -c %Y "$lock" 2>/dev/null || stat -f %m "$lock" 2>/dev/null || echo 0)
    now_ts=$(date +%s)
    if [ "$((now_ts - lock_ts))" -gt "$RESTORE_LOCK_STALE_SECS" ]; then
      rm -rf "$lock"
      if ! mkdir "$lock" 2>/dev/null; then
        log "restore: lock busy; skipping"
        rm -rf "$stage"
        return 0
      fi
    else
      log "restore: lock busy; skipping"
      rm -rf "$stage"
      return 0
    fi
  fi

  # Per-root install: skip if target exists non-empty; else mkdir parent and
  # mv the staged root into place (atomic on same fs).
  local installed=""
  python3 - "$stage" <<'PY' > "$stage/.roots.$$"
import json, sys
m = json.load(open(sys.argv[1] + "/manifest.json"))
for r in m.get("roots", []):
    print(r["name"] + "\t" + r["path"])
PY
  while IFS="$(printf '\t')" read -r name path; do
    [ -n "$name" ] || continue
    if [ -d "$path" ] && [ -n "$(ls -A "$path" 2>/dev/null)" ]; then
      continue  # already warm
    fi
    if [ -d "$path" ]; then
      # dir exists but is empty (checked above); remove so mv renames, not
      # nests (BSD mv moves INTO an existing dir).
      rmdir "$path" 2>/dev/null || rm -rf "$path"
    fi
    if [ ! -d "$stage/$name" ]; then
      continue
    fi
    mkdir -p "$(dirname "$path")"
    if mv "$stage/$name" "$path" 2>/dev/null; then
      installed="${installed:+$installed,}$name"
    elif cp -R "$stage/$name" "$path" 2>/dev/null; then
      installed="${installed:+$installed,}$name"
    fi
  done < "$stage/.roots.$$"
  rm -f "$stage/.roots.$$"

  rm -rf "$lock" "$stage"
  if [ -n "$installed" ]; then
    log "restored ${key: -16} from $store (roots=$installed)"
  else
    log "restored ${key: -16} from $store (nothing to install; already warm)"
  fi
  return 0
}

