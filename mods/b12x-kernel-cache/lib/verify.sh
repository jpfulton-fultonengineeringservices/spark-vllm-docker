# ---------------------------------------------------------------- verify

cmd_verify() {
  local key="${1:-}"
  if [ -z "$key" ]; then
    key=$(key)
  fi
  local res store arch comp magic stage
  if ! res=$(resolve_store read "$key"); then
    log "verify: no archive for ${key: -16}"
    return 1
  fi
  store=$(printf '%s' "$res" | cut -f1)
  arch="$store/$key.tar.zst"
  magic=$(dd if="$arch" bs=4 count=1 2>/dev/null | od -An -tx1 | tr -d ' \n')
  case "$magic" in
    28b52ffd) comp="zstd" ;;
    1f8b*)    comp="gzip" ;;
    *)        log "verify: archive corrupt for ${key: -16}"; return 1 ;;
  esac
  stage=$(mktemp -d "${TMPDIR:-/tmp}/kcache-verify.$$") || return 1
  if ! extract_archive "$arch" "$stage" "$comp" \
     || [ ! -f "$stage/manifest.json" ] || ! verify_manifest "$stage"; then
    log "verify: archive corrupt for ${key: -16}"
    rm -rf "$stage"
    return 1
  fi
  rm -rf "$stage"
  log "verify: archive ok for ${key: -16} ($arch)"
  return 0
}

# ---------------------------------------------------------------- prune

cmd_prune() {
  local cap="${B12X_KCACHE_PRUNE:-3}"
  local res store
  if ! res=$(resolve_store write); then
    log "prune: no writable store"
    return 0
  fi
  store=$(printf '%s' "$res" | cut -f1)
  [ -f "$store/INDEX.tsv" ] || return 0
  python3 - "$store" "$cap" <<'PY'
import os, sys
store, cap = sys.argv[1], int(sys.argv[2])
idx = os.path.join(store, "INDEX.tsv")
rows = []
with open(idx) as f:
    for line in f:
        line = line.rstrip("\n")
        if line:
            rows.append(line.split("\t"))
# sort by created_ts (col 2), keep newest `cap`
rows.sort(key=lambda r: r[1])
drop = rows[:-cap] if len(rows) > cap else []
keep = [r[0] for r in rows[-cap:]] if cap > 0 else []
for r in drop:
    arch = os.path.join(store, r[0] + ".tar.zst")
    if os.path.exists(arch):
        os.remove(arch)
with open(idx, "w") as f:
    for r in rows[-cap:] if cap > 0 else []:
        f.write("\t".join(r) + "\n")
print("\n".join(r[0] for r in drop))
PY
}

