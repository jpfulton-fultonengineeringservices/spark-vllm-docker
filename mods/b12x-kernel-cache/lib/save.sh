# ---------------------------------------------------------------- save

# tar_list <stage-dir>: files under a staged root, excluding locks/tmp.
# Called on the LIVE root before staging. Echoes relpaths sorted.
root_files() { # <root-dir>
  ( cd "$1" 2>/dev/null && find . -type f \
      ! -name '*.lock' ! -name '.tmp.*' \
      ! -path './_archives/*' \
      | LC_ALL=C sort | sed 's|^\./||' )
}

tree_files() { # <dir>: relpaths of all regular files, sorted
  ( cd "$1" 2>/dev/null && find . -type f | LC_ALL=C sort | sed 's|^\./||' )
}

# compress <manifest> | decompress helper: pick zstd when available.
have_zstd() { command -v zstd >/dev/null 2>&1; }

cmd_save() {
  local key="${1:-}"
  if [ -z "$key" ]; then
    key=$(key)
  fi
  local res store shared
  if ! res=$(resolve_store write); then
    log "save: no writable store; skipping"
    return 0
  fi
  store=$(printf '%s' "$res" | cut -f1)
  shared=$(printf '%s' "$res" | cut -f2)
  log "store=$store shared=$shared"

  if [ -f "$store/$key.tar.zst" ]; then
    log "archive already present for ${key: -16}; skipping save"
    return 0
  fi

  local stage="$store/.stage.$key.$$"
  rm -rf "$stage"; mkdir -p "$stage"

  local name path total=0 line
  local tmp_names="$stage/.names.$$"
  : > "$tmp_names"
  while IFS="$(printf '\t')" read -r name path; do
    [ -n "$name" ] || continue
    [ -d "$path" ] || continue
    echo "$name	$path" >> "$tmp_names"
  done < <(roots)
  if [ ! -s "$tmp_names" ]; then
    log "no tuning output found; skipping save"
    rm -rf "$stage"
    return 0
  fi

  # Build per-root payload + manifest.
  local manifest="$stage/manifest.json"
  {
    echo '{'
    echo '  "schema": 1,'
    printf '  "created_ts": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "created_node": "%s",\n' "$(hostname 2>/dev/null || echo unknown)"
    printf '  "key": "%s",\n' "$key"
    printf '  "key_display": "%s",\n' "${key: -16}"
    if have_zstd; then
      echo '  "compression": "zstd",'
    else
      echo '  "compression": "gzip",'
    fi
    echo '  "roots": ['
  } > "$manifest"

  local first="1"
  while IFS="$(printf '\t')" read -r name path; do
    local files file_count tree
    files=$(root_files "$path")
    file_count=$(printf '%s\n' "$files" | grep -c . || true)
    [ "$file_count" -gt 0 ] 2>/dev/null || file_count=0
    total=$((total + file_count))
    if [ "$file_count" -eq 0 ]; then
      # still record the (empty) root so restore can mkdir it
      continue
    fi
    mkdir -p "$stage/$name"
    local f rel
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      mkdir -p "$stage/$name/$(dirname "$f")"
      cp -p "$path/$f" "$stage/$name/$f"
    done <<EOF
$files
EOF
    # tree_sha256: sha over "<rel>:<sha>" lines
    tree=""
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      tree="${tree}${f}:$(sha "$stage/$name/$f")
"
    done <<EOF
$files
EOF
    local tree_sha
    tree_sha=$(printf '%s' "$tree" | stdin_sha)
    [ "$first" = "1" ] || echo ',' >> "$manifest"
    first="0"
    {
      printf '    {"name": "%s", "path": "%s", "file_count": %d, "sha256_by_file": {' \
        "$name" "$path" "$file_count"
      local sep=" "
      while IFS= read -r f; do
        [ -n "$f" ] || continue
        printf '%s"%s": "%s"' "$sep" "$f" "$(sha "$stage/$name/$f")"
        sep=", "
      done <<EOF
$files
EOF
      printf '}, "tree_sha256": "%s"}' "$tree_sha"
    } >> "$manifest"
  done < "$tmp_names"
  rm -f "$tmp_names"

  if [ "$total" -eq 0 ]; then
    log "no tuning output found; skipping save"
    rm -rf "$stage"
    return 0
  fi

  echo '  ]' >> "$manifest"
  echo '}' >> "$manifest"

  # Compress: tar the stage, zstd -3 or gzip fallback.
  local tmp_arch="$store/.tmp.$key.$$"
  local comp
  if have_zstd && tar -cf - -C "$stage" . 2>/dev/null | zstd -3 -q > "$tmp_arch" 2>/dev/null; then
    comp="zstd"
  else
    rm -f "$tmp_arch"
    if tar -czf "$tmp_arch" -C "$stage" . 2>/dev/null; then
      comp="gzip"
    else
      log "save: compression failed; skipping"
      rm -rf "$stage" "$tmp_arch"
      return 0
    fi
  fi
  # Keep manifest consistent with what was actually written.
  if [ "$comp" = "gzip" ]; then
    python3 - "$manifest" <<'PY' 2>/dev/null || true
import json, sys
p = sys.argv[1]
m = json.load(open(p))
m["compression"] = "gzip"
json.dump(m, open(p, "w"), indent=2)
PY
  fi

  local bytes asha
  bytes=$(wc -c < "$tmp_arch" | tr -d ' ')
  asha=$(sha "$tmp_arch")
  mv "$tmp_arch" "$store/$key.tar.zst"

  # INDEX.tsv line.
  printf '%s\t%s\t%s\t%s\t%s\n' "$key" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(hostname 2>/dev/null || echo unknown)" "$bytes" "$asha" \
    >> "$store/INDEX.tsv"

  rm -rf "$stage"
  log "saved ${key: -16} ($total files, $bytes bytes) -> $store/$key.tar.zst"
}

