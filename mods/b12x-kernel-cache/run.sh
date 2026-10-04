#!/bin/bash
set -euo pipefail

# b12x kernel-cache mod.
#
# Installs the b12x-kcache CLI (mods/b12x-kernel-cache/kcache.sh) into the
# container at /usr/local/bin/b12x-kcache so recipes can wrap their serve
# command with `b12x-kcache exec vllm serve ...`. The wrapper is inert until a
# recipe invokes it: it restores a matching autotune-kernel archive if one
# exists (skipping the multi-minute b12x/CuTe tune), then saves a fresh
# archive after the serve becomes healthy. See mods/b12x-kernel-cache/README.md
# for store layout, key inputs, and knobs.
#
# Must not require any recipe/model knowledge: works on any image.
# Idempotent: re-installing overwrites both files.

PREFIX="[b12x-kernel-cache]"
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_LIB=/usr/local/lib/b12x-kcache
DEST_BIN=/usr/local/bin/b12x-kcache

echo "=== b12x kernel-cache mod ==="

if [ ! -f "$MOD_DIR/kcache.sh" ]; then
  echo "$PREFIX kcache.sh missing from $MOD_DIR" >&2
  exit 1
fi

install -D -m 0755 "$MOD_DIR/kcache.sh" "$DEST_LIB/kcache.sh"
install -D -m 0644 "$MOD_DIR/roots.txt" "$DEST_LIB/roots.txt"
install -d -m 0755 "$DEST_LIB/lib"
for _libsh in "$MOD_DIR/lib/"*.sh; do
  install -m 0644 "$_libsh" "$DEST_LIB/lib/$(basename "$_libsh")"
done
cat > "$DEST_BIN" <<'EOF'
#!/bin/bash
exec /usr/local/lib/b12x-kcache/kcache.sh "$@"
EOF
chmod 0755 "$DEST_BIN"

echo "$PREFIX installed b12x-kcache -> $DEST_BIN"
