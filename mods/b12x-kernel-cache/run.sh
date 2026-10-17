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
#
# Also installs a logging filter that silences b12x's per-shape
# "was not prepared before its first use" WARNING (b12x/preparation/
# session.py:_warn_unprepared_declaration): b12x emits one WARNING per
# distinct kernel shape, which spams serving logs during the warmup sweep.
# The filter is message-matched (drops only that first-use notice) rather
# than a logger-level raise, because b12x.preparation also logs genuine
# tuning-candidate rejections at WARNING (session.py:1460) that must stay
# visible; the package logs no ERROR/CRITICAL diagnostics. Delivered via
# a site-packages .pth hook (b12x_prep_quiet.pth -> import b12x_prep_quiet)
# importing /usr/local/lib/b12x-kcache/b12x_prep_quiet.py, so the filter
# installs in every Python process including vLLM's spawned TP workers.
# Remove the .pth to revert.

PREFIX="[b12x-kernel-cache]"
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_LIB=/usr/local/lib/b12x-kcache
DEST_BIN=/usr/local/bin/b12x-kcache
DEST_SITE="${B12X_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"

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

# First-use notice filter (see header comment). .pth import lines execute
# at interpreter startup (site initialization), so the filter is in place
# before b12x is imported.
cat > "$DEST_LIB/b12x_prep_quiet.py" <<'PYEOF'
"""Silence b12x's benign per-shape unprepared-declaration WARNING.

Targets only the first-use notice emitted by
b12x/preparation/session.py:_warn_unprepared_declaration
("... was not prepared before its first use; preparing its default
configuration. Repeated matching declarations are logged at DEBUG.").
Everything else on the b12x.preparation logger (including the genuine
tuning-candidate rejection WARNING) passes through untouched, and the
companion DEBUG line for the full declaration is unaffected.
"""
import logging

_LOGGER = "b12x.preparation"
_FIRST_USE = "was not prepared before its first use"


class _FirstUseNoticeFilter(logging.Filter):
    def filter(self, record):
        return _FIRST_USE not in record.getMessage()


logging.getLogger(_LOGGER).addFilter(_FirstUseNoticeFilter())
PYEOF
if [ -d "$DEST_SITE" ]; then
  # .pth import lines run at site initialization, but $DEST_LIB is not on
  # sys.path at that point, so the hook line appends it before importing.
  printf "import sys; sys.path.append('%s'); import b12x_prep_quiet\n" \
    "$DEST_LIB" > "$DEST_SITE/b12x_prep_quiet.pth"
  echo "$PREFIX installed first-use notice filter -> $DEST_SITE/b12x_prep_quiet.pth"
else
  echo "$PREFIX WARNING: site-packages dir $DEST_SITE missing; notice filter not installed" >&2
fi

echo "$PREFIX installed b12x-kcache -> $DEST_BIN"
