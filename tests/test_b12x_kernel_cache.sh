#!/bin/bash
set -euo pipefail

# b12x kernel-cache mod test: bash 3.2 + 5.x portable, host CPU, no GPU,
# no network. Drives kcache.sh directly with B12X_KCACHE_ROOTS /
# B12X_KCACHE_STORE / B12X_KCACHE_ID_* overrides for deterministic keys.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

KC="$ROOT/mods/b12x-kernel-cache/kcache.sh"
[ -f "$KC" ] || fail "kcache.sh missing"
bash -n "$KC" || fail "kcache.sh does not parse"

STORE="$WORK/store"
ROOTS_TXT="$WORK/roots.txt"
mkdir -p "$STORE"

export B12X_KCACHE_ROOTS="$ROOTS_TXT"
export B12X_KCACHE_STORE="$STORE"
export B12X_KCACHE_ID_B12X_REV=testrev00000000000000000000000000000000
export B12X_KCACHE_ID_CUTLASS_DSL=4.7.0
export B12X_KCACHE_ID_TORCH=2.9.0
export B12X_KCACHE_ID_VLLM=0.1.dev0
export B12X_KCACHE_ID_VLLM_OBJS=
export B12X_KCACHE_ID_CUDA=sm_121a

write_roots() { # write_roots <compile-dir>
  printf 'compile\t%s\npreparation\t%s\n' "$1" "$1-prep" > "$ROOTS_TXT"
  mkdir -p "$1-prep"
}

echo "=== b12x kernel-cache mod ==="

# --- 1. key stability: same inputs twice -> identical; HOSTNAME irrelevant
write_roots "$WORK/roots/compile"
mkdir -p "$WORK/roots/compile"
echo tuning > "$WORK/roots/compile/k1.o"
K1=$(bash "$KC" key "vllm serve FES/MiMo --tensor-parallel-size 2") \
  || fail "key computation failed"
K2=$(bash "$KC" key "vllm serve FES/MiMo --tensor-parallel-size 2") \
  || fail "key computation failed (2nd)"
[ -n "$K1" ] && [ "$K1" = "$K2" ] || fail "key not stable: '$K1' vs '$K2'"
H1=$(printf '%s' "$K1" | awk -F__ '{print $NF}')
OLD_HOST="$HOSTNAME"
HOSTNAME=otherhost
K3=$(bash "$KC" key "vllm serve FES/MiMo --tensor-parallel-size 2") \
  || fail "key computation failed (hostname)"
HOSTNAME="$OLD_HOST"
[ "$K3" = "$K1" ] || fail "HOSTNAME changed the key"

# --- 2. key sensitivity: flipping each override changes the key
K4=$(B12X_KCACHE_ID_B12X_REV=othrev bash "$KC" key "vllm serve FES/MiMo --tensor-parallel-size 2") \
  || fail "key computation failed (b12x_rev flip)"
[ "$K4" != "$K1" ] || fail "b12x_rev flip did not change key"
K5=$(B12X_KCACHE_ID_CUTLASS_DSL=4.8.0 bash "$KC" key "vllm serve FES/MiMo --tensor-parallel-size 2") \
  || fail "key computation failed (cutlass flip)"
[ "$K5" != "$K1" ] || fail "cutlass flip did not change key"
K6=$(bash "$KC" key "vllm serve FES/MiMo --tensor-parallel-size 1") \
  || fail "key computation failed (tp flip)"
[ "$K6" != "$K1" ] || fail "tp flip did not change key"

# --- 3. key hard-fail: missing override -> exit 3, stderr names field
RC=0
OUT=$(env -u B12X_KCACHE_ID_CUDA bash "$KC" key "vllm serve FES/MiMo" 2>&1) || RC=$?
[ "$RC" = "3" ] || fail "unresolvable arch: expected exit 3, got $RC"
printf '%s' "$OUT" | grep -q "cannot resolve arch" \
  || fail "hard-fail stderr does not name field: $OUT"

# --- 4. roundtrip: save -> wipe -> restore -> byte-identical
KS=$(bash "$KC" key "vllm serve FES/MiMo --tensor-parallel-size 2") \
  || fail "key computation failed (roundtrip)"
bash "$KC" save "$KS" > "$WORK/save.log" 2>&1 || fail "save failed: $(cat "$WORK/save.log")"
grep -q "saved" "$WORK/save.log" || fail "save did not log success: $(cat "$WORK/save.log")"
[ -f "$STORE/$KS.tar.zst" ] || fail "archive not created"
# manifest sanity: schema 1, key matches, per-root hashes present
STAGE="$WORK/manifest-check"; mkdir -p "$STAGE"
zstd -d -q -c "$STORE/$KS.tar.zst" | tar -xf - -C "$STAGE" 2>/dev/null \
  || gtar -xf "$STORE/$KS.tar.zst" -C "$STAGE" 2>/dev/null \
  || fail "archive not extractable"
python3 -c "
import json, sys
m = json.load(open('$STAGE/manifest.json'))
assert m['schema'] == 1, m
assert m['key'] == '$KS', m['key']
assert m['roots'], m
for r in m['roots']:
    assert r['sha256_by_file'], r
    assert r['file_count'] >= 1, r
" || fail "manifest malformed"
cp -R "$WORK/roots/compile" "$WORK/orig-compile"
rm -rf "$WORK/roots/compile"; mkdir -p "$WORK/roots/compile"
RESTORE_LOG="$WORK/restore.log"
bash "$KC" restore "$KS" > "$RESTORE_LOG" 2>&1 || fail "restore failed: $(cat "$RESTORE_LOG")"
grep -q "restored" "$RESTORE_LOG" || fail "restore did not log success: $(cat "$RESTORE_LOG")"
diff -r "$WORK/orig-compile" "$WORK/roots/compile" >/dev/null \
  || fail "restore not byte-identical"

# --- 5. corrupt archive: truncated -> restore exits 0, target untouched
KC2=$(bash "$KC" key "vllm serve FES/Other --tensor-parallel-size 2") \
  || fail "key computation failed (corrupt)"
rm -rf "$WORK/roots/compile"; mkdir -p "$WORK/roots/compile"
echo junk > "$WORK/roots/compile/c1.o"
cp -R "$WORK/roots/compile" "$WORK/pre-corrupt"
bash "$KC" save "$KC2" >/dev/null 2>&1 || fail "save (corrupt case) failed"
# rebuild compile as it would be after restore, so we can prove untouched
rm -rf "$WORK/roots/compile"; mkdir -p "$WORK/roots/compile"
echo junk > "$WORK/roots/compile/c1.o"
python3 - <<PY
import sys
p = "$STORE/$KC2.tar.zst"
data = open(p, "rb").read()
open(p, "wb").write(data[: len(data) // 2])
PY
CR_LOG="$WORK/corrupt.log"
bash "$KC" restore "$KC2" > "$CR_LOG" 2>&1; RC=$?
[ "$RC" = "0" ] || fail "corrupt restore exit $RC (expected 0)"
grep -q "archive corrupt" "$CR_LOG" || fail "corrupt restore did not log: $(cat "$CR_LOG")"
diff -r "$WORK/pre-corrupt" "$WORK/roots/compile" >/dev/null \
  || fail "corrupt restore touched the cache"

# --- 6. empty roots: save logs "no tuning output found", writes nothing
rm -rf "$WORK/roots/compile"; mkdir -p "$WORK/roots/compile"
KCE=$(bash "$KC" key "vllm serve FES/Empty --tensor-parallel-size 2") \
  || fail "key computation failed (empty)"
E_LOG="$WORK/empty.log"
bash "$KC" save "$KCE" > "$E_LOG" 2>&1 || fail "empty save failed"
grep -q "no tuning output found" "$E_LOG" || fail "empty save log: $(cat "$E_LOG")"
[ ! -f "$STORE/$KCE.tar.zst" ] || fail "empty save wrote an archive"

# --- 7. recursion guard: _archives/ inside a root is never archived
echo keep > "$WORK/roots/compile/k2.o"
mkdir -p "$WORK/roots/compile/_archives"
echo inner > "$WORK/roots/compile/_archives/inner.o"
KCR=$(bash "$KC" key "vllm serve FES/Recur --tensor-parallel-size 2") \
  || fail "key computation failed (recursion)"
R_LOG="$WORK/recursion.log"
bash "$KC" save "$KCR" > "$R_LOG" 2>&1 || fail "recursion save failed"
STAGE="$WORK/recursion-check"; mkdir -p "$STAGE"
zstd -d -q -c "$STORE/$KCR.tar.zst" | tar -tf - > "$WORK/recursion.list" 2>/dev/null \
  || gtar -tf "$STORE/$KCR.tar.zst" > "$WORK/recursion.list" 2>/dev/null \
  || fail "recursion archive not extractable"
if grep -q "_archives" "$WORK/recursion.list"; then
  fail "recursion guard: _archives got archived"
fi
grep -q "k2.o" "$WORK/recursion.list" || fail "recursion save lost k2.o"

# --- 8. concurrent restore: two processes, same key -> both succeed, no
# partial tree (mkdir lock + atomic renames)
echo tuning2 > "$WORK/roots/compile/k3.o"
echo tuning3 > "$WORK/roots/compile/k4.o"
KCC=$(bash "$KC" key "vllm serve FES/Conc --tensor-parallel-size 2") \
  || fail "key computation failed (concurrency)"
bash "$KC" save "$KCC" >/dev/null 2>&1 || fail "save (concurrency) failed"
rm -rf "$WORK/roots/compile"; mkdir -p "$WORK/roots/compile"
bash "$KC" restore "$KCC" > "$WORK/conc1.log" 2>&1 &
P1=$!
bash "$KC" restore "$KCC" > "$WORK/conc2.log" 2>&1 &
P2=$!
wait "$P1" || fail "concurrent restore 1 failed"
wait "$P2" || fail "concurrent restore 2 failed"
diff -r "$WORK/roots/compile" "$WORK/orig-compile" >/dev/null \
  || true  # conc key has k3.o/k4.o, orig-compile has k1.o: different trees
grep -qE "restored|already warm|lock busy" "$WORK/conc1.log" \
  || fail "conc restore 1 log: $(cat "$WORK/conc1.log")"
grep -qE "restored|already warm|lock busy" "$WORK/conc2.log" \
  || fail "conc restore 2 log: $(cat "$WORK/conc2.log")"
grep -lq "restored" "$WORK/conc1.log" "$WORK/conc2.log" 2>/dev/null \
  || fail "no restore completed (both skipped on lock): $(cat "$WORK/conc1.log" "$WORK/conc2.log")"
[ -f "$WORK/roots/compile/k3.o" ] || fail "concurrent restore lost k3.o"
[ -f "$WORK/roots/compile/k4.o" ] || fail "concurrent restore lost k4.o"
! grep -q "corrupt" "$WORK/conc1.log" || fail "conc restore 1 corrupt"
! grep -q "corrupt" "$WORK/conc2.log" || fail "conc restore 2 corrupt"

# --- 9. prune: 5 archives, B12X_KCACHE_PRUNE=3 -> 3 newest retained
for i in 1 2 3 4 5; do
  KP=$(bash "$KC" key "vllm serve FES/Prune$i --tensor-parallel-size 2") \
    || fail "key computation failed (prune $i)"
  echo x > "$WORK/roots/compile/k5.o"
  bash "$KC" save "$KP" >/dev/null 2>&1 || fail "save (prune $i) failed"
done
export B12X_KCACHE_PRUNE=3
bash "$KC" prune >/dev/null 2>&1 || fail "prune failed"
N=$(ls "$STORE"/*.tar.zst 2>/dev/null | wc -l | tr -d ' ')
[ "$N" = "3" ] || fail "prune left $N archives, expected 3 (global cap)"
IDX_LINES=$(grep -c . "$STORE/INDEX.tsv" || true)
[ "$IDX_LINES" = "$N" ] || fail "INDEX has $IDX_LINES lines, store has $N"
unset B12X_KCACHE_PRUNE

echo "PASS: b12x-kernel-cache mod"
