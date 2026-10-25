#!/bin/bash
set -euo pipefail

# Image-pin regression test: bash 3.2 + 5.x portable, host CPU only.
# Drives launch-cluster.sh's --image-id gate without Docker by sourcing the
# guard with a stubbed `docker` on PATH.
#
# Covers: absent local image -> clear error; mismatched ID -> refuses;
# matching ID -> passes; no pin -> no-op; and the content-vs-tag property that
# makes the pin meaningful (the same tag may hold other content).

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCH="$ROOT/launch-cluster.sh"
[ -f "$LAUNCH" ] || { echo "FAIL: launch-cluster.sh missing" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

bash -n "$LAUNCH" || fail "launch-cluster.sh does not parse"
grep -qF -- '--image-id' "$LAUNCH" || fail "launcher does not accept --image-id"
grep -qF 'verify_pinned_image_id' "$LAUNCH" || fail "pin verification missing"

# The coder390 recipes are pinned to the derived-image digest the repo build
# script produces (docker/build-coder390-derived-image.sh; commit 3377be48d1
# repinned them from the a1fc4f18 nightly). Keep this in sync with the pins:
# the whole point of (f) is that every pinned recipe forwards its pin.
PIN="sha256:e9f6e32941ec5680a58edfbfefaad6997f0527e22344b5b505cf8267de4625f2"
OTHER="sha256:4aef4572b26eb283a1757aeedddc24f5b2bce3ff2c16d961f7334b0f480c3150"

# Extract just the guard so it can run standalone (no cluster/bootstrap).
sed -n '/^verify_pinned_image_id()/,/^}/p' "$LAUNCH" > "$WORK/guard.sh"
[ -s "$WORK/guard.sh" ] || fail "could not extract verify_pinned_image_id"

# Stub docker: reports $STUB_ID for any inspect, or fails when $STUB_MISSING=1.
mkdir -p "$WORK/bin"
cat > "$WORK/bin/docker" <<'EOF'
#!/bin/bash
if [ "${STUB_MISSING:-0}" = "1" ]; then exit 1; fi
case "$1 $2" in
  "image inspect") printf '%s\n' "${STUB_ID}" ;;
  *) exit 0 ;;
esac
EOF
chmod +x "$WORK/bin/docker"

run_guard() { # run_guard <expected-id> ; uses STUB_ID/STUB_MISSING from env
  PATH="$WORK/bin:$PATH" bash -c '
    source "$1"
    IMAGE_NAME=vllm-node-b12x
    EXPECTED_IMAGE_ID="$2"
    verify_pinned_image_id
  ' _ "$WORK/guard.sh" "$1"
}

echo "=== image pin ==="

# (a) absent local image -> clear error, fail closed
if STUB_MISSING=1 STUB_ID= run_guard "$PIN" >"$WORK/a.log" 2>&1; then
  fail "(a) missing image did not fail"
fi
grep -qi 'not present\|cannot be checked' "$WORK/a.log" \
  || fail "(a) unclear missing-image error: $(cat "$WORK/a.log")"

# (b) mismatched content -> refuses, naming both IDs
if STUB_ID="$OTHER" run_guard "$PIN" >"$WORK/b.log" 2>&1; then
  fail "(b) mismatched image did not fail"
fi
grep -qF "$PIN" "$WORK/b.log" || fail "(b) error omits the pinned id"
grep -qF "$OTHER" "$WORK/b.log" || fail "(b) error omits the actual id"

# (c) matching content -> passes
STUB_ID="$PIN" run_guard "$PIN" >"$WORK/c.log" 2>&1 \
  || fail "(c) matching image was rejected: $(cat "$WORK/c.log")"
grep -q 'Verified pinned image_id' "$WORK/c.log" || fail "(c) no verification line"

# (d) no pin -> no-op, even with a mismatched image present
STUB_ID="$OTHER" run_guard "" >"$WORK/d.log" 2>&1 \
  || fail "(d) unpinned run failed: $(cat "$WORK/d.log")"
! grep -q 'Verified pinned' "$WORK/d.log" || fail "(d) unpinned run claimed a pin"

# (e) content-not-tag: the guard keys on the image ID, so the same ID passes
# regardless of which tag name carries it, and another ID fails.
STUB_ID="$PIN" run_guard "$PIN" >/dev/null 2>&1 || fail "(e) pinned id rejected"
if STUB_ID="$OTHER" run_guard "$PIN" >/dev/null 2>&1; then
  fail "(e) divergent tag content accepted"
fi

# (f) recipes that pin must actually reach the launcher with --image-id
for r in qwen3.8-27b-coder390-2x qwen3.8-27b-coder390-4x \
         qwen3.8-27b-coder390-2x-store-l2 qwen3.8-27b-coder390-4x-store-l2; do
  out="$(cd "$ROOT" && python3 run-recipe.py "$r" --dry-run -n 10.0.0.1,10.0.0.2 2>&1)" \
    || fail "(f) $r dry-run failed"
  case "$out" in
    *"--image-id $PIN"*) ;;
    *) fail "(f) $r does not forward --image-id" ;;
  esac
done

# (g) an explicit -t override means "run a different image" and drops the pin
out="$(cd "$ROOT" && python3 run-recipe.py qwen3.8-27b-coder390-2x --dry-run -n 10.0.0.1,10.0.0.2 -t vllm-node-b12x 2>&1)"
case "$out" in
  *"--image-id"*) fail "(g) -t override still forwarded a pin" ;;
esac

echo "PASS: image pin"
