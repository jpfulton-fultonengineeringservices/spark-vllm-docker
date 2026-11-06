#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir "$T/bin"; export TEST_ROOT="$T" TMPDIR="$T"
cat > "$T/bin/gh" <<'SH'
#!/bin/bash
for arg in "$@"; do
    if [ "$arg" = --slurp ]; then echo 'unknown flag: --slurp' >&2; exit 1; fi
done
if [ "${GH_PARTIAL:-0}" = 1 ]; then
    printf '%s\n' '[{"metadata":{"container":{"tags":["vllm-node-20261106-99"]}}}]' '{"message":"Package not found.","status":"404"}'
    echo 'HTTP 404' >&2
    exit 1
fi
if [ "${GH_ERROR_BODY_SET:-0}" = 1 ]; then
    printf '%s' "$GH_ERROR_BODY"
    echo 'gh: Package not found. (HTTP 404)' >&2
    exit 1
fi
if [ "${GH_FAIL:-0}" = 1 ]; then
    printf '%s\n' '{"message":"Package not found.","documentation_url":"https://docs.github.com/rest/packages/packages#list-package-versions-for-a-package-owned-by-an-organization","status":"404"}'
    echo 'gh: Package not found. (HTTP 404)' >&2
    exit 1
fi
if [ "${GH_OTHER_FAIL:-0}" = 1 ]; then echo 'unknown flag: --slurp' >&2; exit 1; fi
if [ "${GH_USE_DATA:-0}" = 1 ]; then printf '%s' "$GH_DATA"; else
    printf '%s\n' '[{"metadata":{"container":{"tags":["vllm-node-20261106-9"]}}}]' '[{"metadata":{"container":{"tags":["vllm-node-20261106-10"]}}}]'
fi
SH
cat > "$T/bin/docker" <<'SH'
#!/bin/bash
printf '%s\n' "$*" >> "$TEST_ROOT/docker.log"
case "${FAIL_DOCKER:-}" in '') ;; *) [[ "$*" != *"$FAIL_DOCKER"* ]] || exit 1;; esac
SH
cat > "$T/bin/date" <<'SH'
#!/bin/bash
echo 20261106
SH
cat > "$T/bin/curl" <<'SH'
#!/bin/bash
printf '%s\n' "$*" >> "$TEST_ROOT/curl.log"
printf 'HTTP/1.1 200 OK\r\n\r\n' > "$TEST_ROOT/fse-headers-temp"
while [[ $# -gt 0 ]]; do case "$1" in -D) cp "$TEST_ROOT/fse-headers-temp" "$2"; shift;; -o) printf '%s' '{"name":"spark-vllm","tags":["vllm-node-20261106-12"]}' > "$2"; shift;; esac; shift; done
printf 200
SH
chmod +x "$T/bin/"*; export PATH="$T/bin:$PATH"
script="$ROOT/push-to-fse-ghcr.sh"
fail() { echo "FAIL: $*" >&2; exit 1; }
"$script" list vllm-node > "$T/list"
grep -q 'vllm-node-20261106-10' "$T/list" || fail list
"$script" pull vllm-node > /dev/null
grep -q 'pull .*vllm-node-20261106-10' "$T/docker.log" || fail latest
: > "$T/docker.log"
"$script" push vllm-node --cluster > "$T/push"
grep -q 'spark-vllm:vllm-node-20261106-13' "$T/push" || fail serial
grep -q 'push 10.100.170.3:5000/vllm-node:latest' "$T/docker.log" || fail alias
grep -q 'https://10.100.170.3:5000/v2/spark-vllm/tags/list' "$T/curl.log" || fail default-address
: > "$T/docker.log"
"$script" push vllm-node --cluster --no-cluster-latest > /dev/null
! grep -q 'push 10.100.170.3:5000/vllm-node:latest' "$T/docker.log" || fail no-alias
: > "$T/docker.log"; : > "$T/curl.log"
FSE_CLUSTER_REGISTRY=registry.example:5443 "$script" push vllm-node --cluster > /dev/null
grep -q 'push registry.example:5443/spark-vllm:vllm-node-20261106-13' "$T/docker.log" || fail override-version
grep -q 'push registry.example:5443/vllm-node:latest' "$T/docker.log" || fail override-alias
grep -q 'https://registry.example:5443/v2/spark-vllm/tags/list' "$T/curl.log" || fail override-catalog
: > "$T/docker.log"; : > "$T/curl.log"
"$script" push vllm-node --cluster --insecure-cluster > /dev/null
grep -q -- '-k' "$T/curl.log" || fail insecure-catalog
for invalid in 'host' 'bad/host:5000' 'host:0' 'host:65536'; do
    : > "$T/docker.log"
    if FSE_CLUSTER_REGISTRY="$invalid" "$script" push vllm-node --cluster >/dev/null 2>&1; then fail invalid-registry; fi
    [ ! -s "$T/docker.log" ] || fail invalid-registry-docker
done
if "$script" push vllm-node:dev >/dev/null 2>&1; then fail nonlatest; fi
if GH_FAIL=1 "$script" push vllm-node >/dev/null 2>&1; then fail 404; fi
GH_FAIL=1 "$script" push vllm-node --new-package >/dev/null 2>&1 || fail new-package
: > "$T/docker.log"
if GH_PARTIAL=1 "$script" push vllm-node --new-package > "$T/failure" 2>&1; then fail partial-ghcr; fi
! grep -q '^push ' "$T/docker.log" || fail partial-ghcr-pushed
for body in '[{"status":"404","message":"not found"}]' '{"message":"not found","status":"403"}' '{"status":"404"}' '{' '[] {"message":"not found","status":"404"}'; do
    : > "$T/docker.log"
    if GH_ERROR_BODY_SET=1 GH_ERROR_BODY="$body" "$script" push vllm-node --new-package > "$T/failure" 2>&1; then fail invalid-error-body; fi
    ! grep -q '^push ' "$T/docker.log" || fail invalid-error-body-pushed
done
GH_ERROR_BODY_SET=1 GH_ERROR_BODY='' "$script" push vllm-node --new-package >/dev/null 2>&1 || fail empty-404-body
GH_ERROR_BODY_SET=1 GH_ERROR_BODY='{"message":"not found","status":404}' "$script" push vllm-node --new-package >/dev/null 2>&1 || fail numeric-404-body
GH_USE_DATA=1 GH_DATA='[]' "$script" list vllm-node > "$T/empty" || fail empty-pages
[ ! -s "$T/empty" ] || fail empty-pages-output
for bad in '' '{"message":"not a page"}' '[] {"bad":true}' '[' '[{}]' '[{"metadata":{"container":{"tags":"invalid"}}}]' '[{"metadata":{"container":{"tags":[42]}}}]'; do
    if GH_USE_DATA=1 GH_DATA="$bad" "$script" push vllm-node > "$T/failure" 2>&1; then fail malformed-page; fi
done
if GH_OTHER_FAIL=1 "$script" list vllm-node > "$T/failure" 2>&1; then fail other-error; fi
grep -q 'verify GitHub CLI authentication' "$T/failure" || fail misleading-non-404
if grep -q '404 may mean' "$T/failure"; then fail misleading-404; fi
if FAIL_DOCKER='push 10.100.170.3:5000/spark-vllm' "$script" push vllm-node --cluster > "$T/failure" 2>&1; then fail partial; fi
grep -q 'already published' "$T/failure" || fail partial-report
echo 'PASS: mocked GHCR push/list/pull and cluster publishing'
