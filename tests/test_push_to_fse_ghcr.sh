#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir "$T/bin"; export TEST_ROOT="$T" TMPDIR="$T"
cat > "$T/bin/gh" <<'SH'
#!/bin/bash
if [ "${GH_FAIL:-0}" = 1 ]; then echo 'HTTP 404' >&2; exit 1; fi
if [ -n "${GH_DATA:-}" ]; then printf '%s\n' "$GH_DATA"; else printf '%s\n' '[[{"metadata":{"container":{"tags":["vllm-node-20261106-9","vllm-node-20261106-10"]}}}]]'; fi
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
grep -q 'push gx10-node1:5000/vllm-node:latest' "$T/docker.log" || fail alias
: > "$T/docker.log"
"$script" push vllm-node --cluster --no-cluster-latest > /dev/null
! grep -q 'push gx10-node1:5000/vllm-node:latest' "$T/docker.log" || fail no-alias
if "$script" push vllm-node:dev >/dev/null 2>&1; then fail nonlatest; fi
if GH_FAIL=1 "$script" push vllm-node >/dev/null 2>&1; then fail 404; fi
GH_FAIL=1 "$script" push vllm-node --new-package >/dev/null 2>&1 || fail new-package
if FAIL_DOCKER='push gx10-node1:5000/spark-vllm' "$script" push vllm-node --cluster > "$T/failure" 2>&1; then fail partial; fi
grep -q 'already published' "$T/failure" || fail partial-report
echo 'PASS: mocked GHCR push/list/pull and cluster publishing'
