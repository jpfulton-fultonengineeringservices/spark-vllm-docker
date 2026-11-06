#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir "$T/bin"; export TEST_ROOT="$T" TMPDIR="$T"
cat > "$T/bin/gh" <<'SH'
#!/bin/bash
printf '%s\n' "$*" >> "$TEST_ROOT/gh.log"
if [ "${1:-}" = api ] && [ "${3:-}" = github.com ] && [ "${4:-}" = user ]; then
    [ "${USER_LOOKUP_FAIL:-0}" != 1 ] || exit 1
    printf '%s\n' "${GH_USERNAME:-example-user}"
    exit 0
fi
if [ "${1:-}" = auth ] && [ "${2:-}" = token ]; then
    [ "${TOKEN_FAIL:-0}" != 1 ] || exit 1
    printf '%s\n' 'mock-gh-token'
    exit 0
fi
if [ "${1:-}" = auth ] && [ "${2:-}" = status ]; then
    [ "${AUTH_FAIL:-0}" != 1 ] || exit 1
    if [ "${OLD_GH:-0}" = 1 ]; then
        for arg in "$@"; do [ "$arg" != --active ] || { echo 'unknown flag: --active' >&2; exit 1; }; done
    fi
    if [ "${MULTI_ACCOUNTS:-0}" = 1 ]; then
        echo "  - Token scopes: 'repo', 'write:packages'"
        echo "  - Token scopes: 'repo'"
        exit 0
    fi
    if [ "${NO_SCOPES:-0}" = 1 ]; then echo 'Logged in to github.com'; exit 0; fi
    if [ "${SCOPE_NONE:-0}" = 1 ]; then echo '  - Token scopes: none'; exit 0; fi
    if [ "${MISSING_SCOPE:-0}" = 1 ] && [ ! -f "$TEST_ROOT/refreshed" ]; then
        echo "  - Token scopes: 'repo', 'read:org'"
    else
        echo "  - Token scopes: 'repo', 'read:org', 'write:packages'"
    fi
    exit 0
fi
if [ "${1:-}" = auth ] && [ "${2:-}" = refresh ]; then
    [ "${REFRESH_FAIL:-0}" != 1 ] || exit 1
    if [ "${REFRESH_NO_SCOPE:-0}" != 1 ]; then touch "$TEST_ROOT/refreshed"; fi
    exit 0
fi
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
if [ "${1:-}" = login ]; then
    IFS= read -r password || exit 1
    [ "$password" = 'mock-gh-token' ] || exit 1
fi
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
! grep -q '^login ' "$T/docker.log" || fail pull-login
: > "$T/docker.log"
"$script" push vllm-node --cluster > "$T/push" 2>&1
grep -q 'spark-vllm:vllm-node-20261106-13' "$T/push" || fail serial
grep -q '^login ghcr.io --username example-user --password-stdin$' "$T/docker.log" || fail docker-login
grep -q '^auth token --hostname github.com$' "$T/gh.log" || fail token-read
! grep -q 'mock-gh-token' "$T/docker.log" "$T/gh.log" "$T/push" || fail token-exposed
login_line=$(grep -n '^login ghcr.io' "$T/docker.log" | head -1 | cut -d: -f1)
push_line=$(grep -n '^push ghcr.io' "$T/docker.log" | head -1 | cut -d: -f1)
[ "$login_line" -lt "$push_line" ] || fail login-order
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
for failure in USER_LOOKUP_FAIL TOKEN_FAIL; do
    : > "$T/docker.log"
    if env "$failure=1" "$script" push vllm-node > "$T/failure" 2>&1; then fail "$failure-accepted"; fi
    ! grep -q '^push ' "$T/docker.log" || fail "$failure-pushed"
    ! grep -q 'mock-gh-token' "$T/failure" "$T/docker.log" || fail "$failure-token-exposed"
done
: > "$T/docker.log"
if GH_USERNAME='invalid/user' "$script" push vllm-node > "$T/failure" 2>&1; then fail bad-user; fi
! grep -q '^login ' "$T/docker.log" || fail bad-user-login
: > "$T/docker.log"
if FAIL_DOCKER='login ghcr.io' "$script" push vllm-node > "$T/failure" 2>&1; then fail docker-login-failure; fi
! grep -q '^push ' "$T/docker.log" || fail login-failure-pushed
: > "$T/gh.log"; : > "$T/docker.log"
if MISSING_SCOPE=1 "$script" push vllm-node > "$T/failure" 2>&1; then fail missing-scope-noninteractive; fi
grep -q 'gh auth refresh --hostname github.com --scopes write:packages' "$T/failure" || fail missing-scope-guidance
! grep -q '^push ' "$T/docker.log" || fail missing-scope-pushed
! grep -q '^auth refresh' "$T/gh.log" || fail noninteractive-refresh
if MISSING_SCOPE=1 GH_TOKEN=env-token "$script" push vllm-node > "$T/failure" 2>&1; then fail env-token-missing-scope; fi
grep -q 'cannot refresh environment tokens' "$T/failure" || fail env-token-guidance
if AUTH_FAIL=1 "$script" push vllm-node > "$T/failure" 2>&1; then fail auth-status; fi
NO_SCOPES=1 "$script" push vllm-node > "$T/noscope" 2>&1 || fail scope-unknown
grep -q 'did not report token scopes' "$T/noscope" || fail scope-unknown-warning
SCOPE_NONE=1 "$script" push vllm-node > "$T/noscope" 2>&1 || fail scope-none
grep -q 'token scopes are unavailable' "$T/noscope" || fail scope-none-warning
OLD_GH=1 "$script" push vllm-node > "$T/old-gh" 2>&1 || fail old-gh-auth
OLD_GH=1 MULTI_ACCOUNTS=1 "$script" push vllm-node > "$T/multi" 2>&1 || fail old-gh-multi
grep -q 'cannot identify active token scopes' "$T/multi" || fail old-gh-multi-warning
if command -v expect >/dev/null 2>&1; then
    cat > "$T/run-in-pty.exp" <<'EXP'
set timeout 20
set mode [lindex $argv 0]
set script [lindex $argv 1]
set ::env(MISSING_SCOPE) 1
set ::env(REFRESH_FAIL) [expr {$mode eq "refresh-fail"}]
set ::env(REFRESH_NO_SCOPE) [expr {$mode eq "refresh-no-scope"}]
spawn -noecho $script push vllm-node
expect eof
set result [wait]
exit [lindex $result 3]
EXP
    rm -f "$T/refreshed"; : > "$T/gh.log"; : > "$T/docker.log"
    expect "$T/run-in-pty.exp" success "$script" > "$T/pty" 2>&1 || fail interactive-refresh
grep -q '^auth refresh --hostname github.com --scopes write:packages$' "$T/gh.log" || fail refresh-not-run
    grep -q '^push ' "$T/docker.log" || fail refresh-no-push
    refresh_line=$(grep -n '^auth refresh' "$T/gh.log" | head -1 | cut -d: -f1)
    token_line=$(grep -n '^auth token' "$T/gh.log" | head -1 | cut -d: -f1)
    [ "$refresh_line" -lt "$token_line" ] || fail token-read-before-refresh
    grep -q '^login ghcr.io --username example-user --password-stdin$' "$T/docker.log" || fail refresh-docker-login
    for mode in refresh-fail refresh-no-scope; do
        rm -f "$T/refreshed"; : > "$T/docker.log"
        if expect "$T/run-in-pty.exp" "$mode" "$script" > "$T/pty" 2>&1; then fail "$mode-accepted"; fi
        ! grep -q '^push ' "$T/docker.log" || fail "$mode-pushed"
    done
else
    echo 'SKIP: interactive refresh tests require expect' >&2
fi
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
