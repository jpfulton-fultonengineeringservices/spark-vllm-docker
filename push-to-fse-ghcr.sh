#!/usr/bin/env bash
set -euo pipefail

GHCR_REPO="ghcr.io/fulton-engineering-services/spark-vllm"
CLUSTER_HOST="gx10-node1:5000"
CLUSTER_REPO="$CLUSTER_HOST/spark-vllm"
API='/orgs/fulton-engineering-services/packages/container/spark-vllm/versions?per_page=100'

usage() {
    cat <<'EOF'
Usage:
  push-to-fse-ghcr.sh push IMAGE [--cluster] [--no-cluster-latest] [--insecure-cluster] [--new-package]
  push-to-fse-ghcr.sh list IMAGE
  push-to-fse-ghcr.sh pull IMAGE [VERSIONED_TAG]

IMAGE is the simple name passed to build-and-copy.sh -t, optionally :latest.
Versioned tags use IMAGE-YYYYMMDD-N (UTC). Pull retags locally as IMAGE:latest.
--cluster also publishes the same version to node1 and IMAGE:latest there;
--no-cluster-latest skips the moving alias. --insecure-cluster skips TLS
verification only for the cluster catalog request, not for Docker.
--new-package acknowledges a GHCR 404 when publishing a new package.
Cross-machine publishing must use a single writer.
EOF
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
[[ $# -gt 0 ]] || { usage >&2; exit 2; }
cmd=$1; shift
case "$cmd" in -h|--help|help) usage; exit 0;; push|pull|list) ;; *) usage >&2; exit 2;; esac
[[ $# -gt 0 ]] || { usage >&2; exit 2; }
input=$1; shift
base=${input%:latest}
[[ "$input" == "$base" || "$input" == "$base:latest" ]] || fail 'only unqualified image names or :latest are supported'
[[ "$base" =~ ^[a-z0-9]+([._-][a-z0-9]+)*$ ]] || fail 'image name must be a simple lowercase Docker repository name'
local_image="$base:latest"
cluster=0; latest=1; insecure=0; new_package=0; selected=''
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster) [[ "$cmd" == push ]] || fail '--cluster is only valid for push'; cluster=1;;
        --no-cluster-latest) [[ "$cmd" == push ]] || fail '--no-cluster-latest is only valid for push'; latest=0;;
        --insecure-cluster) [[ "$cmd" == push ]] || fail '--insecure-cluster is only valid for push'; insecure=1;;
        --new-package) [[ "$cmd" == push ]] || fail '--new-package is only valid for push'; new_package=1;;
        -*) fail "unknown option: $1";;
        *) [[ "$cmd" == pull && -z "$selected" ]] || fail "unexpected argument: $1"; selected=$1;;
    esac
    shift
done
if [[ "$cluster" == 0 && ( "$latest" == 0 || "$insecure" == 1 ) ]]; then fail 'cluster options require --cluster'; fi
for dependency in gh jq; do command -v "$dependency" >/dev/null || fail "$dependency is required"; done
if [[ "$cmd" != list ]]; then command -v docker >/dev/null || fail 'docker is required'; fi
if [[ "$cluster" == 1 ]]; then command -v curl >/dev/null || fail 'curl is required'; fi

work=$(mktemp -d "${TMPDIR:-/tmp}/fse-ghcr.XXXXXXXX") || fail 'cannot create temporary directory'
lock=''
cleanup() {
    [[ -z "$lock" ]] || rmdir "$lock" 2>/dev/null || true
    rm -rf "$work"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# All gh output is buffered and validated before any tag is used.
gh_tags() {
    local allow_new=$1
    if gh api --paginate "$API" > "$work/gh.json" 2> "$work/gh.err"; then
        # Older gh releases emit a stream of page arrays without --slurp.
        # Validate the entire buffered stream before using any page's tags.
        jq -se 'length > 0 and all(.[]; type == "array" and all(.[];
            type == "object" and (.metadata.container.tags | type == "array" and all(.[]; type == "string"))))' \
            "$work/gh.json" >/dev/null || fail 'invalid GHCR package response'
        jq -sr '.[][] | .metadata.container.tags[]' \
            "$work/gh.json" > "$work/gh.tags" || fail 'invalid GHCR tags'
    elif [[ "$allow_new" == 1 ]] && grep -Eq 'HTTP 404' "$work/gh.err" && {
        # gh may emit a JSON error body on stdout. Accept only a single 404
        # error object, never earlier successful pages or malformed output.
        [[ ! -s "$work/gh.json" ]] || jq -se '
            length == 1 and (.[0] | type == "object" and (.status == "404" or .status == 404)
                and (.message | type == "string"))
        ' "$work/gh.json" >/dev/null 2>&1
    }; then
        printf '%s\n' 'Warning: treating GHCR 404 as a new package; 404 can also mean insufficient access.' >&2
        : > "$work/gh.tags"
    else
        cat "$work/gh.err" >&2
        if grep -Eq 'HTTP 404' "$work/gh.err"; then
            fail 'GHCR lookup failed; 404 may mean a missing or inaccessible package (use --new-package only to acknowledge first publication)'
        fi
        fail 'GHCR lookup failed; verify GitHub CLI authentication, permissions and connectivity'
    fi
}

cluster_tags() {
    local url="https://$CLUSTER_HOST/v2/spark-vllm/tags/list?n=1000" status link next count=0
    local curl_opts=(-sS -D "$work/headers" -o "$work/cluster.json" -w '%{http_code}')
    [[ "$insecure" == 0 ]] || curl_opts+=(-k)
    : > "$work/cluster.tags"
    while :; do
        count=$((count + 1)); [[ $count -le 1000 ]] || fail 'cluster pagination exceeded safety limit'
        if ! status=$(curl "${curl_opts[@]}" "$url"); then fail 'cluster registry query failed (network or TLS)'; fi
        if [[ "$status" == 404 ]]; then
            jq -e '.errors | type == "array" and any(.[]; .code == "NAME_UNKNOWN")' "$work/cluster.json" >/dev/null 2>&1 || fail 'cluster registry returned unexpected 404'
            [[ $count == 1 ]] || fail 'cluster pagination returned a missing repository'
            return
        fi
        [[ "$status" == 200 ]] || fail "cluster registry returned HTTP $status"
        jq -e 'type == "object" and (.name == "spark-vllm") and (.tags == null or (.tags | type == "array" and all(.[]; type == "string")))' "$work/cluster.json" >/dev/null || fail 'invalid cluster registry tag response'
        jq -r '.tags[]?' "$work/cluster.json" >> "$work/cluster.tags" || fail 'invalid cluster tags'
        link=$(tr -d '\r' < "$work/headers" | grep -i '^link:' | tail -1 || true)
        [[ -n "$link" ]] || break
        next=$(printf '%s\n' "$link" | sed -nE 's/^[Ll][Ii][Nn][Kk]:[[:space:]]*<([^>]*)>;[[:space:]]*rel="?next"?.*/\1/p')
        [[ -n "$next" ]] || fail 'unrecognized cluster pagination link'
        case "$next" in
            /v2/spark-vllm/tags/list\?*) url="https://$CLUSTER_HOST$next";;
            "https://$CLUSTER_HOST/v2/spark-vllm/tags/list?"*) url="$next";;
            *) fail 'unsafe cluster pagination URL';;
        esac
    done
}
matching_tags() {
    jq -Rrn --arg base "$base" '[inputs | select(startswith($base + "-")) | . as $tag | ltrimstr($base + "-") | capture("^(?<date>[0-9]{8})-(?<serial>0|[1-9][0-9]*)$")? | select(. != null) | select((.serial | length) <= 9) | {tag:$tag,date:.date,serial:(.serial | tonumber)}] | unique_by(.tag) | sort_by(.date,.serial) | .[].tag'
}

if [[ "$cmd" == push ]]; then
    lock="${TMPDIR:-/tmp}/fse-ghcr-${USER:-$(id -u)}-${base}.lock"
    if ! mkdir "$lock" 2>/dev/null; then lock=''; fail 'another local publisher holds the lock (or lock cannot be created)'; fi
    docker image inspect "$local_image" >/dev/null || fail "local image not found: $local_image"
fi
gh_tags "$new_package"
if [[ "$cluster" == 1 ]]; then cluster_tags; else : > "$work/cluster.tags"; fi
cat "$work/gh.tags" "$work/cluster.tags" | matching_tags > "$work/matches"

case "$cmd" in
    list) while IFS= read -r tag; do printf '%s:%s\n' "$GHCR_REPO" "$tag"; done < "$work/matches";;
    pull)
        [[ -n "$selected" ]] || selected=$(tail -1 "$work/matches")
        [[ -n "$selected" ]] || fail "no published versions for $base"
        grep -Fxq -- "$selected" "$work/matches" || fail "tag not found for $base: $selected"
        docker pull "$GHCR_REPO:$selected" || fail "pull failed: $GHCR_REPO:$selected"
        docker tag "$GHCR_REPO:$selected" "$local_image" || fail "local retag failed: $local_image"
        printf 'Pulled %s:%s -> %s\n' "$GHCR_REPO" "$selected" "$local_image";;
    push)
        day=$(date -u +%Y%m%d)
        [[ "$day" =~ ^[0-9]{8}$ ]] || fail 'invalid UTC date'
        serial=$(jq -Rn --arg base "$base" --arg day "$day" '[inputs | ltrimstr($base + "-" + $day + "-") | tonumber] | if length == 0 then 0 else max + 1 end' < <(grep -E "^${base//./\.}-${day}-[0-9]+$" "$work/matches" || true))
        tag="$base-$day-$serial"
        [[ ${#tag} -le 128 ]] || fail 'generated Docker tag exceeds 128 characters'
        # Repeat both listings under the local lock to avoid known collisions.
        gh_tags "$new_package"
        if [[ "$cluster" == 1 ]]; then cluster_tags; fi
        if grep -Fxq -- "$tag" "$work/gh.tags" "$work/cluster.tags"; then fail "version already exists: $tag"; fi
        gh_ref="$GHCR_REPO:$tag"; cluster_ref="$CLUSTER_REPO:$tag"; alias_ref="$CLUSTER_HOST/$base:latest"
        docker tag "$local_image" "$gh_ref" || fail "tag failed: $gh_ref"
        docker push "$gh_ref" || fail "push failed: $gh_ref"
        printf 'Published %s\n' "$gh_ref"
        if [[ "$cluster" == 1 ]]; then
            docker tag "$local_image" "$cluster_ref" || fail "tag failed: $cluster_ref (already published $gh_ref)"
            docker push "$cluster_ref" || fail "push failed: $cluster_ref (already published $gh_ref)"
            printf 'Published %s\n' "$cluster_ref"
            if [[ "$latest" == 1 ]]; then
                docker tag "$local_image" "$alias_ref" || fail "tag failed: $alias_ref (already published $gh_ref and $cluster_ref)"
                docker push "$alias_ref" || fail "push failed: $alias_ref (already published $gh_ref and $cluster_ref)"
                printf 'Published %s\n' "$alias_ref"
            fi
        fi;;
esac
