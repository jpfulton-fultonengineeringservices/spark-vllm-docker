#!/bin/bash
# [coder390-derived] Build and publish the reproducible coder390 serving image.
#
# Derives from the pinned public eugr release tag (see
# docker/Dockerfile.coder390-derived for why this base and not the b12x line),
# bakes the installed-stage patches this repo carries, and publishes the result
# to the node1 registry so the pin is content-addressed and survives upstream
# deletion of the base tag.
#
# Usage:
#   bash docker/build-coder390-derived-image.sh              # build + push
#   bash docker/build-coder390-derived-image.sh --no-push    # build + verify only
#   TAG=coder390-kr-$(date -u +%Y%m%d) bash ...              # override tag
#
# Runs on a node that can reach the registry with certs trusted (/etc/docker/certs.d).
# Requires: docker, network access to the eugr base tag.
set -euo pipefail

PREFIX="[coder390-derived]"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-eugr/spark-vllm:nightly-20261016}"
# Config digest of the previously-pinned a1fc4f18 content. Asserting the base
# resolves to this keeps the recipe's vLLM version fixed; if eugr ever retags
# this date, the build stops instead of silently shipping a different vLLM.
EXPECTED_BASE_ID="${EXPECTED_BASE_ID:-sha256:a1fc4f18d3cb1671986feb92b05243ecd31a21957dd64e28da5c9e6125733aa0}"
REGISTRY="${REGISTRY:-10.100.170.3:5000}"
IMAGE_NAME="${IMAGE_NAME:-spark-vllm}"
TAG="${TAG:-coder390-kr-$(date -u +%Y%m%d)}"
LOCAL_TAG="${LOCAL_TAG:-vllm-node-b12x-kr}"
NO_PUSH=0

for arg in "$@"; do
    case "$arg" in
        --no-push) NO_PUSH=1 ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "$PREFIX unknown argument: $arg" >&2; exit 2 ;;
    esac
done

echo "$PREFIX building from $BASE_IMAGE"

docker pull "$BASE_IMAGE" >/dev/null
BASE_ID="$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")"
if [ -n "$EXPECTED_BASE_ID" ] && [ "$BASE_ID" != "$EXPECTED_BASE_ID" ]; then
    echo "$PREFIX FAIL: base $BASE_IMAGE resolves to $BASE_ID" >&2
    echo "$PREFIX       expected $EXPECTED_BASE_ID (the validated a1fc4f18 content)" >&2
    echo "$PREFIX       The base moved; re-validate the vLLM version before using it." >&2
    exit 1
fi
echo "$PREFIX base verified: $BASE_ID"

docker build \
    -f "$REPO_ROOT/docker/Dockerfile.coder390-derived" \
    -t "$LOCAL_TAG" \
    "$REPO_ROOT"

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$LOCAL_TAG")"
echo "$PREFIX built $LOCAL_TAG = $IMAGE_ID"

# Acceptance gates that do not need a GPU. These catch the two failure modes
# that made earlier source rebuilds unservable (NCCL symbol, InstantTensor gate)
# plus a missing patch.
docker run --rm --entrypoint bash "$LOCAL_TAG" -c '
set -e
python3 -c "import torch" >/dev/null 2>&1 || { echo "FAIL: import torch (NCCL trap)"; exit 1; }
markers=$(grep -c "accept splittable storage block" /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/b12x.py || true)
[ "${markers:-0}" -gt 0 ] || { echo "FAIL: storage-block patch missing"; exit 1; }
gate=$(grep -c _determine_buffer_size /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py || true)
[ "${gate:-0}" -eq 0 ] || { echo "FAIL: InstantTensor gate present, engine will not load"; exit 1; }
python3 -c "import vllm; print(\"vllm\", vllm.__version__)"
' || { echo "$PREFIX FAIL: image gates" >&2; exit 1; }
echo "$PREFIX gates passed"

if [ "$NO_PUSH" -eq 1 ]; then
    echo "$PREFIX --no-push: built and gated $LOCAL_TAG ($IMAGE_ID), skipping publish"
    exit 0
fi

REG_REF="$REGISTRY/$IMAGE_NAME:$TAG"
docker tag "$LOCAL_TAG" "$REG_REF"
docker push "$REG_REF"
echo "$PREFIX published $REG_REF"

# Republishing must not silently change what the tag serves.
PUBLISHED_ID="$(curl -ks -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
    "https://$REGISTRY/v2/$IMAGE_NAME/manifests/$TAG" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["config"]["digest"])')"
if [ "$PUBLISHED_ID" != "$IMAGE_ID" ]; then
    echo "$PREFIX FAIL: registry serves $PUBLISHED_ID, expected $IMAGE_ID" >&2
    exit 1
fi
echo "$PREFIX registry config digest matches: $PUBLISHED_ID"
echo "$PREFIX pin this in the recipes: image_id: $IMAGE_ID"
