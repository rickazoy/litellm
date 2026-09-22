#!/usr/bin/env bash
# Build the fork image with its commit stamped in, or not at all.
#
# The image this replaces was built by hand as witos/litellm:witos-main, a tag
# that moved, from a working tree nobody recorded. The gateway on MI1 runs on
# top of it and cannot say which fork commit it contains. This script exists so
# that stops being possible: it refuses a dirty tree, stamps the exact commit
# into a label and an environment variable, tags the image by that commit, and
# then reads the label back to prove it took.
#
#   scripts/witos/build-image.sh            # builds witos/litellm:<sha>
#   scripts/witos/build-image.sh --also-tag witos-main
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain)" ]; then
  echo "refusing to build: working tree is dirty. An image must name a commit, and a" >&2
  echo "dirty tree is not one. Commit or discard, then build." >&2
  exit 2
fi

SHA=$(git rev-parse --short=10 HEAD)
BUILT_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
IMAGE="witos/litellm:${SHA}"
TAGS=(-t "$IMAGE")
if [ "${1:-}" = "--also-tag" ] && [ -n "${2:-}" ]; then
  TAGS+=(-t "witos/litellm:$2")
fi

echo "building ${IMAGE} from $(git log -1 --format=%h\ %s)"
docker build \
  --build-arg WITOS_FORK_SHA="$SHA" \
  --build-arg WITOS_FORK_BUILT_AT="$BUILT_AT" \
  "${TAGS[@]}" .

# Verify the stamp took. A build step that silently did nothing would leave an
# image that looks built and answers "unknown", which is the defect this fixes.
GOT=$(docker inspect "$IMAGE" --format "{{index .Config.Labels \"org.opencontainers.image.revision\"}}")
if [ "$GOT" != "$SHA" ]; then
  echo "BUILD FAILED: image label reads ${GOT:-<empty>}, expected ${SHA}" >&2
  exit 1
fi
echo "built ${IMAGE}  revision=${GOT}  created=${BUILT_AT}"
