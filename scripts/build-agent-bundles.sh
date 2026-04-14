#!/usr/bin/env bash
# Build per-architecture agent bundles (amd64 + arm64) using docker buildx.
# Runs on any host; cross-architecture builds use QEMU/binfmt emulation.
#
# One-time host setup (registers binfmt handlers for foreign archs):
#   docker run --privileged --rm tonistiigi/binfmt --install all
#
# Output: packages/server/agent-bundles/agent-bundle-{amd64,arm64}.tar.gz
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUT="${ROOT_DIR}/packages/server/agent-bundles"

mkdir -p "$OUT"

# Ensure a buildx builder exists and is selected
if ! docker buildx inspect dgx-bundle-builder >/dev/null 2>&1; then
  docker buildx create --name dgx-bundle-builder --use >/dev/null
else
  docker buildx use dgx-bundle-builder >/dev/null
fi

for ARCH in amd64 arm64; do
  echo "==> Building agent bundle for linux/${ARCH}"
  STAGE_DIR="${OUT}/_${ARCH}"
  rm -rf "$STAGE_DIR"
  docker buildx build \
    --platform "linux/${ARCH}" \
    --file "${ROOT_DIR}/Dockerfile.agent-bundle" \
    --target export \
    --output "type=local,dest=${STAGE_DIR}" \
    "${ROOT_DIR}"
  mv "${STAGE_DIR}/agent-bundle.tar.gz" "${OUT}/agent-bundle-${ARCH}.tar.gz"
  rm -rf "$STAGE_DIR"
done

echo "==> Bundles:"
ls -lh "${OUT}"
