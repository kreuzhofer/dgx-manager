#!/usr/bin/env bash
# Build the agent bundle tarball for HTTP serving.
# Output: packages/server/agent-bundle.tar.gz
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
AGENT_DIR="${ROOT_DIR}/packages/agent"
OUTPUT="${ROOT_DIR}/packages/server/agent-bundle.tar.gz"

echo "Building agent TypeScript..."
npx tsc -p "${AGENT_DIR}/tsconfig.json"

echo "Creating agent bundle tarball..."
tar -czf "$OUTPUT" \
  -C "$AGENT_DIR" \
  dist/ \
  package.json \
  node_modules/

echo "Bundle created: ${OUTPUT} ($(du -h "$OUTPUT" | cut -f1))"
