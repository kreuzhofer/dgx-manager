#!/bin/bash
# Build the agent, sync to NFS, and optionally deploy to all nodes.
# Usage:
#   ./scripts/build-agent.sh          # build + sync only
#   ./scripts/build-agent.sh --deploy  # build + sync + deploy to all nodes
set -e

cd "$(dirname "$0")/.."

echo "Building agent..."
npx tsc -p packages/agent/tsconfig.json

NFS_TARGET="/mnt/tank/src/github/dgx-manager/packages/agent"
echo "Syncing to NFS ($NFS_TARGET)..."
rsync -a packages/agent/dist/ "$NFS_TARGET/dist/"
cp packages/agent/package.json "$NFS_TARGET/package.json"
cp packages/agent/src/ollama-models.json "$NFS_TARGET/ollama-models.json" 2>/dev/null || true

VERSION=$(node -e "console.log(require('./packages/agent/package.json').version)")
echo "Agent v$VERSION built and synced."

if [ "$1" = "--deploy" ]; then
  API_URL="${API_URL:-http://localhost:4000}"
  echo "Deploying to all nodes..."
  NODE_IDS=$(curl -s "$API_URL/api/nodes" | python3 -c "import sys,json; [print(n['id']) for n in json.load(sys.stdin)]")
  for NODE_ID in $NODE_IDS; do
    STATUS=$(curl -s -X POST "$API_URL/api/nodes/$NODE_ID/deploy-agent" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','error'))")
    echo "  $NODE_ID: $STATUS"
  done
  echo "Done. Agents will reconnect with v$VERSION."
fi
