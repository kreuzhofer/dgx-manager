#!/usr/bin/env bash
# build-rtx-vllm-image.sh — build the entrypoint-less vLLM image for the amd64
# RTX-5090 host (aihost01).
#
# There is NO kernel work here and there must not be: the RTX 5090 is sm_120
# consumer Blackwell, which stock upstream vLLM already supports. Unlike
# build-glm52-image.sh / build-glm53-flash-image.sh — which exist because the
# GB10 needs sm_121a kernels nobody publishes — this script adds exactly one
# metadata layer to an official image.
#
#   base   ->  vllm/vllm-openai:<tag>   (pulled, not built — amd64 manifest)
#   final  ->  rtx-vllm:<tag>           (ours: ENTRYPOINT [])
#
# WHY IT EXISTS: dgxrun passes the whole serve argv as the container COMMAND,
# so the image must not carry an ENTRYPOINT of its own. The official image sets
# ENTRYPOINT ["vllm","serve"], which makes the recipe's command land as
# arguments to an already-chosen subcommand and the container dies at once with
# `vllm: error: unrecognized arguments: serve <model>`. See the overlay
# Dockerfile for the full note.
#
# Usage:   ./scripts/build-rtx-vllm-image.sh [node-ip ...]     (default: aihost01)
#          BASE_TAG=v0.28.0 ./scripts/build-rtx-vllm-image.sh
# Overlay: scripts/rtx-vllm-overlay/
# Recipe:  recipes/dgxrun/qwen3.8-27b-nvfp4-rtx.yaml
#
# Pin a DATED release tag, never :latest — `:latest` drifts per node and
# silently changes what a benchmark measured.
set -euo pipefail

BASE_TAG="${BASE_TAG:-v0.28.0}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:${BASE_TAG}}"
FINAL_TAG="${FINAL_TAG:-rtx-vllm:${BASE_TAG}}"
SSH_USER="${SSH_USER:-daniel}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY_SRC="${OVERLAY_SRC:-$REPO_ROOT/scripts/rtx-vllm-overlay}"

NODES=("$@")
if [ ${#NODES[@]} -eq 0 ]; then NODES=(192.168.44.30); fi

die() { echo "ERROR: $*" >&2; exit 1; }
on() { ssh -o ControlMaster=auto -o ControlPath="$HOME/.ssh/cm/%r@%h:%p" \
           -o ControlPersist=5m -o StrictHostKeyChecking=accept-new \
           "${SSH_USER}@$1" "${@:2}"; }

[ -f "$OVERLAY_SRC/Dockerfile.dgxrun-entrypoint" ] || die "overlay missing at $OVERLAY_SRC"
mkdir -p "$HOME/.ssh/cm"

for ip in "${NODES[@]}"; do
  echo "==> [$ip] pulling $BASE_IMAGE"
  # dgxrun never pulls, so the base must be on the node before we layer on it.
  on "$ip" "docker pull '$BASE_IMAGE'" || die "[$ip] pull failed"

  echo "==> [$ip] building $FINAL_TAG (one metadata layer)"
  # The build context is a single Dockerfile piped over stdin — there are no
  # files to ship, and a stdin build cannot go stale on an NFS scratch dir.
  on "$ip" "docker build -t '$FINAL_TAG' --build-arg FROM_TAG='$BASE_IMAGE' -f - . " \
     < "$OVERLAY_SRC/Dockerfile.dgxrun-entrypoint" || die "[$ip] build failed"

  echo "==> [$ip] verifying the entrypoint is actually cleared"
  ep=$(on "$ip" "docker image inspect '$FINAL_TAG' --format '{{json .Config.Entrypoint}}'" | tr -d '\r')
  # A non-null entrypoint here means the build silently kept the base's, which
  # would fail later inside vLLM's arg parser rather than here.
  [ "$ep" = "null" ] || [ "$ep" = "[]" ] || die "[$ip] entrypoint not cleared: $ep"
  echo "==> [$ip] OK — $FINAL_TAG entrypoint=$ep"
done

echo
echo "Built $FINAL_TAG on: ${NODES[*]}"
echo "Deploy with recipes/dgxrun/qwen3.8-27b-nvfp4-rtx.yaml (container: $FINAL_TAG)."
