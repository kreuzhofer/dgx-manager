#!/usr/bin/env bash
# build-glm52-image.sh — reproducibly build a GLM-5.2 arm64/sm_121 vLLM image for the GB10 cluster.
#
# One orchestrator, two variants (same build, different pins):
#   legacy  ->  vllm-node-tf5-glm52-b12x:probe        (the CURRENT production image; ~85-159K, no DCP)
#   dcp     ->  vllm-node-tf5-glm52-b12x-dcp:probe    (long-context; serves 15pct at ~262K via DCP2)
#
# Both = the eugr spark-vllm-docker base compile at a pinned vLLM ref + the CosmicRaisins sm12x
# DSA kernels + b12x overlay + deep_gemm re-bind. The dcp variant points the base at the
# local-inference-lab DCP fork (branch already contains the DCP draft path + MTP top-k
# score-buffer fix aka "PR#72") and refreshes the DCP-matched kernels.
#
# Usage:   ./scripts/build-glm52-image.sh [legacy|dcp]     (default: dcp)
# Docs:    docs/glm-5.2-dcp-image-build.md , docs/glm-5.2-custom-image-build.md
# Recipes: recipes/dgxrun/glm-5.2-awq-15pct{,-144k}.yaml (legacy) , -dcp2.yaml (dcp)
#
# PREREQUISITES (external, not vendored here — pinned + referenced, per each doc):
#   - eugr spark-vllm-docker clone at $SPARK_VLLM_DOCKER (build infra; ~1h arm64 compile)
#   - glm52 overlay context at $OVERLAY_SRC (Dockerfile.glm52-overlay + patch_deep_gemm.py + kernels/)
#   - key-based BatchMode SSH from here to every node; docker on each node
#   - a GB10 node to build on (arm64/sm_121) — the Pi cannot compile this
set -euo pipefail

VARIANT="${1:-dcp}"

case "$VARIANT" in
  legacy)
    VLLM_REF="${VLLM_REF:-ab666069935c1f23e8ef56038b4659ac9e8f19f8}"   # bootstrap.sh VLLM_REF (upstream)
    FORK_URL=""                                                        # upstream vllm-project/vllm — no fork patch
    BASE_TAG="${BASE_TAG:-vllm-node-tf5-glm52-b12x:base}"
    PROBE_TAG="${PROBE_TAG:-vllm-node-tf5-glm52-b12x:probe}"
    OVERLAY_KERNELS=true            # legacy branch lacks the sm12x kernels in-tree — overlay all 10
    REFRESH_KERNELS="${REFRESH_KERNELS:-false}"                        # use overlay kernels as-is (matched to this ref)
    B12X_VER="${B12X_VERSION:-0.23.0}"
    ;;
  dcp)
    VLLM_REF="${VLLM_REF:-e232d262369b8c918cf478a7a96a0fcf8127cf65}"   # codex/dcp-globaltopk-sharddraft-defaults-20260622
    FORK_URL="${FORK_URL:-https://github.com/local-inference-lab/vllm.git}"
    BASE_TAG="${BASE_TAG:-vllm-node-tf5-glm52-b12x-dcp:base}"
    PROBE_TAG="${PROBE_TAG:-vllm-node-tf5-glm52-b12x-dcp:probe}"
    OVERLAY_KERNELS=false           # DCP branch ships its kernels natively — overlaying them REGRESSES it
    REFRESH_KERNELS=false
    B12X_VER="${B12X_VERSION:-0.30.0}"  # 0.30.0 has index_topk_fp8(out_scores); 0.23.0 does not
    ;;
  *) echo "unknown variant '$VARIANT' (expected: legacy | dcp)" >&2; exit 2 ;;
esac

# ---- shared config (override via env) ----------------------------------------
# Repo root, so vendored build inputs resolve in-repo regardless of CWD.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNELS_REPO="${KERNELS_REPO:-CosmicRaisins/glm-5.2-gb10}"
B12X_VERSION="${B12X_VERSION:-0.23.0}"
SPARK_VLLM_DOCKER="${SPARK_VLLM_DOCKER:-/mnt/tank/src/github/spark-vllm-docker}"
OVERLAY_SRC="${OVERLAY_SRC:-$REPO_ROOT/scripts/glm52-overlay}"   # vendored in-repo (was /mnt/tank/src/glm52-overlay)
OVERLAY_BUILD="${OVERLAY_BUILD:-/mnt/tank/src/glm52-overlay-$VARIANT}"   # scratch: wiped+regenerated each build, on NFS so the build node sees it
BUILD_NODE="${BUILD_NODE:-192.168.44.38}"
COPY_NODES="${COPY_NODES:-192.168.44.36,192.168.44.37,192.168.44.39}"
SSH_USER="${SSH_USER:-daniel}"

# The 10 CosmicRaisins kernels (destinations baked into Dockerfile.glm52-overlay).
KERNELS=(sparse_mla_kernels.py sparse_mla_env.py sm12x_sparse_mla_attn.py patch_flashmla_ops.py
         flashmla_sparse.py sm12x_deep_gemm_fallbacks.py sm12x_mqa.py b12x_sparse_helpers.py
         sparse_attn_indexer.py deepseek_v2.py)

say(){ printf '\n\033[1;36m==> [%s] %s\033[0m\n' "$VARIANT" "$*"; }
die(){ printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
on(){ ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_USER}@$1" "${@:2}"; }

# ---- 0. prereqs --------------------------------------------------------------
say "Checking prerequisites (ref=$VLLM_REF fork=${FORK_URL:-<upstream>})"
[ -x "$SPARK_VLLM_DOCKER/build-and-copy.sh" ] || die "eugr build infra missing at $SPARK_VLLM_DOCKER"
[ -f "$OVERLAY_SRC/Dockerfile.glm52-overlay" ] || die "overlay context missing at $OVERLAY_SRC"
[ "$REFRESH_KERNELS" = true ] && { command -v gh >/dev/null || die "gh CLI required to refresh kernels"; }
on "$BUILD_NODE" "true" || die "cannot SSH to build node $BUILD_NODE"

# ---- 1. fork-source patch (dcp only; idempotent, backed up) -------------------
DF="$SPARK_VLLM_DOCKER/Dockerfile"
if [ -n "$FORK_URL" ]; then
  say "Patching fork source into $DF (backup: Dockerfile.pre-dcp-backup)"
  if grep -q "git remote add dcpfork" "$DF"; then echo "  already patched"; else
    cp "$DF" "$DF.pre-dcp-backup"
    python3 - "$DF" "$FORK_URL" <<'PY'
import sys, re
df, fork = sys.argv[1], sys.argv[2]
src = open(df).read()
old = re.search(r'RUN --mount=type=cache,id=repo-cache.*?cp -a /repo-cache/vllm \$VLLM_BASE_DIR/', src, re.S)
if not old: sys.exit("could not locate the vLLM clone RUN block — patch manually")
new = ('RUN --mount=type=cache,id=repo-cache,target=/repo-cache \\\n'
       '    cd /repo-cache && \\\n'
       '    if [ ! -d "vllm" ]; then git clone --recursive https://github.com/vllm-project/vllm.git; fi && \\\n'
       '    cd vllm && \\\n'
       '    (git remote remove dcpfork 2>/dev/null || true) && \\\n'
       f'    git remote add dcpfork {fork} && \\\n'
       '    git fetch dcpfork && git fetch origin --tags --force && \\\n'
       '    (git checkout --detach ${VLLM_REF} 2>/dev/null || git checkout ${VLLM_REF}) && \\\n'
       '    git submodule update --init --recursive && \\\n'
       '    git clean -fdx && git gc --auto && \\\n'
       '    cp -a /repo-cache/vllm $VLLM_BASE_DIR/')
open(df, 'w').write(src[:old.start()] + new + src[old.end():])
print("  patched vLLM clone block for the fork")
PY
  fi
fi

# ---- 2. base image: the ~1h arm64 vLLM compile at the pinned ref --------------
# --vllm-ref triggers a wheel rebuild; use_existing_torch.py keeps the base's torch (no bump).
say "Building base $BASE_TAG @ $VLLM_REF (arm64, ~1h) on $BUILD_NODE"
on "$BUILD_NODE" "cd $SPARK_VLLM_DOCKER && ./build-and-copy.sh --vllm-ref $VLLM_REF -t $BASE_TAG --tf5" \
  || die "base build failed — verify torch is NOT +cpu and the checkout resolved $VLLM_REF"

# ---- 3. overlay ----------------------------------------------------------------
# legacy: overlay the 10 sm12x kernels + b12x + deep_gemm re-bind (Dockerfile.glm52-overlay).
# dcp: e232d26 ships its kernels natively -> overlay b12x ONLY (a generated 2-line Dockerfile);
#      copying the legacy kernels REGRESSES the DCP branch. No deep_gemm patch (native routing).
say "Preparing overlay $OVERLAY_BUILD (kernels=$OVERLAY_KERNELS, b12x=$B12X_VER)"
rm -rf "$OVERLAY_BUILD"; mkdir -p "$OVERLAY_BUILD"
if [ "$OVERLAY_KERNELS" = true ]; then
  cp -r "$OVERLAY_SRC"/. "$OVERLAY_BUILD"/
  if [ "$REFRESH_KERNELS" = true ]; then
    for f in "${KERNELS[@]}"; do
      gh api "repos/$KERNELS_REPO/contents/kernels/$f" --jq '.content' | base64 -d > "$OVERLAY_BUILD/kernels/$f"
      [ -s "$OVERLAY_BUILD/kernels/$f" ] || die "kernel fetch empty: $f"
    done
  fi
  sed -i "s#^FROM .*:base#FROM $BASE_TAG#" "$OVERLAY_BUILD/Dockerfile.glm52-overlay"
  sed -i "s#b12x==[0-9.]*#b12x==$B12X_VER#" "$OVERLAY_BUILD/Dockerfile.glm52-overlay"
  OVERLAY_DF=Dockerfile.glm52-overlay
else
  printf 'FROM %s\nRUN pip install --no-deps b12x==%s\n' "$BASE_TAG" "$B12X_VER" > "$OVERLAY_BUILD/Dockerfile.glm52-overlay-dcp"
  OVERLAY_DF=Dockerfile.glm52-overlay-dcp
fi

say "Building overlay $PROBE_TAG on $BUILD_NODE"
on "$BUILD_NODE" "cd $OVERLAY_BUILD && docker build -f $OVERLAY_DF -t $PROBE_TAG ." \
  || die "overlay build failed"

# ---- 4. smoke test + distribute ----------------------------------------------
say "Smoke-testing $PROBE_TAG"
on "$BUILD_NODE" "docker run --rm --gpus all $PROBE_TAG python3 -c 'import torch,b12x,vllm; print(\"import OK\", torch.__version__)'" \
  || die "import smoke test failed (b12x pin? kernel paths?)"

say "Distributing $PROBE_TAG -> $COPY_NODES"
on "$BUILD_NODE" "cd $SPARK_VLLM_DOCKER && ./build-and-copy.sh --no-build --copy-to $COPY_NODES -t $PROBE_TAG" \
  || die "distribute failed"

IFS=',' read -ra NODES <<< "$BUILD_NODE,$COPY_NODES"
for ip in "${NODES[@]}"; do
  id="$(on "$ip" "docker images -q $PROBE_TAG | head -1")"
  [ -n "$id" ] && echo "  $ip  OK ($id)" || die "  $ip  MISSING $PROBE_TAG"
done

say "Done -> $PROBE_TAG on all nodes."
[ -n "$FORK_URL" ] && echo "  (optional) restore eugr Dockerfile: mv $DF.pre-dcp-backup $DF"
