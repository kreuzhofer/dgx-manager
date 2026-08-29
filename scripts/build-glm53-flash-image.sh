#!/usr/bin/env bash
# build-glm53-flash-image.sh — build the GLM-5.3-Flash sm_121 image for the GB10 cluster.
#
# GLM-5.3-Flash is `glm5_next`: 45 layers, hybrid KDA linear attention, NoPE MLA
# (qk_rope_head_dim=0), natively multimodal. It shares NOTHING with GLM-5.2's
# `glm_moe_dsa` beyond a vendor. Do not reach for build-glm52-image.sh, the
# CosmicRaisins kernels, or b12x — none of them apply here.
#
# Unlike the GLM-5.2 build there is NO vLLM compile: vLLM published a working
# day-0 arm64 image and every layer below is a Python source patch or a pip pin.
# Expect minutes, not the hour build-glm52-image.sh costs.
#
#   base   ->  vllm/vllm-openai:glm53-flash-arm64-cu130   (pulled, not built)
#   v1..v8 ->  glm53:sm121-vN                              (vendored, unmodified)
#   final  ->  vllm-glm53-flash-sm121:probe                (ours: ENTRYPOINT [])
#
# Usage:   ./scripts/build-glm53-flash-image.sh [node-ip ...]     (default: spark-02, spark-03)
# Overlay: scripts/glm53-flash-overlay/  (provenance + per-layer notes in its README)
# Recipe:  recipes/dgxrun/glm-5.3-flash-nvfp4-2x.yaml
#
# Builds INDEPENDENTLY on each node rather than building once and copying: every
# pin in the stack is an exact version, so the builds are reproducible, and a
# ~20 GB docker save/load between nodes costs more than re-running eight thin
# layers. There is deliberately no --copy-to path to drift out of date.
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:glm53-flash-arm64-cu130}"
FINAL_TAG="${FINAL_TAG:-vllm-glm53-flash-sm121:probe}"
SSH_USER="${SSH_USER:-daniel}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY_SRC="${OVERLAY_SRC:-$REPO_ROOT/scripts/glm53-flash-overlay}"
# Scratch build context on NFS so every node sees the same bytes. Wiped per run.
OVERLAY_BUILD="${OVERLAY_BUILD:-/mnt/tank/src/glm53-flash-overlay}"

# The vendored FROM chain, in order. Order is NOT optional: v8 patches files that
# v3's pip install replaces, so building v8 before v3 silently patches a package
# that is about to be overwritten.
LAYERS=(v1 v2 v3 v4 v5 v6 v7 v8)

NODES=("$@")
if [ ${#NODES[@]} -eq 0 ]; then
  NODES=(192.168.44.37 192.168.44.38)   # spark-02, spark-03
fi

say(){ printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die(){ printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
on(){ ssh -o BatchMode=yes -o ConnectTimeout=10 "${SSH_USER}@$1" "${@:2}"; }

# ---- 0. prereqs --------------------------------------------------------------
say "Checking prerequisites"
[ -f "$OVERLAY_SRC/Dockerfile.sm121-v1" ] || die "overlay context missing at $OVERLAY_SRC"
[ -f "$OVERLAY_SRC/Dockerfile.dgxrun-entrypoint" ] || die "entrypoint layer missing at $OVERLAY_SRC"
for ip in "${NODES[@]}"; do
  on "$ip" "true" || die "cannot SSH to $ip"
  arch="$(on "$ip" "uname -m")"
  [ "$arch" = "aarch64" ] || die "$ip is $arch — this image is arm64/sm_121 only"
done

# ---- 1. publish the build context --------------------------------------------
say "Staging overlay context -> $OVERLAY_BUILD"
rm -rf "$OVERLAY_BUILD"
mkdir -p "$OVERLAY_BUILD"
cp "$OVERLAY_SRC"/Dockerfile.sm121-v* "$OVERLAY_SRC"/patch_v7.py \
   "$OVERLAY_SRC"/patch_v8_fp8.py "$OVERLAY_SRC"/Dockerfile.dgxrun-entrypoint "$OVERLAY_BUILD"/

# ---- 2. per-node build -------------------------------------------------------
for ip in "${NODES[@]}"; do
  say "[$ip] Pulling base $BASE_IMAGE"
  on "$ip" "docker pull -q $BASE_IMAGE" || die "[$ip] base pull failed"

  # The entrypoint reset only makes sense if the base HAS one. If upstream ever
  # ships an entrypoint-less image, that layer becomes a silent no-op and the
  # recipe would still work — but we would rather know the assumption changed.
  ep="$(on "$ip" "docker image inspect $BASE_IMAGE --format '{{json .Config.Entrypoint}}'")"
  say "[$ip] base ENTRYPOINT = $ep"
  case "$ep" in
    ""|null|"[]") echo "  NOTE: base has no entrypoint; the reset layer is a no-op now." ;;
  esac

  prev="$BASE_IMAGE"
  for L in "${LAYERS[@]}"; do
    say "[$ip] Building glm53:sm121-$L"
    # Each vendored Dockerfile carries its own FROM (glm53:sm121-v{N-1}); only v1
    # names the upstream base. They are unmodified copies, so the chain already
    # lines up and we just build them in order.
    on "$ip" "cd $OVERLAY_BUILD && docker build -f Dockerfile.sm121-$L -t glm53:sm121-$L ." \
      || die "[$ip] layer $L failed — if it says 'refusing to patch', the base image moved and the overlay needs re-pinning"
    prev="glm53:sm121-$L"
  done

  say "[$ip] Building $FINAL_TAG (ENTRYPOINT reset)"
  on "$ip" "cd $OVERLAY_BUILD && docker build -f Dockerfile.dgxrun-entrypoint --build-arg FROM_TAG=$prev -t $FINAL_TAG ." \
    || die "[$ip] entrypoint layer failed"

  # Import smoke test only. It does NOT prove the sm_121 patches work — that is
  # what the first deploy is for — but it does catch a broken pip pin, which is
  # the failure this stack is most exposed to.
  say "[$ip] Smoke-testing $FINAL_TAG"
  on "$ip" "docker run --rm --entrypoint python3 $FINAL_TAG -c \"
import torch, vllm, flashinfer
print('torch', torch.__version__, '| vllm', vllm.__version__, '| flashinfer', flashinfer.__version__)
assert flashinfer.__version__.startswith('0.6.18'), 'flashinfer pin lost: ' + flashinfer.__version__
import importlib.metadata as md
for pkg, want in (('nvidia-nccl-cu13','2.30.7'), ('nvidia-cutlass-dsl','4.6.2')):
    got = md.version(pkg)
    assert got == want, pkg + ' is ' + got + ', expected ' + want
print('pins OK')
\"" || die "[$ip] smoke test failed — a pip pin did not survive the layer order"
done

# ---- 3. verify ---------------------------------------------------------------
say "Verifying $FINAL_TAG on every node"
for ip in "${NODES[@]}"; do
  id="$(on "$ip" "docker images -q $FINAL_TAG | head -1")"
  [ -n "$id" ] && echo "  $ip  OK ($id)" || die "  $ip  MISSING $FINAL_TAG"
done

say "Done -> $FINAL_TAG on ${NODES[*]}"
echo "  Deploy with recipes/dgxrun/glm-5.3-flash-nvfp4-2x.yaml on exactly these nodes."
