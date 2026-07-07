#!/usr/bin/env bash
# dgx-node-prep.sh — one-time per-node prep for large GLM-5.2 DCP launches on GB10 (DGX Spark).
# Run as root ON EACH NODE (or: for n in 36 37 38 39; do ssh .$n 'sudo bash -s' < scripts/dgx-node-prep.sh; done).
#
# GB10 has NO separate VRAM — weights + KV + CUDA graphs + Linux page cache all share the ~124 GB
# unified pool. This preps the node so CUDA-graph capture / first-run JIT (a transient +20-30 GB
# spike) has headroom, and so an OOM is a RECOVERABLE kill instead of a whole-node freeze.
# See docs/glm-5.2-dcp-nodeprep-retry.md + memory glm52-dcp-stack-build. Idempotent.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)"; exit 1; }

echo "== 1. sysctls (persist + apply) =="
cat > /etc/sysctl.d/90-dgx-dcp.conf <<'EOF'
# Reserve 5 GiB the page cache cannot consume -> guaranteed headroom for capture/JIT.
vm.min_free_kbytes = 5242880
# Reclaim dentry/inode cache aggressively; cap dirty pages so writeback can't hoard DRAM.
vm.vfs_cache_pressure = 200
vm.dirty_ratio = 5
vm.dirty_background_ratio = 2
# Avoid swap thrash on unified memory (swap turns a clean OOM into a whole-node freeze).
vm.swappiness = 1
EOF
sysctl --system >/dev/null
echo "   min_free_kbytes=$(cat /proc/sys/vm/min_free_kbytes)  swappiness=$(cat /proc/sys/vm/swappiness)"

echo "== 2. disable swap (freeze -> recoverable OOM) =="
swapoff -a || true
# comment any swap lines in fstab so it stays off across reboots
sed -i.bak '/\sswap\s/s/^\([^#]\)/#\1/' /etc/fstab 2>/dev/null || true
echo "   active swap: $(swapon --show --noheadings | wc -l) devices"

echo "== 3. earlyoom (keep the node reachable if OOM hits) =="
if command -v earlyoom >/dev/null 2>&1 || apt-get install -y earlyoom >/dev/null 2>&1; then
  mkdir -p /etc/default
  echo 'EARLYOOM_ARGS="-m 2 -s 100 --prefer (^|/)(vllm|python3|python)($|\s)"' > /etc/default/earlyoom
  systemctl enable --now earlyoom 2>/dev/null || true
  echo "   earlyoom: $(systemctl is-active earlyoom 2>/dev/null || echo unavailable)"
else
  echo "   earlyoom not installable (offline?) — skipping; drop_caches + swapoff still protect"
fi

echo "== 4. disable desktop GUI (frees ~2-3 GB) =="
if systemctl get-default | grep -q graphical; then
  systemctl set-default multi-user.target
  echo "   default target -> multi-user (takes effect next boot; 'systemctl isolate multi-user.target' to apply now)"
else
  echo "   already headless (multi-user.target)"
fi

echo "== 5. drop caches now (do this again right before each launch) =="
sync; echo 3 > /proc/sys/vm/drop_caches
echo "   MemAvailable: $(awk '/MemAvailable/{printf "%.1f GB", $2/1024/1024}' /proc/meminfo)   Cached: $(awk '/^Cached:/{printf "%.1f GB", $2/1024/1024}' /proc/meminfo)"
echo
echo "DONE. Before EACH big launch, drop caches on every node:  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
echo "Monitor /proc/meminfo MemAvailable during load (nvidia-smi cannot read GB10 unified memory)."
