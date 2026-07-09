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
# Reserve 1 GiB. A 5 GiB reserve backfired: the kernel refuses to allocate below
# min_free, so torch.compile's 5.00 GiB CUDA-graph capture allocation failed on
# every rank with ~9.7 GiB "free" (usable was only 4.7 GiB). Page cache is kept
# out of the unified pool by the agent's drop-cache loop (every 500 ms during a
# dgxrun deploy), which is the correct tool for that job.
vm.min_free_kbytes = 1048576
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

echo "== 3. earlyoom — DISABLE it (incompatible with memory-maxed DCP deploys) =="
# A gmu-0.90 + forced-KV deploy INTENTIONALLY runs with <2 GB free out of the 124 GB unified pool.
# earlyoom at ANY useful threshold (-m 2 = 2.48 GB) reads that normal state as near-OOM and SIGKILLs
# vllm -> kills the deploy (observed 2026-07-07: earlyoom "SIGKILL to vllm badness 972"). And the
# packaged service ran DEFAULTS (-m 10, no --prefer) -> killed NetworkManager/systemd/netplan -> the
# "Broken pipe / Marlin hang" deploy deaths. There is NO threshold that fits: the deploy's normal free
# memory sits below any real OOM trigger. So disable earlyoom. Safety without it: swapoff (step 2)
# makes an OOM a recoverable kill instead of a freeze, and the kernel OOM-killer fires ONLY at true
# 0-free OOM and kills the biggest hog (vllm). See memory earlyoom-killed-glm52-deploys.
if systemctl list-unit-files earlyoom.service >/dev/null 2>&1; then
  systemctl disable --now earlyoom 2>/dev/null || true
  rm -f /etc/systemd/system/earlyoom.service.d/override.conf 2>/dev/null || true
  systemctl daemon-reload 2>/dev/null || true
  echo "   earlyoom disabled ($(systemctl is-active earlyoom 2>/dev/null || echo inactive)); swapoff + kernel OOM-killer are the safety net"
else
  echo "   earlyoom not present — nothing to disable (swapoff + kernel OOM-killer protect)"
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
