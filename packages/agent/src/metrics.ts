import { execSync } from "child_process";
import { readFileSync, readdirSync } from "fs";

export interface NetInterface {
  name: string;
  rxBytesPerSec: number;
  txBytesPerSec: number;
}

export interface RdmaInterface {
  name: string;
  rxBytesPerSec: number;
  txBytesPerSec: number;
}

export interface DiskDevice {
  name: string;
  readBytesPerSec: number;
  writeBytesPerSec: number;
}

export interface GpuMetrics {
  gpuModel: string;
  vramTotal: number;
  gpuUtil: number;
  vramUsed: number;
  temperature: number | null;
  netInterfaces: NetInterface[];
  rdmaInterfaces: RdmaInterface[];
  diskDevices: DiskDevice[];
}

export async function collectMetrics(): Promise<GpuMetrics> {
  try {
    const output = execSync(
      "nvidia-smi --query-gpu=name,memory.total,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits",
      { timeout: 5000, encoding: "utf-8" }
    );

    const parts = output.trim().split(",").map((s) => s.trim());
    let vramUsed = parseInt(parts[3]) || 0;

    // GB10 shared memory: memory.used shows [N/A], fall back to per-process sum
    if (parts[3] === "[N/A]" || vramUsed === 0) {
      vramUsed = getProcessGpuMemory();
    }

    // GB10: memory.total also [N/A] — use system RAM as the shared pool
    let vramTotal = parseInt(parts[1]) || 0;
    if (parts[1] === "[N/A]" || vramTotal === 0) {
      vramTotal = getSystemMemoryMB();
    }

    return {
      gpuModel: parts[0] || "Unknown GPU",
      vramTotal,
      gpuUtil: parseFloat(parts[2]) || 0,
      vramUsed,
      temperature: parts[4] && parts[4] !== "[N/A]" ? parseFloat(parts[4]) : null,
      netInterfaces: collectNetworkMetrics(),
      rdmaInterfaces: collectRdmaMetrics(),
      diskDevices: collectDiskMetrics(),
    };
  } catch {
    return {
      gpuModel: "Unknown (nvidia-smi unavailable)",
      vramTotal: 0,
      gpuUtil: 0,
      vramUsed: 0,
      temperature: null,
      netInterfaces: collectNetworkMetrics(),
      rdmaInterfaces: collectRdmaMetrics(),
      diskDevices: collectDiskMetrics(),
    };
  }
}

// --- Network interface monitoring ---

const IGNORED_IFACE_PREFIXES = ["lo", "docker", "br-", "veth", "wl"];
const prevBytes = new Map<string, { rx: number; tx: number; time: number }>();

/** Get interfaces with IPv4 addresses, excluding virtual/loopback/wifi. */
function getActiveInterfaces(): string[] {
  try {
    const output = execSync(
      "ip -4 -o addr show | awk '{print $2}'",
      { timeout: 3000, encoding: "utf-8" }
    );
    return output
      .trim()
      .split("\n")
      .map((s) => s.trim())
      .filter((name) => name && !IGNORED_IFACE_PREFIXES.some((p) => name.startsWith(p)));
  } catch {
    return [];
  }
}

/** Read current byte counters and compute per-second rates. */
function collectNetworkMetrics(): NetInterface[] {
  const now = Date.now();
  const results: NetInterface[] = [];

  for (const iface of getActiveInterfaces()) {
    try {
      const rx = parseInt(readFileSync(`/sys/class/net/${iface}/statistics/rx_bytes`, "utf-8").trim());
      const tx = parseInt(readFileSync(`/sys/class/net/${iface}/statistics/tx_bytes`, "utf-8").trim());
      const prev = prevBytes.get(iface);

      if (prev) {
        const elapsed = (now - prev.time) / 1000;
        if (elapsed > 0) {
          results.push({
            name: iface,
            rxBytesPerSec: Math.round((rx - prev.rx) / elapsed),
            txBytesPerSec: Math.round((tx - prev.tx) / elapsed),
          });
        }
      }

      prevBytes.set(iface, { rx, tx, time: now });
    } catch {
      // Interface disappeared or unreadable
    }
  }

  return results;
}

// --- RDMA/InfiniBand monitoring ---

const prevRdmaBytes = new Map<string, { rx: number; tx: number; time: number }>();

/** Read RDMA port counters from InfiniBand sysfs. */
function collectRdmaMetrics(): RdmaInterface[] {
  const now = Date.now();
  const results: RdmaInterface[] = [];
  const ibDir = "/sys/class/infiniband";

  try {
    for (const dev of readdirSync(ibDir)) {
      const counterDir = `${ibDir}/${dev}/ports/1/counters`;
      try {
        // IB counters are in units of 4 bytes (32-bit words), multiply by 4 for bytes
        const rx = parseInt(readFileSync(`${counterDir}/port_rcv_data`, "utf-8").trim()) * 4;
        const tx = parseInt(readFileSync(`${counterDir}/port_xmit_data`, "utf-8").trim()) * 4;
        const prev = prevRdmaBytes.get(dev);

        if (prev) {
          const elapsed = (now - prev.time) / 1000;
          if (elapsed > 0) {
            results.push({
              name: dev,
              rxBytesPerSec: Math.round((rx - prev.rx) / elapsed),
              txBytesPerSec: Math.round((tx - prev.tx) / elapsed),
            });
          }
        }

        prevRdmaBytes.set(dev, { rx, tx, time: now });
      } catch { /* port not readable */ }
    }
  } catch { /* no IB devices */ }

  return results;
}

// --- Disk I/O monitoring ---

// Match whole-device names only: nvme0n1, sda, md0 — partitions like nvme0n1p1
// or sda1 are excluded, as are loop*, ram*, dm-*.
const DISK_DEVICE_RE = /^(nvme\d+n\d+|sd[a-z]+|md\d+)$/;
const prevDiskCounters = new Map<string, { sectorsRead: number; sectorsWritten: number; time: number }>();

/** Read /proc/diskstats and compute per-device read/write bytes per second. */
function collectDiskMetrics(): DiskDevice[] {
  const now = Date.now();
  const results: DiskDevice[] = [];

  let raw: string;
  try {
    raw = readFileSync("/proc/diskstats", "utf-8");
  } catch {
    return results;
  }

  for (const line of raw.split("\n")) {
    // Fields: major minor name reads_completed reads_merged sectors_read
    //         ms_reading writes_completed writes_merged sectors_written ...
    const fields = line.trim().split(/\s+/);
    if (fields.length < 10) continue;

    const name = fields[2];
    if (!DISK_DEVICE_RE.test(name)) continue;

    const sectorsRead = parseInt(fields[5], 10);
    const sectorsWritten = parseInt(fields[9], 10);
    if (Number.isNaN(sectorsRead) || Number.isNaN(sectorsWritten)) continue;

    const prev = prevDiskCounters.get(name);
    if (prev) {
      const elapsed = (now - prev.time) / 1000;
      if (elapsed > 0) {
        // /proc/diskstats sectors are 512 bytes regardless of physical block size.
        results.push({
          name,
          readBytesPerSec: Math.round(((sectorsRead - prev.sectorsRead) * 512) / elapsed),
          writeBytesPerSec: Math.round(((sectorsWritten - prev.sectorsWritten) * 512) / elapsed),
        });
      }
    }

    prevDiskCounters.set(name, { sectorsRead, sectorsWritten, time: now });
  }

  return results;
}

/** Sum GPU memory used by all compute processes (works on GB10 shared memory). */
function getProcessGpuMemory(): number {
  try {
    const output = execSync(
      "nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits",
      { timeout: 5000, encoding: "utf-8" }
    );
    return output
      .trim()
      .split("\n")
      .filter(Boolean)
      .reduce((sum, line) => sum + (parseInt(line.trim()) || 0), 0);
  } catch {
    return 0;
  }
}

/** Get total system RAM in MB (for GB10 shared memory architecture). */
function getSystemMemoryMB(): number {
  try {
    const output = execSync("free -m", { timeout: 3000, encoding: "utf-8" });
    const match = output.match(/Mem:\s+(\d+)/);
    return match ? parseInt(match[1]) : 0;
  } catch {
    return 0;
  }
}
