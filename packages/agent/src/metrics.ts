import { execSync } from "child_process";

export interface GpuMetrics {
  gpuModel: string;
  vramTotal: number;
  gpuUtil: number;
  vramUsed: number;
  temperature: number | null;
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
    };
  } catch {
    return {
      gpuModel: "Unknown (nvidia-smi unavailable)",
      vramTotal: 0,
      gpuUtil: 0,
      vramUsed: 0,
      temperature: null,
    };
  }
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
