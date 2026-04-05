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
    return {
      gpuModel: parts[0] || "Unknown GPU",
      vramTotal: parseInt(parts[1]) || 0,
      gpuUtil: parseFloat(parts[2]) || 0,
      vramUsed: parseInt(parts[3]) || 0,
      temperature: parts[4] ? parseFloat(parts[4]) : null,
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
