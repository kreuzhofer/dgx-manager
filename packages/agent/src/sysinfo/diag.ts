import { readdirSync } from "fs";
import { execSync } from "child_process";
import { readSysInfo, type SysReadings } from "./proc-read.js";

export interface Diag extends SysReadings {
  pidCount: number;
  kmsgTail: string[];
  gpu: string | null;
}

interface DiagDeps {
  readText?: (p: string) => string;
  pidCount?: () => number;
  kmsgTail?: () => string[];
  gpu?: () => string | null;
}

const defaultPidCount = (): number => {
  try {
    return readdirSync("/proc").filter((d) => /^\d+$/.test(d)).length;
  } catch {
    return 0;
  }
};

const defaultGpu = (): string | null => {
  try {
    return execSync(
      "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader",
      { timeout: 4000, encoding: "utf-8" }
    ).trim();
  } catch {
    return null;
  }
};

export function collectDiag(deps: DiagDeps = {}): Diag {
  const sys = readSysInfo(deps.readText);
  let gpu: string | null = null;
  try {
    gpu = (deps.gpu ?? defaultGpu)();
  } catch {
    gpu = null;
  }
  return {
    ...sys,
    pidCount: (deps.pidCount ?? defaultPidCount)(),
    kmsgTail: (deps.kmsgTail ?? (() => []))(),
    gpu,
  };
}
