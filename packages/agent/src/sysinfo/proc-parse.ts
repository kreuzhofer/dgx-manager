export interface MemoryInfo {
  totalMb: number; availableMb: number; freeMb: number;
  cachedMb: number; swapTotalMb: number; swapFreeMb: number;
}

/** Parse /proc/meminfo (kB values) into MB. Missing keys default to 0. */
export function parseMeminfo(text: string): MemoryInfo {
  const kb = (key: string): number => {
    const m = text.match(new RegExp(`^${key}:\\s+(\\d+)\\s+kB`, "m"));
    return m ? Math.round(parseInt(m[1], 10) / 1024) : 0;
  };
  return {
    totalMb: kb("MemTotal"), availableMb: kb("MemAvailable"), freeMb: kb("MemFree"),
    cachedMb: kb("Cached"), swapTotalMb: kb("SwapTotal"), swapFreeMb: kb("SwapFree"),
  };
}
