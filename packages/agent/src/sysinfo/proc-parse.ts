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

export interface PsiLine { avg10: number; avg60: number; avg300: number; total: number; }
export interface Pressure { some: PsiLine; full: PsiLine | null; }

function psiLine(text: string, kind: "some" | "full"): PsiLine | null {
  const m = text.match(new RegExp(`^${kind}\\s+avg10=([\\d.]+)\\s+avg60=([\\d.]+)\\s+avg300=([\\d.]+)\\s+total=(\\d+)`, "m"));
  if (!m) return null;
  return { avg10: parseFloat(m[1]), avg60: parseFloat(m[2]), avg300: parseFloat(m[3]), total: parseInt(m[4], 10) };
}

/** Parse a /proc/pressure/{cpu,memory,io} file. `full` is null for cpu. */
export function parsePressure(text: string): Pressure {
  return { some: psiLine(text, "some") ?? { avg10: 0, avg60: 0, avg300: 0, total: 0 }, full: psiLine(text, "full") };
}
