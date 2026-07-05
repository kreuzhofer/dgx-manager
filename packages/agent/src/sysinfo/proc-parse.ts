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

export interface LoadInfo { load1: number; load5: number; load15: number; runnable: number; totalProcs: number; }
export function parseLoadavg(text: string): LoadInfo {
  const p = text.trim().split(/\s+/);
  const [run, total] = (p[3] ?? "0/0").split("/");
  return { load1: parseFloat(p[0]) || 0, load5: parseFloat(p[1]) || 0, load15: parseFloat(p[2]) || 0,
           runnable: parseInt(run, 10) || 0, totalProcs: parseInt(total, 10) || 0 };
}

export interface FdInfo { allocated: number; max: number; }
export function parseFileNr(text: string): FdInfo {
  const p = text.trim().split(/\s+/);
  return { allocated: parseInt(p[0], 10) || 0, max: parseInt(p[2], 10) || 0 };
}

export const TCP_STATES: Record<string, string> = {
  "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV", "04": "FIN_WAIT1",
  "05": "FIN_WAIT2", "06": "TIME_WAIT", "07": "CLOSE", "08": "CLOSE_WAIT",
  "09": "LAST_ACK", "0A": "LISTEN", "0B": "CLOSING",
};

/** Count rows on `port` (local) in /proc/net/tcp[6] output, tallied by TCP state name. */
export function parseProcNetTcpByPort(text: string, port: number): Record<string, number> {
  const out: Record<string, number> = {};
  const hexPort = port.toString(16).toUpperCase().padStart(4, "0");
  for (const line of text.split("\n")) {
    const cols = line.trim().split(/\s+/);
    if (cols.length < 4 || !cols[1]?.includes(":")) continue;
    const lp = cols[1].split(":")[1]?.toUpperCase();
    if (lp !== hexPort) continue;
    const st = TCP_STATES[cols[3].toUpperCase()] ?? cols[3];
    out[st] = (out[st] ?? 0) + 1;
  }
  return out;
}
