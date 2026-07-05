import { readFileSync, readdirSync } from "fs";
import {
  parseMeminfo, parsePressure, parseLoadavg, parseFileNr, parseProcNetTcpByPort,
  type MemoryInfo, type Pressure, type LoadInfo, type FdInfo,
} from "./proc-parse.js";

export interface SysReadings {
  memory: MemoryInfo;
  pressure: { cpu: Pressure; memory: Pressure; io: Pressure };
  load: LoadInfo; fds: FdInfo; sshd: Record<string, number>; thermalsC: number[];
}
const safe = (read: (p: string) => string, p: string): string => {
  try { return read(p); } catch { return ""; }
};
export function readSysInfo(readText: (p: string) => string = (p) => readFileSync(p, "utf-8")): SysReadings {
  const tcp = parseProcNetTcpByPort(safe(readText, "/proc/net/tcp"), 22);
  const tcp6 = parseProcNetTcpByPort(safe(readText, "/proc/net/tcp6"), 22);
  const sshd: Record<string, number> = { ...tcp };
  for (const [k, v] of Object.entries(tcp6)) sshd[k] = (sshd[k] ?? 0) + v;
  let thermalsC: number[] = [];
  try {
    thermalsC = readdirSync("/sys/class/thermal").filter((d) => d.startsWith("thermal_zone"))
      .map((d) => Math.round((parseInt(safe(readText, `/sys/class/thermal/${d}/temp`), 10) || 0) / 1000))
      .filter((n) => n > 0);
  } catch { /* no thermal zones */ }
  return {
    memory: parseMeminfo(safe(readText, "/proc/meminfo")),
    pressure: {
      cpu: parsePressure(safe(readText, "/proc/pressure/cpu")),
      memory: parsePressure(safe(readText, "/proc/pressure/memory")),
      io: parsePressure(safe(readText, "/proc/pressure/io")),
    },
    load: parseLoadavg(safe(readText, "/proc/loadavg")),
    fds: parseFileNr(safe(readText, "/proc/sys/fs/file-nr")),
    sshd, thermalsC,
  };
}
