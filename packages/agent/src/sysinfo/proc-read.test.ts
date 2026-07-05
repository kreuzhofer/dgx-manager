import { describe, it, expect } from "vitest";
import { readSysInfo } from "./proc-read.js";
describe("readSysInfo", () => {
  it("assembles readings from injected /proc contents", () => {
    const files: Record<string,string> = {
      "/proc/meminfo": "MemTotal: 127535340 kB\nMemAvailable: 7100000 kB",
      "/proc/pressure/cpu": "some avg10=0 avg60=0 avg300=0 total=0",
      "/proc/pressure/memory": "some avg10=5 avg60=1 avg300=0 total=9\nfull avg10=2 avg60=0 avg300=0 total=3",
      "/proc/pressure/io": "some avg10=0 avg60=0 avg300=0 total=0",
      "/proc/loadavg": "1.0 1.0 1.0 2/500 9",
      "/proc/sys/fs/file-nr": "1000 0 900000",
      "/proc/net/tcp": "sl local rem st\n0: 00000000:0016 00000000:0000 0A 0 0",
      "/proc/net/tcp6": "",
    };
    const r = readSysInfo((p) => files[p] ?? "");
    expect(r.memory.totalMb).toBe(124546);
    expect(r.pressure.memory.some.avg10).toBe(5);
    expect(r.sshd.LISTEN).toBe(1);
    expect(r.load.load1).toBe(1.0);
  });
  it("tolerates missing files (readText throws) with zeroed structs", () => {
    const r = readSysInfo(() => { throw new Error("ENOENT"); });
    expect(r.memory.totalMb).toBe(0);
    expect(r.sshd).toEqual({});
  });
});
