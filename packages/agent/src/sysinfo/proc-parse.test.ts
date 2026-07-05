import { describe, it, expect } from "vitest";
import { parseMeminfo, parsePressure, parseLoadavg, parseFileNr } from "./proc-parse.js";

describe("parseMeminfo", () => {
  it("converts kB fields to MB", () => {
    const text = [
      "MemTotal:       127535340 kB",
      "MemFree:          869000 kB",
      "MemAvailable:    7100000 kB",
      "Buffers:          100000 kB",
      "Cached:          9900000 kB",
      "SwapTotal:             0 kB",
      "SwapFree:              0 kB",
    ].join("\n");
    const m = parseMeminfo(text);
    expect(m.totalMb).toBe(124546);
    expect(m.availableMb).toBe(6934);
    expect(m.freeMb).toBe(849);
    expect(m.cachedMb).toBe(9668);
    expect(m.swapTotalMb).toBe(0);
  });
  it("missing fields default to 0", () => {
    expect(parseMeminfo("MemTotal: 1024 kB").availableMb).toBe(0);
  });
});

describe("parsePressure", () => {
  it("parses some + full lines", () => {
    const p = parsePressure(
      "some avg10=1.23 avg60=4.56 avg300=7.89 total=12345\n" +
      "full avg10=0.10 avg60=0.20 avg300=0.30 total=678\n");
    expect(p.some.avg10).toBe(1.23);
    expect(p.some.total).toBe(12345);
    expect(p.full?.avg60).toBe(0.20);
  });
  it("cpu pressure has no full line", () => {
    expect(parsePressure("some avg10=0 avg60=0 avg300=0 total=0").full).toBeNull();
  });
});

describe("parseLoadavg", () => {
  it("parses loads + proc counts", () => {
    const l = parseLoadavg("2.15 1.80 1.44 3/1234 99999");
    expect(l.load1).toBe(2.15); expect(l.runnable).toBe(3); expect(l.totalProcs).toBe(1234);
  });
});

describe("parseFileNr", () => {
  it("parses allocated and max", () => {
    const f = parseFileNr("12800\t0\t9223372");
    expect(f.allocated).toBe(12800); expect(f.max).toBe(9223372);
  });
});
