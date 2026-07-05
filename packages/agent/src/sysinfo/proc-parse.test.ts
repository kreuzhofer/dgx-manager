import { describe, it, expect } from "vitest";
import { parseMeminfo } from "./proc-parse.js";

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
