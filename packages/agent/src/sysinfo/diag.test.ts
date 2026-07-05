import { describe, it, expect } from "vitest";
import { collectDiag } from "./diag.js";

describe("collectDiag", () => {
  it("returns a bundle even when GPU probe fails", () => {
    const d = collectDiag({
      readText: (p) => (p === "/proc/meminfo" ? "MemTotal: 1048576 kB" : ""),
      pidCount: () => 512,
      kmsgTail: () => ["oom-killer invoked"],
      gpu: () => {
        throw new Error("fork failed");
      },
    });
    expect(d.memory.totalMb).toBe(1024);
    expect(d.pidCount).toBe(512);
    expect(d.gpu).toBeNull();
    expect(d.kmsgTail[0]).toContain("oom");
  });
});
