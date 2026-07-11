import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { planRead } from "./log-slice.js";

describe("planRead", () => {
  it("reads the new tail", () => {
    expect(planRead(10, 25)).toEqual({ from: 10, to: 25, truncated: false });
  });

  it("reads nothing when the file has not grown", () => {
    expect(planRead(25, 25)).toEqual({ from: 25, to: 25, truncated: false });
  });

  // Log rotated or the job dir was recreated: start over rather than read garbage.
  it("restarts from zero when the file shrank", () => {
    expect(planRead(100, 20)).toEqual({ from: 0, to: 20, truncated: true });
  });

  it("clamps a negative stored offset", () => {
    expect(planRead(-5, 10)).toEqual({ from: 0, to: 10, truncated: true });
  });

  /** Invariant: the read window is always valid — 0 <= from <= to <= size. */
  test.prop([fc.integer({ min: -50, max: 500 }), fc.nat({ max: 500 })])(
    "always yields a valid window",
    (prev, size) => {
      const r = planRead(prev, size);
      expect(r.from).toBeGreaterThanOrEqual(0);
      expect(r.to).toBe(size);
      expect(r.from).toBeLessThanOrEqual(r.to);
    },
  );

  /**
   * Invariant: successive reads of a growing file reproduce it exactly — no lost
   * bytes, no duplicates. This is what makes reattach-after-manager-restart safe.
   */
  test.prop([fc.array(fc.string({ minLength: 1, maxLength: 8 }), { minLength: 1, maxLength: 20 })])(
    "concatenating successive reads reproduces the log",
    (chunks) => {
      let file = "";
      let offset = 0;
      let seen = "";
      for (const c of chunks) {
        file += c;
        const { from, to } = planRead(offset, file.length);
        seen += file.slice(from, to);
        offset = to;
      }
      expect(seen).toBe(file);
    },
  );
});
