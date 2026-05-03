/**
 * Property + unit tests for the pure VRAM admission decision function.
 *
 * Pattern this file establishes for the repo:
 *   - Pure helpers should be testable WITHOUT the DB. We pulled
 *     `computeVramShortfall` out of the route file specifically so this
 *     file doesn't have to mock Prisma. The DB-coupled orchestrator gets
 *     its own integration test.
 *   - Property tests use `@fast-check/vitest`'s `it.prop` and assert
 *     invariants that should hold across the whole input space, not just
 *     for hand-picked fixtures.
 *   - Each invariant gets a plain-English doc comment so the test reads
 *     like a spec.
 */
import { describe, it, expect } from "vitest";
import { test, fc } from "@fast-check/vitest";
import {
  computeVramShortfall,
  vramShortfallMessage,
  SAFETY_MARGIN_FRACTION,
  type NodeSnapshot,
} from "./vram.js";

const emptyConflicts: NodeSnapshot["conflicts"] = [];

const snapshotArb = fc.record({
  nodeId: fc.string({ minLength: 1, maxLength: 30 }),
  nodeName: fc.option(fc.string({ minLength: 1, maxLength: 30 }), { nil: null }),
  // Realistic GB10/H100/A100-class totals: 16 GB to 256 GB in MB.
  vramTotalMB: fc.integer({ min: 16_000, max: 256_000 }),
  // Used can be anywhere from 0 to total. Generated independently and
  // clamped in the precondition so we get a wide spread.
  vramUsedMB: fc.integer({ min: 0, max: 256_000 }),
  conflicts: fc.constant(emptyConflicts),
});

const utilArb = fc.float({
  min: Math.fround(0.1),
  max: Math.fround(0.99),
  noNaN: true,
});

describe("computeVramShortfall", () => {
  describe("invariants (property tests)", () => {
    test.prop([snapshotArb, utilArb])(
      "returns null when available >= requested + safety margin",
      (rawSnapshot, util) => {
        const snapshot: NodeSnapshot = {
          ...rawSnapshot,
          vramUsedMB: Math.min(rawSnapshot.vramUsedMB, rawSnapshot.vramTotalMB),
        };
        const requested = Math.round(snapshot.vramTotalMB * util);
        const margin = Math.round(snapshot.vramTotalMB * SAFETY_MARGIN_FRACTION);
        const available = snapshot.vramTotalMB - snapshot.vramUsedMB;
        const result = computeVramShortfall(snapshot, util);
        if (available >= requested + margin) {
          expect(result).toBeNull();
        }
      },
    );

    test.prop([snapshotArb, utilArb])(
      "returns a shortfall when available < requested + safety margin",
      (rawSnapshot, util) => {
        const snapshot: NodeSnapshot = {
          ...rawSnapshot,
          vramUsedMB: Math.min(rawSnapshot.vramUsedMB, rawSnapshot.vramTotalMB),
        };
        const requested = Math.round(snapshot.vramTotalMB * util);
        const margin = Math.round(snapshot.vramTotalMB * SAFETY_MARGIN_FRACTION);
        const available = snapshot.vramTotalMB - snapshot.vramUsedMB;
        const result = computeVramShortfall(snapshot, util);
        if (available < requested + margin) {
          expect(result).not.toBeNull();
          expect(result!.vramThresholdMB).toBeGreaterThan(result!.vramAvailableMB);
        }
      },
    );

    test.prop([snapshotArb, utilArb])(
      "non-negative inputs produce non-negative output fields",
      (rawSnapshot, util) => {
        const snapshot: NodeSnapshot = {
          ...rawSnapshot,
          vramUsedMB: Math.min(rawSnapshot.vramUsedMB, rawSnapshot.vramTotalMB),
        };
        const r = computeVramShortfall(snapshot, util);
        if (r) {
          expect(r.vramTotalMB).toBeGreaterThanOrEqual(0);
          expect(r.vramUsedMB).toBeGreaterThanOrEqual(0);
          expect(r.vramAvailableMB).toBeGreaterThanOrEqual(0);
          expect(r.vramRequestedMB).toBeGreaterThanOrEqual(0);
          expect(r.vramThresholdMB).toBeGreaterThanOrEqual(0);
          expect(r.vramSafetyMarginMB).toBeGreaterThanOrEqual(0);
        }
      },
    );

    test.prop([snapshotArb, utilArb])(
      "is deterministic — same input, same output",
      (rawSnapshot, util) => {
        const snapshot: NodeSnapshot = {
          ...rawSnapshot,
          vramUsedMB: Math.min(rawSnapshot.vramUsedMB, rawSnapshot.vramTotalMB),
        };
        expect(computeVramShortfall(snapshot, util)).toEqual(
          computeVramShortfall(snapshot, util),
        );
      },
    );
  });

  describe("hand-picked cases", () => {
    it("real DGX Spark with the qwen3-embedding scenario: rejects the 397B at 0.85", () => {
      // Numbers from the actual incident: spark-03 holding ~15 GB Ollama,
      // 397B FP8 wanted 0.85 × 119.69 GB = 101.74 GB. Should refuse.
      const snapshot: NodeSnapshot = {
        nodeId: "spark-03",
        nodeName: "dgx-spark-03",
        vramTotalMB: 122_502, // 119.69 GiB
        vramUsedMB: 15_360, // 15 GB ollama
        conflicts: [
          {
            id: "ollama-dep-1",
            name: "qwen3-embedding:8b",
            status: "running",
            vramActualMB: 15_360,
            vramEstimateMB: 15_360,
          },
        ],
      };
      const result = computeVramShortfall(snapshot, 0.85);
      expect(result).not.toBeNull();
      expect(result!.vramRequestedMB).toBeGreaterThan(100_000);
      expect(result!.vramAvailableMB).toBe(122_502 - 15_360);
      expect(result!.conflicts[0].name).toBe("qwen3-embedding:8b");
    });

    it("admits a model that fits comfortably with no other usage", () => {
      const snapshot: NodeSnapshot = {
        nodeId: "n",
        nodeName: "n",
        vramTotalMB: 122_502,
        vramUsedMB: 0,
        conflicts: [],
      };
      expect(computeVramShortfall(snapshot, 0.7)).toBeNull();
    });

    it("rejects when required + safety equals exactly available + 1MB", () => {
      // Boundary: any ramp-up that crosses available - margin should fail.
      const snapshot: NodeSnapshot = {
        nodeId: "n",
        nodeName: "n",
        vramTotalMB: 100_000,
        vramUsedMB: 6_000, // 94 GB free; safety margin is 5000; threshold = req + 5000
        conflicts: [],
      };
      // util 0.9 → requested 90000 → threshold 95000 → 94000 < 95000 → reject
      expect(computeVramShortfall(snapshot, 0.9)).not.toBeNull();
      // util 0.88 → requested 88000 → threshold 93000 → 94000 >= 93000 → admit
      expect(computeVramShortfall(snapshot, 0.88)).toBeNull();
    });

    it("clamps available to >=0 even if vramUsed exceeds vramTotal", () => {
      const snapshot: NodeSnapshot = {
        nodeId: "n",
        nodeName: "n",
        vramTotalMB: 100_000,
        vramUsedMB: 110_000, // pathological: usage > total (stale metric / racy report)
        conflicts: [],
      };
      const result = computeVramShortfall(snapshot, 0.5);
      expect(result).not.toBeNull();
      expect(result!.vramAvailableMB).toBe(0);
    });
  });
});

describe("vramShortfallMessage", () => {
  it("formats one shortfall with conflict list and threshold math", () => {
    const msg = vramShortfallMessage([
      {
        nodeId: "abc",
        nodeName: "dgx-spark-03",
        vramTotalMB: 122_000,
        vramUsedMB: 15_000,
        vramAvailableMB: 107_000,
        vramRequestedMB: 103_000,
        vramThresholdMB: 109_000,
        vramSafetyMarginMB: 6_000,
        conflicts: [
          { id: "ollama", name: "qwen3-embedding:8b", status: "running", vramActualMB: null, vramEstimateMB: null },
        ],
      },
    ]);
    expect(msg).toContain("dgx-spark-03");
    expect(msg).toContain("safety margin");
    expect(msg).toContain("qwen3-embedding:8b");
    expect(msg).toContain("running");
  });

  it("joins multiple shortfalls with a semicolon", () => {
    const msg = vramShortfallMessage([
      makeShortfall("a"),
      makeShortfall("b"),
    ]);
    expect(msg.split(";").length).toBe(2);
  });

  it("falls back to short id when nodeName is null", () => {
    const msg = vramShortfallMessage([
      { ...makeShortfall("xxxxxxxxxxxx-abc"), nodeName: null },
    ]);
    expect(msg.startsWith("xxxxxxxxxxxx")).toBe(true);
  });
});

function makeShortfall(name: string) {
  return {
    nodeId: name,
    nodeName: name,
    vramTotalMB: 120_000,
    vramUsedMB: 10_000,
    vramAvailableMB: 110_000,
    vramRequestedMB: 105_000,
    vramThresholdMB: 111_000,
    vramSafetyMarginMB: 6_000,
    conflicts: [],
  };
}
