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
  fineTuneHoldingStatus,
  reclaimForRestart,
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

    test.prop([snapshotArb, utilArb, fc.integer({ min: 0, max: 300_000 })])(
      "reclaimableMB can only relax admission — never turns an admit into a reject",
      (rawSnapshot, util, reclaim) => {
        const snapshot: NodeSnapshot = {
          ...rawSnapshot,
          vramUsedMB: Math.min(rawSnapshot.vramUsedMB, rawSnapshot.vramTotalMB),
        };
        const base = computeVramShortfall(snapshot, util);
        const withReclaim = computeVramShortfall(
          { ...snapshot, reclaimableMB: reclaim },
          util,
        );
        // Subtracting reclaimable VRAM can only free space, so any node that
        // admits with no reclaim must still admit with a non-negative reclaim.
        if (base === null) expect(withReclaim).toBeNull();
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
            kind: "deployment",
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

    it("subtracts reclaimableMB (the restarting deployment's own resident VRAM) before the check", () => {
      // Today's incident (kreuzhofer/dgx-manager#1): spark-02 measured 97069 MB
      // used at a 0.90 restart — but ~95 GB of that IS the deployment being
      // restarted, which is torn down before relaunch. Counting it yields a
      // spurious shortfall.
      const base: NodeSnapshot = {
        nodeId: "spark-02",
        nodeName: "dgx-spark-02",
        vramTotalMB: 124_546,
        vramUsedMB: 97_069,
        conflicts: [],
      };
      // 0.90 util → requested 112091 + margin 6227 = threshold 118318;
      // raw available 27477 → REJECT without reclaim.
      expect(computeVramShortfall(base, 0.9)).not.toBeNull();
      // Marking the restarting deployment's ~95 GB reclaimable → effective used
      // ~2 GB, available ~122 GB ≥ 118318 → ADMIT.
      const admitted = computeVramShortfall({ ...base, reclaimableMB: 95_000 }, 0.9);
      expect(admitted).toBeNull();
    });

    it("clamps reclaimableMB so garbage values can't push used below zero or over-relax", () => {
      const base: NodeSnapshot = {
        nodeId: "n",
        nodeName: "n",
        vramTotalMB: 100_000,
        vramUsedMB: 10_000,
        conflicts: [],
      };
      // Reclaim larger than used → effective used 0, available = total → admit.
      expect(computeVramShortfall({ ...base, reclaimableMB: 999_999 }, 0.5)).toBeNull();
      // Negative reclaim is treated as 0: identical to no reclaim at all.
      expect(computeVramShortfall({ ...base, reclaimableMB: -50_000 }, 0.9)).toEqual(
        computeVramShortfall(base, 0.9),
      );
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
          { id: "ollama", name: "qwen3-embedding:8b", status: "running", kind: "deployment", vramActualMB: null, vramEstimateMB: null },
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

describe("reclaimForRestart", () => {
  describe("invariants (property tests)", () => {
    /**
     * A restart is never credited with more than the share of the node it was
     * authorised to request. Unattributed memory — a training run, a
     * hand-started container, a survivor of an offboarded node — is therefore
     * counted against the restart, not handed to it. This is the invariant
     * kreuzhofer/dgx-manager#118 violated: reclaim was defined as everything
     * the manager could not explain, so an unexplained node read as empty.
     */
    test.prop([
      fc.integer({ min: 0, max: 300_000 }),
      fc.integer({ min: 16_000, max: 256_000 }),
      utilArb,
    ])(
      "never exceeds the authorised share of the node",
      (nodeReadingMB, vramTotalMB, util) => {
        const authorised = Math.round(vramTotalMB * util);
        expect(reclaimForRestart(nodeReadingMB, vramTotalMB, util))
          .toBeLessThanOrEqual(authorised);
      },
    );

    /**
     * Reclaim is also never more than the node is actually holding — a
     * deployment authorised for 100 GB on a node reading 4 GB releases 4 GB.
     * Together with the bound above this keeps effective used memory
     * non-negative without the pure shortfall function having to clamp.
     */
    test.prop([
      fc.integer({ min: 0, max: 300_000 }),
      fc.integer({ min: 16_000, max: 256_000 }),
      utilArb,
    ])(
      "never exceeds the node reading, and is never negative",
      (nodeReadingMB, vramTotalMB, util) => {
        const reclaim = reclaimForRestart(nodeReadingMB, vramTotalMB, util);
        expect(reclaim).toBeLessThanOrEqual(nodeReadingMB);
        expect(reclaim).toBeGreaterThanOrEqual(0);
      },
    );

    /**
     * The brief's invariant for #118, stated end-to-end over the composition
     * the orchestrator performs: for any node reading, the used memory the
     * admission check sees after reclaim is never lower than the node reading
     * minus the restarting deployment's authorised share. A node holding 60 GB
     * of someone else's work can never compute as emptier than 60 GB minus
     * what this deployment was allowed to ask for.
     */
    test.prop([
      fc.integer({ min: 0, max: 256_000 }),
      fc.integer({ min: 16_000, max: 256_000 }),
      utilArb,
      utilArb,
    ])(
      "used-after-reclaim is never below the node reading minus the authorised share",
      (nodeReadingMB, vramTotalMB, savedUtil, requestedUtil) => {
        const authorised = Math.round(vramTotalMB * savedUtil);
        const snapshot: NodeSnapshot = {
          nodeId: "n",
          nodeName: null,
          vramTotalMB,
          vramUsedMB: nodeReadingMB,
          reclaimableMB: reclaimForRestart(nodeReadingMB, vramTotalMB, savedUtil),
          conflicts: [],
        };
        const result = computeVramShortfall(snapshot, requestedUtil);
        // Only a refusal reports used memory; when the node admits there is
        // nothing to check (and admitting is the permissive direction anyway).
        if (result) {
          expect(result.vramUsedMB).toBeGreaterThanOrEqual(nodeReadingMB - authorised);
        }
      },
    );

    /**
     * Unattributed memory counts against the restart. Adding memory to a node
     * can never make it look emptier to admission — the property that fails
     * under #118's rule, where every extra unexplained byte became an extra
     * byte of credit.
     */
    test.prop([
      fc.integer({ min: 0, max: 128_000 }),
      fc.integer({ min: 0, max: 128_000 }),
      fc.integer({ min: 16_000, max: 256_000 }),
      utilArb,
    ])(
      "extra memory on the node never lowers the used figure admission sees",
      (nodeReadingMB, extraMB, vramTotalMB, savedUtil) => {
        const used = (reading: number) =>
          Math.max(0, reading - reclaimForRestart(reading, vramTotalMB, savedUtil));
        expect(used(nodeReadingMB + extraMB)).toBeGreaterThanOrEqual(used(nodeReadingMB));
      },
    );
  });

  describe("hand-picked cases", () => {
    it("credits a restart only up to its authorised share of a node held by a fine-tune job", () => {
      // #118: spark-03 reading 60 GB, all of it a training container. The
      // deployment being restarted was authorised for 0.85 of 122502 MB.
      // Old rule: reclaim = 60000 (nothing else has a row) → node reads empty.
      // New rule: reclaim = min(104127, 60000) = 60000 — still the whole
      // reading, because the share is larger than it. The bound bites when the
      // share is the smaller number:
      expect(reclaimForRestart(60_000, 122_502, 0.85)).toBe(60_000);
      expect(reclaimForRestart(60_000, 122_502, 0.3)).toBe(36_751);
    });

    it("reclaims nothing for a non-finite share, refusing with readable numbers", () => {
      // A corrupt config blob, or a non-numeric `gpuMem` in a request body.
      // Reclaiming 0 charges the restart for the whole reading — it is refused,
      // which is the safe direction — and keeps NaN out of every figure the
      // refusal reports.
      expect(reclaimForRestart(60_000, 100_000, NaN)).toBe(0);
      expect(reclaimForRestart(60_000, 100_000, Infinity)).toBe(0);
      const snapshot: NodeSnapshot = {
        nodeId: "n",
        nodeName: "n",
        vramTotalMB: 100_000,
        vramUsedMB: 60_000,
        reclaimableMB: reclaimForRestart(60_000, 100_000, NaN),
        conflicts: [],
      };
      const result = computeVramShortfall(snapshot, 0.9);
      expect(result).not.toBeNull();
      expect(result!.vramUsedMB).toBe(60_000);
      expect(Number.isFinite(result!.vramAvailableMB)).toBe(true);
    });

    it("treats a nonsensical utilisation as no share at all rather than a licence", () => {
      // A negative or >1 share can only arrive from a corrupt config blob.
      // Clamping to [0,1] keeps a garbage value from crediting the restart
      // with more than a whole node.
      expect(reclaimForRestart(60_000, 100_000, -1)).toBe(0);
      expect(reclaimForRestart(60_000, 100_000, 5)).toBe(60_000);
      expect(reclaimForRestart(150_000, 100_000, 5)).toBe(100_000);
    });
  });
});

describe("fineTuneHoldingStatus", () => {
  /**
   * A job holds a node's memory across three independent columns, and naming
   * the activity matters as much as detecting it: a user told "merging" knows
   * the run has finished training and the merge can be retried, where "held by
   * a fine-tune" alone tells them nothing about what they would be losing.
   */
  const job = (
    status: string,
    mergeStatus: string | null = null,
    quantizationStatus: string | null = null,
  ) => ({ status, mergeStatus, quantizationStatus });

  it.each([
    ["queued", job("pending"), "pending"],
    ["starting", job("starting"), "starting"],
    ["training", job("running"), "running"],
    ["stopping", job("stopping"), "stopping"],
    ["merging after training completed", job("completed", "running"), "merging"],
    ["quantizing after merge", job("completed", "completed", "quantizing"), "quantizing"],
  ])("reports a job that is %s as holding the node, labelled %s", (_label, columns, expected) => {
    expect(fineTuneHoldingStatus(columns)).toBe(expected);
  });

  it.each([
    ["finished cleanly", job("completed")],
    ["failed", job("failed")],
    ["merged and quantized", job("completed", "completed", "quantized")],
    ["a merge that failed", job("failed", "failed")],
  ])("reports a job that is %s as holding nothing", (_label, columns) => {
    expect(fineTuneHoldingStatus(columns)).toBeNull();
  });

  it("names the largest allocation when several columns are in flight at once", () => {
    // Merging loads the base model — the biggest claim a job makes — so it wins
    // over both the job's own status and a concurrent quantization.
    expect(
      fineTuneHoldingStatus({
        status: "running",
        mergeStatus: "running",
        quantizationStatus: "quantizing",
      }),
    ).toBe("merging");
  });
});

describe("vramShortfallMessage — naming the holders", () => {
  const shortfallWith = (conflicts: NodeSnapshot["conflicts"]) => ({
    ...makeShortfall("dgx-spark-03"),
    conflicts,
  });

  it("marks a fine-tune job as one, so it is not mistaken for a restartable deployment", () => {
    const msg = vramShortfallMessage([
      shortfallWith([
        { id: "dep1", name: "gpt-oss-120b", status: "running", kind: "deployment", vramActualMB: 95_000, vramEstimateMB: null },
        { id: "job1", name: "sql-lora-27b", status: "merging", kind: "finetune", vramActualMB: null, vramEstimateMB: null },
      ]),
    ]);
    expect(msg).toContain("gpt-oss-120b (running)");
    expect(msg).toContain("sql-lora-27b (merging, fine-tune job)");
  });

  it("points at the gpuMem lever, the only way through a refusal the user disagrees with", () => {
    // Admission never evicts and there is deliberately no override flag
    // (ADR 0004, Decision 5), so the message has to name the alternative.
    const msg = vramShortfallMessage([shortfallWith([])]);
    expect(msg).toContain("config.gpuMem");
  });

  it("renders nothing at all when there are no shortfalls", () => {
    expect(vramShortfallMessage([])).toBe("");
  });
});
