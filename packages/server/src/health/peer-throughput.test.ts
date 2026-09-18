/**
 * Property + unit tests for the pure peer-throughput comparison (#88).
 *
 * Follows the `admission/vram.test.ts` pattern: the decision function is pure
 * so this file needs no DB and no mocks. The Prisma-coupled orchestrator that
 * loads the window gets its own integration test.
 */
import { describe, it, expect } from "vitest";
import { test, fc } from "@fast-check/vitest";
import {
  evaluatePool,
  median,
  MIN_SAMPLES,
  SUSPECT_RATIO,
  type MemberInput,
} from "./peer-throughput.js";

/** A pool member with sane defaults; override only what the case is about. */
function member(over: Partial<MemberInput> & { nodeName: string; rates: number[] }): MemberInput {
  return {
    deploymentId: `dep-${over.nodeName}`,
    nodeId: `node-${over.nodeName}`,
    modelId: "model-a",
    deploymentsOnNode: 1,
    ...over,
  };
}

/** `n` identical samples — a member holding a steady rate for the window. */
const steady = (rate: number, n = 120): number[] => Array.from({ length: n }, () => rate);

describe("evaluatePool", () => {
  it("flags the member sustaining roughly half its peers' rate", () => {
    const verdicts = evaluatePool([
      member({ nodeName: "spark-02", rates: steady(23) }),
      member({ nodeName: "spark-03", rates: steady(55) }),
      member({ nodeName: "spark-04", rates: steady(56) }),
    ]);

    const slow = verdicts.find((v) => v.node === "spark-02")!;
    expect(slow.state).toBe("suspect");
    expect(slow.ratio!).toBeLessThan(0.8);
  });

  it("reports a single-member pool as not-comparable, not healthy", () => {
    const [only] = evaluatePool([member({ nodeName: "spark-01", rates: steady(30) })]);

    expect(only.state).toBe("not-comparable");
    expect(only.reason).toBe("single-member");
  });

  // 2026-09-15T05 in the real series: spark-03 had one sample, spark-04 two.
  // Their ratio was 0.647 — the worst in 34 hours of paired load, and pure
  // sampling noise. A thin window is a non-answer, never a verdict.
  it("reports a member with too few samples as not-comparable", () => {
    const verdicts = evaluatePool([
      member({ nodeName: "spark-03", rates: [4.4] }),
      member({ nodeName: "spark-04", rates: [6.8, 6.8] }),
    ]);

    expect(verdicts.map((v) => v.state)).toEqual(["not-comparable", "not-comparable"]);
    expect(verdicts[0].reason).toBe("insufficient-samples");
  });

  // A member that served nothing reports no rate at all (the agent collapses a
  // zero rate to null). That is a steady state an operator can ignore, not a
  // "check back later" — so it gets its own reason.
  it("reports a member that served nothing in the window as idle", () => {
    const verdicts = evaluatePool([
      member({ nodeName: "spark-03", rates: [] }),
      member({ nodeName: "spark-04", rates: steady(56) }),
    ]);

    const idle = verdicts.find((v) => v.node === "spark-03")!;
    expect(idle.state).toBe("not-comparable");
    expect(idle.reason).toBe("idle");
    expect(idle.rate).toBeNull();
  });

  // The persisted rate is per NODE and is the sum over every deployment running
  // there. With two deployments on one node the number cannot be attributed to
  // this member — so it gets no verdict, and it must not inflate anyone else's
  // peer median either.
  it("reports a member sharing its node with another deployment as not-comparable", () => {
    const verdicts = evaluatePool([
      member({ nodeName: "spark-02", rates: steady(100), deploymentsOnNode: 2 }),
      member({ nodeName: "spark-03", rates: steady(55) }),
      member({ nodeName: "spark-04", rates: steady(56) }),
    ]);

    const shared = verdicts.find((v) => v.node === "spark-02")!;
    expect(shared.state).toBe("not-comparable");
    expect(shared.reason).toBe("multi-deployment-node");

    const peer = verdicts.find((v) => v.node === "spark-03")!;
    expect(peer.peerRates.map((p) => p.node)).toEqual(["spark-04"]);
    expect(peer.state).toBe("ok");
  });

  // A pool is a name collision, not a guarantee (#90): two deployments can
  // publish one name while serving different models, and their rates are then
  // measuring different things. Refuse the whole pool rather than guess which
  // half is the reference.
  it("reports every member as not-comparable when the pool mixes models", () => {
    const verdicts = evaluatePool([
      member({ nodeName: "spark-03", rates: steady(23), modelId: "glm-5.3-flash" }),
      member({ nodeName: "spark-04", rates: steady(56), modelId: "qwen3.8-27b" }),
    ]);

    expect(verdicts.map((v) => v.state)).toEqual(["not-comparable", "not-comparable"]);
    expect(verdicts.map((v) => v.reason)).toEqual(["model-mismatch", "model-mismatch"]);
  });

  // Found by running this against the live cluster: spark-01 is the only
  // member of its pool AND was idle, and reported "idle" — which reads as
  // "come back when it is serving". It could not be compared either way, so
  // the structural reason has to win over the transient one.
  it("prefers the structural reason when a lone member is also idle", () => {
    const [only] = evaluatePool([member({ nodeName: "spark-01", rates: [] })]);

    expect(only.state).toBe("not-comparable");
    expect(only.reason).toBe("single-member");
  });

  // Distinct from a pool of one: there ARE peers, they just cannot be used
  // this window. Saying "this pool has one member" here would be a lie.
  it("distinguishes a pool whose peers are all unusable from a pool of one", () => {
    const [subject] = evaluatePool([
      member({ nodeName: "spark-03", rates: steady(55) }),
      member({ nodeName: "spark-04", rates: [] }),
    ]);

    expect(subject.state).toBe("not-comparable");
    expect(subject.reason).toBe("no-comparable-peers");
  });

  // ── Calibration guard ──────────────────────────────────────────────────
  //
  // Measured 2026-09-18 over the live two-member `qwen3.8-27b-nvfp4` pool:
  // 34 hours of paired load, hourly median rate per member. Both members were
  // healthy throughout. These are the real numbers, and the threshold was
  // chosen from them — so this test does not drive behaviour, it pins it.
  // If someone raises SUSPECT_RATIO above the healthy floor, this fails.
  const HEALTHY_HOURS: Array<[number, number]> = [
    [24.9, 23.6], [25.5, 25.2], [32.8, 35.0], [32.9, 34.7], [32.8, 34.7],
    [33.0, 34.4], [35.5, 36.4], [34.8, 36.2], [50.3, 49.9], [34.2, 32.4],
    [52.0, 56.0], [55.0, 56.7], [55.4, 56.5], [53.3, 57.1], [55.0, 56.3],
    [54.1, 56.3], [55.0, 55.4], [55.7, 54.7], [30.1, 31.3], [29.5, 29.9],
    [29.5, 30.9], [42.4, 44.3], [54.6, 53.8], [48.8, 56.5], [52.0, 53.7],
    [50.7, 55.0], [51.8, 55.2], [54.4, 53.8], [30.3, 32.2], [28.5, 29.6],
    [28.4, 29.5], [27.6, 29.2], [28.1, 28.6],
  ];

  it("flags nothing across 34 hours of measured healthy load", () => {
    for (const [r03, r04] of HEALTHY_HOURS) {
      const verdicts = evaluatePool([
        member({ nodeName: "spark-03", rates: steady(r03) }),
        member({ nodeName: "spark-04", rates: steady(r04) }),
      ]);
      expect(verdicts.map((v) => v.state), `hour 03=${r03} 04=${r04}`).toEqual(["ok", "ok"]);
    }
  });

  it("leaves headroom under the worst healthy hour on record", () => {
    // 48.8 vs 56.5 — the tightest of the 34, at full sampling on both members.
    const [slower] = evaluatePool([
      member({ nodeName: "spark-03", rates: steady(48.8) }),
      member({ nodeName: "spark-04", rates: steady(56.5) }),
    ]);

    expect(slower.ratio!).toBeCloseTo(0.8637, 3);
    expect(slower.ratio!).toBeGreaterThan(SUSPECT_RATIO);
  });

  // ── Invariants (property tests) ────────────────────────────────────────
  describe("invariants (property tests)", () => {
    const fullWindow = fc.array(fc.double({ min: 1, max: 300, noNaN: true }), {
      minLength: MIN_SAMPLES,
      maxLength: MIN_SAMPLES + 4,
    });

    /**
     * A window thinner than MIN_SAMPLES is never enough to judge a member on,
     * whatever its peers look like. The answer is "cannot compare" — never
     * "healthy", and never "degraded".
     */
    test.prop([
      fc.array(fc.double({ min: 1, max: 300, noNaN: true }), { minLength: 1, maxLength: MIN_SAMPLES - 1 }),
      fullWindow,
    ])("a thin window never yields a verdict", (thin, full) => {
      const [thinVerdict] = evaluatePool([
        member({ nodeName: "thin", rates: thin }),
        member({ nodeName: "full", rates: full }),
      ]);

      expect(thinVerdict.state).toBe("not-comparable");
      expect(thinVerdict.reason).toBe("insufficient-samples");
    });

    /**
     * Members all holding the same rate are by definition not diverging, so a
     * uniform pool of any size must flag nobody. This is the false-alarm floor.
     */
    test.prop([fc.double({ min: 0.5, max: 300, noNaN: true }), fc.integer({ min: 2, max: 5 })])(
      "a pool whose members all hold the same rate flags nobody",
      (rate, size) => {
        const verdicts = evaluatePool(
          Array.from({ length: size }, (_, i) => member({ nodeName: `n${i}`, rates: steady(rate) })),
        );

        expect(verdicts.every((v) => v.state === "ok")).toBe(true);
      },
    );

    /**
     * Degradation is monotone: with its peers held fixed, a member that is
     * already suspect at some rate must still be suspect at any lower one.
     * A detector that could un-flag a node by slowing it further is broken.
     */
    test.prop([
      fc.double({ min: 1, max: 300, noNaN: true }),
      fc.double({ min: 0.01, max: 1, noNaN: true }),
      fc.double({ min: 0.01, max: 1, noNaN: true }),
    ])("slowing a suspect member further keeps it suspect", (peerRate, f1, f2) => {
      const [hi, lo] = f1 >= f2 ? [f1, f2] : [f2, f1];
      const stateAt = (f: number) =>
        evaluatePool([
          member({ nodeName: "subject", rates: steady(peerRate * f) }),
          member({ nodeName: "peer-a", rates: steady(peerRate) }),
          member({ nodeName: "peer-b", rates: steady(peerRate) }),
        ])[0].state;

      if (stateAt(hi) === "suspect") expect(stateAt(lo)).toBe("suspect");
    });

    /**
     * Every verdict shows its work: the reported ratio is exactly the member's
     * own rate over the median of the peer rates it reports. An operator must
     * be able to recompute the judgement from the numbers beside it.
     */
    test.prop([fullWindow, fullWindow, fullWindow])(
      "a verdict's ratio is reproducible from the rates it reports",
      (a, b, c) => {
        const verdicts = evaluatePool([
          member({ nodeName: "a", rates: a }),
          member({ nodeName: "b", rates: b }),
          member({ nodeName: "c", rates: c }),
        ]);

        for (const v of verdicts) {
          if (v.state === "not-comparable") continue;
          expect(v.ratio!).toBeCloseTo(v.rate! / median(v.peerRates.map((p) => p.rate))!, 6);
        }
      },
    );
  });
});