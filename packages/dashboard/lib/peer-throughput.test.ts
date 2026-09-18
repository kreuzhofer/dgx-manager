/**
 * Unit tests for how a peer-throughput verdict (#88) is worded for an operator.
 *
 * The comparison itself is the server's; this is only the sentence the page
 * shows. It lives in lib/ so it can be tested without rendering, the same way
 * cluster-summary and node-power are.
 */
import { describe, it, expect } from "vitest";
import { describeThroughput, suspectedNodes, type ThroughputVerdict } from "./peer-throughput";

const verdict = (over: Partial<ThroughputVerdict>): ThroughputVerdict => ({
  state: "ok",
  rate: 55,
  peerRates: [{ node: "spark-04", rate: 56 }],
  ratio: 0.98,
  windowMinutes: 60,
  ...over,
});

describe("describeThroughput", () => {
  it("says nothing when the server sent no verdict", () => {
    expect(describeThroughput(null)).toBeNull();
  });

  it("names both sides of the comparison when a member is behind", () => {
    const view = describeThroughput(
      verdict({
        state: "suspect",
        rate: 23,
        peerRates: [{ node: "spark-03", rate: 55 }, { node: "spark-04", rate: 56 }],
        ratio: 23 / 55.5,
      }),
    )!;

    expect(view.tone).toBe("suspect");
    expect(view.label).toBe("behind peers");
    // The operator must be able to recompute the judgement from the sentence.
    expect(view.detail).toContain("23.0");
    expect(view.detail).toContain("55.5");
    expect(view.detail).toContain("41%");
    expect(view.detail).toContain("60 min");
  });

  it("states a healthy member's rate without shouting about it", () => {
    const view = describeThroughput(verdict({ state: "ok", rate: 55, ratio: 0.982 }))!;

    expect(view.tone).toBe("ok");
    expect(view.label).toBe("55.0 t/s");
    expect(view.detail).toContain("98%");
  });

  it("explains each reason a member could not be compared", () => {
    const reasons = [
      "single-member",
      "insufficient-samples",
      "idle",
      "multi-deployment-node",
      "model-mismatch",
    ] as const;

    for (const reason of reasons) {
      const view = describeThroughput(verdict({ state: "not-comparable", reason }))!;
      expect(view.tone, reason).toBe("muted");
      expect(view.label, reason).toBe("not compared");
      // Every reason gets a sentence of its own — never a bare enum value.
      expect(view.detail, reason).toBeTruthy();
      expect(view.detail, reason).not.toContain(reason);
    }
  });
});

describe("suspectedNodes", () => {
  const pool = (members: { node: string; state: ThroughputVerdict["state"] }[]) => ({
    publishedName: "pool-a",
    members: members.map((m) => ({
      node: m.node,
      throughput: verdict({ state: m.state, rate: m.state === "suspect" ? 23 : 55 }),
    })),
  });

  it("is empty when every member is keeping up", () => {
    expect(suspectedNodes([pool([{ node: "spark-03", state: "ok" }, { node: "spark-04", state: "ok" }])]).size)
      .toBe(0);
  });

  it("names the node behind its peers, with the verdict that says so", () => {
    const found = suspectedNodes([
      pool([{ node: "spark-02", state: "suspect" }, { node: "spark-03", state: "ok" }]),
    ]);

    expect([...found.keys()]).toEqual(["spark-02"]);
    expect(found.get("spark-02")!.rate).toBe(23);
  });

  // A node hosting two pool members is one node with one badge; being behind
  // in any pool is enough to want a look at it.
  it("keeps the suspect verdict when a node appears in several pools", () => {
    const found = suspectedNodes([
      pool([{ node: "spark-02", state: "ok" }, { node: "spark-03", state: "ok" }]),
      { publishedName: "pool-b", members: [{ node: "spark-02", throughput: verdict({ state: "suspect", rate: 9 }) }] },
    ]);

    expect(found.get("spark-02")!.state).toBe("suspect");
  });

  it("ignores members the server could not judge", () => {
    const found = suspectedNodes([
      { publishedName: "pool-a", members: [
        { node: "spark-03", throughput: verdict({ state: "not-comparable", reason: "idle" }) },
        { node: "spark-04", throughput: null },
      ] },
    ]);

    expect(found.size).toBe(0);
  });
});
