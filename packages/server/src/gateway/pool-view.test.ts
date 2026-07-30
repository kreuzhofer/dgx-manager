import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { buildPoolView, type PoolViewRow } from "./pool-view.js";

const row = (over: Partial<PoolViewRow> = {}): PoolViewRow => ({
  id: "d1",
  publishedName: "glm-5.2",
  status: "running",
  port: 8000,
  nodeId: "n1",
  nodeName: "dgx-spark-01",
  nodeIp: "10.0.0.1",
  runtime: "vllm",
  ...over,
});

const allOnline = () => true;
const noLoad = () => 0;

describe("buildPoolView", () => {
  it("shows a serving member with its node, runtime and in-flight count", () => {
    const [pool] = buildPoolView([row()], allOnline, () => 3);
    expect(pool.publishedName).toBe("glm-5.2");
    expect(pool.servingCount).toBe(1);
    expect(pool.members).toEqual([
      {
        deploymentId: "d1",
        node: "dgx-spark-01",
        runtime: "vllm",
        port: 8000,
        inflight: 3,
        serving: true,
      },
    ]);
  });

  // The whole reason a dedicated view exists: two rows sharing a name are one
  // pool, which a per-deployment table cannot show.
  it("collapses deployments sharing a name into one pool", () => {
    const view = buildPoolView(
      [
        row({ id: "a", nodeId: "na", nodeName: "agenthost" }),
        row({ id: "b", nodeId: "nb", nodeName: "aihost01" }),
      ],
      allOnline,
      noLoad,
    );
    expect(view).toHaveLength(1);
    expect(view[0].members.map((m) => m.node)).toEqual(["agenthost", "aihost01"]);
    expect(view[0].servingCount).toBe(2);
  });

  it("keeps different published names as separate pools", () => {
    const view = buildPoolView(
      [row({ id: "a", publishedName: "zeta" }), row({ id: "b", publishedName: "alpha" })],
      allOnline,
      noLoad,
    );
    expect(view.map((p) => p.publishedName)).toEqual(["alpha", "zeta"]);
  });

  // Eligibility visible before a request fails, rather than only after one does.
  it("marks a member on a dead node as not serving, with the reason", () => {
    const [pool] = buildPoolView([row()], () => false, noLoad);
    expect(pool.servingCount).toBe(0);
    expect(pool.members[0]).toMatchObject({
      serving: false,
      reason: "agent-offline",
    });
    expect(pool.members[0].detail).toMatch(/no live agent connection/);
  });

  it("explains a member that is not running", () => {
    const [pool] = buildPoolView([row({ status: "restarting" })], allOnline, noLoad);
    expect(pool.members[0]).toMatchObject({ serving: false, reason: "not-running" });
  });

  it("lists serving members before excluded ones", () => {
    const [pool] = buildPoolView(
      [
        row({ id: "a", nodeId: "dead", nodeName: "down" }),
        row({ id: "b", nodeId: "live", nodeName: "up" }),
      ],
      (nodeId) => nodeId === "live",
      noLoad,
    );
    expect(pool.members.map((m) => m.node)).toEqual(["up", "down"]);
    expect(pool.servingCount).toBe(1);
  });

  it("is empty when nothing is published", () => {
    expect(buildPoolView([], allOnline, noLoad)).toEqual([]);
  });

  /**
   * Invariant: every published deployment appears exactly once, in exactly one
   * pool, and is either serving or carries a reason it is not. A member that
   * silently vanished from the view would be a deployment the operator cannot
   * account for.
   */
  test.prop([
    fc.array(
      fc.record({
        id: fc.string({ minLength: 1 }),
        publishedName: fc.constantFrom("a", "b", "c"),
        status: fc.constantFrom("running", "restarting", "loading"),
        port: fc.option(fc.integer({ min: 1, max: 65535 }), { nil: null }),
        nodeId: fc.constantFrom("n1", "n2"),
        nodeIp: fc.option(fc.constantFrom("10.0.0.1"), { nil: null }),
      }),
      { maxLength: 12 },
    ),
    fc.func(fc.boolean()),
  ])("accounts for every published deployment exactly once", (rows, online) => {
    const unique = [...new Map(rows.map((r) => [r.id, r])).values()];
    const view = buildPoolView(
      unique.map((r) => ({ ...r, nodeName: `node-${r.nodeId}`, runtime: "vllm" })),
      (nodeId) => online(nodeId),
      noLoad,
    );

    const seen = view.flatMap((p) => p.members.map((m) => m.deploymentId));
    expect(seen.slice().sort()).toEqual(unique.map((r) => r.id).sort());
    for (const pool of view) {
      for (const member of pool.members) {
        if (!member.serving) expect(member.reason).toBeDefined();
      }
      expect(pool.servingCount).toBe(pool.members.filter((m) => m.serving).length);
    }
  });
});
