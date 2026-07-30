import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { assessPool, type MemberCandidate } from "./eligibility.js";

const candidate = (over: Partial<MemberCandidate> = {}): MemberCandidate => ({
  id: "d1",
  status: "running",
  port: 8000,
  nodeId: "n1",
  nodeIp: "10.0.0.1",
  ...over,
});

const allOnline = () => true;

/** The eligible half — most cases below are about who survives the filter. */
const eligibleOf = (c: MemberCandidate[], online: (id: string) => boolean) =>
  assessPool(c, online).eligible;

describe("assessPool", () => {
  it("keeps a running member with a port on a live node", () => {
    expect(eligibleOf([candidate()], allOnline)).toEqual([
      { deploymentId: "d1", nodeId: "n1", baseUrl: "http://10.0.0.1:8000" },
    ]);
  });

  it("drops a member that is not running", () => {
    expect(eligibleOf([candidate({ status: "loading" })], allOnline)).toEqual([]);
  });

  it("drops a member with no port bound", () => {
    expect(eligibleOf([candidate({ port: null })], allOnline)).toEqual([]);
  });

  it("drops a member whose node has no address", () => {
    expect(eligibleOf([candidate({ nodeIp: null })], allOnline)).toEqual([]);
  });

  // The status column can lag a dead node by up to the staleness sweep, so a
  // live agent socket is the stronger signal: without it a request is sent into
  // a black hole and hangs until the socket times out.
  it("drops a member whose node has no live agent connection", () => {
    expect(eligibleOf([candidate()], () => false)).toEqual([]);
  });

  it("keeps every eligible member of a pool, in the order given", () => {
    const members = eligibleOf(
      [
        candidate({ id: "a", nodeId: "na", nodeIp: "10.0.0.1" }),
        candidate({ id: "b", nodeId: "nb", nodeIp: "10.0.0.2", port: 11434 }),
      ],
      allOnline,
    );
    expect(members.map((m) => m.deploymentId)).toEqual(["a", "b"]);
    expect(members[1].baseUrl).toBe("http://10.0.0.2:11434");
  });

  // Each reason has to be distinguishable, because the refusal body is the only
  // thing telling an operator whether to wait or to go and look at a node.
  it.each([
    ["not-running", { status: "loading" }, /'loading'/],
    ["no-port", { port: null }, /no port bound/],
    ["no-node-address", { nodeIp: null }, /no known address/],
  ] as const)("explains a %s exclusion", (reason, over, detailPattern) => {
    const { eligible, excluded } = assessPool([candidate(over)], allOnline);
    expect(eligible).toEqual([]);
    expect(excluded).toHaveLength(1);
    expect(excluded[0]).toMatchObject({ deploymentId: "d1", nodeId: "n1", reason });
    expect(excluded[0].detail).toMatch(detailPattern);
  });

  it("explains an agent-offline exclusion", () => {
    const { excluded } = assessPool([candidate()], () => false);
    expect(excluded[0]).toMatchObject({ reason: "agent-offline" });
    expect(excluded[0].detail).toMatch(/no live agent connection/);
  });

  // Reasons are checked in order of severity, so a stopped deployment on a dead
  // node reports the deployment — the thing the operator changes first.
  it("reports one reason per member, the first that applies", () => {
    const { excluded } = assessPool([candidate({ status: "stopped", port: null })], () => false);
    expect(excluded).toHaveLength(1);
    expect(excluded[0].reason).toBe("not-running");
  });

  it("accounts for every member, whether it can serve or not", () => {
    const { eligible, excluded } = assessPool(
      [candidate({ id: "ok" }), candidate({ id: "bad", status: "failed" })],
      allOnline,
    );
    expect(eligible.map((m) => m.deploymentId)).toEqual(["ok"]);
    expect(excluded.map((m) => m.deploymentId)).toEqual(["bad"]);
  });

  it("consults liveness per node, not globally", () => {
    const members = eligibleOf(
      [
        candidate({ id: "a", nodeId: "dead", nodeIp: "10.0.0.1" }),
        candidate({ id: "b", nodeId: "live", nodeIp: "10.0.0.2" }),
      ],
      (nodeId) => nodeId === "live",
    );
    expect(members.map((m) => m.deploymentId)).toEqual(["b"]);
  });

  /**
   * Invariant: a member is eligible only if every condition holds — running,
   * a bound port, a node address, and a live agent socket. No combination of
   * inputs may produce a member that fails any one of them, because each is
   * the difference between a served request and one that hangs.
   */
  test.prop([
    fc.array(
      fc.record({
        id: fc.string({ minLength: 1 }),
        status: fc.constantFrom("running", "loading", "stopped", "failed", "pending"),
        port: fc.option(fc.integer({ min: 1, max: 65535 }), { nil: null }),
        nodeId: fc.constantFrom("n1", "n2"),
        nodeIp: fc.option(fc.constantFrom("10.0.0.1", "10.0.0.2"), { nil: null }),
      }),
      { maxLength: 12 },
    ),
    fc.func(fc.boolean()),
  ])("never returns a member failing any single condition", (rows, online) => {
    const isOnline = (nodeId: string) => online(nodeId);
    const eligible = eligibleOf(rows, isOnline);

    for (const m of eligible) {
      const row = rows.find((r) => r.id === m.deploymentId)!;
      expect(row.status).toBe("running");
      expect(row.port).not.toBeNull();
      expect(row.nodeIp).not.toBeNull();
      expect(isOnline(row.nodeId)).toBe(true);
    }
    expect(eligible.length).toBeLessThanOrEqual(rows.length);
  });
});
