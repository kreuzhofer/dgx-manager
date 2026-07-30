import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { selectEligibleMembers, type MemberCandidate } from "./eligibility.js";

const candidate = (over: Partial<MemberCandidate> = {}): MemberCandidate => ({
  id: "d1",
  status: "running",
  port: 8000,
  nodeId: "n1",
  nodeIp: "10.0.0.1",
  ...over,
});

const allOnline = () => true;

describe("selectEligibleMembers", () => {
  it("keeps a running member with a port on a live node", () => {
    expect(selectEligibleMembers([candidate()], allOnline)).toEqual([
      { deploymentId: "d1", nodeId: "n1", baseUrl: "http://10.0.0.1:8000" },
    ]);
  });

  it("drops a member that is not running", () => {
    expect(selectEligibleMembers([candidate({ status: "loading" })], allOnline)).toEqual([]);
  });

  it("drops a member with no port bound", () => {
    expect(selectEligibleMembers([candidate({ port: null })], allOnline)).toEqual([]);
  });

  it("drops a member whose node has no address", () => {
    expect(selectEligibleMembers([candidate({ nodeIp: null })], allOnline)).toEqual([]);
  });

  // The status column can lag a dead node by up to the staleness sweep, so a
  // live agent socket is the stronger signal: without it a request is sent into
  // a black hole and hangs until the socket times out.
  it("drops a member whose node has no live agent connection", () => {
    expect(selectEligibleMembers([candidate()], () => false)).toEqual([]);
  });

  it("keeps every eligible member of a pool, in the order given", () => {
    const members = selectEligibleMembers(
      [
        candidate({ id: "a", nodeId: "na", nodeIp: "10.0.0.1" }),
        candidate({ id: "b", nodeId: "nb", nodeIp: "10.0.0.2", port: 11434 }),
      ],
      allOnline,
    );
    expect(members.map((m) => m.deploymentId)).toEqual(["a", "b"]);
    expect(members[1].baseUrl).toBe("http://10.0.0.2:11434");
  });

  it("consults liveness per node, not globally", () => {
    const members = selectEligibleMembers(
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
    const eligible = selectEligibleMembers(rows, isOnline);

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
