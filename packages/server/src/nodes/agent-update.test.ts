import { describe, it, expect } from "vitest";
import { planAgentUpdate, bundleArchQuery, type AgentNode } from "./agent-update.js";

const N = (id: string, agentVersion: string | null, arch: string | null = "arm64"): AgentNode => ({
  id, name: `node-${id}`, agentVersion, arch,
});

describe("planAgentUpdate", () => {
  const target = "0.5.756";
  // Fleet: a=outdated+online, b=current+online, c=outdated+offline, d=never-reported+online
  const nodes = [N("a", "0.5.738"), N("b", "0.5.756"), N("c", "0.5.738"), N("d", null)];
  const onlineExcept = (offlineIds: string[]) => (id: string) => !offlineIds.includes(id);

  it("updates online outdated nodes, skips current, marks offline", () => {
    const plan = planAgentUpdate(nodes, target, onlineExcept(["c"]));
    expect(plan.toUpdate.map((n) => n.id).sort()).toEqual(["a", "d"]);
    expect(plan.skipped.map((n) => n.id)).toEqual(["b"]);
    expect(plan.offline.map((n) => n.id)).toEqual(["c"]);
  });

  it("force includes an online node already on the target", () => {
    const plan = planAgentUpdate(nodes, target, onlineExcept(["c"]), true);
    expect(plan.toUpdate.map((n) => n.id).sort()).toEqual(["a", "b", "d"]);
    expect(plan.skipped).toEqual([]);
    // still can't push to an offline node even with force
    expect(plan.offline.map((n) => n.id)).toEqual(["c"]);
  });

  it("a never-reported (null) version online node is always a target", () => {
    expect(planAgentUpdate([N("x", null)], target, () => true).toUpdate.map((n) => n.id)).toEqual(["x"]);
  });

  it("offline takes precedence over version match", () => {
    const plan = planAgentUpdate([N("b", "0.5.756")], target, () => false);
    expect(plan.offline.map((n) => n.id)).toEqual(["b"]);
    expect(plan.toUpdate).toEqual([]);
    expect(plan.skipped).toEqual([]);
  });

  it("empty fleet yields empty partitions", () => {
    expect(planAgentUpdate([], target, () => true)).toEqual({ toUpdate: [], skipped: [], offline: [] });
  });
});

describe("bundleArchQuery", () => {
  it("emits ?arch= for known arches, empty otherwise", () => {
    expect(bundleArchQuery("amd64")).toBe("?arch=amd64");
    expect(bundleArchQuery("arm64")).toBe("?arch=arm64");
    expect(bundleArchQuery(null)).toBe("");
    expect(bundleArchQuery(undefined)).toBe("");
    expect(bundleArchQuery("riscv64")).toBe("");
  });
});
