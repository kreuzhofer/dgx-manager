import { describe, expect, it } from "vitest";
import { resolveEvalNode } from "./eval-node.js";

const n = (id: string, role: string, status = "online") => ({ id, name: id, role, status });

describe("resolveEvalNode", () => {
  it("picks the single online eval node", () => {
    expect(resolveEvalNode([n("a", "gpu"), n("b", "eval")])).toEqual({ ok: true, nodeId: "b" });
  });

  it("fails when no eval node is online", () => {
    const r = resolveEvalNode([n("a", "gpu"), n("b", "eval", "offline")]);
    expect(r).toMatchObject({ ok: false, reason: "none" });
  });

  it("fails when there are none at all", () => {
    expect(resolveEvalNode([n("a", "gpu")])).toMatchObject({ ok: false, reason: "none" });
  });

  // Silently picking the first would make a run's provenance depend on row order —
  // exactly what runnerNodeId exists to prevent.
  it("refuses to guess between two eval nodes", () => {
    const r = resolveEvalNode([n("a", "eval"), n("b", "eval")]);
    expect(r).toMatchObject({ ok: false, reason: "ambiguous" });
    expect((r as { detail: string }).detail).toContain("EVAL_NODE_ID");
  });

  it("honours an explicit EVAL_NODE_ID", () => {
    expect(resolveEvalNode([n("a", "eval"), n("b", "eval")], "b")).toEqual({ ok: true, nodeId: "b" });
  });

  it("rejects an explicit id that is not an online eval node", () => {
    expect(resolveEvalNode([n("a", "eval"), n("b", "gpu")], "b")).toMatchObject({ ok: false });
  });
});
