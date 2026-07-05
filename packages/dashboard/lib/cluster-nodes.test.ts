import { describe, it, expect } from "vitest";
import { buildClusterNodeIds } from "./cluster-nodes.js";

describe("buildClusterNodeIds", () => {
  it("moves the head to the front, preserving the rest order", () => {
    expect(buildClusterNodeIds("c", ["a", "b", "c", "d"])).toEqual(["c", "a", "b", "d"]);
  });
  it("returns selection unchanged when head is absent from selection", () => {
    expect(buildClusterNodeIds("z", ["a", "b"])).toEqual(["a", "b"]);
  });
  it("returns selection unchanged when head is null/undefined", () => {
    expect(buildClusterNodeIds(null, ["a", "b"])).toEqual(["a", "b"]);
    expect(buildClusterNodeIds(undefined, ["a", "b"])).toEqual(["a", "b"]);
  });
  it("single node -> [id]", () => {
    expect(buildClusterNodeIds("a", ["a"])).toEqual(["a"]);
  });
  it("dedupes", () => {
    expect(buildClusterNodeIds("b", ["a", "b", "b", "a"])).toEqual(["b", "a"]);
  });
  it("accepts a Set", () => {
    expect(buildClusterNodeIds("b", new Set(["a", "b", "c"]))).toEqual(["b", "a", "c"]);
  });
});
