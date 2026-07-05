import { describe, it, expect } from "vitest";
import { selectStaleNodes } from "./staleness.js";

const T = 30_000, NOW = 1_000_000;
const at = (ageMs: number) => new Date(NOW - ageMs);

describe("selectStaleNodes", () => {
  it("selects an online node whose lastSeen is older than the threshold", () => {
    expect(selectStaleNodes([{ id: "a", status: "online", lastSeen: at(40_000) }], NOW, T)).toEqual(["a"]);
  });
  it("does not select a fresh online node", () => {
    expect(selectStaleNodes([{ id: "a", status: "online", lastSeen: at(5_000) }], NOW, T)).toEqual([]);
  });
  it("skips already-offline nodes", () => {
    expect(selectStaleNodes([{ id: "a", status: "offline", lastSeen: at(999_999) }], NOW, T)).toEqual([]);
  });
  it("skips nodes that never heartbeated (null lastSeen)", () => {
    expect(selectStaleNodes([{ id: "a", status: "online", lastSeen: null }], NOW, T)).toEqual([]);
  });
  it("is strict at the boundary (exactly threshold is NOT stale)", () => {
    expect(selectStaleNodes([{ id: "a", status: "online", lastSeen: at(30_000) }], NOW, T)).toEqual([]);
    expect(selectStaleNodes([{ id: "a", status: "online", lastSeen: at(30_001) }], NOW, T)).toEqual(["a"]);
  });
  it("returns only the stale ids from a mixed set", () => {
    expect(selectStaleNodes([
      { id: "fresh", status: "online", lastSeen: at(1_000) },
      { id: "stale", status: "online", lastSeen: at(60_000) },
      { id: "off", status: "offline", lastSeen: at(60_000) },
    ], NOW, T)).toEqual(["stale"]);
  });
});
