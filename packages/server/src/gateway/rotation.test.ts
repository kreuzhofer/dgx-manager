import { afterEach, describe, expect, it } from "vitest";
import { nextRotation, resetRotations } from "./rotation.js";
import { selectLeastOutstanding } from "./selection.js";
import type { EligibleMember } from "./eligibility.js";

afterEach(resetRotations);

const member = (id: string): EligibleMember => ({
  deploymentId: id,
  nodeId: "n",
  baseUrl: "http://10.0.0.1:8000",
});
const idle = () => 0;

describe("nextRotation", () => {
  it("advances for a given pool", () => {
    expect(nextRotation("a")).toBe(1);
    expect(nextRotation("a")).toBe(2);
  });

  it("keeps pools independent", () => {
    nextRotation("a");
    nextRotation("a");
    expect(nextRotation("b")).toBe(1);
  });

  it("stays an exact integer rather than growing without bound", () => {
    for (let i = 0; i < 5; i++) nextRotation("a");
    expect(Number.isSafeInteger(nextRotation("a"))).toBe(true);
  });

  /**
   * The regression this module exists for: a two-member pool interleaved with
   * traffic to another name must still alternate. With one counter shared by
   * the whole gateway it aliased to a single parity and one member was never
   * chosen at all.
   */
  it("does not starve a pool member when another pool takes traffic between requests", () => {
    const pool = [member("A0"), member("A1")];
    const other = [member("solo")];

    const picks: string[] = [];
    for (let i = 0; i < 8; i++) {
      picks.push(selectLeastOutstanding(pool, idle, nextRotation("pooled"))!.deploymentId);
      selectLeastOutstanding(other, idle, nextRotation("solo-name"));
    }

    expect(picks).toContain("A0");
    expect(picks).toContain("A1");
    // Evenly, not merely at least once.
    expect(picks.filter((p) => p === "A0")).toHaveLength(4);
  });
});
