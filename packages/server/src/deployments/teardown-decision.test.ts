import { describe, it, expect } from "vitest";
import { decideTeardown } from "./teardown-decision.js";

describe("decideTeardown", () => {
  it("fans out to every rank of a multi-rank dgxrun failure", () => {
    const out = decideTeardown({ runner: "dgxrun", nodeIds: ["a", "b"], trigger: "failed" });
    expect(out).toEqual({ kind: "fan", nodeIds: ["a", "b"], preserveContainer: true });
  });

  /**
   * #94: the teardown fired on a TP=1 deployment, where "tear down every rank"
   * means "delete the only container". The agent's own health path already
   * stops a failed solo container, so this bought nothing and destroyed the
   * only place the cause could still be read — 19 seconds after the first
   * hard timeout.
   */
  it("SKIPS a single-rank deployment entirely", () => {
    const out = decideTeardown({ runner: "dgxrun", nodeIds: ["only"], trigger: "failed" });
    expect(out.kind).toBe("skip");
    expect(out).toMatchObject({ reason: expect.stringContaining("single-rank") });
  });

  // A post-mortem needs a body: on failure the container is stopped, not removed.
  it("preserves containers when the trigger is a failure", () => {
    expect(decideTeardown({ runner: "dgxrun", nodeIds: ["a", "b"], trigger: "failed" }))
      .toMatchObject({ preserveContainer: true });
  });

  // A routine stop is not a post-mortem; removing is correct and reclaims disk.
  it("does NOT preserve on a routine stop", () => {
    expect(decideTeardown({ runner: "dgxrun", nodeIds: ["a", "b"], trigger: "stopped" }))
      .toMatchObject({ preserveContainer: false });
  });

  it("skips anything that is not a dgxrun deployment", () => {
    expect(decideTeardown({ runner: "sparkrun", nodeIds: ["a", "b"], trigger: "failed" }).kind)
      .toBe("skip");
    expect(decideTeardown({ runner: undefined, nodeIds: ["a", "b"], trigger: "failed" }).kind)
      .toBe("skip");
  });

  it("skips when there are no nodes", () => {
    expect(decideTeardown({ runner: "dgxrun", nodeIds: [], trigger: "failed" }).kind).toBe("skip");
  });
});
