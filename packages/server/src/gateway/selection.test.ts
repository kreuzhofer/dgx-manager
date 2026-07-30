import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { selectLeastOutstanding } from "./selection.js";
import type { EligibleMember } from "./eligibility.js";

const member = (id: string): EligibleMember => ({
  deploymentId: id,
  nodeId: `n-${id}`,
  baseUrl: `http://10.0.0.1:${8000 + id.charCodeAt(0)}`,
});

const counts = (m: Record<string, number>) => (id: string) => m[id] ?? 0;

describe("selectLeastOutstanding", () => {
  it("returns nothing when there is no one to choose", () => {
    expect(selectLeastOutstanding([], counts({}), 0)).toBeNull();
  });

  it("takes the only member of a pool of one", () => {
    expect(selectLeastOutstanding([member("a")], counts({ a: 99 }), 0)?.deploymentId).toBe("a");
  });

  // The reason this rule exists: a pool can span hardware an order of magnitude
  // apart, and the slow member accumulates in-flight requests and stops
  // attracting new ones without anyone configuring a weight.
  it("prefers the member with fewer requests in flight", () => {
    const chosen = selectLeastOutstanding(
      [member("slow"), member("fast")],
      counts({ slow: 3, fast: 0 }),
      0,
    );
    expect(chosen?.deploymentId).toBe("fast");
  });

  it("still prefers the least busy when the busiest comes last", () => {
    const chosen = selectLeastOutstanding(
      [member("idle"), member("busy")],
      counts({ idle: 1, busy: 7 }),
      0,
    );
    expect(chosen?.deploymentId).toBe("idle");
  });

  // With nothing in flight anywhere — the ordinary case between requests — the
  // rule degenerates to round-robin rather than always picking the same member.
  it("rotates between equally idle members", () => {
    const pool = [member("a"), member("b"), member("c")];
    const picks = [0, 1, 2, 3].map((i) => selectLeastOutstanding(pool, counts({}), i)?.deploymentId);
    expect(picks).toEqual(["a", "b", "c", "a"]);
  });

  it("rotates only among the tied minimum, never the busy one", () => {
    const pool = [member("a"), member("busy"), member("b")];
    const picks = [0, 1, 2, 3].map(
      (i) => selectLeastOutstanding(pool, counts({ busy: 5 }), i)?.deploymentId,
    );
    expect(picks).toEqual(["a", "b", "a", "b"]);
  });

  it("tolerates a negative rotation counter", () => {
    const pool = [member("a"), member("b")];
    expect(selectLeastOutstanding(pool, counts({}), -1)).not.toBeNull();
  });

  /**
   * Invariant: the chosen member always has the minimum outstanding count of
   * the pool, and a choice is made whenever the pool is non-empty. Anything
   * else would send work to a member that is already busier than one sitting
   * idle beside it.
   */
  test.prop([
    fc.array(fc.tuple(fc.string({ minLength: 1 }), fc.nat({ max: 50 })), { maxLength: 10 }),
    fc.integer(),
  ])("always picks a member holding the minimum count", (pairs, rotation) => {
    const unique = [...new Map(pairs).entries()];
    const pool = unique.map(([id]) => member(id));
    const lookup = counts(Object.fromEntries(unique));

    const chosen = selectLeastOutstanding(pool, lookup, rotation);

    if (pool.length === 0) {
      expect(chosen).toBeNull();
      return;
    }
    expect(chosen).not.toBeNull();
    const min = Math.min(...pool.map((m) => lookup(m.deploymentId)));
    expect(lookup(chosen!.deploymentId)).toBe(min);
    expect(pool.map((m) => m.deploymentId)).toContain(chosen!.deploymentId);
  });
});
