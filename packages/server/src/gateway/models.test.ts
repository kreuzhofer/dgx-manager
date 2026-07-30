import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { toModelList } from "./models.js";

const at = (iso: string) => new Date(iso);

describe("toModelList", () => {
  it("advertises a published name in the OpenAI list shape", () => {
    const list = toModelList([{ publishedName: "glm-5.2", createdAt: at("2026-07-01T00:00:00Z") }]);
    expect(list.object).toBe("list");
    expect(list.data).toEqual([
      { id: "glm-5.2", object: "model", created: 1782864000, owned_by: "dgx-manager" },
    ]);
  });

  // Two deployments serving one name are a pool, not two models. A client picks
  // a name, never a member, so the name must appear exactly once.
  it("collapses a pool of several deployments into one entry", () => {
    const list = toModelList([
      { publishedName: "qwen3-embedding:8b", createdAt: at("2026-07-02T00:00:00Z") },
      { publishedName: "qwen3-embedding:8b", createdAt: at("2026-07-01T00:00:00Z") },
    ]);
    expect(list.data).toHaveLength(1);
    expect(list.data[0].id).toBe("qwen3-embedding:8b");
  });

  // The oldest member is when the cluster started serving that name.
  it("dates a pooled name from its earliest member", () => {
    const list = toModelList([
      { publishedName: "m", createdAt: at("2026-07-02T00:00:00Z") },
      { publishedName: "m", createdAt: at("2026-07-01T00:00:00Z") },
    ]);
    expect(list.data[0].created).toBe(Math.floor(at("2026-07-01T00:00:00Z").getTime() / 1000));
  });

  it("omits a deployment with no published name yet", () => {
    const list = toModelList([
      { publishedName: null, createdAt: at("2026-07-01T00:00:00Z") },
      { publishedName: "ready", createdAt: at("2026-07-01T00:00:00Z") },
    ]);
    expect(list.data.map((m) => m.id)).toEqual(["ready"]);
  });

  it("is empty when the cluster publishes nothing", () => {
    expect(toModelList([])).toEqual({ object: "list", data: [] });
  });

  /**
   * Invariant: the list is exactly the set of distinct published names, in a
   * stable order. Whatever rows arrive — duplicates, nulls, any ordering — a
   * client sees each name once and sees the same list twice running, so a model
   * picker does not reshuffle between polls.
   */
  test.prop([
    fc.array(
      fc.record({
        publishedName: fc.option(fc.string({ minLength: 1 }), { nil: null }),
        createdAt: fc.date({ min: new Date("2000-01-01"), max: new Date("2100-01-01") }),
      }),
    ),
  ])("lists each published name exactly once, in a stable order", (rows) => {
    const ids = toModelList(rows).data.map((m) => m.id);
    const expected = [...new Set(rows.map((r) => r.publishedName).filter((n): n is string => !!n))].sort();
    expect(ids).toEqual(expected);
    expect(toModelList(rows).data.map((m) => m.id)).toEqual(ids);
  });
});
