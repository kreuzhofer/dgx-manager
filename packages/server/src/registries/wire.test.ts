import { describe, it, expect } from "vitest";
import { registryRowsToWire } from "./wire.js";

const row = (over: Record<string, unknown> = {}) => ({
  id: "x", name: "eugr", url: "https://github.com/eugr/spark-vllm-docker", subpath: "recipes",
  description: null, visible: true, tuningSubpath: null, benchmarkSubpath: null, modsSubpath: "mods",
  sortOrder: 0, createdAt: new Date(0), updatedAt: new Date(0), ...over,
});

describe("registryRowsToWire", () => {
  it("maps camelCase rows to snake_case wire shape, omitting null optionals", () => {
    expect(registryRowsToWire([row()])).toEqual([
      { name: "eugr", url: "https://github.com/eugr/spark-vllm-docker", subpath: "recipes", mods_subpath: "mods" },
    ]);
  });

  it("includes visible only when false", () => {
    expect(registryRowsToWire([row({ visible: false })])[0].visible).toBe(false);
    expect(registryRowsToWire([row({ visible: true })])[0]).not.toHaveProperty("visible");
  });

  it("orders by sortOrder then name", () => {
    const out = registryRowsToWire([row({ name: "b", sortOrder: 1 }), row({ name: "a", sortOrder: 1 }), row({ name: "z", sortOrder: 0 })]);
    expect(out.map((r) => r.name)).toEqual(["z", "a", "b"]);
  });
});
