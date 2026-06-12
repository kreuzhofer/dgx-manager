import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseSparkrunList, parseClusterId, type SparkrunRecipeSummary } from "./sparkrun-parse.js";

const fixture = readFileSync(
  join(__dirname, "__fixtures__/sparkrun-list.json"),  // captured in Phase 0 (48 recipes)
  "utf8",
);

const runFixture = readFileSync(
  join(__dirname, "__fixtures__/sparkrun-run-dryrun.txt"),
  "utf8",
);

describe("parseSparkrunList", () => {
  it("returns one summary per recipe with name + registry", () => {
    const recipes: SparkrunRecipeSummary[] = parseSparkrunList(fixture);
    expect(recipes.length).toBeGreaterThan(0);
    for (const r of recipes) {
      expect(typeof r.ref).toBe("string");
      expect(r.ref.length).toBeGreaterThan(0);
    }
    expect(recipes.some((r) => r.ref.includes("qwen") || r.registry)).toBe(true);
  });

  it("maps fixture[0] fields to the correct SparkrunRecipeSummary fields", () => {
    const recipes: SparkrunRecipeSummary[] = parseSparkrunList(fixture);
    // Pinned against the first entry of sparkrun-list.json
    expect(recipes[0].ref).toBe("@sparkrun-transitional/qwen3-1.7b-llama-cpp");
    expect(recipes[0].registry).toBe("sparkrun-transitional");
    expect(recipes[0].runtime).toBe("llama-cpp");
  });

  it("never throws on empty input", () => {
    expect(parseSparkrunList("")).toEqual([]);
  });
});

describe("parseSparkrunList — deploy defaults", () => {
  it("carries model, minNodes, and tolerates empty tp/gpu_mem", () => {
    const recipes = parseSparkrunList(fixture);
    for (const r of recipes) {
      expect(typeof r.model).toBe("string");
      expect(r.minNodes).toBeGreaterThanOrEqual(1);
      if (r.tpDefault !== undefined) expect(Number.isFinite(r.tpDefault)).toBe(true);
      if (r.gpuMemDefault !== undefined) expect(Number.isFinite(r.gpuMemDefault)).toBe(true);
    }
    expect(recipes.some((r) => r.minNodes >= 2)).toBe(true);
  });
});

describe("parseClusterId", () => {
  it("extracts the sparkrun_<hex> cluster id from run output", () => {
    expect(parseClusterId(runFixture)).toBe("sparkrun_2cf3f3031766");
  });
  it("returns undefined when absent", () => {
    expect(parseClusterId("no cluster line here")).toBeUndefined();
  });
});
