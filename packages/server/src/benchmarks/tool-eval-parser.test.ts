import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseToolEvalResults } from "./tool-eval-parser.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = join(here, "../__tests__/integration/benchmarks.fixtures/tool-eval-result.json");
const fixture = readFileSync(fixturePath, "utf-8");

describe("parseToolEvalResults", () => {
  it("maps headline fields from the real captured sample", () => {
    const r = parseToolEvalResults(fixture);
    expect(r.finalScore).toBe(67);
    expect(r.rating).toBe("★★★ Adequate");
    expect(r.deployability).toBe(48);
    expect(r.responsiveness).toBe(2);
    expect(r.totalScenarios).toBe(15);
    expect(r.totalPoints).toBe(20);
    expect(r.maxPoints).toBe(30);
    expect(r.safetyWarnings).toEqual([]);
  });

  it("maps every category_scores entry 1:1", () => {
    const r = parseToolEvalResults(fixture);
    expect(r.categories.length).toBe(5);
    const a = r.categories.find((c) => c.code === "A")!;
    expect(a).toMatchObject({
      code: "A", label: "Tool Selection", percent: 100,
      earned: 6, maxPoints: 6, passCount: 3, partialCount: 0, failCount: 0,
    });
  });

  it("throws on invalid JSON", () => {
    expect(() => parseToolEvalResults("not json")).toThrow(/failed to parse tool-eval JSON/);
  });

  it("throws when final_score is missing", () => {
    const bad = JSON.stringify({ rating: "x", total_scenarios: 1, scores: { category_scores: [] } });
    expect(() => parseToolEvalResults(bad)).toThrow(/final_score/);
  });

  it("throws when scores.category_scores is missing", () => {
    const bad = JSON.stringify({ final_score: 1, rating: "x", total_scenarios: 1, scores: {} });
    expect(() => parseToolEvalResults(bad)).toThrow(/category_scores/);
  });

  it("defaults safety_warnings to [] and optional ints to null when absent", () => {
    const minimal = JSON.stringify({
      final_score: 50, rating: "★★", total_scenarios: 3,
      scores: { category_scores: [] },
    });
    const r = parseToolEvalResults(minimal);
    expect(r.safetyWarnings).toEqual([]);
    expect(r.deployability).toBeNull();
    expect(r.responsiveness).toBeNull();
    expect(r.totalPoints).toBeNull();
    expect(r.maxPoints).toBeNull();
  });
});
