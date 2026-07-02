import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";
import { parseBenchyResults, summarizeResults } from "./parser.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

const fixture = readFileSync(
  join(__dirname, "../__tests__/integration/benchmarks.fixtures/result.json"),
  "utf-8",
);

describe("parseBenchyResults", () => {
  it("emits two rows (pp + tg) per benchmark entry in the fixture", () => {
    // Fixture has 2 benchmark entries; parser emits 1 pp row + 1 tg row per entry
    const rows = parseBenchyResults(fixture);
    expect(rows).toHaveLength(4);
  });

  it("maps llama-benchy 0.3.7 nested-mean fields to BenchmarkResult fields", () => {
    const rows = parseBenchyResults(fixture);
    expect(rows[0]).toEqual({
      opType: "pp",
      pp: 512,
      tg: 32,
      depth: 0,
      concurrency: 1,
      tps: 1840.4,
      // peakTps is a tg-only metric (llama-benchy: "Peak generation tokens
      // per second total"), so it's null on the prefill row.
      peakTps: null,
      ttfrMs: 142.3,
      estPptMs: 278.0,
      e2eTtftMs: 420.1,
      tpsStdev: 18.2,
      ttfrStdev: 5.1,
    });
    expect(rows[1]).toEqual({
      opType: "tg",
      pp: 512,
      tg: 32,
      depth: 0,
      concurrency: 1,
      tps: 84.5,
      peakTps: 92.1,
      ttfrMs: 142.3,
      estPptMs: 278.0,
      e2eTtftMs: 420.1,
      tpsStdev: 0.9,
      ttfrStdev: 5.1,
    });
  });

  it("returns an empty array for empty input", () => {
    expect(parseBenchyResults('{"benchmarks":[]}')).toEqual([]);
  });

  it("throws a descriptive error on malformed JSON", () => {
    expect(() => parseBenchyResults("not json")).toThrow(/parse/i);
  });

  it("throws when required fields are missing on a benchmark entry", () => {
    const bad = JSON.stringify({ benchmarks: [{ concurrency: 1 }] });
    expect(() => parseBenchyResults(bad)).toThrow(/missing/i);
  });

  it("throws when a required nested metric object is missing its mean", () => {
    const bad = JSON.stringify({
      benchmarks: [{
        concurrency: 1, context_size: 0, prompt_size: 1, response_size: 1,
        pp_throughput: { std: 0, values: [] },
      }],
    });
    expect(() => parseBenchyResults(bad)).toThrow(/missing/i);
  });
});

describe("summarizeResults", () => {
  it("computes mean tps over decode (tg) rows only — prefill must not inflate it", () => {
    const rows = parseBenchyResults(fixture);
    const summary = summarizeResults(rows);
    // fixture tg rows tps: (84.5 + 220.3) / 2 = 152.4 — NOT the pp+tg blend 756.3
    expect(summary.meanTps).toBeCloseTo(152.4, 1);
    // ttfr is per-workload (identical on the pp and tg rows of a workload):
    // (142.3 + 410.0) / 2 = 276.15
    expect(summary.meanTtfrMs).toBeCloseTo(276.15, 1);
  });

  it("returns null meanTps when there are no tg rows", () => {
    const rows = parseBenchyResults(fixture).filter((r) => r.opType === "pp");
    expect(summarizeResults(rows).meanTps).toBeNull();
  });

  it("returns nulls when given no rows", () => {
    expect(summarizeResults([])).toEqual({ meanTps: null, meanTtfrMs: null });
  });
});
