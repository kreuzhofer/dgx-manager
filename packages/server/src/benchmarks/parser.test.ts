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
  it("parses the three rows from the fixture", () => {
    const rows = parseBenchyResults(fixture);
    expect(rows).toHaveLength(3);
  });

  it("maps llama-benchy field names to BenchmarkResult fields", () => {
    const rows = parseBenchyResults(fixture);
    expect(rows[0]).toEqual({
      opType: "pp",
      pp: 512,
      tg: 32,
      depth: 0,
      concurrency: 1,
      tps: 1840.4,
      peakTps: 1955.0,
      ttfrMs: 142.3,
      estPptMs: 278.0,
      e2eTtftMs: 420.1,
      tpsStdev: 18.2,
      ttfrStdev: 5.1,
    });
  });

  it("returns an empty array for empty input", () => {
    expect(parseBenchyResults('{"rows":[]}')).toEqual([]);
  });

  it("throws a descriptive error on malformed JSON", () => {
    expect(() => parseBenchyResults("not json")).toThrow(/parse/i);
  });

  it("throws when required fields are missing on a row", () => {
    const bad = JSON.stringify({ rows: [{ op: "pp", pp: 1 }] });
    expect(() => parseBenchyResults(bad)).toThrow(/missing/i);
  });
});

describe("summarizeResults", () => {
  it("computes mean tps and mean ttfr across all rows", () => {
    const rows = parseBenchyResults(fixture);
    const summary = summarizeResults(rows);
    // (1840.4 + 84.5 + 220.3) / 3 = 715.07
    expect(summary.meanTps).toBeCloseTo(715.07, 1);
    // (142.3 + 142.3 + 410.0) / 3 = 231.53
    expect(summary.meanTtfrMs).toBeCloseTo(231.53, 1);
  });

  it("returns nulls when given no rows", () => {
    expect(summarizeResults([])).toEqual({ meanTps: null, meanTtfrMs: null });
  });
});
