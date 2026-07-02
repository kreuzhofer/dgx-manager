export type BenchmarkResultInput = {
  opType: string;
  pp: number;
  tg: number;
  depth: number;
  concurrency: number;
  tps: number;
  peakTps: number | null;
  ttfrMs: number | null;
  estPptMs: number | null;
  e2eTtftMs: number | null;
  tpsStdev: number | null;
  ttfrStdev: number | null;
};

type RawRow = Record<string, unknown>;

function num(row: RawRow, key: string): number {
  const v = row[key];
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new Error(`benchmark row missing required numeric field: ${key}`);
  }
  return v;
}

// llama-benchy 0.3.7 reports metrics as nested {mean, std, values} objects.
function nestedNum(row: RawRow, key: string, sub: "mean" | "std"): number {
  const v = row[key];
  if (!v || typeof v !== "object") {
    throw new Error(`benchmark row missing required metric object: ${key}`);
  }
  const inner = (v as Record<string, unknown>)[sub];
  if (typeof inner !== "number" || !Number.isFinite(inner)) {
    throw new Error(`benchmark row missing required numeric field: ${key}.${sub}`);
  }
  return inner;
}

function optNestedNum(row: RawRow, key: string, sub: "mean" | "std"): number | null {
  const v = row[key];
  if (!v || typeof v !== "object") return null;
  const inner = (v as Record<string, unknown>)[sub];
  return typeof inner === "number" && Number.isFinite(inner) ? inner : null;
}

// Parse llama-benchy 0.3.7's JSON output. Each entry in `benchmarks` reports
// BOTH prompt-processing and token-generation throughput for a single
// (concurrency, context_size, prompt_size, response_size) workload, so we
// split it into two `BenchmarkResultInput` rows: one with opType="pp" using
// pp_throughput, one with opType="tg" using tg_throughput. The latency-style
// metrics (ttfr, est_ppt, e2e_ttft, peak_throughput) are shared by both rows.
export function parseBenchyResults(jsonText: string): BenchmarkResultInput[] {
  let parsed: { benchmarks?: RawRow[] };
  try {
    parsed = JSON.parse(jsonText);
  } catch (e) {
    throw new Error(`failed to parse llama-benchy JSON: ${(e as Error).message}`);
  }
  const benchmarks = parsed.benchmarks ?? [];
  const out: BenchmarkResultInput[] = [];
  for (const b of benchmarks) {
    const pp = num(b, "prompt_size");
    const tg = num(b, "response_size");
    const depth = num(b, "context_size");
    const concurrency = num(b, "concurrency");
    const ttfrMs = optNestedNum(b, "ttfr", "mean");
    const ttfrStdev = optNestedNum(b, "ttfr", "std");
    const estPptMs = optNestedNum(b, "est_ppt", "mean");
    const e2eTtftMs = optNestedNum(b, "e2e_ttft", "mean");
    // peak_throughput is documented by llama-benchy as "Peak generation
    // tokens per second (total)" — strictly a decode (tg) metric. It's
    // wrong to display it on the prefill (pp) row, where it would appear
    // smaller than the mean and confuse the reader.
    const peakTgTps = optNestedNum(b, "peak_throughput", "mean");

    out.push({
      opType: "pp",
      pp, tg, depth, concurrency,
      tps: nestedNum(b, "pp_throughput", "mean"),
      peakTps: null,
      ttfrMs, estPptMs, e2eTtftMs,
      tpsStdev: optNestedNum(b, "pp_throughput", "std"),
      ttfrStdev,
    });
    out.push({
      opType: "tg",
      pp, tg, depth, concurrency,
      tps: nestedNum(b, "tg_throughput", "mean"),
      peakTps: peakTgTps,
      ttfrMs, estPptMs, e2eTtftMs,
      tpsStdev: optNestedNum(b, "tg_throughput", "std"),
      ttfrStdev,
    });
  }
  return out;
}

export function summarizeResults(rows: BenchmarkResultInput[]): {
  meanTps: number | null;
  meanTtfrMs: number | null;
} {
  if (rows.length === 0) return { meanTps: null, meanTtfrMs: null };
  // meanTps is a DECODE metric: average tg rows only. Averaging prefill tps
  // into it produced numbers like "35.8 tok/s" for a model that generates at
  // 4.6 tok/s (benchmark cmr1a4tm02pp136k42hhf2d9u).
  const tgRows = rows.filter((r) => r.opType === "tg");
  const meanTps =
    tgRows.length === 0
      ? null
      : tgRows.reduce((acc, r) => acc + r.tps, 0) / tgRows.length;
  // ttfr is per-workload (shared by a workload's pp and tg rows) — average it
  // over the same tg rows so both summary metrics describe the same set.
  const ttfrRows = (tgRows.length > 0 ? tgRows : rows).filter(
    (r) => r.ttfrMs !== null,
  ) as Array<BenchmarkResultInput & { ttfrMs: number }>;
  const meanTtfrMs =
    ttfrRows.length === 0
      ? null
      : ttfrRows.reduce((acc, r) => acc + r.ttfrMs, 0) / ttfrRows.length;
  return { meanTps, meanTtfrMs };
}
