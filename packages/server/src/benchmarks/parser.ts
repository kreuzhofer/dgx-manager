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

function optNum(row: RawRow, key: string): number | null {
  const v = row[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function parseBenchyResults(jsonText: string): BenchmarkResultInput[] {
  let parsed: { rows?: RawRow[] };
  try {
    parsed = JSON.parse(jsonText);
  } catch (e) {
    throw new Error(`failed to parse llama-benchy JSON: ${(e as Error).message}`);
  }
  const rows = parsed.rows ?? [];
  return rows.map((r) => ({
    opType: String(r["op"] ?? "tg"),
    pp: num(r, "pp"),
    tg: num(r, "tg"),
    depth: num(r, "depth"),
    concurrency: num(r, "concurrency"),
    tps: num(r, "t/s"),
    peakTps: optNum(r, "peak t/s"),
    ttfrMs: optNum(r, "ttfr (ms)"),
    estPptMs: optNum(r, "est_ppt (ms)"),
    e2eTtftMs: optNum(r, "e2e_ttft (ms)"),
    tpsStdev: optNum(r, "t/s_stdev"),
    ttfrStdev: optNum(r, "ttfr_stdev"),
  }));
}

export function summarizeResults(rows: BenchmarkResultInput[]): {
  meanTps: number | null;
  meanTtfrMs: number | null;
} {
  if (rows.length === 0) return { meanTps: null, meanTtfrMs: null };
  const meanTps =
    rows.reduce((acc, r) => acc + r.tps, 0) / rows.length;
  const ttfrRows = rows.filter((r) => r.ttfrMs !== null) as Array<
    BenchmarkResultInput & { ttfrMs: number }
  >;
  const meanTtfrMs =
    ttfrRows.length === 0
      ? null
      : ttfrRows.reduce((acc, r) => acc + r.ttfrMs, 0) / ttfrRows.length;
  return { meanTps, meanTtfrMs };
}
