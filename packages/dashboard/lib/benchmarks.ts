import { apiFetch } from "@/lib/api";

export type BenchmarkConfig = {
  pp: number[];
  tg: number[];
  depth: number[];
  runs: number;
  concurrency: number[];
  latencyMode: "api" | "generation" | "none";
  enablePrefixCaching: boolean;
  skipCoherence: boolean;
};

export type ToolEvalConfig = {
  short: boolean;
  hardmode: boolean;
  contextPressure: number | null;
  seed: number;
};

export type BenchmarkKind = "throughput" | "tool-eval" | "accuracy";

export type AccuracyConfig = {
  tasks: string[];
  primaryTask: string;
  primaryMetric: string;
  limit: number | null;
  numFewshot: number | null;
  maxGenToks: number;
  applyChatTemplate: boolean;
  reasoning: boolean;
  seed: number;
};

export type AccuracyMetric = {
  task: string;
  metric: string;
  value: number;
  stderr: number | null;
  isGroup: boolean;
  nSamples: number | null;
  /** Extraction filter that produced this value ("strict-match",
   *  "flexible-extract"...). Absent on rows stored before it was recorded. */
  filter?: string | null;
};

/** A metric whose zero is better explained by failed answer extraction than by
 *  the model getting answers wrong. Derived server-side on read. */
export type ExtractionFailure = {
  task: string;
  metric: string;
  zeroFilters: string[];
  bestValue: number;
};

export type ToolEvalCategory = {
  id: string;
  code: string;
  label: string;
  percent: number;
  earned: number;
  maxPoints: number;
  passCount: number;
  partialCount: number;
  failCount: number;
};

export type BenchmarkPreset = {
  id: string;
  label: string;
  description: string;
  kind: BenchmarkKind;
  config: BenchmarkConfig | ToolEvalConfig | AccuracyConfig;
};

export type BenchmarkResult = {
  id: string;
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

export type BenchmarkRun = {
  /** Derived on read; absent from older API responses. */
  extractionFailures?: ExtractionFailure[];
  id: string;
  deploymentId: string | null;
  presetId: string | null;
  modelName: string;
  endpointUrl: string;
  servedModelName: string;
  config: string;
  status: "pending" | "running" | "completed" | "failed" | "canceled";
  error: string | null;
  startedAt: string | null;
  completedAt: string | null;
  meanTps: number | null;
  meanTtfrMs: number | null;
  /** Where the run executed. null = a legacy run measured on the manager (the Pi).
   *  Throughput/latency depend on the client host, so cross-runner runs aren't comparable. */
  runnerNodeId: string | null;
  kind: BenchmarkKind;
  toolEvalScore: number | null;
  toolEvalRating: string | null;
  toolEvalDeployability: number | null;
  toolEvalResponsiveness: number | null;
  toolEvalTotalScenarios: number | null;
  toolEvalTotalPoints: number | null;
  toolEvalMaxPoints: number | null;
  toolEvalSafetyWarnings: string | null;
  toolEvalCategories?: ToolEvalCategory[];
  accuracyScore: number | null;
  accuracyMetrics: string | null;
  rawOutput: string | null;
  createdAt: string;
  results?: BenchmarkResult[];
  deployment?: {
    id: string;
    displayName: string | null;
    node: { id: string; name: string; ipAddress: string | null };
    model: { id: string; name: string; runtime: string };
  } | null;
};

export async function listPresets(): Promise<BenchmarkPreset[]> {
  return apiFetch<BenchmarkPreset[]>("/api/benchmarks/presets");
}

export async function listBenchmarks(opts?: {
  deploymentId?: string;
}): Promise<BenchmarkRun[]> {
  const qs = opts?.deploymentId ? `?deploymentId=${opts.deploymentId}` : "";
  return apiFetch<BenchmarkRun[]>(`/api/benchmarks${qs}`);
}

export async function getBenchmark(id: string): Promise<BenchmarkRun> {
  return apiFetch<BenchmarkRun>(`/api/benchmarks/${id}`);
}

export async function startBenchmark(body: {
  deploymentId: string;
  presetId?: string;
  config?: BenchmarkConfig;
}): Promise<BenchmarkRun> {
  return apiFetch<BenchmarkRun>("/api/benchmarks", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function cancelBenchmark(id: string): Promise<BenchmarkRun> {
  return apiFetch<BenchmarkRun>(`/api/benchmarks/${id}/cancel`, {
    method: "POST",
  });
}

export async function deleteBenchmark(id: string): Promise<void> {
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";
  const res = await fetch(`${API_BASE}/api/benchmarks/${id}`, {
    method: "DELETE",
    cache: "no-store",
  });
  if (!res.ok && res.status !== 204) {
    const body = await res.json().catch(() => ({}));
    throw new Error(
      (body as { error?: string }).error || `delete failed: ${res.status}`,
    );
  }
}

export async function getBenchmarkLog(id: string): Promise<string> {
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";
  const res = await fetch(`${API_BASE}/api/benchmarks/${id}/logs`, {
    cache: "no-store",
  });
  if (!res.ok) return "";
  return res.text();
}
