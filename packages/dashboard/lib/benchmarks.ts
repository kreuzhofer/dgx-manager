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

export type BenchmarkPreset = {
  id: string;
  label: string;
  description: string;
  config: BenchmarkConfig;
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
