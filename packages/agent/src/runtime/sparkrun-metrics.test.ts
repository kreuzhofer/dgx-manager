import { describe, it, expect, vi, beforeEach } from "vitest";
import { parseVllmMetrics, firstErrorLine, computeTps } from "./sparkrun-metrics.js";

// Real Phase 0 shape: metric names carry {labels}; this vLLM uses kv_cache_usage_perc.
const sample = `
vllm:num_requests_running{engine="0",model_name="qwen3-1.7b"} 2.0
vllm:num_requests_waiting{engine="0",model_name="qwen3-1.7b"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="qwen3-1.7b"} 0.37
`;
describe("parseVllmMetrics", () => {
  it("extracts requests running and kv-cache usage from labeled metrics", () => {
    const m = parseVllmMetrics(sample);
    expect(m.numRequestsRunning).toBe(2);
    expect(m.kvCacheUsagePerc).toBeCloseTo(0.37);
  });

  it("handles gpu_cache_usage_perc alias for older vLLM versions", () => {
    const text = `
vllm:num_requests_running{engine="0"} 1.0
vllm:gpu_cache_usage_perc{engine="0"} 0.55
`;
    const m = parseVllmMetrics(text);
    expect(m.numRequestsRunning).toBe(1);
    expect(m.kvCacheUsagePerc).toBeCloseTo(0.55);
  });

  it("returns undefined fields for empty/unmatched text", () => {
    const m = parseVllmMetrics("");
    expect(m.numRequestsRunning).toBeUndefined();
    expect(m.kvCacheUsagePerc).toBeUndefined();
  });

  it("prefers kv_cache_usage_perc over gpu_cache_usage_perc when both present", () => {
    const text = `
vllm:num_requests_running{engine="0"} 3.0
vllm:gpu_cache_usage_perc{engine="0"} 0.10
vllm:kv_cache_usage_perc{engine="0"} 0.20
`;
    const m = parseVllmMetrics(text);
    expect(m.kvCacheUsagePerc).toBeCloseTo(0.20);
  });

  it("extracts the generation_tokens_total counter (for tps)", () => {
    const text = `
vllm:num_requests_running{engine="0"} 1.0
vllm:generation_tokens_total{engine="0",model_name="qwen3-1.7b"} 12345.0
`;
    expect(parseVllmMetrics(text).generationTokensTotal).toBe(12345);
  });

  it("leaves generationTokensTotal undefined when absent", () => {
    expect(parseVllmMetrics(sample).generationTokensTotal).toBeUndefined();
  });
});

// ── computeTps ──────────────────────────────────────────────────────────────

describe("computeTps", () => {
  it("returns null on the first sample (no baseline)", () => {
    expect(computeTps(undefined, { total: 100, time: 1000 })).toBeNull();
  });

  it("computes tokens/sec as counter delta over elapsed seconds", () => {
    // 200 tokens over 2s = 100 tok/s
    expect(computeTps({ total: 100, time: 1000 }, { total: 300, time: 3000 })).toBe(100);
  });

  it("rounds to one decimal place", () => {
    // 100 tokens over 3s = 33.33… → 33.3
    expect(computeTps({ total: 0, time: 0 }, { total: 100, time: 3000 })).toBe(33.3);
  });

  it("returns null when the counter went backwards (container restart reset it)", () => {
    expect(computeTps({ total: 500, time: 1000 }, { total: 10, time: 2000 })).toBeNull();
  });

  it("returns null on a non-positive interval", () => {
    expect(computeTps({ total: 100, time: 2000 }, { total: 300, time: 2000 })).toBeNull();
  });
});

// ── firstErrorLine ──────────────────────────────────────────────────────────

describe("firstErrorLine", () => {
  it("returns the first error-ish line from a multi-line blob", () => {
    const blob = [
      "Starting vllm server...",
      "Loading model weights...",
      "vllm serve: error: argument --compilation-config: Invalid JSON",
      "Traceback (most recent call last):",
    ].join("\n");
    const result = firstErrorLine(blob);
    expect(result).toBe("vllm serve: error: argument --compilation-config: Invalid JSON");
  });

  it("returns undefined when no error-ish line is found", () => {
    const blob = "Starting up\nLoading weights\nAll good";
    expect(firstErrorLine(blob)).toBeUndefined();
  });

  it("matches traceback keyword case-insensitively", () => {
    const blob = "INFO: something\nTraceback (most recent call last):";
    expect(firstErrorLine(blob)).toBe("Traceback (most recent call last):");
  });

  it("returns undefined on empty string", () => {
    expect(firstErrorLine("")).toBeUndefined();
  });
});

// ── checkSparkrunDeployments tests ──────────────────────────────────────────

const { loadDeploymentsMock, isWorkloadRunningMock, inspectSparkrunContainerMock, captureCrashedContainerLogsMock, fetchMock } = vi.hoisted(() => ({
  loadDeploymentsMock: vi.fn(),
  isWorkloadRunningMock: vi.fn(),
  inspectSparkrunContainerMock: vi.fn(),
  captureCrashedContainerLogsMock: vi.fn(),
  fetchMock: vi.fn(),
}));

vi.mock("./deployment-store.js", () => ({ loadDeployments: loadDeploymentsMock }));
vi.mock("./sparkrun.js", () => ({
  isWorkloadRunning: isWorkloadRunningMock,
  inspectSparkrunContainer: inspectSparkrunContainerMock,
  captureCrashedContainerLogs: captureCrashedContainerLogsMock,
}));

// Override global fetch
vi.stubGlobal("fetch", fetchMock);

import { checkSparkrunDeployments } from "./sparkrun-metrics.js";

const metricsBody = `
vllm:num_requests_running{engine="0",model_name="qwen3-1.7b"} 5.0
vllm:kv_cache_usage_perc{engine="0",model_name="qwen3-1.7b"} 0.42
`;

beforeEach(() => {
  loadDeploymentsMock.mockReset();
  isWorkloadRunningMock.mockReset();
  inspectSparkrunContainerMock.mockReset();
  captureCrashedContainerLogsMock.mockReset();
  fetchMock.mockReset();
  // Default: no container found (healthy path — inspect returns null)
  inspectSparkrunContainerMock.mockReturnValue(null);
  captureCrashedContainerLogsMock.mockReturnValue("");
});

describe("checkSparkrunDeployments", () => {
  it("returns a VllmStatus with parsed metrics for a running deployment", async () => {
    loadDeploymentsMock.mockReturnValue([{
      deploymentId: "dep-1",
      recipeFile: "qwen3-1.7b-vllm",
      recipeName: "qwen3-1.7b",
      port: 8000,
      startedAt: "2026-01-01T00:00:00Z",
      clusterNodes: ["10.0.0.1"],
      clusterId: "sparkrun_abc123",
    }]);
    isWorkloadRunningMock.mockReturnValue(true);
    fetchMock.mockResolvedValue({
      ok: true,
      text: async () => metricsBody,
    });

    const results = await checkSparkrunDeployments();
    expect(results).toHaveLength(1);
    const s = results[0];
    expect(s.deploymentId).toBe("dep-1");
    expect(s.alive).toBe(true);
    expect(s.containerRunning).toBe(true);
    expect(s.requestsRunning).toBe(5);
    expect(s.kvCacheUsage).toBeCloseTo(0.42);
    expect(s.port).toBe(8000);
    expect(s.recipeName).toBe("qwen3-1.7b");
  });

  it("returns a not-running VllmStatus when isWorkloadRunning is false", async () => {
    loadDeploymentsMock.mockReturnValue([{
      deploymentId: "dep-2",
      recipeFile: "qwen3-1.7b-vllm",
      recipeName: "qwen3-1.7b",
      port: 8000,
      startedAt: "2026-01-01T00:00:00Z",
      clusterNodes: ["10.0.0.1"],
      clusterId: "sparkrun_dep2abc",
    }]);
    isWorkloadRunningMock.mockReturnValue(false);

    const results = await checkSparkrunDeployments();
    expect(results).toHaveLength(1);
    const s = results[0];
    expect(s.deploymentId).toBe("dep-2");
    expect(s.alive).toBe(false);
    expect(s.containerRunning).toBe(false);
    expect(s.requestsRunning).toBeNull();
    expect(s.kvCacheUsage).toBeNull();
  });

  it("returns not-running when metrics fetch throws (but workload is up)", async () => {
    loadDeploymentsMock.mockReturnValue([{
      deploymentId: "dep-3",
      recipeFile: "qwen3-1.7b-vllm",
      recipeName: "qwen3-1.7b",
      port: 8000,
      startedAt: "2026-01-01T00:00:00Z",
      clusterNodes: ["10.0.0.1"],
      clusterId: "sparkrun_xyz",
    }]);
    isWorkloadRunningMock.mockReturnValue(true);
    fetchMock.mockRejectedValue(new Error("ECONNREFUSED"));

    const results = await checkSparkrunDeployments();
    expect(results).toHaveLength(1);
    const s = results[0];
    // Alive (workload running) but metrics endpoint not yet ready
    expect(s.alive).toBe(true);
    expect(s.containerRunning).toBe(true);
    expect(s.requestsRunning).toBeNull();
  });

  it("returns empty array when no deployments tracked", async () => {
    loadDeploymentsMock.mockReturnValue([]);
    const results = await checkSparkrunDeployments();
    expect(results).toHaveLength(0);
  });

  it("does NOT report a deployment as failed when stopping===true and workload is no longer running", async () => {
    // Invariant: a deployment that cmd:undeploy has marked stopping===true in
    // the store must be excluded from the returned list when the workload is
    // gone — the health loop must never see it as a crash.
    loadDeploymentsMock.mockReturnValue([{
      deploymentId: "dep-stop-1",
      recipeFile: "qwen3-1.7b-vllm",
      recipeName: "qwen3-1.7b",
      port: 8000,
      startedAt: "2026-01-01T00:00:00Z",
      clusterNodes: ["10.0.0.1"],
      clusterId: "sparkrun_abc456",
      stopping: true,
    }]);
    isWorkloadRunningMock.mockReturnValue(false);

    const results = await checkSparkrunDeployments();
    // The stopping-and-gone entry must be absent from the results entirely.
    expect(results).toHaveLength(0);
  });

  it("still reports a stopping deployment that is still running (workload hasn't stopped yet)", async () => {
    // If stopping===true but the workload is still alive, include it so the
    // health loop can continue to report vramActual until it actually stops.
    loadDeploymentsMock.mockReturnValue([{
      deploymentId: "dep-stop-2",
      recipeFile: "qwen3-1.7b-vllm",
      recipeName: "qwen3-1.7b",
      port: 8000,
      startedAt: "2026-01-01T00:00:00Z",
      clusterNodes: ["10.0.0.1"],
      clusterId: "sparkrun_abc789",
      stopping: true,
    }]);
    isWorkloadRunningMock.mockReturnValue(true);
    fetchMock.mockRejectedValue(new Error("ECONNREFUSED"));

    const results = await checkSparkrunDeployments();
    expect(results).toHaveLength(1);
    expect(results[0].deploymentId).toBe("dep-stop-2");
    expect(results[0].containerRunning).toBe(true);
  });

  it("skips a deployment that has no clusterId yet (still launching / downloading)", async () => {
    // Invariant: a deployment without a clusterId has not yet launched a container
    // (the model may still be downloading). The health loop must NOT treat it as a
    // dead container — doing so would call stopSparkrun and kill the download.
    loadDeploymentsMock.mockReturnValue([{
      deploymentId: "d1",
      recipeFile: "fp8-model-vllm",
      recipeName: "fp8-model",
      port: 8000,
      /* no clusterId */
    }]);

    const results = await checkSparkrunDeployments();
    expect(results).toHaveLength(0);
    // isWorkloadRunning and inspectSparkrunContainer must NOT have been called —
    // the guard continues before they are reached.
    expect(isWorkloadRunningMock).not.toHaveBeenCalled();
    expect(inspectSparkrunContainerMock).not.toHaveBeenCalled();
  });

  it("returns crashed status with capturedLog when container is crash-looping", async () => {
    // Invariant: when a sparkrun container has restarted >= CRASH_LOOP_THRESHOLD
    // times, checkSparkrunDeployments must surface the real root error from container
    // logs (not just "not running"). captureCrashedContainerLogs (which stops the
    // restart loop before reading) is used instead of snapshotContainerLogs so that
    // the full accumulated log (run #0 included) is returned reliably.
    const crashLog = [
      "INFO: Starting vllm server...",
      "vllm serve: error: argument --compilation-config: Invalid JSON",
      "Process exited with code 1",
    ].join("\n");

    loadDeploymentsMock.mockReturnValue([{
      deploymentId: "dep-crash-1",
      recipeFile: "qwen3-1.7b-vllm",
      recipeName: "qwen3-1.7b",
      port: 8000,
      startedAt: "2026-01-01T00:00:00Z",
      clusterNodes: ["10.0.0.1"],
      clusterId: "sparkrun_crash01",
    }]);
    isWorkloadRunningMock.mockReturnValue(false);
    inspectSparkrunContainerMock.mockReturnValue({ name: "sparkrun_crash01_solo", state: "restarting", restartCount: 5 });
    captureCrashedContainerLogsMock.mockReturnValue(crashLog);

    const results = await checkSparkrunDeployments();
    expect(results).toHaveLength(1);
    const s = results[0];
    expect(s.deploymentId).toBe("dep-crash-1");
    expect(s.alive).toBe(false);
    expect(s.containerRunning).toBe(false);
    expect(s.crashLoop).toBe(true);
    expect(s.restartCount).toBe(5);
    expect(s.capturedLog).toContain("vllm serve: error: argument --compilation-config: Invalid JSON");
    expect(s.error).toBe("vllm serve: error: argument --compilation-config: Invalid JSON");
  });

  it("reports tps from the generation_tokens counter rate across two scrapes", async () => {
    const dep = {
      deploymentId: "dep-tps",
      recipeFile: "qwen3-1.7b-vllm",
      recipeName: "qwen3-1.7b",
      port: 8000,
      startedAt: "2026-01-01T00:00:00Z",
      clusterNodes: ["10.0.0.1"],
      clusterId: "sparkrun_tps01",
    };
    loadDeploymentsMock.mockReturnValue([dep]);
    isWorkloadRunningMock.mockReturnValue(true);
    const body = (gen: number) => `vllm:num_requests_running{engine="0"} 1.0\nvllm:generation_tokens_total{engine="0"} ${gen}.0\n`;

    const nowSpy = vi.spyOn(Date, "now");

    // First scrape: counter=1000 at t=10s → no baseline yet → tps null.
    nowSpy.mockReturnValue(10_000);
    fetchMock.mockResolvedValueOnce({ ok: true, text: async () => body(1000) });
    let results = await checkSparkrunDeployments();
    expect(results[0].tps).toBeNull();

    // Second scrape: counter=1300 at t=12s → (1300-1000)/2s = 150 tok/s.
    nowSpy.mockReturnValue(12_000);
    fetchMock.mockResolvedValueOnce({ ok: true, text: async () => body(1300) });
    results = await checkSparkrunDeployments();
    expect(results[0].tps).toBe(150);

    nowSpy.mockRestore();
  });

  it("does NOT flag a healthy loading container (state=running, restartCount=0)", async () => {
    // A container that is still starting up has state="running" + restartCount=0.
    // It must NOT be flagged as failing — the health loop should see it as alive.
    loadDeploymentsMock.mockReturnValue([{
      deploymentId: "dep-loading-1",
      recipeFile: "qwen3-1.7b-vllm",
      recipeName: "qwen3-1.7b",
      port: 8000,
      startedAt: "2026-01-01T00:00:00Z",
      clusterNodes: ["10.0.0.1"],
      clusterId: "sparkrun_load01",
    }]);
    isWorkloadRunningMock.mockReturnValue(true);
    inspectSparkrunContainerMock.mockReturnValue({ name: "sparkrun_load01_solo", state: "running", restartCount: 0 });
    fetchMock.mockRejectedValue(new Error("ECONNREFUSED"));

    const results = await checkSparkrunDeployments();
    expect(results).toHaveLength(1);
    const s = results[0];
    expect(s.alive).toBe(true);
    expect(s.containerRunning).toBe(true);
    expect(s.crashLoop).toBeUndefined();
  });
});
