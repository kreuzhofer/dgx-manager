import { describe, it, expect, vi, beforeEach } from "vitest";
import { parseVllmMetrics, firstErrorLine } from "./sparkrun-metrics.js";

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

const { loadDeploymentsMock, isWorkloadRunningMock, inspectSparkrunContainerMock, snapshotContainerLogsMock, fetchMock } = vi.hoisted(() => ({
  loadDeploymentsMock: vi.fn(),
  isWorkloadRunningMock: vi.fn(),
  inspectSparkrunContainerMock: vi.fn(),
  snapshotContainerLogsMock: vi.fn(),
  fetchMock: vi.fn(),
}));

vi.mock("./deployment-store.js", () => ({ loadDeployments: loadDeploymentsMock }));
vi.mock("./sparkrun.js", () => ({
  isWorkloadRunning: isWorkloadRunningMock,
  inspectSparkrunContainer: inspectSparkrunContainerMock,
  snapshotContainerLogs: snapshotContainerLogsMock,
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
  snapshotContainerLogsMock.mockReset();
  fetchMock.mockReset();
  // Default: no container found (healthy path — inspect returns null)
  inspectSparkrunContainerMock.mockReturnValue(null);
  snapshotContainerLogsMock.mockReturnValue("");
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

  it("returns crashed status with capturedLog when container is crash-looping", async () => {
    // Invariant: when a sparkrun container has restarted >= CRASH_LOOP_THRESHOLD
    // times, checkSparkrunDeployments must surface the real error from container
    // logs (not just "not running") so the deployment logstream shows the cause.
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
    snapshotContainerLogsMock.mockReturnValue(crashLog);

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
