import { describe, it, expect } from "vitest";
import { detectPhase, phaseRank, isForwardPhase } from "./deploy-phase.js";

describe("detectPhase — real GLM-5.2 DCP deploy log lines", () => {
  const cases: Array<[string, string | null]> = [
    ["INFO [model_runner.py:312] Loading model from scratch...", "loading"],
    ["Loading safetensors checkpoint shards:   0% Completed | 0/128", "loading"],
    ["INFO [default_loader.py:451] Loading weights took 765.04 seconds", "loading"],
    // The line that caused the bug: a page-cache prefetch AFTER loading, which
    // matches the download heuristic. Classifier still says "downloading" — the
    // forward guard (below) is what stops it from regressing the status.
    ["INFO [weight_utils.py:872] Prefetching checkpoint files into page cache started", "downloading"],
    ["INFO [backends.py:1089] Using cache directory: /root/glm-jit/vllm/torch_compile_cache/6791", "compiling"],
    ["INFO [backends.py:393] Compiling a graph for compile range (1, 2048) takes 20.41 s", "compiling"],
    ["INFO [monitor.py:53] torch.compile took 40.63 s in total", "compiling"],
    ["Capturing CUDA graphs (PIECEWISE):   0%|          | 0/2", "compiling"],
    ["INFO [model_runner.py:766] Graph capturing finished in 9 secs, took 1.09 GiB", "compiling"],
    ["INFO [speculator.py:107] Capturing model for speculator...", "compiling"],
    ["INFO:     Application startup complete.", "running"],
    ["=== downloading model weights ===", "downloading"],
    ["Fetching 5 files: 100%", "downloading"],
    ["Starting ray worker node", "launching"],
    ["INFO some neutral progress line with no phase", null],
  ];
  for (const [line, expected] of cases) {
    it(`${expected ?? "null"} <- ${line.slice(0, 48)}`, () => {
      expect(detectPhase(line)).toBe(expected);
    });
  }
});

describe("phase ordering", () => {
  it("ranks the lifecycle forward", () => {
    expect(phaseRank("starting")).toBeLessThan(phaseRank("downloading"));
    expect(phaseRank("downloading")).toBeLessThan(phaseRank("loading"));
    expect(phaseRank("loading")).toBeLessThan(phaseRank("compiling"));
    expect(phaseRank("compiling")).toBeLessThan(phaseRank("running"));
  });
  it("gives non-progression statuses rank -1", () => {
    expect(phaseRank("failed")).toBe(-1);
    expect(phaseRank("pending")).toBe(-1);
  });
});

describe("isForwardPhase — the monotonic guard", () => {
  // The core fix: a late 'downloading' (the prefetch line) after 'loading' must NOT advance.
  it("blocks the loading -> downloading regression", () => {
    expect(isForwardPhase("loading", "downloading")).toBe(false);
  });
  it("allows loading -> compiling -> running", () => {
    expect(isForwardPhase("loading", "compiling")).toBe(true);
    expect(isForwardPhase("compiling", "running")).toBe(true);
  });
  it("blocks any backward move (running -> loading, compiling -> starting)", () => {
    expect(isForwardPhase("running", "loading")).toBe(false);
    expect(isForwardPhase("compiling", "starting")).toBe(false);
  });
  it("blocks reconnect 'starting' after progress", () => {
    expect(isForwardPhase("loading", "starting")).toBe(false);
  });
  it("advances from the initial starting", () => {
    expect(isForwardPhase("starting", "loading")).toBe(true);
  });
  it("treats unknown target phases as non-advancing", () => {
    expect(isForwardPhase("loading", "failed")).toBe(false);
  });
});
