import { describe, it, expect, vi } from "vitest";
vi.mock("node:child_process", () => ({
  execFileSync: vi.fn(() => JSON.stringify([
    { name: "@reg/qwen3-1.7b-vllm", file: "qwen3-1.7b-vllm", model: "Qwen/Qwen3-1.7B",
      description: "", runtime: "vllm-distributed", min_nodes: 1, tp: 1, gpu_mem: 0.3, registry: "reg" },
    { name: "@reg/big", file: "big", model: "X", runtime: "vllm", min_nodes: 2, tp: 2, gpu_mem: "", registry: "reg" },
  ])),
}));
import { discoverRecipes, toRecipe } from "./recipes.js";
import type { SparkrunRecipeSummary } from "./runtime/sparkrun-parse.js";

describe("toRecipe", () => {
  it("tags an arm64 (official) registry ref with arch=arm64", () => {
    const summary: SparkrunRecipeSummary = {
      ref: "@official/qwen3.6-27b-fp8-vllm",
      name: "Qwen3.6 27B FP8",
      description: "Qwen recipe",
      runtime: "vllm",
      registry: "official",
      model: "Qwen/Qwen3.6-27B-FP8",
      minNodes: 1,
      tpDefault: 1,
      gpuMemDefault: 0.85,
    };
    expect(toRecipe(summary).arch).toBe("arm64");
  });
});

describe("discoverRecipes", () => {
  it("maps sparkrun summaries to the wire Recipe shape", () => {
    const r = discoverRecipes();
    expect(r).toHaveLength(2);
    expect(r[0].file).toBe("@reg/qwen3-1.7b-vllm");
    expect(r[0].defaults.tensor_parallel).toBe(1);
    expect(r[0].defaults.gpu_memory_utilization).toBe(0.3);
    expect(r[0].cluster_only).toBeUndefined();
    expect(r[1].cluster_only).toBe(true);
    expect(r[1].defaults.gpu_memory_utilization).toBe(0.85);
    // TP must be preserved for multi-node recipes; pipeline_parallel must NOT be
    // synthesised (sparkrun list --json never emits it, so we must not invent one —
    // the dashboard dropdown shows TP=N only when pipeline_parallel is absent).
    expect(r[1].defaults.tensor_parallel).toBe(2);
    expect(r[1].defaults.pipeline_parallel).toBeUndefined();
  });
});
