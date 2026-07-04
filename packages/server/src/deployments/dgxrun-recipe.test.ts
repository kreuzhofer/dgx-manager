import { describe, it, expect } from "vitest";
import { resolveDgxrunRecipe } from "./dgxrun-recipe.js";

describe("resolveDgxrunRecipe", () => {
  it("returns isDgxrun:false for a plain sparkrun recipe (no runner)", () => {
    const r = resolveDgxrunRecipe("model: org/m\nruntime: vllm\ncommand: vllm serve x\n");
    expect(r.isDgxrun).toBe(false);
    expect(r.recipe).toBeUndefined();
  });

  it("returns isDgxrun:false for a runner other than dgxrun", () => {
    expect(resolveDgxrunRecipe("runner: sparkrun\ncommand: x\ncontainer: y\n").isDgxrun).toBe(false);
  });

  it("returns isDgxrun:false (not a throw) on malformed YAML", () => {
    expect(resolveDgxrunRecipe("::: not: valid: yaml: [").isDgxrun).toBe(false);
  });

  it("resolves a full dgxrun recipe with env + defaults", () => {
    const yaml = [
      "runner: dgxrun",
      "model: org/glm",
      "container: my-image:tag",
      "cluster_only: true",
      "defaults:",
      "  port: 8000",
      "  tensor_parallel: 4",
      "  gpu_memory_utilization: 0.88",
      "env:",
      "  NCCL_NET: IB",
      "  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: \"5400\"",
      "command: |",
      "  vllm serve {model} -tp {tensor_parallel} --port {port}",
    ].join("\n");
    const r = resolveDgxrunRecipe(yaml);
    expect(r.isDgxrun).toBe(true);
    expect(r.error).toBeUndefined();
    expect(r.recipe?.container).toBe("my-image:tag");
    expect(r.recipe?.model).toBe("org/glm");
    expect(r.recipe?.env?.NCCL_NET).toBe("IB");
    expect(r.recipe?.env?.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS).toBe("5400");
    expect(r.recipe?.defaults?.tensor_parallel).toBe(4);
    expect(r.recipe?.command).toContain("vllm serve {model}");
    expect(r.recipe?.cluster_only).toBe(true);
  });

  it("flags a dgxrun recipe missing container/command with an error", () => {
    expect(resolveDgxrunRecipe("runner: dgxrun\ncommand: x\n").error).toMatch(/container/i);
    expect(resolveDgxrunRecipe("runner: dgxrun\ncontainer: y\n").error).toMatch(/command/i);
  });
});
