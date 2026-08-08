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

  // The manager validates the SHAPE of a mod name (it becomes a bind-mount
  // source, so a path segment that escapes the mods dir must never reach a
  // node); whether that mod is actually installed is the agent's check, because
  // only the node knows which agent bundle it is running.
  describe("mods", () => {
    const withMods = (mods: string) =>
      resolveDgxrunRecipe(
        `runner: dgxrun\ncontainer: img:tag\ncommand: vllm serve x\nmods:\n${mods}`,
      );

    it("carries a list of mod names through to the resolved recipe", () => {
      const r = withMods("  - instanttensor-hybrid-draft-loader\n  - drop-caches\n");
      expect(r.error).toBeUndefined();
      expect(r.recipe?.mods).toEqual(["instanttensor-hybrid-draft-loader", "drop-caches"]);
    });

    it("leaves mods undefined when the recipe declares none", () => {
      const r = resolveDgxrunRecipe("runner: dgxrun\ncontainer: img:tag\ncommand: vllm serve x\n");
      expect(r.recipe?.mods).toBeUndefined();
    });

    it("rejects a mod name that is not a single path segment", () => {
      for (const bad of ["  - ../escape\n", "  - nested/name\n", "  - /absolute\n", '  - ""\n']) {
        const r = withMods(bad);
        expect(r.isDgxrun).toBe(true);
        expect(r.error).toMatch(/mod name/i);
        expect(r.recipe).toBeUndefined();
      }
    });

    it("rejects a mods block that is not a list of strings", () => {
      expect(withMods("  key: value\n").error).toMatch(/mods/i);
    });
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
