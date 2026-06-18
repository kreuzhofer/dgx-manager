import { describe, it, expect } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { deriveRecipeArch, type RecipeArch } from "./recipe-arch.js";

describe("deriveRecipeArch", () => {
  it("maps an amd64 (@rtx) registry ref to amd64", () => {
    expect(deriveRecipeArch("@rtx/qwen3-1.7b-vllm")).toBe("amd64");
  });

  it("maps the forward-compat @dgx-amd64 registry ref to amd64", () => {
    expect(deriveRecipeArch("@dgx-amd64/some-recipe")).toBe("amd64");
  });

  it("maps an ollama: ref to any (arch-agnostic)", () => {
    expect(deriveRecipeArch("ollama:qwen3:8b")).toBe("any");
  });

  it("maps a transitional/official arm64 registry ref to arm64", () => {
    expect(deriveRecipeArch("@sparkrun-transitional/qwen3-1.7b-vllm")).toBe("arm64");
    expect(deriveRecipeArch("@official/qwen3.6-27b-fp8-vllm")).toBe("arm64");
  });

  /** Invariant: the result is always one of the three arch literals. */
  test.prop([fc.string()])("always returns one of the three arch literals", (ref) => {
    const valid: RecipeArch[] = ["amd64", "arm64", "any"];
    expect(valid).toContain(deriveRecipeArch(ref));
  });
});
