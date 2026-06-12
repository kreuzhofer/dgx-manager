import { describe, it, expect } from "vitest";
import { validateInlineRecipe, MAX_INLINE_RECIPE_BYTES } from "./recipe-inline.js";

describe("validateInlineRecipe", () => {
  it("accepts a plausible sparkrun recipe", () => {
    expect(() => validateInlineRecipe("model: Qwen/Qwen3-1.7B\nruntime: vllm\n")).not.toThrow();
  });
  it("rejects empty / whitespace", () => {
    expect(() => validateInlineRecipe("   ")).toThrow(/empty/i);
  });
  it("rejects content that doesn't look like a recipe", () => {
    expect(() => validateInlineRecipe("hello world")).toThrow(/recipe/i);
  });
  it("rejects oversized input", () => {
    expect(() => validateInlineRecipe("model:\n" + "x".repeat(MAX_INLINE_RECIPE_BYTES))).toThrow(/too large/i);
  });
});
