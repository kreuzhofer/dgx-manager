import { describe, it, expect } from "vitest";
import { it as itProp, fc } from "@fast-check/vitest";
import { validateInlineRecipe, MAX_INLINE_RECIPE_BYTES, parseInlineRecipeModel } from "./recipe-inline.js";

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

describe("parseInlineRecipeModel", () => {
  it("extracts the top-level HF model id", () => {
    expect(parseInlineRecipeModel("model: google/gemma-4-12B-it-qat-w4a16-ct\nruntime: vllm\n"))
      .toEqual({ model: "google/gemma-4-12B-it-qat-w4a16-ct" });
  });

  it("extracts served_model_name nested under defaults: (v2 recipe)", () => {
    const yaml = [
      "model: Intel/Qwen3.5-122B-A10B-int4-AutoRound",
      "runtime: vllm",
      "defaults:",
      "  tensor_parallel: 4",
      "  served_model_name: qwen",
    ].join("\n");
    expect(parseInlineRecipeModel(yaml)).toEqual({
      model: "Intel/Qwen3.5-122B-A10B-int4-AutoRound",
      servedModelName: "qwen",
    });
  });

  it("does not mistake the served_model_name line for the top-level model", () => {
    // No top-level model:, only a nested served_model_name: — model must stay absent
    const yaml = "command: vllm serve foo\ndefaults:\n  served_model_name: alias-only\n";
    expect(parseInlineRecipeModel(yaml)).toEqual({ servedModelName: "alias-only" });
  });

  it("strips surrounding quotes and trailing inline comments", () => {
    expect(parseInlineRecipeModel('model: "org/Model-X"  # the base\n'))
      .toEqual({ model: "org/Model-X" });
    expect(parseInlineRecipeModel("model: org/Model-Y # quant\nruntime: vllm\n"))
      .toEqual({ model: "org/Model-Y" });
  });

  it("returns {} for malformed / model-less YAML (caller falls back to inline-<ts>)", () => {
    expect(parseInlineRecipeModel("command: echo hi\nruntime: vllm\n")).toEqual({});
    expect(parseInlineRecipeModel("")).toEqual({});
    expect(parseInlineRecipeModel("not yaml at all")).toEqual({});
  });

  // Invariant: whatever model id we put on a `model:` line at column 0 is the
  // exact string we get back — no truncation, no whitespace leakage — for the
  // unquoted, comment-free model ids that real recipes use.
  itProp.prop([
    fc.stringMatching(/^[A-Za-z0-9][A-Za-z0-9._/-]{0,80}$/),
  ])("round-trips an unquoted model id verbatim", (id) => {
    const parsed = parseInlineRecipeModel(`model: ${id}\nruntime: vllm\n`);
    expect(parsed.model).toBe(id);
  });
});
