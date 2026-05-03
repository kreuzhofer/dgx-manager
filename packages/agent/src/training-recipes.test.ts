/**
 * Unit tests for the agent's tiny YAML parser used to load training recipes.
 *
 * The parser is deliberately minimal — it only handles the shapes that real
 * recipe files use (top-level scalars + one level of nested map). These
 * tests pin the exact behaviour the recipe loader depends on so we notice
 * if the parser is "fixed" in a way that breaks recipes.
 *
 * Pattern this file establishes for the repo:
 *   - Tests live next to source as `<name>.test.ts`.
 *   - Imports use the package's own `.js`-suffixed module specifiers.
 *   - One `describe` per function under test.
 */
import { describe, it, expect } from "vitest";
import { parseRecipeYaml } from "./training-recipes.js";

describe("parseRecipeYaml", () => {
  it("parses top-level scalars", () => {
    const result = parseRecipeYaml(`
recipe_version: "1"
name: Qwen3.6-35B-A3B-Base-LoRA
framework: deepspeed
`);
    expect(result.recipe_version).toBe("1");
    expect(result.name).toBe("Qwen3.6-35B-A3B-Base-LoRA");
    expect(result.framework).toBe("deepspeed");
  });

  it("parses a nested map (scripts: ...)", () => {
    const result = parseRecipeYaml(`
scripts:
  entrypoint: entrypoint.sh
  train: train.py
  launch: launch.sh
  ds_config: ds_config.json
  merge: scripts/merge_qwen3moe.py
`);
    expect(result.scripts).toEqual({
      entrypoint: "entrypoint.sh",
      train: "train.py",
      launch: "launch.sh",
      ds_config: "ds_config.json",
      merge: "scripts/merge_qwen3moe.py",
    });
  });

  it("strips end-of-line comments and tolerates blank lines", () => {
    const result = parseRecipeYaml(`
# leading comment
name: foo  # trailing

scripts:
  # nested comment
  train: train.py
  launch: launch.sh
`);
    expect(result.name).toBe("foo");
    expect(result.scripts).toEqual({ train: "train.py", launch: "launch.sh" });
  });

  it("coerces numeric and boolean scalars", () => {
    const result = parseRecipeYaml(`
defaults:
  batch_size: 1
  learning_rate: 0.0002
  packing: false
  use_wandb: true
`);
    expect(result.defaults).toEqual({
      batch_size: 1,
      learning_rate: 0.0002,
      packing: false,
      use_wandb: true,
    });
  });

  it("strips matching surrounding quotes from string scalars", () => {
    const result = parseRecipeYaml(`
recipe_version: "1"
description: 'single quoted'
`);
    expect(result.recipe_version).toBe("1");
    expect(result.description).toBe("single quoted");
  });

  it("returns nested map under its key, not flattened", () => {
    // Regression guard: an earlier draft of the parser leaked nested keys
    // up to the top level when indentation handling was wrong, which would
    // silently break recipes that have both top-level and nested fields.
    const result = parseRecipeYaml(`
name: foo
scripts:
  train: train.py
`);
    expect(result.name).toBe("foo");
    expect(result.scripts).toEqual({ train: "train.py" });
    expect(result.train).toBeUndefined();
  });
});
