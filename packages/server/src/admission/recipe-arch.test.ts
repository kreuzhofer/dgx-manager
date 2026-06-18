/**
 * Property + unit tests for the pure recipe/node arch-admission decision.
 */
import { describe, it, expect } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { checkRecipeArchAdmission, recipeArchMismatchMessage } from "./recipe-arch.js";
import type { RecipeArch } from "../ws/agent-hub.js";

const archArb = fc.constantFrom<RecipeArch>("amd64", "arm64", "any");

describe("checkRecipeArchAdmission", () => {
  /** An arch-agnostic ("any") recipe is always admitted, on any node. */
  test.prop([archArb])("'any' recipe is always allowed", (nodeArch) => {
    expect(checkRecipeArchAdmission("any", nodeArch)).toBe(true);
  });

  /** Matching arches are always admitted. */
  test.prop([archArb])("matching arch is always allowed", (arch) => {
    expect(checkRecipeArchAdmission(arch, arch)).toBe(true);
  });

  /**
   * The decision is exactly: allowed iff recipe is "any" or arches match.
   * Anything else is rejected — total over every (recipe, node) pair.
   */
  test.prop([archArb, archArb])("total: allowed iff any-or-equal", (recipeArch, nodeArch) => {
    const expected = recipeArch === "any" || recipeArch === nodeArch;
    expect(checkRecipeArchAdmission(recipeArch, nodeArch)).toBe(expected);
  });

  it("rejects a concrete cross-arch deploy (arm64 recipe → amd64 node)", () => {
    expect(checkRecipeArchAdmission("arm64", "amd64")).toBe(false);
    expect(checkRecipeArchAdmission("amd64", "arm64")).toBe(false);
  });
});

describe("recipeArchMismatchMessage", () => {
  it("names both arches", () => {
    const msg = recipeArchMismatchMessage("arm64", "amd64");
    expect(msg).toContain("arm64");
    expect(msg).toContain("amd64");
    expect(msg.toLowerCase()).toContain("mismatch");
  });
});
