import { describe, it, expect } from "vitest";
import { resolveRecipePath } from "./recipe-path.js";

const ROOT = "/mnt/tank";

describe("resolveRecipePath", () => {
  it("accepts an in-tree relative path", () => {
    expect(resolveRecipePath("recipes/dev/my.yaml", ROOT)).toBe("/mnt/tank/recipes/dev/my.yaml");
  });
  it("rejects parent-traversal", () => {
    expect(() => resolveRecipePath("../etc/passwd", ROOT)).toThrow(/outside shared storage/i);
  });
  it("rejects absolute escape", () => {
    expect(() => resolveRecipePath("/etc/passwd", ROOT)).toThrow(/outside shared storage/i);
  });
  it("rejects sneaky traversal that re-enters", () => {
    expect(() => resolveRecipePath("recipes/../../etc/passwd", ROOT)).toThrow(/outside shared storage/i);
  });
});
