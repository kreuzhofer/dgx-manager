import { describe, it, expect } from "vitest";
import { validateRegistry } from "./validate.js";

const ok = { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes" };

describe("validateRegistry", () => {
  it("accepts a minimal valid registry", () => {
    const r = validateRegistry(ok);
    expect(r.ok).toBe(true);
  });

  it.each([
    ["bad name (uppercase)", { ...ok, name: "RTX" }],
    ["bad name (space)", { ...ok, name: "my reg" }],
    ["empty name", { ...ok, name: "" }],
    ["bad url scheme", { ...ok, url: "file:///etc/passwd" }],
    ["empty subpath", { ...ok, subpath: "" }],
    ["path traversal subpath", { ...ok, subpath: "../secrets" }],
    ["leading slash subpath", { ...ok, subpath: "/recipes" }],
    ["path traversal tuningSubpath", { ...ok, tuningSubpath: "../secret" }],
    ["leading slash benchmarkSubpath", { ...ok, benchmarkSubpath: "/absolute" }],
    ["path traversal modsSubpath", { ...ok, modsSubpath: "../x" }],
    ["invalid visible type (string)", { ...ok, visible: "true" }],
    ["invalid sortOrder type (string)", { ...ok, sortOrder: "0" }],
    ["non-object", null],
  ])("rejects %s", (_label, input) => {
    const r = validateRegistry(input);
    expect(r.ok).toBe(false);
  });

  it("accepts scp-style git url", () => {
    expect(validateRegistry({ ...ok, url: "git@github.com:org/repo.git" }).ok).toBe(true);
  });

  it("applies default normalization on success", () => {
    const r = validateRegistry(ok);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.value.visible).toBe(true);
      expect(r.value.sortOrder).toBe(0);
      expect(r.value.tuningSubpath).toBe(null);
      expect(r.value.benchmarkSubpath).toBe(null);
      expect(r.value.modsSubpath).toBe(null);
      expect(r.value.description).toBe(null);
    }
  });
});
