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
    ["non-object", null],
  ])("rejects %s", (_label, input) => {
    const r = validateRegistry(input);
    expect(r.ok).toBe(false);
  });

  it("accepts scp-style git url", () => {
    expect(validateRegistry({ ...ok, url: "git@github.com:org/repo.git" }).ok).toBe(true);
  });
});
