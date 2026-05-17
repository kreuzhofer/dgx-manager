import { describe, it, expect } from "vitest";
import { it as fcIt } from "@fast-check/vitest";
import { fc } from "@fast-check/vitest";
import { normalizeDisplayName, DisplayNameError } from "./display-name.js";

describe("normalizeDisplayName", () => {
  it("returns null for null/undefined input", () => {
    expect(normalizeDisplayName(null)).toBeNull();
    expect(normalizeDisplayName(undefined)).toBeNull();
  });

  it("returns null for empty / whitespace-only strings (treats them as unset)", () => {
    expect(normalizeDisplayName("")).toBeNull();
    expect(normalizeDisplayName("   ")).toBeNull();
    expect(normalizeDisplayName("\t\n")).toBeNull();
  });

  it("trims surrounding whitespace", () => {
    expect(normalizeDisplayName("  chat3d-prod  ")).toBe("chat3d-prod");
  });

  it("accepts URL-safe characters: letters, digits, dot, dash, underscore, colon", () => {
    expect(normalizeDisplayName("Chat3D_v1.0:fp8-prod")).toBe("Chat3D_v1.0:fp8-prod");
  });

  it("rejects strings containing whitespace (would break OpenAI model field)", () => {
    expect(() => normalizeDisplayName("chat 3d")).toThrow(DisplayNameError);
  });

  it("rejects strings containing slashes (would break loadbalancer routing)", () => {
    expect(() => normalizeDisplayName("vendor/chat3d")).toThrow(DisplayNameError);
  });

  it("rejects strings containing other special characters", () => {
    expect(() => normalizeDisplayName("chat@3d")).toThrow(DisplayNameError);
    expect(() => normalizeDisplayName("chat#3d")).toThrow(DisplayNameError);
  });

  it("rejects strings longer than 128 chars", () => {
    const tooLong = "a".repeat(129);
    expect(() => normalizeDisplayName(tooLong)).toThrow(DisplayNameError);
  });

  it("accepts exactly 128 chars", () => {
    const max = "a".repeat(128);
    expect(normalizeDisplayName(max)).toBe(max);
  });

  /**
   * Invariant: for any string drawn from the URL-safe character set, the
   * normalizer is idempotent — calling it twice produces the same result.
   */
  fcIt.prop([
    fc.stringMatching(/^[A-Za-z0-9._:-]{1,128}$/),
  ])("is idempotent on already-normalized inputs", (s) => {
    const once = normalizeDisplayName(s);
    const twice = normalizeDisplayName(once);
    expect(twice).toBe(once);
  });
});
