import { describe, it, expect } from "vitest";
import { verifyExtractedBundle, healthCheckPasses } from "./updater.js";

describe("verifyExtractedBundle", () => {
  it("ok when package.json version matches", () => {
    const r = verifyExtractedBundle("/opt/dgx-agent-new", "0.5.720", () => JSON.stringify({ version: "0.5.720" }));
    expect(r.ok).toBe(true);
  });
  it("fails on version mismatch", () => {
    const r = verifyExtractedBundle("/d", "0.5.720", () => JSON.stringify({ version: "0.5.719" }));
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/version/i);
  });
  it("fails when package.json unreadable/unparseable", () => {
    expect(verifyExtractedBundle("/d", "0.5.720", () => { throw new Error("ENOENT"); }).ok).toBe(false);
    expect(verifyExtractedBundle("/d", "0.5.720", () => "not json").ok).toBe(false);
  });
});

describe("healthCheckPasses", () => {
  it("true when marker written after restart", () => {
    expect(healthCheckPasses(2000, 1000, 90000)).toBe(true);
  });
  it("false when marker missing", () => {
    expect(healthCheckPasses(null, 1000, 90000)).toBe(false);
  });
  it("false when marker is stale (pre-restart)", () => {
    expect(healthCheckPasses(500, 1000, 90000)).toBe(false);
  });
});
