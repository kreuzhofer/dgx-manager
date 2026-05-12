import { describe, expect, it } from "vitest";
import { ollamaVramEstimateMB } from "./vram-estimate.js";

describe("ollamaVramEstimateMB", () => {
  it.each([
    ["8b", 4506],     // 8 × 0.55 GB × 1024 ≈ 4506 MB
    ["70b", 39424],   // 70 × 0.55 GB × 1024
    ["405b", 228096], // 405 × 0.55 GB × 1024
    ["1.5b", 845],    // 1.5 × 0.55 GB × 1024
    ["270m", 152],    // 0.27 × 0.55 GB × 1024 = 152.064 ≈ 152 MB
    ["e2b", 1126],    // Gemma "effective 2b" — parsed as 2b
    ["e4b", 2253],
  ])("parses %s as ~%d MB", (size, expected) => {
    const got = ollamaVramEstimateMB(size);
    // Allow ±2 MB for rounding drift.
    expect(got).toBeGreaterThanOrEqual(expected - 2);
    expect(got).toBeLessThanOrEqual(expected + 2);
  });

  it.each(["", "huge", "12.34xyz", "GB", null as unknown as string, undefined as unknown as string])(
    "returns null for unparseable input %p",
    (s) => {
      expect(ollamaVramEstimateMB(s)).toBeNull();
    },
  );
});
