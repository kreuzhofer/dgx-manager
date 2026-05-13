import { describe, expect, it } from "vitest";
import { detectQuantizeProgress } from "./finetune-quantize.js";

describe("detectQuantizeProgress", () => {
  it("emits load progress for 'Loading model'", () => {
    expect(detectQuantizeProgress("[quantize_fp8] Loading model from /foo"))
      .toEqual({ phase: "loading", progress: 0.1 });
  });
  it("emits quantize progress for 'Applying FP8_DYNAMIC'", () => {
    expect(detectQuantizeProgress("[quantize_fp8] Applying FP8_DYNAMIC W8A8 quantization"))
      .toEqual({ phase: "quantizing", progress: 0.5 });
  });
  it("emits saving progress for 'Saving FP8 model'", () => {
    expect(detectQuantizeProgress("[quantize_fp8] Saving FP8 model to /foo"))
      .toEqual({ phase: "saving", progress: 0.85 });
  });
  it("emits final progress for 'OK'", () => {
    expect(detectQuantizeProgress("[quantize_fp8] OK"))
      .toEqual({ phase: "saving", progress: 1.0 });
  });
  it("returns null for unrelated lines", () => {
    expect(detectQuantizeProgress("some unrelated log line")).toBeNull();
    expect(detectQuantizeProgress("")).toBeNull();
  });
});
