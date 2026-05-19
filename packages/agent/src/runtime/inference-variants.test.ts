import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import {
  inferenceVariantIdFromFilename,
  inferenceFilenameForId,
} from "./inference-template.js";

describe("inferenceVariantIdFromFilename", () => {
  it("maps the bare inference.yaml to the special id 'default'", () => {
    expect(inferenceVariantIdFromFilename("inference.yaml")).toBe("default");
  });

  it("strips the inference- prefix and .yaml suffix", () => {
    expect(inferenceVariantIdFromFilename("inference-fp8.yaml")).toBe("fp8");
    expect(inferenceVariantIdFromFilename("inference-int4.yaml")).toBe("int4");
    expect(inferenceVariantIdFromFilename("inference-low-ctx.yaml")).toBe("low-ctx");
  });

  it("returns null for filenames that don't fit the convention", () => {
    expect(inferenceVariantIdFromFilename("recipe.yaml")).toBeNull();
    expect(inferenceVariantIdFromFilename("inference.yml")).toBeNull(); // .yml not .yaml
    expect(inferenceVariantIdFromFilename("not-inference-fp8.yaml")).toBeNull();
    expect(inferenceVariantIdFromFilename("inference-.yaml")).toBeNull(); // empty slug
  });
});

describe("inferenceFilenameForId", () => {
  it("maps 'default' back to inference.yaml", () => {
    expect(inferenceFilenameForId("default")).toBe("inference.yaml");
  });

  it("maps legacy 'bf16' back to inference.yaml for back-compat", () => {
    // Saved deployments before this feature stored "bf16" — keep them working.
    expect(inferenceFilenameForId("bf16")).toBe("inference.yaml");
  });

  it("maps any other id to inference-<id>.yaml", () => {
    expect(inferenceFilenameForId("fp8")).toBe("inference-fp8.yaml");
    expect(inferenceFilenameForId("int4")).toBe("inference-int4.yaml");
    expect(inferenceFilenameForId("low-ctx")).toBe("inference-low-ctx.yaml");
  });

  // Invariant: for any slug we'd derive from a real filename, the round-trip
  // back through inferenceFilenameForId returns the original filename.
  test.prop([
    fc.stringMatching(/^[a-z0-9][a-z0-9-]{0,30}$/),
  ])("round-trips filename → id → filename for any plausible slug", (slug) => {
    const filename = slug === "default" ? "inference.yaml" : `inference-${slug}.yaml`;
    const id = inferenceVariantIdFromFilename(filename);
    expect(id).toBe(slug);
    expect(inferenceFilenameForId(id!)).toBe(filename);
  });
});
