import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import {
  inferenceVariantIdFromFilename,
  inferenceFilenameForId,
  listInferenceVariants,
} from "./inference-template.js";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

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

describe("listInferenceVariants", () => {
  it("returns [] for a recipe dir with no inference templates", () => {
    const dir = mkdtempSync(join(tmpdir(), "variants-"));
    try {
      writeFileSync(join(dir, "recipe.yaml"), "name: x\n");
      expect(listInferenceVariants(dir)).toEqual([]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("returns a single variant when only inference.yaml exists", () => {
    const dir = mkdtempSync(join(tmpdir(), "variants-"));
    try {
      writeFileSync(join(dir, "inference.yaml"),
        `name: my-recipe-bf16\ndescription: Default BF16 serve.\nmodel: x\n`);
      const out = listInferenceVariants(dir);
      expect(out).toEqual([{
        id: "default",
        filename: "inference.yaml",
        name: "my-recipe-bf16",
        description: "Default BF16 serve.",
      }]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("enumerates multiple variants sorted with 'default' first then alphabetical", () => {
    const dir = mkdtempSync(join(tmpdir(), "variants-"));
    try {
      writeFileSync(join(dir, "inference.yaml"),         `name: r-bf16\ndescription: bf16.\n`);
      writeFileSync(join(dir, "inference-fp8.yaml"),     `name: r-fp8\ndescription: fp8 on-load.\n`);
      writeFileSync(join(dir, "inference-int4.yaml"),    `name: r-int4\ndescription: int4 awq.\n`);
      writeFileSync(join(dir, "recipe.yaml"),            `name: r-train\n`);
      writeFileSync(join(dir, "not-an-inference.txt"),   `noise`);

      const out = listInferenceVariants(dir);
      expect(out.map((v) => v.id)).toEqual(["default", "fp8", "int4"]);
      expect(out[1].filename).toBe("inference-fp8.yaml");
      expect(out[1].name).toBe("r-fp8");
      expect(out[2].description).toBe("int4 awq.");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("omits description when the YAML doesn't declare one", () => {
    const dir = mkdtempSync(join(tmpdir(), "variants-"));
    try {
      writeFileSync(join(dir, "inference.yaml"), `name: bare\n`);
      const out = listInferenceVariants(dir);
      expect(out[0].name).toBe("bare");
      expect(out[0].description).toBeUndefined();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("falls back to the variant id when name: is missing from the YAML", () => {
    // A malformed template still appears in the list; name defaults to the id
    // so the UI can render something instead of crashing.
    const dir = mkdtempSync(join(tmpdir(), "variants-"));
    try {
      writeFileSync(join(dir, "inference-fp8.yaml"), `model: x\n`);
      const out = listInferenceVariants(dir);
      expect(out).toEqual([{
        id: "fp8",
        filename: "inference-fp8.yaml",
        name: "fp8",
        description: undefined,
      }]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
