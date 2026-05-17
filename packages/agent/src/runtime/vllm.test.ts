import { describe, it, expect } from "vitest";
import { buildLaunchArgs } from "./vllm.js";

describe("buildLaunchArgs servedModelName passthrough", () => {
  it("appends --served-model-name <name> to the post-`--` passthrough args when servedModelName is set", () => {
    const args = buildLaunchArgs({
      recipeName: "test-recipe",
      options: {
        port: 8000,
        servedModelName: "my-custom-name",
      },
    });

    // Find the `--` separator and the flag after it.
    const dashIdx = args.indexOf("--");
    expect(dashIdx).toBeGreaterThan(-1);

    const flagIdx = args.indexOf("--served-model-name");
    expect(flagIdx).toBeGreaterThan(dashIdx); // must be in the passthrough section
    expect(args[flagIdx + 1]).toBe("my-custom-name");
  });

  it("does NOT append --served-model-name when servedModelName is undefined", () => {
    const args = buildLaunchArgs({
      recipeName: "test-recipe",
      options: { port: 8000 },
    });
    expect(args.indexOf("--served-model-name")).toBe(-1);
  });

  it("composes both pipelineParallel AND servedModelName in the passthrough block when both are set", () => {
    const args = buildLaunchArgs({
      recipeName: "test-recipe",
      options: {
        port: 8000,
        pipelineParallel: 4,
        servedModelName: "combo-name",
      },
    });
    const dashIdx = args.indexOf("--");
    expect(dashIdx).toBeGreaterThan(-1);
    // Both flags must appear after `--`.
    const ppIdx = args.indexOf("-pp");
    const sgnIdx = args.indexOf("--served-model-name");
    expect(ppIdx).toBeGreaterThan(dashIdx);
    expect(sgnIdx).toBeGreaterThan(dashIdx);
  });
});
