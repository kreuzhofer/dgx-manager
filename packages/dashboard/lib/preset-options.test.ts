import { describe, expect, it } from "vitest";
import { presetOptions } from "./preset-options";

const run = (presetId: string | null) => ({ presetId }) as { presetId: string | null };

describe("presetOptions", () => {
  // The filter list was hardcoded and silently drifted: acc-gpqa-diamond-full-longgen
  // existed server-side for weeks and could never be selected. Deriving it from the
  // runs themselves means a new preset is filterable the moment a run uses it.
  it("derives the options from the runs present", () => {
    expect(presetOptions([run("acc-gpqa-diamond-full"), run("acc-ifeval-quick")]))
      .toEqual(["acc-ifeval-quick", "acc-gpqa-diamond-full"].sort());
  });

  it("de-duplicates", () => {
    expect(presetOptions([run("a"), run("a"), run("b")])).toEqual(["a", "b"]);
  });

  // A custom run stores presetId null; it must not become an empty option that
  // silently filters everything out.
  it("drops null and empty preset ids", () => {
    expect(presetOptions([run(null), run(""), run("a")])).toEqual(["a"]);
  });

  it("returns nothing for no runs", () => {
    expect(presetOptions([])).toEqual([]);
  });
});
