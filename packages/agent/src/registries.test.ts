import { describe, it, expect } from "vitest";
import { it as itProp, fc } from "@fast-check/vitest";
import { parse } from "yaml";
import { renderRegistriesYaml, type RegistryWire } from "./registries.js";

describe("renderRegistriesYaml", () => {
  it("renders a known registry to parseable YAML", () => {
    const out = renderRegistriesYaml([
      { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes", description: "amd64 RTX", visible: false, tuning_subpath: "tuning" },
    ]);
    expect(out.startsWith("registries:\n")).toBe(true);
    expect(parse(out)).toEqual({
      registries: [
        { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes", description: "amd64 RTX", visible: false, tuning_subpath: "tuning" },
      ],
    });
  });

  it("escapes quotes and backslashes in descriptions", () => {
    const out = renderRegistriesYaml([
      { name: "x", url: "https://h/r.git", subpath: "recipes", description: 'has "quotes" and \\ slash' },
    ]);
    expect(parse(out).registries[0].description).toBe('has "quotes" and \\ slash');
  });

  /**
   * Invariant: rendering an arbitrary registry list then YAML-parsing it yields
   * back exactly the same logical data — name/url/subpath always present, optional
   * string fields preserved verbatim (escaping is correct), and `visible` is `false`
   * in the parsed output iff it was `false` in the input.
   */
  itProp.prop([
    fc.array(
      fc.record({
        name: fc.string({ minLength: 1 }),
        url: fc.string({ minLength: 1 }),
        subpath: fc.string({ minLength: 1 }),
        description: fc.option(fc.string(), { nil: undefined }),
        visible: fc.option(fc.boolean(), { nil: undefined }),
        tuning_subpath: fc.option(fc.string({ minLength: 1 }), { nil: undefined }),
      }),
      { maxLength: 8 },
    ),
  ])("round-trips through a real YAML parser", (regs: RegistryWire[]) => {
    const parsed = parse(renderRegistriesYaml(regs)) as { registries: RegistryWire[] };
    expect(parsed.registries).toHaveLength(regs.length);
    regs.forEach((r, i) => {
      const p = parsed.registries[i];
      expect(p.name).toBe(r.name);
      expect(p.url).toBe(r.url);
      expect(p.subpath).toBe(r.subpath);
      if (r.description != null) expect(p.description).toBe(r.description);
      if (r.tuning_subpath != null) expect(p.tuning_subpath).toBe(r.tuning_subpath);
      expect(p.visible === false).toBe(r.visible === false);
    });
  });
});
