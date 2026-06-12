import { describe, it, expect } from "vitest";
import { it as itProp, fc } from "@fast-check/vitest";
import { buildSparkrunArgs } from "./sparkrun-args.js";

describe("buildSparkrunArgs", () => {
  it("solo deploy: ref, --no-follow, port forwarded; no host flag", () => {
    const args = buildSparkrunArgs({ recipeRef: "qwen3-1.7b-vllm", hosts: ["10.0.0.1"], port: 8000 });
    expect(args).toContain("run");
    expect(args).toContain("qwen3-1.7b-vllm");
    expect(args).toContain("--no-follow");
    expect(args.join(" ")).toContain("--port 8000");
    expect(args).not.toContain("-H");
  });

  it("cluster deploy: -H lists head first, --tp equals host count", () => {
    const args = buildSparkrunArgs({ recipeRef: "big-model", hosts: ["10.0.0.1", "10.0.0.2", "10.0.0.3"] });
    const hIdx = args.indexOf("-H");
    expect(hIdx).toBeGreaterThanOrEqual(0);
    expect(args[hIdx + 1]).toBe("10.0.0.1,10.0.0.2,10.0.0.3");
    const tpIdx = args.indexOf("--tp");
    expect(args[tpIdx + 1]).toBe("3");
  });

  it("forwards -o overrides verbatim", () => {
    const args = buildSparkrunArgs({ recipeRef: "r", hosts: ["h"], options: { max_model_len: 8192, foo: "bar" } });
    expect(args).toContain("-o");
    expect(args.join(" ")).toContain("max_model_len=8192");
    expect(args.join(" ")).toContain("foo=bar");
  });

  it("never emits eugr run-recipe.sh flags", () => {
    const args = buildSparkrunArgs({ recipeRef: "r", hosts: ["a", "b"] });
    expect(args).not.toContain("--eth-if");
    expect(args).not.toContain("--ib-if");
    expect(args).not.toContain("--setup");
  });

  /** Invariant: for any non-empty host list, --tp (when not explicitly overridden)
   * equals the number of hosts, because each DGX Spark contributes exactly one GPU. */
  itProp.prop([fc.array(fc.string({ minLength: 1 }), { minLength: 1, maxLength: 8 })])(
    "tp defaults to host count",
    (hosts) => {
      const args = buildSparkrunArgs({ recipeRef: "r", hosts });
      const tpIdx = args.indexOf("--tp");
      expect(args[tpIdx + 1]).toBe(String(hosts.length));
    },
  );
});
