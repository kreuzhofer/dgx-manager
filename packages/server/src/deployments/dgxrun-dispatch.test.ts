import { describe, it, expect } from "vitest";
import { it as itProp, fc } from "@fast-check/vitest";
import { buildDgxrunDeploys, DEFAULT_MASTER_PORT } from "./dgxrun-dispatch.js";
import type { DgxrunResolvedRecipe } from "./dgxrun-recipe.js";

const recipe: DgxrunResolvedRecipe = {
  runner: "dgxrun",
  container: "img:tag",
  command: "vllm serve {model}",
  defaults: { tensor_parallel: 4 },
};

describe("buildDgxrunDeploys", () => {
  const ctx = {
    deploymentId: "dep1",
    recipe,
    clusterNodeIds: ["n0", "n1", "n2", "n3"],
    clusterNodeIps: ["10.0.0.10", "10.0.0.11", "10.0.0.12", "10.0.0.13"],
  };

  it("emits one deploy per node, head first = rank 0", () => {
    const d = buildDgxrunDeploys(ctx);
    expect(d.map((x) => x.nodeId)).toEqual(["n0", "n1", "n2", "n3"]);
    expect(d.map((x) => x.payload.rank)).toEqual([0, 1, 2, 3]);
  });

  it("sets masterAddr to the head mgmt IP on every rank", () => {
    for (const d of buildDgxrunDeploys(ctx)) {
      expect(d.payload.masterAddr).toBe("10.0.0.10");
      expect(d.payload.nnodes).toBe(4);
      expect(d.payload.kind).toBe("dgxrun");
      expect(d.payload.masterPort).toBe(DEFAULT_MASTER_PORT);
    }
  });

  it("sets headless only for workers (rank>0)", () => {
    expect(buildDgxrunDeploys(ctx).map((x) => x.payload.headless)).toEqual([false, true, true, true]);
  });

  it("honors an explicit masterPort override", () => {
    expect(buildDgxrunDeploys({ ...ctx, masterPort: 29501 })[0].payload.masterPort).toBe(29501);
  });

  it("throws when IPs don't align 1:1 with node ids", () => {
    expect(() => buildDgxrunDeploys({ ...ctx, clusterNodeIps: ["10.0.0.10"] })).toThrow(/align/i);
  });

  /** Invariant: for any cluster size, exactly one head (rank 0, not headless)
   *  and every worker headless; masterAddr is always the first IP. */
  itProp.prop([fc.integer({ min: 1, max: 12 })])("head-first rank invariant", (n) => {
    const ids = Array.from({ length: n }, (_, i) => `n${i}`);
    const ips = Array.from({ length: n }, (_, i) => `10.0.0.${i}`);
    const d = buildDgxrunDeploys({ deploymentId: "d", recipe, clusterNodeIds: ids, clusterNodeIps: ips });
    expect(d.filter((x) => !x.payload.headless)).toHaveLength(1);
    expect(d[0].payload.headless).toBe(false);
    expect(d.every((x) => x.payload.masterAddr === "10.0.0.0")).toBe(true);
  });
});
