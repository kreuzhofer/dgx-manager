import { describe, it, expect } from "vitest";
import { loadDgxrunCatalog, resolveDgxrunRecipeFile } from "./dgxrun-catalog.js";

const VALID = `runner: dgxrun
model: CosmicRaisins/GLM-5.2-AWQ-INT4-15pct
container: vllm-node-tf5-glm52-b12x:probe
cluster_only: true
defaults:
  tensor_parallel: 4
  gpu_memory_utilization: 0.88
  port: 8000
  max_model_len: 87040
command: vllm serve {model}`;

describe("loadDgxrunCatalog", () => {
  const deps = (files: Record<string, string>) => ({
    readDir: () => Object.keys(files),
    readFile: (p: string) => files[p.split("/").pop()!],
  });
  it("maps a valid dgxrun yaml to a CatalogRecipe under @dgxrun/", () => {
    const r = loadDgxrunCatalog("/recipes/dgxrun", deps({ "glm-5.2-awq-15pct.yaml": VALID }));
    expect(r).toHaveLength(1);
    expect(r[0].file).toBe("@dgxrun/glm-5.2-awq-15pct");
    expect(r[0].source).toBe("dgxrun");
    expect(r[0].container).toBe("dgxrun");
    expect(r[0].cluster_only).toBe(true);
    expect(r[0].defaults.tensor_parallel).toBe(4);
    expect(r[0].defaults.max_model_len).toBe(87040);
  });
  it("skips a malformed file but keeps the good ones", () => {
    const r = loadDgxrunCatalog("/d", deps({ "bad.yaml": ": not: yaml:", "ok.yaml": VALID }));
    expect(r.map((x) => x.file)).toEqual(["@dgxrun/ok"]);
  });
  it("skips a yaml without runner: dgxrun", () => {
    const r = loadDgxrunCatalog("/d", deps({ "spark.yaml": "container: foo\ncommand: bar" }));
    expect(r).toEqual([]);
  });
  it("missing dir -> []", () => {
    const r = loadDgxrunCatalog("/nope", { readDir: () => { throw new Error("ENOENT"); }, readFile: () => "" });
    expect(r).toEqual([]);
  });
});

describe("resolveDgxrunRecipeFile", () => {
  it("maps @dgxrun/<name> to <dir>/<name>.yaml", () => {
    expect(resolveDgxrunRecipeFile("@dgxrun/glm-5.2-awq-15pct", "/app/recipes/dgxrun"))
      .toBe("/app/recipes/dgxrun/glm-5.2-awq-15pct.yaml");
  });
  it("rejects non-@dgxrun refs", () => {
    expect(resolveDgxrunRecipeFile("@community/foo", "/d")).toBeNull();
    expect(resolveDgxrunRecipeFile("plain", "/d")).toBeNull();
  });
  it("rejects path traversal / separators", () => {
    expect(resolveDgxrunRecipeFile("@dgxrun/../../etc/passwd", "/d")).toBeNull();
    expect(resolveDgxrunRecipeFile("@dgxrun/sub/evil", "/d")).toBeNull();
    expect(resolveDgxrunRecipeFile("@dgxrun/", "/d")).toBeNull();
  });
});
