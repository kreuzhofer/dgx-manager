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

const AMD64 = `runner: dgxrun
arch: amd64
cluster_only: false
model: unsloth/Qwen3.8-27B-NVFP4
container: vllm/vllm-openai:v0.28.0
defaults:
  tensor_parallel: 1
  port: 8000
command: vllm serve {model}`;

describe("loadDgxrunCatalog arch + cluster_only", () => {
  const deps = (files: Record<string, string>) => ({
    readDir: () => Object.keys(files),
    readFile: (p: string) => files[p.split("/").pop()!],
  });

  /** A recipe that declares `arch: amd64` is tagged amd64, so the deploy-time
   *  arch admission guard lets it onto the RTX-5090 host. */
  it("reads arch from the recipe", () => {
    const r = loadDgxrunCatalog("/d", deps({ "qwen.yaml": AMD64 }));
    expect(r[0].arch).toBe("amd64");
  });

  /** Backward compatibility: every recipe written before `arch:` existed
   *  targets the arm64 Sparks, so an absent field must still mean arm64. */
  it("defaults arch to arm64 when the recipe omits it", () => {
    const r = loadDgxrunCatalog("/d", deps({ "glm.yaml": VALID }));
    expect(r[0].arch).toBe("arm64");
  });

  /** cluster_only is likewise read, not assumed: a single-GPU amd64 recipe is
   *  deployable solo and must not be forced into the cluster-only UI path. */
  it("reads cluster_only: false from the recipe", () => {
    const r = loadDgxrunCatalog("/d", deps({ "qwen.yaml": AMD64 }));
    expect(r[0].cluster_only).toBe(false);
  });

  /** Backward compatibility: the pre-existing recipes are all cluster_only. */
  it("defaults cluster_only to true when the recipe omits it", () => {
    const noFlag = VALID.replace("cluster_only: true\n", "");
    const r = loadDgxrunCatalog("/d", deps({ "glm.yaml": noFlag }));
    expect(r[0].cluster_only).toBe(true);
  });

  /** Fail loud, not silently: an unrecognised arch is a typo that would
   *  otherwise route the recipe to the wrong hardware, so the recipe is
   *  dropped from the catalog rather than defaulted. */
  it("skips a recipe whose arch is not amd64/arm64", () => {
    const bad = AMD64.replace("arch: amd64", "arch: x86");
    const r = loadDgxrunCatalog("/d", deps({ "qwen.yaml": bad, "ok.yaml": VALID }));
    expect(r.map((x) => x.file)).toEqual(["@dgxrun/ok"]);
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
