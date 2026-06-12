import { describe, it, expect } from "vitest";
import { applyFinetuneSubstitutions, MERGED_PATH_PLACEHOLDER, renderSparkrunFinetuneRecipe } from "./inference-template.js";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { findInferenceTemplate } from "./inference-template.js";

describe("applyFinetuneSubstitutions", () => {
  it("replaces {{MERGED_MODEL_PATH}} placeholder and injects served_model_name", () => {
    const input = `recipe_version: "1"
name: qwen3.6-27b-bf16
model: ${MERGED_PATH_PLACEHOLDER}
container: vllm-node

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 1
  max_model_len: 32768

command: |
  vllm serve ${MERGED_PATH_PLACEHOLDER} \\
    --host {host} \\
    --port {port} \\
    --max-model-len {max_model_len} \\
    -tp {tensor_parallel}
`;

    const out = applyFinetuneSubstitutions(input, {
      modelPath: "/workspace/outputs/cmp073lno00mn36p0bhffd2q4/merged",
      servedModelName: "chat3d-build123d-01",
    });

    // Placeholder is replaced wherever it appears
    expect(out).not.toContain(MERGED_PATH_PLACEHOLDER);
    expect(out).toContain("model: /workspace/outputs/cmp073lno00mn36p0bhffd2q4/merged");
    expect(out).toContain("vllm serve /workspace/outputs/cmp073lno00mn36p0bhffd2q4/merged");
    // served_model_name added to defaults
    expect(out).toMatch(/^defaults:[\s\S]*?served_model_name: chat3d-build123d-01/m);
    // Other content preserved verbatim
    expect(out).toContain("tensor_parallel: 1");
    expect(out).toContain("port: 8000");
  });

  it("is idempotent if served_model_name is already declared", () => {
    const input = `defaults:
  port: 8000
  served_model_name: existing-name

command: |
  vllm serve ${MERGED_PATH_PLACEHOLDER}
`;
    const out = applyFinetuneSubstitutions(input, {
      modelPath: "/path/to/merged",
      servedModelName: "new-name",
    });
    // Existing served_model_name wins — author intent is preserved
    expect(out).toContain("served_model_name: existing-name");
    expect(out).not.toContain("served_model_name: new-name");
  });

  it("throws when template has no defaults: block AND no served_model_name", () => {
    const input = `name: minimal
command: |
  vllm serve ${MERGED_PATH_PLACEHOLDER}
`;
    expect(() =>
      applyFinetuneSubstitutions(input, {
        modelPath: "/path",
        servedModelName: "x",
      }),
    ).toThrow(/defaults/);
  });

  it("does not treat --served-model-name inside command: heredoc as already-declared", () => {
    // The command: block has `--served-model-name foo` as a vLLM flag, but
    // there is no top-level `served_model_name:` YAML key. Injection should
    // still happen into defaults:.
    const input = `defaults:
  port: 8000

command: |
  vllm serve ${MERGED_PATH_PLACEHOLDER} \\
    --served-model-name foo
`;
    const out = applyFinetuneSubstitutions(input, {
      modelPath: "/m",
      servedModelName: "real-name",
    });
    expect(out).toContain("served_model_name: real-name");
    expect(out).toContain("--served-model-name foo"); // original kept
  });
});

describe("findInferenceTemplate", () => {
  it("returns the path when inference.yaml exists in the recipe dir", () => {
    const tmp = mkdtempSync(join(tmpdir(), "inftpl-"));
    try {
      const recipeDir = join(tmp, "recipes", "qwen3.6-27b-base-lora-attn-mlp");
      mkdirSync(recipeDir, { recursive: true });
      const target = join(recipeDir, "inference.yaml");
      writeFileSync(target, "recipe_version: \"1\"\nname: test\n");

      expect(findInferenceTemplate(recipeDir)).toBe(target);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("returns null when the recipe dir has no inference.yaml", () => {
    const tmp = mkdtempSync(join(tmpdir(), "inftpl-"));
    try {
      const recipeDir = join(tmp, "recipes", "qwen3.6-27b-base-lora-attn-only");
      mkdirSync(recipeDir, { recursive: true });
      // Note: no inference.yaml written
      expect(findInferenceTemplate(recipeDir)).toBeNull();
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("returns null when the recipe dir does not exist at all", () => {
    expect(findInferenceTemplate("/nonexistent/path/never/created")).toBeNull();
  });
});

describe("findInferenceTemplate(variant)", () => {
  it("returns inference.yaml when variant is bf16 (or omitted)", () => {
    const dir = mkdtempSync(join(tmpdir(), "tpl-"));
    writeFileSync(join(dir, "inference.yaml"), "name: bf16\n");
    expect(findInferenceTemplate(dir, "bf16")).toBe(join(dir, "inference.yaml"));
    expect(findInferenceTemplate(dir)).toBe(join(dir, "inference.yaml"));
    rmSync(dir, { recursive: true, force: true });
  });

  it("returns inference-fp8.yaml when variant is fp8 and file exists", () => {
    const dir = mkdtempSync(join(tmpdir(), "tpl-"));
    writeFileSync(join(dir, "inference.yaml"), "name: bf16\n");
    writeFileSync(join(dir, "inference-fp8.yaml"), "name: fp8\n");
    expect(findInferenceTemplate(dir, "fp8")).toBe(join(dir, "inference-fp8.yaml"));
    rmSync(dir, { recursive: true, force: true });
  });

  it("returns null when variant is fp8 but inference-fp8.yaml is missing", () => {
    const dir = mkdtempSync(join(tmpdir(), "tpl-"));
    writeFileSync(join(dir, "inference.yaml"), "name: bf16\n");
    expect(findInferenceTemplate(dir, "fp8")).toBeNull();
    rmSync(dir, { recursive: true, force: true });
  });

  it("returns null when neither variant exists", () => {
    const dir = mkdtempSync(join(tmpdir(), "tpl-"));
    expect(findInferenceTemplate(dir, "bf16")).toBeNull();
    expect(findInferenceTemplate(dir, "fp8")).toBeNull();
    rmSync(dir, { recursive: true, force: true });
  });
});

describe("renderSparkrunFinetuneRecipe", () => {
  it("emits a sparkrun recipe pointing at the merged model with runtime vllm", () => {
    const yaml = renderSparkrunFinetuneRecipe({
      mergedModelPath: "/workspace/outputs/abc/merged",
      servedModelName: "My-Model",
      container: "vllm-node",
    });
    expect(yaml).toContain("runtime: vllm");
    expect(yaml).toContain("/workspace/outputs/abc/merged");
    expect(yaml).toContain("My-Model");
  });

  it("uses defaults for gpuMem and maxModelLen when not provided", () => {
    const yaml = renderSparkrunFinetuneRecipe({
      mergedModelPath: "/mnt/tank/outputs/xyz/merged",
      servedModelName: "test-model",
      container: "vllm-node",
    });
    expect(yaml).toContain("gpu_memory_utilization: 0.85");
    expect(yaml).toContain("max_model_len: 4096");
  });

  it("uses provided gpuMem and maxModelLen when specified", () => {
    const yaml = renderSparkrunFinetuneRecipe({
      mergedModelPath: "/workspace/merged",
      servedModelName: "custom-model",
      container: "vllm-custom",
      gpuMem: 0.95,
      maxModelLen: 8192,
    });
    expect(yaml).toContain("gpu_memory_utilization: 0.95");
    expect(yaml).toContain("max_model_len: 8192");
    expect(yaml).toContain("container: vllm-custom");
  });

  it("includes the vllm serve command with placeholder substitutions", () => {
    const yaml = renderSparkrunFinetuneRecipe({
      mergedModelPath: "/workspace/outputs/abc/merged",
      servedModelName: "My-Model",
      container: "vllm-node",
    });
    expect(yaml).toContain("vllm serve /workspace/outputs/abc/merged");
    expect(yaml).toContain("{host}");
    expect(yaml).toContain("{port}");
    expect(yaml).toContain("{tensor_parallel}");
    expect(yaml).toContain("{served_model_name}");
  });
});

describe("findInferenceTemplate — variant id back-compat", () => {
  it("'default' resolves to inference.yaml (new canonical id)", () => {
    const dir = mkdtempSync(join(tmpdir(), "find-"));
    try {
      writeFileSync(join(dir, "inference.yaml"), "name: x");
      expect(findInferenceTemplate(dir, "default")).toBe(join(dir, "inference.yaml"));
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  it("'bf16' still resolves to inference.yaml (legacy alias)", () => {
    const dir = mkdtempSync(join(tmpdir(), "find-"));
    try {
      writeFileSync(join(dir, "inference.yaml"), "name: x");
      expect(findInferenceTemplate(dir, "bf16")).toBe(join(dir, "inference.yaml"));
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  it("arbitrary slug resolves to inference-<slug>.yaml", () => {
    const dir = mkdtempSync(join(tmpdir(), "find-"));
    try {
      writeFileSync(join(dir, "inference-int4.yaml"), "name: x");
      expect(findInferenceTemplate(dir, "int4")).toBe(join(dir, "inference-int4.yaml"));
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });
});
