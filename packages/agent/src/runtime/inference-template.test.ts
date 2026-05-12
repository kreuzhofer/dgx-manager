import { describe, it, expect } from "vitest";
import { applyFinetuneSubstitutions, MERGED_PATH_PLACEHOLDER } from "./inference-template.js";

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
