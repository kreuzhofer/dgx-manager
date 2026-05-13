import { existsSync } from "fs";
import { join } from "path";

/**
 * Materialize a fine-tune's inference template by substituting the merged-model
 * path placeholder and injecting served_model_name into the defaults block.
 *
 * Inference templates live next to training recipes (e.g.
 * `<fine-tune-recipes>/recipes/<name>/inference.yaml`). They look like a
 * regular spark-vllm-docker recipe but use the literal placeholder
 * {{MERGED_MODEL_PATH}} wherever the local model path needs to land.
 *
 * Substitution is plain text — we don't parse YAML — to avoid imposing a
 * specific YAML library on the agent and to keep round-trips byte-exact
 * for hand-tuned comments and whitespace.
 */
export const MERGED_PATH_PLACEHOLDER = "{{MERGED_MODEL_PATH}}";

export interface SubstitutionParams {
  modelPath: string;        // absolute path inside the container, e.g. /workspace/outputs/<jobId>/merged
  servedModelName: string;  // friendly name to report via /v1/models
}

export function applyFinetuneSubstitutions(
  yaml: string,
  params: SubstitutionParams,
): string {
  // 1. Replace every occurrence of {{MERGED_MODEL_PATH}} with the merged path.
  let out = yaml.split(MERGED_PATH_PLACEHOLDER).join(params.modelPath);

  // 2. Inject served_model_name into defaults: block — unless author already
  // declared one (then we preserve their intent; see test 2).
  // Scope detection to a YAML key at exactly 2 spaces of indent (the
  // convention for defaults: children) so we don't false-positive on
  // `--served-model-name foo` inside a `command: |` literal block scalar.
  const hasServedName = /^ {2}served_model_name:\s*\S/m.test(out);
  if (!hasServedName) {
    if (!/^defaults:\s*\n/m.test(out)) {
      throw new Error(
        "inference template: must contain a top-level 'defaults:' block (followed by a newline) OR an explicit served_model_name: line; got neither",
      );
    }
    out = out.replace(
      /^defaults:\s*\n/m,
      `defaults:\n  served_model_name: ${params.servedModelName}\n`,
    );
  }

  return out;
}

/**
 * Return the absolute path to the inference template for a given
 * artifact variant, or null if no template exists for it.
 *
 *   bf16 (default) → <recipeDir>/inference.yaml
 *   fp8            → <recipeDir>/inference-fp8.yaml
 *
 * Used by the deploy path to decide whether to inherit a hand-authored
 * serve config or fall through to the minimal auto-gen.
 */
export function findInferenceTemplate(
  recipeDir: string,
  variant: "bf16" | "fp8" = "bf16",
): string | null {
  const filename = variant === "fp8" ? "inference-fp8.yaml" : "inference.yaml";
  const candidate = join(recipeDir, filename);
  return existsSync(candidate) ? candidate : null;
}
