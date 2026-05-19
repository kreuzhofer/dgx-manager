import { existsSync, readdirSync, readFileSync } from "fs";
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
 * Convert an inference filename into its variant id slug.
 *
 *   "inference.yaml"       → "default"
 *   "inference-fp8.yaml"   → "fp8"
 *   "inference-low-ctx.yaml" → "low-ctx"
 *
 * Returns null for filenames that don't match the convention (so callers can
 * skip non-inference files during a directory scan).
 */
export function inferenceVariantIdFromFilename(filename: string): string | null {
  if (filename === "inference.yaml") return "default";
  const m = filename.match(/^inference-([a-z0-9][a-z0-9-]*)\.yaml$/);
  return m ? m[1] : null;
}

/**
 * Convert a variant id back into the filename to look up in a recipe dir.
 * Legacy back-compat: "bf16" is treated as an alias for "default" so saved
 * deployments from before this feature keep resolving to inference.yaml.
 */
export function inferenceFilenameForId(id: string): string {
  if (id === "default" || id === "bf16") return "inference.yaml";
  return `inference-${id}.yaml`;
}

export interface InferenceVariant {
  /** Slug derived from filename. `inference.yaml` → "default";
   *  `inference-fp8.yaml` → "fp8". Used as the wire id everywhere. */
  id: string;
  /** Filename relative to the recipe dir. */
  filename: string;
  /** From the YAML's top-level `name:` field. Falls back to the id when
   *  the file doesn't declare one (malformed templates still appear). */
  name: string;
  /** From the YAML's top-level `description:` field. Optional. */
  description?: string;
}

/**
 * Enumerate every inference template in a training-recipe dir. Each
 * `inference*.yaml` file becomes one entry. Reads `name:` and
 * `description:` from each YAML's top of file using a lightweight
 * regex — we don't import a YAML parser for two fields. The list is
 * sorted with "default" first, then alphabetical by id, so the UI
 * order is deterministic without the dashboard having to re-sort.
 */
export function listInferenceVariants(recipeDir: string): InferenceVariant[] {
  let entries: string[];
  try {
    entries = readdirSync(recipeDir);
  } catch {
    return [];
  }
  const out: InferenceVariant[] = [];
  for (const filename of entries) {
    const id = inferenceVariantIdFromFilename(filename);
    if (!id) continue;
    const full = join(recipeDir, filename);
    let text = "";
    try { text = readFileSync(full, "utf-8"); } catch { /* fall through */ }
    const nameMatch = text.match(/^name:\s*(.+?)\s*$/m);
    const descMatch = text.match(/^description:\s*(.+?)\s*$/m);
    out.push({
      id,
      filename,
      name: nameMatch ? stripYamlQuotes(nameMatch[1]) : id,
      description: descMatch ? stripYamlQuotes(descMatch[1]) : undefined,
    });
  }
  out.sort((a, b) => {
    if (a.id === "default") return -1;
    if (b.id === "default") return 1;
    return a.id.localeCompare(b.id);
  });
  return out;
}

function stripYamlQuotes(v: string): string {
  if ((v.startsWith('"') && v.endsWith('"')) ||
      (v.startsWith("'") && v.endsWith("'"))) {
    return v.slice(1, -1);
  }
  return v;
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
