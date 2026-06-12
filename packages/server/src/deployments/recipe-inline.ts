export const MAX_INLINE_RECIPE_BYTES = 512 * 1024;

/**
 * Validate an inline sparkrun recipe YAML body (D7). Fail-fast: throws on empty, oversized,
 * or content that doesn't look like a recipe. NOTE: an accepted recipe's `command:` runs in a
 * container — arbitrary-execution surface; callers must audit-log.
 */
export function validateInlineRecipe(yaml: string): void {
  if (!yaml || !yaml.trim()) throw new Error("recipeYaml is empty");
  if (Buffer.byteLength(yaml, "utf8") > MAX_INLINE_RECIPE_BYTES) {
    throw new Error(`recipeYaml too large (> ${MAX_INLINE_RECIPE_BYTES} bytes)`);
  }
  if (!/^\s*(model|command|runtime)\s*:/m.test(yaml)) {
    throw new Error("recipeYaml does not look like a sparkrun recipe (needs model:/command:/runtime:)");
  }
}

/**
 * Best-effort extraction of the human-meaningful model identity from an inline
 * sparkrun recipe YAML, so inline deploys show a real name in the dashboard
 * instead of a synthetic `inline-<ts>` key. Pure + fail-safe: returns `{}` when
 * a field is absent or the YAML is malformed; the caller falls back to the
 * synthetic key, so this is never worse than the previous behaviour.
 *
 * - `model`: the top-level `model:` scalar — the HF model id, e.g.
 *   `google/gemma-4-12B-it-qat-w4a16-ct`. Sparkrun recipes place this at column 0.
 * - `servedModelName`: the `served_model_name:` scalar — the endpoint alias,
 *   e.g. `gemma4-12b-unified`; in v2 recipes it sits indented under `defaults:`.
 *
 * Targeted line-parsing (not a full YAML load) keeps this dependency-free and
 * matches `validateInlineRecipe`'s regex approach. Quotes and trailing inline
 * comments are stripped.
 */
export function parseInlineRecipeModel(yaml: string): { model?: string; servedModelName?: string } {
  const scalar = (raw: string | undefined): string | undefined => {
    if (raw == null) return undefined;
    let v = raw.trim();
    if (!v) return undefined;
    const q = v[0];
    if (q === '"' || q === "'") {
      const close = v.indexOf(q, 1); // quoted scalar — take inside, ignore trailing comment
      if (close > 0) return v.slice(1, close) || undefined;
      // unterminated quote: fall through and treat the rest literally
    }
    const comment = v.indexOf(" #"); // unquoted: ` #...` is a trailing comment
    if (comment >= 0) v = v.slice(0, comment).trim();
    return v || undefined;
  };
  // Top-level `model:` only (column 0) — the leading `^model:` won't match the
  // indented `served_model_name:` line or any other nested key.
  const model = scalar(/^model:[ \t]*(.+)$/m.exec(yaml)?.[1]);
  // `served_model_name:` at any indentation (typically nested under defaults:).
  const servedModelName = scalar(/^[ \t]*served_model_name:[ \t]*(.+)$/m.exec(yaml)?.[1]);
  return { ...(model ? { model } : {}), ...(servedModelName ? { servedModelName } : {}) };
}
