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
