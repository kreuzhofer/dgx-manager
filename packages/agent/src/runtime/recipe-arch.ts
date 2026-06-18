/**
 * Derive a recipe's target CPU architecture from its launch ref.
 *
 * Pure, total function — used both on the agent (to tag each wire Recipe with
 * an `arch`) and as the source of truth for arch-based filtering/admission on
 * the server.
 */

/** Target arch of a recipe. `"any"` means arch-agnostic (e.g. Ollama). */
export type RecipeArch = "amd64" | "arm64" | "any";

/**
 * Registries whose recipes target amd64 hosts (e.g. aihost01 / RTX 5090).
 * `@rtx` is the registry Daniel created (rtx-recipe-registry, `name: rtx`);
 * `@dgx-amd64` is kept for forward-compat with a future amd64 DGX registry.
 */
const AMD64_REGISTRIES = ["@rtx", "@dgx-amd64"];

/**
 * Classify a recipe ref:
 *   - `ollama:` prefix → `"any"` (Ollama is arch-agnostic)
 *   - ref under an amd64 registry (`@rtx/…`, `@dgx-amd64/…`) → `"amd64"`
 *   - everything else (transitional/official sparkrun registries) → `"arm64"`
 */
export function deriveRecipeArch(ref: string): RecipeArch {
  if (ref.startsWith("ollama:")) return "any";
  if (AMD64_REGISTRIES.some((reg) => ref.startsWith(reg + "/"))) return "amd64";
  return "arm64";
}
