/**
 * Pure arch-admission decision for registry-ref deploys.
 *
 * A recipe built for one CPU arch cannot run on a node of another — e.g. an
 * arm64 (DGX Spark) sparkrun recipe deployed to the amd64 RTX-5090 host. We
 * fail fast at deploy time rather than letting the agent crash mid-launch.
 *
 * Kept pure (no Prisma, no IO) so it is property-testable; the route wires it
 * in over the recipe's `arch` (from the catalog) and the node's `arch`.
 */
import type { RecipeArch } from "../ws/agent-hub.js";

/**
 * Returns true when a recipe of `recipeArch` may be deployed to a node of
 * `nodeArch`. Allowed iff the arches match, or the recipe is arch-agnostic
 * (`"any"`). Any other combination is rejected.
 */
export function checkRecipeArchAdmission(
  recipeArch: RecipeArch,
  nodeArch: RecipeArch,
): boolean {
  return recipeArch === "any" || recipeArch === nodeArch;
}

/** Human-readable reason for a rejected deploy, for the HTTP error body. */
export function recipeArchMismatchMessage(
  recipeArch: RecipeArch,
  nodeArch: RecipeArch,
): string {
  return `Recipe targets ${recipeArch} but node is ${nodeArch} — architecture mismatch.`;
}
