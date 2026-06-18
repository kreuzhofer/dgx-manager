import { execFileSync } from "node:child_process";
import { parseSparkrunList, type SparkrunRecipeSummary } from "./runtime/sparkrun-parse.js";
import { deriveRecipeArch, type RecipeArch } from "./runtime/recipe-arch.js";

export const SPARKRUN_PKG = "sparkrun==0.2.38";

export interface Recipe {
  file: string;
  name: string;
  description?: string;
  model?: string;
  container: string;
  cluster_only?: boolean;
  solo_only?: boolean;
  /** Target CPU arch derived from the recipe ref; used for per-node filtering. */
  arch: RecipeArch;
  defaults: Record<string, unknown>;
}

export function toRecipe(s: SparkrunRecipeSummary): Recipe {
  return {
    file: s.ref,
    name: s.name,
    description: s.description,
    model: s.model,
    container: "sparkrun",
    arch: deriveRecipeArch(s.ref),
    cluster_only: s.minNodes > 1 ? true : undefined,
    solo_only: undefined,
    defaults: {
      tensor_parallel: s.tpDefault ?? 1,
      gpu_memory_utilization: s.gpuMemDefault ?? 0.85,
      port: 8000,
      max_model_len: "",
    },
  };
}

/** Discover available recipes by running `sparkrun list --json`. */
export function discoverRecipes(): Recipe[] {
  try {
    const out = execFileSync(
      "uvx",
      ["--from", SPARKRUN_PKG, "sparkrun", "list", "--json"],
      { encoding: "utf8", timeout: 30_000 }
    );
    const recipes = parseSparkrunList(out).map(toRecipe);
    console.log(`Discovered ${recipes.length} sparkrun recipes`);
    return recipes;
  } catch (err) {
    console.error("Failed to discover sparkrun recipes:", err);
    return [];
  }
}
