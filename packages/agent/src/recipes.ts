import { execFileSync } from "node:child_process";
import { parseSparkrunList, type SparkrunRecipeSummary } from "./runtime/sparkrun-parse.js";
import { deriveRecipeArch, type RecipeArch } from "./runtime/recipe-arch.js";

/**
 * sparkrun version constraint for every `uvx --from` invocation.
 *
 * A floor, not a pin. This was `==0.2.38` and went stale: 0.2.38 does not
 * unescape a recipe command's `{{...}}`, and its placeholder regex swallows any
 * `{placeholder}` nested inside a JSON-valued flag. We diagnosed that correctly
 * and then reported it upstream as a registry bug, which it was not — 0.3.x
 * masks literal braces before substitution and renders those recipes fine.
 * See https://github.com/spark-arena/recipe-registry/issues/20.
 *
 * 0.3.3 is the floor because it is the version whose CLI we verified against
 * every flag and subcommand used here — `run` (`-H --tp --pp --port --gpu-mem
 * --max-model-len --served-model-name --no-follow -o`), `list --json`,
 * `logs` (`--tail` is an alias of `-n/--lines`), `stop -H`,
 * `cluster check-job -H`, and `registry update`.
 *
 * Trade-off worth knowing: with a floor, nodes resolve at first use, so a fleet
 * provisioned weeks apart can land on different 0.3.x versions. That is
 * acceptable while the surface we use is this small and stable; if a future
 * release changes it, pin again deliberately rather than drifting.
 */
export const SPARKRUN_PKG = "sparkrun>=0.3.3";

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

/**
 * Re-pull the recipe registries from git (`sparkrun registry update`).
 *
 * `sparkrun list` reads whatever is in sparkrun's cached registry clones and
 * does NOT `git pull` them, and `sparkrun run` reads the same clones — so after
 * a recipe is edited upstream, a plain rescan surfaces the old content and a
 * deploy launches the stale recipe. Calling this before discovery makes
 * `POST /api/recipes/refresh` actually reflect (and deploy) the latest recipes.
 * Best-effort: logs and returns on failure rather than aborting the rescan.
 */
export function updateRegistries(): void {
  try {
    execFileSync(
      "uvx",
      ["--from", SPARKRUN_PKG, "sparkrun", "registry", "update"],
      { encoding: "utf8", timeout: 60_000 },
    );
    console.log("Updated sparkrun recipe registries from git");
  } catch (err) {
    console.error("Failed to update sparkrun registries:", err);
  }
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
