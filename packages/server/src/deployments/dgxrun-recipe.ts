import { parse } from "yaml";

/**
 * Server-side resolution of a dgxrun recipe from YAML text the manager can
 * read directly (inline `recipeYaml`, or a `recipePath` file on shared
 * storage). The manager parses it, and — if `runner: dgxrun` — fans the full
 * resolved object to each cluster node so the agent needn't re-fetch.
 *
 * Registry-ref (`recipeFile`) recipes are NOT resolvable here in v1: their raw
 * YAML lives only in the agent's sparkrun cache, not on the manager. That's a
 * documented v1 follow-up (needs manager-side registry access). Backward
 * compatible: a recipe WITHOUT `runner: dgxrun` returns `isDgxrun:false` and
 * the existing sparkrun path is taken unchanged.
 */
export interface DgxrunResolvedRecipe {
  runner?: string;
  model?: string;
  container: string;
  env?: Record<string, string | number | boolean>;
  command: string;
  defaults?: Record<string, unknown>;
  cluster_only?: boolean;
}

export interface DgxrunRecipeResult {
  isDgxrun: boolean;
  recipe?: DgxrunResolvedRecipe;
  /** Populated when isDgxrun is true but the recipe is malformed for dgxrun. */
  error?: string;
}

function asEnv(raw: unknown): Record<string, string | number | boolean> | undefined {
  if (raw == null || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const out: Record<string, string | number | boolean> = {};
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") out[k] = v;
    else if (v != null) out[k] = String(v);
  }
  return out;
}

/**
 * Parse recipe YAML text and decide whether it's a dgxrun deploy.
 *
 * Fail-safe: any parse error (or non-object YAML) returns `isDgxrun:false` so a
 * bad/foreign document never diverts a plain sparkrun deploy. Only when
 * `runner: dgxrun` is explicitly present do we validate the dgxrun essentials
 * (container + command) and surface a clear error if they're missing.
 */
export function resolveDgxrunRecipe(yamlText: string): DgxrunRecipeResult {
  let doc: unknown;
  try {
    doc = parse(yamlText);
  } catch {
    return { isDgxrun: false };
  }
  if (doc == null || typeof doc !== "object" || Array.isArray(doc)) {
    return { isDgxrun: false };
  }
  const o = doc as Record<string, unknown>;
  if (o.runner !== "dgxrun") return { isDgxrun: false };

  const container = typeof o.container === "string" ? o.container : "";
  const command = typeof o.command === "string" ? o.command : "";
  if (!container) return { isDgxrun: true, error: "dgxrun recipe missing container image" };
  if (!command.trim()) return { isDgxrun: true, error: "dgxrun recipe missing command" };

  const recipe: DgxrunResolvedRecipe = {
    runner: "dgxrun",
    model: typeof o.model === "string" ? o.model : undefined,
    container,
    command,
    env: asEnv(o.env),
    defaults: (o.defaults && typeof o.defaults === "object" && !Array.isArray(o.defaults))
      ? (o.defaults as Record<string, unknown>)
      : undefined,
    cluster_only: o.cluster_only === true,
  };
  return { isDgxrun: true, recipe };
}
