import { readdirSync, readFileSync } from "fs";
import { join } from "path";
import { parse } from "yaml";

/** Local structural interface compatible with agent's Recipe type (avoids cross-package rootDir issue). */
interface Recipe {
  file: string;
  name: string;
  description?: string;
  model?: string;
  container: string;
  cluster_only?: boolean;
  arch: string;
  defaults: Record<string, unknown>;
}

export type CatalogRecipe = Recipe & { source: "dgxrun" };

interface CatalogDeps {
  readDir?: (d: string) => string[];
  readFile?: (p: string) => string;
}

export function loadDgxrunCatalog(dir: string, deps: CatalogDeps = {}): CatalogRecipe[] {
  const readDir = deps.readDir ?? ((d) => readdirSync(d));
  const readFile = deps.readFile ?? ((p) => readFileSync(p, "utf-8"));
  let files: string[];
  try {
    files = readDir(dir).filter((f) => f.endsWith(".yaml") || f.endsWith(".yml"));
  } catch {
    return [];
  }
  const out: CatalogRecipe[] = [];
  for (const f of files.sort()) {
    const base = f.replace(/\.ya?ml$/, "");
    let doc: unknown;
    try {
      doc = parse(readFile(join(dir, f)));
    } catch (e) {
      console.warn(`[dgxrun-catalog] skip ${f}: parse error ${(e as Error).message}`);
      continue;
    }
    if (doc == null || typeof doc !== "object" || Array.isArray(doc)) {
      console.warn(`[dgxrun-catalog] skip ${f}: not a mapping`);
      continue;
    }
    const o = doc as Record<string, unknown>;
    if (o.runner !== "dgxrun") {
      console.warn(`[dgxrun-catalog] skip ${f}: not runner:dgxrun`);
      continue;
    }
    const d = (o.defaults && typeof o.defaults === "object" ? o.defaults : {}) as Record<string, unknown>;
    out.push({
      file: `@dgxrun/${base}`,
      name: typeof o.name === "string" ? o.name : base,
      description: typeof o.description === "string" ? o.description : undefined,
      model: typeof o.model === "string" ? o.model : undefined,
      container: "dgxrun",
      source: "dgxrun",
      arch: "arm64",
      cluster_only: true,
      defaults: {
        tensor_parallel: d.tensor_parallel ?? 4,
        gpu_memory_utilization: d.gpu_memory_utilization ?? 0.85,
        port: d.port ?? 8000,
        max_model_len: d.max_model_len ?? "",
      },
    });
  }
  return out;
}

const DGXRUN_PREFIX = "@dgxrun/";
export function resolveDgxrunRecipeFile(recipeFile: string, dir: string): string | null {
  if (typeof recipeFile !== "string" || !recipeFile.startsWith(DGXRUN_PREFIX)) return null;
  const name = recipeFile.slice(DGXRUN_PREFIX.length);
  if (!name || name.includes("/") || name.includes("\\") || name.includes("..")) return null;
  return join(dir, `${name}.yaml`);
}

/**
 * Directory the in-repo dgxrun recipe catalog is loaded from. Overridable via
 * env for tests (point at a temp fixture dir) and for deployments where the
 * working directory isn't the repo root.
 */
export const DGXRUN_RECIPES_DIR = process.env.DGXRUN_RECIPES_DIR || join(process.cwd(), "recipes/dgxrun");

let _cache: CatalogRecipe[] | null = null;

/** Memoized catalog read — call refreshDgxrunCatalog() to invalidate. */
export function getDgxrunCatalog(): CatalogRecipe[] {
  if (_cache == null) _cache = loadDgxrunCatalog(DGXRUN_RECIPES_DIR);
  return _cache;
}

/** Clears the memoized catalog so the next getDgxrunCatalog() call re-reads disk. */
export function refreshDgxrunCatalog(): void {
  _cache = null;
}
