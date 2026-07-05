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
