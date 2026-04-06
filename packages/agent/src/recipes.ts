import { execSync } from "child_process";
import { existsSync, readdirSync, readFileSync } from "fs";
import { join, basename } from "path";

const VLLM_REPO_URL = "https://github.com/kreuzhofer/spark-vllm-docker.git";
const VLLM_REPO_PATH =
  process.env.VLLM_REPO_PATH || "/mnt/tank/src/github/spark-vllm-docker";

export interface Recipe {
  file: string;
  name: string;
  description?: string;
  model?: string;
  container: string;
  cluster_only?: boolean;
  solo_only?: boolean;
  defaults: Record<string, unknown>;
}

/** Ensure the spark-vllm-docker repo is cloned locally. */
function ensureRepo(): boolean {
  if (existsSync(join(VLLM_REPO_PATH, "recipes"))) {
    return true;
  }
  try {
    console.log(`Cloning ${VLLM_REPO_URL} to ${VLLM_REPO_PATH}...`);
    execSync(`git clone ${VLLM_REPO_URL} ${VLLM_REPO_PATH}`, {
      timeout: 120_000,
      stdio: "inherit",
    });
    return true;
  } catch (err) {
    console.error("Failed to clone spark-vllm-docker:", err);
    return false;
  }
}

/**
 * Minimal YAML parser for recipe files.
 * Handles the flat key: value structure and simple nested maps used by recipes.
 * Avoids adding a YAML dependency to the agent.
 */
function parseRecipeYaml(text: string): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  let currentMap: Record<string, unknown> | null = null;
  let currentMapKey: string | null = null;

  for (const rawLine of text.split("\n")) {
    // Strip comments (but not inside quoted strings — good enough for recipes)
    const line = rawLine.replace(/#.*$/, "").trimEnd();
    if (!line.trim()) continue;

    const indent = line.length - line.trimStart().length;

    // Nested map entry (indented key: value)
    if (indent >= 2 && currentMapKey && currentMap) {
      const m = line.trim().match(/^([A-Za-z_][\w]*)\s*:\s*(.+)?$/);
      if (m) {
        currentMap[m[1]] = parseValue(m[2] ?? "");
        continue;
      }
    }

    // Top-level key
    const topMatch = line.match(/^([A-Za-z_][\w]*)\s*:\s*(.*)?$/);
    if (topMatch) {
      const key = topMatch[0].split(":")[0].trim();
      const val = topMatch[0].slice(topMatch[0].indexOf(":") + 1).trim();

      // Block scalar (command: |)
      if (val === "|" || val === ">") {
        // We don't need the command text for recipe listing — skip
        currentMapKey = null;
        currentMap = null;
        continue;
      }

      // Empty value = start of a map or list
      if (!val || val === "{}") {
        if (!val) {
          currentMapKey = key;
          currentMap = {};
          result[key] = currentMap;
        } else {
          result[key] = {};
          currentMapKey = null;
          currentMap = null;
        }
        continue;
      }

      result[key] = parseValue(val);
      currentMapKey = null;
      currentMap = null;
    }
  }
  return result;
}

function parseValue(s: string): unknown {
  const trimmed = s.trim();
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  if (trimmed === "{}") return {};
  if (/^-?\d+$/.test(trimmed)) return parseInt(trimmed, 10);
  if (/^-?\d+\.\d+$/.test(trimmed)) return parseFloat(trimmed);
  // Strip quotes
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

/** Scan the recipes directory and return parsed recipe metadata. */
export function discoverRecipes(): Recipe[] {
  if (!ensureRepo()) return [];

  const recipesDir = join(VLLM_REPO_PATH, "recipes");
  const recipes: Recipe[] = [];

  function scanDir(dir: string) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        scanDir(join(dir, entry.name));
      } else if (entry.name.endsWith(".yaml") || entry.name.endsWith(".yml")) {
        try {
          const filePath = join(dir, entry.name);
          const raw = readFileSync(filePath, "utf-8");
          const parsed = parseRecipeYaml(raw);

          recipes.push({
            file: filePath.slice(VLLM_REPO_PATH.length + 1), // relative path
            name: (parsed.name as string) || basename(entry.name, ".yaml"),
            description: parsed.description as string | undefined,
            model: parsed.model as string | undefined,
            container: (parsed.container as string) || "vllm-node",
            cluster_only: parsed.cluster_only as boolean | undefined,
            solo_only: parsed.solo_only as boolean | undefined,
            defaults: (parsed.defaults as Record<string, unknown>) || {},
          });
        } catch (err) {
          console.error(`Failed to parse recipe ${entry.name}:`, err);
        }
      }
    }
  }

  scanDir(recipesDir);
  console.log(`Discovered ${recipes.length} vLLM recipes`);
  return recipes;
}
