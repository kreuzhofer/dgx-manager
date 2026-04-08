import { execSync } from "child_process";
import { existsSync, readdirSync, readFileSync } from "fs";
import { join } from "path";

const TRAINING_REPO_URL =
  "https://github.com/kreuzhofer/dgx-manager-fine-tune-recipes.git";
const TRAINING_REPO_PATH =
  process.env.TRAINING_REPO_PATH ||
  "/mnt/tank/src/github/dgx-manager-fine-tune-recipes";

export interface TrainingRecipe {
  file: string; // relative dir path (e.g., "recipes/gemma4-e2b-lora")
  name: string;
  description?: string;
  base_model: string;
  framework: string;
  method: string;
  dataset_format?: string;
  container: { image: string; name: string; build_context?: string };
  scripts: {
    entrypoint: string;
    train: string;
    launch: string;
    ds_config?: string;
  };
  defaults: Record<string, unknown>;
  hardware: { min_nodes: number; gpus_per_node: number; vram_estimate_mb: number };
}

/** Ensure the training recipes repo is cloned locally. */
function ensureRepo(): boolean {
  if (existsSync(join(TRAINING_REPO_PATH, "recipes"))) {
    return true;
  }
  try {
    console.log(`Cloning ${TRAINING_REPO_URL} to ${TRAINING_REPO_PATH}...`);
    execSync(`git clone ${TRAINING_REPO_URL} ${TRAINING_REPO_PATH}`, {
      timeout: 120_000,
      stdio: "inherit",
    });
    return true;
  } catch (err) {
    console.error("Failed to clone training recipes:", err);
    return false;
  }
}

/**
 * Minimal YAML parser for recipe files.
 * Handles flat key: value and simple nested maps used by recipes.
 */
function parseRecipeYaml(text: string): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  let currentMap: Record<string, unknown> | null = null;
  let currentMapKey: string | null = null;

  for (const rawLine of text.split("\n")) {
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

      if (val === "|" || val === ">") {
        currentMapKey = null;
        currentMap = null;
        continue;
      }

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
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

/** Get the absolute path to the training repo. */
export function getTrainingRepoPath(): string {
  return TRAINING_REPO_PATH;
}

/** Scan the recipes directory and return parsed training recipe metadata. */
export function discoverTrainingRecipes(): TrainingRecipe[] {
  if (!ensureRepo()) return [];

  const recipesDir = join(TRAINING_REPO_PATH, "recipes");
  if (!existsSync(recipesDir)) return [];

  const recipes: TrainingRecipe[] = [];

  for (const entry of readdirSync(recipesDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;

    const recipeYaml = join(recipesDir, entry.name, "recipe.yaml");
    if (!existsSync(recipeYaml)) continue;

    try {
      const raw = readFileSync(recipeYaml, "utf-8");
      const parsed = parseRecipeYaml(raw);

      const container = (parsed.container as Record<string, unknown>) || {};
      const scripts = (parsed.scripts as Record<string, unknown>) || {};
      const hardware = (parsed.hardware as Record<string, unknown>) || {};
      const defaults = (parsed.defaults as Record<string, unknown>) || {};

      recipes.push({
        file: `recipes/${entry.name}`,
        name: (parsed.name as string) || entry.name,
        description: parsed.description as string | undefined,
        base_model: (parsed.base_model as string) || "",
        framework: (parsed.framework as string) || "deepspeed",
        method: (parsed.method as string) || "lora",
        dataset_format: parsed.dataset_format as string | undefined,
        container: {
          image: (container.image as string) || "",
          name: (container.name as string) || "dgx-finetune",
          build_context: container.build_context as string | undefined,
        },
        scripts: {
          entrypoint: (scripts.entrypoint as string) || "entrypoint.sh",
          train: (scripts.train as string) || "train.py",
          launch: (scripts.launch as string) || "launch.sh",
          ds_config: scripts.ds_config as string | undefined,
        },
        defaults,
        hardware: {
          min_nodes: (hardware.min_nodes as number) || 1,
          gpus_per_node: (hardware.gpus_per_node as number) || 1,
          vram_estimate_mb: (hardware.vram_estimate_mb as number) || 0,
        },
      });
    } catch (err) {
      console.error(`Failed to parse training recipe ${entry.name}:`, err);
    }
  }

  console.log(`Discovered ${recipes.length} training recipe(s)`);
  return recipes;
}
